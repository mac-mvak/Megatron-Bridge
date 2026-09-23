"""Native offloading checkpoints load natively and through Bridge before HF export."""

import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from safetensors.torch import load_file, save_file
from test_apertus2_conversion import _checkpoint_weights
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.bridge import AutoBridge
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint


@pytest.fixture(scope="module", params=["ep", "pp"])
def distributed(request):
    if not os.environ.get("APERTUS2_HF_REFERENCE"):
        pytest.skip("Set APERTUS2_HF_REFERENCE to an Apertus2 HF export")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if request.param == "pp":
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", rank))
        parallel_state.initialize_model_parallel(
            pipeline_model_parallel_size=torch.distributed.get_world_size(),
        )
    else:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", rank))
        parallel_state.initialize_model_parallel(
            expert_model_parallel_size=torch.distributed.get_world_size(), expert_tensor_parallel_size=1
        )
    model_parallel_cuda_manual_seed(123)
    yield ProcessGroupCollection.use_mpu_process_groups()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


def _reference():
    config = AutoConfig.from_pretrained(
        os.environ["APERTUS2_HF_REFERENCE"], trust_remote_code=True, local_files_only=True
    )
    overrides = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        no_rope_layers=[0, 0, 0],
        moe_layer_freq=[0, 1, 1],
        first_k_dense_replace=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        moe_latent_size=128,
        n_shared_experts=2,
        vocab_size=256,
        max_position_embeddings=128,
        embedding_multiplier=16.0,
        residual_multiplier=6**-0.5,
        linear_attn_a_log_per_channel=True,
        linear_attn_output_gate_bias=False,
    )
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Missing HF configuration field: {name}")
        setattr(config, name, value)
    torch.manual_seed(42)
    reference = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
    reference = reference.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, value in list(reference.named_parameters()) + list(reference.named_buffers()):
            if name.endswith((".A_log", ".dt_bias", ".qb_beta", ".e_score_correction_bias")):
                value.data = value.data.float()
                if not name.endswith(".e_score_correction_bias"):
                    value.copy_(torch.linspace(-0.1234567, 0.2345678, value.numel()).view_as(value))
    return reference, _checkpoint_weights(reference.state_dict())


def _copy_to_offloaded(source, destination):
    source_params = dict(source.named_parameters())
    with torch.no_grad():
        for name, value in destination.named_parameters():
            match = re.fullmatch(r"(.*\.mlp\.experts\.)weight([12])(?:_expert_(\d+))?", name)
            if match:
                prefix, projection, expert = match.groups()
                if expert is None:
                    expected = torch.stack(
                        [
                            source_params[f"{prefix}linear_fc{projection}.weight{index}"]
                            for index in range(value.shape[0])
                        ]
                    )
                else:
                    expected = source_params[f"{prefix}linear_fc{projection}.weight{expert}"].T
            else:
                expected = source_params[name]
            value.copy_(expected)
        source_buffers = dict(source.named_buffers())
        for name, value in destination.named_buffers():
            if name in source_buffers:
                value.copy_(source_buffers[name])


@pytest.mark.parametrize("offloading_mode", ["fine-grained", "coarse-grained"])
def test_native_offloaded_checkpoint_load_and_hf_export(distributed, tmp_path, offloading_mode):
    reference, expected = _reference()
    hf_path = tmp_path / "hf"
    hf_path.mkdir()
    reference.config.save_pretrained(hf_path)
    for filename in ("configuration_apertus2.py", "modeling_apertus2.py"):
        shutil.copyfile(Path(os.environ["APERTUS2_HF_REFERENCE"]) / filename, hf_path / filename)
    save_file(expected, hf_path / "model.safetensors")
    auto = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    provider = auto.to_megatron_provider(load_weights=False)
    pipeline_export = parallel_state.get_pipeline_model_parallel_world_size() > 1
    provider.expert_model_parallel_size = parallel_state.get_expert_model_parallel_world_size()
    provider.pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if provider.pipeline_model_parallel_size > 1:
        provider.num_layers_in_first_pipeline_stage = 1
    provider.expert_tensor_parallel_size = 1
    provider.gradient_accumulation_fusion = False
    provider.parallel_output = False
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False, pg_collection=distributed)
    auto.load_hf_weights(models)

    native_provider = auto.to_megatron_provider(load_weights=False)
    native_provider.expert_model_parallel_size = provider.expert_model_parallel_size
    native_provider.pipeline_model_parallel_size = provider.pipeline_model_parallel_size
    native_provider.num_layers_in_first_pipeline_stage = provider.num_layers_in_first_pipeline_stage
    native_provider.expert_tensor_parallel_size = 1
    native_provider.moe_use_offloading_experts = True
    native_provider.moe_use_inplace_fp8_param = True
    native_provider.moe_use_extra_fp8_param_storage = True
    native_provider.moe_offloading_mode = offloading_mode
    native_provider.moe_offloading_num_chunks = 1
    native_provider.moe_offloading_num_stages = 1
    native_provider.gradient_accumulation_fusion = True
    native_provider.perform_initialization = False
    native_provider.finalize()
    native = native_provider.provide_distributed_model(
        wrap_with_ddp=False, use_cpu_initialization=True, pg_collection=distributed
    )
    _copy_to_offloaded(models[0], native[0])
    expected_native = {name: value.detach().clone() for name, value in native[0].named_parameters()}
    checkpoint = [str(tmp_path / "native") if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(checkpoint, src=0)
    if torch.distributed.get_rank() == 0:
        Path(checkpoint[0]).mkdir()
    torch.distributed.barrier()
    dist_checkpointing.save(
        {"model": native[0].module.sharded_state_dict(), "args": SimpleNamespace(moe_use_offloading_experts=True)},
        checkpoint[0],
    )
    metadata = dist_checkpointing.load_tensors_metadata(checkpoint[0])
    assert "decoder.layers.1.mlp.experts.experts.weight1" in metadata
    assert "decoder.layers.1.mlp.experts.experts.linear_fc1.weight" not in metadata
    with torch.no_grad():
        for model in [*native, *models]:
            for parameter in model.parameters():
                parameter.zero_()
            for name, buffer in model.named_buffers():
                if name.endswith(".qb_beta"):
                    buffer.zero_()
    # First prove that the native offloading implementation restores its own format.
    _load_model_weights_from_checkpoint(checkpoint[0], native)
    for name, value in native[0].named_parameters():
        torch.testing.assert_close(value, expected_native[name], atol=0, rtol=0, msg=name)
    # Then load the identical checkpoint through Bridge's standard expert path.
    _load_model_weights_from_checkpoint(checkpoint[0], models)
    if pipeline_export:
        export_path = str(Path(checkpoint[0]).parent / "exported")
        auto.save_hf_weights(
            models, export_path, show_progress=False, strict=True, distributed_save=True, weight_dtype=torch.bfloat16
        )
        saved = load_file(Path(export_path) / "model.safetensors")
        assert set(saved) == set(expected)
        for name, value in expected.items():
            torch.testing.assert_close(saved[name], value, atol=0, rtol=0, msg=name)
    exported = dict(auto.export_hf_weights(models, cpu=True, show_progress=False, weight_dtype=torch.bfloat16))
    assert set(exported) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value, atol=0, rtol=0, msg=name)
    if pipeline_export:
        return
    tokens = (torch.arange(16, device="cuda").view(1, 16) + 7) % reference.config.vocab_size
    positions = torch.arange(16, device="cuda").view(1, 16)
    models[0].eval()
    reference = reference.cuda().eval()
    with torch.no_grad():
        actual = models[0](tokens, positions, None).float()
        wanted = reference(tokens, use_cache=False).logits.float()
    assert F.cosine_similarity(actual.flatten(), wanted.flatten(), dim=0) >= 0.99
    assert torch.equal(actual[:, -1].argmax(-1), wanted[:, -1].argmax(-1))
