"""Small GPU parity test using the HF implementation shipped with an Apertus2 export.

Set APERTUS2_HF_REFERENCE to that export directory. Run with one or two torchrun
processes; WORLD_SIZE is the tensor-parallel size. No production weights are loaded.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.bridge.models.apertus2 import Apertus2Bridge, Apertus2ModelBuilder
from megatron.bridge.models.conversion.auto_bridge import AutoBridge


def _checkpoint_weights(state):
    """Convert Transformers' stacked experts to the checkpoint's per-expert keys."""
    result = {}
    for name, value in state.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for index, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                result[f"{prefix}.{index}.gate_proj.weight"] = gate.contiguous()
                result[f"{prefix}.{index}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for index, expert in enumerate(value):
                result[f"{prefix}.{index}.down_proj.weight"] = expert.contiguous()
        else:
            result[name] = value.contiguous()
    return result


@pytest.fixture(scope="module")
def distributed():
    if not os.environ.get("APERTUS2_HF_REFERENCE"):
        pytest.skip("Set APERTUS2_HF_REFERENCE to an Apertus2 HF export")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", rank))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=torch.distributed.get_world_size())
    model_parallel_cuda_manual_seed(123)
    init_num_microbatches_calculator(rank, None, global_batch_size=1, micro_batch_size=1, data_parallel_size=1)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_num_microbatches_calculator()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("construction", ["provider", "builder"])
def test_megachonk_hybrid_roundtrip_and_forward(distributed, tmp_path, construction):
    reference = os.environ.get("APERTUS2_HF_REFERENCE")
    if reference is None:
        pytest.skip("Set APERTUS2_HF_REFERENCE to an Apertus2 HF export")
    config = AutoConfig.from_pretrained(reference, trust_remote_code=True, local_files_only=True)
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
            raise ValueError(f"Missing HF config field: {name}")
        setattr(config, name, value)
    torch.manual_seed(42)
    hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
    hf_model = hf_model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, value in list(hf_model.named_parameters()) + list(hf_model.named_buffers()):
            if name.endswith((".A_log", ".dt_bias", ".qb_beta", ".e_score_correction_bias")):
                value.data = value.data.float()
                if not name.endswith(".e_score_correction_bias"):
                    value.copy_(torch.linspace(-0.1234567, 0.2345678, value.numel()).view_as(value))
    expected = _checkpoint_weights(hf_model.state_dict())
    model_path = tmp_path / "hf"
    model_path.mkdir()
    config.save_pretrained(model_path)
    for filename in ("configuration_apertus2.py", "modeling_apertus2.py"):
        shutil.copyfile(Path(reference) / filename, model_path / filename)
    save_file(expected, model_path / "model.safetensors")
    auto = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
    tp_size = torch.distributed.get_world_size()
    if construction == "provider":
        provider = auto.to_megatron_provider(load_weights=False)
        provider.tensor_model_parallel_size = tp_size
        provider.sequence_parallel = tp_size > 1
        provider.parallel_output = False
        provider.gradient_accumulation_fusion = False
        provider.finalize()
        models = provider.provide_distributed_model(wrap_with_ddp=False, pg_collection=distributed)
    else:
        model_config = Apertus2Bridge().hf_config_to_model_config(config)
        model_config.tensor_model_parallel_size = tp_size
        model_config.sequence_parallel = tp_size > 1
        model_config.parallel_output = False
        model_config.gradient_accumulation_fusion = False
        model_config.finalize()
        models = Apertus2ModelBuilder(model_config).build_distributed_models(distributed, wrap_with_ddp=False)
    auto.load_hf_weights(models)
    # Exercise the native on-disk checkpoint contract as well as HF mappings.
    checkpoint_path = [str(tmp_path / "native") if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(checkpoint_path, src=0)
    if torch.distributed.get_rank() == 0:
        Path(checkpoint_path[0]).mkdir()
    torch.distributed.barrier()
    dist_checkpointing.save(models[0].sharded_state_dict(), checkpoint_path[0])
    with torch.no_grad():
        for parameter in models[0].parameters():
            parameter.zero_()
        for name, buffer in models[0].named_buffers():
            if name.endswith(".qb_beta"):
                buffer.zero_()
    restored = dist_checkpointing.load(models[0].sharded_state_dict(), checkpoint_path[0])
    models[0].load_state_dict(restored)
    exported = dict(auto.export_hf_weights(models, cpu=True, show_progress=False, weight_dtype=torch.bfloat16))
    assert set(exported) == set(expected), (set(expected) - set(exported), set(exported) - set(expected))
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value, atol=0, rtol=0, msg=name)
    for name, parameter in models[0].named_parameters():
        if name.endswith((".A_log", ".dt_bias")):
            assert parameter.dtype is torch.float32, name
    hf_model = hf_model.cuda().eval()
    models[0].eval()
    tokens = (torch.arange(16, device="cuda").view(1, 16) + 7) % config.vocab_size
    positions = torch.arange(16, device="cuda").view(1, 16)
    with torch.no_grad():
        expected_logits = hf_model(tokens, use_cache=False).logits.float()
        logits = models[0](tokens, positions, None).float()
        cosine = F.cosine_similarity(logits.flatten(), expected_logits.flatten(), dim=0).item()
        assert cosine >= 0.99, cosine
        assert torch.equal(logits[:, -1].argmax(-1), expected_logits[:, -1].argmax(-1))
        prefill = hf_model(tokens[:, :-1], use_cache=True)
        decoded = hf_model(tokens[:, -1:], past_key_values=prefill.past_key_values, use_cache=True).logits.float()
        torch.testing.assert_close(decoded, expected_logits[:, -1:], atol=0.03, rtol=0.03)
    models[0].train()
    models[0](tokens, positions, None).float().square().mean().backward()
    for name, parameter in models[0].named_parameters():
        if name.endswith((".A_log", ".dt_bias")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    diagnostics = {
        "tp": tp_size,
        "construction": construction,
        "cosine": cosine,
        "max_logit_error": (logits - expected_logits).abs().max().item(),
        "mean_logit_error": (logits - expected_logits).abs().mean().item(),
        "tensors": len(exported),
    }
    (tmp_path / "parity.json").write_text(json.dumps(diagnostics, indent=2))
