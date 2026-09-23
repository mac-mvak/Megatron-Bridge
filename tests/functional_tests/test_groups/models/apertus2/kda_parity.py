"""Offline KDA conversion, forward and backward check; run with torchrun (1 or 2 GPUs)."""

import argparse
import faulthandler
import logging
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention, get_kimi_delta_attention_module_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.apertus2.apertus2_bridge import Apertus2Bridge
from megatron.bridge.models.apertus2.apertus2_mapping import build_apertus2_mapping_registry


logger = logging.getLogger(__name__)


def check_variant(source: str, per_channel: bool, gate_bias: bool) -> None:
    """Compare native KDA against the actual exported HF implementation."""
    config_cls = get_class_from_dynamic_module("configuration_apertus2.Apertus2Config", source)
    model_cls = get_class_from_dynamic_module("modeling_apertus2.Apertus2KimiDeltaAttention", source)
    config = config_cls(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_attn_a_log_per_channel=per_channel,
        linear_attn_output_gate_bias=gate_bias,
        gate_lower_bound=-5.0,
        hidden_act="sssglu",
        no_rope_layers=[0],
        moe_layer_freq=[0],
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        moe_latent_size=64,
        sandwich_norm=True,
        embedding_multiplier=128**0.5,
        residual_multiplier=2**-0.5,
        use_quantile_balancing=True,
        dtype="bfloat16",
        attention_bias=False,
        attention_dropout=0.0,
    )
    model_config = Apertus2Bridge().hf_config_to_model_config(config)
    model_config.tensor_model_parallel_size = dist.get_world_size()
    model_config.sequence_parallel = False
    model_config.finalize()
    pg = ProcessGroupCollection.use_mpu_process_groups()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    # Deliberately contradict the explicit flag: serialized config must win.
    os.environ["KDA_ALOG_PER_CHANNEL"] = "0" if per_channel else "1"
    native = KimiDeltaAttention(
        model_config.transformer,
        get_kimi_delta_attention_module_spec(model_config.transformer).submodules,
        layer_number=1,
        pg_collection=pg,
        a_log_per_channel=per_channel,
        output_gate_bias=gate_bias,
    ).cuda()
    torch.manual_seed(987)
    reference = model_cls(config, 0).cuda().bfloat16()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name in ("A_log", "dt_bias"):
                parameter.data = torch.linspace(-0.5, 0.5, parameter.numel(), device="cuda").reshape(parameter.shape)
            elif name == "o_norm.weight":
                parameter.fill_(1)
            else:
                parameter.copy_(torch.randn(parameter.shape, device="cuda") * 0.04)
    hf_state = {
        f"model.layers.0.self_attn.{name}": parameter.detach() for name, parameter in reference.named_parameters()
    }
    hf_state["model.layers.0.attention_layernorm.weight"] = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    registry = build_apertus2_mapping_registry(config)
    converted = {}
    logger.info("Rank %s: checking KDA weights", dist.get_rank())
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            logger.debug("Rank %s: mapping %s", dist.get_rank(), name)
            mapping = registry.megatron_to_hf_lookup(f"decoder.layers.0.self_attention.{name}")
            assert mapping is not None, name
            module = native.get_submodule(name.rsplit(".", 1)[0]) if "." in name else native
            # Conv1d has no config; conversion attaches the owning model config too.
            module.config = model_config.transformer
            mapping.set_process_groups_from_pg_collection(pg)
            weights = (
                {key: hf_state[value] for key, value in mapping.hf_param.items()}
                if isinstance(mapping.hf_param, dict)
                else hf_state[mapping.hf_param]
            )
            parameter.copy_(mapping.hf_to_megatron(weights, module))
            converted.update(mapping.megatron_to_hf(parameter, module))
    assert set(converted) == set(hf_state)
    for name, expected in hf_state.items():
        torch.testing.assert_close(converted[name], expected, rtol=0, atol=0)
    assert native.A_log.numel() == (128 if per_channel else 2) // dist.get_world_size()
    assert ("bias" in dict(native.gate_out_proj.named_parameters())) == gate_bias
    assert "bias" not in dict(native.in_proj.named_parameters())
    assert "bias" not in dict(native.decay_out_proj.named_parameters())
    hidden = torch.randn(128, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dist.broadcast(hidden.detach(), 0)
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
    native_output, _ = native(hidden, None)
    reference_output, _ = reference(normalized.to(hidden.dtype).transpose(0, 1).contiguous())
    reference_output = reference_output.transpose(0, 1)
    cosine = torch.nn.functional.cosine_similarity(
        native_output.float().flatten(), reference_output.float().flatten(), dim=0
    )
    assert cosine > 0.99, cosine.item()
    difference = (native_output.float() - reference_output.float()).abs()
    native_output.float().square().mean().backward()
    for name in ("A_log", "dt_bias"):
        grad = getattr(native, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    logger.info(
        "PASS TP=%s per_channel=%s gate_bias=%s tensors=%s cosine=%.8f max=%.6g mean=%.6g",
        dist.get_world_size(),
        per_channel,
        gate_bias,
        len(converted),
        cosine.item(),
        difference.max().item(),
        difference.mean().item(),
    )


def main() -> None:
    """Run all checkpoint-layout variants with real collectives and FLA kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-source", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    faulthandler.dump_traceback_later(120)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    try:
        for per_channel, gate_bias in ((True, False), (False, True), (True, True), (False, False)):
            check_variant(args.hf_source, per_channel, gate_bias)
        dist.barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
