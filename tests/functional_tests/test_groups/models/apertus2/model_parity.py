"""Offline hybrid Apertus2 weight roundtrip and logit parity using exported HF code."""

import argparse
import faulthandler
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.apertus2.apertus2_bridge import Apertus2Bridge
from megatron.bridge.models.apertus2.apertus2_builder import Apertus2ModelBuilder
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import StateDict


logger = logging.getLogger(__name__)


def check_model(source: str, checkpoint_dir: str | None = None) -> None:
    """Exercise dense and latent MoE layers with KDA, QB, sandwich norms and NoPE."""
    config_cls = get_class_from_dynamic_module("configuration_apertus2.Apertus2Config", source)
    model_cls = get_class_from_dynamic_module("modeling_apertus2.Apertus2ForCausalLM", source)
    values = json.loads((Path(source) / "config.json").read_text())
    values.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        no_rope_layers=[0, 0],
        moe_layer_freq=[0, 1],
        first_k_dense_replace=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        moe_intermediate_size=64,
        moe_latent_size=64,
        embedding_multiplier=128**0.5,
        residual_multiplier=0.5,
    )
    config = config_cls(**values)
    config._attn_implementation = "eager"
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    reference = model_cls(config).cuda().bfloat16().eval()
    # Distinct FP32 values detect silent BF16 truncation of router and decay state.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith((".A_log", ".dt_bias")):
                parameter.data = torch.linspace(-0.1234567, 0.2345678, parameter.numel(), device="cuda").reshape(
                    parameter.shape
                )
        for name, buffer in reference.named_buffers():
            if name.endswith(".e_score_correction_bias"):
                buffer.data = buffer.data.float()
            if name.endswith(".qb_beta"):
                buffer.data = torch.linspace(-0.0234567, 0.0345678, buffer.numel(), device="cuda").reshape(
                    buffer.shape
                )
    state = {}
    for name, weight in reference.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert, fused in enumerate(weight):
                state[f"{prefix}.{expert}.gate_proj.weight"], state[f"{prefix}.{expert}.up_proj.weight"] = fused.chunk(
                    2, dim=0
                )
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert, projection in enumerate(weight):
                state[f"{prefix}.{expert}.down_proj.weight"] = projection
        else:
            state[name] = weight
    pretrained = PreTrainedCausalLM()
    pretrained.config = config
    pretrained._state_dict_accessor = StateDict(state)
    bridge = Apertus2Bridge()
    model_config = bridge.hf_config_to_model_config(config)
    model_config.tensor_model_parallel_size = dist.get_world_size()
    model_config.sequence_parallel = dist.get_world_size() > 1
    model_config.parallel_output = False
    model_config.finalize()
    models = Apertus2ModelBuilder(model_config).build_distributed_models(
        ProcessGroupCollection.use_mpu_process_groups(),
        wrap_with_ddp=False,
    )
    logger.info("Built native hybrid model on rank %s", dist.get_rank())
    bridge.load_weights_hf_to_megatron(pretrained, models)
    logger.info("Rank %s: exporting native weights", dist.get_rank())
    exported = dict(bridge.stream_weights_megatron_to_hf(models, pretrained, cpu=False, show_progress=False))
    assert set(exported) == set(state), (set(state) - set(exported), set(exported) - set(state))
    for name, expected in state.items():
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)
    if checkpoint_dir is not None:
        from megatron.core import dist_checkpointing

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if any(checkpoint_path.iterdir()):
            raise ValueError("The checkpoint test requires an empty output directory")
        dist.barrier()
        logger.info("Rank %s: saving distributed checkpoint", dist.get_rank())
        dist_checkpointing.save({"model": models[0].sharded_state_dict()}, checkpoint_dir)
        with torch.no_grad():
            for parameter in models[0].parameters():
                parameter.zero_()
        restored = dist_checkpointing.load({"model": models[0].sharded_state_dict()}, checkpoint_dir)
        models[0].load_state_dict(restored["model"])
        reloaded = dict(bridge.stream_weights_megatron_to_hf(models, pretrained, cpu=False, show_progress=False))
        for name, expected in state.items():
            torch.testing.assert_close(reloaded[name], expected, rtol=0, atol=0)
        logger.info("PASS distributed checkpoint save/reload TP=%s", dist.get_world_size())
    model = models[0].eval()
    tokens = torch.arange(16, device="cuda").reshape(1, -1)
    positions = torch.arange(tokens.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        expected = reference(tokens, use_cache=False).logits.float()
        actual = model(tokens, positions, None).float()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    difference = (actual - expected).abs()
    assert cosine >= 0.99, cosine
    assert torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
    logger.info(
        "PASS hybrid TP=%s tensors=%s cosine=%.8f max=%.6g mean=%.6g next_token=%s",
        dist.get_world_size(),
        len(state),
        cosine,
        difference.max().item(),
        difference.mean().item(),
        actual[:, -1].argmax(-1).tolist(),
    )


def main() -> None:
    """Run a tiny copy of the requested architecture, without loading its huge weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-source", required=True)
    parser.add_argument("--checkpoint-dir", help="Optional empty shared directory for a native checkpoint round trip")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    faulthandler.dump_traceback_later(300)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    try:
        check_model(args.hf_source, args.checkpoint_dir)
    except Exception:
        logger.exception("Native model validation failed on rank %s", dist.get_rank())
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
