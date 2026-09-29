"""Legacy KDA flags must come from checkpoint shapes, independent of the environment."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from megatron.bridge.models.apertus2.apertus2_checkpoint import infer_kda_checkpoint_flags
from megatron.bridge.training.mlm_compat.arguments import _load_args_from_checkpoint


pytestmark = pytest.mark.unit


def _metadata(*, per_channel=True, bias=False):
    result = {}
    for index in (0, 2):
        prefix = f"decoder.layers.{index}.self_attention"
        result[f"{prefix}.A_log"] = SimpleNamespace(global_shape=(32 if per_channel else 4,))
        if bias:
            result[f"{prefix}.gate_out_proj.bias"] = SimpleNamespace(global_shape=(32,))
    return result


def _infer(metadata):
    return infer_kda_checkpoint_flags(
        metadata,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_value_heads=4,
        key_head_dim=8,
    )


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_layout_inference(per_channel, bias):
    assert _infer(_metadata(per_channel=per_channel, bias=bias)) == {
        "linear_attn_a_log_per_channel": per_channel,
        "linear_attn_output_gate_bias": bias,
    }


@pytest.mark.parametrize("shape", [None, (5,), (4, 8)])
def test_invalid_or_missing_decay_tensor(shape):
    metadata = _metadata()
    if shape is None:
        del metadata["decoder.layers.2.self_attention.A_log"]
    else:
        metadata["decoder.layers.2.self_attention.A_log"].global_shape = shape
    with pytest.raises(ValueError, match="expected A_log shape"):
        _infer(metadata)


def test_mixed_layout_rejected():
    metadata = _metadata()
    metadata["decoder.layers.2.self_attention.A_log"].global_shape = (4,)
    with pytest.raises(ValueError, match="Mixed KDA A_log"):
        _infer(metadata)


def test_mixed_bias_rejected():
    metadata = _metadata(bias=True)
    del metadata["decoder.layers.2.self_attention.gate_out_proj.bias"]
    with pytest.raises(ValueError, match="Mixed KDA output-gate bias"):
        _infer(metadata)


def test_native_loader_recovers_flags_before_building_config(monkeypatch):
    args = Namespace(
        sssglu=True,
        experimental_attention_variant="kda",
        num_layers=3,
        linear_attention_freq=[1, 0, 1],
        linear_num_value_heads=4,
        linear_key_head_dim=8,
    )
    monkeypatch.setattr(
        "megatron.bridge.training.mlm_compat.arguments.dist_checkpointing.load_common_state_dict",
        lambda path: {"args": args},
    )
    monkeypatch.setattr(
        "megatron.bridge.training.mlm_compat.arguments.dist_checkpointing.load_tensors_metadata",
        lambda path: _metadata(),
    )
    recovered = _load_args_from_checkpoint("checkpoint")
    assert recovered.linear_attn_a_log_per_channel is True
    assert recovered.linear_attn_output_gate_bias is False
