# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, Swiss AI Initiative. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Apertus2 KDA parameter mappings."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.apertus2.apertus2_mapping import (
    Apertus2QBMapping,
    Apertus2QKVGMapping,
    KDAConv1dMapping,
    KDAInProjMapping,
    _pack_tp_sections,
    _unpack_tp_sections,
    build_apertus2_mapping_registry,
)
from megatron.bridge.models.conversion.param_mapping import QKVGMapping, QKVMapping


def _in_proj_mapping():
    return KDAInProjMapping(
        "decoder.layers.0.self_attention.in_proj.weight",
        query="model.layers.0.self_attn.q_proj.weight",
        key="model.layers.0.self_attn.k_proj.weight",
        value="model.layers.0.self_attn.v_proj.weight",
        decay_low_rank="model.layers.0.self_attn.f_a_proj.weight",
        gate_low_rank="model.layers.0.self_attn.g_a_proj.weight",
        beta="model.layers.0.self_attn.b_proj.weight",
    )


def _schedule_config(**overrides):
    values = {
        "attention_output_gate": True,
        "layer_types": (
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ),
        "moe_layer_freq": [0, 1, 0, 1],
        "moe_router_enable_expert_bias": False,
        "num_hidden_layers": 4,
        "qk_layernorm": True,
        "sandwich_norm": False,
        "use_quantile_balancing": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


pytestmark = pytest.mark.unit


def test_qb_mapping_preserves_thresholds_and_hf_zero_buffer():
    mapping = Apertus2QBMapping("qb_beta", beta="gate.qb_beta", correction="gate.e_score_correction_bias")
    module = torch.nn.Linear(4, 4, bias=False)
    module.config = SimpleNamespace(params_dtype=torch.bfloat16)
    beta = torch.tensor([0.1234567, -0.2345678, 0.3456789, -0.4567891])
    weights = {"beta": beta, "correction": torch.zeros(4, dtype=torch.float32)}
    loaded = mapping.hf_to_megatron(weights, module)
    result = mapping.megatron_to_hf(loaded, module)
    torch.testing.assert_close(result["gate.qb_beta"], beta, rtol=0, atol=0)
    torch.testing.assert_close(result["gate.e_score_correction_bias"], weights["correction"], rtol=0, atol=0)


def test_qb_mapping_rejects_lossy_nonzero_correction_buffer():
    mapping = Apertus2QBMapping("qb_beta", beta="gate.qb_beta", correction="gate.e_score_correction_bias")
    with pytest.raises(ValueError, match="zero e_score_correction_bias"):
        mapping.hf_to_megatron({"beta": torch.zeros(4), "correction": torch.ones(4)}, None)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("shapes", [(8, 8, 16, 4, 4, 4), (8, 8, 16)])
def test_kda_sections_match_native_rank_layout(tp_size, shapes):
    sections = [torch.arange(size * 3).reshape(size, 3) + 1000 * index for index, size in enumerate(shapes)]
    packed = _pack_tp_sections(sections, tp_size)
    for rank, shard in enumerate(packed.chunk(tp_size)):
        expected = torch.cat([section.chunk(tp_size)[rank] for section in sections])
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)
    for expected, restored in zip(sections, _unpack_tp_sections(packed, shapes, tp_size)):
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_kda_rejects_projection_indivisible_by_tp():
    with pytest.raises(ValueError, match="divisible"):
        _pack_tp_sections([torch.zeros(3, 2), torch.zeros(5, 2)], 2)
    with pytest.raises(ValueError, match="Invalid KDA split shapes"):
        _unpack_tp_sections(torch.zeros(8, 2), (3, 5), 2)


@pytest.mark.parametrize("bias", [True, False, None])
def test_kda_gate_bias_mapping_follows_config(bias):
    registry = build_apertus2_mapping_registry(_schedule_config(linear_attn_output_gate_bias=bias))
    assert (registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.gate_out_proj.bias") is not None) == (
        bias is not False
    )


def test_kda_in_proj_export_splits_globally_gathered_weight(monkeypatch):
    mapping = _in_proj_mapping()
    split_shapes = (8, 8, 16, 4, 4, 4)
    fused = torch.arange(44 * 6).reshape(44, 6)
    module = SimpleNamespace(weight=SimpleNamespace(kda_split_shapes=split_shapes))
    monkeypatch.setattr(
        mapping._tp_mapping,
        "megatron_to_hf",
        lambda megatron_weights, megatron_module: {"weight": fused},
    )
    monkeypatch.setattr(
        mapping,
        "broadcast_obj_from_pp_rank",
        lambda value, description: value,
    )

    result = mapping.megatron_to_hf(torch.empty(0), module)
    expected = torch.split(fused, split_shapes, dim=0)

    assert list(result) == [str(mapping.hf_param[name]) for name in mapping._names]
    for name, section in zip(mapping._names, expected):
        assert torch.equal(result[str(mapping.hf_param[name])], section)


def test_kda_in_proj_export_requires_split_metadata(monkeypatch):
    mapping = _in_proj_mapping()
    monkeypatch.setattr(
        mapping._tp_mapping,
        "megatron_to_hf",
        lambda megatron_weights, megatron_module: {"weight": torch.empty(44, 6)},
    )
    monkeypatch.setattr(
        mapping,
        "broadcast_obj_from_pp_rank",
        lambda value, description: value,
    )

    with pytest.raises(ValueError, match="missing kda_split_shapes metadata"):
        mapping.megatron_to_hf(torch.empty(0), SimpleNamespace(weight=SimpleNamespace()))


def test_kda_conv_sections_use_global_geometry(monkeypatch):
    mapping = KDAConv1dMapping(
        "decoder.layers.0.self_attention.conv1d.weight",
        query="model.layers.0.self_attn.q_conv1d.weight",
        key="model.layers.0.self_attn.k_conv1d.weight",
        value="model.layers.0.self_attn.v_conv1d.weight",
    )
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=8,
    )
    module = SimpleNamespace(config=config)
    monkeypatch.setattr(mapping, "broadcast_obj_from_pp_rank", lambda value, description: value)

    assert mapping._sections(module) == (8, 8, 16)


def test_apertus2_qkvg_mapping_preserves_channelwise_gate(monkeypatch):
    mapping = Apertus2QKVGMapping("linear_qkv.weight", q="q", k="k", v="v", g="g")
    config = SimpleNamespace(
        attention_output_gate=True,
        hidden_size=5,
        kv_channels=3,
        num_attention_heads=4,
        num_query_groups=2,
    )
    module = SimpleNamespace(config=config)
    weights = {
        "q": torch.arange(12 * 5).reshape(12, 5),
        "k": torch.arange(6 * 5).reshape(6, 5) + 1_000,
        "v": torch.arange(6 * 5).reshape(6, 5) + 2_000,
        "g": torch.arange(12 * 5).reshape(12, 5) + 3_000,
    }
    monkeypatch.setattr(
        mapping._tp_mapping,
        "hf_to_megatron",
        lambda merged, megatron_module: merged,
    )
    fused = mapping.hf_to_megatron(weights, module)
    assert fused.shape == (36, 5)

    monkeypatch.setattr(
        mapping._tp_mapping,
        "megatron_to_hf",
        lambda megatron_weights, megatron_module: {"weight": fused},
    )
    monkeypatch.setattr(
        mapping,
        "broadcast_obj_from_pp_rank",
        lambda value, description: value,
    )
    restored = mapping.megatron_to_hf(fused, module)

    assert restored["g"].shape == (12, 5)
    for name, expected in weights.items():
        assert torch.equal(restored[name], expected)


def test_mapping_registry_follows_attention_and_moe_schedules():
    registry = build_apertus2_mapping_registry(_schedule_config())

    assert isinstance(
        registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight"),
        KDAInProjMapping,
    )
    assert isinstance(
        registry.megatron_to_hf_lookup("decoder.layers.1.self_attention.linear_qkv.weight"),
        Apertus2QKVGMapping,
    )
    assert registry.megatron_to_hf_lookup("decoder.layers.0.mlp.linear_fc1.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.0.mlp.router.weight") is None
    assert registry.megatron_to_hf_lookup("decoder.layers.1.mlp.router.weight") is not None
    assert registry.megatron_to_hf_lookup("decoder.layers.1.mlp.router.qb_beta") is not None


def test_mapping_registry_uses_qkv_without_attention_output_gate():
    registry = build_apertus2_mapping_registry(_schedule_config(attention_output_gate=False))

    mapping = registry.megatron_to_hf_lookup("decoder.layers.1.self_attention.linear_qkv.weight")
    assert isinstance(mapping, QKVMapping)
    assert not isinstance(mapping, QKVGMapping)
