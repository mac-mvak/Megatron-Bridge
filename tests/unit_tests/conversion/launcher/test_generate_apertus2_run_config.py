# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for native Apertus2 run-config generation."""

from types import SimpleNamespace

import pytest
from scripts.conversion.generate_apertus2_run_config import _infer_kda_checkpoint_flags, _LegacyConfigView


def _kda_metadata(*, per_channel=True, bias=False):
    config = {
        "layer_types": ["linear_attention", "full_attention", "linear_attention"],
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
    }
    metadata = {}
    for index in (0, 2):
        prefix = f"decoder.layers.{index}.self_attention"
        metadata[f"{prefix}.A_log"] = SimpleNamespace(global_shape=(32 if per_channel else 4,))
        if bias:
            metadata[f"{prefix}.gate_out_proj.bias"] = SimpleNamespace(global_shape=(32,))
    return config, metadata


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_checkpoint_shapes_determine_kda_flags(per_channel, bias):
    config, metadata = _kda_metadata(per_channel=per_channel, bias=bias)
    assert _infer_kda_checkpoint_flags(config, metadata) == {
        "linear_attn_a_log_per_channel": per_channel,
        "linear_attn_output_gate_bias": bias,
    }


@pytest.mark.parametrize("shape", [None, (31,), (4, 8)])
def test_checkpoint_kda_missing_or_invalid_shape_fails(shape):
    config, metadata = _kda_metadata()
    key = "decoder.layers.2.self_attention.A_log"
    if shape is None:
        del metadata[key]
    else:
        metadata[key].global_shape = shape
    with pytest.raises(ValueError, match="expected A_log shape"):
        _infer_kda_checkpoint_flags(config, metadata)


def test_checkpoint_mixed_kda_layouts_fail():
    config, metadata = _kda_metadata()
    metadata["decoder.layers.2.self_attention.A_log"].global_shape = (4,)
    with pytest.raises(ValueError, match="Mixed KDA A_log"):
        _infer_kda_checkpoint_flags(config, metadata)


def test_checkpoint_mixed_gate_biases_fail():
    config, metadata = _kda_metadata()
    metadata["decoder.layers.2.self_attention.gate_out_proj.bias"] = SimpleNamespace(global_shape=(32,))
    with pytest.raises(ValueError, match="Mixed KDA output-gate bias"):
        _infer_kda_checkpoint_flags(config, metadata)


def test_softmax_checkpoint_needs_no_kda_metadata():
    assert _infer_kda_checkpoint_flags({"layer_types": ["full_attention"]}, {}) == {}


def test_legacy_precision_flags_override_transformer_defaults():
    transformer_config = SimpleNamespace(bf16=False, fp16=False, fp8=None, fp8_param=False)
    legacy_args = SimpleNamespace(bf16=True, fp16=False, fp8="hybrid", fp8_param=True)

    view = _LegacyConfigView(transformer_config, legacy_args)

    assert view.bf16 is True
    assert view.fp16 is False
    assert view.fp8 == "hybrid"
    assert view.fp8_param is True
