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

"""Native MCore specification for Apertus2's mixed attention schedule."""

import copy
from typing import Any, cast

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.ssm.kimi_delta_attention import get_kimi_delta_attention_module_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def _attention_layer_types(config) -> tuple[str, ...]:
    """Normalize the explicit HF schedule or MCore's linear-attention frequency."""
    num_layers = int(config.num_layers)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        frequency = getattr(config, "linear_attention_freq", None)
        if frequency is None:
            return ("full_attention",) * num_layers
        if isinstance(frequency, int):
            if frequency <= 0:
                raise ValueError(f"linear_attention_freq must be positive, got {frequency}")
            layer_types = [
                ("full_attention" if index % frequency == frequency - 1 else "linear_attention")
                for index in range(num_layers)
            ]
        else:
            if len(frequency) != num_layers or any(entry not in (0, 1) for entry in frequency):
                raise ValueError(f"linear_attention_freq must contain one 0/1 entry per layer, got {frequency!r}")
            layer_types = ["linear_attention" if entry else "full_attention" for entry in frequency]

    layer_types = tuple(layer_types)
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries; expected {num_layers}")
    unsupported = set(layer_types) - {"full_attention", "linear_attention"}
    if unsupported:
        raise ValueError(f"Unsupported Apertus2 attention layer types: {sorted(unsupported)}")
    return layer_types


def build_apertus2_spec(config, vp_stage=None):
    """Build native MCore layers, replacing attention only on KDA layers."""
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Apertus2 does not support virtual pipeline parallelism")

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        qk_l2_norm=getattr(config, "qk_l2_norm", False),
        vp_stage=vp_stage,
    )
    layer_offset = get_transformer_layer_offset(config, vp_stage)
    layer_types = _attention_layer_types(config)
    layer_specs = block_spec.layer_specs
    if layer_specs is None:
        raise ValueError("Apertus2 requires an explicit per-layer decoder specification")

    for local_idx, original_spec in enumerate(layer_specs):
        global_index = layer_offset + local_idx
        layer_spec = copy.deepcopy(original_spec)
        if layer_types[global_index] == "linear_attention":
            # KDA's fused input projection owns the pre-attention norm.
            submodules = cast(Any, layer_spec.submodules)
            submodules.input_layernorm = IdentityOp
            submodules.self_attention = get_kimi_delta_attention_module_spec(config)
        layer_specs[local_idx] = layer_spec
    return block_spec


__all__ = ["build_apertus2_spec"]
