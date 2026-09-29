# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Recover Apertus2 layout options absent from legacy checkpoint arguments."""

from collections.abc import Mapping, Sequence

from megatron.core.dist_checkpointing.mapping import ShardedTensor


def infer_kda_checkpoint_flags(
    metadata: Mapping[str, ShardedTensor],
    *,
    layer_types: Sequence[str],
    num_value_heads: int,
    key_head_dim: int,
) -> dict[str, bool]:
    """Infer decay layout and gate-bias presence from every KDA layer's metadata.

    Args:
        metadata: Distributed checkpoint tensor metadata, without tensor payloads.
        layer_types: Global attention schedule in layer order.
        num_value_heads: Global KDA value head count.
        key_head_dim: Channels per KDA key head.

    Returns:
        The two HF-compatible KDA layout flags, or an empty mapping for dense attention.

    Raises:
        ValueError: A KDA tensor is missing, malformed, or inconsistent across layers.
    """
    layers = [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
    if not layers:
        return {}
    channels = num_value_heads * key_head_dim
    layouts = set()
    biases = set()
    for index in layers:
        prefix = f"decoder.layers.{index}.self_attention"
        key = f"{prefix}.A_log"
        entry = metadata.get(key)
        shape = tuple(entry.global_shape) if entry is not None else None
        if shape not in ((num_value_heads,), (channels,)):
            raise ValueError(f"{key}: expected A_log shape {(num_value_heads,)} or {(channels,)}, got {shape}")
        layouts.add(shape)
        biases.add(f"{prefix}.gate_out_proj.bias" in metadata)
    if len(layouts) != 1:
        raise ValueError("Mixed KDA A_log layouts cannot be represented by one HF config")
    if len(biases) != 1:
        raise ValueError("Mixed KDA output-gate bias presence cannot be represented by one HF config")
    return {
        "linear_attn_a_log_per_channel": layouts == {(channels,)} and channels != num_value_heads,
        "linear_attn_output_gate_bias": biases == {True},
    }
