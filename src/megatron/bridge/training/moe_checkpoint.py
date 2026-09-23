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

"""Read native offloaded expert checkpoints into standard expert modules."""

from dataclasses import replace
from typing import Any

import torch
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)


def _offloaded_key(key: str) -> str:
    if ".experts." in key:
        for projection in (1, 2):
            suffix = f".linear_fc{projection}.weight"
            if key.endswith(suffix):
                return key[: -len(suffix)] + f".weight{projection}"
    return key


def _transpose_shard(shard: ShardedTensor) -> ShardedTensor:
    if shard.flattened_range is not None or len(shard.local_shape) < 2:
        raise ValueError(f"Offloaded expert loading requires unflattened matrix shards: {shard.key}")

    def swap_last_axes(values: tuple[int, ...]) -> tuple[int, ...]:
        return (*values[:-2], values[-1], values[-2])

    return replace(
        shard,
        key=_offloaded_key(shard.key),
        data=shard.data.transpose(-1, -2) if shard.data is not None else None,
        local_shape=swap_last_axes(shard.local_shape),
        global_shape=swap_last_axes(shard.global_shape),
        global_offset=swap_last_axes(shard.global_offset),
        axis_fragmentations=swap_last_axes(shard.axis_fragmentations),
    )


def _map_tensors(value: Any, *, loading: bool) -> Any:
    if isinstance(value, dict):
        return {key: _map_tensors(item, loading=loading) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, loading=loading) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, loading=loading) for item in value)
    if loading and isinstance(value, torch.Tensor):
        return value.transpose(-1, -2)
    if not loading and isinstance(value, ShardedTensor):
        return _transpose_shard(value)
    raise TypeError(f"Unexpected offloaded expert factory value: {type(value).__name__}")


def _offloaded_factory(original: ShardedTensor | ShardedTensorFactory) -> ShardedTensorFactory:
    def build(key: str, data: torch.Tensor, replica_id: Any, flattened_range: slice | None) -> Any:
        if isinstance(original, ShardedTensorFactory):
            native = original.build_fn(key, data, replica_id, flattened_range)
        else:
            native = replace(original, key=key, data=data, replica_id=replica_id, flattened_range=flattened_range)
        return _map_tensors(native, loading=False)

    def merge(loaded: Any) -> torch.Tensor:
        native = _map_tensors(loaded, loading=True)
        return original.merge_fn(native) if isinstance(original, ShardedTensorFactory) else native

    return ShardedTensorFactory(
        original.key,
        original.data,
        build,
        merge,
        replica_id=original.replica_id,
        flattened_range=original.flattened_range,
    )


def adapt_offloaded_expert_state_dict(state_dict: dict[str, Any], *, has_te_extra_state: bool) -> None:
    """Adapt model load requests to native ``weight1/weight2`` expert storage.

    The native offloading module stores matrices in input/output orientation.
    Standard experts use output/input orientation and different checkpoint keys.
    Wrap their existing factories so GLU splitting and TP/EP offsets are retained,
    transpose requests on read, and restore the original layout before loading
    model parameters. Checkpoint files and the model's save schema are unchanged.

    Args:
        state_dict: Model sharded state dictionary, modified in place for loading.
        has_te_extra_state: Whether the source stores TE expert extra state.
    """
    for key, value in state_dict.items():
        if isinstance(value, dict):
            adapt_offloaded_expert_state_dict(value, has_te_extra_state=has_te_extra_state)
        elif isinstance(value, (ShardedTensor, ShardedTensorFactory)) and _offloaded_key(value.key) != value.key:
            state_dict[key] = _offloaded_factory(value)
        elif (
            not has_te_extra_state
            and isinstance(value, ShardedObject)
            and ".experts." in value.key
            and value.key.endswith((".linear_fc1._extra_state", ".linear_fc2._extra_state"))
        ):
            state_dict[key] = LocalNonpersistentObject(value.data)
