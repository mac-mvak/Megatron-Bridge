"""Native offloaded-expert read requests preserve expert and GLU/TP coordinates."""

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

from megatron.bridge.training.moe_checkpoint import adapt_offloaded_expert_state_dict


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("expert", [0, 2])
@pytest.mark.parametrize("projection", [1, 2])
def test_offloaded_read_slices_and_merges_native_matrices(tp_size, expert, projection):
    full = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    source = full.transpose(-1, -2).contiguous()
    key = f"decoder.layers.3.mlp.experts.experts.linear_fc{projection}.weight"
    for tp_rank in range(tp_size):
        if projection == 1:
            gate, up = full[expert].chunk(2)
            local = torch.cat([gate.chunk(tp_size)[tp_rank], up.chunk(tp_size)[tp_rank]])
            original = ShardedTensor.from_rank_offsets(
                key, local, (0, expert, 4), (1, tp_rank, tp_size), prepend_axis_num=1
            )
            original = apply_swiglu_sharded_factory(original, ((0, expert, 4),), False)
        else:
            local = full[expert].chunk(tp_size, dim=1)[tp_rank].contiguous()
            original = ShardedTensor.from_rank_offsets(
                key, local, (0, expert, 4), (2, tp_rank, tp_size), prepend_axis_num=1
            )
        state = {"expert": original}
        adapt_offloaded_expert_state_dict(state, has_te_extra_state=False)
        factory = state["expert"]
        requests = factory.build_fn(factory.key, factory.data, factory.replica_id, None)

        def read(shard):
            assert shard.key == key.replace(f"linear_fc{projection}.weight", f"weight{projection}")
            assert shard.global_shape == tuple(source.shape)
            expert_offset, row, column = shard.global_offset
            rows, columns = shard.local_shape
            return source[expert_offset, row : row + rows, column : column + columns].clone()

        payloads = [read(shard) for shard in requests] if isinstance(requests, list) else read(requests)
        torch.testing.assert_close(factory.merge_fn(payloads), local, atol=0, rtol=0)
        assert original.key == key


def test_offloaded_adaptation_preserves_native_requests_and_nonexpert_weights():
    native = ShardedTensor.from_rank_offsets("decoder.layers.3.mlp.experts.experts.weight1", torch.ones(3, 4))
    dense = ShardedTensor.from_rank_offsets("decoder.layers.0.mlp.linear_fc1.weight", torch.ones(3, 4))
    state = {"model": {"native": native, "dense": dense}}
    adapt_offloaded_expert_state_dict(state, has_te_extra_state=False)
    assert state["model"]["native"] is native
    assert state["model"]["dense"] is dense


@pytest.mark.parametrize("has_extra", [True, False])
def test_native_checkpoint_without_te_expert_state_uses_local_state(has_extra):
    extra = ShardedObject(
        "decoder.layers.3.mlp.experts.experts.linear_fc1._extra_state",
        {"initialized": True},
        (4,),
        (2,),
        replica_id=0,
    )
    state = {"model": {"extra": extra}}
    adapt_offloaded_expert_state_dict(state, has_te_extra_state=has_extra)
    value = state["model"]["extra"]
    assert isinstance(value, ShardedObject if has_extra else LocalNonpersistentObject)
    assert (value.data if has_extra else value.unwrap()) == extra.data
