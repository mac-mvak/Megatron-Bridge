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

"""Virtual-key mappings for Apertus2 Hugging Face conversion."""

from typing import cast

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVGMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)


class KDAInProjMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Convert the runtime-fused KDA input projection to six HF matrices.

    MCore exposes ``in_proj.weight`` as one parameter while DCP expands it to virtual
    ``weight.query``/... keys.  Bridge conversion walks runtime parameters, so it needs this
    companion mapping; the explicit virtual mappings below remain authoritative for DCP keys.
    """

    _names = ("query", "key", "value", "decay_low_rank", "gate_low_rank", "beta")

    def __init__(self, megatron_param: str, **hf_params: str):
        if tuple(hf_params) != self._names:
            raise ValueError(f"KDA input projection names must be {self._names}, got {tuple(hf_params)}")
        super().__init__(megatron_param, hf_params)
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def resolve(self, captures):
        resolved_megatron, resolved_hf = self._resolve_names(captures)
        return type(self)(resolved_megatron, **cast(dict[str, str], resolved_hf))

    def hf_to_megatron(self, hf_weights, megatron_module):
        merged = None
        if self.tp_rank == 0:
            merged = torch.cat([hf_weights[name] for name in self._names], dim=0)
        return self._tp_mapping.hf_to_megatron(cast(torch.Tensor, merged), megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        gathered = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not gathered:
            return {}
        fused = next(iter(gathered.values()))
        weight = getattr(megatron_module, "weight", None) if megatron_module is not None else None
        split_shapes = self.broadcast_obj_from_pp_rank(
            getattr(weight, "kda_split_shapes", None), "kda_in_proj_split_shapes"
        )
        if split_shapes is None:
            raise ValueError("KDA in_proj.weight is missing kda_split_shapes metadata")
        sections = torch.split(fused, list(split_shapes), dim=0)
        if len(sections) != len(self._names):
            raise ValueError(f"KDA in_proj split has {len(sections)} sections, expected 6")
        hf_param = cast(dict[str, str], self.hf_param)
        return {hf_param[name]: section for name, section in zip(self._names, sections)}


class KDAConv1dMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Convert the fused KDA depthwise convolution to q/k/v HF tensors."""

    _names = ("query", "key", "value")

    def __init__(self, megatron_param: str, **hf_params: str):
        if tuple(hf_params) != self._names:
            raise ValueError(f"KDA convolution names must be {self._names}, got {tuple(hf_params)}")
        super().__init__(megatron_param, hf_params)
        # The Conv1d module itself has no parallelism class, but MCore marks its
        # output channels as TP-sharded in GDN's checkpoint path.
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def _sections(self, megatron_module) -> tuple[int, int, int]:
        sections = None
        if megatron_module is not None:
            config = self._get_config(megatron_module)
            qk_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
            value_dim = int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
            sections = (qk_dim, qk_dim, value_dim)
        sections = self.broadcast_obj_from_pp_rank(sections, "kda_conv_sections")
        if sections is None:
            raise ValueError("KDA conv1d geometry is unavailable")
        return tuple(sections)

    def resolve(self, captures):
        resolved_megatron, resolved_hf = self._resolve_names(captures)
        return type(self)(resolved_megatron, **cast(dict[str, str], resolved_hf))

    def hf_to_megatron(self, hf_weights, megatron_module):
        merged = None
        if self.tp_rank == 0:
            merged = torch.cat([hf_weights[name] for name in self._names], dim=0)
        return self._tp_mapping.hf_to_megatron(cast(torch.Tensor, merged), megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        gathered = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not gathered:
            return {}
        fused = next(iter(gathered.values()))
        sections = torch.split(fused, list(self._sections(megatron_module)), dim=0)
        hf_param = cast(dict[str, str], self.hf_param)
        return {hf_param[name]: section for name, section in zip(self._names, sections)}


def _model_mappings(config, *, include_expert_bias: bool) -> MegatronMappingRegistry:
    """Build mappings for the actual per-layer schedules in ``config``."""
    num_layers = int(config.num_hidden_layers)
    layer_types = tuple(getattr(config, "layer_types", None) or ("full_attention",) * num_layers)
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries; expected {num_layers}")
    if set(layer_types) - {"full_attention", "linear_attention"}:
        raise ValueError(f"Unsupported Apertus2 layer_types: {layer_types!r}")

    moe_freq = getattr(config, "moe_layer_freq", None)
    if moe_freq is None:
        first_dense = int(getattr(config, "first_k_dense_replace", 0))
        moe_freq = [int(index >= first_dense) for index in range(num_layers)]
    elif isinstance(moe_freq, int):
        if moe_freq <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {moe_freq}")
        moe_freq = [int(index % moe_freq == 0) for index in range(num_layers)]
    else:
        moe_freq = list(moe_freq)
    if len(moe_freq) != num_layers or any(value not in (0, 1) for value in moe_freq):
        raise ValueError(f"moe_layer_freq must contain one 0/1 entry per layer, got {moe_freq!r}")

    mappings = [
        AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        AutoMapping("output_layer.weight", "lm_head.weight"),
        ReplicatedMapping("decoder.final_layernorm.weight", "model.norm.weight"),
    ]
    for index, layer_type in enumerate(layer_types):
        layer = f"decoder.layers.{index}"
        hf_layer = f"model.layers.{index}"
        attn = f"{layer}.self_attention"
        hf_attn = f"{hf_layer}.self_attn"
        if layer_type == "linear_attention":
            mappings.append(
                KDAInProjMapping(
                    f"{attn}.in_proj.weight",
                    query=f"{hf_attn}.q_proj.weight",
                    key=f"{hf_attn}.k_proj.weight",
                    value=f"{hf_attn}.v_proj.weight",
                    decay_low_rank=f"{hf_attn}.f_a_proj.weight",
                    gate_low_rank=f"{hf_attn}.g_a_proj.weight",
                    beta=f"{hf_attn}.b_proj.weight",
                )
            )
            mappings.append(
                KDAConv1dMapping(
                    f"{attn}.conv1d.weight",
                    query=f"{hf_attn}.q_conv1d.weight",
                    key=f"{hf_attn}.k_conv1d.weight",
                    value=f"{hf_attn}.v_conv1d.weight",
                )
            )
            mappings.extend(
                ColumnParallelMapping(f"{attn}.in_proj.weight.{name}", f"{hf_attn}.{hf_name}.weight")
                for name, hf_name in (
                    ("query", "q_proj"),
                    ("key", "k_proj"),
                    ("value", "v_proj"),
                    ("decay_low_rank", "f_a_proj"),
                    ("gate_low_rank", "g_a_proj"),
                    ("beta", "b_proj"),
                )
            )
            mappings.extend(
                ColumnParallelMapping(f"{attn}.conv1d.weight.{name}", f"{hf_attn}.{hf_name}_conv1d.weight")
                for name, hf_name in (("query", "q"), ("key", "k"), ("value", "v"))
            )
            mappings.extend(
                [
                    ReplicatedMapping(
                        f"{attn}.in_proj.layer_norm_weight",
                        f"{hf_attn.rsplit('.', 1)[0]}.attention_layernorm.weight",
                    ),
                    ColumnParallelMapping(f"{attn}.A_log", f"{hf_attn}.A_log"),
                    ColumnParallelMapping(f"{attn}.dt_bias", f"{hf_attn}.dt_bias"),
                    ColumnParallelMapping(f"{attn}.decay_out_proj.weight", f"{hf_attn}.f_b_proj.weight"),
                    ColumnParallelMapping(f"{attn}.gate_out_proj.weight", f"{hf_attn}.g_b_proj.weight"),
                    ColumnParallelMapping(f"{attn}.gate_out_proj.bias", f"{hf_attn}.g_b_proj.bias"),
                    ReplicatedMapping(f"{attn}.out_norm.weight", f"{hf_attn}.o_norm.weight"),
                    RowParallelMapping(f"{attn}.out_proj.weight", f"{hf_attn}.o_proj.weight"),
                ]
            )
        else:
            qkv_mapping = (
                QKVGMapping(
                    f"{attn}.linear_qkv.weight",
                    q=f"{hf_attn}.q_proj.weight",
                    k=f"{hf_attn}.k_proj.weight",
                    v=f"{hf_attn}.v_proj.weight",
                    g=f"{hf_attn}.g_proj.weight",
                )
                if getattr(config, "attention_output_gate", False)
                else QKVMapping(
                    f"{attn}.linear_qkv.weight",
                    q=f"{hf_attn}.q_proj.weight",
                    k=f"{hf_attn}.k_proj.weight",
                    v=f"{hf_attn}.v_proj.weight",
                )
            )
            mappings.extend(
                [
                    ReplicatedMapping(
                        f"{attn}.linear_qkv.layer_norm_weight",
                        f"{hf_layer}.attention_layernorm.weight",
                    ),
                    qkv_mapping,
                    RowParallelMapping(f"{attn}.linear_proj.weight", f"{hf_attn}.o_proj.weight"),
                ]
            )
            if getattr(config, "qk_layernorm", getattr(config, "use_qk_norm", False)):
                mappings.extend(
                    [
                        ReplicatedMapping(f"{attn}.q_layernorm.weight", f"{hf_attn}.q_norm.weight"),
                        ReplicatedMapping(f"{attn}.k_layernorm.weight", f"{hf_attn}.k_norm.weight"),
                    ]
                )

        if getattr(config, "sandwich_norm", False):
            mappings.extend(
                [
                    ReplicatedMapping(
                        f"{layer}.post_self_attn_layernorm.weight",
                        f"{hf_layer}.post_attention_layernorm.weight",
                    ),
                    ReplicatedMapping(
                        f"{layer}.post_mlp_layernorm.weight",
                        f"{hf_layer}.post_feedforward_layernorm.weight",
                    ),
                ]
            )

        if moe_freq[index]:
            mappings.extend(
                [
                    ReplicatedMapping(
                        f"{layer}.pre_mlp_layernorm.weight",
                        f"{hf_layer}.feedforward_layernorm.weight",
                    ),
                    ReplicatedMapping(f"{layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
                    ReplicatedMapping(
                        f"{layer}.mlp.fc1_latent_proj.weight",
                        f"{hf_layer}.mlp.latent_down_proj.weight",
                    ),
                    ReplicatedMapping(
                        f"{layer}.mlp.fc2_latent_proj.weight",
                        f"{hf_layer}.mlp.latent_up_proj.weight",
                    ),
                    # The runtime TEGroupedMLP names are ``experts.linear_fc*``;
                    # torch_dist's expanded virtual schema in the target artifact adds
                    # the second ``experts`` segment. Keep both spellings mapped.
                    GatedMLPMapping(
                        f"{layer}.mlp.experts.linear_fc1.weight*",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                    RowParallelMapping(
                        f"{layer}.mlp.experts.linear_fc2.weight*",
                        f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                    GatedMLPMapping(
                        f"{layer}.mlp.experts.experts.linear_fc1.weight*",
                        gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                        up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
                    ),
                    RowParallelMapping(
                        f"{layer}.mlp.experts.experts.linear_fc2.weight*",
                        f"{hf_layer}.mlp.experts.*.down_proj.weight",
                    ),
                    GatedMLPMapping(
                        f"{layer}.mlp.shared_experts.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
                        up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
                    ),
                    RowParallelMapping(
                        f"{layer}.mlp.shared_experts.linear_fc2.weight",
                        f"{hf_layer}.mlp.shared_experts.down_proj.weight",
                    ),
                ]
            )
            if include_expert_bias:
                mappings.append(
                    ReplicatedMapping(
                        f"{layer}.mlp.router.expert_bias",
                        f"{hf_layer}.mlp.gate.e_score_correction_bias",
                    )
                )
            if getattr(config, "use_quantile_balancing", False):
                mappings.append(ReplicatedMapping(f"{layer}.mlp.router.qb_beta", f"{hf_layer}.mlp.gate.qb_beta"))
        else:
            mappings.extend(
                [
                    ReplicatedMapping(
                        f"{layer}.mlp.linear_fc1.layer_norm_weight",
                        f"{hf_layer}.feedforward_layernorm.weight",
                    ),
                    GatedMLPMapping(
                        f"{layer}.mlp.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.gate_proj.weight",
                        up=f"{hf_layer}.mlp.up_proj.weight",
                    ),
                    RowParallelMapping(
                        f"{layer}.mlp.linear_fc2.weight",
                        f"{hf_layer}.mlp.down_proj.weight",
                    ),
                ]
            )
    return MegatronMappingRegistry(*mappings)


def build_apertus2_mapping_registry(config=None) -> MegatronMappingRegistry:
    """Return schedule-aware mappings; ``config=None`` gives the target default schedule."""
    if config is None:
        # Keep a useful registry for direct callers without an HF config.  Conversion callers pass
        # their config and therefore take the exact schedule-specific path above.
        class _Default:
            num_hidden_layers = 12
            layer_types = (
                ("linear_attention",) * 3
                + ("full_attention",)
                + ("linear_attention",) * 3
                + ("full_attention",)
                + ("linear_attention",) * 3
                + ("full_attention",)
            )
            moe_layer_freq = [0] + [1] * 11
            attention_output_gate = False
            qk_layernorm = True
            sandwich_norm = False
            use_quantile_balancing = True
            moe_router_enable_expert_bias = False

        config = _Default()
    return _model_mappings(
        config,
        include_expert_bias=bool(
            getattr(
                config,
                "moe_router_enable_expert_bias",
                not bool(getattr(config, "use_quantile_balancing", False)),
            )
        ),
    )


__all__ = ["build_apertus2_mapping_registry"]
