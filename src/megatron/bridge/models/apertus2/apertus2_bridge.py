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

"""Hugging Face adapter for the native Apertus2 model family."""

from typing import Any

import torch
from megatron.core.activations import sssglu_act
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.apertus2.apertus2_builder import (
    Apertus2ModelConfig,
    Apertus2TransformerConfig,
)
from megatron.bridge.models.apertus2.apertus2_mapping import (
    build_apertus2_mapping_registry,
)
from megatron.bridge.models.apertus2.apertus2_provider import Apertus2ModelProvider
from megatron.bridge.models.apertus2.apertus2_spec import build_apertus2_spec
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf


def _schedule_from_hf(hf_config: Any) -> tuple[str, ...]:
    """Read and validate the authoritative per-layer attention schedule."""
    num_layers = int(hf_config.num_hidden_layers)
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * num_layers
    layer_types = tuple(layer_types)
    if getattr(hf_config, "sliding_window", None) is not None:
        raise ValueError("Apertus2 Bridge does not support sliding_window attention")
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries; expected {num_layers}")
    unsupported = set(layer_types) - {"full_attention", "linear_attention"}
    if unsupported:
        raise ValueError(
            f"Apertus2 Bridge does not support sliding attention; unsupported layer_types: {sorted(unsupported)}"
        )
    return layer_types


def _moe_schedule_from_hf(hf_config: Any) -> list[int]:
    """Normalize the HF dense/MoE schedule to MCore's explicit zero-based list."""
    num_layers = int(hf_config.num_hidden_layers)
    schedule = getattr(hf_config, "moe_layer_freq", None)
    if schedule is None:
        first_dense = int(getattr(hf_config, "first_k_dense_replace", 0))
        schedule = [0] * first_dense + [1] * (num_layers - first_dense)
    elif isinstance(schedule, int):
        if schedule <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {schedule}")
        schedule = [int(i % schedule == 0) for i in range(num_layers)]
    else:
        schedule = list(schedule)
    if len(schedule) != num_layers or any(value not in (0, 1) for value in schedule):
        raise ValueError(f"moe_layer_freq must contain one 0/1 entry per layer, got {schedule!r}")
    return [int(value) for value in schedule]


def _no_rope_freq_from_hf(hf_config: Any) -> list[int]:
    """Invert HF's ``1=rotated, 0=NoPE`` polarity to MCore's ``1=NoPE`` polarity."""
    num_layers = int(hf_config.num_hidden_layers)
    keep_rope = getattr(hf_config, "no_rope_layers", None)
    if keep_rope is None:
        return [0] * num_layers
    keep_rope = list(keep_rope)
    if len(keep_rope) != num_layers or any(value not in (0, 1) for value in keep_rope):
        raise ValueError(f"no_rope_layers must contain one 0 (NoPE) or 1 (RoPE) entry per layer; got {keep_rope!r}")
    return [1 - int(value) for value in keep_rope]


def _rope_kwargs(hf_config: Any) -> dict[str, Any]:
    """Translate supported HF RoPE settings to MCore's GPT fields."""
    try:
        rotary_base = rope_theta_from_hf(hf_config)
    except ValueError:
        rotary_base = 500000
    rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
    if not isinstance(rope_parameters, dict):
        raise ValueError(f"Unsupported rope_parameters type: {type(rope_parameters).__name__}")
    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
    partial = rope_parameters.get("partial_rotary_factor", 1.0)
    if partial != 1.0:
        raise ValueError(f"Apertus2 requires full-head RoPE (partial_rotary_factor=1), got {partial}")
    if rope_type not in (None, "default", "llama3"):
        raise ValueError(f"Unsupported Apertus2 RoPE variant: {rope_type!r}")
    kwargs: dict[str, Any] = {
        "rotary_base": rotary_base,
        "rotary_percent": 1.0,
        "position_embedding_type": "rope",
        "rope_scaling": rope_type == "llama3",
        "rope_scaling_factor": float(rope_parameters.get("factor", 1.0)),
    }
    if kwargs["rope_scaling"] and kwargs["rope_scaling_factor"] <= 0:
        raise ValueError("llama3 RoPE scaling factor must be positive")
    return kwargs


def _activation_from_hf(name: str):
    """Resolve both standard and MCore's authoritative SSSGLU activation."""
    if name == "sssglu":
        return sssglu_act
    return MegatronModelBridge.hf_to_megatron_activation(name)


def _multiplier_flag(value: Any, expected: float, field_name: str) -> bool:
    """Represent only MCore's exact boolean multiplier semantics."""
    if value is None:
        return False
    value = float(value)
    if abs(value - 1.0) < 1e-12:
        return False
    if abs(value - expected) >= 1e-9 * max(1.0, abs(expected)):
        raise ValueError(f"{field_name}={value} cannot be represented by MCore; expected 1 or {expected}")
    return True


@MegatronModelBridge.register_bridge(  # ty: ignore[invalid-argument-type]
    source="Apertus2ForCausalLM",
    target=GPTModel,
    provider=Apertus2ModelProvider,
    model_type="apertus2",
)
class Apertus2Bridge(MegatronModelBridge[Any, Apertus2ModelProvider, GPTModel]):
    """Bridge Apertus2 configs while retaining MCore's virtual KDA checkpoint keys."""

    MODEL_CONFIG_CLASS = Apertus2ModelConfig
    TRANSFORMER_CONFIG_CLASS = Apertus2TransformerConfig

    def _apertus2_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Translate all config controls that affect model math or state layout."""
        layer_types = _schedule_from_hf(hf_config)
        moe_schedule = _moe_schedule_from_hf(hf_config)
        has_kda = any(kind == "linear_attention" for kind in layer_types)
        bound = getattr(hf_config, "gate_lower_bound", None) if has_kda else None
        if bound is not None and not -5.0 <= float(bound) < 0.0:
            raise ValueError(f"gate_lower_bound must be in [-5, 0), got {bound}")
        if has_kda:
            geometry_names = (
                "linear_conv_kernel_dim",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_num_key_heads",
                "linear_num_value_heads",
            )
            geometry = {name: getattr(hf_config, name, None) for name in geometry_names}
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in geometry.values()):
                raise ValueError(f"KDA geometry must be positive integers, got {geometry!r}")
            if geometry["linear_num_key_heads"] != geometry["linear_num_value_heads"]:
                raise ValueError("KDA key/value head counts must match")
            # ``None`` is the HF config's omitted/default value and means the fork's
            # checkpoint contract (the g_b_proj bias is present).  False is reserved for
            # Kimi-Linear-style bias-free exports, which this adapter intentionally rejects.
            gate_bias = getattr(hf_config, "linear_attn_output_gate_bias", None)
            if gate_bias is not None and gate_bias is not True:
                raise ValueError("Apertus2 Bridge requires the exported KDA g_b_proj bias")
            if bool(getattr(hf_config, "linear_attention_full_rank_output_gate", False)):
                raise ValueError("Apertus2 Bridge requires the low-rank KDA output gate")
        else:
            geometry = {}

        tie_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
        if tie_embeddings:
            raise ValueError("Apertus2 requires untied embeddings and output weights")
        if bool(getattr(hf_config, "attention_bias", False)):
            raise ValueError("Apertus2 attention_bias=True is unsupported by the bias-free checkpoint")

        use_qb = bool(getattr(hf_config, "use_quantile_balancing", False))
        qb_method = getattr(hf_config, "moe_router_quantile_balancing_method", "sigmoid")
        qb_method = {"sigmoid": "histogram", "legacy": "legacy_average"}.get(qb_method, qb_method)
        if use_qb and qb_method not in ("average", "legacy_average", "histogram"):
            raise ValueError(f"Unsupported quantile balancing method: {qb_method!r}")
        # The canonical Apertus2 checkpoints use quantile balancing alone. MCore also
        # accepts a list with an independent sequence auxiliary-loss term; preserve that
        # explicit list when a caller supplies one instead of forcing it on every HF export.
        configured_routing = getattr(hf_config, "moe_router_load_balancing_type", None)
        if configured_routing is not None:
            load_balancing_type = configured_routing
            routing_methods = (
                [load_balancing_type] if isinstance(load_balancing_type, str) else list(load_balancing_type)
            )
            if ("quantile_balancing" in routing_methods) != use_qb:
                raise ValueError(
                    "use_quantile_balancing must match moe_router_load_balancing_type; "
                    f"got use_quantile_balancing={use_qb} and {load_balancing_type!r}"
                )
        elif use_qb:
            load_balancing_type = "quantile_balancing"
        else:
            load_balancing_type = "aux_loss"
        if isinstance(load_balancing_type, list):
            coefficients = getattr(hf_config, "moe_aux_loss_coeff", None)
            if coefficients is None:
                coefficients = [0.0 if name == "quantile_balancing" else 1e-3 for name in load_balancing_type]
            moe_aux_loss_coeff: float | list[float] = list(coefficients)
        else:
            moe_aux_loss_coeff = float(getattr(hf_config, "moe_aux_loss_coeff", 1e-3 if use_qb else 0.0))
        if use_qb and (
            getattr(hf_config, "n_group", 1) not in (None, 1) or getattr(hf_config, "topk_group", 1) not in (None, 1)
        ):
            raise ValueError("Quantile balancing cannot be combined with group-limited routing")
        groups = getattr(hf_config, "n_group", 1)
        group_topk = getattr(hf_config, "topk_group", 1)
        if groups == 1 and group_topk == 1:
            groups = group_topk = None

        topk = int(getattr(hf_config, "num_experts_per_tok", 1))
        norm_topk = bool(getattr(hf_config, "norm_topk_prob", topk > 1))
        if (topk == 1 and norm_topk) or (topk > 1 and not norm_topk):
            raise ValueError("norm_topk_prob does not match MCore's top-k normalization semantics")
        n_shared = int(getattr(hf_config, "n_shared_experts", 1))
        moe_intermediate = int(getattr(hf_config, "moe_intermediate_size", 0))
        if n_shared < 1 or moe_intermediate < 1:
            raise ValueError("Apertus2 requires positive shared and routed expert dimensions")

        hidden_size = int(hf_config.hidden_size)
        num_layers = int(hf_config.num_hidden_layers)
        embedding_multiplier = getattr(hf_config, "embedding_multiplier", 1.0)
        residual_multiplier = getattr(hf_config, "residual_multiplier", 1.0)
        kwargs: dict[str, Any] = {
            "vocab_size": hf_config.vocab_size,
            # Keep the outer GPT model faithful in both the legacy provider and
            # builder-backed config paths. Apertus2 checkpoints always carry a
            # separate LM head; do not rely on the generic GPT default here.
            "share_embeddings_and_output_weights": False,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "kv_channels": hf_config.head_dim,
            "ffn_hidden_size": hf_config.intermediate_size,
            "num_moe_experts": getattr(hf_config, "n_routed_experts", None),
            "moe_ffn_hidden_size": moe_intermediate,
            "moe_router_topk": topk,
            "moe_shared_expert_intermediate_size": n_shared * moe_intermediate,
            "moe_layer_freq": moe_schedule,
            "moe_router_topk_scaling_factor": float(getattr(hf_config, "routed_scaling_factor", 1.0)),
            "moe_router_num_groups": groups,
            "moe_router_group_topk": group_topk,
            "moe_router_pre_softmax": False,
            "moe_router_score_function": "sigmoid",
            "moe_router_dtype": "fp32",
            # The HF router always carries the correction buffer. Enabling it for
            # non-QB routing preserves nonzero checkpoints, while an all-zero buffer is inert.
            "moe_router_enable_expert_bias": bool(getattr(hf_config, "moe_router_enable_expert_bias", not use_qb)),
            "moe_router_bias_update_rate": 0.0,
            "moe_router_load_balancing_type": load_balancing_type,
            "moe_aux_loss_coeff": moe_aux_loss_coeff,
            "moe_router_quantile_balancing_method": qb_method,
            "moe_latent_size": getattr(hf_config, "moe_latent_size", None),
            "layer_types": layer_types,
            "experimental_attention_variant": "kda" if has_kda else None,
            "linear_attention_freq": [int(kind == "linear_attention") for kind in layer_types],
            "linear_attention_qk_norm": "l2norm",
            "linear_attention_v_norm": "none",
            "linear_attention_use_output_gate": True,
            "linear_attention_full_rank_output_gate": False,
            "linear_attention_safe_output_gate": bound is not None,
            "linear_attention_safe_output_gate_lower_bound": (float(bound) if bound is not None else -5.0),
            "linear_attention_output_gate_form": "per_channel",
            "linear_attn_output_gate_bias": True,
            "normalization": "RMSNorm",
            "qk_layernorm": bool(getattr(hf_config, "use_qk_norm", True)),
            "gated_linear_unit": True,
            "add_bias_linear": False,
            "add_qkv_bias": False,
            "attention_output_gate": bool(getattr(hf_config, "attention_output_gate", False)),
            "sandwich_norm": bool(getattr(hf_config, "sandwich_norm", False)),
            "scale_embeddings_by_sqrt_hidden": _multiplier_flag(
                embedding_multiplier, hidden_size**0.5, "embedding_multiplier"
            ),
            "residual_output_scaling": _multiplier_flag(
                residual_multiplier, (2 * num_layers) ** -0.5, "residual_multiplier"
            ),
            "no_rope_freq": _no_rope_freq_from_hf(hf_config),
            "attention_dropout": float(getattr(hf_config, "attention_dropout", 0.0)),
            "hidden_dropout": 0.0,
            "layernorm_epsilon": float(getattr(hf_config, "rms_norm_eps", 1e-5)),
            "init_method_std": float(getattr(hf_config, "initializer_range", 0.02)),
            "seq_length": int(getattr(hf_config, "max_position_embeddings", 8192)),
            "attention_softmax_in_fp32": True,
            "transformer_impl": "transformer_engine",
            "moe_grouped_gemm": True,
            "params_dtype": self.dtype_from_hf(hf_config, default=torch.bfloat16),
            "activation_func": _activation_from_hf(getattr(hf_config, "hidden_act", "silu")),
            "transformer_layer_spec": build_apertus2_spec,
        }
        kwargs.update(geometry)
        kwargs.update(_rope_kwargs(hf_config))
        return kwargs

    def hf_config_to_provider_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Translate an HF config for the legacy provider path."""
        return self._apertus2_kwargs(hf_config)

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Translate an HF config for the Apertus2 builder path."""
        kwargs = self._apertus2_kwargs(hf_config)
        return {key: value for key, value in kwargs.items() if key != "transformer_layer_spec"}

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        """Translate Apertus2's custom activation to its Hugging Face name."""
        if activation_func is sssglu_act or activation_func == "sssglu":
            return "sssglu"
        return super().megatron_to_hf_activation(activation_func)

    @classmethod
    def megatron_to_hf_config(cls, provider: Apertus2ModelProvider) -> dict:
        """Convert a Megatron Apertus2 config into a reloadable HF config."""
        hf_config = super().megatron_to_hf_config(provider)

        # These generic aliases are not fields in the strict Apertus2Config.
        for key in (
            "aux_loss_alpha",
            "hidden_dropout",
            "mlp_bias",
            "num_experts",
            "num_local_experts",
            "partial_rotary_factor",
            "rope_theta",
            "router_aux_loss_coef",
            "scoring_func",
        ):
            hf_config.pop(key, None)

        num_layers = int(provider.num_layers)
        layer_types = getattr(provider, "layer_types", None)
        if layer_types is None:
            linear_attention_freq = getattr(provider, "linear_attention_freq", None)
            if linear_attention_freq is None:
                layer_types = ["full_attention"] * num_layers
            elif isinstance(linear_attention_freq, int):
                if linear_attention_freq <= 0:
                    raise ValueError(f"linear_attention_freq must be positive, got {linear_attention_freq}")
                layer_types = [
                    ("full_attention" if (index + 1) % linear_attention_freq == 0 else "linear_attention")
                    for index in range(num_layers)
                ]
            else:
                pattern = list(linear_attention_freq)
                if len(pattern) != num_layers or any(value not in (0, 1) for value in pattern):
                    raise ValueError(f"linear_attention_freq must contain one 0/1 entry per layer; got {pattern!r}")
                layer_types = ["linear_attention" if value else "full_attention" for value in pattern]
        else:
            layer_types = list(layer_types)
        if len(layer_types) != num_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries; expected {num_layers}")
        unsupported = set(layer_types) - {"full_attention", "linear_attention"}
        if unsupported:
            raise ValueError(f"Unsupported Apertus2 layer types: {sorted(unsupported)}")
        has_kda = "linear_attention" in layer_types

        moe_layer_freq = getattr(provider, "moe_layer_freq", None)
        if moe_layer_freq is None:
            moe_layer_freq = [1] * num_layers
        elif isinstance(moe_layer_freq, int):
            if moe_layer_freq <= 0:
                raise ValueError(f"moe_layer_freq must be positive, got {moe_layer_freq}")
            moe_layer_freq = [int(index % moe_layer_freq == 0) for index in range(num_layers)]
        else:
            moe_layer_freq = list(moe_layer_freq)
        if len(moe_layer_freq) != num_layers or any(value not in (0, 1) for value in moe_layer_freq):
            raise ValueError(f"moe_layer_freq must contain one 0/1 entry per layer; got {moe_layer_freq!r}")
        first_dense = 0
        while first_dense < num_layers and not moe_layer_freq[first_dense]:
            first_dense += 1

        no_rope_freq = getattr(provider, "no_rope_freq", None)
        if no_rope_freq is None:
            skips_rope = [0] * num_layers
        elif isinstance(no_rope_freq, int):
            if no_rope_freq <= 0 or num_layers % no_rope_freq:
                raise ValueError(
                    f"no_rope_freq must be positive and divide num_layers; got {no_rope_freq} for {num_layers} layers"
                )
            skips_rope = ([0] * (no_rope_freq - 1) + [1]) * (num_layers // no_rope_freq)
        else:
            skips_rope = list(no_rope_freq)
        if len(skips_rope) != num_layers or any(value not in (0, 1) for value in skips_rope):
            raise ValueError(f"no_rope_freq must contain one 0/1 entry per layer; got {skips_rope!r}")

        routing = getattr(provider, "moe_router_load_balancing_type", None)
        routing_methods = [routing] if isinstance(routing, str) else list(routing or ())
        use_qb = "quantile_balancing" in routing_methods
        qb_method = getattr(provider, "moe_router_quantile_balancing_method", "histogram")
        qb_method = {
            "average": "sigmoid",
            "histogram": "sigmoid",
            "legacy_average": "legacy",
        }.get(qb_method, qb_method)
        if use_qb and qb_method not in ("sigmoid", "legacy"):
            raise ValueError(f"Unsupported quantile balancing method: {qb_method!r}")

        shared_size = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe_intermediate_size = getattr(provider, "moe_ffn_hidden_size", None)
        if not shared_size or not moe_intermediate_size or shared_size % moe_intermediate_size:
            raise ValueError("moe_shared_expert_intermediate_size must be a positive multiple of moe_ffn_hidden_size")
        topk = int(provider.moe_router_topk)
        groups = getattr(provider, "moe_router_num_groups", None) or 1
        group_topk = getattr(provider, "moe_router_group_topk", None) or 1

        rope_parameters: dict[str, Any] = {
            "rope_type": "default",
            "rope_theta": float(provider.rotary_base),
            "partial_rotary_factor": 1.0,
        }
        if bool(getattr(provider, "rope_scaling", False)):
            rope_parameters.update(
                rope_type="llama3",
                factor=float(provider.rope_scaling_factor),
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            )

        hf_config.update(
            {
                "architectures": ["Apertus2KDAForCausalLM" if has_kda else "Apertus2ForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_apertus2.Apertus2Config",
                    "AutoModel": "modeling_apertus2.Apertus2Model",
                    "AutoModelForCausalLM": "modeling_apertus2.Apertus2ForCausalLM",
                },
                "model_type": "apertus2",
                "rope_parameters": rope_parameters,
                "layer_types": layer_types,
                "no_rope_layers": [1 - int(value) for value in skips_rope],
                "moe_layer_freq": [int(value) for value in moe_layer_freq],
                "first_k_dense_replace": first_dense,
                "n_routed_experts": provider.num_moe_experts,
                "n_shared_experts": shared_size // moe_intermediate_size,
                "moe_intermediate_size": moe_intermediate_size,
                "num_experts_per_tok": topk,
                "routed_scaling_factor": float(provider.moe_router_topk_scaling_factor),
                "n_group": groups,
                "topk_group": group_topk,
                "norm_topk_prob": topk > 1,
                "sandwich_norm": bool(getattr(provider, "sandwich_norm", False)),
                "moe_latent_size": getattr(provider, "moe_latent_size", None),
                "use_quantile_balancing": use_qb,
                "moe_router_quantile_balancing_method": qb_method,
                "attention_output_gate": bool(getattr(provider, "attention_output_gate", False)),
                "embedding_multiplier": (
                    float(provider.hidden_size) ** 0.5
                    if bool(getattr(provider, "scale_embeddings_by_sqrt_hidden", False))
                    else 1.0
                ),
                "residual_multiplier": (
                    (2 * num_layers) ** -0.5 if bool(getattr(provider, "residual_output_scaling", False)) else 1.0
                ),
            }
        )
        if has_kda:
            hf_config.update(
                {
                    "linear_num_key_heads": provider.linear_num_key_heads,
                    "linear_num_value_heads": provider.linear_num_value_heads,
                    "linear_key_head_dim": provider.linear_key_head_dim,
                    "linear_value_head_dim": provider.linear_value_head_dim,
                    "linear_conv_kernel_dim": provider.linear_conv_kernel_dim,
                    "gate_lower_bound": (
                        float(provider.linear_attention_safe_output_gate_lower_bound)
                        if bool(getattr(provider, "linear_attention_safe_output_gate", False))
                        else None
                    ),
                    "linear_attn_output_gate_bias": bool(getattr(provider, "linear_attn_output_gate_bias", True)),
                }
            )
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return schedule-specific virtual-key mappings."""
        return build_apertus2_mapping_registry(self.hf_config)


__all__ = ["Apertus2Bridge"]
