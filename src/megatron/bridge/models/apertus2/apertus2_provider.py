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

"""Native Apertus2 provider defaults."""

from dataclasses import dataclass
from typing import Callable

import torch
from megatron.core.activations import sssglu_act
from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention

from megatron.bridge.models.apertus2.apertus2_spec import build_apertus2_spec
from megatron.bridge.models.gpt_provider import GPTModelProvider


def _preserve_kda_decay_parameters(model: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Keep native KDA decay parameters in FP32 through mixed-precision wrapping."""
    for model_chunk in model:
        for module in model_chunk.modules():
            if not isinstance(module, KimiDeltaAttention):
                continue
            module.A_log.data = module.A_log.data.float()
            module.dt_bias.data = module.dt_bias.data.float()
            module._keep_in_float32_parameter_names = ("A_log", "dt_bias")
    return model


@dataclass
class Apertus2ModelProvider(GPTModelProvider):
    """Provider for Apertus2's standard attention / native MCore KDA stack."""

    def __post_init__(self) -> None:
        """Install the native KDA mixed-precision preservation hook."""
        super().__post_init__()
        if not hasattr(self, "_pre_wrap_hooks") or _preserve_kda_decay_parameters not in self._pre_wrap_hooks:
            self.register_pre_wrap_hook(_preserve_kda_decay_parameters, prepend=True)

    # Authoritative Apertus2 checkpoint semantics.  These defaults are intentionally different
    # from the generic GPT provider and prevent a plain provider construction from silently tying
    # the LM head or enabling dropout/biases.
    transformer_layer_spec: Callable = build_apertus2_spec
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    gated_linear_unit: bool = True
    activation_func: Callable = sssglu_act
    qk_layernorm: bool = True
    attention_softmax_in_fp32: bool = True
    scale_embeddings_by_sqrt_hidden: bool = True
    residual_output_scaling: bool = True
    share_embeddings_and_output_weights: bool = False
    transformer_impl: str = "transformer_engine"
    moe_grouped_gemm: bool = True

    # Routing is sigmoid, computed in FP32, and uses a static correction buffer from HF.
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = "sigmoid"
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = False
    moe_router_bias_update_rate: float = 0.0
    moe_router_load_balancing_type: str | list[str] = "quantile_balancing"
    moe_aux_loss_coeff: float | list[float] = 1e-3
    moe_router_quantile_balancing_method: str = "histogram"
    moe_router_quantile_balancing_num_bins: int = 1000
    moe_router_topk: int = 8
    moe_router_topk_scaling_factor: float = 2.5

    # Apertus2 schedules and KDA geometry.
    layer_types: tuple[str, ...] | None = None
    linear_attention_full_rank_output_gate: bool = False
    linear_attention_safe_output_gate: bool = False
    linear_attention_safe_output_gate_lower_bound: float = -5.0
    linear_attention_output_gate_form: str = "per_channel"
    linear_attn_output_gate_bias: bool = True
    linear_attn_a_log_per_channel: bool = False


__all__ = ["Apertus2ModelProvider"]
