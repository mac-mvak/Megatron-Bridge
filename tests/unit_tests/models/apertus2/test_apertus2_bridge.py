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

"""Unit tests for the Apertus2 model bridge."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.activations import sssglu_act
from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention
from transformers import PretrainedConfig

from megatron.bridge.models.apertus2.apertus2_bridge import Apertus2Bridge
from megatron.bridge.models.apertus2.apertus2_mapping import (
    build_apertus2_mapping_registry,
)
from megatron.bridge.models.apertus2.apertus2_provider import (
    Apertus2ModelProvider,
    _preserve_kda_decay_parameters,
)


def _hf_config(**overrides):
    values = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_output_gate": True,
        "embedding_multiplier": 8.0,
        "gate_lower_bound": -5.0,
        "head_dim": 8,
        "hidden_act": "sssglu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_attn_output_gate_bias": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 8,
        "max_position_embeddings": 8192,
        "moe_intermediate_size": 32,
        "moe_latent_size": 16,
        "moe_layer_freq": [0, 1, 1, 1],
        "moe_router_quantile_balancing_method": "sigmoid",
        "n_group": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "no_rope_layers": [1, 1, 0, 1],
        "norm_topk_prob": True,
        "num_attention_heads": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "num_key_value_heads": 4,
        "residual_multiplier": (2 * 4) ** -0.5,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "partial_rotary_factor": 1.0,
        },
        "routed_scaling_factor": 2.5,
        "sandwich_norm": True,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": torch.bfloat16,
        "use_qk_norm": True,
        "use_quantile_balancing": True,
        "vocab_size": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider(**overrides):
    values = {
        "activation_func": sssglu_act,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "attention_dropout": 0.0,
        "attention_output_gate": True,
        "bf16": True,
        "ffn_hidden_size": 128,
        "hidden_dropout": 0.0,
        "hidden_size": 64,
        "init_method_std": 0.02,
        "kv_channels": 8,
        "layer_types": (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        "layernorm_epsilon": 1e-5,
        "linear_attention_freq": [1, 1, 1, 0],
        "linear_attention_safe_output_gate": True,
        "linear_attention_safe_output_gate_lower_bound": -5.0,
        "linear_attn_output_gate_bias": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 8,
        "moe_ffn_hidden_size": 32,
        "moe_latent_size": 16,
        "moe_layer_freq": [0, 1, 1, 1],
        "moe_router_group_topk": None,
        "moe_router_load_balancing_type": "quantile_balancing",
        "moe_router_num_groups": None,
        "moe_router_quantile_balancing_method": "histogram",
        "moe_router_score_function": "sigmoid",
        "moe_router_topk": 2,
        "moe_router_topk_scaling_factor": 2.5,
        "moe_shared_expert_intermediate_size": 32,
        "no_rope_freq": [0, 0, 1, 0],
        "normalization": "RMSNorm",
        "num_attention_heads": 8,
        "num_layers": 4,
        "num_moe_experts": 8,
        "num_query_groups": 4,
        "residual_output_scaling": True,
        "rope_scaling": True,
        "rope_scaling_factor": 8.0,
        "rotary_base": 500000.0,
        "rotary_percent": 1.0,
        "sandwich_norm": True,
        "scale_embeddings_by_sqrt_hidden": True,
        "seq_length": 8192,
        "share_embeddings_and_output_weights": False,
        "vocab_size": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
class TestApertus2ConfigConversion:
    @pytest.mark.parametrize("per_channel", [True, False, None])
    @pytest.mark.parametrize("bias", [True, False, None])
    def test_kda_flags_roundtrip(self, per_channel, bias):
        config = _hf_config(linear_attn_a_log_per_channel=per_channel, linear_attn_output_gate_bias=bias)
        provider_kwargs = Apertus2Bridge().hf_config_to_provider_kwargs(config)
        builder = Apertus2Bridge().hf_config_to_model_config(config)
        expected_channel = per_channel is True
        expected_bias = bias is not False
        for converted in (SimpleNamespace(**provider_kwargs), builder):
            assert converted.linear_attn_a_log_per_channel is expected_channel
            assert converted.linear_attn_output_gate_bias is expected_bias
            restored = Apertus2Bridge.megatron_to_hf_config(converted)
            assert restored["linear_attn_a_log_per_channel"] is expected_channel
            assert restored["linear_attn_output_gate_bias"] is expected_bias

    @pytest.mark.parametrize("field", ["linear_attn_a_log_per_channel", "linear_attn_output_gate_bias"])
    @pytest.mark.parametrize("value", ["false", 1, 0, []])
    def test_kda_flags_reject_non_boolean_values(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be a boolean"):
            Apertus2Bridge().hf_config_to_provider_kwargs(_hf_config(**{field: value}))

    def test_megachonk_config_preserves_checkpoint_semantics(self):
        config = PretrainedConfig(**json.loads((Path(__file__).parent / "fixtures/megachonk_config.json").read_text()))
        result = Apertus2Bridge().hf_config_to_model_config(config)
        assert result.num_layers == 61
        assert result.linear_attention_freq.count(1) == 45
        assert result.moe_layer_freq == [0] * 3 + [1] * 58
        assert result.no_rope_freq == [1] * 61
        assert result.linear_attn_a_log_per_channel is True
        assert result.linear_attn_output_gate_bias is False
        assert result.linear_key_head_dim == result.linear_value_head_dim == 128
        assert result.linear_num_key_heads == result.linear_num_value_heads == 64
        assert result.moe_latent_size == 4096
        assert result.num_moe_experts == 256
        assert result.sandwich_norm and result.attention_output_gate
        assert result.scale_embeddings_by_sqrt_hidden and result.residual_output_scaling
        assert result.bf16 and result.params_dtype is torch.bfloat16
        assert result.moe_router_quantile_balancing_method == "histogram"

    def test_hf_to_megatron_preserves_apertus2_semantics(self):
        result = Apertus2Bridge().hf_config_to_provider_kwargs(_hf_config())

        assert result["layer_types"] == tuple(_hf_config().layer_types)
        assert result["linear_attention_freq"] == [1, 1, 1, 0]
        assert result["no_rope_freq"] == [0, 0, 1, 0]
        assert result["moe_layer_freq"] == [0, 1, 1, 1]
        assert result["moe_router_load_balancing_type"] == "quantile_balancing"
        assert result["moe_router_quantile_balancing_method"] == "histogram"
        assert result["moe_router_enable_expert_bias"] is False
        assert result["bf16"] is True
        assert result["fp16"] is False
        assert result["params_dtype"] is torch.bfloat16
        assert result["activation_func"] is sssglu_act

    def test_builder_config_preserves_apertus2_fields(self):
        result = Apertus2Bridge().hf_config_to_model_config(_hf_config())

        assert result.num_layers == 4
        assert result.num_moe_experts == 8
        assert result.moe_layer_freq == [0, 1, 1, 1]
        assert result.experimental_attention_variant == "kda"
        assert result.linear_attention_freq == [1, 1, 1, 0]
        assert result.moe_router_load_balancing_type == "quantile_balancing"
        assert result.moe_router_quantile_balancing_method == "histogram"
        assert result.moe_shared_expert_intermediate_size == 32
        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype is torch.bfloat16
        assert result.share_embeddings_and_output_weights is False
        assert result.transformer.layer_types == tuple(_hf_config().layer_types)
        assert result.transformer.linear_attn_output_gate_bias is True
        assert result.transformer_layer_spec is None

    def test_non_qb_routing_enables_static_correction_buffer(self):
        result = Apertus2Bridge().hf_config_to_provider_kwargs(_hf_config(use_quantile_balancing=False))

        assert result["moe_router_load_balancing_type"] == "aux_loss"
        assert result["moe_router_enable_expert_bias"] is True

    @pytest.mark.parametrize(
        ("use_qb", "routing"),
        [
            (False, "quantile_balancing"),
            (True, "aux_loss"),
            (False, ["quantile_balancing", "seq_aux_loss"]),
        ],
    )
    def test_rejects_contradictory_quantile_balancing_config(self, use_qb, routing):
        config = _hf_config(
            use_quantile_balancing=use_qb,
            moe_router_load_balancing_type=routing,
        )

        with pytest.raises(ValueError, match="use_quantile_balancing must match"):
            Apertus2Bridge().hf_config_to_provider_kwargs(config)

    def test_megatron_to_hf_config_is_reloadable(self):
        result = Apertus2Bridge.megatron_to_hf_config(_provider())

        assert result["architectures"] == ["Apertus2KDAForCausalLM"]
        assert result["model_type"] == "apertus2"
        assert result["hidden_act"] == "sssglu"
        assert result["torch_dtype"] == "bfloat16"
        assert result["layer_types"] == list(_provider().layer_types)
        assert result["no_rope_layers"] == [1, 1, 0, 1]
        assert result["moe_layer_freq"] == [0, 1, 1, 1]
        assert result["first_k_dense_replace"] == 1
        assert result["n_routed_experts"] == 8
        assert result["n_shared_experts"] == 1
        assert result["use_quantile_balancing"] is True
        assert result["moe_router_quantile_balancing_method"] == "sigmoid"
        assert result["rope_parameters"] == {
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "partial_rotary_factor": 1.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        assert result["linear_num_key_heads"] == 4
        assert result["linear_num_value_heads"] == 4
        assert result["gate_lower_bound"] == -5.0
        assert result["linear_attn_output_gate_bias"] is True
        assert "num_experts" not in result
        assert "num_local_experts" not in result
        assert "rope_theta" not in result

    def test_export_keeps_kda_decay_parameters_in_fp32(self):
        weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, dtype=torch.float32),
            "model.layers.0.self_attn.A_log": torch.ones(2, dtype=torch.bfloat16),
            "model.layers.0.self_attn.dt_bias": torch.ones(2, dtype=torch.bfloat16),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(4, dtype=torch.bfloat16),
            "integer_buffer": torch.ones(2, dtype=torch.int64),
        }

        result = Apertus2Bridge._cast_export_weight_dtype(weights, torch.bfloat16)

        assert result["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16
        assert result["model.layers.0.self_attn.A_log"].dtype is torch.float32
        assert result["model.layers.0.self_attn.dt_bias"].dtype is torch.float32
        assert result["model.layers.1.mlp.gate.e_score_correction_bias"].dtype is torch.float32
        assert result["integer_buffer"].dtype is torch.int64

    def test_softmax_export_uses_standard_architecture_and_omits_kda_geometry(self):
        result = Apertus2Bridge.megatron_to_hf_config(
            _provider(
                layer_types=("full_attention",) * 4,
                linear_attention_freq=[0, 0, 0, 0],
                linear_attention_safe_output_gate=False,
            )
        )

        assert result["architectures"] == ["Apertus2ForCausalLM"]
        assert "linear_num_key_heads" not in result
        assert "gate_lower_bound" not in result


@pytest.mark.unit
class TestApertus2MixedPrecision:
    @pytest.mark.parametrize("construction", ["provider", "builder"])
    def test_router_state_survives_bfloat16_wrapping(self, construction):
        from megatron.training.models.dist_utils import _wrap_with_mp_wrapper

        from megatron.bridge.models.model_provider import _apply_mixed_precision_wrapper

        class Router(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("qb_beta", torch.tensor([-0.1234567, 0.2345678]))
                self.register_buffer("expert_bias", torch.tensor([0.01234567, -0.02345678]))

            def _maintain_float32_expert_bias(self):
                self.qb_beta = self.qb_beta.float()
                self.expert_bias = self.expert_bias.float()

        router = Router()
        expected = {name: value.clone() for name, value in router.named_buffers()}
        config = SimpleNamespace(bf16=True, fp16=False)

        def wrapper(config, model):
            return model.bfloat16()

        if construction == "provider":
            _apply_mixed_precision_wrapper([router], config, wrapper)
        else:
            _wrap_with_mp_wrapper([router], config, wrapper)
        for name, value in router.named_buffers():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_builder_wrapper_keeps_fp32_decay_values_exact(self):
        from megatron.training.models.dist_utils import _wrap_with_mp_wrapper

        module = KimiDeltaAttention.__new__(KimiDeltaAttention)
        torch.nn.Module.__init__(module)
        values = torch.tensor([0.1234567, -0.2345678], dtype=torch.float32)
        module.A_log = torch.nn.Parameter(values.clone())
        module.dt_bias = torch.nn.Parameter(values.clone())
        _preserve_kda_decay_parameters([module])
        _wrap_with_mp_wrapper([module], SimpleNamespace(fp16=False, bf16=True), lambda config, model: model.bfloat16())
        assert module.A_log.dtype is torch.float32
        assert module.dt_bias.dtype is torch.float32
        torch.testing.assert_close(module.A_log, values, rtol=0, atol=0)
        torch.testing.assert_close(module.dt_bias, values, rtol=0, atol=0)

    @pytest.mark.parametrize("field", ["a_log_per_channel", "output_gate_bias"])
    def test_native_kda_rejects_non_boolean_options(self, field):
        with pytest.raises(ValueError, match=f"{field} must be a boolean"):
            KimiDeltaAttention(None, None, **{field: "false"})

    def test_provider_registers_kda_precision_hook(self):
        provider = Apertus2ModelProvider()

        assert _preserve_kda_decay_parameters in provider._pre_wrap_hooks

    def test_preserves_kda_decay_parameters_in_fp32(self):
        module = KimiDeltaAttention.__new__(KimiDeltaAttention)
        torch.nn.Module.__init__(module)
        module.A_log = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        module.dt_bias = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

        result = _preserve_kda_decay_parameters([module])

        assert result == [module]
        assert module.A_log.dtype is torch.float32
        assert module.dt_bias.dtype is torch.float32
        assert module._keep_in_float32_parameter_names == ("A_log", "dt_bias")


@pytest.mark.unit
class TestApertus2MappingRegistry:
    def test_non_qb_registry_maps_expert_correction_bias(self):
        registry = build_apertus2_mapping_registry(_hf_config(use_quantile_balancing=False))
        megatron_params = {str(mapping.megatron_param) for mapping in registry.mappings}

        assert "decoder.layers.1.mlp.router.expert_bias" in megatron_params
        assert "decoder.layers.1.mlp.router.qb_beta" not in megatron_params

    def test_qb_registry_maps_beta_instead_of_correction_bias(self):
        registry = build_apertus2_mapping_registry(_hf_config())
        megatron_params = {str(mapping.megatron_param) for mapping in registry.mappings}

        assert "decoder.layers.1.mlp.router.qb_beta" in megatron_params
        assert "decoder.layers.1.mlp.router.expert_bias" not in megatron_params
