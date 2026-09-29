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

"""Builder-backed configuration for native Apertus2 models."""

import os
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, cast

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.apertus2.apertus2_provider import (
    _preserve_kda_decay_parameters,
    _validate_kda_a_log_layout,
)
from megatron.bridge.models.apertus2.apertus2_spec import build_apertus2_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.model_provider import _apply_mixed_precision_wrapper
from megatron.bridge.models.transformer_config import TransformerConfig


def _wrap_preserving_fp32(
    config: TransformerConfig,
    model: MegatronModule,
    *,
    wrapper: Callable[[TransformerConfig, MegatronModule], MegatronModule],
) -> MegatronModule:
    """Use Bridge's precision-preserving wrapper on MCore's builder path too."""
    return _apply_mixed_precision_wrapper([model], config, wrapper)[0]


@dataclass(kw_only=True)
class Apertus2TransformerConfig(TransformerConfig):
    """Apertus2-only schedule fields kept on the transformer config."""

    layer_types: tuple[str, ...] | None = None
    linear_attn_output_gate_bias: bool = True
    linear_attn_a_log_per_channel: bool = False


@dataclass(kw_only=True)
class Apertus2ModelConfig(BridgeGPTModelConfig):
    """Serializable GPT config that resolves to :class:`Apertus2ModelBuilder`."""

    builder: ClassVar[str] = "megatron.bridge.models.apertus2.Apertus2ModelBuilder"
    transformer_config_class: ClassVar[type[Apertus2TransformerConfig]] = Apertus2TransformerConfig


class Apertus2ModelBuilder(GPTModelBuilder):
    """Build Apertus2 using native MCore KDA and standard transformer layers."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build without mutating the caller's model or transformer config."""
        original = cast(Apertus2ModelConfig, self._model_config)
        model_config = copy(original)
        model_config.transformer = deepcopy(original.transformer)
        # GPTModelBuilder reads this selector from the outer GPT model config, not
        # from the nested TransformerConfig. Keep the caller's config untouched.
        model_config.transformer_layer_spec = build_apertus2_spec

        # TODO: Remove this env shim once Megatron-LM-MoE exposes a saved KDA layout CLI argument.
        previous = os.environ.get("KDA_ALOG_PER_CHANNEL")
        per_channel = model_config.transformer.linear_attn_a_log_per_channel
        os.environ["KDA_ALOG_PER_CHANNEL"] = "1" if per_channel else "0"
        try:
            model = GPTModelBuilder(model_config).build_model(
                pg_collection,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
        finally:
            if previous is None:
                os.environ.pop("KDA_ALOG_PER_CHANNEL", None)
            else:
                os.environ["KDA_ALOG_PER_CHANNEL"] = previous
        _validate_kda_a_log_layout(model, per_channel)
        _preserve_kda_decay_parameters([model])
        return model

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        mixed_precision_wrapper: Callable[[TransformerConfig, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[GPTModel]:
        """Build distributed models while preserving KDA and router state in FP32."""
        wrapper = (
            partial(_wrap_preserving_fp32, wrapper=mixed_precision_wrapper)
            if mixed_precision_wrapper is not None
            else None
        )
        return super().build_distributed_models(
            pg_collection,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
            mixed_precision_wrapper=wrapper,
            model_type=model_type,
        )


__all__ = ["Apertus2ModelBuilder", "Apertus2ModelConfig", "Apertus2TransformerConfig"]
