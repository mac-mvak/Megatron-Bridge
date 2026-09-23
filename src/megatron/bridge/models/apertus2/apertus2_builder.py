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

from copy import copy, deepcopy
from dataclasses import dataclass
from typing import ClassVar, cast

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.apertus2.apertus2_spec import build_apertus2_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


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
        model = GPTModelBuilder(model_config).build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        from megatron.bridge.models.apertus2.apertus2_provider import _preserve_kda_decay_parameters

        _preserve_kda_decay_parameters([model])
        return model


__all__ = ["Apertus2ModelBuilder", "Apertus2ModelConfig", "Apertus2TransformerConfig"]
