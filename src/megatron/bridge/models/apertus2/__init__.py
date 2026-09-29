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

"""Native Apertus2 model provider, specification, builder, and mappings."""

from megatron.bridge.models.apertus2.apertus2_bridge import Apertus2Bridge
from megatron.bridge.models.apertus2.apertus2_builder import (
    Apertus2ModelBuilder,
    Apertus2ModelConfig,
    Apertus2TransformerConfig,
)
from megatron.bridge.models.apertus2.apertus2_mapping import (
    build_apertus2_mapping_registry,
)
from megatron.bridge.models.apertus2.apertus2_provider import Apertus2ModelProvider
from megatron.bridge.models.apertus2.apertus2_spec import build_apertus2_spec


__all__ = [
    "Apertus2Bridge",
    "Apertus2ModelBuilder",
    "Apertus2ModelConfig",
    "Apertus2TransformerConfig",
    "Apertus2ModelProvider",
    "build_apertus2_spec",
    "build_apertus2_mapping_registry",
]
