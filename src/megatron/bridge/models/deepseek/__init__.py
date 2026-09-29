# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from importlib.util import find_spec


# TODO: Remove this guard when the Apertus2 MCore fork provides the MLA QK-norm resolver.
# Other model families import deepseek.common without requiring DeepSeek model classes.
DEEPSEEK_AVAILABLE = find_spec("megatron.core.transformer.mla_qk_norm_config") is not None

if DEEPSEEK_AVAILABLE:
    from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge

    __all__ = ["DeepSeekV2Bridge", "DeepSeekV3Bridge", "DeepSeekV4Bridge"]
else:
    __all__ = []
