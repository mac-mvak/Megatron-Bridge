# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Nemotron Omni model family (Vision-Language + Audio) for Megatron Bridge."""

from megatron.core.models.multimodal import context_parallel as _context_parallel


# TODO: Remove this guard when the Apertus2 MCore fork provides dynamic-resolution CP.
NEMOTRON_OMNI_AVAILABLE = all(
    hasattr(_context_parallel, name)
    for name in ("gather_from_context_parallel_ranks_dynamic_res", "split_to_context_parallel_ranks_dynamic_res")
)

if NEMOTRON_OMNI_AVAILABLE:
    from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
    from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
    from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
        NemotronOmniBridge,
        NemotronOmniLlavaBridge,
    )
    from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
        NemotronOmniLlavaModelProvider,
        NemotronOmniModelProvider,
        NemotronVLModelProvider,
    )

    __all__ = [
        "NemotronOmniModel",
        "NemotronOmniBridge",
        "NemotronOmniModelProvider",
        "NemotronOmniLlavaModel",
        "NemotronOmniLlavaBridge",
        "NemotronOmniLlavaModelProvider",
        "NemotronVLModelProvider",
        "BridgeSoundEncoder",
    ]
else:
    __all__ = []


def __getattr__(name: str):
    if name == "BridgeSoundEncoder" and NEMOTRON_OMNI_AVAILABLE:
        from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder

        return BridgeSoundEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
