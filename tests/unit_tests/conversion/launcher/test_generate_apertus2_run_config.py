# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for native Apertus2 run-config generation."""

from types import SimpleNamespace

from scripts.conversion.generate_apertus2_run_config import _LegacyConfigView


def test_legacy_precision_flags_override_transformer_defaults():
    transformer_config = SimpleNamespace(bf16=False, fp16=False, fp8=None, fp8_param=False)
    legacy_args = SimpleNamespace(bf16=True, fp16=False, fp8="hybrid", fp8_param=True)

    view = _LegacyConfigView(transformer_config, legacy_args)

    assert view.bf16 is True
    assert view.fp16 is False
    assert view.fp8 == "hybrid"
    assert view.fp8_param is True
