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

"""Generate a Bridge run config for a native Apertus2 Megatron-LM checkpoint."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from megatron.core.activations import sssglu_act
from transformers import PretrainedConfig

from megatron.bridge.models.apertus2.apertus2_bridge import Apertus2Bridge
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.model_load_save import load_model_config


class _LegacyConfigView:
    """Expose fields from both TransformerConfig and legacy Megatron arguments."""

    def __init__(self, config: Any, args: argparse.Namespace):
        self._config = config
        self._args = args

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._config, name, None)
        if value is not None:
            return value
        if name == "vocab_size":
            return self._args.padded_vocab_size
        if name == "share_embeddings_and_output_weights":
            return not self._args.untie_embeddings_and_output_weights
        if hasattr(self._args, name):
            return getattr(self._args, name)
        if hasattr(self._config, name):
            return value
        raise AttributeError(name)


def _validate_generated_config(candidate_dir: Path, expected: Any) -> None:
    loaded, legacy_args = load_model_config(str(candidate_dir))
    if legacy_args is not None:
        raise RuntimeError("Generated run_config.yaml was not recognized as a Bridge checkpoint config")

    fields = (
        "num_layers",
        "hidden_size",
        "vocab_size",
        "seq_length",
        "num_moe_experts",
        "moe_latent_size",
        "linear_attention_freq",
        "moe_layer_freq",
        "no_rope_freq",
    )
    mismatches = {
        name: (getattr(expected, name, None), getattr(loaded, name, None))
        for name in fields
        if getattr(expected, name, None) != getattr(loaded, name, None)
    }
    if mismatches:
        raise RuntimeError(f"Generated run config failed round-trip validation: {mismatches}")
    if loaded.activation_func is not sssglu_act:
        raise RuntimeError(f"Generated run config restored the wrong activation: {loaded.activation_func!r}")


def generate_run_config(checkpoint: Path, output: Path, *, overwrite: bool = False) -> None:
    """Generate and validate ``run_config.yaml`` without changing checkpoint tensors."""
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "common.pt").is_file():
        raise FileNotFoundError(f"Native Megatron checkpoint has no common.pt: {checkpoint}")
    if output.name != "run_config.yaml":
        raise ValueError(f"Output filename must be run_config.yaml, got: {output.name}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite to replace it")

    transformer_config, args = load_model_config(str(checkpoint))
    if args is None:
        raise ValueError(f"Checkpoint already uses a Bridge run config: {checkpoint}")
    if not getattr(args, "sssglu", False):
        raise ValueError("Apertus2 checkpoint does not declare sssglu=True")
    transformer_config.activation_func = sssglu_act

    hf_dict = Apertus2Bridge.megatron_to_hf_config(_LegacyConfigView(transformer_config, args))
    hf_config = PretrainedConfig(**hf_dict)
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    provider.perform_initialization = False

    config = ConfigContainer(
        model=provider,
        train=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        tokenizer=None,
        checkpoint=None,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".run-config-", dir=output.parent) as temp_dir:
        candidate = Path(temp_dir) / "run_config.yaml"
        config.to_yaml(str(candidate))
        _validate_generated_config(Path(temp_dir), provider)
        os.replace(candidate, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native Megatron iteration directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination run_config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Generate a validated Apertus2 Bridge run config from CLI arguments."""
    args = _parse_args()
    generate_run_config(args.checkpoint, args.output, overwrite=args.overwrite)
    print(f"Wrote and validated {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
