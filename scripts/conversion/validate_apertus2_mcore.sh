#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inside a Bridge development container with two GPUs. The HF reference
# supplies the model implementation and config; production weights are not loaded.
set -euo pipefail
stage="${1:-all}"
case "$stage" in
    all|unit|tp|offloading) ;;
    *) echo "Usage: $0 [all|unit|tp|offloading]" >&2; exit 2 ;;
esac
: "${APERTUS2_HF_REFERENCE:?Set APERTUS2_HF_REFERENCE to an Apertus2 HF export}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src:$PWD/3rdparty/Megatron-LM:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -n "${BRIDGE_TEST_PYTHON:-}" ]]; then
    uv_run=(uv run --no-project --python "$BRIDGE_TEST_PYTHON")
else
    uv_run=(uv run --no-sync)
fi
if [[ "$stage" == all || "$stage" == unit ]]; then
    "${uv_run[@]}" python -m pytest \
        tests/unit_tests/models/apertus2/ \
        tests/unit_tests/training/test_checkpointing.py::TestLegacyModelShardedStateDictMetadata \
        tests/unit_tests/training/mlm_compat/test_arguments.py \
        tests/unit_tests/conversion/launcher/test_generate_apertus2_run_config.py \
        -q --tb=short -p no:cacheprovider
fi
if [[ "$stage" == all || "$stage" == tp ]]; then
    "${uv_run[@]}" python -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m pytest tests/functional_tests/test_groups/models/apertus2/test_apertus2_conversion.py \
        -q -x --tb=short -o faulthandler_timeout=180 -p no:cacheprovider
fi
if [[ "$stage" == all || "$stage" == offloading ]]; then
    for topology in ep pp; do
        APERTUS2_TEST_PARALLELISM="$topology" \
            "${uv_run[@]}" python -m torch.distributed.run --standalone --nproc_per_node=2 \
            -m pytest tests/functional_tests/test_groups/models/apertus2/test_offloaded_checkpoint.py \
            -q -x --tb=short -o faulthandler_timeout=180 -p no:cacheprovider
    done
fi
