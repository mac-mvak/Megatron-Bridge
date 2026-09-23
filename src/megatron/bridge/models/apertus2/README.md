# Apertus 2 KDA checkpoint support

The adapter supports explicit per-channel KDA decay scales and an optional bias on
the second output-gate projection. Both options are preserved through HF import,
provider/builder construction, serialized model configuration, and HF export.
Missing flags retain the historical defaults: per-head decay and output-gate bias.
An explicit configuration takes precedence over `KDA_ALOG_PER_CHANNEL`.

The `megachonk_iter_0000008` configuration uses 61 layers: 45 KDA layers, 16 full
attention layers, three initial dense MLPs, and 58 latent MoE layers. Its KDA
`A_log` contains 8192 values per layer and its output-gate projection has no bias.
The adapter also retains QB routing, SSSGLU, shared experts, sandwich norms, QK
norms, attention output gating, NoPE, and embedding/residual scaling.

KDA input projections and convolution weights shard each Q/K/V/gate component
independently. The mappings rearrange these components before TP scatter and
after TP gather. Decay parameters and QB router state retain FP32 precision during
mixed-precision wrapping and export. BF16 source decay values can therefore be
represented exactly in FP32, while their serialized dtype changes.

Native offloaded experts retain their `weight1`/`weight2` checkpoint format.
During a weight-only load, Bridge detects `moe_use_offloading_experts` in the
checkpoint arguments and adapts standard expert requests to those keys and their
input/output matrix orientation. Existing GLU factories and TP/EP offsets are
preserved. Native offloading modules load their own keys directly. This read
adapter does not change checkpoint files or either implementation's save format.

## Native dependency

Use the accompanying Swiss Megatron-LM `apertus2/bridge-support` changes based on
`96b9751981de6a259baa8d5bce471416f4c362a2`. They add explicit KDA constructor
options and the builder/configuration APIs needed by this Bridge revision. The
upstream Bridge-pinned Megatron revision alone lacks the Swiss Apertus 2 modules;
the unpatched Swiss revision lacks the newer builder APIs. See the companion
repository's `BRIDGE_COMPATIBILITY.md` for backport provenance and scope.

## Reproduce validation

Use Python 3.12 with the Bridge dependencies and the native fork. Set
`APERTUS2_HF_REFERENCE` to an export containing the trusted
`configuration_apertus2.py` and `modeling_apertus2.py`. The functional tests read
that implementation and construct small random models; they do not load the
production checkpoint weights or download a tokenizer.

```bash
export PYTHONPATH="$BRIDGE_ROOT/src:$MCORE_ROOT"
export APERTUS2_HF_REFERENCE="$HF_EXPORT"
cd "$BRIDGE_ROOT"

uv run python -m pytest \
  tests/unit_tests/models/apertus2 \
  tests/unit_tests/conversion/launcher/test_generate_apertus2_run_config.py -q

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m pytest --confcutdir=tests/functional_tests/test_groups/models/apertus2 \
  tests/functional_tests/test_groups/models/apertus2/test_apertus2_conversion.py -q -x

uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest --confcutdir=tests/functional_tests/test_groups/models/apertus2 \
  tests/functional_tests/test_groups/models/apertus2/test_apertus2_conversion.py -q -x

uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest --confcutdir=tests/functional_tests/test_groups/models/apertus2 \
  tests/functional_tests/test_groups/models/apertus2/test_offloaded_checkpoint.py -q -x
```

The hybrid test covers both provider and builder construction, native distributed
checkpoint save/reload, exact equality of all 95 logical HF tensors, nonzero FP32
QB state, distinct channel decay scales,
logit cosine similarity of at least 0.99, matching next-token predictions, HF
prefill/cached-decode agreement, and native backward passes. Each pytest output
directory also contains `parity.json` with measured errors.

The offloading test writes real native `weight1`/`weight2` distributed checkpoints
using fused expert BF16 master weights in both fine-grained and coarse-grained
offloading modes. After zeroing parameters, it verifies native reload, standard
Bridge expert reload, and exact HF tensor values and dtypes. It covers EP=2 and
PP=2 separately, writes and reloads HF safetensors with PP=2, and checks logits
and next-token agreement with EP=2. Unit tests also verify TP=1/2/4 shard offsets
and GLU reconstruction for both expert projections.

For the KDA layer matrix, run `kda_parity.py` in the same test directory with
`--hf-source "$HF_EXPORT"` under one- and two-process torchrun. It checks all four
combinations of per-head/per-channel decay and bias present/absent. The native
`tests/unit_tests/ssm/test_kimi_delta_attention_reference.py` additionally checks
bounded and unbounded decay and gradients.

For an existing native checkpoint, generate configuration in a separate directory:

```bash
uv run python scripts/conversion/generate_apertus2_run_config.py \
  --checkpoint "$NATIVE_ITERATION" --output "$CHECKPOINT_VIEW/run_config.yaml"
```

Generation reads metadata from every KDA layer, rejects mixed or invalid layouts,
and validates the YAML round trip. It does not rewrite checkpoint tensors. A
checkpoint view used for conversion must also contain the original checkpoint
files, for example through links. Loading offloaded-expert checkpoints requires
the compatible native fork and the Bridge read adapter described above.

## Validation boundary

The full `megachonk_iter_0000008` native checkpoint was loaded through Bridge on
2026-09-23 using 32 GPUs, TP=1, PP=8 and EP=4. All 593,659,004,544 parameter
elements were materialized on GPUs and passed finite-value checks. The same
loaded model exported successfully to 239 HF safetensors shards containing
46,051 tensors. Loading took 342.4 seconds and export took 273.7 seconds in this
run. Temporary expert merges used Megatron's CPU fallback when GPU allocation
failed; every final model parameter remained on a GPU.

Every exported tensor was then compared against the independent hfconverter
export: identical shapes and values for all 46,051 tensors, with 90 KDA decay
tensors promoted losslessly from BF16 to FP32. Other tensor dtypes match the
reference. The exported model/configuration Python files are byte-identical to
the hfconverter implementation. A separate full native-offloading load also
passed using the original `weight1`/`weight2` checkpoint format.

Small-model GPU tests additionally verify forward/backward behavior. Full-model
inference/training and FSDP parity have not been established. Native KDA cached
inference remains unsupported by the native implementation; HF cached decoding
is tested.
Sliding-window attention and virtual pipeline parallelism retain their explicit
unsupported checks in this adapter.

## Additional native checkpoint round trip

Set `CUDA_DEVICE_MAX_CONNECTIONS=1` before the GPU process starts; this avoids
TE TP backward collective/GEMM ordering hangs in the tested environment.
`model_parity.py` also provides a standalone hybrid test using the source
export's configuration and model code:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/functional_tests/test_groups/models/apertus2/model_parity.py \
  --hf-source "$HF_EXPORT" --checkpoint-dir "$EMPTY_SHARED_DIRECTORY"
```

The optional checkpoint directory must be empty. This test saves the native
distributed checkpoint, zeroes the model parameters, reloads it, checks all 56
logical HF tensors exactly, and repeats logit and next-token checks. It passed
on two GPUs with logit cosine similarity 0.99990928.

When using hfconverter on the resulting native checkpoint, use its matching
FP32 decay support: shard planning must read each source tensor's dtype to
preserve FP32 KDA masters alongside BF16 ordinary weights.
