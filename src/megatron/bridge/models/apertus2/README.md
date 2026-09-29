# Apertus2 conversion with native MCore expert layouts

This integration builds on `swiss-ai/Megatron-Bridge:apertus2/main` at
`b8543ca6552b567f789f32ca50e6b5ea969f5f2a`. It retains the KDA conversion and
router-state fixes from `mac-mvak/Megatron-Bridge:apertus2/kda_channels` while
using MCore for offloaded-expert checkpoint conversion.

The MCore submodule is pinned to
`andresnowak/Megatron-LM-MoE:feat/nemo-rl-integration-v2` at
`51bf6fa83547388d3fc51883a680c394672db4ac`. Initialize it before setting up the
Bridge environment:

```bash
git submodule update --init 3rdparty/Megatron-LM
```

## Expert checkpoint loading

Bridge's model-weight loader recovers `moe_expert_checkpoint_schema` from saved
metadata. For older checkpoints with `args.moe_use_offloading_experts=True`
and no explicit schema, it requests `legacy_offloading`. MCore's
`sharded_state_dict` factories then load the `weight1`/`weight2` tensors and
transpose them into standard expert parameters. Bridge contains no additional
offloading transpose adapter.

New checkpoints use canonical `experts.linear_fc1.weight` and
`experts.linear_fc2.weight` keys. Their explicit `sequential` schema takes
precedence over the old offloading flag, preventing a second transpose.

This covers model-weight conversion. It does not establish optimizer-state
resume compatibility between different expert implementations. In-place FP8
offloading requires `moe_use_extra_fp8_param_storage=True` when saving, as
required by MCore, to preserve the BF16 parameters.
SSSGLU expert offloading requires this in-place FP8 path on the pinned MCore
branch; ordinary BF16 offloading is tested with SiLU instead.

## KDA and router state

- Fused KDA input and convolution projections retain the correct rank-local
  ordering at TP > 1.
- Both per-head and per-channel `A_log` layouts are supported. Native checkpoint
  loading infers this layout and gate-bias presence from tensor metadata before
  reconstructing the provider, including the path without `run_config.yaml`.
- KDA output-gate bias is independent of bias on other projections. The pinned
  MCore constructor uses its legacy environment flag for `A_log`; Bridge scopes
  and restores that flag during construction.
- `A_log`, `dt_bias`, router `expert_bias`, and `qb_beta` retain FP32 values
  through mixed-precision wrapping and HF export.
- Non-quantile routing maps HF `e_score_correction_bias` to native
  `router.expert_bias`, including nonzero values. Quantile-balanced exports
  preserve `qb_beta` and the HF schema's zero correction buffer. A nonzero
  correction buffer is rejected when native expert bias is disabled.

The `megachonk_iter_0000008` HF export uses per-channel KDA, bias-free KDA output
gates, and quantile balancing. Its router schema contains both `qb_beta` and
`e_score_correction_bias`.

## Conversion and validation workflow

Use the existing [conversion launcher](../../../../../scripts/conversion/README.md)
for import/export. For example, from the repository root in the configured
container:

```bash
# Set these scheduler and topology variables for the full model first.
launch=(--executor slurm --device gpu --account "$ACCOUNT" --partition "$PARTITION"
  --container-image "$BRIDGE_IMAGE" --nodes "$NODES" --gpus-per-node "$GPUS_PER_NODE"
  --mount /path/to/shared/workspace --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP")
bash scripts/conversion/convert.sh import "${launch[@]}" \
  --hf-model /path/to/hf-export/megachonk_iter_0000008 \
  --megatron-path /path/to/output/megatron --trust-remote-code

bash scripts/conversion/convert.sh export "${launch[@]}" \
  --hf-model /path/to/hf-export/megachonk_iter_0000008 \
  --megatron-path /path/to/native/iter_0000008 \
  --hf-path /path/to/output/hf --trust-remote-code
```

These commands illustrate the interfaces; size the production topology and
memory for the full model. The two-GPU test workflow below instead creates tiny
models from the HF reference implementation and config without loading its
production weights:

```bash
APERTUS2_HF_REFERENCE=/path/to/hf-export/megachonk_iter_0000008 \
  bash scripts/conversion/validate_apertus2_mcore.sh
```

The workflow checks TP=2 HF/native parity and exact checkpoint/export equality
for legacy and canonical offloaded checkpoints with EP=2 or PP=2. It covers
ordinary BF16 and in-place FP8 parameter storage, quantile balancing, and
nonzero expert correction bias. It does not run the offloaded FP8 forward
kernel. Set `BRIDGE_TEST_PYTHON` to use an existing development environment.
Pass `unit`, `tp`, or `offloading` to the validation script to rerun one stage.
