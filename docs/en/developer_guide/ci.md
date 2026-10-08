# CI (Continuous Integration)

Vime uses Buildkite for continuous integration. The committed pipeline is
`.buildkite/pipeline.yml`.

## Always-on checks

Every pull request runs these CPU steps:

| Step | Coverage |
|---|---|
| `pre-commit` | formatting, lint, and repository policy |
| `plugin-contracts` | customization contracts and CPU tests |
| `agent-adapter` | agent adapter behavior |
| `upstream-sync-cpu` | CPU tests synchronized from upstream |
| `utils` | `tests/utils` |

The authoritative commands and queue configuration are in
`.buildkite/pipeline.yml`.

## GPU suites

After the CPU steps pass, the Buildkite build exposes a block step named
`Run GPU test suites?`. Select one or more suites:

- `short`
- `vllm-config`
- `megatron`
- `vime-customized`
- `precision`
- `ckpt`

`.buildkite/gpu_suites.py` expands each selected suite into one Buildkite job
per test. Set `VIME_CI_IMAGE` to an immutable candidate digest when validating
Dockerfile or vLLM patch changes. Jobs otherwise use `vllm/vime:latest`, which
must not be updated before the change merges.

The `megatron` suite includes `test_straw_checkpoint_fork.py` for checkpoint
step selection, rollback branches and indexed debug archives.

`test_qwen2.5_0.5B_pipeline_rl.py` runs three GRPO steps with fully async rollout
on 4 GPUs. Its probes check that requests continue across weight updates and
that policy weights change. The matrix covers NCCL and full disk sync with
`--flush-cache-interval 0`, plus periodic refresh with NCCL and interval `2`.
`test_pipeline_rl.py` checks the schedule on CPU. These tests do not measure
learning quality or throughput gains.

### Manual Megatron Restart

`test_qwen2.5_0.5B_training_recovery.py` uses 4 GPUs and two successive training jobs on the same Ray cluster. The first uses TP=1 and deliberately triggers a real CUDA OOM. After verifying that serving still responds when the job exits, it resubmits training with TP=2, which also changes the DP size.

The test checks that healthy vLLM processes, routers, and GPU placements are reused; replayed batch contents match; and training scheduler progress, finite nonzero gradients, and the final checkpoint are correct. The fixed test list includes:

| Data storage | RolloutManager state | Checks |
|---|---|---|
| Straw with online GC | Remains alive | Reconnect trainers and replay completed training batches that were not checkpointed. |
| Straw with online GC | Killed after failure | Reconnect a new manager to the original serving cluster and replay the same batches. |
| Straw with a model/optimizer checkpoint and Megatron YAML configuration | Killed during training | Restore from the checkpoint and check configuration and recovery state. |
| Rollout debug files | Killed after failure | Restore data from debug files and reconnect a new manager to the original serving cluster. |
| Straw with disk-delta weight synchronization | Killed after failure | Publish restored weights as a new full baseline, then continue delta updates. |
| Straw with PD/NIXL serving | Killed after failure | Wedge the prefill actor, replace it within the reset timeout, and retain the healthy decode actor. |

`test_qwen3_30B_A3B_training_recovery.py` uses 8 GPUs for the same OOM/checkpoint/manager-loss workflow with a MoE model, R3, and stateless Adam. It omits optimizer tensors while checking scheduler progress, compares persisted routing bytes across the TP/DP change, and completes training after recovery. The dense cases cover ordinary Adam with optimizer checkpoints.

Internal serving health checks are enabled with or without the compatibility flag `--use-fault-tolerance`. CPU coverage includes configuration and checkpoint boundaries in `test_training_recovery.py`, lost disk-update replies in `test_disk_delta_recovery.py`, and real Ray manager SIGKILL or conversion-reply loss in `test_rollout_manager_recovery.py`.

### Removing Failed Engines at Rollout Completion

`test_qwen2.5_0.5B_rollout_health.py` stops a real vLLM HTTP server just before rollout completes, leaving its router registration intact. Two four-GPU cases either retain or kill the corresponding Ray actor. They check bounded rollout completion, deregistration before training, engine recovery at weight update, and a final training checkpoint.

Both cases omit `--use-fault-tolerance` and set the background interval and initial wait to 600 seconds. This verifies that rollout-completion checks run immediately without waiting for background checks.

## Registering tests

- Add always-on CPU tests to the appropriate command in
  `.buildkite/pipeline.yml`.
- Add GPU tests to a suite in `.buildkite/gpu_suites.py` and update the suite
  count shown by `.buildkite/pipeline.yml`.
- Keep `.buildkite/README.md` synchronized with pipeline behavior.

Run the exact command locally before triggering its remote Buildkite job. For
GPU failures, reproduce on an H200 node with the same image and environment,
then rerun the remote suite only after the local test passes.
