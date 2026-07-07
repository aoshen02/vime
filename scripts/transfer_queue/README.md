# TransferQueue Artifact Training

This directory contains helper scripts for running VIME against rollout artifacts published through TransferQueue. The intended closed-loop path is:

1. Start or reuse the shared Ray and TransferQueue service.
2. Dispatch prompts to a VIME-managed vLLM rollout server with `artifact_transfer_params`.
3. Consume the resulting `vllm.trajectory/v1alpha1` artifacts in VIME with `--trajectory-source transfer_queue`.
4. Keep policy updates on VIME's normal rollout-engine update path by also passing `--tq-manage-rollout-servers`.

TransferQueue is used for rollout artifacts in this flow. Policy weights are not sent through the TransferQueue policy-deployment helpers; closed-loop training should let `actor_model.update_weights()` update the VIME-managed rollout engines as it does in non-artifact-transfer training.

## Service Scripts

- `start_services.sh`: starts Ray if needed, starts the TransferQueue controller, and exports a client config.
- `tq_service.py`: long-running TransferQueue controller process.
- `export_config.py`: writes the resolved TransferQueue client config for jobs that should not attach to the service Ray cluster directly.
- `health_check.py`: non-destructive health check for the Ray and TransferQueue service.

## Rollout Dispatch

- `dispatch_rollouts.py`: sends JSONL prompts to an OpenAI-compatible vLLM completions endpoint and includes artifact metadata so vLLM can publish trajectory artifacts.
- `phase6_prompts.jsonl`: tiny smoke-test prompts for end-to-end validation.

Example dispatch shape:

```bash
python scripts/transfer_queue/dispatch_rollouts.py \
  --endpoint http://<vime-managed-router>/v1/completions \
  --model <served-model-name> \
  --run-id <run-id> \
  --policy-version <policy-version> \
  --prompts scripts/transfer_queue/phase6_prompts.jsonl \
  --n-samples-per-prompt 2 \
  --max-tokens 32 \
  --output /tmp/dispatched_rollouts.jsonl
```

## VIME Training Flags

Use these flags for closed-loop artifact-transfer training:

```text
--trajectory-source transfer_queue
--tq-manage-rollout-servers
--tq-ray-address ray://<transfer-queue-host>:20001
--tq-run-id <run-id>
--tq-policy-version <policy-version>
--num-rollout <explicit-count>
```

`--tq-manage-rollout-servers` is the important distinction from offline artifact consumption. It keeps VIME in charge of rollout server lifecycle and weight updates while only replacing the sample source with TransferQueue artifacts.
