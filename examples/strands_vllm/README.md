# vime x Strands-vLLM

This example connects `vime` with [`strands-vllm`](https://github.com/horizon-rl/strands-vllm) (vLLM extension for the agentic scaffolding [`strands`](https://github.com/strands-agents/sdk-python)) for agentic RL training.

## Why `strands-vllm`?

| Component                                                          | Agent Loop                          | TITO Support                           |
| ------------------------------------------------------------------ | ----------------------------------- | -------------------------------------- |
| [Strands-Agents](https://github.com/strands-agents/sdk-python)     | ✅ Handles agent loop, custom hooks | ❌ text-based, requires retokenization |
| [vLLM](https://github.com/sgl-project/vllm)                    | ❌ Single generation only           | ✅ Native `input_ids` in/out           |
| **[strands-vllm](https://github.com/horizon-rl/strands-vllm)** | ✅ Via Strands                      | ✅ Via vLLM's native API             |

`strands-vllm` bridges the gap by extending `strands` with vLLM's native `/generate` endpoint:

- Captures exact token IDs during generation (no retokenization drift)
- Automatically tracks `loss_mask` via the `Rollout` tracker (`model.rollout`)
- Provides `ToolLimiter` for clean trajectory truncation

## Install Dependencies

1. Pull the `vimerl/vime:latest` image and enter it
2. Go to vime folder: `cd /root/vime`
3. Install vime: `pip install -e . --no-deps`
4. Go to the example folder: `cd /root/vime/examples/strands_vllm`
5. Install `strands-vllm`: `pip install strands-vllm==0.4.2`

> NOTE: The `execute_python_code` tool runs code via `subprocess_interpreter.py`, a self-contained interpreter vendored from camel-ai so this example does not depend on the full `camel-ai` package. It runs model-generated code in a local subprocess with **no isolation**, which is NOT a good practice; it is here only for the convenience of this example. Use a sandboxed interpreter (Docker, e2b, microsandbox, ...) for anything beyond local experimentation.

## Prepare Model

```bash
# hf checkpoint
hf download Qwen/Qwen3-8B --local-dir /root/models/Qwen/Qwen3-8B

# mcore checkpoint
cd /root/vime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen/Qwen3-8B \
    --save /root/models/Qwen/Qwen3-8B_torch_dist
```

## Prepare Dataset

Following [Retool](https://arxiv.org/abs/2504.11536), we use `dapo-math-17k` as training data:

```python
from datasets import load_dataset
ds = load_dataset("zhuzilin/dapo-math-17k", split="train")
ds.to_json("/root/data/dapo-math-17k.jsonl", orient="records", lines=True)
```

and `aime-2024` as eval data:

```python
from datasets import load_dataset
ds = load_dataset("zhuzilin/aime-2024", split="train")
ds.to_json("/root/data/aime-2024.jsonl", orient="records", lines=True)
```

## Run Training

```bash
cd /root/vime
export WANDB_KEY=$your_wandb_key
bash examples/strands_vllm/strands_qwen3_8b.sh
```
