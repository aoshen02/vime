# 可复现性

可复现性是科研进展的基础。vime 通过结合 vLLM 的 [batch-invariant 确定性推理](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) 与 Megatron-LM 的 deterministic 模式，支持 bitwise 级的实验复现。

为了开启确定性训练，你需要通过 `pip uninstall flash_attn_3 -y` 卸载 flash attention 3，并设置：

```bash
  # vLLM config
  --vllm-enable-deterministic-inference
  --vllm-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

以及设置如下环境变量：

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

我们提供了一个完全确定性的，用 Qwen2.5 0.5B 训练 GSM8K 的脚本。

可以用如下脚本初始化训练数据和 ckpt：

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd vime/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

可以使用如下脚本进行训练：

```bash
bash scripts/run-qwen2.5-0.5B-reproducibility.sh
```

这个 PR 中记录了 wandb 的截图 [pull#370](https://github.com/THUDM/slime/pull/370)。

## Train/rollout log-prob alignment（GLM-5）

Vime 当前固定版本的 vLLM **尚不支持这条对齐路径**。Megatron 侧对齐 hook
已存在，但 rollout 侧 sparse MLA、DeepGEMM 和 DeepEP 的数值契约尚未完整验证。

`tests/test_glm52_6layer_deterministic_e2e.py` 和
`tests/test_glm52_layerwise_zero_e2e.py` 当前会明确报“不支持”，不是已通过的
回归 gate。Slime 报告的 log-prob 误差小于1e-6、逐层输出严格相等，是上游
参考目标，不能作为 Vime 的实验结果。

缺失的引擎行为及对应 vLLM PR 记录在
[feature-gap ledger](https://github.com/Inferact/vime-sync-skills/blob/main/knowledge/sglang-vllm-feature-gap-ledger.md)。
