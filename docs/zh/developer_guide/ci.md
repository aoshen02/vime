# CI（持续集成）

Vime 使用 Buildkite 进行持续集成。提交到仓库中的 pipeline 是
`.buildkite/pipeline.yml`。

## 始终运行的检查

每个 pull request 都会运行以下 CPU step：

| Step | 覆盖范围 |
|---|---|
| `pre-commit` | 格式化、lint 与仓库规则 |
| `plugin-contracts` | customization contract 与 CPU 测试 |
| `agent-adapter` | agent adapter 行为 |
| `upstream-sync-cpu` | 从上游同步的 CPU 测试 |
| `utils` | `tests/utils` |

权威命令与队列配置位于 `.buildkite/pipeline.yml`。

## GPU 套件

CPU step 通过后，Buildkite build 会显示名为 `Run GPU test suites?` 的
block step。可以选择一个或多个套件：

- `short`
- `vllm-config`
- `megatron`
- `vime-customized`
- `precision`
- `ckpt`

`.buildkite/gpu_suites.py` 会把所选套件展开为每个测试一个 Buildkite
job。验证 Dockerfile 或 vLLM patch 修改时，通过 `VIME_CI_IMAGE` 指定不可变的
候选镜像 digest；未设置时使用 `vllm/vime:latest`，且 PR 合入前不得更新该标签。

`megatron` 套件包含 `test_straw_checkpoint_fork.py`，覆盖 checkpoint
步骤选择、回退分支和带索引的 debug 归档。

`test_qwen2.5_0.5B_pipeline_rl.py` 使用 4 张 GPU，运行 fully async rollout
和三个真实 GRPO step。探针检查同一请求跨权重更新持续生成，并确认训练改变了
策略权重。固定矩阵覆盖 `--flush-cache-interval 0` 下的 NCCL 和 full disk
权重同步，以及 NCCL 下 interval `2` 的周期性刷新。`test_pipeline_rl.py`
在 CPU 上检查刷新周期。这些测试不衡量学习效果或吞吐增益。

### Megatron 手动重启

`test_qwen2.5_0.5B_training_recovery.py` 使用 4 张 GPU，在同一个 Ray 集群中先后运行两次训练任务。第一次使用 TP=1，故意触发真实的 CUDA OOM；确认训练任务退出后推理服务仍能响应，再改成 TP=2 重新提交，此时 DP 大小也会改变。

测试检查是否复用了健康的 vLLM 进程、路由器和 GPU 资源，重放的批次内容是否一致，以及训练调度器进度、非零且有限的梯度和最终 checkpoint。固定测试列表包括：

| 数据保存方式 | RolloutManager 状态 | 检查内容 |
|---|---|---|
| straw，开启在线 GC | 保持存活 | 重新连接训练进程，重放已经训练但尚未保存到 checkpoint 的批次。 |
| straw，开启在线 GC | 失败后被杀掉 | 新 manager 接回原推理集群，并重放同样的批次。 |
| straw，已有模型和优化器 checkpoint，使用 Megatron YAML 配置 | 训练过程中被杀掉 | 从 checkpoint 恢复，核对配置与恢复状态。 |
| Rollout 调试文件 | 失败后被杀掉 | 从调试文件恢复数据，新 manager 接回原推理集群。 |
| straw，使用 disk-delta 同步权重 | 失败后被杀掉 | 以恢复后的权重发布新的完整基准，再继续 delta 更新。 |
| straw，使用 PD/NIXL 推理 | 失败后被杀掉 | 卡住 prefill actor，在连接重置超时后替换它，并保留健康的 decode actor。 |

`test_qwen3_30B_A3B_training_recovery.py` 使用 8 张 GPU，在 MoE 模型、R3 和 stateless Adam 配置下覆盖 OOM、checkpoint 恢复及 manager 丢失。它不保存优化器张量，但会检查 scheduler 进度，比对 TP/DP 改变前后的持久化路由字节，并在恢复后完成训练。Dense 测试覆盖普通 Adam 的优化器 checkpoint 恢复。

无论是否传入兼容参数 `--use-fault-tolerance`，内部推理健康检查都会启用。CPU 测试还包括：`test_training_recovery.py` 的配置与 checkpoint 边界检查，`test_disk_delta_recovery.py` 的权重更新应答丢失，以及 `test_rollout_manager_recovery.py` 中真实 Ray manager 的 SIGKILL 和转换应答丢失。

### Rollout 收尾时清理失效引擎

`test_qwen2.5_0.5B_rollout_health.py` 在 rollout 收尾前停掉真实的 vLLM HTTP 服务，但保留它在路由器中的注册信息。两个四卡场景分别保留或杀掉对应的 Ray actor，检查收尾是否有超时限制、返回训练前是否注销失效服务、更新权重时能否恢复引擎，以及训练能否保存最终 checkpoint。

两种情况都不传 `--use-fault-tolerance`，并将后台检查间隔和首次等待时间设为 600 秒，以验证收尾检查会立即执行，不必等待后台定时检查。

## 注册测试

- 始终运行的 CPU 测试加入 `.buildkite/pipeline.yml` 中对应的命令。
- GPU 测试加入 `.buildkite/gpu_suites.py` 中对应的套件，并同步更新
  `.buildkite/pipeline.yml` 显示的测试数量。
- `.buildkite/README.md` 必须与 pipeline 行为保持一致。

触发远程 Buildkite job 前，应先在本地运行完全相同的命令。GPU 测试失败
时，先使用相同镜像和环境在 H200 节点复现并修复；本地通过后再重跑远程
套件。
