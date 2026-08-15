# CI（持续集成）

Vime 使用 Buildkite 进行持续集成。提交到仓库中的 pipeline 是
`.buildkite/pipeline.yml`；Vime 不使用上游自动生成的
`.github/workflows/pr-test.yml` 文件。

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
job。GPU 测试默认使用 `vllm/vime:latest`。如果 pull request 修改了镜像
或 vLLM patch，应将 `VIME_CI_IMAGE` 设置为该 commit 构建出的不可变镜像
tag。

## 注册测试

- 始终运行的 CPU 测试加入 `.buildkite/pipeline.yml` 中对应的命令。
- GPU 测试加入 `.buildkite/gpu_suites.py` 中对应的套件，并同步更新
  `.buildkite/pipeline.yml` 显示的测试数量。
- `.buildkite/README.md` 必须与 pipeline 行为保持一致。

触发远程 Buildkite job 前，应先在本地运行完全相同的命令。GPU 测试失败
时，先使用相同镜像和环境在 H200 节点复现并修复；本地通过后再重跑远程
套件。
