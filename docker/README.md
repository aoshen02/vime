# Docker release rule

vime ships one image based on the official vLLM 0.27.1 image, published as
`vllm/vime:latest`. Supports GB200/300 and H100/200.

Build locally:

```bash
cd docker
VIME_COMMIT=<commit-or-branch> just build
```

The command builds on the native host and pushes an architecture-specific
digest. Run it once on amd64 and once on arm64, then combine the two digests
with `just manifest` as documented in `docker/justfile`.

Before each update, validate the following model matrix on the available GPU
cluster. Record the actual GPU shape and logs in the release PR rather than
assuming a fixed 64xH100 environment:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-106B-A12B sync
