# Docker release rule

vime ships one image based on the official vllm image, published as
`vllm/vime:latest`. Supports GB200/300 and H100/200.

Build and publish from the `docker/` directory:


```bash
just build
```

Run `just build` on both native AMD64 and ARM64 hosts. After both digests are
available, publish the version tag and `latest`:

```bash
just manifest "" sha256:AMD64_DIGEST sha256:ARM64_DIGEST
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-106B-A12B sync
