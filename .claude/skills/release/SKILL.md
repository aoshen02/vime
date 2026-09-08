---
name: release
description: Prepare and verify a Vime release, including version metadata, Docker patch-stack validation, and release-specific checks. Use when cutting or auditing a Vime release.
---

# Release Vime

Prepare a release without creating Git tags, GitHub releases, or publishing
images unless the user explicitly requests those external actions.

## Establish the release baseline

- Preserve unrelated changes and compare the previous Vime release tag.
- Confirm the package version, the pinned `BASE_IMAGE`, and the Docker patch
  stack in `docker/patch/latest/`.
- Do not upgrade the vLLM base image as part of a release unless Slime has
  upgraded its corresponding inference-image baseline.

## Prepare the release PR

- Update `setup.py` and `docs/conf.py` to the requested package version.
- Give `docker/version.txt` a new unique dated image tag.
- Review every remaining occurrence of the old Vime version rather than making
  a blind repository-wide replacement.
- Verify every patch under `docker/patch/latest/` is consumed in Dockerfile
  application order and applies to its target in separate clean checkouts of
  the pinned vLLM and Megatron revisions. Do not validate patch application
  against a dirty developer checkout.

## Validate and publish

- Run `python .claude/skills/release/scripts/check_release.py --repo .
  --expected-version <version>`, `python setup.py --version`, and
  `git diff --check`.
- Build a candidate image from the release commit and run the required E2E
  tests before promoting an image tag.
- Merge the green release PR, then create the matching Git tag and GitHub
  release at its merge commit.
- Publish the versioned image first. Only update `vllm/vime:latest` after the
  candidate has passed and every required vLLM patch has merged upstream.
