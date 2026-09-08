#!/usr/bin/env python3
"""Check Vime release metadata and its Docker patch stack."""

import argparse
import ast
import re
import sys
from pathlib import Path


def setup_version(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "version":
                return ast.literal_eval(keyword.value)
    raise ValueError(f"setup version not found in {path}")


def assigned_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    repo = args.repo.resolve()
    errors: list[str] = []
    package_version = setup_version(repo / "setup.py")
    docs_version = assigned_string(repo / "docs/conf.py", "__version__")
    if package_version != docs_version:
        errors.append(f"setup.py={package_version} but docs/conf.py={docs_version}")
    if args.expected_version and package_version != args.expected_version:
        errors.append(f"release version is {package_version}, expected {args.expected_version}")

    dockerfile = (repo / "docker/Dockerfile").read_text()
    image_tag = (repo / "docker/version.txt").read_text().strip()
    if not image_tag:
        errors.append("docker/version.txt is empty")
    if not re.search(r"^ARG BASE_IMAGE=", dockerfile, re.MULTILINE):
        errors.append("docker/Dockerfile does not pin BASE_IMAGE")
    if not re.search(r"^ARG PATCH_VERSION=latest$", dockerfile, re.MULTILINE):
        errors.append("docker/Dockerfile must build from docker/patch/latest")

    patch_dir = repo / "docker/patch/latest"
    patches = {path.name for path in patch_dir.glob("*.patch")}
    copied = {
        name
        for name in re.findall(r"COPY docker/patch/\$\{PATCH_VERSION\}/([^\s]+\.patch)", dockerfile)
        if "*" not in name
    }
    if "megatron*.patch" in dockerfile:
        copied.add("megatron.patch")
    if patches != copied:
        errors.append(
            "Dockerfile patch set differs from docker/patch/latest: "
            f"only_patches={sorted(patches - copied)}, "
            f"only_dockerfile={sorted(copied - patches)}"
        )
    applied = set(
        re.findall(
            r"git apply(?:\s+--?[\w-]+)*\s+(?:/tmp/)?([^ \\]+\.patch)",
            dockerfile,
        )
    )
    if patches != applied:
        errors.append(
            "Dockerfile does not apply every patch: "
            f"not_applied={sorted(patches - applied)}, "
            f"unknown={sorted(applied - patches)}"
        )
    for patch in sorted(patches):
        if not (patch_dir / patch).read_text().startswith("diff --git "):
            errors.append(f"invalid git patch: {patch}")

    justfile = (repo / "docker/justfile").read_text()
    if 'VERSION="$(cat docker/version.txt | tr -d' not in justfile:
        errors.append("docker/justfile does not source docker/version.txt")

    if errors:
        print(*[f"ERROR: {error}" for error in errors], sep="\n", file=sys.stderr)
        return 1

    print(f"release={package_version}, image={image_tag}, " f"patches={','.join(sorted(patches))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
