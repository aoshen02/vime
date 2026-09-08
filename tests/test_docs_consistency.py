"""Guard documentation references that can be checked without a GPU runtime."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

NUM_GPUS = 0

ROOT = Path(__file__).resolve().parents[1]
DOC_SOURCES = (
    ROOT / "README.md",
    ROOT / "README_zh.md",
    ROOT / "docs" / "en",
    ROOT / "docs" / "zh",
    ROOT / "examples",
    ROOT / "docker",
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
COMMAND_PATH_RE = re.compile(r"\b(?:bash|python)\s+((?:scripts?|tests)/[A-Za-z0-9_.+/-]+\.(?:sh|py))")
LOCAL_ANCHOR_RE = re.compile(r"\]\(#([A-Za-z0-9_-]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _markdown_files():
    for source in DOC_SOURCES:
        if source.is_file():
            yield source
        else:
            for path in sorted(source.rglob("*.md")):
                if "_examples_synced" not in path.parts:
                    yield path


def _local_link_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]

    if target.startswith(("#", "/")) or "://" in target or target.startswith(("mailto:", "data:")):
        return None

    return unquote(target.split("#", 1)[0].split("?", 1)[0]) or None


def test_local_markdown_links_exist():
    missing = []
    for markdown_file in _markdown_files():
        for raw_target in MARKDOWN_LINK_RE.findall(markdown_file.read_text(encoding="utf-8")):
            target = _local_link_target(raw_target)
            if (
                target is not None
                and "_examples_synced" not in Path(target).parts
                and not (markdown_file.parent / target).exists()
            ):
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {target}")

    assert not missing, "Broken local Markdown links:\n" + "\n".join(missing)


def test_documented_script_and_test_commands_exist():
    missing = []
    for markdown_file in _markdown_files():
        text = markdown_file.read_text(encoding="utf-8")
        for command_path in COMMAND_PATH_RE.findall(text):
            if not (ROOT / command_path).is_file():
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {command_path}")

    assert not missing, "Documented command paths do not exist:\n" + "\n".join(missing)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_customization_anchor_links_exist(language):
    text = (ROOT / "docs" / language / "get_started" / "customization.md").read_text(encoding="utf-8")
    headings = set()
    for heading in HEADING_RE.findall(text):
        heading = re.sub(r"[^\w\s-]", "", heading.replace("`", "").lower())
        headings.add(re.sub(r"\s+", "-", heading).strip("-"))

    missing = sorted(set(LOCAL_ANCHOR_RE.findall(text)) - headings)
    assert not missing, f"Local anchors without matching headings: {missing}"


@pytest.mark.parametrize("language", ["en", "zh"])
def test_agent_guide_is_in_get_started_toctree(language):
    text = (ROOT / "docs" / language / "index.rst").read_text(encoding="utf-8")
    assert re.search(r"^   get_started/agent\.md$", text, re.MULTILINE)


def test_ci_skill_references_current_buildkite_sources():
    text = (ROOT / ".claude/skills/add-tests-and-ci/SKILL.md").read_text(encoding="utf-8")
    for source in (".buildkite/pipeline.yml", ".buildkite/gpu_suites.py"):
        assert source in text
        assert (ROOT / source).is_file()
    assert ".github/workflows/pr-test" not in text
    assert "generate_github_workflows.py" not in text


@pytest.mark.parametrize("language, filename", [("en", "README.md"), ("zh", "README_zh.md")])
def test_readme_links_deployment_and_correctness_guides(language, filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    for guide in (
        "advanced/vllm-config.md",
        "advanced/pd-disaggregation.md",
        "advanced/delta-weight-sync.md",
        "advanced/external-rollout-engines.md",
        "advanced/reproducibility.md",
        "advanced/fault-tolerance.md",
        "developer_guide/ci.md",
        "developer_guide/debug.md",
        "developer_guide/trace.md",
        "developer_guide/profiling.md",
    ):
        assert f"docs/{language}/{guide}" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
