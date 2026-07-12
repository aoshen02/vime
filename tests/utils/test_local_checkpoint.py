import json

import numpy as np
import pytest
import safetensors.numpy
import zstandard

from vime.backends.vllm_utils.local_checkpoint import pull
from vime.utils.disk_delta import checksum, overwrite_encode


def _write_delta(source_dir, old, new, *, encoding, digest=None):
    version_dir = source_dir / "weight_v000001"
    version_dir.mkdir()
    if encoding == "xor":
        delta = new ^ old
    else:
        delta = overwrite_encode(new, new != old)
    compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(delta), dtype=np.uint8)
    safetensors.numpy.save_file(
        {"weight": compressed},
        version_dir / "model-00000-of-00001.safetensors",
        metadata={"weight": digest or checksum("xxh3-128", new)},
    )
    index = {
        "metadata": {
            "version": "000001",
            "base_version": "000000",
            "delta_encoding": encoding,
            "compression_format": "zstd",
            "checksum_format": "xxh3-128",
        },
        "weight_map": {"weight": "model-00000-of-00001.safetensors"},
    }
    (version_dir / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_pull_materializes_base_and_applies_delta(tmp_path, encoding):
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "published"
    local_dir = tmp_path / "local"
    base_dir.mkdir()
    source_dir.mkdir()

    old = np.arange(16, dtype=np.uint8)
    new = old.copy()
    new[[1, 7, 15]] = [91, 92, 93]
    safetensors.numpy.save_file({"weight": old}, base_dir / "model.safetensors")
    _write_delta(source_dir, old, new, encoding=encoding)

    pull(str(local_dir), str(base_dir), str(source_dir), target_version=1)
    np.testing.assert_array_equal(safetensors.numpy.load_file(local_dir / "model.safetensors")["weight"], new)

    pull(str(local_dir), str(base_dir), str(source_dir), target_version=1)
    np.testing.assert_array_equal(safetensors.numpy.load_file(local_dir / "model.safetensors")["weight"], new)


def test_pull_rejects_checksum_mismatch(tmp_path):
    base_dir = tmp_path / "base"
    source_dir = tmp_path / "published"
    local_dir = tmp_path / "local"
    base_dir.mkdir()
    source_dir.mkdir()

    old = np.arange(8, dtype=np.uint8)
    new = old + 1
    safetensors.numpy.save_file({"weight": old}, base_dir / "model.safetensors")
    _write_delta(source_dir, old, new, encoding="xor", digest="not-the-right-checksum")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        pull(str(local_dir), str(base_dir), str(source_dir), target_version=1)
