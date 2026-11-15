"""Every fast format path must equal the bit-level definition.

Opt-in checks cover all f32 patterns for indexing and bf16/f16 casts."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests import exhaustive
from tests.golden import data_path

pytestmark = [
    pytest.mark.skipif(os.environ.get("THIN_EXHAUSTIVE") != "1", reason="set THIN_EXHAUSTIVE=1"),
    pytest.mark.slow("exhaustive"),
]

CAST_SETS = [
    (case["dtype"], case["kwargs"])
    for case in json.loads((data_path("formats", "encode.json")).read_text())["cases"]
    if case["dtype"] in ("bf16", "f16")
]


@pytest.fixture
def device():
    """A GPU when there is one, else the CPU, with torch."""
    torch = pytest.importorskip("torch")
    return torch, "cuda" if torch.cuda.is_available() else "cpu"


def _cached(request, torch, device: str):
    """Whether this check passed on this build and device, and how to record a pass."""
    from thin import _native, formats

    digest = hashlib.sha256()
    for module in (_native, formats):
        digest.update(Path(module.__file__).read_bytes())
    name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    digest.update(f"{torch.__version__} {name}".encode())
    cache = getattr(request.config, "cache", None)
    key = f"thin/exhaustive/{digest.hexdigest()[:16]}/{request.node.name}"
    passed = cache is not None and cache.get(key, False)
    return passed, (lambda: cache.set(key, True)) if cache is not None else (lambda: None)


def test_table_index_everywhere(request, device):
    torch, where = device
    passed, record = _cached(request, torch, where)
    if passed:
        return
    for piece in range(exhaustive.PIECES):
        bad = exhaustive.first_bad_index(torch, piece, where)
        assert bad is None, f"the table index of f32 pattern {bad:#010x} is wrong"
    record()


def test_bf16_and_f16_casts_everywhere(request, device):
    torch, where = device
    passed, record = _cached(request, torch, where)
    if passed:
        return
    for piece in range(exhaustive.PIECES):
        found = exhaustive.first_bad_cast(torch, CAST_SETS, piece, where)
        for (name, kwargs), bad in zip(CAST_SETS, found, strict=True):
            assert bad is None, f"{name} {kwargs}: f32 pattern {bad:#010x} differs"
    record()
