"""WMMA model in whole emulated WMMAs: every wmma.json.gz case
(measured and derived), 16 to a WMMA on its diagonal,
D bit for bit. The D element alone is replayed in crates/thin-emu/tests."""

import collections
import gzip
import json
from pathlib import Path

import pytest

from tests.hw import _wmma as wm

PATH = Path(__file__).parents[1] / "crates/thin-emu/tests/data/emu/wmma.json.gz"


def _forms():
    with gzip.open(PATH) as f:
        doc = json.load(f)
    by = collections.defaultdict(list)
    for c in doc["cases"]:
        by[c["form"]].append(c)
    return by


BY = _forms()


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


@pytest.mark.parametrize("form", sorted(BY))
def test_every_case_in_whole_wmmas(form):
    cases = BY[form]
    emu, _ = wm.diagonal(form, cases, gpu=False)
    want = wm.codes_of(form, [c["d"] for c in cases])
    bad = [(c["probe"], hex(e), hex(w)) for c, e, w in zip(cases, emu, want, strict=True) if e != w]
    assert not bad, (len(bad), bad[:3])


RANDOMS = Path(__file__).parents[1] / "crates/thin-emu/tests/data/emu/wmma_randoms.json.gz"


def _randoms():
    with gzip.open(RANDOMS) as f:
        return json.load(f)


RDOC = _randoms()


@pytest.mark.parametrize("form", list(RDOC["summary"]))
def test_the_recorded_randoms(form, record_property):
    """each recorded WMMA as a whole emulated instruction; D equals
    the GPU's at every output not listed in model_mismatch. The listed ones are
    reported, and their count is the file's summary."""
    import numpy as np

    agree = total = 0
    for w in (w for w in RDOC["wmmas"] if w["form"] == form):
        emu, _ = wm.run(form, np.array(w["A"]), np.array(w["B"]), np.array(w["C"]), gpu=False)
        want = np.array([[int(x, 16) for x in row] for row in w["D"]], np.uint64)
        listed = {tuple(mn) for mn in w["model_mismatch"]}
        same = emu == want
        unlisted = [(m, n) for m in range(16) for n in range(16) if not same[m, n]]
        bad = [mn for mn in unlisted if mn not in listed]
        assert not bad, (len(bad), [(mn, hex(emu[mn]), hex(want[mn])) for mn in bad[:3]])
        agree += int(same.sum())
        total += same.size
    s = RDOC["summary"][form]
    record_property("agree", f"{agree}/{total}")
    assert (agree, total) == (s["model_agrees"], s["outputs"])
