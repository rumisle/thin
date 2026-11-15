"""Check matrix layouts and measured summation on gfx1201.

Layouts follow thin_target::MatrixForm. Whole WMMA instructions are compared
with the CPU model across measured cases and randomized inputs."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hw import _wmma as wm

pytestmark = pytest.mark.gfx1201
pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


CASES = [
    (f, s)
    for f, (e, *_) in wm.FORMS.items()
    for s in ((False, True) if e.startswith("iu") else (False,))
]


@pytest.mark.parametrize(
    ("form", "signed"), CASES, ids=[f"{f}{'-signed' if s else ''}" for f, s in CASES]
)
def test_the_layouts_where_sums_are_exact(form, signed):
    elem, _, k, _ = wm.FORMS[form]
    rng = np.random.default_rng(0)
    if elem.startswith("iu"):
        lo, hi = (
            ((-8, 8) if elem == "iu4" else (-128, 128))
            if signed
            else ((0, 16) if elem == "iu4" else (0, 256))
        )
        A, B = rng.integers(lo, hi, (16, k)), rng.integers(lo, hi, (k, 16))  # noqa: N806
        C = rng.integers(-1000, 1000, (16, 16))  # noqa: N806
    else:  # small integers: every order of every sum is exact
        A = rng.integers(-3, 4, (16, k)).astype(float)  # noqa: N806
        B = rng.integers(-3, 4, (k, 16)).astype(float)  # noqa: N806
        C = rng.integers(-3, 4, (16, 16)).astype(float)  # noqa: N806
    emu, gpu = wm.run(form, A, B, C, signed=signed)
    assert np.array_equal(wm.decode(form, gpu), A @ B + C)
    assert np.array_equal(emu, gpu)


GROUPED = {"f": 24, "mode": "floor", "combine": "acc_in_each", "order": "halves", "step": "rne"}
"""The measured structure: groups in half-interleaved order, the accumulator a
term of each group's alignment, F fraction bits below the largest, RNE between."""


def test_the_measured_order_of_an_f16_sum():
    """+2^25, -2^25 and 1 at every k positions (i, j, l): the GPU keeps the 1
    exactly where the grouped model does (groups of 4 summed in the order k
    0-3, 8-11, 4-7, 12-15), in 800 of 3360 triples, and the emulator with it."""
    import itertools

    from tests.hw.wmma_fit import model
    from tests.hw.wmma_probe import _dots

    triples = list(itertools.permutations(range(16), 3))
    rows = []
    for i, j, l_ in triples:
        p = np.zeros(16)
        p[i], p[j], p[l_] = 2.0**25, -(2.0**25), 1.0
        rows.append(p)
    gpu, emu = _dots("f32_16x16x16_f16", rows)
    g = GROUPED
    want = [model(list(r), 0.0, "f32", 4, g["f"], g["mode"], g["combine"], "rne", g["order"],
                  g["step"]) for r in rows]  # fmt: skip
    assert int((gpu == 1.0).sum()) == 800
    assert gpu.tolist() == want
    assert emu.tolist() == gpu.tolist()


def test_where_c_enters():
    """C starts the accumulator: C = 2^25 with products 1 and -2^25 keeps the 1
    exactly when -2^25's group comes first (96 of 240), and C = 1 is lost to
    any ±2^25 pair (0 of 240)."""
    from tests.hw.wmma_probe import c_entry

    r = c_entry("f32_16x16x16_f16", E=25)
    big, small = r["C = 2^E, products 1 and -2^E"], r["C = 1, products 2^E and -2^E"]
    assert big["gpu_exact"] == big["model_exact"] == 96
    assert small["gpu_exact"] == small["model_exact"] == 0


def test_fp8_fits_the_grouped_model():
    """fp8 x fp8 -> f32: groups of 8 aligned on the products' exponent sums at 24
    fraction bits by floor, RNE to binary32 after each (4095 of 4096 measured)."""
    import tests.hw.wmma_fit as wf

    saved = wf.MODELS
    wf.MODELS = [(8, 24, "floor", "acc_in_each", "rne", "halves", "rne", "esum")]
    try:
        r = wf.fit("f32_16x16x16_fp8_fp8", randoms=16)
    finally:
        wf.MODELS = saved
    assert r["best"][0]["agree"] >= r["outputs"] - 4, r


def test_the_measured_vectors_in_whole_wmmas():
    """wmma.json.gz's measured cases, 16 to a WMMA on its diagonal:
    the GPU's D equals the case's d and the emulator's, bit for bit."""
    import collections
    import gzip
    import json
    from pathlib import Path

    path = Path(__file__).parents[2] / "crates/thin-emu/tests/data/emu/wmma.json.gz"
    by = collections.defaultdict(list)
    with gzip.open(path) as f:
        doc = json.load(f)
    for c in doc["cases"]:
        if c["source"].startswith("measured"):
            by[c["form"]].append(c)
    assert sum(map(len, by.values())) == 7877
    for form, cases in by.items():
        emu, gpu = wm.diagonal(form, cases)
        want = wm.codes_of(form, [c["d"] for c in cases])
        bad = [
            (c["probe"], hex(g), hex(w)) for c, g, w in zip(cases, gpu, want, strict=True) if g != w
        ]
        assert not bad, (form, len(bad), bad[:3])
        assert emu == gpu, form


def test_random_fp8_sums_agree_with_the_emulator():
    """is normative for fp8: of 4096 random outputs at least 4090 agree with
    the emulator."""
    from tests.hw.wmma_probe import random_outputs

    r = random_outputs("f32_16x16x16_fp8_fp8", groups=16)
    assert r["outputs"] == 4096 and r["agree"] >= 4090, r


EXACT = ["f32_16x16x16_f16", "f32_16x16x16_bf16", "f16_16x16x16_f16", "bf16_16x16x16_bf16"]


@pytest.mark.parametrize("form", EXACT)
def test_random_f16_and_bf16_sums_are_the_gpus(form):
    """f16 and bf16 inputs run through gfx1201's dot unit (thin_emu's
    machine::dot): every one of 4096 random outputs, bit for bit."""
    from tests.hw.wmma_probe import random_outputs

    r = random_outputs(form, groups=16)
    assert r["agree"] == r["outputs"] == 4096, r


def test_random_bf8_sums_are_reported(record_property):
    """bf8's random agreement, reported, not asserted (4089 of 4096 on the
    recorded draw, wmma_randoms.json.gz)."""
    from tests.hw.wmma_probe import random_outputs

    r = random_outputs("f32_16x16x16_bf8_bf8", groups=16)
    record_property("agree", f"{r['agree']}/{r['outputs']}")
    print(r)
