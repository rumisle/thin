"""Measure how gfx1201 WMMA operations sum and round."""

from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

from tests.hw import _wmma as W  # noqa: N812 - the harness, as W

ULP1 = {"f32": 2.0**-23, "f16": 2.0**-10, "bf16": 2.0**-7}
"""The spacing of D's numbers just above 1."""


def _scales(elem, products):
    """One power-of-two scale a k for all rows, such that the scale and every
    A = p / s are held exactly by the input format (nearest the middle first)."""
    products = np.asarray(products, float)
    s = np.ones(products.shape[1])
    for kk in range(products.shape[1]):
        col = products[:, kk]
        mags = [abs(x) for x in col if x]
        if not mags:
            continue
        mid = (math.log2(min(mags)) + math.log2(max(mags))) / 2
        for e in sorted(range(-160, 161), key=lambda e: abs(e - mid)):
            sc = 2.0**e
            if W.held(elem, [sc])[0] == sc and np.array_equal(W.held(elem, col / sc), col / sc):
                s[kk] = sc
                break
        else:
            raise ValueError(f"no scale holds k = {kk}'s products in {elem}")
    return s


def _dots(form, products, c=None):
    """(GPU, model) D[m][0] of rows of K products each, the products split as
    A = p / s, s[k] = the scale that keeps A in the input format."""
    elem, _, k, _ = W.FORMS[form]
    products = np.asarray(products, float).reshape(-1, k)
    s = _scales(elem, products)
    gpu, emu = [], []
    for at in range(0, len(products), 16):
        chunk = products[at : at + 16]
        A = np.zeros((16, k))  # noqa: N806
        A[: len(chunk)] = chunk / s
        B = np.repeat(s[:, None], 16, axis=1)  # noqa: N806
        C = np.zeros((16, 16))  # noqa: N806
        if c is not None:
            C[: len(chunk)] = np.asarray(c[at : at + 16], float)[:, None]
        assert np.array_equal(W.held(elem, A), A), "an A value the format does not hold"
        assert np.array_equal(W.held(elem, B), B), "a scale the format does not hold"
        e, g = W.run(form, A, B, C)
        gpu += list(W.decode(form, g)[: len(chunk), 0])
        emu += list(W.decode(form, e)[: len(chunk), 0])
    return np.array(gpu), np.array(emu)


def cancel_order(form, E):  # noqa: N803
    """p[i] = +2^E, p[j] = -2^E, p[l] = 1 for every (i, j, l): the exact sum is 1.

    Where the GPU loses the 1, it was added to 2^E before the cancellation at
    a precision that cannot hold 2^E + 1."""
    k = W.FORMS[form][2]
    triples = list(itertools.permutations(range(k), 3))
    rows = []
    for i, j, l in triples:  # noqa: E741
        p = np.zeros(k)
        p[i], p[j], p[l] = 2.0**E, -(2.0**E), 1.0
        rows.append(p)
    gpu, emu = _dots(form, rows)
    lost = [t for t, g in zip(triples, gpu, strict=True) if g != 1.0]
    out = {"exact": 1.0, "big": f"2^{E}", "triples": len(triples), "gpu_lost": len(lost),
           "model_lost": int((emu != 1.0).sum()),
           "gpu_values": sorted({float(g) for g in gpu})}  # fmt: skip
    if lost:  # which (i, j, l) lose: l between i and j in k order, or by groups?
        between = sum(min(i, j) < l < max(i, j) for i, j, l in lost)  # noqa: E741
        out["lost_with_l_between_i_and_j"] = between
        out["lost_examples"] = lost[:10]
        kept = [t for t in triples if t not in set(lost)]
        out["kept_examples"] = kept[:10]
    return out


def sticky_precision(form, deepest):
    """p = 1, 2^-24, 2^-t (f32 D): above the tie only while 2^-t is kept.

    The largest t that still rounds up to 1 + 2^-23 says how far below 1 the
    sum keeps bits (the model keeps all: up to `deepest`)."""
    k = W.FORMS[form][2]
    ts = list(range(25, deepest + 1))
    rows = []
    for t in ts:
        p = np.zeros(k)
        p[0], p[1], p[2] = 1.0, 2.0**-24, 2.0**-t
        rows.append(p)
    gpu, emu = _dots(form, rows)
    up = 1 + 2.0**-23
    kept = [t for t, g in zip(ts, gpu, strict=True) if g == up]
    model_kept = [t for t, m in zip(ts, emu, strict=True) if m == up]
    return {
        "gpu_kept_up_to_t": max(kept, default=None),
        "model_kept_up_to_t": max(model_kept, default=None),
        "gpu": {t: float(g) for t, g in zip(ts, gpu, strict=True)},
    }


def rounding_mode(form):
    """Sums between two of D's neighbours just above 1, both signs."""
    k = W.FORMS[form][2]
    d = W.FORMS[form][3]
    u = ULP1[d]
    cases = {"1 + u/2 (tie, even below)": [1.0, u / 2],
             "1 + u + u/2 (tie, even above)": [1.0, u, u / 2],
             "1 + 3u/4 (above the tie)": [1.0, u / 2, u / 4],
             "1 + u/4 (below the tie)": [1.0, u / 4]}  # fmt: skip
    rows, names = [], []
    for name, ps in cases.items():
        for sign in (1.0, -1.0):
            p = np.zeros(k)
            p[: len(ps)] = sign * np.asarray(ps)
            rows.append(p)
            names.append(("+" if sign > 0 else "-") + name)
    gpu, emu = _dots(form, rows)
    return {n: {"exact": math.fsum(r), "gpu": float(g), "model": float(m),
                "gpu_in_ulps_above_1": round((abs(float(g)) - 1) / u, 3)}
            for n, r, g, m in zip(names, rows, gpu, emu, strict=True)}  # fmt: skip


def c_entry(form, E):  # noqa: N803
    """Where C enters: C = 2^E with p[i] = 1, p[j] = -2^E (exact 1), and C = 1
    with p[i] = 2^E, p[j] = -2^E (exact 1), for every i != j."""
    k = W.FORMS[form][2]
    pairs = list(itertools.permutations(range(k), 2))
    big_c, small_c = [], []
    for i, j in pairs:
        p = np.zeros(k)
        p[i], p[j] = 1.0, -(2.0**E)
        big_c.append(p)
        p = np.zeros(k)
        p[i], p[j] = 2.0**E, -(2.0**E)
        small_c.append(p)
    g1, m1 = _dots(form, big_c, c=[2.0**E] * len(pairs))
    g2, m2 = _dots(form, small_c, c=[1.0] * len(pairs))
    return {"C = 2^E, products 1 and -2^E": {"gpu_exact": int((g1 == 1).sum()),
                                              "model_exact": int((m1 == 1).sum()),
                                              "gpu_values": sorted({float(x) for x in g1})},
            "C = 1, products 2^E and -2^E": {"gpu_exact": int((g2 == 1).sum()),
                                              "model_exact": int((m2 == 1).sum()),
                                              "gpu_values": sorted({float(x) for x in g2})},
            "cases_each": len(pairs), "big": f"2^{E}"}  # fmt: skip


def subnormals(form, tiny_products, tiny_inputs):
    """A subnormal product or input alone: kept or flushed."""
    k = W.FORMS[form][2]
    out = {}
    for label, p_ in tiny_products:
        p = np.zeros(k)
        p[0] = p_
        g, m = _dots(form, [p])
        out[label] = {"gpu": float(g[0]), "model": float(m[0])}
    for label, a in tiny_inputs:  # a subnormal A times a scale of 1
        A = np.zeros((16, k))  # noqa: N806
        A[0, 0] = a
        B = np.ones((k, 16))  # noqa: N806
        e, g = W.run(form, A, B, np.zeros((16, 16)))
        out[label] = {
            "gpu": float(W.decode(form, g)[0, 0]),
            "model": float(W.decode(form, e)[0, 0]),
        }
    return out


def random_models(form, groups=64, seed=0):
    """Random wide-range inputs: how often the GPU equals each candidate model."""
    elem, _, k, d = W.FORMS[form]
    rng = np.random.default_rng(seed)
    spread = {"f16": 5, "bf16": 12, "e4m3fn": 3, "e5m2": 5}[elem]
    f32 = np.float32
    names = ["k-order binary64 model (+ C, one rounding)", "exact sum, one rounding"]
    if d == "f32":
        names += ["binary32 in k order from C", "binary32 in k order, then + C",
                  "binary32 pairwise tree, then + C"]  # fmt: skip
    agree, total = dict.fromkeys(names, 0), 0
    worst = 0.0
    for _ in range(groups):

        def draw(shape, fmt):
            x = rng.standard_normal(shape) * 2.0 ** rng.integers(-spread, spread + 1, shape)
            return W.held(fmt, x)

        A, B, C = draw((16, k), elem), draw((k, 16), elem), draw((16, 16), d)  # noqa: N806
        e, g = W.run(form, A, B, C)
        G, M = W.decode(form, g), W.decode(form, e)  # noqa: N806
        for i in range(16):
            for j in range(16):
                p = [float(A[i][q]) * float(B[q][j]) for q in range(k)]
                exact = W.round_to(d, math.fsum([*p, float(C[i][j])]))
                total += 1
                agree[names[0]] += G[i][j] == M[i][j]
                agree[names[1]] += G[i][j] == exact
                if d == "f32":
                    s = f32(C[i][j])
                    for x in p:
                        s = f32(s + f32(x))
                    agree[names[2]] += G[i][j] == float(s)
                    s = f32(0)
                    for x in p:
                        s = f32(s + f32(x))
                    agree[names[3]] += G[i][j] == float(f32(s + f32(C[i][j])))
                    t = [f32(x) for x in p]
                    while len(t) > 1:
                        t = [f32(t[q] + t[q + 1]) for q in range(0, len(t), 2)]
                    agree[names[4]] += G[i][j] == float(f32(t[0] + f32(C[i][j])))
                if exact and np.isfinite(exact):
                    worst = max(worst, abs(G[i][j] - exact) / abs(exact))
    out = {n: f"{v}/{total}" for n, v in agree.items()}
    out["largest relative error against the exact sum"] = worst
    return out


def random_outputs(form, groups=16, seed=0, record=None):
    """Random wide-range WMMAs (random_models' draw): how many D elements the GPU
    and the emulator agree on, bit for bit.

    record: a list that gains each WMMA's inputs and both D's codes, for the
    measurement report."""
    elem, _, k, d = W.FORMS[form]
    rng = np.random.default_rng(seed)
    spread = {"f16": 5, "bf16": 12, "e4m3fn": 3, "e5m2": 5}[elem]
    agree = total = 0
    for _ in range(groups):

        def draw(shape, fmt):
            x = rng.standard_normal(shape) * 2.0 ** rng.integers(-spread, spread + 1, shape)
            return W.held(fmt, x)

        A, B, C = draw((16, k), elem), draw((k, 16), elem), draw((16, 16), d)  # noqa: N806
        e, g = W.run(form, A, B, C)
        agree += int((e == g).sum())
        total += e.size
        if record is not None:
            record.append({"form": form, "A": A.tolist(), "B": B.tolist(), "C": C.tolist(),
                           "gpu_d": [[hex(int(x)) for x in r] for r in g],
                           "emu_d": [[hex(int(x)) for x in r] for r in e]})  # fmt: skip
    return {"agree": agree, "outputs": total}


def record_randoms(path, forms=None, groups=16, seed=0):
    """random_outputs of every float form into a gzipped JSON file (the GPU's
    D codes and the emulator's, per WMMA): python -m tests.hw.wmma_probe --randoms PATH."""
    import gzip

    import torch

    torch.cuda.set_device(0)
    forms = forms or [f for f, (e, *_) in W.FORMS.items() if not e.startswith("iu")]
    rows, summary = [], {}
    for form in forms:
        summary[form] = random_outputs(form, groups, seed, rows)
        print(form, summary[form], flush=True)
    doc = {"notation": "Random WMMAs on gfx1201 (tests/hw/wmma_probe.py random_outputs): "
           "A, B, C as values (held by the formats), gpu_d and emu_d as D's codes "
           ", D[m][n] at [m][n].",
           "device": torch.cuda.get_device_name(0), "hip": torch.version.hip,
           "seed": seed, "summary": summary, "wmmas": rows}  # fmt: skip
    with gzip.open(path, "wt") as f:
        json.dump(doc, f)
    return summary


PLAN = {
    "f32_16x16x16_f16": [("cancel_order", cancel_order, {"E": 25}),
                         ("sticky_precision", sticky_precision, {"deepest": 48}),
                         ("rounding_mode", rounding_mode, {}),
                         ("c_entry", c_entry, {"E": 25}),
                         ("subnormals", subnormals, {"tiny_products": [],
                          "tiny_inputs": [("f16 subnormal input 2^-20", 2.0**-20)]}),
                         ("random_models", random_models, {})],
    "f32_16x16x16_bf16": [("cancel_order", cancel_order, {"E": 60}),
                          ("sticky_precision", sticky_precision, {"deepest": 160}),
                          ("rounding_mode", rounding_mode, {}),
                          ("c_entry", c_entry, {"E": 60}),
                          ("subnormals", subnormals, {
                              "tiny_products": [("product 2^-140 (an f32 subnormal)", 2.0**-140)],
                              "tiny_inputs": [("bf16 subnormal input 2^-130", 2.0**-130)]}),
                          ("random_models", random_models, {})],
    "f16_16x16x16_f16": [("rounding_mode", rounding_mode, {}),
                         ("subnormals", subnormals, {
                             "tiny_products": [("product 2^-20 (an f16 subnormal D)", 2.0**-20)],
                             "tiny_inputs": [("f16 subnormal input 2^-20", 2.0**-20)]}),
                         ("random_models", random_models, {})],
    "bf16_16x16x16_bf16": [("rounding_mode", rounding_mode, {}),
                           ("random_models", random_models, {})],
    "f32_16x16x16_fp8_fp8": [("random_models", random_models, {})],
    "f32_16x16x16_bf8_bf8": [("random_models", random_models, {})],
}  # fmt: skip


def main(path=None, forms=None):
    import torch

    torch.cuda.set_device(0)
    report = {}
    for form, probes in PLAN.items():
        if forms and form not in forms:
            continue
        report[form] = {}
        for name, fn, kw in probes:
            try:
                report[form][name] = fn(form, **kw)
            except Exception as e:  # recorded: a probe that does not fit says why
                report[form][name] = f"{type(e).__name__}: {e}"
        print(form, json.dumps(report[form], default=str)[:4000], flush=True)
    if path:
        with open(path, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return report


if __name__ == "__main__":
    if sys.argv[1:2] == ["--randoms"]:
        record_randoms(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
