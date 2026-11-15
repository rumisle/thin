"""Fit exact summation models to measured gfx1201 WMMA outputs."""

from __future__ import annotations

import itertools
import json
import math
import sys
from fractions import Fraction

import numpy as np

from tests.hw import _wmma as wm


def _exp(v) -> int:
    return math.floor(math.log2(abs(v)))


def _align(vals, f, mode, refs=None):
    """vals aligned to the largest reference exponent (each value's own leading
    bit, or the given refs: a product's exponent sum), F fraction bits below it."""
    nz = [(v, r) for v, r in zip(vals, refs or [None] * len(vals), strict=True) if v]
    if not nz:
        return [Fraction(0)] * len(vals)
    top = max(_exp(v) if r is None else r for v, r in nz)
    q = Fraction(2) ** (top - f)
    out = []
    for v in vals:
        x = v / q
        n = math.floor(x) if mode == "floor" else int(x)  # int() truncates toward zero
        out.append(n * q)
    return out


def _f32(x: Fraction) -> Fraction:
    return Fraction(float(np.float32(float(x))))  # x exact in f64 here (few bits)


def _round(d, x: Fraction, mode):
    v = float(x)  # exact: an aligned sum has few significant bits
    if mode == "rne":
        return wm.round_to(d, v)
    # toward zero: the format's RNE of v, stepped back if it went past v
    r = wm.round_to(d, v)
    if abs(r) > abs(v):
        bits = {"f32": (np.float32, np.uint32), "f16": (np.float16, np.uint16)}
        if d in bits:
            t, u = bits[d]
            r = float(np.asarray(r, t).view(u).__sub__(1).view(t))
        else:  # bf16: one step down in its 16-bit code
            code = (np.asarray(r, np.float32).view(np.uint32) >> 16) - 1
            r = float((np.asarray(code, np.uint32) << 16).view(np.float32))
    return r


def _step(x: Fraction, how) -> Fraction:
    """The accumulator's rounding to binary32 between groups."""
    v = float(x)
    r = float(np.float32(v))
    if how == "zero" and abs(r) > abs(v):
        r = float(np.nextafter(np.float32(r), np.float32(0)))
    return Fraction(r)


def _order(k, g, order):
    """Group starts in the order they are summed: by k, or interleaving the two
    halves of a lane's operand (k < K/2 and k >= K/2), group by group."""
    starts = list(range(0, k, g))
    if order == "k" or g >= k // 2:
        return starts
    per = (k // 2) // g
    return [h * (k // 2) + i * g for i in range(per) for h in (0, 1)]


def model(products, c, d, g, f, mode, combine, final, order="k", step="rne", ref="lead",
          refs=None):  # fmt: skip
    """One D element under a model (refs: each product's exponent sum, for ref="esum")."""
    p = [Fraction(x) for x in products]
    c = Fraction(c)
    k = len(p)
    if combine == "acc_in_each":  # the accumulator is a term of every group's alignment
        acc = c
        for s in _order(k, g, order):
            rs = None
            if ref == "esum":
                rs = [None, *[refs[q] if p[q] else None for q in range(s, s + g)]]
            acc = _step(sum(_align([acc, *p[s : s + g]], f, mode, rs)), step)
        return _round(d, acc, final)
    groups = [p[i : i + g] for i in range(0, len(p), g)]
    if combine == "c_in_g0":
        groups[0] = [c, *groups[0]]
    sums = [sum(_align(gr, f, mode)) for gr in groups]
    if combine == "align":
        total = sum(_align([c, *sums], f, mode))
    elif combine == "c_in_g0":
        total = sum(_align(sums, f, mode))
    else:  # f32_seq
        acc = _f32(c)
        for s_ in sums:
            acc = _f32(acc + s_)
        total = acc
    return _round(d, total, final)


MODELS = [
    (g, f, mode, "acc_in_each", final, order, step, ref)
    for g in (4, 8)
    for f in (22, 23, 24, 25, 26, 27)
    for mode in ("floor", "zero")
    for final in ("rne", "zero")
    for order in ("halves",)
    for step in ("rne", "zero")
    for ref in ("lead", "esum")
]
"""The family the probes point to (a first fit over sequential f32, re-alignment
and C in group 0 found none above 70%)."""


def _cases(form, rng, randoms):
    """(A, B, C, probe) cases: the probes' rows (f16 and bf16 inputs), then random ones."""
    from tests.hw.wmma_probe import _scales

    elem, _, k, d = wm.FORMS[form]
    out = []

    def rows_to_mats(rows, cs=None):
        rows = np.asarray(rows, float)
        s = _scales(elem, rows)
        for at in range(0, len(rows), 16):
            a = np.zeros((16, k))
            ch = rows[at : at + 16]
            a[: len(ch)] = ch / s
            b = np.repeat(s[:, None], 16, axis=1)
            cm = np.zeros((16, 16))
            if cs is not None:
                cm[: len(ch)] = np.asarray(cs[at : at + 16])[:, None]
            out.append((a, b, cm, True))

    if elem in ("f16", "bf16"):
        big = 2.0**25 if elem == "f16" else 2.0**60
        rows = []
        for i, j, l_ in itertools.permutations(range(k), 3):
            r = np.zeros(k)
            r[i], r[j], r[l_] = big, -big, 1.0
            rows.append(r)
        rows_to_mats(rows)
        for cval, pair in ((big, (1.0, -big)), (1.0, (big, -big))):
            rows = []
            for i, j in itertools.permutations(range(k), 2):
                r = np.zeros(k)
                r[i], r[j] = pair
                rows.append(r)
            rows_to_mats(rows, [cval] * len(rows))
        u = {"f32": 2.0**-23, "f16": 2.0**-10, "bf16": 2.0**-7}[d]
        rows = []
        for ps in (
            [1, u / 2],
            [1, u, u / 2],
            [1, u / 2, u / 4],
            [1, u / 4],
            [1, 2.0**-24, 2.0**-30],
        ):
            for sign in (1, -1):
                r = np.zeros(k)
                r[: len(ps)] = sign * np.asarray(ps)
                rows.append(r)
        rows_to_mats(rows)
    spread = {"f16": 5, "bf16": 12, "e4m3fn": 3, "e5m2": 5}[elem]
    for _ in range(randoms):

        def draw(shape, fmt):
            x = rng.standard_normal(shape) * 2.0 ** rng.integers(-spread, spread + 1, shape)
            return wm.held(fmt, x)

        out.append((draw((16, k), elem), draw((k, 16), elem), draw((16, 16), d), False))
    return out


def fit(form, randoms=16, seed=1):
    """Each model's agreement with the GPU on every output of every case; a
    model is dropped once it misses more than 2% (after 1000 outputs)."""
    _, _, k, d = wm.FORMS[form]
    rng = np.random.default_rng(seed)
    alive = {m: 0 for m in MODELS}
    total = 0
    for a, b, c, probe in _cases(form, rng, randoms):
        _, gpu = wm.run(form, a, b, c, emulated=False)
        got = wm.decode(form, gpu)
        cols = [0] if probe else range(16)
        for i in range(16):
            for j in cols:
                prods = [float(a[i][q]) * float(b[q][j]) for q in range(k)]
                refs = [_exp(a[i][q]) + _exp(b[q][j]) if a[i][q] and b[q][j] else None
                        for q in range(k)]  # fmt: skip
                total += 1
                for m in list(alive):
                    alive[m] += model(prods, float(c[i][j]), d, *m, refs=refs) == got[i][j]
        if total > 1000:
            alive = {m: v for m, v in alive.items() if v >= 0.98 * total} or alive
    best = sorted(alive.items(), key=lambda kv: -kv[1])[:6]
    names = ("group", "fraction_bits", "align", "combine", "final", "order", "step", "ref")
    return {
        "outputs": total,
        "best": [{"model": dict(zip(names, m, strict=True)), "agree": int(v)} for m, v in best],
    }


def main(path=None):
    import torch

    torch.cuda.set_device(0)
    report = {}
    for form in ("f32_16x16x16_f16", "f32_16x16x16_bf16", "f16_16x16x16_f16", "bf16_16x16x16_bf16",
                 "f32_16x16x16_fp8_fp8", "f32_16x16x16_bf8_bf8"):  # fmt: skip
        report[form] = fit(form)
        print(form, json.dumps(report[form]), flush=True)
    if path:
        with open(path, "w") as f:
            json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
