"""One WMMA on gfx1201 and on the emulator, from logical matrices.

`run(form, A, B, C)` packs A (16 x K), B (K x 16) and C (16 x 16) into the
lanes of layout, runs one `v_wmma_<form>` (D = A.B + C) on the GPU
and on the emulated device, and returns both D as logical 16 x 16 arrays.
"""

from __future__ import annotations

import numpy as np

# form -> (A/B element, bits, K, D element)
FORMS = {
    "f32_16x16x16_f16": ("f16", 16, 16, "f32"),
    "f32_16x16x16_bf16": ("bf16", 16, 16, "f32"),
    "f16_16x16x16_f16": ("f16", 16, 16, "f16"),
    "bf16_16x16x16_bf16": ("bf16", 16, 16, "bf16"),
    "i32_16x16x16_iu8": ("iu8", 8, 16, "i32"),
    "i32_16x16x16_iu4": ("iu4", 4, 16, "i32"),
    "i32_16x16x32_iu4": ("iu4", 4, 32, "i32"),
    "f32_16x16x16_fp8_fp8": ("e4m3fn", 8, 16, "f32"),
    "f32_16x16x16_bf8_bf8": ("e5m2", 8, 16, "f32"),
}


def _dwords(form):
    _, bits, k, d = FORMS[form]
    ab = (k // 2) * bits // 32
    cd = 8 * (16 if d in ("f16", "bf16") else 32) // 32
    return ab, cd


def gen_wmma(form: str, signed: bool = False):
    """A KernelDef: each lane loads its A, B, C dwords, one WMMA, D stored."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    ab, cd = _dwords(form)

    def load(b, V, reg, n, base, buf, name):  # noqa: N803
        for at in range(0, n, 4):
            w = min(4, n - at)
            op = {4: "global_load_b128", 2: "global_load_b64", 1: "global_load_b32"}[w]
            b.emit(O.op(op, reg[at : at + w] if n > 1 else reg, base, buf, offset=4 * at or None))

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        a, bb, c, d = k.load()
        lane = V.u32(k.tid, name="lane")
        oa, oc = V.u32(lane * (4 * ab), name="oa"), V.u32(lane * (4 * cd), name="oc")
        A, B = V.u32(n=ab, name="A"), V.u32(n=ab, name="B")  # noqa: N806
        C, D = V.u32(n=cd, name="C"), V.u32(n=cd, name="D")  # noqa: N806
        load(b, V, A, ab, oa, a, "A")
        load(b, V, B, ab, oa, bb, "B")
        load(b, V, C, cd, oc, c, "C")
        b.wait(A, B, C, "A, B, C landed")
        kw = {"neg_lo": (1, 1, 0)} if signed else {}
        b.emit(O.op(f"v_wmma_{form}", D, A, B, C, **kw))
        for at in range(0, cd, 4):
            w = min(4, cd - at)
            op = {4: "global_store_b128", 2: "global_store_b64"}[w]
            b.emit(O.op(op, oc, D[at : at + w], d, offset=4 * at or None))

    args = Args(a=ptr, b=ptr, c=ptr, d=ptr)
    return KernelDef(f"wmma_{form}{'_s' if signed else ''}", args, body, wgsize=32, max_grid=(1,))


def _bits(values, elem):
    """Element codes (uint) of float values or ints in the element format."""
    from thin import formats

    if elem in ("iu8", "iu4"):
        return np.asarray(values, np.int64).astype(np.uint64) & (
            (1 << (8 if elem == "iu8" else 4)) - 1
        )
    if elem == "f16":
        return np.asarray(values, np.float16).view(np.uint16).astype(np.uint64)
    if elem in ("i32",):
        return np.asarray(values, np.int64).astype(np.uint64) & 0xFFFFFFFF
    if elem == "f32":
        return np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    f = np.asarray(values, np.float32)
    return np.asarray(formats.encode(f, formats.DTYPES[elem], saturate=False), np.uint64)


def _pack(codes, bits):
    """Codes of one lane, element e at bit e*bits, as little-endian u32 words."""
    word = 0
    for e, c in enumerate(codes):
        word |= int(c) << (e * bits)
    n = max(1, len(codes) * bits // 32)
    return [(word >> (32 * i)) & 0xFFFFFFFF for i in range(n)]


def pack(form, A, B, C):  # noqa: N803
    """(a, b, c) u32 buffers of lanes."""
    elem, bits, k, d = FORMS[form]
    half = k // 2
    a, b, c = [], [], []
    ca = _bits(A, elem).reshape(16, k)
    cb = _bits(B, elem).reshape(k, 16)
    cc = _bits(C, d).reshape(16, 16)
    cbits = 16 if d in ("f16", "bf16") else 32
    for lane in range(32):
        r, h = lane % 16, lane // 16
        a += _pack(ca[r, half * h : half * h + half], bits)
        b += _pack(cb[half * h : half * h + half, r], bits)
        c += _pack(cc[8 * h : 8 * h + 8, r], cbits)
    return tuple(np.array(x, np.uint32) for x in (a, b, c))


def unpack(form, words):
    """D's codes as a logical 16 x 16 array (uint)."""
    d = FORMS[form][3]
    cbits = 16 if d in ("f16", "bf16") else 32
    per = 8 * cbits // 32
    out = np.zeros((16, 16), np.uint64)
    for lane in range(32):
        r, h = lane % 16, lane // 16
        word = 0
        for i, w in enumerate(words[lane * per : lane * per + per]):
            word |= int(w) << (32 * i)
        for e in range(8):
            out[8 * h + e, r] = (word >> (e * cbits)) & ((1 << cbits) - 1)
    return out


def decode(form, codes):
    """D's codes as numbers: float64 for float forms, int64 for i32."""
    d = FORMS[form][3]
    if d == "f32":
        return codes.astype(np.uint32).view(np.float32).astype(np.float64)
    if d == "f16":
        return codes.astype(np.uint16).view(np.float16).astype(np.float64)
    if d == "bf16":
        return (codes.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return codes.astype(np.uint32).view(np.int32).astype(np.int64)


_KERNELS: dict = {}


def kernel(form, signed=False):
    import thin

    key = (form, signed)
    if key not in _KERNELS:
        before = thin.use_store(None)
        try:
            _KERNELS[key] = thin.compile(gen_wmma, form=form, signed=signed)
        finally:
            thin.use_store(before)
    return _KERNELS[key]


def run(form, A, B, C, signed=False, gpu=True, emulated=True):  # noqa: N803
    """(emulated D codes or None, GPU D codes or None) of one WMMA."""
    k = kernel(form, signed)
    a, b, c = pack(form, A, B, C)
    _, cd = _dwords(form)
    out = []
    if emulated:
        d = np.zeros(32 * cd, np.uint32)
        got = k.bind((1,), {"a": a, "b": b, "c": c, "d": d}).emulate()
        out.append(unpack(form, np.asarray(got.get("d", d))))
    else:
        out.append(None)
    if gpu:
        import torch

        t = {
            n: torch.from_numpy(v.view(np.int32).copy()).cuda()
            for n, v in (("a", a), ("b", b), ("c", c))
        }
        t["d"] = torch.zeros(32 * cd, dtype=torch.int32, device="cuda")
        k.bind((1,), t).run()
        torch.cuda.synchronize()
        out.append(unpack(form, t["d"].cpu().numpy().view(np.uint32)))
    else:
        out.append(None)
    return tuple(out)


def held(fmt, values):
    """The values as the format holds them (RNE), as float64."""
    from thin import formats

    x = np.asarray(values, np.float64)
    if fmt == "f32":
        return x.astype(np.float32).astype(np.float64)
    if fmt == "f16":
        return x.astype(np.float16).astype(np.float64)
    if fmt in ("iu8", "iu4", "i32"):
        return np.rint(x)
    f = formats.DTYPES[fmt]
    codes = formats.encode(x.astype(np.float32), f, saturate=False)
    return np.asarray(formats.decode(codes, f), np.float64)


def round_to(fmt, x: float) -> float:
    """One rounding of an exact sum (a float64 already) to D's format."""
    return float(held(fmt, np.array([x]))[0])


def diagonal(form, cases, gpu=True):
    """(emulated, GPU) D codes of function-level cases (wmma.json.gz:
    dicts with a, b, c), 16 to a WMMA: case t in row t of A, column t of B and
    C[t][t], every other entry 0; D[t][t] depends on those alone."""
    k = FORMS[form][2]
    emu, out = [], []
    for at in range(0, len(cases), 16):
        chunk = cases[at : at + 16]
        A, B, C = np.zeros((16, k)), np.zeros((k, 16)), np.zeros((16, 16))  # noqa: N806
        for t, c in enumerate(chunk):
            A[t], B[:, t], C[t, t] = c["a"], c["b"], c["c"]
        e, g = run(form, A, B, C, gpu=gpu)
        emu += [int(e[t, t]) for t in range(len(chunk))]
        if gpu:
            out += [int(g[t, t]) for t in range(len(chunk))]
    return emu, out


def codes_of(form, values):
    """D's codes of numbers in D's format (held exactly)."""
    return [int(x) for x in _bits(np.asarray(values, float), FORMS[form][3])]
