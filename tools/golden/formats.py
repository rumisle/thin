"""Format inputs constructed from boundary values, Cartesian products and seed 0.

The random stream is consumed in a fixed order: encoding, MX, packing, layouts.
No tests/data file is read. Numerical/byte results are produced by thin.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess

import numpy as np

from thin import formats as f
from tools.golden import ROOT, encode

BASE = "crates/thin-formats/tests/data/formats/"
OUTPUTS = tuple(
    BASE + name
    for name in (
        "decode.json",
        "encode.json",
        "mx.json",
        "pack.json",
        "layout.json",
        "msgpack.json",
        "recipe.json",
        "codeobject.json",
        "codeobject_k.co",
    )
)


def hexes(values):
    """f32 arrays as exact bit patterns, including payloads and signed zero."""
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    return np.vectorize(lambda v: f"{v:08x}", otypes=[str])(words).tolist()


def native(requests):
    """Call thin-formats directly for internal Recipe, pack and MessagePack APIs."""
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-formats", "--example", "golden"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def decode_data():
    """Exhaust the small codes; probe normal/subnormal/sign/NaN 16-bit edges."""
    result = {}
    for name in ("e4m3fn", "e5m2", "e2m1", "e2m3", "e3m2", "e8m0"):
        dtype = f.DTYPES[name]
        bits = np.arange(1 << dtype.width, dtype=dtype.numpy)
        result[name] = hexes(f.decode(bits, dtype))
    # IEEE-half and bfloat boundaries, combined so both formats see each code.
    codes = {0, 1, 1 << 15}
    for exponent, mantissa in ((5, 10), (8, 7)):
        exp = (1 << exponent) - 1
        codes |= {exp << mantissa, (1 << 15) | (exp << mantissa), (exp << mantissa) | 1}
    codes |= {0x3C00, 0x3F80, 0x7E00, 0xFE01, 0x7BFF, 0x400, 0x3FF}
    for name in ("bf16", "f16"):
        bits = np.array(sorted(codes), dtype=np.uint16)
        result[name] = dict(
            zip((f"{v:04x}" for v in bits), hexes(f.decode(bits, f.DTYPES[name])), strict=True)
        )
    return result


def encode_data(rng):
    """Boundary midpoints, extreme f32s, both signs and 200 random bit patterns."""
    finite = [
        1,
        17 / 16,
        9 / 8,
        19 / 16,
        6,
        6.5,
        7,
        448,
        464,
        480,
        65504,
        65520,
        65536,
        np.finfo(np.float32).max,
        2.0**-2,
        2.0**-4,
        2.0**-7,
        2.0**-8,
        2.0**-9,
        2.0**-24,
        2.0**-25,
        np.nextafter(np.float32(2.0**-25), np.float32(np.inf)),
        2.0**-14 - 2.0**-24,
        2.0**-14,
        1 + 2.0**-8,
        1 + 3 * 2.0**-8,
        1 + 3 * 2.0**-9,
        np.nextafter(np.float32(1), np.float32(np.inf)),
        np.nextafter(np.float32(1), np.float32(0)),
        3,
        5,
        5.5,
    ]
    words = [
        0,
        1 << 31,
        0xFF << 23,
        (1 << 31) | (0xFF << 23),
        (0xFF << 23) | (1 << 22),
        (1 << 31) | (0xFF << 23) | (1 << 22) | 1,
        (0xFF << 23) | 1,
        1,
        (1 << 31) | ((1 << 23) - 1),
    ]
    values = np.asarray(finite, dtype=np.float32).view(np.uint32).tolist()
    words += values + [v | (1 << 31) for v in values]
    words += rng.integers(0, 1 << 32, size=200, dtype=np.uint32).tolist()
    x = np.array(words, dtype=np.uint32).view(np.float32)
    configs = []
    for name in ("bf16", "e4m3fn"):
        for saturate in (False, True) if name == "bf16" else (True, False):
            configs.append((name, {"saturate": saturate}))
    configs.extend(
        ("e2m1", kw)
        for kw in (
            {"saturate": True, "nan": 0},
            {"saturate": True, "nan": 7},
            {"saturate": False, "nan": 0, "inf": 7},
            {"saturate": False, "nan": 3, "inf": 5},
        )
    )
    for name in ("f16", "e5m2"):
        configs.extend((name, {"saturate": sat}) for sat in (False, True))
    for name in ("e2m3", "e3m2"):
        configs.extend(
            (name, {"saturate": sat, "nan": 0, **({} if sat else {"inf": 31})})
            for sat in (True, False)
        )
    configs.append(("f32", {"saturate": False}))
    cases = []
    for name, kwargs in configs:
        bits = f.encode(x, f.DTYPES[name], **kwargs)
        cases.append(
            {"dtype": name, "kwargs": kwargs, "bits": bits.view(f"u{bits.itemsize}").tolist()}
        )
    return {"inputs_f32_hex": [f"{v:08x}" for v in words], "cases": cases}


def mx_data(rng):
    """Two-row MX blocks over four magnitude regimes plus empty/malformed shapes."""
    cases = []
    for name in ("e2m1", "e4m3fn"):
        dtype = f.DTYPES[name]
        for scale in (1.0, 1e-3, 3e4, 1e-40):
            x = (rng.standard_normal((2, 64)) * scale).astype(np.float32)
            x[0, 5] = 0
            x[1, 0] = 1.75 * scale
            if name == "e2m1" and scale == 1:
                x[1, 33] = np.nan
            codes, scales = f.quantize(x, dtype)
            cases.append(
                {
                    "dtype": name,
                    "x_f32_hex": hexes(x),
                    "codes": codes.tolist(),
                    "scales": scales.tolist(),
                }
            )
        amax = np.array(
            [0, 1, 1.5, 1.74, 1.75, 2, 3.5, 448, 6, 1e-38, 1e30, 3e38], dtype=np.float32
        )
        cases.append(
            {
                "dtype": name,
                "scale_bytes_amax_f32_hex": hexes(amax),
                "bytes": f.scale_bytes(amax, dtype).tolist(),
            }
        )
    for name in ("e2m1", "e4m3fn"):
        for shape in ((3, 0), (0,), (0, 32), (2, 0, 64), (), (3, 31)):
            case = {"dtype": name, "shape": list(shape)}
            try:
                codes, scales = f.quantize(np.zeros(shape, dtype=np.float32), f.DTYPES[name])
                case.update(codes_shape=list(codes.shape), scales_shape=list(scales.shape))
            except ValueError as error:
                case["refused"] = type(error).__name__
            cases.append(case)
    return cases


def pack_data(rng):
    """Sub-byte tails, whole bytes and 16-bit storage, at odd/even row widths."""
    cases, requests = [], []
    for name in ("e2m1", "e2m3", "e3m2", "u8", "bf16"):
        dtype = f.DTYPES[name]
        for columns in (1, 3, 4, 5, 8):
            bits = rng.integers(0, 1 << dtype.width, size=(2, 2, columns)).astype(dtype.numpy)
            cases.append({"dtype": name, "bits": bits.tolist()})
            requests.append(
                {
                    "op": "pack",
                    "dtype": name,
                    "shape": list(bits.shape),
                    "bytes": bits.tobytes().hex(),
                }
            )
            requests[-1]["bytes"] = list(bytes.fromhex(requests[-1]["bytes"]))
    for case, result in zip(cases, native(requests), strict=True):
        case.update(result)
    return cases


def layout_data(rng):
    """Layout constructor/type products and out-of-range sub-byte storage."""
    cases = []
    for key in ("row_major", "row_major(ld=6)", "row_major(pad_to=4)", "col_major"):
        for name, shape in (("u8", (2, 3, 5)), ("bf16", (1, 2, 3)), ("e2m1", (1, 3, 6))):
            if key == "col_major" and name == "e2m1":
                continue
            dtype = f.DTYPES[name]
            bits = rng.integers(0, 1 << dtype.width, size=shape).astype(dtype.numpy)
            layout = f.Layout.parse(key)
            raw = f.to_bytes(bits, layout, dtype)
            cases.append(
                {
                    "key": key,
                    "dtype": name,
                    "shape": list(shape),
                    "bits": bits.tolist(),
                    "bytes": raw.tolist(),
                    "nbytes": f.nbytes(layout, dtype, shape),
                    "coords": [
                        a.tolist() for a in f.coords(layout, dtype, shape, np.arange(len(raw)))
                    ],
                }
            )
    for name, invalid in (("e2m1", 1 << 4), ("e2m3", 1 << 6), ("e3m2", 255)):
        bits = np.array([[[0, invalid]]], dtype=np.uint8)
        case = {"key": "row_major", "dtype": name, "shape": list(bits.shape), "bits": bits.tolist()}
        try:
            f.to_bytes(bits, f.row_major(), f.DTYPES[name])
        except ValueError as error:
            case.update(error=type(error).__name__, layout_error=isinstance(error, f.LayoutError))
        cases.append(case)
    return cases


def msgpack_data():
    """Integer-width transitions, collection headers, key validity and refusals."""
    values = [None, True, False, 0]
    for bits in (7, 8, 16):
        values += [(1 << bits) - 1, 1 << bits]
    values += [
        1 << 32,
        1 << 63,
        -1,
        -(1 << 5),
        -(1 << 5) - 1,
        -(1 << 7),
        -(1 << 7) - 1,
        -(1 << 31),
        -(1 << 63),
    ]
    values += ["a" * n for n in (0, 31, 32, 256)]
    values += [
        [],
        list(range(16)),
        {"b": 1, "a": [1, "x"]},
        {f"k{i:02}": i for i in range(16)},
        1 << 64,
        -(1 << 63) - 1,
        {"é": 1},
        1.5,
        {"a": {"b": [1 << 70]}},
        {"a": 1.0},
    ]
    return [
        {"value": value, **result}
        for value, result in zip(
            values, native([{"op": "msgpack", "value": v} for v in values]), strict=True
        )
    ]


def recipe_data():
    """Four combinations of padding/repetition on one or separate axes."""
    cases = []
    plans = (
        ("pad only", [[0, 0], [1, 0], [1, 2]], []),
        ("repeat only", [], [[1, 3]]),
        ("pad and repeat on different dimensions", [[0, 0], [0, 0], [1, 1]], [[1, 2]]),
        ("pad and repeat on the same dimension", [[0, 0], [1, 2], [0, 0]], [[1, 2]]),
    )
    requests = []
    for name, pad, repeat in plans:
        logical = np.arange(1, 7, dtype=np.uint8).reshape(2, 1, 3)
        case = {
            "name": name,
            "rows": 1,
            "row_bytes": 3,
            "dims": [2, 1, 3],
            "perm": list(range(3)),
            "fill": 0xEE,
            "pad": pad,
            "repeat": repeat,
            "logical": logical.tolist(),
        }
        cases.append(case)
        requests.append(
            {
                "op": "recipe",
                "shape": list(logical.shape),
                "bytes": logical.ravel().tolist(),
                **case,
            }
        )
    for case, result in zip(cases, native(requests), strict=True):
        case.update(result)
    return {
        "notation": (
            "Native Recipe padding/repetition over six ascending logical bytes; "
            "coordinates identify source bytes or padding."
        ),
        "cases": cases,
    }


def codeobject_data():
    """Construct synthetic container bytes and round-trip all structured metadata."""
    symbol = "k"
    # This is an arbitrary synthetic byte payload, not a shipped kernel.
    text = bytes(range(256)) * 2 + struct.pack("<I", 0xBF810000)
    descriptor = struct.pack("<IIQ", 1024, 0, 16) + bytes(48)
    metadata = {
        "amdhsa.version": [1, 2],
        "amdhsa.kernels": [
            {
                ".name": symbol,
                ".symbol": symbol + ".kd",
                ".kernarg_segment_size": 16,
                ".group_segment_fixed_size": 1024,
                ".args": [{".offset": 0, ".size": 8, ".value_kind": "global_buffer"}],
            }
        ],
        "amdhsa.target": "amdgcn-amd-amdhsa--gfx1201",
    }
    flags = 0x4E

    def args(meta):
        return {
            "symbol": symbol,
            "text_hex": text.hex(),
            "descriptor_hex": descriptor.hex(),
            "metadata": meta,
            "e_flags": flags,
        }

    image = f.write(symbol, text, descriptor, metadata, flags)
    obj, code = f.read(image), f.code(image)
    doc = {
        "write_args": args(metadata),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "image_len": len(image),
        "image_file": "codeobject_k.co",
        "sections": [list(vars(section).values()) for section in obj.sections],
        "descriptor_after_write_hex": obj.descriptor.hex(),
        "code_metadata": code.metadata,
        "code_digests": dict(code.digests),
        "target": obj.target,
        "lds": obj.lds,
        "kernarg": obj.kernarg,
    }
    minimal = {"amdhsa.kernels": [{".name": symbol, ".symbol": symbol + ".kd"}]}
    flag_cases = []
    for flag in (
        flags,
        flags | 1 << 8,
        flags | 3 << 8,
        flags | 0xFFFFFF00,
        flags - 2,
        (flags - 2) | 5 << 8,
        0x3F,
        0,
    ):
        case = {"e_flags": flag}
        try:
            back = f.read(f.write(symbol, struct.pack("<I", 0xBF810000), bytes(64), minimal, flag))
            case.update(read_e_flags=back.e_flags, target=back.target)
        except f.CodeObjectError as error:
            case["refused"] = type(error).__name__
        flag_cases.append(case)
    doc["write_e_flags"] = {
        "metadata": minimal,
        "text_hex": struct.pack("<I", 0xBF810000).hex(),
        "descriptor_hex": bytes(64).hex(),
        "cases": flag_cases,
    }
    padded_meta = {**metadata, "x": "abc"}
    padded_image = f.write(symbol, text, descriptor, padded_meta, flags)
    back = f.read(padded_image)
    note = next(section for section in back.sections if section.name == ".note")
    raw_note = padded_image[note.offset : note.offset + note.size]
    namesz, descsz, kind = struct.unpack_from("<3I", raw_note)
    doc["note_padding"] = {
        "write_args": args(padded_meta),
        "msgpack_len": descsz,
        "namesz": namesz,
        "descsz": descsz,
        "type": kind,
        "note_size": note.size,
        "note_hex": raw_note.hex(),
        "image_sha256": hashlib.sha256(padded_image).hexdigest(),
        "image_len": len(padded_image),
        "read_back_metadata_equal": back.metadata == padded_meta,
    }
    return doc, image


def generate():
    """Evaluate all host-side format cases with a single deterministic RNG stream."""
    rng = np.random.default_rng(0)
    encoded = encode_data(rng)
    mx = mx_data(rng)
    packed = pack_data(rng)
    layouts = layout_data(rng)
    container, image = codeobject_data()
    documents = [
        decode_data(),
        encoded,
        mx,
        packed,
        layouts,
        msgpack_data(),
        recipe_data(),
        container,
    ]
    return {
        **{path: encode(doc) for path, doc in zip(OUTPUTS[:-1], documents, strict=True)},
        OUTPUTS[-1]: image,
    }
