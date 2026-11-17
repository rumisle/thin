"""Golden checks of the Python format API.

Rust golden tests exercise the same encodings and byte containers directly."""

import json

import numpy as np
import pytest

from tests.golden import cases, data_path, whole
from thin import formats
from thin.formats import DTYPES

PARAMS = ("vector", "case_id", "case", "doc")


def f32(hexes) -> np.ndarray:
    """f32 values from their bits in hex."""
    return np.array([int(h, 16) for h in hexes], dtype=np.uint32).view(np.float32)


@pytest.mark.parametrize(PARAMS, cases("formats", "decode.json"))
def test_decode(vector, case_id, case, doc):
    name, key = case_id[:-1].split("[")
    dtype = DTYPES[name]
    pattern = int(key) if isinstance(doc[name], list) else int(key, 16)
    bits = np.array([pattern], dtype=dtype.numpy)
    assert formats.decode(bits, dtype).view(np.uint32)[0] == int(case, 16)


@pytest.mark.parametrize(PARAMS, cases("formats", "encode.json", "cases"))
def test_encode(vector, case_id, case, doc):
    dtype = DTYPES[case["dtype"]]
    bits = formats.encode(f32(doc["inputs_f32_hex"]), dtype, **case["kwargs"])
    assert bits.dtype == np.dtype(dtype.numpy)
    assert bits.view(f"u{bits.itemsize}").tolist() == case["bits"]


@pytest.mark.parametrize(PARAMS, cases("formats", "mx.json"))
def test_mx(vector, case_id, case, doc):
    dtype = DTYPES[case["dtype"]]
    if "scale_bytes_amax_f32_hex" in case:
        got = formats.scale_bytes(f32(case["scale_bytes_amax_f32_hex"]), dtype)
        assert got.tolist() == case["bytes"]
        return
    if "shape" in case:
        x = np.zeros(case["shape"], dtype=np.float32)
        if "refused" in case:
            with pytest.raises(Exception) as caught:
                formats.quantize(x, dtype)
            assert type(caught.value).__name__ == case["refused"]
            return
        codes, scales = formats.quantize(x, dtype)
        assert list(codes.shape) == case["codes_shape"]
        assert list(scales.shape) == case["scales_shape"]
        return
    x = np.stack([f32(row) for row in case["x_f32_hex"]])
    codes, scales = formats.quantize(x, dtype)
    assert (codes.tolist(), scales.tolist()) == (case["codes"], case["scales"])


@pytest.mark.parametrize(PARAMS, cases("formats", "layout.json"))
def test_layout(vector, case_id, case, doc):
    dtype = DTYPES[case["dtype"]]
    layout = formats.Layout.parse(case["key"])
    shape = tuple(case["shape"])
    bits = np.array(case["bits"], dtype=dtype.numpy)
    if "error" in case:
        with pytest.raises(Exception) as caught:
            formats.to_bytes(bits, layout, dtype)
        assert type(caught.value).__name__ == case["error"]
        assert isinstance(caught.value, formats.LayoutError) == case["layout_error"]
        return
    assert formats.nbytes(layout, dtype, shape) == case["nbytes"]
    raw = formats.to_bytes(bits, layout, dtype)
    assert raw.tolist() == case["bytes"]
    assert np.array_equal(formats.from_bytes(raw, layout, dtype, shape), bits)
    got = formats.coords(layout, dtype, shape, np.arange(case["nbytes"]))
    assert [c.tolist() for c in got] == case["coords"]


@pytest.mark.parametrize(PARAMS, whole("formats", "codeobject.json"))
def test_codeobject(vector, case_id, case, doc):
    args = doc["write_args"]
    image = formats.write(
        args["symbol"],
        bytes.fromhex(args["text_hex"]),
        bytes.fromhex(args["descriptor_hex"]),
        args["metadata"],
        args["e_flags"],
    )
    assert image == (data_path("formats", doc["image_file"])).read_bytes()
    obj = formats.read(image)
    assert [list(vars(s).values()) for s in obj.sections] == doc["sections"]
    assert obj.descriptor.hex() == doc["descriptor_after_write_hex"]
    assert (obj.symbol, obj.target, obj.lds, obj.kernarg) == (
        args["symbol"],
        doc["target"],
        doc["lds"],
        doc["kernarg"],
    )
    normalized = formats.code(image)
    assert normalized.metadata == doc["code_metadata"]
    assert dict(normalized.digests) == doc["code_digests"]
    flags = doc["write_e_flags"]
    parts = (bytes.fromhex(flags["text_hex"]), bytes.fromhex(flags["descriptor_hex"]))
    for each in flags["cases"]:
        if "refused" in each:
            with pytest.raises(formats.CodeObjectError):
                formats.write("k", *parts, flags["metadata"], each["e_flags"])
            continue
        back = formats.read(formats.write("k", *parts, flags["metadata"], each["e_flags"]))
        assert (back.e_flags, back.target) == (each["read_e_flags"], each["target"])


def test_codeobject_note_padding():
    """descsz is the unpadded msgpack length; the note is the padded total."""
    import hashlib
    import struct

    from tests.golden import load

    doc = load("formats", "codeobject.json")["note_padding"]
    args = doc["write_args"]
    image = formats.write(
        args["symbol"],
        bytes.fromhex(args["text_hex"]),
        bytes.fromhex(args["descriptor_hex"]),
        args["metadata"],
        args["e_flags"],
    )
    assert (len(image), hashlib.sha256(image).hexdigest()) == (
        doc["image_len"],
        doc["image_sha256"],
    )
    note = next(s for s in formats.read(image).sections if s.name == ".note")
    data = image[note.offset : note.offset + note.size]
    assert (note.size, data.hex()) == (doc["note_size"], doc["note_hex"])
    namesz, descsz, kind = struct.unpack_from("<3I", data)
    assert (namesz, descsz, kind) == (doc["namesz"], doc["descsz"], doc["type"])
    assert descsz == doc["msgpack_len"] and note.size == 12 + 8 + -(-descsz // 4) * 4
    assert (formats.read(image).metadata == args["metadata"]) == doc["read_back_metadata_equal"]


KERNELS = {"amdhsa.kernels": [{".name": "k", ".symbol": "k.kd"}]}


@pytest.mark.parametrize(PARAMS, cases("formats", "msgpack.json"))
def test_msgpack(vector, case_id, case, doc):
    """msgpack is internal to the code object, so its values ride in the
    metadata of a written code object: refused as given, or written with
    their bytes and read back equal."""
    metadata = {**KERNELS, "v": case["value"]}
    if "refused" in case:
        with pytest.raises(getattr(formats, case["refused"])):
            formats.write("k", b"\0", bytes(64), metadata, 0x4E)
        return
    image = formats.write("k", b"\0", bytes(64), metadata, 0x4E)
    assert b"\xa1v" + bytes.fromhex(case["hex"]) in image
    assert formats.read(image).metadata["v"] == case["value"]


# --- The Python layer's refusals and edge cases ----------

E4M3, E2M1, BF16, F16 = (DTYPES[n] for n in ("e4m3fn", "e2m1", "bf16", "f16"))


def test_dtype_records():
    assert repr(BF16) == "bf16" and BF16.torch == "bfloat16" and BF16.dlpack == (4, 16)
    assert (E4M3.encoding.max, E2M1.encoding.max) == (448.0, 6.0)
    f8 = DTYPES["e4m3fnuz"]
    b8 = DTYPES["e5m2fnuz"]
    assert (f8.torch, f8.dlpack, f8.encoding.nan, f8.encoding.inf) == (
        "float8_e4m3fnuz",
        (11, 8),
        "NEG_ZERO",
        False,
    )
    assert (f8.encoding.bias, f8.encoding.max_bits, f8.encoding.max) == (8, 0x7F, 240.0)
    assert (b8.torch, b8.dlpack, b8.encoding.nan, b8.encoding.inf) == (
        "float8_e5m2fnuz",
        (13, 8),
        "NEG_ZERO",
        False,
    )
    assert (b8.encoding.bias, b8.encoding.max_bits, b8.encoding.max) == (16, 0x7F, 57344.0)
    assert DTYPES["u8"].encoding is None
    with pytest.raises(AttributeError):
        BF16.name = "x"


def test_fnuz_encode_and_decode():
    f8, b8 = DTYPES["e4m3fnuz"], DTYPES["e5m2fnuz"]
    xs = np.array(
        [
            0.0,
            -0.0,
            -1e-30,
            2**-10,
            -(2**-10),
            1.0,
            240.0,
            -240.0,
            248.0,
            -248.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    nosat = formats.encode(xs, f8, saturate=False).tolist()
    sat = formats.encode(xs, f8, saturate=True).tolist()
    assert nosat == [0x00, 0x00, 0x00, 0x01, 0x81, 0x40, 0x7F, 0xFF, 0x80, 0x80, 0x80, 0x80, 0x80]
    assert sat == [0x00, 0x00, 0x00, 0x01, 0x81, 0x40, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF, 0x80]
    dec = formats.decode(np.array([0x00, 0x80, 0x01, 0x7F, 0xFF], dtype=np.uint8), f8)
    assert dec.view(np.uint32)[1] == 0x7FC00000
    assert dec[[0, 2, 3, 4]].tolist() == [0.0, 2**-10, 240.0, -240.0]

    bxs = np.array(
        [0.0, -0.0, -1e-30, 2**-17, 57344.0, -57344.0, 61440.0, -np.inf, np.nan], np.float32
    )
    assert formats.encode(bxs, b8, saturate=False).tolist() == [
        0x00,
        0x00,
        0x00,
        0x01,
        0x7F,
        0xFF,
        0x80,
        0x80,
        0x80,
    ]
    assert formats.encode(bxs, b8, saturate=True).tolist() == [
        0x00,
        0x00,
        0x00,
        0x01,
        0x7F,
        0xFF,
        0x7F,
        0xFF,
        0x80,
    ]
    bdec = formats.decode(np.array([0x00, 0x80, 0x01, 0x7F, 0xFF], dtype=np.uint8), b8)
    assert bdec.view(np.uint32)[1] == 0x7FC00000
    assert bdec[[0, 2, 3, 4]].tolist() == [0.0, 2**-17, 57344.0, -57344.0]


@pytest.mark.parametrize(
    ("x", "kw", "error"),
    [
        ([1.0], {"saturate": True}, TypeError),
        (np.ones(2, np.float64), {"saturate": True}, TypeError),
        (np.ones(2, np.float32), {"saturate": 1}, TypeError),
        (np.ones(2, np.float32), {"saturate": True, "nan": True}, TypeError),
        (np.ones(2, np.float32), {"saturate": True, "nan": 1.0}, TypeError),
        (np.ones(2, np.float32), {"saturate": True, "nan": 1 << 70}, ValueError),
        (np.ones(2, np.float32), {"saturate": True, "nan": 8}, ValueError),
        (np.ones(2, np.float32), {"saturate": True}, TypeError),
        (np.ones(2, np.float32), {"saturate": False, "nan": 0}, TypeError),
        (np.ones(2, np.float32), {"saturate": True, "nan": 0, "inf": 1}, ValueError),
    ],
)
def test_encode_refusals(x, kw, error):
    with pytest.raises(error):
        formats.encode(x, E2M1, **kw)


def test_encode_refusals_by_dtype():
    x = np.ones(2, np.float32)
    with pytest.raises(TypeError):
        formats.encode(x, "e4m3fn", saturate=True)
    for name, kw in [("u8", {}), ("e8m0", {}), ("f32", {"saturate": True}), ("bf16", {"nan": 0})]:
        with pytest.raises(ValueError):
            formats.encode(x, DTYPES[name], **{"saturate": False, **kw})


def test_encode_keeps_shape_and_takes_any_byte_order():
    assert formats.encode(np.array(1.5, np.float32), E4M3, saturate=True).shape == ()
    big = np.arange(6, dtype=">f4").reshape(2, 3)[:, ::2]
    got = formats.encode(big, BF16, saturate=False)
    assert got.shape == (2, 2) and got.dtype == np.uint16
    assert np.array_equal(got, formats.encode(big.astype(np.float32), BF16, saturate=False))
    copy = formats.encode(np.float32([np.nan]), DTYPES["f32"], saturate=False)
    assert copy.dtype == np.float32


def test_decode_refusals():
    with pytest.raises(TypeError):
        formats.decode(np.zeros(2, np.uint16), E4M3)
    with pytest.raises(ValueError):
        formats.decode(np.array([16], np.uint8), E2M1)
    with pytest.raises(ValueError):
        formats.decode(np.zeros(2, np.uint8), DTYPES["u8"])


def test_sixteen_bit_decodes_every_pattern():
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    assert np.array_equal(formats.decode(bits, BF16).view(np.uint32), bits.astype(np.uint32) << 16)
    half = formats.decode(bits, F16).view(np.uint32)
    with np.errstate(invalid="ignore"):
        widened = bits.view(np.float16).astype(np.float32).view(np.uint32)
    nan = np.isnan(bits.view(np.float16))
    nan_bits = bits[nan].astype(np.uint32)
    assert np.array_equal(half[~nan], widened[~nan])
    assert np.array_equal(
        half[nan], (nan_bits & 0x8000) << 16 | (nan_bits & 0x3FF) << 13 | 0x7FC00000
    )


def test_rounded():
    x = np.float32([1.0625, 500.0])
    assert formats.rounded(x, E4M3, saturate=True).tolist() == [1.0, 448.0]


def test_quantize_refusals():
    with pytest.raises(ValueError):
        formats.quantize(np.array(1.0, np.float32), E2M1)
    assert formats.quantize(np.zeros((3, 0), np.float32), E2M1)[1].shape == (3, 0)
    with pytest.raises(ValueError):
        formats.quantize(np.zeros(48, np.float32), E2M1)
    with pytest.raises(ValueError):
        formats.quantize(np.zeros(32, np.float32), DTYPES["e5m2"])
    with pytest.raises(TypeError):
        formats.quantize(np.zeros(32, np.float64), E2M1)


@pytest.mark.parametrize("dtype", [E2M1, E4M3])
def test_mx_dequantizes_to_a_nearest_value(dtype):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((64, 64)) * 10.0 ** rng.integers(-6, 6, (64, 1))).astype(np.float32)
    codes, scales = formats.quantize(x, dtype)
    scale = np.exp2(scales.astype(np.float64) - 127).repeat(32, axis=-1)
    dequantized = formats.decode(codes, dtype).astype(np.float64) * scale
    grid = formats.decode(np.arange(1 << dtype.width, dtype=np.uint8), dtype).astype(np.float64)
    grid = np.unique(grid[np.isfinite(grid)])
    q = x / scale
    nearest = np.abs(q[..., None] - grid).min(axis=-1)
    assert np.all(np.abs(q - dequantized / scale) <= nearest * (1 + 1e-12))
    assert np.all(np.abs(dequantized / scale) <= dtype.encoding.max)


def test_layout_constructors():
    import inspect

    assert (
        str(inspect.signature(formats.row_major))
        == "(ld: int = 0, pad_to: int = 0) -> thin.formats.Layout"
    )
    assert (
        formats.row_major(6) == formats.row_major(ld=6) == formats.Layout.parse("row_major(ld=6)")
    )
    assert formats.row_major(0, 0).key == "row_major" and formats.col_major().args == ()
    assert formats.row_major(pad_to=4).args == (("pad_to", 4),)
    assert hash(formats.row_major(ld=6)) == hash(formats.Layout("row_major", [("ld", 6)]))
    assert set(formats.LAYOUTS) == {"row_major", "col_major"}
    for bad in [lambda: formats.row_major(ld=True), lambda: formats.row_major(ld=1.0)]:
        with pytest.raises(TypeError):
            bad()
    with pytest.raises(TypeError):
        formats.row_major(width=1)
    with pytest.raises(formats.LayoutError):
        formats.row_major(ld=2, pad_to=2)
    with pytest.raises(formats.LayoutError):
        formats.row_major(ld=-1)
    with pytest.raises(TypeError):
        formats.Layout.parse(1)
    with pytest.raises(AttributeError):
        formats.row_major().name = "x"


@pytest.mark.parametrize(
    "key", ["row_major(ld=0)", "row_major( ld=6)", "row_major(ld=6", "rowmajor", "row_major(x=1)"]
)
def test_layout_parse_is_strict(key):
    with pytest.raises(formats.LayoutError):
        formats.Layout.parse(key)


def test_layout_refusals():
    layout, u8 = formats.row_major(), DTYPES["u8"]
    for shape in [(1, 2), [1, 2, 3], (1, 0, 3), (1, 2, True), (1, 2, 3.0)]:
        with pytest.raises(formats.LayoutError):
            formats.nbytes(layout, u8, shape)
    with pytest.raises(TypeError):
        formats.nbytes("row_major", u8, (1, 1, 1))
    with pytest.raises(TypeError):
        formats.nbytes(layout, "u8", (1, 1, 1))
    with pytest.raises(formats.LayoutError):
        formats.to_bytes(np.zeros((2, 3), np.uint8), layout, u8)
    with pytest.raises(TypeError):
        formats.to_bytes(np.zeros((1, 2, 3), np.int8), layout, u8)
    with pytest.raises(formats.LayoutError):
        formats.from_bytes(np.zeros(5, np.uint8), layout, u8, (1, 2, 3))
    with pytest.raises(formats.LayoutError):
        formats.from_bytes(np.zeros((2, 3), np.uint8), layout, u8, (1, 2, 3))
    with pytest.raises(TypeError):
        formats.from_bytes(np.zeros(6, np.int8), layout, u8, (1, 2, 3))
    with pytest.raises(formats.LayoutError):
        formats.coords(layout, u8, (1, 2, 3), [6])
    with pytest.raises(formats.LayoutError):
        formats.coords(layout, u8, (1, 2, 3), [-1])
    with pytest.raises(TypeError):
        formats.coords(layout, u8, (1, 2, 3), [0.5])
    with pytest.raises(ValueError):
        formats.to_bytes(np.full((1, 1, 2), 16, np.uint8), formats.row_major(), E2M1)


def test_coords_keep_the_offsets_shape():
    matrix, row, byte = formats.coords(formats.row_major(ld=4), DTYPES["u8"], (2, 2, 3), [[3, 4]])
    assert (matrix.tolist(), row.tolist(), byte.tolist()) == ([[-1, 0]], [[-1, 1]], [[-1, 0]])


def test_code_object_refusals():
    image = (data_path("formats", "codeobject_k.co")).read_bytes()
    with pytest.raises(TypeError):
        formats.read("image")
    with pytest.raises(formats.CodeObjectError):
        formats.read(image[:40])
    with pytest.raises(formats.CodeObjectError):
        formats.code(image[:-700])
    meta = {"amdhsa.kernels": [{".name": "k", ".symbol": "k.kd"}]}
    with pytest.raises(formats.CodeObjectError):
        formats.write(b"k", b"\0", bytes(64), meta, 0x4E)
    with pytest.raises(formats.CodeObjectError):
        formats.write("k", b"\0", bytes(64), meta, 0xFF)
    import enum

    class Flag(enum.IntEnum):
        ONE = 1

    for bad in [{1.5: 0}, (1,), b"x", Flag.ONE, 1 << 200, {"é": 1}]:
        with pytest.raises(formats.CodeObjectError):
            formats.write("k", b"\0", bytes(64), {**meta, "x": [{"y": bad}]}, 0x4E)
    for e_flags, why in ((-1, "not a u32"), (1 << 32 | 0x4E, "not a u32"), (0x4E00, "known")):
        with pytest.raises(formats.CodeObjectError, match=why):
            formats.write("k", b"\0", bytes(64), meta, e_flags)
    with pytest.raises(formats.CodeObjectError, match="an int, not float"):
        formats.write("k", b"\0", bytes(64), meta, 78.0)
    with pytest.raises(ValueError, match="0x4e"):
        formats.target_of(0xFF)
    assert formats.target_of(0x54F) == "gfx950"
    assert formats.read(formats.write("k", b"\0", bytes(64), meta, 0x4E)).symbol == "k"


# --- torch equals numpy, on the CPU and on a GPU -------------


def _devices():
    return [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif("not _has_cuda()", reason="no GPU")),
    ]


def _has_cuda() -> bool:
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available()


def _host(tensor) -> np.ndarray:
    """A tensor's bytes on the host, as unsigned integers of its element size."""
    import torch

    flat = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return flat.view(f"u{tensor.element_size()}").reshape(tuple(tensor.shape))


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("device", _devices())
def test_torch_encode_equals_numpy(torch, device):
    doc = json.loads((data_path("formats", "encode.json")).read_text())
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 1 << 32, 1 << 16, dtype=np.uint64).astype(np.uint32).view(np.float32)
    x = np.concatenate([f32(doc["inputs_f32_hex"]), noise])[: 1 << 16].reshape(256, 256)
    tensor = torch.from_numpy(x).to(device)
    for case in [
        *doc["cases"],
        {"dtype": "e4m3fnuz", "kwargs": {"saturate": False}},
        {"dtype": "e4m3fnuz", "kwargs": {"saturate": True}},
        {"dtype": "e5m2fnuz", "kwargs": {"saturate": False}},
        {"dtype": "e5m2fnuz", "kwargs": {"saturate": True}},
    ]:
        dtype = DTYPES[case["dtype"]]
        want = formats.encode(x, dtype, **case["kwargs"])
        got = formats.encode(tensor, dtype, **case["kwargs"])
        assert got.dtype == getattr(torch, dtype.torch) and got.device == tensor.device
        assert np.array_equal(_host(got), want.view(f"u{want.itemsize}")), case


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("source", ["bfloat16", "float16"])
def test_torch_encode_takes_narrow_floats(torch, device, source):
    x = torch.linspace(-500, 500, 4096).to(getattr(torch, source)).to(device)
    want = formats.encode(x.float().cpu().numpy(), E4M3, saturate=True)
    assert np.array_equal(_host(formats.encode(x, E4M3, saturate=True)), want)
    with pytest.raises(TypeError):
        formats.encode(x.double(), E4M3, saturate=True)


@pytest.mark.parametrize("device", _devices())
def test_torch_decode_equals_numpy(torch, device):
    for name in (
        "e4m3fn",
        "e5m2",
        "e4m3fnuz",
        "e5m2fnuz",
        "e2m1",
        "e2m3",
        "e3m2",
        "e8m0",
        "bf16",
        "f16",
        "f32",
    ):
        dtype = DTYPES[name]
        patterns = np.arange(min(1 << dtype.width, 1 << 16), dtype=np.uint32)
        bits = patterns.astype(dtype.numpy) if dtype.width < 32 else patterns.view(np.float32)
        want = formats.decode(bits, dtype).view(np.uint32)
        storage = torch.from_numpy(bits.view(f"i{bits.itemsize}")).to(device)
        for kind in {dtype.torch, dtype.numpy}:
            if hasattr(torch, kind) and dtype.width >= 8:
                tensor = storage.view(getattr(torch, kind))
            elif hasattr(torch, kind):
                tensor = storage.view(torch.uint8)
            else:
                continue
            got = formats.decode(tensor, dtype)
            assert got.dtype == torch.float32 and got.device == tensor.device
            assert np.array_equal(_host(got), want), (name, kind)
    with pytest.raises(ValueError):
        formats.decode(torch.full((3,), 16, dtype=torch.uint8, device=device), E2M1)
    with pytest.raises(TypeError):
        formats.decode(torch.zeros(3, dtype=torch.uint8, device=device), BF16)


@pytest.mark.parametrize("device", _devices())
def test_torch_quantize_and_layouts_equal_numpy(torch, device):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 96)).astype(np.float32)
    tensor = torch.from_numpy(x).to(device)
    codes, scales = formats.quantize(tensor, E4M3)
    want_codes, want_scales = formats.quantize(x, E4M3)
    assert codes.dtype == torch.float8_e4m3fn and codes.device == tensor.device
    assert np.array_equal(_host(codes), want_codes) and np.array_equal(_host(scales), want_scales)
    assert np.array_equal(
        _host(formats.scale_bytes(tensor[:, 0], E2M1)), formats.scale_bytes(x[:, 0], E2M1)
    )
    bits = formats.encode(x.reshape(2, 4, 48), BF16, saturate=False)
    on_device = formats.encode(tensor.reshape(2, 4, 48), BF16, saturate=False)
    for layout in (formats.row_major(pad_to=64), formats.col_major()):
        raw = formats.to_bytes(on_device, layout, BF16)
        assert raw.device == tensor.device
        assert np.array_equal(_host(raw), formats.to_bytes(bits, layout, BF16))
        back = formats.from_bytes(raw, layout, BF16, (2, 4, 48))
        assert back.dtype == torch.bfloat16 and np.array_equal(_host(back), bits)
        offsets = torch.arange(raw.numel(), device=device)
        got = formats.coords(layout, BF16, (2, 4, 48), offsets)
        want = formats.coords(layout, BF16, (2, 4, 48), np.arange(raw.numel()))
        assert all(
            g.device == offsets.device and np.array_equal(g.cpu().numpy(), w)
            for g, w in zip(got, want, strict=True)
        )


@pytest.mark.parametrize("device", _devices()[1:])
def test_device_scratch_is_a_few_pieces(torch, device):
    """above input and output, an encode's peak device
    memory stays a few pieces, whatever the input's size."""
    extra = []
    for n in (2 * formats.PIECE, 4 * formats.PIECE):
        x = torch.zeros(n, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = formats.encode(x, E4M3, saturate=True)
        torch.cuda.synchronize()
        extra.append(torch.cuda.max_memory_allocated() - base - out.numel())
        del x, out
    piece_bytes = 4 * formats.PIECE
    assert max(extra) <= 8 * piece_bytes and extra[1] <= extra[0] + piece_bytes
