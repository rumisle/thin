"""Element formats, layouts and code objects.

Every bit-level algorithm is native (thin._native). This module holds what
is about arrays: which arguments are refused before the core sees them,
numpy and torch inputs, and the kind and device of results.

- numpy arrays go through Rust without a copy: the native call borrows the
  input and writes into an output array allocated here, with the GIL
  released.
- torch tensors are encoded and decoded on their device, in pieces of
  ``PIECE`` elements, by gathering from tables Rust generates; bf16 and
  f16 encodes use torch's cast with fix-ups whose patterns Rust encodes.
  The other operations (quantize, scale_bytes, layouts, coords) run once
  in Rust on the host, and the result goes back to the input's device.

The module never imports torch: a value is a tensor only if a ``torch``
module is already loaded and the value is an instance of its ``Tensor``.
"""

from __future__ import annotations

import inspect
import numbers
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from thin import _native

__all__ = [
    "DTYPES",
    "LAYOUTS",
    "PIECE",
    "Code",
    "CodeObject",
    "CodeObjectError",
    "Dtype",
    "Encoding",
    "Layout",
    "LayoutError",
    "Section",
    "code",
    "col_major",
    "coords",
    "decode",
    "encode",
    "from_bytes",
    "nbytes",
    "quantize",
    "read",
    "rounded",
    "row_major",
    "scale_bytes",
    "target_of",
    "to_bytes",
    "write",
]

PIECE = 1 << 24
"""The most elements a device encode or decode works on at once."""

_I64 = range(-(1 << 63), 1 << 63)


class LayoutError(ValueError):
    """A layout refusal."""


class CodeObjectError(ValueError):
    """A code-object refusal."""


# --- Dtypes ---------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """A float encoding's facts."""

    exp: int
    """Exponent bits."""

    man: int
    """Mantissa bits."""

    inf: bool
    """Whether it has +-inf."""

    nan: str
    """The NaN kind: ``"IEEE"``, ``"ALL_ONES"`` or ``"none"``."""

    signed: bool
    """Whether it has a sign bit."""

    subnormals: bool
    """Whether exponent field 0 holds subnormals."""

    bias: int
    """2^(exp-1) - 1."""

    max_bits: int
    """The magnitude pattern of the largest finite value."""

    max: float
    """The largest finite value."""


@dataclass(frozen=True, repr=False)
class Dtype:
    """An immutable dtype record; its repr is its name."""

    name: str
    """The identity, used in serialized forms."""

    width: int
    """Bits per element."""

    numpy: str
    """The numpy storage type's name."""

    torch: str
    """The torch dtype's name."""

    dlpack: tuple[int, int]
    """The DLPack (type code, bits)."""

    encoding: Encoding | None
    """The float encoding, or None for an integer dtype."""

    def __repr__(self) -> str:
        return self.name


DTYPES: Mapping[str, Dtype] = MappingProxyType(
    {
        name: Dtype(name, width, numpy, torch, dlpack, enc and Encoding(*enc))
        for name, width, numpy, torch, dlpack, enc in _native.dtypes()
    }
)
"""The registry: name to dtype."""


def _check_dtype(dtype) -> Dtype:
    if not isinstance(dtype, Dtype):
        raise TypeError(f"a formats Dtype, not {type(dtype).__name__}")
    return dtype


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _int(value, what: str, error: type[Exception]) -> int:
    """An int argument as the core takes it: TypeError for another type."""
    if not _is_int(value):
        raise TypeError(f"{what} must be an int, not {type(value).__name__}")
    if int(value) not in _I64:
        raise error(f"{what}={value} is out of range")
    return int(value)


# --- Arrays and tensors --------------------------------------------------


def _torch():
    """The loaded torch module, or None; never imports it."""
    return sys.modules.get("torch")


def _is_tensor(x) -> bool:
    torch = _torch()
    return torch is not None and isinstance(x, torch.Tensor)


def _storage_types(dtype: Dtype) -> set:
    """The torch dtypes a tensor in a dtype's storage may have."""
    torch = _torch()
    return {getattr(torch, n) for n in (dtype.torch, dtype.numpy) if hasattr(torch, n)}


def _host(x, dtype: Dtype, what: str) -> np.ndarray:
    """An array or tensor in a dtype's storage, as a C-contiguous numpy array."""
    if _is_tensor(x):
        if x.dtype not in _storage_types(dtype):
            raise TypeError(f"{what}: a tensor of {x.dtype} is not {dtype.name} storage")
        raw = x.detach().contiguous().cpu()
        flat = raw.reshape(-1).view(_torch().uint8).numpy()
        return flat.view(dtype.numpy).reshape(tuple(x.shape))
    if not isinstance(x, np.ndarray) or x.dtype != np.dtype(dtype.numpy):
        kind = x.dtype if isinstance(x, np.ndarray) else type(x).__name__
        raise TypeError(f"{what}: {kind} is not {dtype.name} storage ({dtype.numpy})")
    return np.ascontiguousarray(x)


def _bytes(a: np.ndarray) -> np.ndarray:
    """The bytes of a C-contiguous array, flat."""
    return a.reshape(-1).view(np.uint8)


def _like(result: np.ndarray, like, torch_dtype: str | None = None):
    """A host result as the kind of ``like``.

    For a tensor, that is a tensor on its device, viewed as ``torch_dtype``
    when given.
    """
    if not _is_tensor(like):
        return result
    torch = _torch()
    tensor = torch.from_numpy(np.ascontiguousarray(result).reshape(-1).view(np.uint8))
    tensor = tensor.view(getattr(torch, torch_dtype or result.dtype.name))
    return tensor.reshape(result.shape).to(like.device)


# --- Encode and decode ----------------------------------------------


def _options(saturate, nan, inf) -> tuple[bool, int | None, int | None]:
    if not isinstance(saturate, bool):
        raise TypeError(f"encode: saturate must be a bool, not {type(saturate).__name__}")
    patterns = [
        None if v is None else _int(v, f"encode: {k}", ValueError)
        for k, v in (("nan", nan), ("inf", inf))
    ]
    return saturate, patterns[0], patterns[1]


def encode(x, dtype, *, saturate, nan=None, inf=None):
    """The bits of f32 values in a dtype's storage, bit-exact.

    ``x`` is a numpy float32 array, or a torch tensor of float32, bfloat16
    or float16; the result has its shape and kind, on its device.
    """
    dtype = _check_dtype(dtype)
    saturate, nan, inf = _options(saturate, nan, inf)
    if _is_tensor(x):
        return _encode_tensor(x, dtype, saturate, nan, inf)
    if not isinstance(x, np.ndarray) or x.dtype.type is not np.float32:
        kind = x.dtype if isinstance(x, np.ndarray) else type(x).__name__
        raise TypeError(f"encode: a float32 array or a tensor, not {kind}")
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    out = np.empty(x.shape, dtype=dtype.numpy)
    _native.encode_into(bits, dtype.name, saturate, _bytes(out), nan, inf)
    return out


def decode(bits, dtype):
    """The exact f32 value of every pattern.

    On torch, a sub-byte dtype's range check syncs the device, so it must
    not run under graph capture.
    """
    dtype = _check_dtype(dtype)
    if _is_tensor(bits):
        return _decode_tensor(bits, dtype)
    if not isinstance(bits, np.ndarray) or bits.dtype != np.dtype(dtype.numpy):
        kind = bits.dtype if isinstance(bits, np.ndarray) else type(bits).__name__
        raise TypeError(f"decode: {kind} is not {dtype.name} storage ({dtype.numpy})")
    out = np.empty(bits.shape, dtype=np.uint32)
    _native.decode_into(_bytes(np.ascontiguousarray(bits)), dtype.name, out)
    return out.view(np.float32)


def rounded(x, dtype, **kw):
    """``decode(encode(x, dtype, **kw), dtype)``."""
    return decode(encode(x, dtype, **kw), dtype)


_TABLES: dict = {}
"""Rust's tables, per (what, dtype, options, device), as device tensors."""


def _table(key, build, device):
    table = _TABLES.get((key, device))
    if table is None:
        table = _torch().from_numpy(build()).to(device)
        _TABLES[(key, device)] = table
    return table


def _encode_table(dtype: Dtype, options, device):
    saturate, nan, inf = options

    def build():
        out = np.empty(1 << 17, dtype=np.uint8)
        _native.encode_table_into(dtype.name, saturate, out, nan, inf)
        return out

    return _table(("encode", dtype.name, options), build, device)


def _fixups(dtype: Dtype, options) -> list[int]:
    """The patterns of NaN, +inf and -inf under the options.

    They are what a native cast's NaN and infinities become.
    """
    saturate, nan, inf = options
    specials = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    out = np.empty(3, dtype=dtype.numpy)
    _native.encode_into(specials.view(np.uint32), dtype.name, saturate, _bytes(out), nan, inf)
    return out.view(np.int16).tolist()


def _table_index(bits):
    """The encode table's index of f32 bits held as int32.

    It is the top 16 bits, then a sticky bit for the low 16, as Rust's
    ``encode_table`` lays out its entries.
    """
    torch = _torch()
    return ((bits >> 15) & 0x1FFFE | ((bits & 0xFFFF) != 0).to(torch.int32)).long()


def _encode_tensor(x, dtype: Dtype, saturate: bool, nan, inf):
    torch = _torch()
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"encode: a tensor of float32, bfloat16 or float16, not {x.dtype}")
    options = (saturate, nan, inf)
    _native.check_encode(dtype.name, *options)
    flat = x.detach().reshape(-1)
    if dtype.width == 32:
        return flat.float().clone().reshape(x.shape)
    narrow = dtype.encoding.man <= 6
    storage = torch.uint8 if narrow else torch.int16
    out = torch.empty(flat.shape, dtype=storage, device=x.device)
    if narrow:
        table = _encode_table(dtype, options, x.device)
    else:
        cast = getattr(torch, dtype.torch)
        nan_bits, pinf_bits, ninf_bits = _fixups(dtype, options)
    for start in range(0, flat.numel(), PIECE):
        piece = flat[start : start + PIECE].float()
        if narrow:
            out[start : start + PIECE] = table[_table_index(piece.view(torch.int32))]
            continue
        y = piece.to(cast)
        bits = y.view(torch.int16)
        bits = torch.where(y == float("inf"), pinf_bits, bits)
        bits = torch.where(y == float("-inf"), ninf_bits, bits)
        out[start : start + PIECE] = torch.where(torch.isnan(piece), nan_bits, bits)
    return out.view(getattr(torch, dtype.torch)).reshape(x.shape)


def _decode_table(dtype: Dtype, device):
    def build():
        out = np.empty(1 << dtype.width, dtype=np.uint32)
        _native.decode_table_into(dtype.name, out)
        return out.view(np.int32)

    return _table(("decode", dtype.name), build, device)


def _decode_tensor(bits, dtype: Dtype):
    torch = _torch()
    if dtype.encoding is None:
        raise ValueError(f"decode: {dtype.name} is an integer dtype")
    if bits.dtype not in _storage_types(dtype):
        raise TypeError(f"decode: a tensor of {bits.dtype} is not {dtype.name} storage")
    flat = bits.detach().reshape(-1)
    if dtype.width == 32:
        return flat.clone().view(torch.float32).reshape(bits.shape)
    signed = torch.int8 if dtype.width <= 8 else torch.int16
    mask = (1 << (8 * flat.element_size())) - 1
    table = _decode_table(dtype, bits.device)
    out = torch.empty(flat.shape, dtype=torch.int32, device=bits.device)
    for start in range(0, flat.numel(), PIECE):
        index = flat[start : start + PIECE].view(signed).long() & mask
        if dtype.width < 8 and bool((index >= len(table)).any()):
            raise ValueError(
                f"decode: a value above {len(table) - 1} for the {dtype.width}-bit {dtype.name}"
            )
        out[start : start + PIECE] = table[index]
    return out.view(torch.float32).reshape(bits.shape)


# --- MX -------------------------------------------------------------


def _f32_host(x, what: str) -> np.ndarray:
    """f32 values of an array or a tensor, as a C-contiguous numpy array."""
    if _is_tensor(x):
        if x.dtype != _torch().float32:
            raise TypeError(f"{what}: a float32 tensor, not {x.dtype}")
        return x.detach().contiguous().cpu().numpy()
    if not isinstance(x, np.ndarray) or x.dtype.type is not np.float32:
        kind = x.dtype if isinstance(x, np.ndarray) else type(x).__name__
        raise TypeError(f"{what}: a float32 array, not {kind}")
    return np.ascontiguousarray(x, dtype=np.float32)


def scale_bytes(amax, dtype):
    """One e8m0 scale byte per group maximum."""
    dtype = _check_dtype(dtype)
    values = _f32_host(amax, "scale_bytes")
    out = np.empty(values.shape, dtype=np.uint8)
    _native.scale_bytes_into(values.view(np.uint32), dtype.name, out)
    return _like(out, amax)


def quantize(x, dtype):
    """MX codes and scales of f32 rows ``[..][K]``, K a multiple of 32.

    The codes have x's shape in the element storage; the scales are uint8
    ``[..][K/32]``. K = 0 and other empty dimensions are accepted; a 0-d x
    is refused.
    """
    dtype = _check_dtype(dtype)
    values = _f32_host(x, "quantize")
    k = values.shape[-1] if values.ndim else 0
    codes = np.empty(values.shape, dtype=np.uint8)
    scales = np.empty((*values.shape[:-1], k // 32), dtype=np.uint8)
    _native.quantize_into(values.view(np.uint32), list(values.shape), dtype.name, codes, scales)
    return _like(codes, x, dtype.torch), _like(scales, x)


# --- Layouts --------------------------------------------------------


class Layout:
    """A layout's identity: its name and non-default arguments.

    The registered constructors (``LAYOUTS``, ``row_major``, …) and
    ``Layout.parse`` make layouts; ``Layout(name, args)`` normalizes its
    arguments the same way.
    """

    __slots__ = ("args", "key", "name")

    name: str
    """The registered name."""

    args: tuple[tuple[str, int], ...]
    """The arguments that differ from their defaults, in parameter order."""

    key: str
    """``name``, or ``name(k=v,k=v)`` with no spaces."""

    def __init__(self, name: str, args=()):
        if not isinstance(name, str):
            raise TypeError(f"a layout name is a str, not {type(name).__name__}")
        pairs = [(k, _int(v, f"{name}({k}=)", LayoutError)) for k, v in args]
        self._set(*_native.layout_new(name, pairs))

    def _set(self, name: str, args, key: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "key", key)

    @classmethod
    def parse(cls, key: str) -> Layout:
        """Rebuilds a layout, refusing any key that is not exactly its own."""
        if not isinstance(key, str):
            raise TypeError(f"a layout key is a str, not {type(key).__name__}")
        layout = object.__new__(cls)
        layout._set(*_native.layout_parse(key))
        return layout

    def __setattr__(self, name, value):
        raise AttributeError("a Layout is immutable")

    def __eq__(self, other) -> bool:
        return isinstance(other, Layout) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key


def _constructor(name: str, params):
    """The constructor of a registered definition.

    Its parameters are the definition's, with their defaults, positional or
    keyword.
    """
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    signature = inspect.Signature(
        [inspect.Parameter(p, kind, default=d, annotation=int) for p, d in params],
        return_annotation=Layout,
    )

    def construct(*args, **kwargs) -> Layout:
        bound = signature.bind(*args, **kwargs)
        return Layout(name, bound.arguments.items())

    construct.__signature__ = signature
    construct.__name__ = construct.__qualname__ = name
    construct.__doc__ = f"The {name} layout ."
    return construct


LAYOUTS: Mapping = MappingProxyType(
    {name: _constructor(name, params) for name, params in _native.layout_definitions()}
)
"""The registry: each definition's name to its constructor."""

row_major = LAYOUTS["row_major"]
col_major = LAYOUTS["col_major"]


def _shape(shape) -> tuple[int, int, int]:
    """A shape as the core takes it, refusing anything but 3 ints ≥ 1."""
    if not isinstance(shape, tuple) or len(shape) != 3 or not all(map(_is_int, shape)):
        raise LayoutError(f"a shape is a 3-tuple of ints, not {shape!r}")
    if not all(1 <= int(s) < (1 << 63) for s in shape):
        raise LayoutError(f"shape {shape}: every size must be at least 1")
    return tuple(int(s) for s in shape)


def _layout_args(layout, dtype) -> tuple[Layout, Dtype]:
    if not isinstance(layout, Layout):
        raise TypeError(f"a Layout, not {type(layout).__name__}")
    return layout, _check_dtype(dtype)


def nbytes(layout, dtype, shape) -> int:
    """The bytes a layout occupies for a dtype and shape."""
    layout, dtype = _layout_args(layout, dtype)
    return _native.layout_nbytes(layout.key, dtype.name, _shape(shape))


def to_bytes(bits, layout, dtype):
    """Flat uint8 bytes of bits ``[stack][rows][cols]`` in a layout.

    The result is of the input's kind and on its device.
    """
    layout, dtype = _layout_args(layout, dtype)
    host = _host(bits, dtype, "to_bytes")
    if host.ndim != 3:
        raise LayoutError(f"to_bytes: bits are [stack][rows][cols], not {host.ndim}-D")
    out = _native.to_bytes(_bytes(host), layout.key, dtype.name, _shape(host.shape))
    return _like(out, bits)


def from_bytes(raw, layout, dtype, shape):
    """Bits back from a layout's bytes.

    Padding is ignored, and the copies of a repeated dimension must agree.
    """
    layout, dtype = _layout_args(layout, dtype)
    shape = _shape(shape)
    host = _host(raw, DTYPES["u8"], "from_bytes")
    if host.ndim != 1:
        raise LayoutError(f"from_bytes: raw bytes are flat, not {host.ndim}-D")
    out = np.empty(shape, dtype=dtype.numpy)
    _native.from_bytes_into(host, layout.key, dtype.name, shape, _bytes(out))
    return _like(out, raw, dtype.torch)


def coords(layout, dtype, shape, offsets):
    """The (matrix, row, logical byte) of each offset.

    They come as three int64 arrays, -1 where the offset is padding.
    """
    layout, dtype = _layout_args(layout, dtype)
    shape = _shape(shape)
    host = offsets.detach().cpu().numpy() if _is_tensor(offsets) else np.asarray(offsets)
    if host.dtype.kind not in "iu":
        raise TypeError(f"coords: offsets are ints, not {host.dtype}")
    if host.size and host.min() < 0:
        raise LayoutError(f"coords: offset {host.min()} is negative")
    out = np.empty((host.size, 3), dtype=np.int64)
    flat = np.ascontiguousarray(host.reshape(-1), dtype=np.uint64)
    _native.coords_into(layout.key, dtype.name, shape, flat, out)
    columns = np.ascontiguousarray(out.T).reshape((3, *host.shape))
    return tuple(_like(c, offsets) for c in columns)


# --- Code objects -----------------------------------------------


@dataclass(frozen=True)
class Section:
    """A section header of a code object."""

    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int


@dataclass(frozen=True)
class CodeObject:
    """A code object's parts."""

    ident: bytes
    """The OS ABI and its version (image bytes 7 and 8)."""

    e_flags: int
    sections: tuple[Section, ...]
    """Every section in header order, the null one included."""

    symbol: str
    """The kernel's symbol: the ``.kd`` symbol without ``.kd``."""

    descriptor: bytes
    """The 64-byte kernel descriptor."""

    metadata: dict
    """The decoded AMDGPU metadata note."""

    text: bytes
    lds: int
    """The descriptor's LDS bytes (u32 at 0)."""

    private: int
    """The descriptor's private bytes (u32 at 4)."""

    kernarg: int
    """The descriptor's kernarg bytes (u32 at 8)."""

    @property
    def target(self) -> str:
        """The target its e_flags select."""
        return target_of(self.e_flags)


@dataclass(frozen=True)
class Code:
    """The normalized "what runs" of a code object."""

    ident: bytes
    e_flags: int
    text: bytes
    descriptor: bytes
    """The descriptor with its entry offset (bytes 16–23) zeroed."""

    metadata: str
    """The metadata without the kernel's ``.name`` and ``.symbol``, as canonical JSON."""

    digests: Mapping[str, str]
    """sha256 hex of ident, e_flags, text, descriptor and metadata, by those names."""


def _image(img, what: str) -> bytes:
    if not isinstance(img, (bytes, bytearray, memoryview)):
        raise TypeError(f"{what}: an image is bytes, not {type(img).__name__}")
    return bytes(img)


def read(img) -> CodeObject:
    """Read and validate a code object's container, descriptor and metadata."""
    ident, e_flags, sections, symbol, descriptor, metadata, text, sizes = _native.read_code_object(
        _image(img, "read")
    )
    parts = (ident, e_flags, tuple(Section(*s) for s in sections), symbol, descriptor)
    return CodeObject(*parts, metadata, text, *sizes)


def code(img) -> Code:
    """The normalized code of a code object and its digests."""
    *parts, digests = _native.code(_image(img, "code"))
    return Code(*parts, MappingProxyType(dict(digests)))


def write(symbol, text, descriptor, metadata, e_flags) -> bytes:
    """A code object HIP loads, with no linker."""
    if not isinstance(symbol, str):
        raise CodeObjectError(f"the symbol is a str, not {type(symbol).__name__}")
    if not _is_int(e_flags):
        raise CodeObjectError(f"e_flags is an int, not {type(e_flags).__name__}")
    if not 0 <= e_flags < (1 << 32):
        raise CodeObjectError(f"e_flags {e_flags} is not a u32")
    text, descriptor = _image(text, "write"), _image(descriptor, "write")
    return _native.write_code_object(symbol, text, descriptor, metadata, int(e_flags))


def target_of(e_flags) -> str:
    """The target an ELF's e_flags select."""
    if not _is_int(e_flags):
        raise TypeError(f"e_flags is an int, not {type(e_flags).__name__}")
    if not 0 <= e_flags < (1 << 32):
        raise ValueError(f"e_flags {e_flags} is not a u32")
    return _native.target_of(int(e_flags))
