"""The bit-exact numbers: native functions by their names."""

from __future__ import annotations

import struct

import numpy as np

from thin import _native

__all__ = [
    "ALL_LANES",
    "CVT_NAN",
    "LANES",
    "LANE_IDX",
    "QNAN",
    "bf16_canon",
    "bf16_round",
    "code_values",
    "cvt_down",
    "cvt_f32",
    "cvt_nan_bits",
    "cvt_scale_e",
    "cvt_up",
    "f16_canon",
    "f16_of_f32",
    "f16_rtz",
    "f16_val",
    "f32",
    "fbits",
    "fcanon",
    "lanebits",
    "popcount64",
    "u32",
]

QNAN = 0x7FC00000
"""The canonical f32 NaN."""
CVT_NAN = {"f32": 0xFFC00000, "bf16": 0xFFC0, "f16": 0xFE00}
"""The scaled forms' negative quiet NaN per destination."""
LANES = _native.target_info(_native.target_names()[0])[0]
"""W of the default target."""
LANE_IDX = np.arange(LANES, dtype=np.uint32)
"""Lane numbers."""
ALL_LANES = (1 << LANES) - 1
"""Every lane's bit."""


def f32(bits: int) -> float:
    """The f32 value of bits."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def u32(x: float) -> int:
    """The bits of an f32 value."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def fcanon(bits: int) -> int:
    """canon32 of bits."""
    return _native.emu_fcanon(bits)


def fbits(x: float) -> int:
    """canon32 of an f32 value's bits."""
    return fcanon(u32(x))


def bf16_round(bits: int) -> int:
    """bf16_rne of f32 bits."""
    return _native.emu_bf16_round(bits)


def bf16_canon(bits: int) -> int:
    """bf16_conv of f32 bits."""
    return _native.emu_bf16_canon(bits)


def f16_of_f32(x, bits: bool = False) -> int:
    """f32 (a value, or bits when ``bits``) to f16 bits, RNE."""
    return _native.emu_f16_of_f32(x if bits else u32(x))


def f16_canon(x: float) -> int:
    """One RNE rounding of an exact value to f16 bits."""
    return _native.emu_f16_one(x)


def f16_rtz(bits: int) -> int:
    """f32 bits to f16 bits toward zero."""
    return _native.emu_f16_rtz(bits)


def f16_val(h: int) -> float:
    """A binary16 pattern's value."""
    return _native.emu_f16_val(h)


def code_values(codes, fmt: str) -> list[float]:
    """The values of codes of a small format."""
    table = _native.emu_code_values(fmt)
    return [table[c] for c in codes]


def cvt_scale_e(scale: int) -> int:
    """The scale exponent E of a scale operand."""
    return _native.emu_cvt_scale_e(scale)


def cvt_down(x: float, s: int, fmt: str) -> int:
    """down(x, scale, fmt)."""
    return _native.emu_cvt_down(x, s, fmt)


def cvt_up(code: int, s: int, fmt: str) -> float:
    """up(code, scale, fmt) as a real value."""
    return _native.emu_cvt_up(code, s, fmt)[0]


def cvt_f32(code: int, s: int, fmt: str) -> int:
    """up(code, scale, fmt) as f32 bits."""
    return _native.emu_cvt_up(code, s, fmt)[1]


def cvt_nan_bits(bits: int, v: float, fmt: str) -> int:
    """The destination NaN of ``fmt`` when v is a NaN, else bits."""
    return CVT_NAN[fmt] if v != v else bits


def popcount64(x: int) -> int:
    """Set bits of a 64-bit value."""
    return int(x & 0xFFFFFFFFFFFFFFFF).bit_count()


def lanebits(mask: int) -> np.ndarray:
    """A W-lane boolean view of a lane mask."""
    return ((int(mask) >> LANE_IDX.astype(np.uint64)) & 1).astype(bool)
