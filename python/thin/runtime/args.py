"""Argument checking and packing by the manifest's records."""

from __future__ import annotations

import operator
import struct

from thin.runtime.arith import magic
from thin.runtime.errors import LaunchError

__all__ = ["RANGES", "check", "pack"]

RANGES = {
    "u32": (0, (1 << 32) - 1),
    "i32": (-(1 << 31), (1 << 31) - 1),
    "u64": (0, (1 << 64) - 1),
}
"""Each integer kind's own range."""


def _int_like(v) -> bool:
    if isinstance(v, bool):
        return False
    try:
        operator.index(v)
    except TypeError:
        return False
    return True


def _cut(v, n: int = 60) -> str:
    text = repr(v)
    return text if len(text) <= n else text[:n]


def _article(kind: str) -> str:
    """The article of a kind as it is read aloud: an i32, an f32, a u32."""
    return "an" if kind[:1] in "aefhilmnorsx" else "a"


def _own(k, r, v) -> None:
    name, owned = r["name"], r["owned"]
    if owned == "kept" and isinstance(v, (bytes, bytearray)) and len(v):
        return
    if not (type(v) is int and v >= 1):
        tail = ", or its first bytes" if owned == "kept" else ""
        raise LaunchError(
            f"{k}: argument {name} is the kernel's own {owned} memory: give its size, an int of "
            f"bytes (1 or more){tail} - not {_cut(v)}"
        )
    if owned == "gens" and v % 4:
        raise LaunchError(
            f"{k}: argument {name} is generation words: {v} bytes, not whole u32 words"
        )


def _integer(k, r, v) -> None:
    name, kind = r["name"], r["kind"]
    if not _int_like(v):
        raise LaunchError(
            f"{k}: argument {name} is {_article(kind)} {kind}: an int, not {type(v).__name__}"
        )
    v = operator.index(v)
    lo0, hi0 = RANGES[kind]
    lo = lo0 if r.get("lo") is None else r["lo"]
    hi = hi0 if r.get("hi") is None else r["hi"]
    if not lo <= v <= hi:
        raise LaunchError(f"{k}: argument {name} = {v}: outside its range {lo} .. {hi}")
    m = r.get("multiple") or 1
    if v % m:
        raise LaunchError(f"{k}: argument {name} = {v}: not a multiple of {m}")


def _f32(k, r, v) -> None:
    ok = not isinstance(v, bool) and (isinstance(v, (int, float)) or hasattr(type(v), "__float__"))
    if not ok:
        raise LaunchError(f"{k}: argument {r['name']} is an f32: a number, not {type(v).__name__}")
    try:
        struct.pack("<f", float(v))
    except OverflowError:
        raise LaunchError(f"{k}: argument {r['name']} = {v!r}: past f32's range") from None


def check(symbol, records, values, is_pointer) -> None:
    """Checks the values given against the records."""
    names = [r["name"] for r in records]
    if set(values) != set(names):
        missing = sorted(set(names) - set(values))
        unknown = sorted(set(values) - set(names))
        raise LaunchError(
            f"{symbol}: arguments {sorted(names)}, given {sorted(values)} "
            f"(missing {missing}, unknown {unknown})"
        )
    for r in records:
        v, kind = values[r["name"]], r["kind"]
        if kind == "ptr" and r.get("owned"):
            _own(symbol, r, v)
        elif kind == "ptr":
            if not is_pointer(v):
                raise LaunchError(
                    f"{symbol}: argument {r['name']} is a pointer: a buffer, not {type(v).__name__}"
                )
        elif kind in RANGES:
            _integer(symbol, r, v)
        elif kind == "f32":
            _f32(symbol, r, v)
        else:
            raise LaunchError(
                f"{symbol}: argument {r['name']} of kind {kind!r}: not one a kernel is given by "
                "name"
            )


def pack(records, nbytes, values, address) -> bytes:
    """The kernarg block: each record at its offset, little-endian."""
    out = bytearray(nbytes)
    for r in records:
        name, kind, at = r["name"], r["kind"], r["offset"]
        if kind == "ptr" and r.get("ranks") is not None:
            addrs = address(name)
            want = r["bytes"] // 8
            if len(addrs) != want:
                raise LaunchError(f"argument {name}: {want} ranks' addresses, given {len(addrs)}")
            struct.pack_into(f"<{want}Q", out, at, *addrs)
        elif kind == "ptr":
            struct.pack_into("<Q", out, at, address(name))
        elif kind == "f32":
            struct.pack_into("<f", out, at, float(values[name]))
        elif r.get("divisor"):
            d = operator.index(values[name])
            struct.pack_into("<3I", out, at, d, *magic(d))
        elif kind == "i32":
            struct.pack_into("<i", out, at, operator.index(values[name]))
        elif kind == "u64":
            struct.pack_into("<Q", out, at, operator.index(values[name]))
        else:
            struct.pack_into("<I", out, at, operator.index(values[name]))
    return bytes(out)
