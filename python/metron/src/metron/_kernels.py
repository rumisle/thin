"""The harness's own kernels, kept per target.

Each is a code object and a JSON description in ``kernels/<arch>/``: the
copy compare (``differ``), the instruction-cache invalidate (``icache_inv``)
and the empty kernel (``empty``). They are built by ``metron.maker`` and
kept, so the harness needs no compiler.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DIR", "NAMES", "kept", "pack"]

#: Where the kernels are kept.
DIR = Path(__file__).parent / "kernels"
#: The own kernels.
NAMES = ("differ", "icache_inv", "empty")


def kept(arch: str, name: str) -> tuple[dict, bytes]:
    """(description, code object) of an own kernel for an arch.

    A missing directory is a FileNotFoundError (the race fails when it starts).
    """
    at = DIR / arch
    desc = json.loads((at / f"{name}.json").read_text())
    return desc, (at / f"{name}.co").read_bytes()


def pack(desc: dict, args: dict) -> bytes:
    """The argument block of a kept kernel's description.

    kernarg_bytes zeros, each argument little-endian at its offset (ptr 8 bytes, u32 4).
    """
    out = bytearray(desc["kernarg_bytes"])
    for a in desc["args"]:
        width = 8 if a["kind"] == "ptr" else 4
        out[a["offset"] : a["offset"] + width] = int(args[a["name"]]).to_bytes(width, "little")
    return bytes(out)
