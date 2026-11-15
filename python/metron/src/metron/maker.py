"""The maker of the own kernels.

One of the two modules that import the compiler.

It builds the copy compare, the instruction-cache invalidate and the empty
kernel with the compiler for every target it knows (the invalidate from the
target's hook), and writes ``<arch>/<name>.co`` and
``<arch>/<name>.json`` under ``metron/kernels``. ``python -m metron.maker``
rewrites them; a test rebuilds them and compares with the kept bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

# metron[thin]: without thin installed, importing this module raises ImportError.
import thin
from thin.dsl import Args, O, mul_wide, ptr, u32
from thin.dsl import min as dmin

__all__ = ["build", "gen_differ", "gen_empty", "gen_icache_inv", "make"]

CHUNK = 32768
"""The bytes one workgroup of the compare covers."""
STRIDE_MAX = 2**32 - 2**16
"""The largest stride: a copy's offsets fit in 32 bits."""
COPIES_MAX = 65536
"""The most copies one launch compares."""


def gen_differ():
    """The copy compare: flags[y] = 1 where copy y's stride bytes differ from ref's."""

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        base, ref, flags, stride = k.load()
        copy = S.ptr(base + mul_wide(k.wgid_y, stride), name="copy")
        last = S.u32(stride - 16, name="last")
        offs = [
            V.u32(dmin(k.wgid * CHUNK + k.tid * 16 + 4096 * j, last), name=f"off{j}")
            for j in range(8)
        ]
        pairs = []
        for j, off in enumerate(offs):
            x, y = V.u32(n=4, name=f"x{j}"), V.u32(n=4, name=f"y{j}")
            b.emit(O.global_load(x, off, copy))
            b.emit(O.global_load(y, off, ref))
            pairs.append((x, y))
        acc = V.u32(0, name="acc")
        for j, (x, y) in enumerate(pairs):
            b.wait(x, y, comment=f"piece {j} of both")
            for d in range(4):
                acc[...] = acc | (x[d] ^ y[d])
        with b.lanes(acc != 0):
            at = V.u32(k.wgid_y * 4, name="at")
            b.emit(O.global_store(at, V.u32(1, name="one"), flags))

    stride_kind = u32.within(16, STRIDE_MAX).multiple_of(16)
    args = Args(base=ptr, ref=ptr, flags=ptr, stride=stride_kind)
    meta = {"chunk": CHUNK, "stride_max": STRIDE_MAX, "copies_max": COPIES_MAX}
    grid = (-(-STRIDE_MAX // CHUNK), COPIES_MAX)
    return thin.KernelDef("metron_differ", args, body, wgsize=256, max_grid=grid, wg=2, meta=meta)


def gen_icache_inv():
    """The instruction-cache invalidate: the target's invalidate, then its no-ops."""
    from thin.target import target

    instruction, nop, nops, per_cu = target().icache_inv

    def body(b, k):
        b.emit(O.op(instruction))
        for _ in range(nops):
            b.emit(O.op(nop, 0))

    args = Args(unused=u32)
    meta = {"per_cu": per_cu}
    return thin.KernelDef("metron_icache_inv", args, body, wgsize=64, meta=meta)


def gen_empty():
    """The empty kernel: a launch's floor."""

    def body(b, k):
        pass

    return thin.KernelDef("metron_empty", Args(unused=u32), body, wgsize=64, max_grid=(1,))


GENERATORS = {"differ": gen_differ, "icache_inv": gen_icache_inv, "empty": gen_empty}


def _description(k) -> dict:
    return {
        "symbol": k.name,
        "block": [k.manifest.wgsize, 1],
        "kernarg_bytes": k.manifest.kernarg_bytes,
        "args": [
            {"name": a["name"], "kind": a["kind"], "offset": a["offset"]} for a in k.manifest.args
        ],
        "meta": dict(k.definition.meta),
    }


def build(target_name: str) -> dict[str, tuple[bytes, str]]:
    """Name -> (code object, description JSON text) of every own kernel for a target."""
    from thin.target import targeting

    out = {}
    before = thin.use_store(None)
    try:
        with targeting(target_name):
            for name, gen in GENERATORS.items():
                k = thin.compile(gen)
                text = json.dumps(_description(k), indent=1, sort_keys=True) + "\n"
                out[name] = (k.co, text)
    finally:
        thin.use_store(before)
    return out


def make(root: Path | None = None) -> list[Path]:
    """Writes every target's own kernels under root (the kept directory); the paths written."""
    from metron import _kernels
    from thin.target import target, target_names

    root = Path(root or _kernels.DIR)
    written = []
    for name in target_names():
        at = root / target(name).name
        at.mkdir(parents=True, exist_ok=True)
        for kernel, (co, text) in build(name).items():
            (at / f"{kernel}.co").write_bytes(co)
            (at / f"{kernel}.json").write_text(text)
            written += [at / f"{kernel}.co", at / f"{kernel}.json"]
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in make():
        print(p)
