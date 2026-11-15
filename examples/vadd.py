"""Instruction-level f32 vector addition with optional lap timing."""

from __future__ import annotations

from thin import KernelDef, cdiv
from thin.dsl import Args, O, ptr, u32


def build_vadd(
    BLOCK: int = 256,  # noqa: N803 - a build parameter
    n: tuple = (1, (1 << 30) - 1),
    laps: bool = False,
) -> KernelDef:
    """The vadd kernel of BLOCK lanes a workgroup, serving n's range.

    With ``laps`` each wave times its loads and its add and store:
    lap ``load`` is the time from ``start`` to x and y landed.
    """
    from thin.lib import Laps

    clock = Laps(["start", "load", "store"], BLOCK, keep="first", on=laps)

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, z, count = k.load("x", "y", "z", "n")
        lp = clock.begin(b, k)
        i = V.u32(k.wgid * BLOCK + k.tid, name="i")
        with b.lanes(i < count):
            off = V.u32(i * 4, name="off")
            lp.lap("start")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            b.emit(O.global_load(c, off, y))
            b.wait(a, c, "x, y landed")
            lp.lap("load")
            a += c
            b.emit(O.global_store(off, a, z))
            lp.lap("store")
        lp.end()

    def host(bind, x, y, z, n):
        grid = cdiv(n, BLOCK)
        return bind(grid=grid, args={"x": x, "y": y, "z": z, "n": n, **clock.sizes(grid)})

    args = Args(x=ptr, y=ptr, z=ptr, n=u32.within(*n), **clock.args())
    grid = cdiv((1 << 30) - 1, BLOCK)
    name = f"vadd_b{BLOCK}" + ("_laps" if laps else "")
    return KernelDef(
        name,
        args,
        body,
        wgsize=BLOCK,
        max_grid=(grid,),
        host=host,
        meta={"laps": clock} if laps else {},
    )
