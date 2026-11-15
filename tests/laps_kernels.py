"""A laps kernel the emulator and the GPU tests share: two rounds of two laps a wave."""

from __future__ import annotations


def laps_decl(keep):
    from thin.lib import Laps

    return Laps(["load", "store"], 64, keep=keep)


def gen_laps(keep):
    """Two rounds of two laps a wave, kept by `keep`, around a scaled store."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    decl = laps_decl(keep)

    def body(b, k):
        (out,) = k.load("out")
        lp = decl.begin(b, k)
        i = b.v.u32(k.wgid * 64 + k.tid, name="i")
        o = b.v.u32(i * 4, name="o")
        for r in range(2):
            v = b.v.u32(i * (3 + r), name=f"v{r}")
            lp.lap("load")
            b.emit(O.global_store(o, v, out))
            lp.lap("store")
            if r == 0:
                lp.record()
        lp.end()  # records the last round

    return KernelDef("laps", Args(out=ptr, **decl.args()), body, wgsize=64, max_grid=(2,))
