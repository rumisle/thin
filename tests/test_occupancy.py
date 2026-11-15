"""Check per-dword register occupancy and overlapping live ranges."""

from __future__ import annotations

import re

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def gen_quad(BLOCK: int = 32):  # noqa: N803 - a build parameter
    """A quad loaded whole whose dwords die one by one, folded into a sum,
    with two values live across it so the low registers are taken."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def body(b, k):
        from thin.dsl import O

        V = b.v  # noqa: N806
        x, y = k.load()
        off = V.u32(k.tid * 16, name="off")
        a = V.u32(k.tid + 3, name="a")
        c = V.u32(k.tid + 5, name="c")
        q = V.u32(n=4, name="q")
        b.emit(O.global_load(q, off, x))
        b.wait(q)
        s0 = V.u32(q[0] + 1, name="s0")
        s1 = V.u32(q[1] + s0, name="s1")
        s2 = V.u32(q[2] + s1, name="s2")
        s3 = V.u32(q[3] + s2, name="s3")
        t = V.u32(s3 + a, name="t")
        t += c
        t += k.tid
        o4 = V.u32(k.tid * 4, name="o4")
        b.emit(O.global_store(o4, t, y))

    return KernelDef("quad", Args(x=ptr, y=ptr), body, wgsize=BLOCK, max_grid=(1,))


def _body(asm: str) -> list:
    out = []
    for line in asm.split("\t.section")[0].splitlines():
        code = line.split("//")[0].strip()
        if code and not code.startswith((";", ".")) and not code.endswith(":"):
            out.append(code)
    return out


def test_a_dead_dword_of_a_live_quad_takes_another_value():
    import thin

    k = thin.compile(gen_quad)
    body = _body(k.asm)
    load = next(i for i, c in enumerate(body) if c.startswith("global_load_b128"))
    lo = int(re.search(r"v\[(\d+):", body[load]).group(1))
    quad = set(range(lo, lo + 4))

    # A later write lands on one of the quad's registers while another quad
    # register is still read after it.
    def dst(c):
        m = re.match(r"v_\w+ v(\d+),", c)
        return int(m.group(1)) if m else None

    def reads(c):
        return (
            {int(r) for r in re.findall(r"\bv(\d+)\b", c.split(",", 1)[1])} if "," in c else set()
        )

    reused = [
        i for i in range(load + 1, len(body))
        if dst(body[i]) in quad and any(reads(body[j]) & (quad - {dst(body[i])})
                                        for j in range(i + 1, len(body)))
    ]  # fmt: skip
    assert reused, "\n".join(body)
    # The result equals the sum, on the emulator.
    x = np.arange(32 * 4, dtype=np.uint32)
    y = np.zeros(32, np.uint32)
    got = np.asarray(k.bind((1,), {"x": x, "y": y}).emulate().get("y", y))
    tid = np.arange(32, dtype=np.uint32)
    want = x.reshape(32, 4).sum(axis=1) + 1 + (tid + 3) + (tid + 5) + tid
    assert got.tolist() == want.tolist()


def test_the_load_counts_dwords():
    """Whole quads would count 9 where s0 is written (the quad's 4, s0, off, a,
    c, tid); per dword the dead q[0] is not counted, and the peak is 8 (at the
    load: the quad, off, a, c and tid, plus x's pointer pair is SGPRs)."""
    art = gen_quad().compile()
    m = re.search(r"v derived: (\d+) registers for a peak of (\d+)", art.summary)
    assert m and int(m.group(2)) == 8, art.summary
