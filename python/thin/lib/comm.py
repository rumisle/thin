"""System-scope communication payloads and polling.

A completed poll guarantees that every read carries this launch's payload.
Polls are unbounded; the group launch contract must guarantee progress.
"""

from __future__ import annotations

from thin.dsl import O, popcount
from thin.lib import _hooks

__all__ = ["ll_all", "payload", "poll", "put", "tagged", "word"]


def word(b):
    """A word's four fresh u32 VGPRs."""
    return b.v.u32(n=4, name="wd")


def tagged(w, t):
    """Puts the tag in dwords 2 and 3; returns the word."""
    w[2] = t
    w[3] = t
    return w


def payload(w):
    """The word's 8-byte payload: a view of its own registers."""
    return w[0:2]


def put(b, w, recv, off, offset=0) -> None:
    """Stores the word into every rank's receive buffer at `off` (+ `offset`)."""
    for q, base in enumerate(recv):
        store = O.global_store(
            base + off, w, offset=offset, **_hooks.system(), comment=f"-> rank {q}"
        )
        b.emit(store)


def poll(b, reads, t, name="poll", cache=None) -> None:
    """Waits until every read's word carries the tag `t`."""
    for lp in b.loop(name=name):
        for w, address, lanes in reads:
            with b.lanes(*([] if lanes is None else [lanes])):
                b.emit(O.global_load(w, address, **(_hooks.system() if cache is None else cache)))
        b.wait(*[w for w, _, _ in reads], comment="the round")
        stale = None
        for w, _, lanes in reads:
            with b.lanes(*([] if lanes is None else [lanes])):
                mine = b.s.mask(w[3] != t, name="stale")
            if stale is None:
                stale = mine
            else:
                stale[...] = stale | mine
        lp.break_when(popcount(stale) == 0, comment="every word this launch's")


def ll_all(b, recv, rank, slot, g1, t, c, words, covers) -> dict:
    """Exchanges this lane's words with every rank: `{(q, j): word}`."""
    world = len(recv)
    # Each word takes 16 bytes of the rank's slot, from 16 c on: what is
    # known while building is checked (the buffer's size is the caller's).
    first = c if isinstance(c, int) else 0
    if 16 * (first + len(words)) > slot:
        raise ValueError(
            f"ll_all: chunk {first} and {len(words)} more of 16 bytes end at byte "
            f"{16 * (first + len(words))}, past the {slot}-byte slot"
        )
    pws = b.s.u32((g1 & 1) * (world * slot), name="pws")
    b.wait(*words, comment="the payloads")
    vw = b.v.u32(c * 16 + rank * slot + pws, name="vw")
    for j, (w, cover) in enumerate(zip(words, covers, strict=True)):
        with b.lanes(*([] if cover is None else [cover])):
            put(b, tagged(w, t), recv, vw, offset=16 * j)
    vp = b.v.u32(c * 16 + pws, name="vp")
    got, reads = {}, []
    for q in range(world):
        vq = vp if q == 0 else b.v.u32(vp + q * slot, name="vq")
        for j, cover in enumerate(covers):
            w = b.v.u32(n=4, name="w")
            address = recv.own + vq if j == 0 else recv.own + vq + 16 * j
            got[q, j] = w
            reads.append((w, address, cover))
    poll(b, reads, t)
    return got
