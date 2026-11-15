"""The memory of a race.

The constants, the sizing, the copies of every tensor and the pools of the
arms' owned memory.
"""

from __future__ import annotations

import math
import os

__all__ = [
    "ALIGN",
    "COLD",
    "GAP",
    "GENS_START",
    "GRAPH_CALLS",
    "GUARD",
    "HYGIENE",
    "MIN_ROUNDS",
    "OWNED_ALIGN",
    "PLAIN_MOST",
    "POISON",
    "PROBE",
    "REPLAY_US",
    "SWEEP_US",
    "calls",
    "copies",
    "fitted",
    "graph_calls",
    "graph_replays_ok",
    "most",
]

#: The rotated copies together span at least this.
COLD = 2 << 30
#: Bytes after each copy and each owned pool.
GAP = 4096
#: Each copy starts at a multiple of this.
ALIGN = 16
#: The graph calls alive by stack: launches in all of a race's graphs together.
GRAPH_CALLS = {
    "ROCm 7.x": 4096,
    "other stacks": 4096,
}
#: Stacks whose graph replays crash: HIP major version -> the GPU archs.
REPLAY_FAULTS = {7: ("gfx1201",)}
#: `most` on the plain path: nothing is captured, so no budget.
PLAIN_MOST = 65536
#: A replay's length, roughly, in us.
REPLAY_US = 10000
#: A sweep (every copy once) lasts at most this, in us.
SWEEP_US = 200000
#: The calls a sweep's duration is measured on.
PROBE = 64
#: The smallest n whose extremes bound the median at 95 %.
MIN_ROUNDS = 6
#: The fill of an output's bytes and of an input's gap.
POISON = 0xFF
#: The gap after an output and after a pool.
GUARD = 0x5B
#: An input this large holding one value is flagged.
HYGIENE = 64 << 10
#: Each owned buffer's place is rounded up to this.
OWNED_ALIGN = 256
#: A gens word's first value.
GENS_START = 2**32 - 3


_ASK = object()
"""Read it: the environment variable, or the HIP runtime's version."""


def graph_calls(env=_ASK, hip_version=_ASK) -> int:
    """The graph calls alive a race may hold.

    METRON_GRAPH_CALLS when set, else the stack's (by HIP's major version).
    """
    if env is _ASK:
        env = os.environ.get("METRON_GRAPH_CALLS")
    if env is not None:
        if not (env.isascii() and env.isdigit() and int(env) >= 1):
            raise ValueError(
                f"METRON_GRAPH_CALLS={env!r}: a whole number of graph calls, at least 1"
            )
        return int(env)
    if hip_version is _ASK:
        from metron import _device

        hip_version = _device.hip_version()
    if hip_version // 10**7 == 7:
        return GRAPH_CALLS["ROCm 7.x"]
    return GRAPH_CALLS["other stacks"]


def graph_replays_ok(env=_ASK, hip_version=_ASK, archs=_ASK) -> bool:
    """Whether the stack replays graphs without limit.

    METRON_GRAPHS when set ("1" or "0"), else false for HIP major 7 with any
    gfx1201 rank: those stacks crash after about 2^14 replayed launches.
    """
    if env is _ASK:
        env = os.environ.get("METRON_GRAPHS")
    if env is not None:
        if env not in ("0", "1"):
            raise ValueError(f"METRON_GRAPHS={env!r}: 1 (graphs) or 0 (plain launches)")
        return env == "1"
    if hip_version is _ASK or archs is _ASK:
        from metron import _device

        be = _device.backend()
        if hip_version is _ASK:
            hip_version = be.hip_version()
        if archs is _ASK:
            archs = [be.arch]
    faulty = REPLAY_FAULTS.get(hip_version // 10**7, ())
    return not any(a in faulty for a in archs)


def most(budget: int, ranks: int, arms: int, cold: bool, key: str = "<key>") -> int:
    """The most calls one graph may hold, the race's graphs within the budget.

    ValueError when fewer than 2 (a sweep needs two copies).
    """
    w = 2 if cold else 1
    per = ranks * (arms * w + 1 + (2 if cold else 0))
    m = budget // per
    if m < 2:
        cold_text = ", cold code" if cold else ""
        raise ValueError(
            f"race at {key}: {arms} arms \u00d7 {ranks} ranks{cold_text} need {2 * per} graph "
            f"calls alive at the least; this stack holds {budget} (set METRON_GRAPH_CALLS to "
            "change it)"
        )
    return m


def copies(rotated: int, most: int, cold: int = COLD) -> int:
    """The copies that make a sweep span COLD bytes."""
    return min(most, max(2, -(-cold // max(1, rotated))))


def fitted(n: int, call_us: float) -> int:
    """The copies a sweep of SWEEP_US holds."""
    return max(2, min(n, int(SWEEP_US // max(call_us, 0.001))))


def calls(n: int, sweep_us: float, most: int) -> int:
    """Whole sweeps lasting about REPLAY_US."""
    sweeps = math.ceil(REPLAY_US / max(sweep_us, 0.001))
    return n * max(1, min(most // n, sweeps))


#: The copy compare's largest stride: a copy's offsets fit 32 bits.
STRIDE_MAX = 2**32 - 2**16


def _bytes_of(t):
    """A tensor's bytes as a flat u8 tensor (contiguous)."""
    import torch

    flat = t.contiguous().reshape(-1)
    return flat.view(torch.uint8) if flat.numel() else torch.empty(0, dtype=torch.uint8)


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def stride_of(name: str, nbytes: int) -> int:
    """A copy's stride: its bytes rounded up to ALIGN, then GAP."""
    stride = -(-nbytes // ALIGN) * ALIGN + GAP
    if stride > STRIDE_MAX:
        raise ValueError(f"{name}: {nbytes} bytes: a copy of at most {STRIDE_MAX - GAP}")
    return stride


class Copies:
    """A tensor's n copies in one allocation, each followed by its gap."""

    def __init__(self, be, rank, name, tensor, io, n):
        """Lays out n copies.

        An input's row is its bytes then 0xFF; an output's is 0xFF for its bytes then 0x5B.
        """
        import torch

        self.name, self.io, self.n = name, io, n
        self.dtype, self.shape = tensor.dtype, tuple(tensor.shape)
        self.nbytes = _nbytes(tensor)
        self.stride = stride_of(name, self.nbytes)
        where = tensor.device if io == "in" and tensor.device.type != "meta" else "cpu"
        row = torch.full((self.stride,), GUARD if io == "out" else POISON, dtype=torch.uint8)
        row = row.to(where)
        if io == "in":
            row[: self.nbytes] = _bytes_of(tensor)
        else:
            row[: self.nbytes] = POISON
        self.original = row.to(be.device(rank))
        self.buf = be.alloc(rank, n * self.stride)

    def restore(self) -> None:
        """Copy 0 from the input, then doubling until all n are written."""
        s = self.stride
        self.buf[:s].copy_(self.original)
        have = 1
        while have < self.n:
            k = min(have, self.n - have)
            self.buf[have * s : (have + k) * s].copy_(self.buf[: k * s])
            have += k

    def view(self, i):
        """Copy i mod n as a tensor of the tensor's dtype and shape."""
        at = (i % self.n) * self.stride
        u8 = self.buf[at : at + self.nbytes]
        return u8.view(self.dtype).view(self.shape) if self.nbytes else u8

    def row(self, i):
        """Copy i's stride bytes (the copy and its gap) as a u8 view."""
        at = (i % self.n) * self.stride
        return self.buf[at : at + self.stride]


def hygiene(ranks_inputs) -> list[str]:
    """The hygiene flags of a point's inputs.

    `constant:<name>` for every input of HYGIENE bytes or more whose four slices
    all equal its first element, in rank then input order.
    """
    import torch

    if isinstance(ranks_inputs, dict) or hasattr(ranks_inputs, "items"):
        ranks_inputs = [ranks_inputs]
    flags = []
    several = len(ranks_inputs) > 1
    ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    for r, inputs in enumerate(ranks_inputs):
        for name, t in inputs.items():
            if _nbytes(t) < HYGIENE:
                continue
            flat = t.reshape(-1).contiguous().view(ints[t.element_size()])
            rows, k = flat.numel(), HYGIENE // 4 // t.element_size()
            first = flat[0]
            starts = (0, rows // 3, 2 * rows // 3, rows - k)
            if all(bool((flat[s : s + k] == first).all()) for s in starts):
                flags.append(f"constant:{name}" + (f" rank {r}" if several else ""))
    return flags


def _round(n: int) -> int:
    return -(-n // OWNED_ALIGN) * OWNED_ALIGN


def owned_layout(arms) -> tuple[list, dict]:
    """The owned memory's places and each kind's pool size.

    Places are [(arm, name, kind, offset, nbytes)]: the sized ones first, then
    the kept buffers given first contents.
    """
    from metron._nouns import KINDS

    sized, firsts, sizes = [], [], {}
    for kind in KINDS:
        end = 0
        for arm in arms:
            at = 0
            for name, (size, k) in arm.owned.items():
                if k == kind and not isinstance(size, bytes):
                    sized.append((arm.name, name, kind, at, size))
                    at += _round(size)
            end = max(end, at)
        for arm in arms:
            for name, (size, k) in arm.owned.items():
                if k == kind and isinstance(size, bytes):
                    firsts.append((arm.name, name, kind, end, len(size)))
                    end += _round(len(size))
        sizes[kind] = end
    order = {(a.name, n): i for i, a in enumerate(arms) for n in a.owned}
    sized.sort(key=lambda p: order[p[0], p[1]])
    firsts.sort(key=lambda p: order[p[0], p[1]])
    return sized + firsts, sizes


class Pools:
    """One pool per owned kind, shared by every arm of the race."""

    def __init__(self, be, rank, arms):
        """Lays the pools out and allocates them, then a fresh()."""
        from metron._nouns import UNCACHED

        self.places, sizes = owned_layout(arms)
        self.at = {(arm, name): (kind, off, n) for arm, name, kind, off, n in self.places}
        self.pools = {}
        for kind, size in sizes.items():
            if kind in UNCACHED and not size:
                continue
            buf = (be.uncached if kind in UNCACHED else be.alloc)(rank, size + GAP)
            buf[:size].fill_(POISON if kind == "scratch" else 0)
            buf[size:].fill_(GUARD)
            self.pools[kind] = (buf, size)
        for arm in arms:
            for name, (contents, kind) in arm.owned.items():
                if isinstance(contents, bytes):
                    import torch

                    _, off, n = self.at[arm.name, name]
                    data = torch.frombuffer(bytearray(contents), dtype=torch.uint8)
                    self.pools[kind][0][off : off + n].copy_(data)
        self.first = {
            k: buf[:size].clone() for k, (buf, size) in self.pools.items() if k.startswith("kept")
        }
        self.fresh()

    def view(self, arm, name):
        """The arm's buffer as a u8 view."""
        kind, off, n = self.at[arm, name]
        return self.pools[kind][0][off : off + n]

    def fresh(self, kept=False) -> None:
        """Resets the pools before an arm's calls.

        Carried and mapped are zeroed, gens set to GENS_START; with kept, kept goes
        back to its first contents. Scratch keeps what the last arm left.
        """
        import torch

        for kind, (buf, size) in self.pools.items():
            base = kind.split(".")[0]
            if base in ("carried", "mapped"):
                buf[:size].zero_()
            elif base == "gens":
                words = buf[: size - size % 4].view(torch.int32)
                words.fill_(GENS_START - 2**32)
            elif base == "kept" and kept:
                buf[:size].copy_(self.first[kind])

    def kept_changed(self, arm=None) -> list | None:
        """The arm's kept buffers that differ from their first contents.

        ["(outside its buffers)"] when the change is elsewhere; None when every
        kept pool is as found.
        """
        import torch

        changed = [
            k
            for k, first in self.first.items()
            if not torch.equal(self.pools[k][0][: first.numel()], first)
        ]
        if not changed:
            return None
        names = []
        for a, name, kind, off, n in self.places:
            if (arm is None or a == arm) and kind in changed:
                buf = self.pools[kind][0]
                if not torch.equal(buf[off : off + n], self.first[kind][off : off + n]):
                    names.append(name)
        return names or ["(outside its buffers)"]

    def gap_written(self) -> bool:
        """Whether any pool's gap holds a byte other than GUARD."""
        return any(bool((buf[size:] != GUARD).any()) for buf, size in self.pools.values())

    def check(self, arm) -> str | None:
        """The owned check of an arm after its replay."""
        names = self.kept_changed(arm)
        if names is not None:
            return f"kept:{','.join(names)}: not left as found"
        if self.gap_written():
            return "bounds:owned: a gap after the owned memory written"
        return None
