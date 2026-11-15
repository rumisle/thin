"""Clusters: several ranks' launches over one memory."""

from __future__ import annotations

from thin import _native

__all__ = ["Cluster", "RankMachine"]


def _per_rank(value, kind=int) -> list:
    """An option given once or per rank, as a list."""
    return (
        []
        if value is None
        else [kind(value)]
        if isinstance(value, (int, float, str))
        else list(value)
    )


def _order(value) -> list[str]:
    """`order`: None, "reverse", an int seed, or one per rank."""
    one = {type(None): lambda v: [""], str: lambda v: [v], int: lambda v: [str(v)]}
    return one.get(type(value), lambda v: [_order(x)[0] for x in v])(value)


def _launches(kernarg) -> list[int]:
    """A rank's kernarg buffers' addresses, in launch order."""
    return [kernarg.base] if hasattr(kernarg, "base") else [k.base for k in kernarg]


def _carry(mode) -> tuple:
    """An LDS carry record: a mode name, or (fill data, tags)."""
    return (
        (mode, b"", [])
        if isinstance(mode, str) or mode is None
        else ("fill", bytes(mode[0]), list(mode[1]))
    )


class RankMachine:
    """One rank's machine after a run: its reports."""

    def __init__(self, cluster, rank):
        self.rank, self._reports = rank, cluster._native.reports(rank)

    @property
    def stats(self) -> dict:
        """The machine's counts."""
        return self._reports["stats"]

    @property
    def banks(self) -> dict:
        """Instruction index to [executions, cycles, conflict cycles]."""
        return self._reports["banks"]

    @property
    def vmw(self) -> dict:
        """The VMEM-wait coupling report."""
        return self._reports["vmw"]

    @property
    def ldsu(self) -> dict:
        """LDS reads of unwritten bytes."""
        return self._reports["ldsu"]


class Cluster:
    """Several ranks' launches over one memory."""

    def __init__(
        self,
        text,
        man,
        mem,
        kernargs,
        grid=(1,),
        hooks=None,
        resident=0,
        quantum=64,
        skew=None,
        jitter=None,
        wave_jitter=None,
        stall_rounds=200,
        speed=None,
        sum_records=False,
        trace=False,
        wg_speed=None,
        order=None,
        lockstep=False,
        hog=None,
        hog_stall=2000,
        lds_seed=None,
        lds_carry=None,
        max_ins=0,
        target=None,
    ):
        """A cluster; `man` = {wgsize, vgpr, agpr, lds}; kernargs[r] one buffer or a list."""
        from thin.emu import _target

        g = (*tuple(grid), 1, 1)
        carry = [_carry(c) for c in (lds_carry or [lds_seed] * bool(lds_seed))]
        options = {
            "resident": _per_rank(resident),
            "quantum": quantum,
            "order": _order(order),
            "skew": _per_rank(skew),
            "speed": _per_rank(speed, float),
            "wg_speed": {int(k): float(v) for k, v in (wg_speed or {}).items()},
            "jitter": jitter,
            "wave_jitter": wave_jitter,
            "stall_rounds": stall_rounds,
            "lockstep": lockstep,
            "hog": None if hog is None else list(hog),
            "hog_stall": hog_stall,
            "carry": carry,
            "max_ins": max_ins,
        }
        counts = (man["vgpr"], man.get("agpr", 0), man.get("lds", 0), man["wgsize"])
        launches = [_launches(k) for k in kernargs]
        self.mem, self.staged, self.ranks, self.hooks = mem, [], len(launches), hooks
        self._native = _native.EmuCluster(
            text,
            target or _target(),
            counts,
            launches,
            (g[0], g[1]),
            _Options(options),
            (sum_records, trace),
        )

    def run(self) -> list[RankMachine]:
        """Runs every rank; returns the machines in rank order."""
        gemm = self.hooks._native() if hasattr(self.hooks, "_take") else None
        self._native.run(self.mem._native, list(self.staged), gemm)
        if gemm is not None:
            self.hooks._take(gemm)
        return [RankMachine(self, r) for r in range(self.ranks)]

    @property
    def rounds(self) -> int:
        """Rounds run."""
        return self._native.rounds()

    @property
    def texts(self) -> list:
        """(text, line, comment) of each instruction."""
        return self._native.texts()


class _Options(dict):
    """The options as attributes, for the native side's extraction."""

    def __getattr__(self, name):
        return self[name]
