"""The GEMM checker's Python names."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from thin import _native
from thin.emu import Hooks

__all__ = [
    "ACC_MARK",
    "GemmHooks",
    "Layout",
    "Problem",
    "fresh_tables",
    "poison_tag",
    "record_of",
    "record_pid",
    "record_tag",
    "rowmajor",
    "rowmajor_d",
    "union_records",
]

ACC_MARK = -1
"""The accumulator mark."""


def record_tag(m, n, kset, stat=False, pid=0):
    """The canonical tag of a record (n one column or a tuple of them)."""
    cols = list(n) if isinstance(n, tuple) else [n]
    return _native.emu_record_tag(m, cols, sorted(kset), stat, pid)


def record_of(tag):
    """(m, n, kset, stat) of a record tag."""
    m, n, k, stat, _ = _native.emu_record_of(tag)
    return m, (n[0] if len(n) == 1 else tuple(n)), frozenset(k), stat


def record_pid(tag):
    """A record's problem."""
    return _native.emu_record_of(tag)[4]


def poison_tag(msg):
    """The poison record of a reason."""
    return _native.emu_poison_tag(msg)


def union_records(xt, yt, stat=False):
    """Combines records per element."""
    return _native.emu_union_records(list(map(int, xt)), list(map(int, yt)), stat)


@dataclass(frozen=True)
class Layout:
    """A check's layout: data, made only by the constructors below."""

    shape: str
    """rowmajor or rowmajor_d."""
    params: tuple
    """The constructor's parameters, (name, value) pairs."""

    def encoded(self) -> dict:
        """The layout as the native checker reads it."""
        return {"shape": self.shape, **dict(self.params)}

    def decode(self, offsets):
        """The coordinates of byte offsets: int64 array to a dict of int64 arrays."""
        o = np.ascontiguousarray(offsets, dtype=np.int64)
        return _native.emu_layout_decode(json.dumps(self.encoded()), o)


def rowmajor(kind, ld_bytes, elem_bytes=1, rows_per_batch=0, batch_bytes=0) -> Layout:
    """A row-major operand."""
    p = {
        "kind": kind,
        "ld_bytes": ld_bytes,
        "elem_bytes": elem_bytes,
        "rows_per_batch": rows_per_batch,
        "batch_bytes": batch_bytes,
    }
    return Layout("rowmajor", tuple(p.items()))


def rowmajor_d(ld_bytes, elem_bytes=4, rows_per_batch=0, batch_bytes=0) -> Layout:
    """A row-major D."""
    p = {
        "kind": "D",
        "ld_bytes": ld_bytes,
        "elem_bytes": elem_bytes,
        "rows_per_batch": rows_per_batch,
        "batch_bytes": batch_bytes,
    }
    return Layout("rowmajor_d", tuple(p.items()))


def _table(t):
    return None if t is None else np.asarray(t, dtype=np.int64).tolist()


class Problem:
    """What a kernel is supposed to compute, all data.

    ``d_rows`` is an int64 table [M], ``d_cols`` one of [N] or [M][N] (a tuple
    of them for a fused epilogue), ``d_kblocks`` a dict of D row to k-block
    set, and ``record_operand`` (side, source problem[, (row, k) tables]). No
    argument is a callable: checkers never call back into Python. A table
    indexed past its end is a ValueError.
    """

    def __init__(
        self,
        layouts,
        K,
        d_cols=None,
        d_rows=None,
        d_kblocks=None,
        N=None,
        record_operand=None,
        scatter=None,
        pid=-1,
        fail_fast=True,
    ):
        """A problem; its pid is registered now unless given."""
        self.layouts, self.K, self.N = dict(layouts), K, N
        self.d_cols, self.d_rows, self.d_kblocks = d_cols, d_rows, d_kblocks
        self.record_operand, self.scatter = record_operand, scatter
        self.fail_fast = fail_fast
        source = record_operand[1].pid if record_operand else None
        self.pid = pid if pid >= 0 else _native.emu_register_problem(K, N, source)
        self.errors: list[str] = []
        self.store_log: list = []
        self.stores: dict = {}

    def encoded(self) -> dict:
        """The problem as the native checker reads it."""
        ro = self.record_operand
        link = None
        if ro:
            maps = {"rows": _table(ro[2][0]), "ks": _table(ro[2][1])} if len(ro) > 2 else {}
            link = {"side": {"A": 0, "B": 1}.get(ro[0], ro[0]), "source": ro[1].pid, **maps}
        cols = self.d_cols
        cols = {"tuple": [_table(c) for c in cols]} if isinstance(cols, tuple) else _table(cols)
        sets = lambda d: None if d is None else {str(k): sorted(v) for k, v in d.items()}  # noqa: E731
        return {
            "layouts": {
                n: (lay.encoded() if lay is not None else None) for n, lay in self.layouts.items()
            },
            "K": self.K,
            "N": self.N,
            "d_rows": _table(self.d_rows),
            "d_cols": cols,
            "d_kblocks": sets(self.d_kblocks),
            "scatter": sets(self.scatter),
            "record_operand": link,
            "pid": self.pid,
            "fail_fast": self.fail_fast,
        }

    def _take(self, results) -> None:
        errors, log, stores = results
        self.errors, self.store_log = list(errors), list(log)
        self.stores = dict(stores)

    def kblocks(self) -> set:
        """{0,..., ceil(K/32) - 1}, or {0} below 32."""
        return set(range(-(-self.K // 32))) if self.K >= 32 else {0}


class GemmHooks(Hooks):
    """The native GEMM checker over one or more problems.

    Like ``Hooks``, an object to pass, never a class to subclass.
    """

    products = False

    def __init__(self, prob, mem):
        """A checker of these problems over a memory's buffers."""
        object.__setattr__(self, "probs", list(prob) if isinstance(prob, (list, tuple)) else [prob])
        object.__setattr__(self, "mem", mem)

    @property
    def prob(self):
        """The first problem."""
        return self.probs[0]

    def _native(self):
        names = [b.name for b in self.mem.bufs]
        problems = json.dumps([p.encoded() for p in self.probs])
        return _native.EmuGemm(problems, names)

    def _take(self, native) -> None:
        for p, r in zip(self.probs, native.results(), strict=True):
            p._take(r)

    def __init_subclass__(cls, **kwargs):
        raise TypeError("checkers are native only: pass GemmHooks(...)")


def fresh_tables():
    """A test-only context: its block runs on fresh tag tables.

    The record, k-set, problem and sum tables are append-only for the life
    of the process; a vector that pins a fresh table's numbering
    (encoding.json) runs inside this block. The process's tables come back
    at its end. Nothing else resets them.
    """
    return _native.FreshTables()
