"""The emulator's Python binding.

The machine and its checkers are native: this package converts values and
names the classes. ``memory`` owns buffer descriptions; ``numbers`` contains bit-level
conversion helpers. ``gemm`` supplies experimental provenance checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from thin import _native
from thin.emu.memory import LDS, POISON, Buffer, Memory
from thin.emu.numbers import LANES

__all__ = [
    "LANES",
    "LDS",
    "POISON",
    "TIMEOUTS",
    "Buffer",
    "Cluster",
    "Deadlock",
    "EmuError",
    "EmuLimit",
    "Hooks",
    "Imm",
    "Ins",
    "IpcPolicy",
    "Label",
    "Machine",
    "Memory",
    "Operand",
    "Program",
    "Reg",
    "RegisterInFlight",
    "ScopeModel",
    "Wave",
    "assume_claims",
    "assume_false",
    "held_text",
    "load_kernel",
    "load_program",
    "parse_line",
    "parse_text",
    "read_program",
    "undefined_claims",
    "uniform_claims",
    "user_sgprs",
]

TIMEOUTS = 64
"""Bounded-poll time-outs in a row that are a deadlock."""


class EmuError(Exception):
    """A refusal or fault of a run; ``cause`` is its class's cause."""

    cause = "fault"


class EmuLimit(EmuError):
    """Something the emulator does not model; never the kernel's fault."""

    cause = "emulator"


class Deadlock(EmuError):
    """A run where no wave can run or none makes progress."""

    cause = "deadlock"


class IpcPolicy(EmuError):
    """An IPC buffer accessed without the policy bits."""

    cause = "ipc"


class ScopeModel(EmuError):
    """A read of bytes another L2 holds, or of a stale kept line."""

    cause = "scope"


class RegisterInFlight(EmuError):  # noqa: N818 - the name
    """A register read or written while a load into it is still pending: a missing wait."""

    cause = "fault"


def _target() -> str:
    from thin.target import target

    return target().name


@dataclass(frozen=True)
class Reg:
    """A register: file v, s or a, or a modelled named register's name (repr v5, s[4:7], exec)."""

    file: str
    """The file letter, or the named register."""
    lo: int
    """The first register (a named register's part)."""
    n: int
    """The count."""

    def __repr__(self) -> str:
        if len(self.file) > 1:
            return self.file
        if self.n == 1:
            return f"{self.file}{self.lo}"
        return f"{self.file}[{self.lo}:{self.lo + self.n - 1}]"


@dataclass(frozen=True)
class Imm:
    """A 32-bit literal and its text."""

    value: int
    """The 32-bit pattern."""
    text: str = ""
    """The writer's text."""

    def __repr__(self) -> str:
        return self.text or hex(self.value)


@dataclass(frozen=True)
class Label:
    """A label reference."""

    name: str
    """The label, leading dots stripped."""

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Operand:
    """An operand with its source modifiers."""

    base: Reg | Imm | Label
    """What it is."""
    neg: bool = False
    """Negated."""
    abs: bool = False
    """Absolute value (always false for a literal)."""

    def encoded(self) -> dict:
        """The decoded JSON encoding of the operand."""
        flags = {k: True for k in ("neg", "abs") if getattr(self, k)}
        return {**_BASES[type(self.base)](self.base), **flags}


_BASES = {
    Reg: lambda b: {"reg": [b.file, b.lo, b.n]},
    Imm: lambda b: {"lit": [f"{b.value & 0xFFFFFFFF:#010x}", b.text]},
    Label: lambda b: {"label": b.name},
}

_OPERANDS = {
    "reg": lambda r: Operand(Reg(r[1], r[2], r[3]), r[4], r[5]),
    "lit": lambda r: Operand(Imm(r[2], r[1]), r[4]),
    "label": lambda r: Operand(Label(r[1])),
}

_FIELDS = {"true": lambda f: True, "int": lambda f: f[2], "list": lambda f: list(f[3])}


@dataclass(frozen=True)
class Ins:
    """A decoded instruction, the machine's input."""

    canon: str
    """The canonical op name the semantics are keyed by."""
    kind: str
    """The form kind."""
    args: tuple[Operand, ...]
    """The operands in order."""
    mods: dict | None = None
    """The fields by name, in order."""
    matrix: object = None
    """The matrix-form record of a matrix op."""
    text: str = ""
    """The text for messages; defaults to ``canon a, b, c``."""
    comment: str = ""
    """The markers' source."""
    line: int = 0
    """The source line, or 0."""

    def __post_init__(self):
        spelled = ", ".join(repr(a.base) for a in self.args)
        object.__setattr__(self, "text", self.text or " ".join(filter(None, [self.canon, spelled])))

    @classmethod
    def of(cls, encoded: dict) -> Ins:
        """An instruction from decoded JSON."""
        return Program.of({"ins": [encoded]}).ins[0]

    @classmethod
    def _of_row(cls, row) -> Ins:
        canon, kind, ops, fields, matrix, text, comment, line = row
        mods = {f[0]: _FIELDS[f[1]](f) for f in fields}
        return cls(
            canon, kind, tuple(_OPERANDS[o[0]](o) for o in ops), mods, matrix, text, comment, line
        )

    def encoded(self) -> dict:
        """The decoded JSON encoding of the instruction."""
        return {
            "op": self.canon,
            "kind": self.kind,
            "operands": [a.encoded() for a in self.args],
            "fields": dict(self.mods or {}),
            "text": self.text,
            "comment": self.comment,
            "line": self.line,
            "matrix": self.matrix,
        }


@dataclass(frozen=True)
class Program:
    """A decoded program: instructions, labels, section lines, user SGPRs."""

    ins: tuple[Ins, ...]
    """The instructions in order."""
    labels: dict[str, int]
    """Label to the index of the instruction after it."""
    sections: tuple = field(default=())
    """(index of the next instruction, text, line) of each note line."""
    user_sgprs: int = 2
    """The descriptor's user SGPR count."""
    workgroup_ids: tuple[bool, bool, bool] | None = None
    """Descriptor x/y/z enables; None for a body without a descriptor."""
    kernarg_sgpr: int | None = None
    """First SGPR of the enabled kernarg pointer, if any."""
    pc_layout: tuple[tuple[tuple[int, int], ...], int] | None = None
    """Explicit byte ranges and text end; absent unless PC instructions need them."""

    @classmethod
    def of(cls, encoded: dict) -> Program:
        """A program from decoded JSON, masks at the target's W."""
        lanes = _native.target_info(_target())[0]
        return cls._of_rows(_native.emu_program_of(json.dumps(encoded), lanes))

    @classmethod
    def _of_rows(cls, rows) -> Program:
        ins, labels, sections, user_sgprs, _, ids, kernarg, pc = rows
        return cls(
            tuple(Ins._of_row(r) for r in ins),
            dict(labels),
            tuple(sections),
            user_sgprs,
            None if ids is None else tuple(ids),
            kernarg,
            None if pc is None else (tuple(pc[0]), pc[1]),
        )

    def encoded(self) -> dict:
        """The decoded JSON encoding of the program."""
        return {
            "ins": [i.encoded() for i in self.ins],
            "labels": dict(self.labels),
            "sections": [list(s) for s in self.sections],
            "user_sgprs": self.user_sgprs,
            "workgroup_ids": self.workgroup_ids,
            "kernarg_sgpr": self.kernarg_sgpr,
            "pc_layout": None
            if self.pc_layout is None
            else {"ranges": self.pc_layout[0], "end": self.pc_layout[1]},
        }


def read_program(text: str) -> Program:
    """The program of a.s: the target's reader, then its decoding map."""
    return Program._of_rows(_native.emu_read_program(text, _target()))


def parse_text(text: str, sections: bool = False):
    """(ins, labels), or (ins, labels, sections), of a.s."""
    p = read_program(text)
    return {False: (list(p.ins), p.labels), True: (list(p.ins), p.labels, list(p.sections))}[
        bool(sections)
    ]


def parse_line(line: str, lineno: int = 0):
    """One line read the same way: Ins, a label name, or None."""
    text = "\n" * max(lineno - 1, 0) + line
    p = read_program(text)
    labels = list(p.labels)
    return (list(p.ins) + labels + [None])[0]


_CLAIMS: dict = {}


def _claims(text: str, kind: str):
    key = (text, kind, _target())
    found = _CLAIMS.get(key)
    found = found or _CLAIMS.setdefault(key, _native.emu_claims(text, _target(), kind))
    return found


def uniform_claims(text: str) -> dict:
    """{index: (regs, text)} of a .s's uniform claims, memoised by text."""
    uniform, _, _, refused, _ = _claims(text, "uniform")
    return {"error": refused} if refused else {i: (regs, t) for i, regs, t in uniform}


def assume_claims(text: str) -> dict:
    """{index: [(file, first, count, lo, hi, text)]}, or {error: message}."""
    _, assume, _, refused, _ = _claims(text, "assume")
    return {"error": refused} if refused else dict(assume)


def undefined_claims(text: str) -> dict:
    """{index: [(file, first, count)]}, or {error: message}.

    An edge claim is keyed ``edge:<from>-><to>``.
    """
    _, _, undefined, refused, edges = _claims(text, "undefined")
    if refused:
        return {"error": refused}
    out = dict(undefined)
    out.update({f"edge:{a}->{b}": regs for (a, b), regs in edges})
    return out


def assume_false(text, pc, wave, lane, value) -> str:
    """The message of a false assume claim."""
    return _native.emu_assume_false(text, pc, wave, lane, value)


def held_text(values) -> str:
    """The held-values spelling of a uniform claim (``[0x0, [0x1, 0x2]]``)."""
    return _native.emu_held_text([[v] if isinstance(v, int) else list(v) for v in values])


def user_sgprs(text: str) -> int:
    """The descriptor's user SGPR count of a.s."""
    return _native.emu_user_sgprs(text)


@dataclass(frozen=True)
class Hooks:
    """The plain checker: an object to pass, not a class to subclass.

    Checkers are native only; the plain checker and ``GemmHooks`` are the
    only ones, and the machine never calls back into Python.
    """

    products: bool = True
    """Whether the machine computes matrix-op values."""

    def __init_subclass__(cls, **kwargs):
        if cls.__module__ != "thin.emu.gemm":
            raise TypeError("checkers are native only: pass Hooks() or GemmHooks(...)")
        super().__init_subclass__(**kwargs)


@dataclass
class Wave:
    """A wave after a run: its registers and state."""

    wid: int
    """Its id in its workgroup."""
    V: object
    """VGPRs, an array [nv][W]."""
    S: list
    """SGPRs."""
    exec: int
    """EXEC."""
    vcc: int
    """VCC."""
    scc: int
    """SCC."""
    m0: int
    """M0."""
    pc: int
    """The next instruction."""
    done: bool
    """Ended."""

    @classmethod
    def _of_row(cls, row, lanes) -> Wave:
        wid, v, s, exec_, vcc, scc, m0, pc, done = row
        return cls(wid, v.reshape(-1, lanes), list(s), exec_, vcc, int(scc), m0, pc, done)


class Machine:
    """One rank's interpreter of one kernel."""

    def __init__(self, native, mem, lanes):
        self._native, self.mem, self.lanes = native, mem, lanes

    @property
    def wg_sgpr(self) -> int:
        """U of: the program's user SGPR count."""
        return self._native.user_sgprs()

    def configure(self, sum_records=False, ipc_policy=True, rank=0, trace=False) -> Machine:
        """Sets sum mode, the IPC policy, the rank and tracing; returns the machine."""
        self._native.configure(sum_records, ipc_policy, rank, trace)
        return self

    def run_wg(self, kernarg, wgid, waves):
        """Runs one workgroup (wgid (x, y) or (x, y, z)); returns its waves."""
        rows = self._native.run_wg(self.mem._native, kernarg, tuple(wgid), waves)
        return [Wave._of_row(r, self.lanes) for r in rows]

    @property
    def reports(self) -> dict:
        """stats, banks, vmw, ldsu."""
        return self._native.reports()

    @property
    def stats(self) -> dict:
        """The machine's counts."""
        return self.reports["stats"]

    @property
    def banks(self) -> dict:
        """Instruction index to [executions, cycles, conflict cycles]."""
        return self.reports["banks"]


def load_program(program, nv, na, lds_bytes, mem, hooks=None, trace=False) -> Machine:
    """A Machine of a decoded program, with its claims."""
    lanes = _native.target_info(_target())[0]
    native = _native.EmuMachine.from_json(
        json.dumps(program.encoded()), _target(), (nv, na, lds_bytes)
    )
    return Machine(native, mem, lanes).configure(trace=trace)


def load_kernel(text, nv, na, lds_bytes, mem, hooks=None, trace=False, parsed=None) -> Machine:
    """load_program(read_program(text), …)."""
    lanes = _native.target_info(_target())[0]
    native = _native.EmuMachine.from_text(text, _target(), (nv, na, lds_bytes))
    return Machine(native, mem, lanes).configure(trace=trace)


from thin.emu.cluster import Cluster  # noqa: E402  (it builds on the names above)
