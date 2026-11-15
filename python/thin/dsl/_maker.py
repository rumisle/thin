"""The instruction maker ``O`` and the memory families.

The native maker behind Python calls.
"""

from __future__ import annotations

import weakref

from thin import _native
from thin.dsl._expr import Tree
from thin.dsl._values import BUILDERS, TypedValue, as_reg
from thin.ir import B32, Instr, Places, Published, Value, VReg, _Row
from thin.target import PHYS, Phys, target


class Counter(str):
    """A counter or counter group named as a wait argument."""

    __slots__ = ()


VM = Counter("vm")
"""The vector-memory counter group."""
LGKM = Counter("lgkm")
"""The LDS, GDS, constant and message counter group."""

_LIT = Value(-1, "s", 1, 1, B32, "_lit")
LIT = VReg(_LIT, 0, 1)
"""The staging placeholder register."""


class Mod:
    """An operand with source modifiers (``O.neg``, ``O.abs_``)."""

    __slots__ = ("abs", "neg", "x")

    def __init__(self, x, neg: bool, abs_: bool):
        self.x, self.neg, self.abs = x, neg, abs_

    @property
    def byte(self) -> int:
        """The modifier byte, ``neg·1 | abs·2``."""
        return int(self.neg) | int(self.abs) << 1


OWNER: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""A Builder's values to its identity, for the register texts."""


def _builder(xs):
    for x in xs:
        native = getattr(x, "_typed", None)
        if native is not None:
            return BUILDERS.get(native.builder)
        b = OWNER.get(x.value) if isinstance(x, VReg) else None
        if b is not None:
            return BUILDERS.get(b)
    return None


def native_reg(r: VReg) -> tuple:
    """A register as the native Builder names it."""
    v = r.value
    return (v.id, v.file, r.lo, r.n)


def operand(x, mods: int = 0) -> tuple:
    """The native maker operand of a Python one."""
    if isinstance(x, Mod):
        return operand(x.x, x.byte)
    address = getattr(x, "_address", None)
    if address is not None:
        return ("a", address())
    x = as_reg(x)
    if isinstance(x, VReg):
        return ("r", native_reg(x), mods)
    if isinstance(x, Phys):
        return ("p", x.name, mods)
    if x is None:
        return ("n",)
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, int):
        return ("i", x)
    if isinstance(x, float):
        return ("f", x)
    return ("s", str(x))


def val(x) -> tuple:
    """A family's operand: a register, an int, a formula, an address or a text."""
    address = getattr(x, "_address", None)
    if address is not None:
        return ("a", address())
    if isinstance(x, Tree):
        return ("t", ("t", x._n))
    x = as_reg(x)
    if isinstance(x, VReg):
        return ("r", native_reg(x))
    if x is None:
        return ("n",)
    if isinstance(x, int) and not isinstance(x, bool):
        return ("i", x)
    return ("s", str(x))


def _regs(b, regs) -> tuple:
    values = b.values if b is not None else []
    return tuple(
        None if r is None else VReg(_LIT if r[0] < 0 else values[r[0]], r[1], r[2]) for r in regs
    )


def _tag(tag):
    return tuple(tag) if tag else None


def _cache(th, scope, sc0=False, sc1=False, nt=False, glc=False) -> tuple:
    flags = [k for k, on in (("sc0", sc0), ("sc1", sc1), ("nt", nt), ("glc", glc)) if on]
    return (th, scope, flags)


def _opts(offset, cache, comment, tag, bytes, signed, ordered) -> tuple:
    exact = bytes is None or (isinstance(bytes, int) and not isinstance(bytes, bool))
    b, written = (bytes, None) if exact else (None, repr(bytes))
    th, scope, flags = cache if len(cache) == 3 else (*cache, [])
    return (
        val(offset),
        th or "",
        scope or "",
        flags,
        comment,
        _tag(tag),
        b,
        written,
        signed,
        ordered,
    )


class Made:
    """What a family made that is not yet an instruction; ``b.emit`` emits it."""

    __slots__ = ("_m",)

    def __init__(self, m):
        self._m = m

    def __asm_emit__(self, b, abl=()):
        """Emits itself."""
        b._at()
        return b._row(b._n.emit_made(self._m, list(abl)))


def _out(b, result):
    ins, regs, made = result
    if made is not None:
        return Made(made)
    tgt = target()
    fid, name = ins.form()
    desc = tgt._forms[fid] if fid >= 0 else tgt._own[name]
    return Instr._of(_Row(desc, ins, _regs(b, regs), 0, None))


class Maker:
    """Makes instructions by dotted op or mnemonic."""

    VCC, EXEC, SCC, M0 = PHYS["vcc"], PHYS["exec"], PHYS["scc"], PHYS["m0"]
    VCC_LO, VCC_HI, EXEC_LO, EXEC_HI = (
        PHYS["vcc_lo"],
        PHYS["vcc_hi"],
        PHYS["exec_lo"],
        PHYS["exec_hi"],
    )
    NULL = PHYS["null"]
    VM, LGKM, LIT = VM, LGKM, LIT

    def op(self, name, *operands, comment="", tag=(), abl=(), stage=(), **fields):
        """Makes one instruction with the maker's checks."""
        b = _builder(operands)
        ops = [operand(x) for x in operands]
        st = _stage(stage)
        f = [(k, as_reg(v)) for k, v in fields.items()]
        result = _native.maker_op(
            target().name, b and b._n, name, ops, f, comment, _tag(tag), list(abl), st
        )
        return _out(b, result)

    @staticmethod
    def neg(x) -> Mod:
        """``-x`` as a source modifier."""
        return Mod(x.x, not x.neg, x.abs) if isinstance(x, Mod) else Mod(x, True, False)

    @staticmethod
    def abs_(x) -> Mod:
        """``|x|`` as a source modifier."""
        return Mod(x.x if isinstance(x, Mod) else x, False, True)

    @staticmethod
    def phys(name) -> Phys:
        """The named register of that name."""
        return PHYS[name]

    @staticmethod
    def place(ins, *names):
        """Puts ``Places(names)`` on ins and returns it."""
        ins.put(Places(tuple(names)))
        return ins

    @staticmethod
    def published(ins, why):
        """Puts ``Published(why)`` on ins and returns it."""
        if not why:
            raise ValueError("published(): say why")
        ins.put(Published(why))
        return ins

    @staticmethod
    def buffer_off(off) -> str:
        """``""`` for 0, else ``" offset:<off>"``, in the target's buffer offset range."""
        bits = next(
            f.bits
            for form in target().forms()
            if form.mem == "buffer"
            for f in form.fields
            if f.name == "offset"
        )
        top = (1 << bits) - 1
        if not 0 <= off <= top:
            raise ValueError(f"buffer offset {off} out of range (0..{top})")
        return f" offset:{off}" if off else ""

    def _family(self, family, data, xs, opts):
        b = _builder([data, *xs])
        d = native_reg(as_reg(data))
        result = _native.maker_family(
            target().name, b and b._n, family, d, [val(x) for x in xs], opts
        )
        return _out(b, result)

    def s_load(
        self, dst, base, offset=0, *, th=None, scope=None, glc=False, comment="", tag=()
    ):
        """A scalar load."""
        o = _opts(0, _cache(th, scope, glc=glc), comment, tag, None, False, "")
        return self._family("s_load", dst, [base, offset], o)

    def s_store(self, data, base, offset=0, *, th=None, scope=None, glc=False, comment=""):
        """A scalar store."""
        o = _opts(0, _cache(th, scope, glc=glc), comment, (), None, False, "")
        return self._family("s_store", data, [base, offset], o)

    def global_load(
        self,
        dst,
        addr,
        base=None,
        *,
        offset=0,
        th=None,
        scope=None,
        sc0=False,
        sc1=False,
        nt=False,
        comment="",
        tag=(),
        bytes=None,
        signed=False,
    ):
        """A global load."""
        c = _cache(th, scope, sc0=sc0, sc1=sc1, nt=nt)
        o = _opts(offset, c, comment, tag, bytes, signed, "")
        return self._family("global_load", dst, [addr, base], o)

    def global_store(
        self,
        addr,
        data,
        base=None,
        *,
        offset=0,
        th=None,
        scope=None,
        sc0=False,
        sc1=False,
        nt=False,
        comment="",
        tag=(),
        bytes=None,
    ):
        """A global store."""
        c = _cache(th, scope, sc0=sc0, sc1=sc1, nt=nt)
        o = _opts(offset, c, comment, tag, bytes, False, "")
        return self._family("global_store", data, [addr, base], o)

    def buffer_load(
        self,
        dst,
        voffset,
        desc,
        soffset=0,
        *,
        offset=0,
        th=None,
        scope=None,
        sc0=False,
        sc1=False,
        nt=False,
        comment="",
        tag=(),
        bytes=None,
        signed=False,
    ):
        """A buffer load."""
        c = _cache(th, scope, sc0=sc0, sc1=sc1, nt=nt)
        o = _opts(offset, c, comment, tag, bytes, signed, "")
        return self._family("buffer_load", dst, [voffset, desc, soffset], o)

    def buffer_store(
        self,
        data,
        voffset,
        desc,
        soffset=0,
        *,
        offset=0,
        th=None,
        scope=None,
        sc0=False,
        sc1=False,
        nt=False,
        comment="",
        tag=(),
        bytes=None,
    ):
        """A buffer store."""
        c = _cache(th, scope, sc0=sc0, sc1=sc1, nt=nt)
        o = _opts(offset, c, comment, tag, bytes, False, "")
        return self._family("buffer_store", data, [voffset, desc, soffset], o)

    def ds_read(
        self, dst, addr, offset=0, *, comment="", tag=(), ordered="", bytes=None, signed=False
    ):
        """An LDS load."""
        o = _opts(0, (None, None), comment, tag, bytes, signed, ordered)
        return self._family("ds_read", dst, [addr, offset], o)

    def ds_write(self, addr, data, offset=0, *, comment="", tag=(), ordered="", bytes=None):
        """An LDS store."""
        o = _opts(0, (None, None), comment, tag, bytes, False, ordered)
        return self._family("ds_write", data, [addr, offset], o)

    def mfma(
        self,
        form: str,
        dst,
        a,
        b,
        c=None,
        *,
        cbsz: int = 0,
        abid: int = 0,
        blgp: int = 0,
        neg=None,
        comment: str = "",
        tag=(),
        abl=(),
    ):
        """Makes a dense MFMA instruction (``c=None`` accumulates in-place into ``dst``)."""
        b_ctx = _builder([dst, a, b, c])
        c_op = None if c is None else operand(c)
        opts = _mfma_opts(cbsz, abid, blgp, neg, comment, tag, abl)
        result = _native.maker_mfma(
            target().name,
            b_ctx and b_ctx._n,
            form,
            operand(dst),
            operand(a),
            operand(b),
            c_op,
            opts,
        )
        return _out(b_ctx, result)

    def smfmac(
        self,
        form: str,
        dst,
        a,
        b,
        idx,
        *,
        cbsz: int = 0,
        abid: int = 0,
        comment: str = "",
        tag=(),
        abl=(),
    ):
        """Makes a sparse 4:2 SMFMAC instruction updating ``dst`` in-place."""
        b_ctx = _builder([dst, a, b, idx])
        opts = _mfma_opts(cbsz, abid, 0, None, comment, tag, abl)
        result = _native.maker_smfmac(
            target().name,
            b_ctx and b_ctx._n,
            form,
            operand(dst),
            operand(a),
            operand(b),
            operand(idx),
            opts,
        )
        return _out(b_ctx, result)


def _mfma_opts(cbsz, abid, blgp, neg, comment, tag, abl) -> tuple:
    if isinstance(cbsz, bool) or not isinstance(cbsz, int):
        raise TypeError(f"cbsz={cbsz!r}: an int")
    if isinstance(abid, bool) or not isinstance(abid, int):
        raise TypeError(f"abid={abid!r}: an int")
    if isinstance(blgp, bool) or not isinstance(blgp, int):
        raise TypeError(f"blgp={blgp!r}: an int")
    neg_list = None
    if neg is not None:
        if not isinstance(neg, (list, tuple)):
            raise TypeError(f"neg={neg!r}: a 3-element tuple or list")
        neg_list = [int(x) for x in neg]
    return (cbsz, abid, blgp, neg_list, comment, _tag(tag), list(abl))


def _stage(stage):
    if not stage:
        return None
    if stage[0] == "add":
        return ("add", native_reg(as_reg(stage[1])), stage[2])
    return ("mov", stage[1])


O = Maker()  # noqa: E741 - public instruction-maker name
"""The instruction maker."""

__all__ = ["LGKM", "LIT", "VM", "Counter", "Made", "Maker", "O", "TypedValue"]
