"""Trees and the expression functions.

The Python operators delegate to the native trees.
"""

from __future__ import annotations

import builtins

from thin import _native
from thin.target import Phys

_I128 = 1 << 127


def leaf(x) -> tuple:
    """The native leaf of a Python operand (a tree, a value, a register, a constant, …)."""
    if isinstance(x, Tree):
        return ("t", x._n)
    native = getattr(x, "_typed", None)
    if native is not None:
        return ("v", native)
    if isinstance(x, Phys):
        return ("p", x.name)
    if isinstance(x, bool):
        return ("b", repr(x))
    if isinstance(x, int):
        return ("i", x) if -_I128 <= x < _I128 else ("big", repr(x))
    if isinstance(x, float):
        return ("f", x)
    return ("o", repr(x))


def tree(n) -> Tree:
    """The Python handle of a native tree."""
    t = Tree.__new__(Tree)
    t._n = n
    return t


def _bin(name):
    def op(self, other):
        return tree(_native.tree_binop(name, leaf(self), leaf(other)))

    def rop(self, other):
        return tree(_native.tree_binop(name, leaf(other), leaf(self)))

    return op, rop


def _cmp(kind):
    def op(self, other):
        return tree(_native.tree_compare(kind, leaf(self), leaf(other)))

    return op


def _divmod(name):
    def op(self, d):
        typed = getattr(d, "_typed", None)
        c = d if isinstance(d, int) and not isinstance(d, bool) else None
        text = repr(d) if typed is None else ""
        return tree(_native.tree_divmod(name, leaf(self), c, typed, text))

    return op


class Expr:
    """The operators of trees and typed values."""

    __slots__ = ()

    __add__, __radd__ = _bin("add")
    __sub__, __rsub__ = _bin("sub")
    __mul__, __rmul__ = _bin("mul")
    __and__, __rand__ = _bin("and")
    __or__, __ror__ = _bin("or")
    __xor__, __rxor__ = _bin("xor")
    __lshift__, __rlshift__ = _bin("shl")
    __rshift__, __rrshift__ = _bin("shr")
    __lt__ = _cmp("lt")
    __le__ = _cmp("le")
    __gt__ = _cmp("gt")
    __ge__ = _cmp("ge")
    __eq__ = _cmp("eq")
    __ne__ = _cmp("ne")
    __floordiv__ = _divmod("div")
    __mod__ = _divmod("mod")

    def __invert__(self):
        return tree(_native.tree_invert(leaf(self)))

    def __neg__(self):
        return tree(_native.tree_sign("neg", leaf(self)))

    def __abs__(self):
        return tree(_native.tree_sign("abs", leaf(self)))

    def __bool__(self):
        from thin.ir import ExprError

        raise ExprError(
            f"{_text(self)}: an expression is not a Python bool (a branch is b.when(c); a chain of "
            "compares is (a < b) & (b < c); a value is select(c, 1, 0))"
        )

    def __hash__(self):
        return id(self)


def _text(x) -> str:
    return x._n.text() if isinstance(x, Tree) else _native.tree_leaf(leaf(x)).text()


class Tree(Expr):
    """An expression over typed values and constants; emits nothing until written."""

    __slots__ = ("__weakref__", "_n")

    @property
    def op(self) -> str:
        """Its operator's name."""
        return self._n.op

    @property
    def args(self) -> list:
        """Its operands."""
        return [tree(a) for a in self._n.args]

    @property
    def type(self):
        """Its type, or None."""
        from thin.dsl._types import TYPES

        return TYPES.get(self._n.ty)

    @property
    def file(self) -> str | None:
        """Its file, or None."""
        return self._n.file

    @property
    def val(self):
        """Its value, as the node's kind holds it."""
        v = self._n.val
        if isinstance(v, _native.DslTyped):
            from thin.dsl._values import typed

            return typed(v)
        return v

    @property
    def text(self) -> str:
        """Its text at the top level."""
        return self._n.text()

    def __repr__(self) -> str:
        return self._n.text()

    def __asm_cond__(self, b, comment=None):
        """Lowers it as a condition."""
        return b._cond(self, comment)


def call(name: str, *xs) -> Tree:
    """The function ``name`` of trees."""
    return tree(_native.tree_call(name, [leaf(x) for x in xs]))


def _exprs(xs) -> bool:
    return builtins.any(isinstance(x, Expr) for x in xs)


def select(c, a, b):
    """``select(c, a, b)``: a lane-wise choice."""
    return tree(_native.tree_select(leaf(c), leaf(a), leaf(b)))


def _fn(name):
    def f(*xs):
        return call(name, *xs)

    f.__name__ = f.__qualname__ = name
    f.__doc__ = f"``{name}(…)`` ."
    return f


fma = _fn("fma")
exp2 = _fn("exp2")
log2 = _fn("log2")
rcp = _fn("rcp")
rsq = _fn("rsq")
sqrt = _fn("sqrt")
min3 = _fn("min3")
max3 = _fn("max3")
med3 = _fn("med3")
mul24 = _fn("mul24")
mad24 = _fn("mad24")
mulhi = _fn("mulhi")
mul_wide = _fn("mul_wide")
cvt_f32 = _fn("cvt_f32")
cvt_u32 = _fn("cvt_u32")
cvt_i32 = _fn("cvt_i32")
bf16_lo = _fn("bf16_lo")
bf16_hi = _fn("bf16_hi")
pk_bf16 = _fn("pk_bf16")
bf16 = _fn("bf16")
pair = _fn("pair")
popcount = _fn("popcount")
first = _fn("first")
first_lane = _fn("first_lane")
lane_read = _fn("lane_read")
lane_write = _fn("lane_write")
shuffle = _fn("shuffle")


def rank(m, base=0):
    """``rank(m, base=0)``."""
    return call("rank", m, base)


def min(*xs, **kw):
    """A tree's minimum, else Python's builtin."""
    return call("min", *xs) if _exprs(xs) else builtins.min(*xs, **kw)


def max(*xs, **kw):
    """A tree's maximum, else Python's builtin."""
    return call("max", *xs) if _exprs(xs) else builtins.max(*xs, **kw)


def _lanes(x) -> bool:
    return isinstance(x, (Expr, Phys))


def any(x, *rest):
    """``any(c)`` of a condition, else Python's builtin."""
    return call("any", x) if _lanes(x) and not rest else builtins.any(x, *rest)


def all(x, *rest):
    """``all(c)`` of a condition, else Python's builtin."""
    return call("all", x) if _lanes(x) and not rest else builtins.all(x, *rest)


def across(x, op, lanes):
    """A reduction across lanes."""
    return tree(_native.tree_lanes(False, leaf(x), op, lanes))


def scan(x, op, lanes=64):
    """A prefix scan across lanes."""
    return tree(_native.tree_lanes(True, leaf(x), op, lanes))


def bfe(a, offset, width, signed=None):
    """``bfe(a, offset, width)``: a bit field."""
    return tree(_native.tree_bfe(leaf(a), offset, width, signed))


def _fmt(fmt) -> str:
    return getattr(fmt, "name", fmt) if not isinstance(fmt, str) else fmt


def _at(at) -> tuple:
    exact = isinstance(at, int) and not isinstance(at, bool)
    return at, exact or at is None


def _opt(x):
    return None if x is None else leaf(x)


def unpack(w, fmt, at=None):
    """A storage format's element as an f32."""
    a, exact = _at(at)
    return tree(_native.tree_unpack(False, leaf(w), _fmt(fmt), a, exact))


def unpack2(w, fmt, at=None):
    """Two storage format elements as an f32x2."""
    a, exact = _at(at)
    return tree(_native.tree_unpack(True, leaf(w), _fmt(fmt), a, exact))


def pack2(a, b, fmt, into=None, at=None):
    """Two f32 packed into a storage format."""
    k, exact = _at(at)
    return tree(_native.tree_pack2(leaf(a), leaf(b), _fmt(fmt), k, exact, _opt(into)))


def op(name, *args, ty=None, **fields):
    """Any instruction as a node."""
    t = None if ty is None else ty.name
    return tree(_native.tree_instr(name, [leaf(x) for x in args], t, list(fields.items())))
