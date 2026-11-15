"""A generator's identity and the held forms of code.

The identity is (qualified name, sha256 of its source file, the line its
definition starts at, its captures): the closure's captured values, held
as build parameters are, plus **code held forms**: a function by its own
identity and captures (recursively, a cycle cut by name), a class or a
module by its name and file. This is how Triton keys a jitted function by
what it calls.

Loads only the standard library.
"""

from __future__ import annotations

import hashlib
import os
import types

__all__ = ["CodeHeld", "code_held", "file_sha", "generator_identity", "refusal"]

_SHAS: dict[str, str] = {}
"""Real path -> sha256 of its bytes, once a process (later edits are seen by the next)."""


_BY_NAME: dict[str, str] = {}
"""The path as given -> its sha (skips realpath's system calls after the first)."""


def file_sha(path: str) -> str:
    """The sha256 hex of a file's bytes, by its real path, once a process."""
    found = _BY_NAME.get(path)
    if found is not None:
        return found
    real = os.path.realpath(path)
    found = _SHAS.get(real)
    if found is None:
        with open(real, "rb") as f:
            found = hashlib.sha256(f.read()).hexdigest()
        _SHAS[real] = found
    _BY_NAME[path] = found
    return found


class CodeHeld:
    """A function, class or module as an address holds it: by its text."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:
        return self.text

    def __eq__(self, other) -> bool:
        return isinstance(other, CodeHeld) and other.text == self.text

    def __hash__(self) -> int:
        return hash(("code", self.text))


class UnheldError(TypeError):
    """A value with no held form; ``kind`` says which refusal, ``value`` what."""

    def __init__(self, kind: str, value):
        super().__init__(kind)
        self.kind, self.value = kind, value


_BUILTINS = (
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
)


def _readable(path) -> bool:
    return isinstance(path, str) and os.path.isfile(path)


def _class_file(cls) -> str | None:
    import sys

    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    return getattr(module, "__file__", None)


def code_held(v, walking: tuple = ()):
    """A function, class or module's code held form; None for anything else.

    Raises UnheldError("source") for one with no readable source file, and lets a
    capture's own refusal through (the walk is recursive).
    """
    if isinstance(v, types.MethodType):  # a bound method: its repr carries an address
        return None
    if isinstance(v, types.FunctionType):
        code = v.__code__
        if not _readable(code.co_filename):
            raise UnheldError("source", v)
        name = v.__qualname__
        if v in walking:
            return CodeHeld(f"function-ref({name!r})")
        caps = ", ".join(f"{n}={t}" for n, t in captures(v, (*walking, v)))
        sha = file_sha(code.co_filename)
        return CodeHeld(f"function({name!r}, {sha!r}, {code.co_firstlineno}, ({caps}))")
    if isinstance(v, type):
        path = _class_file(v)
        if not _readable(path):
            raise UnheldError("source", v)
        return CodeHeld(f"class({v.__qualname__!r}, {file_sha(path)!r})")
    if isinstance(v, types.ModuleType):
        path = getattr(v, "__file__", None)
        if not _readable(path):
            raise UnheldError("source", v)
        return CodeHeld(f"module({v.__name__!r}, {file_sha(path)!r})")
    if isinstance(v, _BUILTINS):
        raise UnheldError("source", v)
    return None


def held_value(v, walking: tuple = ()):
    """The held form of a build parameter or a captured value."""
    from thin.runtime.params import _PLAIN, HeldSet, _is_field

    if type(v) in _PLAIN:
        return v
    if type(v) in (tuple, list):
        return tuple(held_value(m, walking) for m in v)
    if type(v) in (set, frozenset):
        return HeldSet(sorted((held_value(m, walking) for m in v), key=repr))
    if _is_field(v):
        raise UnheldError("field", v)
    code = code_held(v, walking)
    if code is not None:
        return code
    if " at 0x" in repr(v):
        raise UnheldError("address", v)
    raise UnheldError("other", v)


def captures(fn, walking: tuple = ()) -> tuple:
    """(name, text of the held form) of each captured value of fn, in name order."""
    code = fn.__code__
    cells = dict(zip(code.co_freevars, fn.__closure__ or (), strict=True))
    out = []
    for name in sorted(cells):
        try:
            value = cells[name].cell_contents
        except ValueError:
            raise UnheldError("unbound", name) from None
        out.append((name, repr(held_value(value, walking))))
    return tuple(out)


def refusal(e: UnheldError, who: str, name: str, *, parameter: bool) -> TypeError:
    """Or TypeError for a value with no held form.

    `who` and `name`: the generator's qualname and the top-level capture, or
    (parameter) the build parameter's name.
    """
    v = e.value
    tn = type(v).__name__
    head = f"build parameter {name}" if parameter else f"{who}: captured {name}"
    if e.kind == "source":
        return TypeError(
            f"{head} is a {tn} with no source file (a builtin, a C function, or code made by "
            "exec or eval): it cannot be held"
        )
    if e.kind == "address":
        return TypeError(
            f"{head} is a {tn} whose repr carries a memory address: it has no held form (capture "
            "a held value, a function, a class or a module)"
        )
    if e.kind == "unbound":
        return TypeError(
            f"{head} is not bound yet (a closure's captured values are part of its identity)"
        )
    if e.kind == "field":
        if parameter:
            return TypeError(
                f"{head} = {v!r}: a run-time field's value; the kernel serves its range - pass "
                f"{name}.range (or .lo, .hi) to compile and the value to bind"
            )
        return TypeError(
            f"{head} = {v!r} is a run-time field's value; the kernel serves its range - capture "
            f"{name}.range (or .lo, .hi)"
        )
    if parameter:
        return TypeError(
            f"{head} = {v!r}: None, a bool, an int, a float, a str, a tuple / set of them, or a "
            "function, class or module with a source file"
        )
    return TypeError(
        f"{head} = {v!r}: a generator's captured values are held as its build parameters are "
        "(None, a bool, an int, a float, a str, a tuple / set of them, or a function, class or "
        "module with a source file)"
    )


def generator_identity(generator) -> tuple:
    """(qualname, file sha256, first line, captures) as a render's address holds it.

    TypeError for a generator without a source file, or a capture with no held
    form (at any level of the walk; the message names the top-level capture).
    """
    name = generator.__qualname__
    code = getattr(generator, "__code__", None)
    path = code.co_filename if code is not None else None
    if not _readable(path):
        raise TypeError(f"{name}: a generator lives in a file (its bytes are its identity)")
    cells = dict(zip(code.co_freevars, generator.__closure__ or (), strict=True))
    caps = []
    for cap in sorted(cells):
        try:
            try:
                value = cells[cap].cell_contents
            except ValueError:
                raise UnheldError("unbound", cap) from None
            caps.append((cap, repr(held_value(value, (generator,)))))
        except UnheldError as e:
            raise refusal(e, name, cap, parameter=False) from None
    return (name, file_sha(path), code.co_firstlineno, tuple(caps))
