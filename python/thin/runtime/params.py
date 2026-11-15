"""Build parameters, overriding and declare."""

from __future__ import annotations

import contextlib
import contextvars
import inspect

__all__ = ["HeldSet", "bind_params", "declare", "held", "held_pairs", "overriding"]

_OVERRIDES: contextvars.ContextVar[dict] = contextvars.ContextVar("thin.overriding")


class HeldSet(tuple):
    """A set parameter's held form: members sorted by their text, distinct from a tuple."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "{" + ", ".join(repr(m) for m in self) + "}" if self else "set()"

    def __eq__(self, other) -> bool:
        return isinstance(other, HeldSet) and tuple.__eq__(self, other)

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("set", tuple(self)))


_PLAIN = (type(None), bool, int, float, str)


def held(name: str, v):
    """A parameter's held form.

    A function, class or module is held by its code held form.
    """
    from thin.runtime.identity import UnheldError, held_value, refusal

    try:
        return held_value(v)
    except UnheldError as e:
        raise refusal(e, "", name, parameter=True) from None


def _is_field(v) -> bool:
    import sys

    opspec = sys.modules.get("thin.opspec")
    field = getattr(opspec, "Field", None)
    return field is not None and isinstance(v, field)


def bind_params(generator, params: dict) -> dict:
    """The given parameters, the overrides on top, the defaults applied."""
    given = {**params, **_OVERRIDES.get({}).get(generator, {})}
    sig = inspect.signature(generator)
    try:
        bound = sig.bind(**given)
    except TypeError as e:
        raise TypeError(f"{generator.__qualname__}: {e}") from None
    bound.apply_defaults()
    out = dict(bound.arguments)
    # A **kwargs parameter's dict: its keywords are parameters of their own.
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            out.update(out.pop(name, {}))
    return out


def held_pairs(params: dict) -> tuple:
    """(name, text of the held form) for every parameter, sorted by name."""
    return tuple(sorted((n, repr(held(n, v))) for n, v in params.items()))


@contextlib.contextmanager
def overriding(params):
    """In the block, compile and declare take these parameters over the caller's."""
    outer = _OVERRIDES.get({})
    merged = {g: dict(v) for g, v in outer.items()}
    for g, v in params.items():
        merged[g] = {**merged.get(g, {}), **v}
    token = _OVERRIDES.set(merged)
    try:
        yield
    finally:
        _OVERRIDES.reset(token)


_DEFS: dict = {}


def declare(generator, **params):
    """The KernelDef of a generator at these parameters, run once a process."""
    from thin.runtime.kernel import KernelDef

    bound = bind_params(generator, params)
    key = (generator, held_pairs(bound), _target_name())
    found = _DEFS.get(key)
    if found is None:
        found = generator(**bound)
        if not isinstance(found, KernelDef):
            raise TypeError(
                f"{generator.__qualname__} returned {type(found).__name__}: a generator returns "
                "thin.KernelDef(symbol, args, body, wgsize=, max_grid=, host=)"
            )
        _DEFS[key] = found
    return found


def _target_name() -> str:
    from thin.target import target

    return target().name
