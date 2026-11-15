"""The launch arithmetic: the divisor pair and a rounded-up division."""

from __future__ import annotations

from thin.runtime.errors import LaunchError

__all__ = ["cdiv", "magic"]


def _exact_int(x) -> bool:
    return type(x) is int


def magic(d):
    """(mul, shift) for dividing a u32 below 2^31 by d."""
    if not _exact_int(d) or not 1 <= d <= 1 << 31:
        raise LaunchError(f"magic({d!r}): a divisor in 1 .. 2^31")
    shift = (d - 1).bit_length()
    mul = (-(-(1 << (32 + shift)) // d) - (1 << 32)) % (1 << 32)
    return mul, shift


def cdiv(a, b):
    """``a / b`` rounded up, refusing 0."""
    if b == 0:
        raise LaunchError("division by 0")
    return -(-a // b)
