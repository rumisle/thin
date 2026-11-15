"""folds: the element-wise sum of bf16 pairs, rounded once."""

from __future__ import annotations

from thin.dsl import bf16_hi, bf16_lo, pair, pk_bf16

__all__ = ["bf16_sum"]


def bf16_sum(b, words):
    """A u32 tree: the sum of dwords of two bf16 each, in f32 in list order, rounded once.

    An empty list is refused.
    """
    if not words:
        raise ValueError("bf16_sum: no words to sum")
    first = words[0]
    s = b.v.f32x2(pair(bf16_lo(first), bf16_hi(first)))
    for w in words[1:]:
        s += pair(bf16_lo(w), bf16_hi(w))
    return pk_bf16(s.lo, s.hi)
