"""Statistics.

An order-statistic interval of a median, the verdict on a ratio of times, and
when a race's times are settled.
"""

from __future__ import annotations

import math

__all__ = ["median_ci", "rank", "settled", "verdict"]


def _below(n: int, k: int) -> float:
    """P(B < k) for B ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k)) / 2**n


def rank(n: int, confidence: float = 0.95) -> int:
    """The order statistic bounding a median of n samples.

    The k whose k-th smallest and k-th largest of n samples bound their median
    with at least `confidence`: the largest k with P(B < k) <= (1 - confidence) / 2,
    at most (n + 1) // 2, at least 1.
    """
    tail = (1 - confidence) / 2
    k = 1
    while k + 1 <= (n + 1) // 2 and _below(n, k + 1) <= tail:
        k += 1
    return k


def median_ci(xs) -> tuple:
    """(low, median, high) of samples.

    The k-th order statistics from each end, k = rank(n), and the usual median.
    """
    s = sorted(xs)
    n = len(s)
    k = rank(n)
    mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return s[k - 1], mid, s[n - k]


def verdict(low, high, within) -> str:
    """The verdict on an interval of a ratio arm / reference."""
    if low >= 1 / (1 + within) and high <= 1 + within:
        return "tie"
    if high < 1:
        return "faster"
    if low > 1:
        return "slower"
    return "undecided"


def ratio_verdict(a, b, within) -> tuple:
    """(ratio, (low, high), verdict) of times a against b, round by round."""
    low, mid, high = median_ci([x / y for x, y in zip(a, b, strict=True)])
    return mid, (low, high), verdict(low, high, within)


def settled(times: dict, reference: str, within: float) -> bool:
    """Whether a race's times are settled.

    With the baseline and another arm timed, no verdict against it undecided;
    else every timed arm's interval no wider than `within` x its median.
    """
    others = [n for n in times if n != reference]
    if reference in times and others:
        return all(
            ratio_verdict(times[n], times[reference], within)[2] != "undecided" for n in others
        )
    for xs in times.values():
        low, mid, high = median_ci(xs)
        if high - low > within * mid:
            return False
    return True
