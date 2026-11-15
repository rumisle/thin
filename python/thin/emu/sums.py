"""Sum records."""

from __future__ import annotations

__all__ = ["sum_record_tags", "sum_sources"]


def sum_sources(tag) -> tuple:
    """A sum record's leaves, left to right."""
    raise NotImplementedError


def sum_record_tags(xt, yt):
    """The sum records of two operands' tags."""
    raise NotImplementedError
