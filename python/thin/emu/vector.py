"""Vector tag helpers."""

from __future__ import annotations

__all__ = ["float_tags", "lane_source"]


def lane_source(t):
    """A dword's one tag."""
    raise NotImplementedError


def float_tags(mach, xt, yt, stat=False, add=False):
    """The float rule."""
    raise NotImplementedError
