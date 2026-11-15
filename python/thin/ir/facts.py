"""Readers of an instruction's attributes.

Public: import these helpers from here.
"""

from __future__ import annotations

from thin.ir import Comment, InRegion, Instr, LdsAccess, Places, Staged, Tag, WavesTaken

__all__ = ["answers", "label", "region", "said", "say", "staged", "tag", "waves_taken"]


def label(i: Instr) -> str:
    """A label's ``name`` field, else ``i.target``, else "".

    This is the reader; ``thin.ir.stream.label`` makes a label.
    """
    if i.desc.pseudo == "Label":
        return i.field("name")
    return i.target or ""


def answers(i: Instr) -> frozenset[str]:
    """The LdsAccess region's tags (when present) and the Places names."""
    names = set(p.names) if (p := i.attr(Places)) else set()
    access = i.attr(LdsAccess)
    tags = getattr(access.region, "tags", None) if access else None
    return frozenset(names | set(tags() if tags else ()))


def waves_taken(i: Instr | None) -> frozenset[int] | None:
    """The WavesTaken set, or None (also for None)."""
    taken = i.attr(WavesTaken) if i is not None else None
    return taken.waves if taken else None


def said(i: Instr) -> str:
    """The Comment's text, or ""."""
    c = i.attr(Comment)
    return c.text if c else ""


def say(i: Instr, text: str) -> None:
    """Sets the comment; an empty text drops it."""
    i.put(Comment(text))


def tag(i: Instr) -> tuple:
    """The Tag's value, or."""
    t = i.attr(Tag)
    return t.value if t else ()


def staged(i: Instr) -> tuple:
    """The Staged stage, or."""
    s = i.attr(Staged)
    return s.stage if s else ()


def region(i: Instr) -> int | None:
    """The InRegion id, or None."""
    r = i.attr(InRegion)
    return r.id if r else None
