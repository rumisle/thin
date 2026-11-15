"""Line helpers.

Instructions the layers make, and the printer's forms over register
files. Public: import these helpers from here.
"""

from __future__ import annotations

from thin import _native
from thin.ir import (
    Ablation,
    Comment,
    Imm,
    Instr,
    Staged,
    Tag,
    _line,
    _own,
    annotation,
    own,
    text,
)
from thin.ir.facts import answers, said

__all__ = [
    "LDS_FENCE",
    "SPIN",
    "bare",
    "comment",
    "imm",
    "label",
    "made",
    "named",
    "passed",
    "render",
    "waits",
]


LDS_FENCE = "[lds fence]"
"""The comment that marks a barrier as an LDS fence."""


SPIN = "[spin]"
"""The comment of a bounded poll's round counter."""


def comment(text: str, *, comment="", tag=(), abl=(), stage=(), loc=0, on=None) -> Instr:
    """An ``ir.comment`` with text ``"; " + text``, passed through ``made``.

    The notes are exactly made's keywords (``comment`` puts a Comment on
    this instruction); any other keyword is a TypeError. ``on``: the
    target, by default the current one.
    """
    i = Instr(own("ir.comment", on), (), {"text": "; " + text})
    return made(i, comment=comment, tag=tag, abl=abl, stage=stage, loc=loc)


def label(name: str, *, on=None) -> Instr:
    """An ``ir.label`` of that name on a target, by default the current one.

    This is the maker; ``thin.ir.facts.label`` reads a label.
    """
    return Instr(own("ir.label", on), (), {"name": name})


def made(i: Instr, comment="", tag=(), abl=(), stage=(), loc=0) -> Instr:
    """Puts Comment, Tag, Ablation and Staged where given and sets a nonzero loc."""
    if comment:
        i.put(Comment(comment))
    if tag:
        i.put(Tag(tag))
    if abl:
        i.put(Ablation(frozenset(abl)))
    if stage:
        i.put(Staged(stage))
    if loc:
        i.loc = loc
    return i


def waits(target, counts: dict, comment="", loc=0) -> list[Instr]:
    """A wait count vector as the target's count-less wait instructions."""
    out = []
    for mnemonic, count_fields in target.waits:
        mine = {
            target.counters[c]: n for c, n in counts.items() if target.counters[c] in count_fields
        }
        if mine:
            out.append(made(Instr(target.form(mnemonic, 0), (), mine), comment=comment, loc=loc))
    return out


def bare(i: Instr, files) -> str:
    """The line with ``files[f].text`` (physical once placed)."""
    return text(i) if _own(i) else _line(i, lambda r: files[r.file].text(r))


def named(i: Instr, files) -> str:
    """The line with ``files[f].name_text``."""
    return text(i) if _own(i) else _line(i, lambda r: files[r.file].name_text(r))


def render(i: Instr, files) -> str:
    """The line the .s body holds for i."""
    claim = annotation(i, lambda r: files[r.file].text(r))
    if claim is not None:
        return claim
    if i.kind in ("label", "comment"):
        return text(i)
    return _native.commented(bare(i, files), said(i) or None)


def passed(i: Instr, files) -> str:
    """``bare`` with label serials masked.

    A counter form with answers gets ``" @"`` and the sorted answers.
    """
    line = _native.mask_serials(bare(i, files))
    names = answers(i) if i.counter is not None else ()
    return line + " @" + ",".join(sorted(names)) if names else line


def imm(t: str) -> Imm:
    """Parses an immediate's text: int(t, 0), else float(t)."""
    try:
        value: int | float = int(t, 0)
    except ValueError:
        value = float(t)
    return Imm(value, "" if str(value) == t else t)
