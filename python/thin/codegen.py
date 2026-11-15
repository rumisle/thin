"""The pipeline's Python names.

The pipeline is native (thin._native); these names hand it a
kernel and the options in force, and wrap its results.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
from dataclasses import dataclass, field

from thin import _native

__all__ = [
    "Artifact",
    "Manifest",
    "Options",
    "code_object",
    "compile_kernel",
    "counter_waits",
    "current",
    "derive",
    "descriptor",
    "emit_kernel",
    "encode",
    "finish",
    "metadata",
    "place",
    "read_s",
    "render",
    "rendered",
    "split_manifest",
    "using",
]


from thin._options import Options, current, using

# --- The placement offered to the audit's second run ----------------


class _Offer:
    """A placement offered to the first Builder made while it is set."""

    __slots__ = ("placement", "taker")

    def __init__(self, placement):
        self.placement = placement
        self.taker = None


_OFFER: contextvars.ContextVar[_Offer | None] = contextvars.ContextVar("thin.offer", default=None)


def take_placement(b):
    """The placement offered to Builders, if this one is the first made under it."""
    offer = _OFFER.get()
    if offer is None or offer.taker is not None:
        return None
    offer.taker = b
    return offer.placement


def _options(o: Options) -> tuple:
    return (
        o.audit,
        o.keep_dead,
        o.waits_unique,
        o.safe_waits,
        o.regions,
        o.peak_list,
        o.peak_report,
        o.pressure,
        o.lds_wave_order,
    )


# --- The manifest ------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """The manifest record; ``line`` is its text."""

    symbol: str = ""
    kernarg_bytes: int = 0
    lds: int = 0
    wgsize: int = 0
    wgy: int = 0
    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    args: list = field(default_factory=list)
    build: dict = field(default_factory=dict)
    resident: int = 0
    mix: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    grid: list = field(default_factory=list)
    phases: dict = field(default_factory=dict)
    max_grid: list = field(default_factory=list)
    needs: dict = field(default_factory=dict)

    @classmethod
    def of(cls, json_text: str) -> Manifest:
        """The record of a line's JSON: defaults filled, an unknown key refused."""
        return cls(**json.loads(_native.manifest_record(json_text)))

    def record(self) -> dict:
        """The record as a dict."""
        return dataclasses.asdict(self)

    @property
    def line(self) -> str:
        """``; thin.manifest `` and the record as JSON, with a newline."""
        return _native.manifest_line(json.dumps(self.record()))

    def arg(self, name: str):
        """The first argument of that name, or None."""
        return next((a for a in self.args if a["name"] == name), None)

    @classmethod
    def read(cls, path) -> Manifest:
        """The manifest of a rendered file."""
        with open(path) as f:
            _, m = split_manifest(f.read())
        if m is None:
            raise ValueError(
                f"{path}: no thin.manifest line: not rendered by the compiler, "
                "or rendered before it"
            )
        return m


@dataclass(frozen=True)
class Artifact:
    """compile_kernel's result: the .s text, the manifest and a summary."""

    text: str
    """The .s text; its last line is the manifest."""

    manifest: Manifest
    """The manifest record."""

    summary: str
    """The summary."""


def split_manifest(text: str):
    """(body, Manifest or None) of a.s."""
    body, record = _native.split_manifest(text)
    return body, None if record is None else Manifest.of(record)


# --- The phases ----------------------------------------------------


def _kernel(result):
    from thin.dsl import Builder
    from thin.ir import AssignError

    b = result[0] if isinstance(result, tuple) and result else result
    if not isinstance(b, Builder):
        raise AssignError(
            f"derive: build() returned {type(result).__name__}, "
            "not a Builder or a tuple starting with one"
        )
    return b


def finish(b, nops=True):
    """Seals the kernel and runs FINISH, then POST_PLACE with ``nops``."""
    _native.pipeline_finish(b._n, _options(b.options), nops)


def place(result, build=None):
    """Places the kernel's registers, with the audit when a generator is given."""
    from thin.ir import AssignError

    b = _kernel(result)
    audit = b.options.audit and build is not None
    placement = _native.pipeline_place(b._n, _options(b.options), audit)
    if placement is None:
        return result
    offer = _Offer(placement)
    token = _OFFER.set(offer)
    try:
        again = build()
    finally:
        _OFFER.reset(token)
    second = _kernel(again)
    if offer.taker is not second:
        raise AssignError(
            "the generator's second run returned another Builder than the first it made "
            "(derive: the Builder it builds)"
        )
    _native.pipeline_audit(b._n, second._n, _options(second.options))
    return again


def derive(build):
    """``place(build(), build)``."""
    return place(build(), build)


def rendered(b):
    """(lines, snapshot) after RENDER."""
    from thin.ir import Snapshot

    lines, snap = _native.pipeline_rendered(b._n, _options(b.options))
    return lines, Snapshot(snap, list(b.ins._layout))


def render(b):
    """The kernel's lines."""
    return rendered(b)[0]


def compile_kernel(kernel, symbol, *, wgsize, args=None, lds=0, build=None, grid=(), max_grid=()):
    """Runs every phase, renders the .s file and its manifest."""
    entered = kernel.entered
    if entered is not None and args is not None and args is not entered.args:
        raise ValueError(
            f"{symbol}: compile_kernel(args=) and Builder.entry's block differ "
            "(give it once, to entry)"
        )
    block = None if args is None else args._a
    call = (
        symbol,
        wgsize,
        block,
        lds or 0,
        json.dumps(build or {}),
        list(grid),
        list(max_grid or ()),
    )
    text, record, summary = _native.pipeline_compile(kernel._n, call, _options(kernel.options))
    return Artifact(text, Manifest.of(record), summary)


def emit_kernel(out, symbol, kernel, **kwargs):
    """Compiles and writes the .s to ``out``; prints the summary."""
    art = compile_kernel(kernel, symbol, **kwargs)
    with open(out, "w") as f:
        f.write(art.text)
    print(f"{out}: {art.summary}")
    return art.manifest


# --- From a.s ----------------------------------------------------------


def code_object(text: str) -> bytes:
    """The code object of a.s."""
    return _native.code_object(text)


def encode(text: str) -> bytes:
    """The.text bytes of a.s."""
    return _native.encode(text)


def descriptor(text: str) -> bytes:
    """The 64-byte descriptor of a.s."""
    return _native.descriptor(text)


def metadata(text: str) -> dict:
    """The metadata a.s carries."""
    return json.loads(_native.metadata(text))


def read_s(text: str):
    """The .s reader: the text's code lines' encodings, metadata and descriptor."""
    return {"text": encode(text), "descriptor": descriptor(text), "metadata": metadata(text)}


def counter_waits(text: str) -> list[str]:
    """gfx1201's split-counter wait findings, as messages."""
    return [m for _, m in _native.counter_waits(text)]
