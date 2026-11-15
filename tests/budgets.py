"""The test budgets of tests/budgets.toml, and how much a process used.

The file is a flat subset of TOML (sections and ``key = number`` lines),
read here without tomllib, which Python 3.10 lacks.

The time budgets (every key ending in ``seconds``) are calibrated on a
developer machine. A slower machine says how much slower it is in
``THIN_BUDGET_SCALE`` (docs/dev/ci.md): unset, they are as written; set,
they are multiplied by it, from 1 to ``MAX_SCALE``, and every report says
so. Memory budgets are never scaled.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

PATH = Path(__file__).resolve().parent / "budgets.toml"

SCALE_VARIABLE = "THIN_BUDGET_SCALE"
MAX_SCALE = 4.0
"""Past this a machine is too unlike the calibration one for the budgets to
speak for it; a guard that wide would no longer catch a regression."""


def time_scale(environ: Mapping[str, str] = os.environ) -> float:
    """The factor the time budgets are multiplied by: 1 unless
    ``THIN_BUDGET_SCALE`` sets one; an unreadable or out-of-range one is
    an error, never ignored."""
    raw = environ.get(SCALE_VARIABLE)
    if raw is None:
        return 1.0
    try:
        scale = float(raw)
    except ValueError:
        raise ValueError(f"{SCALE_VARIABLE}={raw!r}: not a number") from None
    if not 1.0 <= scale <= MAX_SCALE:
        raise ValueError(f"{SCALE_VARIABLE}={raw!r}: not in [1, {MAX_SCALE:g}]")
    return scale


def load(path: Path = PATH, scale: float | None = None) -> dict[str, dict[str, float]]:
    """Each section's budgets, the time ones multiplied by ``scale``
    (``time_scale()`` unless given)."""
    scale = time_scale() if scale is None else scale
    budgets: dict[str, dict[str, float]] = {}
    section = budgets.setdefault("", {})
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = budgets.setdefault(line[1:-1].strip(), {})
            continue
        key, sep, value = (part.strip() for part in line.partition("="))
        if not sep or not key:
            raise ValueError(f"{path}:{number}: not `key = number`: {raw!r}")
        section[key] = float(value) * (scale if key.endswith("seconds") else 1.0)
    return budgets


def peak_rss_mb(children: bool = False) -> float | None:
    """The peak resident set of this process (or the largest of its waited-
    for children) in MB, None where the platform does not report it."""
    try:
        import resource
    except ImportError:
        return None
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    peak = resource.getrusage(who).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def rss_mb() -> float | None:
    """The current resident set of this process in MB (Linux), else None."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
    except (OSError, ValueError, IndexError):
        return None
    import os

    return pages * os.sysconf("SC_PAGE_SIZE") / (1 << 20)


def torch_baseline_mb() -> float | None:
    """What loading torch costs this process, in MB: its import and, on a
    CUDA machine, a context and the kernel modules torch loads lazily for
    the operations thin's torch paths run (formats' encode and decode of
    every element format, on a few elements); None without torch, or where
    the resident set cannot be read.

    thin.formats (numpy and the native extension) is loaded before the
    first reading, so none of thin is counted. The session's peak minus
    this is thin's and its tests' own.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return None
    from thin import formats

    before = rss_mb()
    try:
        import torch
    except ImportError:
        return None
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        x = torch.linspace(-500, 500, 256, device=device)
        for dtype in formats.DTYPES.values():
            for kwargs in ({"saturate": True}, {"saturate": False}):
                try:
                    bits = formats.encode(x, dtype, **kwargs)
                    formats.decode(bits, dtype)
                except (TypeError, ValueError):
                    pass
        if device == "cuda":
            torch.cuda.synchronize()
    after = rss_mb()
    return None if before is None or after is None else after - before
