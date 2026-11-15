"""Construct golden inputs in code and evaluate them with thin or measured hardware.

Generators must not read tests/data. Existing golden files are opened only by
the command's comparison layer, after all fresh outputs have been produced.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def encode(doc, *, compressed=False) -> bytes:
    """Serialize deterministically, including reproducible gzip headers."""
    text = json.dumps(doc, ensure_ascii=False, indent=None if compressed else 1)
    raw = (text + "\n").encode()
    return gzip.compress(raw, mtime=0) if compressed else raw


def decode(raw: bytes, path: str):
    """Read a JSON document for semantic comparison, ignoring serialization."""
    return json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)


def data_files(root: Path = ROOT) -> set[str]:
    """All published test data, including binary images."""
    return {
        p.relative_to(root).as_posix()
        for base in (root / "crates", root / "tests" / "data")
        for p in base.rglob("*")
        if p.is_file() and ("tests/data/" in p.as_posix())
    }


_GUARDS = [0]
"""How many no_checked_data blocks are open; the audit hook refuses only then."""


def _refuse_checked_data(event, args):
    """Audit hook: refuses opening checked-in test data while a guard is open.

    The ``open`` audit event is raised by every way of opening a file by
    name (builtins.open, io.open, os.open, and Path.open however a Python
    version binds it: 3.10's pathlib holds its own reference to io.open), so
    the guard does not depend on which function object a caller reached.
    """
    if not _GUARDS[0] or event != "open":
        return
    file = args[0]
    if isinstance(file, (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(file)).as_posix()
        if "tests/data/" in path:
            raise PermissionError(f"generator accessed checked-in test data: {path}")


sys.addaudithook(_refuse_checked_data)


@contextlib.contextmanager
def no_checked_data():
    """Forbid fixture I/O during generator import and input/output construction.

    Comparison and installation happen only after this guard exits. Native
    examples consume stdin requests; they must not open golden files either.
    The guard is process-wide (every thread), as the open it intercepts is.
    """
    _GUARDS[0] += 1
    try:
        yield
    finally:
        _GUARDS[0] -= 1
