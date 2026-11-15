"""Load golden cases shared by Python and Rust integration tests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def data_path(component: str, file: str) -> Path:
    """Locate the owning crate's golden data, or Python-only test data."""
    owner = {"emu-tags": "emu", "target": "targets"}.get(component, component)
    core = ROOT / "crates" / f"thin-{owner}" / "tests" / "data" / component / file
    return core if core.parent.exists() else ROOT / "tests" / "data" / component / file


def load(component: str, file: str):
    """Read a golden JSON document, optionally compressed."""
    path = data_path(component, file)
    opener = gzip.open if file.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _items(doc, key: str | None):
    if key is not None:
        return [(f"{key}[{i}]", case) for i, case in enumerate(doc[key])]
    if isinstance(doc, list):
        return [(f"[{i}]", case) for i, case in enumerate(doc)]
    items = []
    for section, value in doc.items():
        if "notation" in section:
            continue
        if isinstance(value, list):
            items.extend((f"{section}[{i}]", case) for i, case in enumerate(value))
        elif isinstance(value, dict):
            items.extend((f"{section}[{k}]", c) for k, c in value.items())
    return items


def cases(component: str, file: str, key: str | None = None) -> list:
    """One pytest parameter per case: (golden file, case id, case, whole file)."""
    name = f"{component}/{file}"
    doc = load(component, file)
    return [
        pytest.param(name, case_id, case, doc, id=f"{file}:{case_id}")
        for case_id, case in _items(doc, key)
    ]


def cases_at(component: str, file: str, *path) -> list:
    """One pytest parameter per case of the list at ``doc[path[0]][path[1]]…``."""
    name = f"{component}/{file}"
    doc = load(component, file)
    found = doc
    for key in path:
        found = found[key]
    where = ".".join(path)
    return [
        pytest.param(
            name,
            f"{where}[{i}]",
            case,
            doc,
            id=f"{file}:{where}[{i}]",
        )
        for i, case in enumerate(found)
    ]


def whole(component: str, file: str) -> list:
    """The whole golden document as one test parameter."""
    doc = load(component, file)
    return [pytest.param(f"{component}/{file}", "file", doc, doc, id=file)]


def undecoded(what: str):
    """Reject an unsupported test notation."""
    raise NotImplementedError(f"decode the golden notation of {what}")
