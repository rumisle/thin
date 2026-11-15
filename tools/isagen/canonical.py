"""Canonical JSON, the form whose sha256 the manifest pins."""

from __future__ import annotations

import hashlib
import json


def dumps(value) -> str:
    """Keys sorted, separators ``,`` and ``:``, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(value) -> str:
    """The sha256 hex digest of a value's canonical JSON."""
    return hashlib.sha256(dumps(value).encode("ascii")).hexdigest()
