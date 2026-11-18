"""thin's version has one source: the Cargo workspace."""

import re
from pathlib import Path

import thin

ROOT = Path(__file__).resolve().parents[1]


def test_the_version_is_the_cargo_workspace_version():
    cargo = (ROOT / "Cargo.toml").read_text()
    workspace = cargo.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    version = re.search(r'^version = "([^"]+)"$', workspace, re.M).group(1)
    assert thin.__version__ == version
