"""metron imports nothing of the compiler but its adapter and maker, and works
without thin installed."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import metron
from metron import record

ALLOWED = {"thin.py", "maker.py"}


def test_only_the_adapter_and_the_maker_import_thin():
    root = Path(metron.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name in ALLOWED:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] == "thin"]
    assert offenders == []


CORE = ["metron", "metron.stats", "metron.record", "metron._memory", "metron._nouns",
        "metron._device", "metron._kernels", "metron._race"]  # fmt: skip


def _without_thin(code: str) -> subprocess.CompletedProcess:
    """Runs code in a fresh interpreter where importing thin fails, as with
    metron installed alone."""
    block = "import sys; sys.modules['thin'] = None\n"
    return subprocess.run(
        [sys.executable, "-c", block + code], capture_output=True, text=True, timeout=60
    )


def test_the_core_imports_without_thin():
    done = _without_thin(f"import importlib\nfor m in {CORE!r}: importlib.import_module(m)")
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("module", ["metron.thin", "metron.maker"])
def test_the_adapter_and_maker_need_thin(module):
    done = _without_thin(f"import {module}")
    assert done.returncode != 0 and "Error: import of thin" in done.stderr


def test_the_version_is_sixteen_hex_and_stable():
    """a function of the core's own source, not of thin."""
    v = record.version()
    assert len(v) == 16 and int(v, 16) >= 0
    done = _without_thin("from metron import record; print(record.version())")
    assert done.stdout.strip() == v
