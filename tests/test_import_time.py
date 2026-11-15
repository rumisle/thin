"""Import cost.

Each import runs in a fresh interpreter, best of three, so the figure is
the import alone. ``import thin.formats`` must also leave torch unloaded
.
"""

import subprocess
import sys

import pytest

from tests import budgets

PROBE = """
import sys, time
start = time.perf_counter()
import {module}
print(time.perf_counter() - start, "torch" in sys.modules)
"""


def _import(module: str) -> tuple[float, bool]:
    runs = []
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", PROBE.format(module=module)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        runs.append((float(out[0]), out[1] == "True"))
    return min(runs)


@pytest.mark.parametrize(
    ("module", "key"), [("thin", "thin_seconds"), ("thin.formats", "formats_seconds")]
)
def test_import_is_fast(module, key):
    seconds, torch_loaded = _import(module)
    limit = budgets.load()["import"][key]
    assert seconds <= limit, f"import {module}: {seconds:.3f} s > {limit:g} s"
    assert not torch_loaded
