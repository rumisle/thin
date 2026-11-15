"""Graph replay stack regression checks, opt-in: METRON_GRAPH_RETEST=1.

For each new ROCm or torch: graph replays far past the ~2^14 launches that
crash ROCm 7.1 on gfx1201, each case in a fresh process. A case passes on its
completion marker with the expected count, whatever its exit status (torch's
graph teardown at exit can segfault alone). Maintainers should enable a new
stack in graph_replays_ok() only after all stress cases pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gfx1201,
    pytest.mark.skipif(
        os.environ.get("METRON_GRAPH_RETEST") != "1", reason="opt-in: METRON_GRAPH_RETEST=1"
    ),
]
REPRO = Path(__file__).parent / "repro"
CASES = {
    "one graph of 341, 200 replays": ("replays done 68200", "graph_replays.py", "341", "200"),
    "one graph of 100, 1000 replays": ("replays done 100000", "graph_replays.py", "100", "1000"),
    "two graphs of 341, 100 replays each": (
        "replays done 68200",
        "graph_replays_spread.py",
        "two",
        "341",
        "100",
    ),
    "a fresh graph, 200 times": (
        "replays done 68200",
        "graph_replays_spread.py",
        "fresh",
        "341",
        "200",
    ),
    "20 A/A races, METRON_GRAPHS=1": ("races done 20", "graph_races.py"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_graph_retest(case):
    marker, script, *args = CASES[case]
    env = {k: v for k, v in os.environ.items() if k != "METRON_GRAPHS"}
    if script == "graph_races.py":
        env["METRON_GRAPHS"] = "1"
    run = subprocess.run(
        [sys.executable, str(REPRO / script), *args],
        env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    done = marker in run.stdout.splitlines()
    assert done, ("the fault reproduced", run.returncode, run.stdout[-500:], run.stderr[-1500:])
