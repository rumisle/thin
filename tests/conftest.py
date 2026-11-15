"""Measures the test session against tests/budgets.toml.

Every run prints the slowest tests, the session's wall time and peak RSS,
and fails when one exceeds its budget: a test not marked ``slow`` above
``test_seconds``, the session above ``suite_seconds`` (slow tests not
counted), or the process above its RSS ceiling. A test marked
``slow("<section>")`` is held to that section's ``seconds`` and
``peak_rss_mb`` instead.

A test marked ``gfx1201`` needs that GPU: it skips unless HIP has one,
runs with the first one current (torch's and HIP's), and is held to the
``[gfx1201]`` budgets as a slow test is to its section. ``pytest -m
gfx1201`` runs the hardware suite alone.

``THIN_BUDGET_SCALE`` multiplies the time budgets on a slower machine
(tests/budgets.py); the header, the cost line and every time violation
name the factor whenever it is not 1.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import pytest

from tests import budgets

SCALE = budgets.time_scale()
BUDGETS = budgets.load(scale=SCALE)
DURATIONS: dict[str, float] = defaultdict(float)
"""Each test's setup, call and teardown seconds."""

SLOW: dict[str, str] = {}
"""The budget section of each test marked slow."""

VIOLATIONS: list[str] = []
STARTED = time.perf_counter()
TORCH_MB: list[float | None] = [None]
"""What loading torch cost the session (budgets.torch_baseline_mb)."""
TORCH_S: list[float] = [0.0]
"""What loading torch took the session (its import and context, measured first):
not ours, so the suite's time is held net of it, as its RSS is."""
TORCH_PEAK_MB: list[float | None] = [None]
"""The session's peak RSS once torch was loaded: some builds (ROCm's)
peak far above what they keep while importing, none of it ours."""


def _scaled(seconds: float) -> str:
    """A time budget as reported: with the factor it was scaled by, if any."""
    if SCALE == 1.0:
        return f"{seconds:g} s"
    return f"{seconds:g} s: {seconds / SCALE:g} s scaled by {budgets.SCALE_VARIABLE}={SCALE:g}"


def pytest_report_header():
    """Names a time-budget factor where one is set."""
    if SCALE != 1.0:
        return [
            f"time budgets scaled by {budgets.SCALE_VARIABLE}={SCALE:g} (memory budgets as written)"
        ]
    return []


def pytest_addoption(parser):
    """``--stored=DIR``: the store the GPU tests compile through."""
    parser.addoption(
        "--stored", default="~/.cache/thin-store", help="the store tests compile through"
    )


@pytest.fixture
def stored(request):
    """Compiles through the ``--stored`` store for the test, then restores the one before."""
    import thin

    before = thin.use_store(request.config.getoption("stored"))
    yield request.config.getoption("stored")
    thin.use_store(before)


def pytest_configure(config):
    """Registers the slow marker, and loads torch first to measure its cost."""
    config.addinivalue_line(
        "markers", "slow(section): an opt-in heavy test, held to that section of budgets.toml"
    )
    config.addinivalue_line(
        "markers", "gfx1201: needs a gfx1201 (RDNA 4) GPU; runs on the first one, skips without"
    )
    config.addinivalue_line("markers", "gpu: a GPU cross-check; every gfx1201 test is one")
    config.addinivalue_line(
        "markers",
        "isolated: runs in a child process of its own (graph capture has a per-process limit)",
    )
    t = time.perf_counter()
    TORCH_MB[0] = budgets.torch_baseline_mb()
    TORCH_S[0] = time.perf_counter() - t if TORCH_MB[0] is not None else 0.0
    if TORCH_MB[0] is not None:
        TORCH_PEAK_MB[0] = budgets.peak_rss_mb()


def pytest_runtest_logreport(report):
    """Adds up each test's time."""
    DURATIONS[report.nodeid] += report.duration


_GPU: list[int | None] = []
"""HIP's index of the first gfx1201, or None: asked once a session."""


def _gfx1201() -> int | None:
    """HIP's index of the first gfx1201, or None (none, or no torch to run on it)."""
    if not _GPU:
        import importlib.util

        from thin.runtime import hip

        archs, _ = hip.gpus()
        torch = importlib.util.find_spec("torch") is not None
        _GPU.append(archs.index("gfx1201") if "gfx1201" in archs and torch else None)
    return _GPU[0]


@pytest.fixture(autouse=True)
def _on_the_gfx1201(request):
    """Makes the gfx1201 current for a test marked so: a machine may have
    others (an APU's integrated GPU enumerates too), and the runtime runs on
    the current one."""
    if request.node.get_closest_marker("gfx1201") is None:
        yield
        return
    import torch

    index = _gfx1201()
    before = torch.cuda.current_device()
    torch.cuda.set_device(index)
    assert torch.cuda.get_device_properties(index).gcnArchName.split(":")[0] == "gfx1201"
    yield
    torch.cuda.set_device(before)


_ISOLATED: set = set()
"""The node ids of tests run in a child of their own."""
_CHILD = "THIN_ISOLATED_CHILD"
"""Set in an isolated test's child process."""


def _in_a_child(item):
    """Runs item in a pytest child of its own; fails with the child's tail."""
    import os
    import subprocess

    env = {**os.environ, _CHILD: "1"}
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--runxfail"]
    done = subprocess.run(
        [*args, item.nodeid], capture_output=True, text=True, env=env, timeout=900
    )
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-25:])
        pytest.fail(f"in its own process (exit {done.returncode}):\n{tail}", pytrace=False)


def pytest_collection_modifyitems(items):
    """Notes which tests are slow, and under which budget; skips the
    hardware tests away from their GPU."""
    for item in items:
        marker = item.get_closest_marker("slow")
        if marker is not None:
            SLOW[item.nodeid] = marker.args[0]
        import os

        if item.get_closest_marker("isolated") is not None and not os.environ.get(_CHILD):
            item.runtest = lambda item=item: _in_a_child(item)
            _ISOLATED.add(item.nodeid)
        if item.get_closest_marker("gfx1201") is not None:
            SLOW.setdefault(item.nodeid, "gfx1201")  # an explicit slow section wins
            item.add_marker(pytest.mark.gpu)
            if _gfx1201() is None:
                item.add_marker(pytest.mark.skip(reason="needs a gfx1201 GPU and torch"))


def _cost() -> tuple[float, float | None, float | None]:
    """The session's seconds (slow tests not counted) and peak RSS, of
    this process and of its largest child, in MB."""
    seconds = time.perf_counter() - STARTED - TORCH_S[0]
    seconds -= sum(DURATIONS[n] for n in SLOW if n in DURATIONS)
    return seconds, budgets.peak_rss_mb(), budgets.peak_rss_mb(children=True)


def _torch_mb() -> float:
    """Torch's measured cost; where it could not be measured but torch is
    loaded, the budget file's fallback."""
    if TORCH_MB[0] is not None:
        return TORCH_MB[0]
    return BUDGETS["pytest"]["torch_mb_unmeasured"] if "torch" in sys.modules else 0.0


def _violations() -> list[str]:
    """Every budget the session exceeded."""
    pytest_budget = BUDGETS["pytest"]
    seconds, rss, children = _cost()
    found = []
    for nodeid, spent in DURATIONS.items():
        limit = (
            BUDGETS[SLOW[nodeid]]["seconds"] if nodeid in SLOW else pytest_budget["test_seconds"]
        )
        if spent > limit:
            found.append(f"{nodeid}: {spent:.2f} s > {_scaled(limit)}")
    if seconds > pytest_budget["suite_seconds"]:
        found.append(f"the suite: {seconds:.2f} s > {_scaled(pytest_budget['suite_seconds'])}")
    # With torch, its import and CUDA context (measured as the session
    # starts) are not ours: the ceiling is the same budget above them.
    ceiling = max(pytest_budget["peak_rss_mb"] + _torch_mb(), TORCH_PEAK_MB[0] or 0.0)
    ran_slow = {SLOW[n] for n in DURATIONS if n in SLOW}
    ceiling = max([ceiling, *(BUDGETS[s]["peak_rss_mb"] for s in ran_slow)])
    # A child's peak includes the parent's pages it was forked with, so
    # children are held to a budget only where slow tests spawn workers.
    # An isolated test's child is a pytest session holding its own budgets: then the
    # children's peak (each one's torch import) is not judged again here.
    isolated = any(n in _ISOLATED for n in DURATIONS)
    judged = ran_slow and not isolated
    measured = [("this process", rss)] + ([("a child process", children)] if judged else [])
    for who, mb in measured:
        if mb is not None and mb > ceiling:
            found.append(f"peak RSS of {who}: {mb:.0f} MB > {ceiling:.0f} MB")
    return found


def pytest_terminal_summary(terminalreporter):
    """The cost of the session, then a per-file table of passing cases, as
    the Rust golden harness prints."""
    seconds, rss, children = _cost()
    terminalreporter.section("cost")
    slowest = sorted(DURATIONS.items(), key=lambda item: -item[1])
    for nodeid, spent in slowest[: int(BUDGETS["pytest"]["slowest"])]:
        terminalreporter.write_line(f"  {spent:7.3f} s  {nodeid}")
    mb = lambda v: "n/a" if v is None else f"{v:.0f} MB"  # noqa: E731
    torch = _torch_mb()
    if rss is None or not torch:
        ours = ""
    elif TORCH_PEAK_MB[0] and rss <= TORCH_PEAK_MB[0] + 1:
        ours = f" (torch's import peak; torch keeps {torch:.0f} MB, ours fit under it)"
    else:
        ours = f" (torch {torch:.0f} MB of it, ours {rss - torch:.0f} MB)"
    terminalreporter.write_line(
        f"  suite {seconds:.2f} s{f' (+ torch {TORCH_S[0]:.2f} s)' if TORCH_S[0] else ''} "
        f"(budget {_scaled(BUDGETS['pytest']['suite_seconds'])}), "
        f"peak RSS {mb(rss)}{ours} (budget {BUDGETS['pytest']['peak_rss_mb']:g} MB ours), "
        f"largest child {mb(children)}"
    )
    for violation in VIOLATIONS:
        terminalreporter.write_line(f"  OVER BUDGET: {violation}", red=True)


_RAN_GPU: list[bool] = [False]
"""Whether a gfx1201 test ran (not skipped) this session."""


def pytest_runtest_setup(item):
    """Notes a hardware test about to run."""
    if item.get_closest_marker("gfx1201") is not None and _gfx1201() is not None:
        _RAN_GPU[0] = True


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Leaves at once after a session that ran gfx1201 tests, its report written.

    On ROCm 7.1 tearing down torch's captured graphs at interpreter exit
    segfaults (the ROCm stack) after every test passed; the exit
    status is the session's, not the crash's.
    """
    if _RAN_GPU[0]:
        import os

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(getattr(config, "_thin_status", 0)))


def pytest_sessionfinish(session):
    """Fails the session when a budget is exceeded."""
    VIOLATIONS.extend(_violations())
    if VIOLATIONS and session.exitstatus == 0:
        session.exitstatus = 1
    session.config._thin_status = session.exitstatus
