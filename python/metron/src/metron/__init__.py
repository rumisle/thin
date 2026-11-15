"""metron: races of GPU kernels on the device.

A race times several arms (ways of computing the same outputs) on one point
(the inputs, the outputs' shapes and a check) under one state: every arm is
judged first, then the right and unchecked ones are timed in shuffled
rounds, interleaved on the same copies of the data, until each one's
verdict against the baseline (arm 0) is reached or the budget is spent.

The package imports nothing of the compiler, except its adapter
(``metron.thin``) and its kernel maker (``metron.maker``), which need the
``metron[thin]`` extra.
"""

from __future__ import annotations

from metron._memory import calls, copies, fitted, graph_calls, graph_replays_ok, hygiene, most
from metron._nouns import Arm, Call, Point, State, co_arm, torch_arm
from metron._race import ArmResult, Race, Verdict, race
from metron.record import current, read

__all__ = [
    "Arm",
    "ArmResult",
    "Call",
    "Point",
    "Race",
    "State",
    "Verdict",
    "calls",
    "co_arm",
    "copies",
    "current",
    "fitted",
    "graph_calls",
    "graph_replays_ok",
    "hygiene",
    "most",
    "race",
    "read",
    "torch_arm",
]
