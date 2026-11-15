"""Generate IR program outputs through thin's native analyses and Python API.

Pure target enumerations (pseudo.json) and numeric set operations (sets.json)
are deliberately not owned here.  All inputs are explicit programs in the
ir_programs, ir_verify and ir_host modules.  Golden fixtures are never opened
while constructing outputs; the comparison layer reads stored outputs afterward.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import subprocess
from pathlib import Path

from tools.golden import ROOT, encode
from tools.golden import ir_programs as programs
from tools.golden.ir_host import loc_data, model_data, passes_data, stream_data
from tools.golden.ir_verify import verify_data

BASE = "crates/thin-ir/tests/data/ir/"
NAMES = (
    "cfg",
    "exec",
    "flight",
    "m0",
    "solvers",
    "waves",
    "verify",
    "instr",
    "stream",
    "model",
    "loc",
    "passes",
)
OUTPUTS = tuple(BASE + name + ".json" for name in NAMES)


def native(requests):
    """Evaluate internal IR queries in thin-ir's Rust implementation."""
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-ir", "--example", "golden_authored"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    answers = json.loads(result.stdout)
    if len(answers) != len(requests):
        raise AssertionError(
            f"native IR returned {len(answers)} answers for {len(requests)} requests"
        )
    for request, answer in zip(requests, answers, strict=True):
        if isinstance(answer, dict) and "error" in answer:
            raise RuntimeError(
                f"native IR {request['kind']} refused authored input: {answer['error']}"
            )
    return answers


def native_data():
    """Build all native input plans first, then attach computed native answers."""
    cfg = programs.cfg_plans()
    solvers = programs.solver_plans()
    execute = programs.exec_plans()
    flight = programs.flight_plans()
    m0 = programs.m0_plans()
    lines = programs.instruction_lines()
    dpps = programs.dpp_plans()
    wave_lines = programs.wave_program()
    wave_cases = programs.WAVE_CASES

    requests = [{"kind": "cfg", **case} for case in cfg]
    requests += [{"kind": "solvers", "stream": case["stream"]} for case in solvers]
    requests += [{"kind": "exec", "lines": case["lines"]} for case in execute]
    requests += [
        {"kind": "flight", "lines": case["lines"], "loads": case["loads"]} for case in flight
    ]
    requests += [{"kind": "m0", "lines": case["lines"]} for case in m0]
    requests.append({"kind": "instrs", "lines": lines})
    requests += [{"kind": "dpp", "fields": fields} for fields in dpps]
    requests += [
        {"kind": "waves", "lines": wave_lines, "nwaves": n, "by_wave": by_wave}
        for _name, n, by_wave in wave_cases
    ]
    answers = iter(native(requests))

    cfg_cases = []
    for case in cfg:
        answer = next(answers)
        cfg_cases.append({**case, **answer})
    solver_cases = []
    for case in solvers:
        answer = next(answers)
        solver_cases.append({**case, **answer})
    exec_cases = []
    for case in execute:
        exec_cases.append({**case, "exec": next(answers)})
    flight_cases = []
    for case in flight:
        flight_cases.append({**case, "in_flight": next(answers)})
    m0_cases = []
    for case in m0:
        m0_cases.append({**case, "redundant": next(answers)})

    instrs = next(answers)
    dpp_cases = [{"fields": fields, "reads_old": next(answers)} for fields in dpps]
    waves = []
    for name, n, by_wave in wave_cases:
        answer = next(answers)
        answer.update(nwaves=n, by_wave=by_wave)
        waves.append({"name": name, **answer})

    notation = (
        "Authored gfx1201 notation: vN/sN and v[a:b]/s[a:b] name values placed at physical "
        "registers; named physical registers, integer immediates and @label operands are "
        "supported. {name=value|...} gives fields. '; m0_set=C' and '; waves=0bMASK' attach "
        "IR attributes. Abstract stream op/br/brw/jmp/end/label tokens denote concrete native "
        "instructions. Native results are entry states unless their key says otherwise."
    )
    return {
        "cfg": {
            "notation": (
                "Abstract streams recorded in each case; both CFG block policies "
                "are native results."
            ),
            "cases": cfg_cases,
        },
        "solvers": {
            "notation": (
                notation + " A caps block-length sums at 10; B increments with widening after "
                "patience 2; C follows only forward block edges; D unions visited blocks over all "
                "blocks; E is the backward union. visits is the native transfer evaluation order."
            ),
            "cases": solver_cases,
        },
        "exec": {
            "target": "gfx1201",
            "notation": notation,
            "result": "EXEC on entry to each row",
            "cases": exec_cases,
        },
        "flight": {
            "target": "gfx1201",
            "notation": notation,
            "result": "Tracked load rows still in flight on entry to each row",
            "cases": flight_cases,
        },
        "m0": {"target": "gfx1201", "notation": notation, "cases": m0_cases},
        "instr": {
            "target": "gfx1201",
            "notation": notation + " defs/uses are [value id, lo, n] over one shared value table.",
            "instrs": instrs,
            "dpp_reads_old": dpp_cases,
        },
        "waves": {
            "target": "gfx1201",
            "notation": notation,
            "lines": wave_lines,
            "cases": waves,
        },
    }


@contextlib.contextmanager
def _no_fixture_reads():
    """Refuse Python opens of any checked-in tests/data file during generation."""
    real_open, real_io_open, real_path_open = builtins.open, io.open, Path.open

    def checked(opener):
        """Construct checked inputs."""

        def open_checked(file, *args, **kwargs):
            """Construct open checked inputs."""
            if isinstance(file, (str, bytes, Path)):
                path = Path(file).as_posix()
                if "tests/data/" in path:
                    raise AssertionError(f"IR generation opened checked-in data: {path}")
            return opener(file, *args, **kwargs)

        return open_checked

    builtins.open = checked(real_open)
    io.open = checked(real_io_open)
    Path.open = checked(real_path_open)
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = real_open, real_io_open, real_path_open


def generate() -> dict[str, bytes]:
    """Construct every output without reading any existing golden fixture."""
    with _no_fixture_reads():
        docs = native_data()
        docs["verify"] = verify_data()
        docs["stream"] = stream_data()
        docs["model"] = model_data()
        docs["loc"] = loc_data()
        docs["passes"] = passes_data()
    return {path: encode(docs[name]) for path, name in zip(OUTPUTS, NAMES, strict=True)}
