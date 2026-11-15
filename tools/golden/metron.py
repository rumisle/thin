"""Timing statistics and sizing from explicit boundaries and seeded samples."""

from __future__ import annotations

import random

from metron import _memory, _race, stats
from tools.golden import encode

BASE = "tests/data/metron/"
OUTPUTS = tuple(
    BASE + name
    for name in (
        "stats.json",
        "sizing.json",
        "rounds.json",
        "kernels.json",
        "record.json",
        "nouns.json",
        "hygiene.json",
        "layout.json",
        "judging.json",
        "adapter.json",
    )
)


def stats_data():
    """Order statistics at every sample count and ratio-boundary decisions."""
    confidence = 0.95
    rng = random.Random(5)
    samples = [[5, 1, 9, 3, 7, 2, 8, 4, 6, 10], [3.0, 1.0, 2.0], [1] * 6, list(range(2, 15, 2))]
    samples += [[rng.uniform(0.9, 1.1) for _ in range(n)] for n in (20, 33)]
    samples += [[7.5]]
    medians = []
    for xs in samples:
        low, mid, high = stats.median_ci(xs)
        medians.append(dict(xs=xs, low=low, median=mid, high=high, k=stats.rank(len(xs))))
    ratios = [
        (0.997, 1.004, 0.005),
        (0.99, 0.998, 0.005),
        (1.002, 1.01, 0.005),
        (0.99, 1.01, 0.005),
        (0.999, 0.9995, 0.005),
        (1 / 1.02, 1.02, 0.02),
        (1 / 1.02 - 1e-12, 1.0, 0.02),
        (0.95, 0.999999, 0.02),
        (1.0, 1.05, 0.02),
        (1.000001, 1.05, 0.02),
        (0.95, 1.0, 0.02),
        (0.5, 0.6, 0.0),
        (1.0, 1.0, 0.0),
    ]
    verdicts = [
        {"low": lo, "high": hi, "within": w, "is": stats.verdict(lo, hi, w)} for lo, hi, w in ratios
    ]
    return {
        "notation": "Seeded samples and exact decision boundaries evaluated by metron.stats.",
        "confidence": confidence,
        "rank": [dict(n=n, k=stats.rank(n, confidence)) for n in range(1, 121)],
        "median_ci": medians,
        "verdict": verdicts,
    }


def called(case, name, function):
    """Evaluate a value or preserve the exact class and message of its refusal."""
    try:
        case[name] = function()
    except Exception as error:
        case.update(raises=type(error).__name__, message=str(error))
    return case


def sizing_data():
    """Memory/call-count boundaries for host launches and graph budgets."""
    constants = {
        name: getattr(_memory, name)
        for name in (
            "COLD",
            "GAP",
            "ALIGN",
            "REPLAY_US",
            "SWEEP_US",
            "PROBE",
            "MIN_ROUNDS",
            "POISON",
            "GUARD",
            "HYGIENE",
            "OWNED_ALIGN",
            "GENS_START",
            "GRAPH_CALLS",
            "PLAIN_MOST",
        )
    }
    kib, mib, gib = 1 << 10, 1 << 20, 1 << 30
    copy_inputs = (
        [(v, 1 << 16) for v in (0, 1, 128)]
        + [(128, 1 << 15)]
        + [
            (v, 1 << 16)
            for v in (
                3 * gib,
                2 * gib,
                2 * gib - 1,
                gib,
                256 * mib,
                96 * mib,
                6 * mib,
                12 * mib,
                32 * kib,
            )
        ]
    )
    copy_inputs += (
        [(128, n) for n in (1365, 585)]
        + [(v, 1365) for v in (768 * kib, 1536 * kib, 1536 * kib - 1, 12 * mib)]
        + [(768 * kib, 2), (256 * kib, 682)]
    )
    fitted_inputs = [
        (65536, 1.0),
        (65536, 71.0),
        (22, 1e6),
        (22, 0.0),
        (100, 2000.0),
        (100, 1999.0),
        (5, 3e4),
        (2, 1e9),
    ]
    call_inputs = [
        (22, 1000.0, 65536),
        (22, 1e6, 65536),
        (65536, 1.0, 65536),
        (8, 10.0, 65536),
        (8, 0.0, 65536),
        (7, 3333.0, 65536),
        (2, 1.0, 65536),
        (3, 1.0, 32768),
        (32768, 5.0, 32768),
        (22, 10000.0, 65536),
        (22, 9999.0, 65536),
        (1365, 1000.0, 1365),
        (171, 500.0, 1365),
        (171, 50.0, 1365),
        (585, 1.0, 585),
        (2, 1.0, 2),
        (2, 1.0, 3),
        (683, 100.0, 1365),
    ]
    data = {
        "notation": (
            "Sizing boundaries and explicit "
            "stack/environment inputs, evaluated "
            "through metron's API."
        ),
        "constants": constants,
        "copies": [
            dict(rotated_bytes=n, most=m, copies=_memory.copies(n, m)) for n, m in copy_inputs
        ],
        "fitted": [
            dict(copies=n, call_us=t, fitted=_memory.fitted(n, t)) for n, t in fitted_inputs
        ],
        "calls": [
            dict(copies=n, sweep_us=t, most=m, calls=_memory.calls(n, t, m))
            for n, t, m in call_inputs
        ],
    }
    graph_inputs = (
        [(None, v) for v in (70152802, 70000000, 60443483, 0)]
        + [("8192", 70152802), ("1", 70152802), ("65536", 60443483)]
        + [(v, 70152802) for v in ("0", "-5", "4k", "", "1.5")]
    )
    data["graph_calls"] = [
        called(
            dict(env=env, hip_version=version),
            "graph_calls",
            lambda e=env, v=version: _memory.graph_calls(e, v),
        )
        for env, version in graph_inputs
    ]
    most_inputs = [
        (4096, 1, 1, False),
        (4096, 1, 2, False),
        (4096, 1, 2, True),
        (4096, 1, 3, False),
        (4096, 2, 2, False),
        (4096, 8, 2, False),
        (4096, 8, 4, True),
        (4096, 1, 1, True),
        (65536, 1, 2, False),
        (6, 1, 2, False),
        (5, 1, 2, False),
        (13, 1, 2, True),
        (14, 1, 2, True),
        (1, 1, 1, False),
        (4096, 64, 31, True),
    ]
    data["most"] = [
        called(
            dict(budget=b, ranks=r, arms=a, cold=c),
            "most",
            lambda b=b, r=r, a=a, c=c: _memory.most(b, r, a, c),
        )
        for b, r, a, c in most_inputs
    ]
    arch = "gfx1201"
    replay_inputs = [
        (None, 70152802, [arch]),
        (None, 70000000, [arch] * 2),
        (None, 70152802, ["gfx1036", arch]),
        (None, 70152802, ["gfx1100"]),
        (None, 60443483, [arch]),
        (None, 80000000, [arch]),
        ("1", 70152802, [arch]),
        ("0", 60443483, ["gfx1100"]),
        ("yes", 70152802, [arch]),
        ("", 70152802, [arch]),
        ("true", 60443483, ["gfx1100"]),
    ]
    data["graph_replays_ok"] = [
        called(
            dict(env=e, hip_version=v, archs=a),
            "ok",
            lambda e=e, v=v, a=a: _memory.graph_replays_ok(e, v, a),
        )
        for e, v, a in replay_inputs
    ]
    return data


def round_record(name, raw, cold):
    """Evaluate the actual race statistics and stopping condition for every prefix."""
    case = dict(case=name, within=0.02, reference="r", cold=cold, raw=raw)

    def at(n=None):
        selected = {name: xs[:n] for name, xs in raw.items()}
        return _race.stats_of(selected, {name: [xs] for name, xs in selected.items()}, "r", cold)

    prelude, empty, adjusted, _, flag = at()
    arms = {}
    for arm, times in adjusted.items():
        low, mid, high = stats.median_ci(times)
        arms[arm] = dict(times=times, us=mid, ci=[low, high])
    verdicts = {}
    if "r" in adjusted:
        for name, times in adjusted.items():
            if name != "r":
                ratio, ci, said = stats.ratio_verdict(times, adjusted["r"], 0.02)
                verdicts[name] = {"ratio": ratio, "ci": list(ci), "is": said}
    rounds = len(next(iter(raw.values())))
    after = next(
        (k for k in range(_memory.MIN_ROUNDS, rounds + 1) if stats.settled(at(k)[2], "r", 0.02)),
        None,
    )
    case.update(
        prelude_us=prelude,
        empty_us=empty,
        arms=arms,
        verdicts=verdicts,
        settled_after=after,
        settled_final=stats.settled(adjusted, "r", 0.02),
        flags=[flag] if flag else [],
    )
    return case


def rounds_data():
    """Seeded exchangeable timings, paired differences and launch-bound edges."""
    rng = random.Random(11)
    plans = [
        ("a/a tie", 12, False, [("r", 100, 0.003), ("b", 100, 0.003), ("(empty)", 3, 0.05)]),
        ("b faster by 10 %", 8, False, [("r", 100, 0.01), ("b", 90, 0.01), ("(empty)", 3, 0.05)]),
        (
            "b slower by 1 %, noisy: undecided",
            10,
            False,
            [("r", 100, 0.03), ("b", 101, 0.03), ("(empty)", 3, 0.05)],
        ),
        (
            "cold: prelude taken off",
            7,
            True,
            [("r", 50, 0.002), ("b", 60, 0.002), ("(empty)", 3, 0.01), ("(prelude)", 8, 0.01)],
        ),
        ("one arm: interval width rule", 9, False, [("r", 100, 0.004), ("(empty)", 3, 0.05)]),
        ("one arm too noisy", 9, False, [("r", 100, 0.05), ("(empty)", 3, 0.05)]),
        (
            "reference untimed: width rule over the others",
            9,
            False,
            [("b", 100, 0.004), ("c", 80, 0.05), ("(empty)", 3, 0.05)],
        ),
        (
            "a slow episode costs one sample",
            10,
            False,
            [("r", 100, 0.002), ("b", 100, 0.002), ("(empty)", 3, 0.05)],
        ),
        (
            "cold: a launch-bound invalidate, prelude below the empty kernel: 0 and flagged",
            9,
            True,
            [
                ("r", 84.25, 0.0005),
                ("b", 84.25, 0.0005),
                ("(empty)", 2.2, 0.01),
                ("(prelude)", 2.19, 0.01),
            ],
        ),
        (
            "cold: prelude a hair above but within noise: 0 and flagged",
            8,
            True,
            [("r", 50, 0.002), ("b", 55, 0.002), ("(empty)", 2.2, 0.02), ("(prelude)", 2.21, 0.02)],
        ),
        (
            "cold: prelude above noise on every round: taken off, no flag",
            8,
            True,
            [
                ("r", 50, 0.002),
                ("b", 55, 0.002),
                ("(empty)", 2.2, 0.005),
                ("(prelude)", 2.5, 0.005),
            ],
        ),
    ]
    cases = []
    for i, (name, n, cold, arms) in enumerate(plans):
        if i == 8:
            rng = random.Random(55)
        raw = {
            name: [base * (1 + rng.uniform(-noise, noise)) for _ in range(n)]
            for name, base, noise in arms
        }
        if i == 7:
            raw["r"][4] *= 1.5
        cases.append(round_record(name, raw, cold))
    entries = _race.entries(["arm0", "arm1"], True)
    rng = random.Random(0)
    shuffle = []
    for _ in range(3):
        row = list(entries)
        rng.shuffle(row)
        shuffle.append(row)
    launch_inputs = [
        ("empty_as_arm", [4.1, 4.12, 4.09, 4.15, 4.11, 4.1], [4.05, 4.08, 4.02, 4.11, 4.06, 4.04]),
        ("vadd_2_23", [84.24, 84.25, 84.23, 84.26, 84.25, 84.24], [3.9, 4.1, 4.0, 4.2, 3.8, 4.0]),
        ("at_the_edge", [5.1] * 6, [5.0] * 6),
        ("just_past_the_edge", [5.11] * 6, [5.0] * 6),
        (
            "host_slower_than_device",
            [12.0, 12.4, 11.9, 12.2, 12.1, 12.0, 12.3],
            [12.1, 12.3, 12.0, 12.2, 12.2, 12.1, 12.4],
        ),
    ]
    launches = [
        dict(
            arm=name,
            raw=raw,
            host_us=host,
            within=0.02,
            flag=_race.launch_bound(name, raw, host, 0.02),
        )
        for name, raw, host in launch_inputs
    ]
    return {
        "notation": (
            "Seeded timing plans evaluated through metron's race/statistics implementation."
        ),
        "cases": cases,
        "shuffle_seed0": shuffle,
        "launch_bound": {
            "notation": "Median device/host timing boundary cases.",
            "cases": launches,
        },
    }


def generate():
    """Generate native timing-harness outputs from explicit synthetic plans."""
    try:
        import torch  # noqa: F401 - required by the host tensor/adapter generators
    except ImportError as error:
        raise RuntimeError(
            "metron generation requires torch; use a torch-enabled environment"
        ) from error
    from tools.golden.metron_host import (
        adapter_data,
        hygiene_data,
        judging_data,
        layout_data,
        nouns_data,
    )

    return {
        path: encode(doc)
        for path, doc in zip(
            OUTPUTS,
            (
                stats_data(),
                sizing_data(),
                rounds_data(),
                kernels_data(),
                record_data(),
                nouns_data(),
                hygiene_data(),
                layout_data(),
                judging_data(),
                adapter_data(),
            ),
            strict=True,
        )
    }


def kernels_data():
    """Compile the kept kernels and execute synthetic compare faults on the CPU."""
    import hashlib
    import json

    import numpy as np

    import thin
    from metron import maker
    from thin.target import target_names

    name = target_names()[0]
    kernels = {
        n: {
            "description": json.loads(desc),
            "code_object": {"sha256": hashlib.sha256(co).hexdigest(), "bytes": len(co)},
        }
        for n, (co, desc) in maker.build(name).items()
    }
    kib = 1 << 10
    chunk = maker.CHUNK
    plans = [
        (kib, 4, []),
        (4 * kib + 48, 5, [(2, 4 * kib + 47), (4, 0)]),
        (chunk + 4 * kib + 16, 3, [(1, chunk + 4 * kib + 3)]),
        (16, 3, [(1, 15)]),
        (16, 3, [(0, 0), (2, 7)]),
        (chunk, 2, [(1, chunk - 1)]),
        (chunk + 16, 2, [(1, chunk + 15)]),
        (chunk + 16, 2, [(1, chunk)]),
        (100 * kib + 16, 4, [(3, 100 * kib + 15), (1, chunk + 5)]),
        (4 * kib + 16, 6, [(5, 4 * kib), (5, 4 * kib + 1), (0, 4 * kib + 15)]),
    ]
    before = thin.use_store(None)
    try:
        kernel = thin.compile(maker.gen_differ)
        cases = []
        for stride, copies, changes in plans:
            rng = np.random.default_rng(stride + copies)
            ref = rng.integers(0, 256, stride, dtype=np.uint8)
            base = np.tile(ref, copies)
            for copy, byte in changes:
                base[copy * stride + byte] ^= 0x40
            flags = np.zeros(copies, dtype=np.uint32)
            grid = [-(-stride // chunk), copies]
            call = kernel.bind(grid=grid, args=dict(base=base, ref=ref, flags=flags, stride=stride))
            got = call.emulate().get("flags", flags).tolist()
            cases.append(
                dict(
                    stride=stride,
                    copies=copies,
                    changes=[list(pair) for pair in changes],
                    grid=grid,
                    flags=got,
                    flagged=[i for i, flag in enumerate(got) if flag],
                )
            )
    finally:
        thin.use_store(before)
    return {
        "notation": (
            "Thin-compiled harness kernels and "
            "CPU-measured copy-compare outputs for "
            "explicit chunk/tail faults."
        ),
        "chunk": chunk,
        "kernels": kernels,
        "compare": cases,
    }


def record_data():
    """Evaluate line serialization, verdicts and schema/group parsing from fake records."""
    import json
    import tempfile
    from pathlib import Path

    from metron import ArmResult, Race, record

    # These are synthetic record inputs, not observations of the current host.
    def arm(us, ci, times, fingerprint, wrong=None, ranks_us=None):
        return dict(
            us=us,
            ci=ci,
            times=times,
            wrong=wrong,
            checked=True,
            fingerprint=fingerprint,
            ranks_us=ranks_us,
        )

    fields = dict(
        point="vadd n=1024 seed 0",
        state=dict(warm=[], code="warm"),
        status="decided",
        rounds=6,
        arms={
            "ref": arm(
                10.0,
                [9.9, 10.2],
                [10.0, 9.9, 10.2, 10.1, 9.95, 10.05],
                "torch 2.x",
                ranks_us=[10.0],
            ),
            "b": arm(9.0, [8.9, 9.1], [9.0, 8.9, 9.1, 9.05, 8.95, 9.0], "abcd", ranks_us=[9.0]),
            "wrong": arm(
                None, None, [], "ef01", wrong="check: copy 0: z[3] = nan, want 1.0 (1 wrong)"
            ),
        },
    )
    validators = dict(
        metron="0123456789abcdef",
        torch="T",
        hip="H",
        amdgpu="D",
        vbios="V",
        arch="gfx1201",
        launches="plain",
    )
    conditions = dict(
        budget_s=5.0,
        bytes=dict(x=4096),
        calls=2,
        cold_copies=2,
        copies=2,
        empty_us=2.5,
        prelude_us=0.0,
        rotated=["x"],
        rounds_changed_inputs=0,
        seed=0,
        within=0.02,
        world=1,
    )
    device = dict(
        amdgpu="D",
        arch="gfx1201",
        cus=32,
        exclusive="1",
        firmware=dict(mec_fw_version="0x1"),
        name="AMD Radeon RX 9070 XT",
        pci="0000:03:00.0",
        vbios="V",
    )
    arms = {
        name: ArmResult(**{**a, "ci": None if a["ci"] is None else tuple(a["ci"])})
        for name, a in fields["arms"].items()
    }
    race = Race(
        point=fields["point"],
        state=fields["state"],
        arms=arms,
        status=fields["status"],
        rounds=fields["rounds"],
        conditions=conditions,
        flags=["constant:y"],
        device=device,
        validators=validators,
        phases_s=dict(after=0.01, judge=0.2, lay_out=0.1, rounds=1.0),
        wall_s=1.3,
    )
    rows = record.lines(race)
    for row in rows:
        row["time"] = "#TIME#"
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    verdicts = {
        name: {"ratio": v.ratio, "ci": list(v.ci), "is": v.verdict}
        for name, v in race.verdicts.items()
    }
    try:
        race.versus("wrong", "ref")
    except ValueError as error:
        versus_error = dict(error=type(error).__name__, message=str(error))

    def row(arm, validators=validators, schema=record.SCHEMA):
        return json.dumps(dict(schema=schema, validators=validators, arm=arm))

    old = {key: value for key, value in validators.items() if key != "launches"}
    read_lines = [
        row("a"),
        "",
        "not json",
        json.dumps(dict(schema=record.SCHEMA, arm="no validators")),
        row("b", {**validators, "torch": "other"}),
        row("old", old, "metron/1.2"),
        row("c", {**dict(reversed(list(old.items()))), "launches": "plain"}),
        row("d", {**validators, "launches": "graph"}),
        row("old4", old, "metron/4"),
    ]
    with tempfile.TemporaryDirectory(prefix="thin-record-") as temp:
        path = Path(temp) / "records.jsonl"
        path.write_text("\n".join(read_lines) + "\n")
        groups = [[v, [r["arm"] for r in rs]] for v, rs in record.read(path)]
    return {
        "notation": (
            "Synthetic race fields evaluated by "
            "metron record serialization and "
            "schema/group reading; timestamps are "
            "masked."
        ),
        "schema": record.SCHEMA,
        "race_fields": fields,
        "lines": lines,
        "verdicts": verdicts,
        "ranked": race.ranked(),
        "versus_error": versus_error,
        "read": dict(lines=read_lines, groups=groups),
    }
