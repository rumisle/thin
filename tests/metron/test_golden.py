"""Golden checks of timing-harness statistics, schemas and host protocols."""

from __future__ import annotations

import json
import re

import pytest

from tests.golden import cases, cases_at, data_path
from tools.golden.metron_inputs import (
    _act,
    _arms_of,
    _host,
    _hygiene_tensor,
    _judging_layout,
    _owned_of,
    _point_case,
    _result,
    _rle,
    _torch,
    _xy,
)

PARAMS = ("vector", "case_id", "case", "doc")


# --- stats.json, sizing.json -------------------------------------------------------------


@pytest.mark.parametrize(PARAMS, cases("metron", "stats.json"))
def test_stats(vector, case_id, case, doc):
    from metron import stats

    section = case_id.split("[")[0]
    if section == "rank":
        assert stats.rank(case["n"], doc["confidence"]) == case["k"]
    elif section == "median_ci":
        assert stats.median_ci(case["xs"]) == (case["low"], case["median"], case["high"])
        assert stats.rank(len(case["xs"])) == case["k"]
    else:
        assert stats.verdict(case["low"], case["high"], case["within"]) == case["is"]


@pytest.mark.parametrize(PARAMS, cases("metron", "sizing.json"))
def test_sizing(vector, case_id, case, doc):
    import metron
    from metron import _memory

    section = case_id.split("[")[0]
    if section == "constants":
        assert getattr(_memory, case_id.split("[")[1].rstrip("]")) == case
    elif section == "copies":
        assert metron.copies(case["rotated_bytes"], case["most"]) == case["copies"]
    elif section == "fitted":
        assert metron.fitted(case["copies"], case["call_us"]) == case["fitted"]
    elif section == "graph_calls":
        got = _result(lambda: metron.graph_calls(case["env"], case["hip_version"]))
        want = case.get("graph_calls")
        assert got == (
            want if want is not None else {"error": case["raises"], "message": case["message"]}
        )
    elif section == "graph_replays_ok":
        args = (case["env"], case["hip_version"], case["archs"])
        got = _result(lambda: metron.graph_replays_ok(*args))
        want = case.get("ok")
        assert got == (
            want if want is not None else {"error": case["raises"], "message": case["message"]}
        )
    elif section == "most":
        args = (case["budget"], case["ranks"], case["arms"], case["cold"])
        got = _result(lambda: metron.most(*args))
        want = case.get("most")
        assert got == (
            want if want is not None else {"error": case["raises"], "message": case["message"]}
        )
    else:
        assert metron.calls(case["copies"], case["sweep_us"], case["most"]) == case["calls"]


# --- nouns.json --------------------------------------------------------------------------


@pytest.mark.parametrize(PARAMS, cases("metron", "nouns.json", "cases"))
def test_nouns(vector, case_id, case, doc):
    import metron

    what, want = case["what"], case["result"]
    if what == "Point":
        got = _result(lambda: _point_case(case["case"]) and "ok")
    elif what == "Arm":
        name = "" if case["case"] == "empty name" else "a"

        def made():
            owned, read = _owned_of(case), tuple(case["read"])
            return metron.Arm(name, lambda a: None, owned=owned, read=read) and "ok"

        got = _result(made)
    elif what == "State":
        code = case["case"].removeprefix("code ")
        got = _result(lambda: metron.State(code=code) and "ok")
    elif what == "State.said":
        got = metron.State(warm={"y", "x"}, code="cold").said()
    elif what == "race refusal":
        x, y = _xy()
        p = metron.Point("p", {"x": x}, {"y": y})
        arms = [metron.Arm(n, lambda a: None) for n in case["arms"]]
        state = metron.State(warm=frozenset(case["warm"]))
        from metron import _race

        got = _result(lambda: _race._refuse(arms, p, state) or "ok")
    else:
        got = metron.Arm("a", lambda a: None).unchecked().checked
    assert got == want


# --- hygiene.json ------------------------------------------------------------------------


def test_hygiene():
    from metron import hygiene

    doc = json.loads((data_path("metron", "hygiene.json")).read_text())
    inputs = {c["name"]: _hygiene_tensor(c) for c in doc["cases"]}
    assert hygiene(inputs) == doc["flags"]
    for c in doc["cases"]:
        assert bool(hygiene({c["name"]: inputs[c["name"]]})) == c["flagged"], c["name"]


@pytest.mark.parametrize(PARAMS, cases("metron", "hygiene.json", "cases"))
def test_hygiene_case(vector, case_id, case, doc):
    from metron import hygiene

    assert bool(hygiene({case["name"]: _hygiene_tensor(case)})) == case["flagged"]


# --- layout.json -------------------------------------------------------------------------


@pytest.mark.parametrize(PARAMS, cases("metron", "layout.json", "copies"))
def test_layout_copies(vector, case_id, case, doc):
    torch = _torch()
    from metron._memory import Copies

    t = case["tensor"]
    shape, dtype = tuple(t["shape"]), getattr(torch, t["dtype"])
    tensor = (
        torch.zeros(shape, dtype=dtype)
        if case["io"] == "in"
        else torch.empty(shape, dtype=dtype, device="meta")
    )
    c = Copies(_host(), 0, "t", tensor, case["io"], 3)
    assert (c.nbytes, c.stride) == (case["nbytes"], case["stride"])
    c.restore()
    for i in range(3):
        row = c.row(i)
        assert bool((row[c.nbytes :] == case["gap_byte"]).all())
        first = int(row[0]) if c.nbytes else None
        assert first == case["first_byte"]


def test_layout_copies_refusal():
    from metron._memory import stride_of

    case = json.loads((data_path("metron", "layout.json")).read_text())["copies_refusal"]
    nbytes = int(re.search(r"(\d+) bytes", case["result"]["message"]).group(1))
    with pytest.raises(ValueError) as e:
        stride_of("huge", nbytes)
    assert str(e.value) == case["result"]["message"]


@pytest.mark.parametrize(PARAMS, cases("metron", "layout.json", "owned"))
def test_layout_owned(vector, case_id, case, doc):
    _torch()
    from metron._memory import GAP, Pools

    pools = Pools(_host(), 0, _arms_of(case))
    at = [
        {"arm": a, "name": n, "kind": k, "offset": o, "nbytes": b} for a, n, k, o, b in pools.places
    ]
    assert at == case["at"]
    assert {k: size for k, (_, size) in pools.pools.items()} == case["pools"]
    for kind, (buf, size) in pools.pools.items():
        assert _rle(buf[:size], words=kind.startswith("gens")) == case["contents"][kind], kind
        assert _rle(buf[size : size + GAP]) == case["gap"][kind], kind


# --- judging.json ------------------------------------------------------------------------


@pytest.mark.parametrize(PARAMS, cases("metron", "judging.json", "cases"))
def test_judging(vector, case_id, case, doc):
    lay, arm = _judging_layout(case)
    _act(lay, case["case"])
    assert lay.judge(arm) == case["wrong"]
    assert lay.check_calls == case["check_calls"]
    if "again_same_outputs" in case:
        lay.check_calls = 0
        assert lay.judge(arm) == case["again_same_outputs"]["wrong"]
        assert lay.check_calls == case["again_same_outputs"]["check_calls"]


# --- rounds.json -------------------------------------------------------------------------


def _close(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)


def _round_stats(case, k=None):
    from metron import _race

    raw = {n: xs[:k] for n, xs in case["raw"].items()}
    ranks_raw = {n: [xs] for n, xs in raw.items()}
    return _race.stats_of(raw, ranks_raw, case["reference"], case["cold"])


@pytest.mark.parametrize(PARAMS, cases("metron", "rounds.json", "cases"))
def test_rounds(vector, case_id, case, doc):
    from metron import stats
    from metron._memory import MIN_ROUNDS

    prelude, empty, adjusted, _ranks_us, flag = _round_stats(case)
    assert _close(prelude, case["prelude_us"]) and _close(empty, case["empty_us"])
    assert ([flag] if flag else []) == case.get("flags", [])
    for name, want in case["arms"].items():
        low, mid, high = stats.median_ci(adjusted[name])
        assert _close(adjusted[name], want["times"]), name
        assert _close(mid, want["us"]) and _close([low, high], want["ci"]), name
    ref = case["reference"]
    got = {}
    if ref in adjusted:
        for name in adjusted:
            if name != ref:
                ratio, ci, said = stats.ratio_verdict(adjusted[name], adjusted[ref], case["within"])
                got[name] = {"ratio": ratio, "ci": list(ci), "is": said}
    assert got.keys() == case["verdicts"].keys()
    for name, v in case["verdicts"].items():
        assert _close(got[name]["ratio"], v["ratio"]) and _close(got[name]["ci"], v["ci"])
        assert got[name]["is"] == v["is"]
    n = len(next(iter(case["raw"].values())))
    after = None
    for k in range(MIN_ROUNDS, n + 1):
        if stats.settled(_round_stats(case, k)[2], ref, case["within"]):
            after = k
            break
    assert after == case["settled_after"]
    assert stats.settled(adjusted, ref, case["within"]) == case["settled_final"]


@pytest.mark.parametrize(PARAMS, cases_at("metron", "rounds.json", "launch_bound", "cases"))
def test_launch_bound(vector, case_id, case, doc):
    """the plain path's launch-bound flag."""
    from metron import _race

    assert (
        _race.launch_bound(case["arm"], case["raw"], case["host_us"], case["within"])
        == (case["flag"])
    )


def test_rounds_shuffle_seed0():
    """Seeded shuffle orders for four entries."""
    import random

    from metron import _race

    want = json.loads((data_path("metron", "rounds.json")).read_text())["shuffle_seed0"]
    rng = random.Random(0)
    entries = _race.entries(["arm0", "arm1"], True)
    got = []
    for _ in want:
        this = list(entries)
        rng.shuffle(this)
        got.append(this)
    assert got == want


# --- record.json -------------------------------------------------------------------------


def _record_race():
    from metron import ArmResult, Race

    doc = json.loads((data_path("metron", "record.json")).read_text())
    f = doc["race_fields"]
    first = json.loads(doc["lines"][0])
    arms = {
        n: ArmResult(
            us=a["us"],
            ci=None if a["ci"] is None else tuple(a["ci"]),
            times=a["times"],
            wrong=a["wrong"],
            checked=a["checked"],
            fingerprint=a["fingerprint"],
            ranks_us=a["ranks_us"],
        )
        for n, a in f["arms"].items()
    }
    race = Race(
        point=f["point"],
        state=f["state"],
        arms=arms,
        status=f["status"],
        rounds=f["rounds"],
        conditions=first["conditions"],
        flags=first["flags"],
        device=first["device"],
        validators=first["validators"],
        phases_s=first["phases_s"],
        wall_s=1.3,
    )
    return doc, race


@pytest.mark.parametrize(PARAMS, cases("metron", "record.json"))
def test_record(vector, case_id, case, doc, tmp_path):
    import metron
    from metron import record

    key = case_id.split("[")[0]
    data, race = _record_race()
    case = data[key]  # the whole value: a sub-case checks it all
    if key == "schema":
        assert case == record.SCHEMA
    elif key == "race_fields":
        assert (race.point, race.status, race.rounds, list(race.arms)) == (
            case["point"], case["status"], case["rounds"], list(case["arms"])
        )  # fmt: skip
    elif key == "lines":
        path = tmp_path / "r.jsonl"
        race.save(path)
        stamp = r'"time": "[^"]*"'
        got = [re.sub(stamp, '"time": "#TIME#"', line) for line in path.read_text().splitlines()]
        assert got == data["lines"]
    elif key == "verdicts":
        got = {
            n: {"ratio": v.ratio, "ci": list(v.ci), "is": v.verdict}
            for n, v in race.verdicts.items()
        }
        assert got == case
    elif key == "ranked":
        assert race.ranked() == case
    elif key == "versus_error":
        with pytest.raises(ValueError) as e:
            race.versus("wrong", "ref")
        assert str(e.value) == case["message"]
    else:
        path = tmp_path / "r.jsonl"
        path.write_text("\n".join(case["lines"]) + "\n")
        got = [[v, [rec["arm"] for rec in recs]] for v, recs in metron.read(path)]
        assert got == case["groups"]


# --- kernels.json ------------------------------------------------------------------------


def _compare_rows(case):
    """(base: copies of a random row with the case's changes, ref: the row)."""
    import numpy as np

    rng = np.random.default_rng(case["stride"] + case["copies"])
    ref = rng.integers(0, 256, case["stride"], dtype=np.uint8)
    base = np.tile(ref, case["copies"])
    for copy, byte in case["changes"]:
        base[copy * case["stride"] + byte] ^= 0x40
    return base, ref


@pytest.mark.parametrize(PARAMS, cases("metron", "kernels.json"))
def test_kernels(vector, case_id, case, doc):
    import hashlib

    from metron import _kernels

    key = case_id.split("[")[0]
    if key == "chunk":
        assert _kernels.kept("gfx1201", "differ")[0]["meta"]["chunk"] == case
    elif key == "kernels":
        name = case_id.split("[")[1].split("]")[0]
        want = doc["kernels"][name]
        desc, co = _kernels.kept("gfx1201", name)
        assert desc == want["description"]
        want_co = {k: v for k, v in want["code_object"].items() if k != "derived"}
        assert {"sha256": hashlib.sha256(co).hexdigest(), "bytes": len(co)} == want_co
    else:
        torch = _torch()
        base, ref = _compare_rows(case)
        flags = torch.zeros(4 * case["copies"], dtype=torch.uint8)
        _host().differ(0, torch.from_numpy(base), torch.from_numpy(ref), case["stride"],
                       case["copies"], flags)  # fmt: skip
        got = flags.view(torch.int32).tolist()
        assert got == case["flags"]
        assert [i for i, f in enumerate(got) if f] == case["flagged"]


@pytest.mark.parametrize("n", [6, 7, 10, 20, 50, 101])
def test_rank_is_the_largest_k_within_the_tail(n):
    """rank(n) is the largest k with P(B < k) <= 0.025, B ~ Binomial(n, 1/2)."""
    from math import comb

    from metron import stats

    def below(k):
        return sum(comb(n, i) for i in range(k)) / 2**n

    k = stats.rank(n)
    assert below(k) <= 0.025 < below(k + 1)
