"""Generate authored compiler-pass programs and evaluate every result natively.

``generate()`` never reads a fixture.  The Rust example receives only the input
structures below and computes streams, intervals, reports, counts, placements,
and diagnostics with the same pass APIs as ``crates/thin-passes/tests/passes.rs``.
"""

from __future__ import annotations

import json
import subprocess

from tools.golden import ROOT, encode, no_checked_data
from tools.golden.passes_liveness import lives_plans, races_plans
from tools.golden.passes_programs import (
    _pipeline_body,
    alias_plans,
    checks_plans,
    hazards_plans,
    ranges_plans,
    regalloc_plans,
    solve_plans,
    transfer_plans,
    waits_plans,
)

BASE = "crates/thin-passes/tests/data/passes/"
FILES = ("alias", "checks", "hazards", "lives", "ranges", "regalloc", "waits")
OUTPUTS = tuple(BASE + name + ".json" for name in FILES)


def native(requests):
    """Evaluate a batch using the production pass implementations."""
    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "thin-passes", "--example", "golden_authored"],
        cwd=ROOT,
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    answers = json.loads(proc.stdout)
    if len(answers) != len(requests):
        raise RuntimeError("native example returned a different batch length")
    for request, answer in zip(requests, answers, strict=True):
        if "request_error" in answer:
            raise RuntimeError(f"{request.get('name', request['kind'])}: {answer['request_error']}")
    return answers


def _clean(plan):
    return {k: v for k, v in plan.items() if not k.startswith("_") and k != "refuses"}


def _documents():
    aliases, checks, hazards, waits = alias_plans(), checks_plans(), hazards_plans(), waits_plans()
    ranges, transfers = ranges_plans(), transfer_plans()
    allocs, solves = regalloc_plans(), solve_plans()
    lives, races = lives_plans(), races_plans()

    # Pipeline reconstruction itself is computed through the wait solver, rather
    # than importing a stored final.s body or hand-writing expected wait counts.
    body_answer = native([{"kind": "waits", "input": _pipeline_body()}])[0]
    if body_answer.get("fixpoint") is not True or "error" in body_answer:
        raise RuntimeError(f"could not construct pipeline body: {body_answer}")
    pipeline_body = body_answer["output"]

    records = []
    for p in aliases:
        records.append(
            (
                "alias",
                p,
                {
                    **_clean(p),
                    "kind": "alias",
                    "groups": [[name, spec] for name, spec in sorted(p["groups"])],
                    "source": p["input"],
                },
            )
        )
    for p in checks:
        records.append(
            (
                "checks",
                p,
                {
                    "kind": "checks",
                    "name": p["name"],
                    "input": p["input"],
                    "check": p["check"],
                    "entries": p.get("entries", []),
                },
            )
        )
    for p in hazards:
        records.append(("hazards", p, {"kind": "hazards", "name": p["name"], "input": p["input"]}))
    for p in waits:
        req = {"kind": "waits", "pipeline": p.get("pipeline", False), "input": p["input"]}
        if req["pipeline"]:
            req.update(source=p["source"], body=pipeline_body)
        records.append(("waits", p, req))
    for p in transfers:
        records.append(
            (
                "transfer",
                p,
                {"kind": "transfer", "op": p["op"], "operands": p["operands"], "k": p["k"]},
            )
        )
    for p in ranges:
        records.append(
            ("ranges", p, {"kind": "ranges", "input": p["input"], "launch": p["launch"]})
        )
    for p in solves:
        records.append(("solve", p, {"kind": "solve", **_clean(p)}))
    for p in allocs:
        records.append(("regalloc", p, {"kind": "regalloc", **_clean(p)}))
    for p in lives:
        records.append(
            (
                "lives",
                p,
                {
                    "kind": "lives",
                    "name": p["name"],
                    "listing": p["listing"],
                    "source": p["source"],
                    "table_lines": p["table_lines"],
                },
            )
        )
    for p in races:
        records.append(
            (
                "races",
                p,
                {
                    "kind": "races",
                    "name": p["name"],
                    "listing": p["listing"],
                    "source": p["source"],
                    "wgsize": p["wgsize"],
                    "wave_lanes": p["wave_lanes"],
                    "per_wave_bytes": p["per_wave_bytes"],
                },
            )
        )

    for _, plan, request in records:
        request.setdefault("name", plan["name"])
    answers = native([request for _, _, request in records])
    docs = {
        "alias": {
            "target": "gfx1201",
            "notation": (
                "independently authored alias groups and "
                "waivers; findings are computed by "
                "thin_passes::alias"
            ),
            "cases": [],
        },
        "checks": {
            "target": "gfx1201",
            "notation": "native pass diagnostics; null error means the check accepts the program",
            "cases": [],
        },
        "hazards": {
            "target": "gfx1201",
            "notation": "+ marks inserted lines; delay text is rendered by the native pass",
            "cases": [],
        },
        "waits": {
            "target": "gfx1201",
            "notation": (
                "new source markers and independently "
                "spelled CFGs; counts, uniqueness, waits and "
                "diagnostics are native"
            ),
            "cases": [],
        },
        "lives": {
            "target": "gfx1201",
            "notation": (
                "new P/Q union and 16-element address "
                "programs; four-wave liveness uses "
                "wave_order; races use three waves and "
                "explicit byte maps"
            ),
            "cases": [],
            "races": [],
        },
        "ranges": {
            "target": "gfx1201",
            "transfer": [],
            "notation": (
                "independent interval vectors and launch "
                "facts; all transfer/report/error results "
                "are native"
            ),
            "cases": [],
        },
        "regalloc": {
            "target": "gfx1201",
            "solve": [],
            "notation": (
                "new derived kernels and seeded abstract CFG "
                "facts; native solver derives occupancy and "
                "placement"
            ),
            "cases": [],
        },
    }
    for (kind, plan, _), answer in zip(records, answers, strict=True):
        if kind == "alias":
            docs[kind]["cases"].append(
                {
                    **_clean(plan),
                    "groups": dict(sorted(plan["groups"])),
                    "source": plan["input"],
                    "input": answer["input"],
                    "findings": answer["findings"],
                }
            )
        elif kind == "checks":
            docs[kind]["cases"].append({**_clean(plan), "error": answer["error"]})
        elif kind == "hazards":
            item = {
                **_clean(plan),
                "output": answer["output"],
                "rerun_needs_more": answer["rerun_needs_more"],
            }
            docs[kind]["cases"].append(item)
        elif kind == "waits":
            if not plan.get("pipeline") and answer.get("fixpoint") is not True:
                raise RuntimeError(f"{plan['name']}: non-fixpoint native counts")
            item = _clean(plan)
            if "output" in answer:
                item["output"] = answer["output"]
            if "error" in answer:
                item["error"] = answer["error"]
            if not item.get("pipeline"):
                item["input"] = plan["input"]
                item["counts"] = dict(answer["counts"])
                item["unique"] = answer["unique"]
            else:
                item.pop("input", None)
            docs[kind]["cases"].append(item)
        elif kind == "transfer":
            docs["ranges"]["transfer"].append({**_clean(plan), "result": answer["result"]})
        elif kind == "ranges":
            docs[kind]["cases"].append({**_clean(plan), **answer})
        elif kind == "solve":
            docs["regalloc"]["solve"].append({**_clean(plan), **answer})
        elif kind == "regalloc":
            if answer.get("check_error") is not None:
                raise RuntimeError(
                    f"{plan['name']}: native placement check failed: {answer['check_error']}"
                )
            item = {**_clean(plan)}
            if "error" in answer:
                item["error"] = answer["error"]
            else:
                item.update(
                    placement=dict(answer["placement"]),
                    vgprs_used=answer["vgprs_used"],
                    report=answer["report"],
                )
            docs[kind]["cases"].append(item)
        elif kind == "lives":
            item = _clean(plan)
            if "error" in answer:
                item["error"] = answer["error"]
            else:
                item.update(report=answer["report"], epochs=dict(answer["epochs"]))
            docs[kind]["cases"].append(item)
        elif kind == "races":
            if "proof_error" in answer:
                raise RuntimeError(
                    f"{plan['name']}: proof failed before race check: {answer['proof_error']}"
                )
            docs["lives"]["races"].append({**_clean(plan), "error": answer["error"]})

    # Classification checks guard against accidentally reducing a refusal case
    # to a passing program (or vice versa) while composing fresh source.
    for (kind, plan, _), answer in zip(records, answers, strict=True):
        if "refuses" in plan:
            if kind == "races":
                refused = answer["error"] is not None
            elif kind in ("checks", "ranges", "regalloc", "lives"):
                refused = answer.get("error") is not None
            elif kind == "waits":
                refused = "error" in answer
            else:
                continue
            if refused != plan["refuses"]:
                raise RuntimeError(f"{kind} {plan['name']}: wrong success/refusal class: {answer}")
    return [docs[name] for name in FILES]


def generate() -> dict[str, bytes]:
    """Evaluate and encode the independently authored input plans."""
    with no_checked_data():
        documents = _documents()
    return {path: encode(doc) for path, doc in zip(OUTPUTS, documents, strict=True)}
