"""Golden checks of executable operation declarations and implementations.

Each fixture runs in an isolated namespace with its supplied synthetic filename."""

from __future__ import annotations

import builtins
import functools
import json
import re

import pytest

from tests.golden import load

FILES = ["spec.json", "implement.json", "runner.json", "group.json"]
CALLS = (("call", None), ("lowered_call", "lowered"), ("candidates_call", "candidates"))
SKIP = {"ok", "calls", "setup", "teardown", "other_setup"}


def _steps(node):
    """The file's texts in document order: ("exec", text) or ("case", outcome, text, case)."""
    if isinstance(node, list):
        for item in node:
            yield from _steps(item)
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("setup"), str) and ("implementations" in node or "teardown" in node):
        yield ("exec", node["setup"])  # a scenario's or the group section's (others are notes)
    for key, outcome in CALLS:
        if isinstance(node.get(key), str):
            yield ("case", node if outcome is None else node[outcome], node[key], node)
    for key, value in node.items():
        if key == "other_setup":
            yield ("exec", value)
        elif key not in SKIP and isinstance(value, (list, dict)):
            yield from _steps(value)
    if isinstance(node.get("teardown"), str):
        yield ("exec", node["teardown"])


def _namespace() -> dict:
    import numpy as np

    import thin.opspec as o
    from thin.runtime import Bound, DeviceFault, LaunchError

    ns = {n: getattr(o, n) for n in o.__all__}
    ns.update(np=np, LaunchError=LaunchError, DeviceFault=DeviceFault, Bound=Bound)
    try:
        import torch

        ns["torch"] = torch
    except ImportError:
        pass
    return ns


def _plain(v):
    """r as JSON would hold it (tuples as lists, numpy scalars as numbers)."""
    return json.loads(json.dumps(v, default=lambda x: x.item() if hasattr(x, "item") else repr(x)))


def _stubbed(fn):
    from thin.opspec import _check

    saved = (_check.to_device, _check.to_host, _check.wait)
    _check.to_device, _check.to_host = (lambda a, t: a), (lambda v, t: v)
    _check.wait = lambda devices=None: None
    try:
        return fn()
    finally:
        _check.to_device, _check.to_host, _check.wait = saved


def _outcome(ns, outcome, text, case, filename):
    """None when the case's text gives its outcome, else why not."""
    code = compile(text, filename, "exec")
    ns.pop("r", None)

    def run():
        exec(code, ns)

    try:
        _stubbed(run) if case.get("gpu_copies_stubbed") else run()
    except Exception as e:  # the vector names the exception
        if "refused" not in outcome:
            return f"raised {type(e).__name__}: {e}"
        got = e.args[0] if isinstance(e, AttributeError | KeyError) and e.args else str(e)
        got = re.sub(r"\((?:/|<test>/)[^)]*\)", "(<file>)", got)  # (<file>) masks a source path
        if (type(e).__name__, got) != (outcome["refused"], outcome["message"]):
            want = f"{outcome['refused']}: {outcome['message']!r}"
            return f"raised {type(e).__name__}: {got!r}, want {want}"
        return None
    if "refused" in outcome:
        return f"no {outcome['refused']} ({outcome['message']!r})"
    if "ok" in outcome and _plain(ns.get("r")) != outcome["ok"]:
        return f"r = {_plain(ns.get('r'))!r}, want {outcome['ok']!r}"
    if "calls" in case and _plain(ns.get("LOG")) != case["calls"]:
        return f"LOG = {_plain(ns.get('LOG'))!r}, want {case['calls']!r}"
    return None


@functools.cache
def _replay(file: str) -> tuple:
    """Every case's verdict, in order, from one run of the file's texts."""
    doc = load("opspec", file)
    filename = doc["file"]
    ns = _namespace()
    exec(compile(doc["setup"], filename, "exec"), ns)
    verdicts = []
    for step in _steps({k: v for k, v in doc.items() if k != "setup"}):
        if step[0] == "exec":
            exec(compile(step[1], filename, "exec"), ns)
            continue
        _, outcome, text, case = step
        if "torch" in text and "torch" not in ns:
            verdicts.append((text, "skip"))
            continue
        verdicts.append((text, _outcome(ns, outcome, text, case, filename)))
    return tuple(verdicts)


def _params():
    out = []
    for file in FILES:
        name = f"opspec/{file}"
        marks = []
        count = sum(1 for s in _steps(load("opspec", file)) if s[0] == "case")
        out += [
            pytest.param(name, i, file, None, id=f"{file}:{i}", marks=marks) for i in range(count)
        ]
    return out


@pytest.mark.parametrize(("vector", "case_id", "case", "doc"), _params())
def test_opspec(vector, case_id, case, doc):
    text, why = _replay(case)[case_id]
    if why == "skip":
        pytest.skip("needs torch")
    assert why is None, f"{text}: {why}"


def test_the_namespace_is_the_public_api():
    ns = _namespace()
    assert {"Spec", "group", "check", "LaunchError", "np"} <= set(ns)
    assert not set(ns) & set(dir(builtins)) - {"np"}
