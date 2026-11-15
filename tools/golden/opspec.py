"""Executable op-layer scenarios: declarations, implementations, check runs and groups.

Each file is a document of Python texts (a setup, then calls) written in
``opspec_spec``, ``opspec_impl``, ``opspec_runner`` and ``opspec_group``. The
expected outcome of every call is not written there: this module replays the
texts the way tests/test_opspec.py does (one namespace a file, the public
opspec names bound, the same step order) and records what thin returned or
raised. Torch is required: tensors given as CPU torch tensors are cases too.
"""

from __future__ import annotations

import builtins
import json
import re

from tools.golden import encode

BASE = "tests/data/opspec/"
FILES = ("spec.json", "implement.json", "runner.json", "group.json")
OUTPUTS = tuple(BASE + name for name in FILES)

CALLS = (("call", None), ("lowered_call", "lowered"), ("candidates_call", "candidates"))
SKIP = {"ok", "calls", "setup", "teardown", "other_setup"}
PRIVATE = ("_effect", "_log")
"""Builder-only keys: a call that sets no r, a call whose LOG is kept."""

PATH = re.compile(r"\((?:/|<test>/)[^)]*\)")


def case(text, *, effect=False, log=False, **facts) -> dict:
    """A call node: facts first (documentation), then the call; outcomes are filled by replay."""
    node = dict(facts)
    node["call"] = text
    if effect:
        node["_effect"] = True
    if log:
        node["_log"] = True
    return node


def steps(node, path=""):
    """("exec", text) and ("case", node, key, path) in the order the replay test runs them."""
    if isinstance(node, list):
        for i, item in enumerate(node):
            yield from steps(item, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("setup"), str) and ("implementations" in node or "teardown" in node):
        yield ("exec", node["setup"])
    for key, _ in CALLS:
        if isinstance(node.get(key), str):
            yield ("case", node, key, path)
    for key, value in list(node.items()):
        if key == "other_setup":
            yield ("exec", value)
        elif key not in SKIP and isinstance(value, (list, dict)):
            yield from steps(value, f"{path}.{key}" if path else key)
    if isinstance(node.get("teardown"), str):
        yield ("exec", node["teardown"])


def namespace() -> dict:
    """The public names a document's texts see."""
    import numpy as np
    import torch

    import thin.opspec as o
    from thin.runtime import Bound, DeviceFault, LaunchError

    ns = {n: getattr(o, n) for n in o.__all__}
    ns.update(np=np, LaunchError=LaunchError, DeviceFault=DeviceFault, Bound=Bound, torch=torch)
    return ns


def plain(v):
    """A value as JSON holds it."""
    return json.loads(json.dumps(v, default=lambda x: x.item() if hasattr(x, "item") else repr(x)))


def stubbed(fn):
    """Runs fn with the GPU copies and synchronisation replaced by identity."""
    from thin.opspec import _check

    saved = (_check.to_device, _check.to_host, _check.wait)
    _check.to_device, _check.to_host = (lambda a, t: a), (lambda v, t: v)
    _check.wait = lambda devices=None: None
    try:
        return fn()
    finally:
        _check.to_device, _check.to_host, _check.wait = saved


def run_case(ns, text, filename, gpu_stub):
    """(r or None, (class, message) or None) of one text."""
    code = compile(text, filename, "exec")
    ns.pop("r", None)

    def go():
        """Construct go inputs."""
        exec(code, ns)

    try:
        stubbed(go) if gpu_stub else go()
    except Exception as e:  # the outcome is the refusal, recorded as raised
        got = e.args[0] if isinstance(e, AttributeError | KeyError) and e.args else str(e)
        return None, (type(e).__name__, PATH.sub("(<file>)", got))
    return ns.get("r", ...), None


def replay(doc, *, record=True) -> list:
    """Run a document; record outcomes into it, or (record=False) return mismatches."""
    from thin.opspec import group

    filename = doc["file"]
    ns = namespace()
    emulate = group.emulate
    bad = []
    try:
        exec(compile(doc["setup"], filename, "exec"), ns)
        body = {k: v for k, v in doc.items() if k != "setup"}
        for step in list(steps(body)):
            if step[0] == "exec":
                exec(compile(step[1], filename, "exec"), ns)
                continue
            _, node, key, path = step
            outcome_key = dict(CALLS)[key]
            r, refused = run_case(ns, node[key], filename, node.get("gpu_copies_stubbed"))
            if not record:
                bad += check_outcome(ns, node, outcome_key, r, refused, path, key)
                continue
            out = {} if outcome_key is not None else node
            if refused is not None:
                out["refused"], out["message"] = refused
            elif not node.get("_effect"):
                if r is ...:
                    raise ValueError(f"{filename} {path}.{key}: the text sets no r")
                out["ok"] = plain(r)
            if node.get("_log"):
                node["calls"] = plain(ns.get("LOG"))
            if outcome_key is not None:
                node[outcome_key] = out
    finally:
        group.emulate = emulate
    return bad


def check_outcome(ns, node, outcome_key, r, refused, path, key) -> list:
    """The replay test's verdict on one recorded case, as a list of complaints."""
    want = node if outcome_key is None else node[outcome_key]
    where = f"{path}.{key}"
    if refused is not None:
        if "refused" not in want or refused != (want["refused"], want["message"]):
            return [f"{where}: raised {refused}"]
        return []
    if "refused" in want:
        return [f"{where}: no {want['refused']}"]
    if "ok" in want and plain(None if r is ... else r) != want["ok"]:
        return [f"{where}: r changed"]
    if "calls" in node and plain(ns.get("LOG")) != node["calls"]:
        return [f"{where}: LOG changed"]
    return []


def strip(node):
    """The document without builder-only keys."""
    if isinstance(node, list):
        return [strip(x) for x in node]
    if isinstance(node, dict):
        return {k: strip(v) for k, v in node.items() if k not in PRIVATE}
    return node


def documents() -> dict:
    """File name → the authored document, outcomes not yet recorded."""
    from tools.golden import opspec_group, opspec_impl, opspec_runner, opspec_spec

    return {
        "spec.json": opspec_spec.document(),
        "implement.json": opspec_impl.document(),
        "runner.json": opspec_runner.document(),
        "group.json": opspec_group.document(),
    }


def evaluated() -> dict:
    """File name → (document with outcomes and builder keys, final document)."""
    if builtins.__dict__.get("__thin_opspec_golden_running__"):
        raise RuntimeError("opspec generation is not reentrant")
    out = {}
    for name, doc in documents().items():
        replay(doc)
        final = strip(doc)
        complaints = replay(json.loads(json.dumps(doc)), record=False)
        if complaints:  # a second run must agree: no case depends on the generator's process
            raise ValueError(f"{name}: replay disagrees with the recorded outcomes: {complaints}")
        out[name] = (doc, final)
    return out


def generate():
    """Author the four documents, replay them with thin and record every outcome."""
    return {BASE + name: encode(final) for name, (_, final) in evaluated().items()}
