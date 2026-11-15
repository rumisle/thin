"""The compiler adapter on example.vadd.

Every kernel is built for gfx1201. These need thin (the adapter's tests).
"""

from __future__ import annotations

import pytest

from tests.golden import cases
from tools.golden.metron_inputs import _arms, _refusal, _tensor_said

pytest.importorskip("thin")
pytest.importorskip("torch")

PARAMS = ("vector", "case_id", "case", "doc")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


@pytest.mark.parametrize(PARAMS, cases("metron", "adapter.json", "points"))
def test_points(vector, case_id, case, doc):
    import torch

    import metron.thin as mn
    from examples.vadd_op import VAdd

    op = VAdd(n=int(case["op"].split("=")[1].rstrip("}")))
    p = mn.point(op, case["seed"], case["dist"])
    assert p.key == case["key"]
    assert {n: _tensor_said(t) for n, t in p.inputs.items()} == case["inputs"]
    assert {n: _tensor_said(t) for n, t in p.outputs.items()} == case["outputs"]
    right = {"z": p.inputs["x"] + p.inputs["y"]}
    assert p.check(p.inputs, right) == case["check_right"]
    assert p.check(p.inputs, {"z": torch.zeros_like(right["z"])}) == case["check_wrong"]


@pytest.mark.parametrize(PARAMS, cases("metron", "adapter.json", "arms"))
def test_arms(vector, case_id, case, doc):
    arms = _arms(case)
    arm = next(a for a in arms if a.name == case["name"])
    assert {n: list(v) for n, v in arm.owned.items()} == case["owned"]
    assert (arm.checked, list(arm.read)) == (case["checked"], case["read"])
    assert len(arm.fingerprint.split(" ")) == case["fingerprint_words"]


@pytest.mark.parametrize(PARAMS, cases("metron", "adapter.json", "refusals"))
def test_refusals(vector, case_id, case, doc):
    want = case["result"]
    with pytest.raises(Exception) as e:
        _refusal(case["case"])
    assert type(e.value).__name__ == want["error"]
    assert str(e.value) == want["message"]
