"""Golden DSL tests on a fresh gfx1201 Builder.

Construction is checked separately from full-pipeline renders and refusals."""

import pytest

from tests import notation
from tests.golden import cases

PARAMS = ("vector", "case_id", "case", "doc")
FILES = [
    "values.json",
    "expr.json",
    "maker.json",
    "builder.json",
    "flow.json",
    "lanes.json",
    "waits.json",
    "entry.json",
    "lds.json",
]


def _snippets():
    return [param for file in FILES for param in cases("dsl", file, "cases")]


@pytest.mark.parametrize(PARAMS, _snippets())
def test_snippet(vector, case_id, case, doc):
    if "error" in case:
        with pytest.raises(Exception) as caught:
            notation.run_snippet(case["src"])
        assert type(caught.value).__name__ == case["error"]["class"]
        assert _message(caught.value) == case["error"]["message"]
        return
    b = notation.run_snippet(case["src"])
    assert notation.built(b) == case["built"]
    assert notation.values(b) == case["values"]
    assert notation.records(b) == case.get("records", {})


def _message(e) -> str:
    """The refusal's message: a Diag's own, else the exception's text."""
    return getattr(e, "msg", None) if hasattr(e, "msg") else (e.args[0] if e.args else "")


_DEFAULTS = {
    "lo": None,
    "hi": None,
    "divisor": False,
    "multiple": 1,
    "owned": None,
    "library": None,
}
_DEFAULTS.update(mem=None, ranks=None)


@pytest.mark.parametrize(PARAMS, cases("dsl", "entry.json", "args"))
def test_argument_block(vector, case_id, case, doc):
    from thin import dsl

    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    if "error" in case:
        with pytest.raises(Exception) as caught:
            eval(case["src"], scope)
        assert type(caught.value).__name__ == case["error"]["class"]
        assert _message(caught.value) == case["error"]["message"]
        return
    block = eval(case["src"], scope)
    assert block.size == case["size"]
    assert block.own() == case["own"]
    shown = [
        {k: v for k, v in block[n].items() if _DEFAULTS.get(k, ...) != v} for n in block.names()
    ]
    assert shown == case["args"]


RENDER_DEVIATIONS: dict = {}


@pytest.mark.parametrize(
    PARAMS, [p for p in _snippets() if {"render", "render_error"} & set(p.values[2])]
)
def test_render(vector, case_id, case, doc):
    """The pipeline's output for each snippet: its lines, or its refusal."""
    from thin import codegen

    if "error" in case:
        pytest.skip("the snippet itself is refused")
    if (vector, case_id) in RENDER_DEVIATIONS:
        pytest.xfail(RENDER_DEVIATIONS[(vector, case_id)])
    b = notation.run_snippet(case["src"])
    if "render_error" in case:
        with pytest.raises(Exception) as caught:
            codegen.render(b)
        want = case["render_error"]
        # The vectors ran each snippet as a file named kernel.py; its refusal's
        # message is the first line of the error's text, its location included.
        first = str(caught.value).split("\n")[0].replace("<string>", "kernel.py")
        assert {"class": type(caught.value).__name__, "message": first} == want
        return
    assert codegen.render(b) == case["render"]
