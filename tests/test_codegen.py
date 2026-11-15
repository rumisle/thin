"""Golden checks of DSL-built code-generation artifacts."""

import hashlib
import re

import pytest

from tests import notation
from tests.golden import cases

PARAMS = ("vector", "case_id", "case", "doc")


@pytest.mark.parametrize(PARAMS, cases("codegen", "kernels.json", "kernels"))
def test_kernel(vector, case_id, case, doc):
    from thin import codegen, formats

    kernel, artifact = notation.compile_case(case)
    assert artifact.text == case["s"]
    assert artifact.manifest.line == case["manifest_line"] + "\n"
    assert artifact.manifest.record() == {**_EMPTY, **case["manifest"]}
    assert _masked(artifact.summary) == case["summary"]
    assert list(kernel.pass_times) == case["pass_times_keys"]
    image = codegen.code_object(artifact.text)
    want = case["code_object"]
    assert hashlib.sha256(image).hexdigest() == want["sha256"]
    assert len(image) == want["bytes"]
    obj = formats.read(image)
    assert obj.text.hex() == want["text_hex"]
    assert obj.descriptor.hex() == want["descriptor_hex"]
    assert obj.e_flags == want["e_flags"]
    assert dict(formats.code(image).digests) == want["digests"]


_EMPTY = {"phases": {}, "max_grid": [], "needs": {}}
"""The record's fields the line leaves out when empty."""


def _masked(text: str) -> str:
    """The text with the placement's seconds masked, as the vectors write them."""
    return re.sub(r"\d+\.\d\d s", "#.## s", text)


@pytest.mark.parametrize(PARAMS, cases("codegen", "refusals.json", "cases"))
def test_refusal(vector, case_id, case, doc):
    if "ok_of" in case:
        scope = notation.run_call(case)
        assert eval(case["ok_of"], scope) == case["ok"]
        return
    with pytest.raises(Exception) as caught:
        notation.run_call(case)
    assert type(caught.value).__name__ == case["refused"]
    assert _masked(str(caught.value)) == case["message"]


@pytest.mark.parametrize(PARAMS, cases("codegen", "abi.json", "args"))
def test_abi_args(vector, case_id, case, doc):
    """each declaration, evaluated with the DSL's names, lays out as given."""
    from thin import dsl
    from thin.target import targeting

    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    with targeting("gfx1201"):
        if "ok" in case:
            block = eval(case["src"], scope)
            got = {"size": block.size, "args": [dict(block[n]) for n in block.names()]}
            assert got == case["ok"]
        else:
            import builtins

            with pytest.raises(getattr(builtins, case["refused"])) as e:
                eval(case["src"], scope)
            assert str(e.value) == case["message"]
