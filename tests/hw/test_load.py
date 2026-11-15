"""Load thin-generated gfx1201 code objects and resolve their kernels."""

from __future__ import annotations

import pytest

from tests import notation
from tests.golden import load

pytestmark = pytest.mark.gfx1201

KERNELS = load("codegen", "kernels.json")["kernels"]


def test_hip_reports_the_arch_by_the_properties_offset():
    """the arch name read at its offset in the properties record."""
    from thin.runtime import hip

    archs, version = hip.gpus()
    assert "gfx1201" in archs, archs
    assert version >= 70000000, version


@pytest.mark.parametrize("case", KERNELS, ids=[c["name"] for c in KERNELS])
def test_every_kernel_of_kernels_json_loads(case):
    from thin import codegen
    from thin.runtime import hip

    _, art = notation.compile_case(case)
    image = codegen.code_object(art.text)
    mod, fn = hip.module(image, case["symbol"], case["name"])
    assert mod.value and fn.value


def test_vadd_loads():
    import thin
    from examples.vadd import build_vadd
    from thin.runtime import hip

    before = thin.use_store(None)
    try:
        k = thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))
    finally:
        thin.use_store(before)
    mod, fn = hip.module(k.co, k.name, k.name)
    assert mod.value and fn.value
