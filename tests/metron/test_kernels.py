"""The own kernels with the compiler: the maker's rebuild
is byte-identical to the kept files, and the copy compare gives
kernels.json's flags on the emulator. These need thin (the maker's tests)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.golden import data_path

pytest.importorskip("thin")


@pytest.fixture
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def test_the_maker_rebuilds_the_kept_kernels_byte_for_byte(no_store):
    from metron import _kernels, maker
    from thin.target import target, target_names

    for name in target_names():
        arch = target(name).name
        for kernel, (co, text) in maker.build(name).items():
            assert (_kernels.DIR / arch / f"{kernel}.co").read_bytes() == co, (arch, kernel)
            assert (_kernels.DIR / arch / f"{kernel}.json").read_text() == text, (arch, kernel)


def test_the_invalidate_is_the_targets_with_its_no_ops(no_store):
    import thin
    from metron import maker
    from thin.target import targeting

    with targeting("gfx1201"):
        k = thin.compile(maker.gen_icache_inv)
    body = [line.split("//")[0].strip() for line in k.asm.splitlines() if line.startswith("\t")]
    body = [line for line in body if line and not line.startswith(".")]
    # After the entry's own moves: the invalidate, its no-ops, the end.
    assert body[-18:] == ["s_icache_inv", *["s_nop 0"] * 16, "s_endpgm"]
    assert "s_icache_inv" not in body[:-18]


def _rows(case):
    rng = np.random.default_rng(case["stride"] + case["copies"])
    ref = rng.integers(0, 256, case["stride"], dtype=np.uint8)
    base = np.tile(ref, case["copies"])
    for copy, byte in case["changes"]:
        base[copy * case["stride"] + byte] ^= 0x40
    return base, ref


COMPARE = json.loads((data_path("metron", "kernels.json")).read_text())["compare"]


@pytest.mark.parametrize(
    "case", COMPARE, ids=[f"stride{c['stride']}x{c['copies']}" for c in COMPARE]
)
def test_the_copy_compare_on_the_emulator(no_store, case):
    """kernels.json `compare`: the kept kernel's source run on the emulator."""
    import thin
    from metron import maker
    from thin.target import targeting

    with targeting("gfx1201"):
        k = thin.compile(maker.gen_differ)
    base, ref = _rows(case)
    flags = np.zeros(case["copies"], np.uint32)
    grid = (-(-case["stride"] // maker.CHUNK), case["copies"])
    assert list(grid) == case["grid"]
    k.bind(
        grid=grid, args={"base": base, "ref": ref, "flags": flags, "stride": case["stride"]}
    ).emulate()
    assert flags.tolist() == case["flags"]
