"""counter_waits, gfx1201's independent wait check: it
reports nothing on every rendered gfx1201 program, and, on random derived
kernels with waits removed at random, it reports exactly where the emulator
 raises RegisterInFlight."""

from __future__ import annotations

import glob
import json
import random
import re
import struct

import pytest

from tests import notation
from tests.golden import data_path, load


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _vector_renders():
    out = [
        (f"kernels.json {c['name']}", c["s"]) for c in load("codegen", "kernels.json")["kernels"]
    ]
    paths = [
        *sorted(glob.glob(str(data_path("dsl", "*.json")))),
        str(data_path("lib", "gfx1201.json")),
    ]
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        for k, c in enumerate(doc.get("cases", [])):
            r = c.get("render") if isinstance(c, dict) else None
            if r:
                out.append(
                    (f"{path.split('/')[-1]}[{k}]", "\n".join(r) if isinstance(r, list) else r)
                )
    for k, c in enumerate(load("passes", "waits.json")["cases"]):
        if c.get("output"):
            out.append((f"waits.json[{k}]", "\n".join(c["output"])))
    return out


RENDERS = _vector_renders()


@pytest.mark.parametrize(("name", "text"), RENDERS, ids=[n for n, _ in RENDERS])
def test_nothing_on_a_rendered_program(name, text):
    from thin import codegen

    assert codegen.counter_waits(text) == []


def test_nothing_on_the_compiled_kernels_and_metrons():
    import thin
    from metron import maker
    from thin import codegen
    from thin.target import targeting

    for case in load("codegen", "kernels.json")["kernels"]:
        assert codegen.counter_waits(notation.compile_case(case)[1].text) == [], case["name"]
    with targeting("gfx1201"):
        for name, gen in maker.GENERATORS.items():
            assert codegen.counter_waits(thin.compile(gen).asm) == [], name


WAIT = re.compile(r"^\ts_wait_(loadcnt|dscnt|kmcnt) ")


def _emulate(text, man):
    """Runs a gfx1201.s on one wave: x = 0..127 (u32), y 32 dwords; the
    emulator's error message, or None."""
    from thin.emu import Cluster, EmuError, Memory

    m = Memory()
    x = m.alloc("x", 512)
    x.write(0, struct.pack("<128I", *range(128)))
    y = m.alloc("y", 128)
    ka = m.alloc("kernarg", 16, tagged=False)
    ka.write(0, struct.pack("<QQ", x.base, y.base))
    try:
        Cluster(text, man, m, [ka]).run()
    except EmuError as e:
        return type(e).__name__, str(e)
    return None


@pytest.mark.parametrize("seed", range(24))
def test_it_reports_where_the_emulator_raises(seed):
    import thin
    from tests.test_load_invariant import gen_fuzz
    from thin import codegen

    with codegen.using(keep_dead=True):
        try:
            k = thin.compile(gen_fuzz, seed=seed)
        except Exception:
            pytest.skip("a generated shape the compiler refuses")
    rng = random.Random(seed)
    lines = k.asm.split("\n")
    kept = [ln for ln in lines if not (WAIT.match(ln) and rng.random() < 0.5)]
    text = "\n".join(kept)
    found = codegen.counter_waits(text)
    man = {"wgsize": 32, "vgpr": k.manifest.vgpr, "lds": k.manifest.lds}
    got = _emulate(text, man)
    if got is None:
        assert found == [], found
        return
    cls, msg = got
    assert cls == "RegisterInFlight", msg
    ins = re.match(r"(.*?): [vsa]\d+ is still being loaded", msg).group(1)
    lines_with = {int(f.split(":")[0].split()[1]) for f in found}
    texts = {kept[n - 1].split("//")[0].strip() for n in lines_with}
    assert ins.strip() in texts, (msg, found)
