"""Golden checks of library declarations, generated streams and rendering."""

import re

import numpy as np
import pytest

from tests import notation
from tests.golden import cases

PARAMS = ("vector", "case_id", "case", "doc")
_DEFAULTS = {
    "lo": None,
    "hi": None,
    "divisor": False,
    "multiple": 1,
    "owned": None,
    "library": None,
    "mem": None,
    "ranks": None,
}


def _message(e) -> str:
    return getattr(e, "msg", None) if hasattr(e, "msg") else (e.args[0] if e.args else "")


def _records(args: dict) -> list:
    """An argument block's records with the defaults left out."""
    from thin.dsl import Args

    block = Args(**args)
    return [
        {k: v for k, v in block[n].items() if _DEFAULTS.get(k, ...) != v} for n in block.names()
    ]


def _kw(kw: dict) -> dict:
    """A declaration's keywords, a JSON list tag back as its tuple."""
    out = dict(kw)
    if isinstance(out.get("tag"), list):
        out["tag"] = tuple(out["tag"])
    return out


def _call(section: str, case: dict):
    from thin.lib import Laps, laps, xwg

    call = case["call"]
    if call.startswith("Laps(['a'], wgsize=64, on=False)"):
        return Laps(["a"], wgsize=64, on=False).decode(b"", workgroups=1)
    if "uint8 array of 252" in call:
        return Laps(["a"], 64, keep="sum").decode(np.zeros(252, np.uint8), workgroups=1)
    if call == "Counts(wgsize=64).sizes(0)":
        return xwg.Counts(wgsize=64).sizes(0)
    make = {"Laps": laps.Laps, "Gens": xwg.Gens, "Slots": xwg.Slots, "Counts": xwg.Counts}[call]
    return make(**_kw(case["kw"]))


@pytest.mark.parametrize(
    PARAMS,
    [p for s in ("laps", "gens", "slots", "counts") for p in cases("lib", "declarations.json", s)],
)
def test_declaration(vector, case_id, case, doc):
    section = case_id.split("[")[0]
    if "error" in case:
        with pytest.raises(Exception) as caught:
            _call(section, case)
        assert type(caught.value).__name__ == case["error"]["class"]
        assert _message(caught.value) == case["error"]["message"]
        return
    decl = _call(section, case)
    assert _records(decl.args()) == case["args"]
    if section == "laps":
        assert (decl.waves, decl.rows) == (case["waves"], case["rows"])
        assert decl.sizes(case["workgroups"]) == case["sizes"]
    elif section == "gens":
        assert decl.waves == case["waves"]
        assert decl.sizes() == case["sizes"]
    elif section == "slots":
        for f in ("kind", "stride", "flags_at", "lane_dwords", "bits", "at"):
            if f in case:
                assert getattr(decl, f) == case[f], f
        assert decl.bytes() == case["bytes"]
        for f in ("mask", "keep"):
            if f in case:
                assert hex(getattr(decl, f)) == case[f]
        assert decl.sizes() == case["sizes"]
    else:
        assert decl.sizes(3) == case["sizes(3)"]


@pytest.mark.parametrize(PARAMS, cases("lib", "tags.json", "cases"))
def test_tags(vector, case_id, case, doc):
    from thin.lib.xwg import GEN_START, bits_tag, full_tag

    g = (GEN_START + case["launch"]) % 2**32
    g1 = (g + 1) % 2**32
    assert (g, g1) == (case["g"], case["g1"])
    assert full_tag(g1) == case["tag"]
    assert full_tag(g1, ahead=1) == case["tag_ahead1"]
    for key, want in case.items():
        if key.startswith("tag_bits("):
            k, at = map(int, key[len("tag_bits(") : -1].split(","))
            assert hex(bits_tag(g1, k, at)) == want, key


@pytest.mark.parametrize(PARAMS, cases("lib", "counts.json", "cases"))
def test_counts(vector, case_id, case, doc):
    from thin.lib.xwg import arrival_fields, leave_mine, reset_amount

    word, n, saw = 0, case["n"], {}
    for step in case["trace"]:
        op, who = step["op"], step["who"]
        if op == "arrive":
            old = word
            word = (word + (1 << 16)) % 2**32
            last, left = arrival_fields(old, n)
            saw[who] = old
            assert step["returns"] == {"old": old, "last": last, "left": left}
        elif op == "leave":
            old = word
            word = (word + 1) % 2**32
            mine = leave_mine(old, n)
            assert step["returns"] == {"old": old, "mine": mine}
            assert step["taken_back"] == bool(mine)
            if mine:
                word = (word + 0xFFFF_FFFF) % 2**32
        else:
            word = (word + reset_amount(saw[who])) % 2**32
        assert word == step["word"], step
    assert word == case["end_word"] == 0


@pytest.mark.parametrize(PARAMS, cases("lib", "laps_decode.json", "cases"))
def test_laps_decode(vector, case_id, case, doc):
    from thin.lib import Laps

    decl = Laps(**case["kw"])
    words = np.zeros(case["workgroups"] * decl.waves * decl.rows * 32, np.uint32)
    for at, v in case["words"].items():
        words[int(at)] = v
    t = decl.decode(words.view(np.uint8), case["workgroups"])
    assert t.rounds == case["rounds"]
    mhz = [[None if m is None else round(m, 6) for m in row] for row in t.clock_mhz]
    assert mhz == case["clock_mhz"]
    assert t.laps == case["laps"]
    if "phases" in case:
        assert t.phases() == case["phases"]
    else:
        with pytest.raises(ValueError, match="keep sum: the totals are the laps"):
            t.phases()
    assert t.summary() == case["summary"]


def _render_message(e) -> str:
    """A render Diag's text with the library's own line as `<module>`."""
    first = str(e).split("\n")[0]
    return re.sub(r"^(xwg|comm|laps|folds)\.py:\d+:", r"<\1>:", first)


@pytest.mark.parametrize(PARAMS, cases("lib", "gfx1201.json", "cases"))
def test_gfx1201(vector, case_id, case, doc):
    from thin import codegen

    if "error" in case:
        with pytest.raises(Exception) as caught:
            notation.run_snippet(case["src"], library=True)
        assert type(caught.value).__name__ == case["error"]["class"]
        assert _message(caught.value) == case["error"]["message"]
        return
    b = notation.run_snippet(case["src"], library=True)
    assert notation.built(b) == case["built"]
    assert notation.values(b) == case["values"]
    assert notation.records(b) == case.get("records", {})
    if "render" in case:
        assert codegen.render(b) == case["render"]
    else:
        with pytest.raises(Exception) as caught:
            codegen.render(b)
        assert type(caught.value).__name__ == case["render_error"]["class"]
        assert _render_message(caught.value) == case["render_error"]["message"]


def test_gfx1201_supplies_the_hooks_target_md_lists():
    """gfx1201's values, None where it supplies none yet."""
    from thin.target import targeting

    with targeting("gfx1201") as t:
        assert t.lib_hooks == {
            "lane_granule": "32",
            "core_clock": "halves s_getreg_b32 63517 63518",
            "real_clock": "pair s_sendmsg_rtn_b64 131",
            "real_clock_mhz": "100",
            "coherent": "SCOPE_DEV",
            "system": "SCOPE_SYS",
            "returning_atomic_add": "global_atomic_add_u32",
            "atomic_return": "TH_ATOMIC_RETURN",
            "scalar_add": "s_add_co_u32",
            "mask_shr": "s_lshr_b32",
            "sleep": "s_sleep",
        }


def test_the_library_takes_every_spelling_from_the_target(monkeypatch):
    """Sizes, spellings and policies are the target's hooks; a missing one is
    refused where the library first needs it, naming the hook."""
    from thin.lib import _hooks, xwg
    from thin.target import target

    hooks = dict(target().lib_hooks)
    monkeypatch.setattr(target(), "lib_hooks", {**hooks, "lane_granule": "16"})
    assert xwg.Gens(per="member", words=1, wgsize=64).waves == 4
    monkeypatch.setattr(target(), "lib_hooks", {**hooks, "coherent": None})
    with pytest.raises(ValueError, match="gfx1201 supplies no coherent hook"):
        _hooks.coherent()


def test_the_library_uses_only_the_public_language():
    """every library module imports the language's public names, the
    package root, the current target's hooks and its own modules only (numpy
    for the host decode)."""
    import ast
    from pathlib import Path

    import thin
    import thin.dsl
    import thin.lib

    public = set(thin.dsl.__all__)
    for path in sorted(Path(thin.lib.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                mod, names = node.module, {a.name for a in node.names}
                if mod == "thin.dsl":
                    assert names <= public, (path.name, names - public)
                elif mod == "thin":
                    assert names <= set(thin.__all__), (path.name, names)
                elif mod == "thin.lib":
                    assert all(n in thin.lib.__all__ or n == "_hooks" for n in names), path.name
                elif mod == "thin.target":
                    assert names == {"target"}, (path.name, names)
                else:
                    assert mod == "__future__" or mod.startswith("thin.lib."), (path.name, mod)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name == "numpy", (path.name, a.name)


def test_decode_reads_the_bytes_of_any_buffer():
    """A u32 array, its byte view and a bytes object decode the same records."""
    from thin.lib import Laps

    decl = Laps(["a"], 32, keep="sum")
    words = np.zeros(32, np.uint32)
    words[0] = 0x0102_0305
    for data in (words, words.view(np.uint8), words.tobytes(), bytearray(words.tobytes())):
        assert decl.decode(data, 1).rounds == [[0x0102_0305]]


def test_ki12_count_fields_carry_past_65535_leaves():
    """the 65536th leave carries into the arrivals' field; nothing checks."""
    from thin.lib.xwg import arrival_fields

    word = 0xFFFF + 1  # 65536 leaves read as one arrival and no leave
    assert arrival_fields(word, 2) == (1, 0)


def test_a_summed_lap_never_taken_says_so():
    from thin.lib import Laps

    decl = Laps(["a", "b"], 32, keep="sum")
    words = np.zeros(decl.waves * decl.rows * 32, np.uint32)
    words[0], words[1], words[3], words[5], words[7] = 1, 10, 100, 110, 2100
    words[9] = 500  # a's total; b's stays 0
    lines = decl.decode(words.view(np.uint8), 1).summary().splitlines()
    assert lines[0].endswith("clock 2000 MHz (waves 2000-2000)")
    assert lines[1].split()[:3] == ["a", "500", "clk"]
    assert lines[2].split() == ["b", "never", "taken"]


def test_an_empty_fold_is_refused():
    from thin.lib import bf16_sum

    with pytest.raises(ValueError, match="bf16_sum: no words to sum"):
        bf16_sum(None, [])


def test_the_off_laps_check_their_names():
    from thin.lib import Laps

    off = Laps(["a"], 64, on=False).begin(None, None)
    off.lap("a")
    off.record()
    off.end()
    with pytest.raises(ValueError, match=r"lap 'not declared': not one of \['a'\]"):
        off.lap("not declared")


def test_a_short_span_gives_its_clocks_error():
    """A span of 8 ticks of the 100 MHz clock is good to 1 in 8."""
    from thin.lib import Laps

    decl = Laps(["a"], 32, keep="sum")
    words = np.zeros(decl.waves * decl.rows * 32, np.uint32)
    words[0], words[1], words[3], words[5], words[7] = 1, 10, 100, 18, 260
    lines = decl.decode(words.view(np.uint8), 1).summary().splitlines()
    assert lines[0].endswith("clock 2000 MHz (waves 2000-2000, short spans: ±12%)"), lines[0]


def test_the_clock_pools_the_waves_spans():
    """Waves of 2 and 1000 ticks: the pooled clock weighs each by its span."""
    from thin.lib import Laps

    decl = Laps(["a"], 64, keep="sum")
    words = np.zeros(decl.waves * decl.rows * 32, np.uint32)
    for w, (ticks, cycles) in enumerate([(2, 90), (1000, 20_000)]):
        row = w * 32
        words[row], words[row + 5], words[row + 7] = 1, ticks, cycles
    t = decl.decode(words.view(np.uint8), 1)
    assert t.clock_mhz == [[4500.0, 2000.0]]
    assert t.clock == pytest.approx((90 + 20_000) / 1002 * 100)
    assert "clock 2005 MHz (waves 2000-4500, short spans: ±50%)" in t.summary()
