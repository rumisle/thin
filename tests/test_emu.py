"""The emulator's Python binding over its native core."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.golden import data_path

COPY = """\
\ts_load_b64 s[4:5], s[0:1], 0x0
\ts_load_b64 s[6:7], s[0:1], 0x8
\ts_wait_kmcnt 0
\tv_lshlrev_b32 v1, 2, v0
\tglobal_load_b32 v2, v1, s[4:5]
\ts_wait_loadcnt 0
\tglobal_store_b32 v1, v2, s[6:7]
\ts_endpgm
"""


def test_read_program_and_the_decoded_encoding_agree():
    from thin.emu import Ins, Program, read_program

    p = read_program("\tv_add_f32 v1, -v2, |v3|\n.Lx:\n\ts_endpgm\n")
    assert [i.canon for i in p.ins] == ["v.add.f32", "s.endpgm"] and p.labels == {".Lx": 1}
    assert repr(p.ins[0].args[0].base) == "v1" and p.ins[0].args[1].neg and p.ins[0].args[2].abs
    again = Program.of(p.encoded())
    assert again.ins == p.ins and again.labels == p.labels
    enc = {"op": "v.cndmask.b32", "kind": "valu", "operands": [{"reg": ["vcc", 0, 2]}]}
    assert Ins.of(enc).args[0].base.n == 1  # a lane mask at W = 32


def test_parse_line_and_a_reading_error():
    from thin.emu import parse_line

    assert parse_line("La:") == "La" and parse_line("; a note") is None
    assert parse_line("\ts_nop 0", 7).line == 7
    with pytest.raises(ValueError, match="line 3: operand not understood: `@x`"):
        parse_line("\tv_mov_b32 v0, @x", 3)


def test_claims_as_claims_json_gives_them():
    from thin.emu import (
        assume_claims,
        assume_false,
        held_text,
        undefined_claims,
        uniform_claims,
        user_sgprs,
    )

    doc = json.loads((data_path("emu", "claims.json")).read_text())
    for c in doc["cases"]:
        u = uniform_claims(c["text"])
        got = u if "error" in u else {str(k): [[list(r) for r in v[0]], v[1]] for k, v in u.items()}
        assert got == c["uniform"]
        a = assume_claims(c["text"])
        want = a if "error" in a else {str(k): [list(x) for x in v] for k, v in a.items()}
        assert want == c["assume"]
        undefined = undefined_claims(c["text"])
        got = (
            undefined
            if "error" in undefined
            else {str(key): [list(reg) for reg in regs] for key, regs in undefined.items()}
        )
        assert got == c["undefined"]
        assert user_sgprs(c["text"]) == c["user_sgprs"]
    inputs = doc["message_inputs"]
    assert held_text(inputs["held"]) == doc["messages"]["held_text"]
    assert assume_false(**inputs["sgpr"]) == doc["messages"]["assume_false_sgpr"]
    assert assume_false(**inputs["lane"]) == doc["messages"]["assume_false_lane"]


def test_numbers_as_numbers_json_gives_them():
    import gzip

    from thin.emu import numbers as n

    doc = json.loads(gzip.decompress((data_path("emu", "numbers.json.gz")).read_bytes()))
    for x, want in doc["bf16_rne"][:50]:
        assert n.bf16_round(int(x, 16)) == int(want, 16)


def test_memory_places_buffers_and_refuses_an_unmapped_address():
    from thin.emu import EmuError, Memory

    m = Memory()
    a, b = m.alloc("a", 64), m.alloc("b", 1000)
    assert (a.base, b.base) == (0x1000_FFFF_FF00, 0x1002_FFFF_FE00)
    assert m.resolve(b.base + 5) == (1, 5) and m.index("b") == 1
    b.write(4, b"\x01\x02")
    assert b.read(3, 4) == b"\x00\x01\x02\x00" and m.version[1] == 1
    with pytest.raises(EmuError, match="address 0x7 is in no buffer"):
        m.resolve(7)


def _copy_run(hooks_of=None):
    from thin.emu import Cluster, Memory

    m = Memory()
    x = m.alloc("x", 128)
    x.write(0, np.arange(32, dtype=np.uint32).tobytes())
    y = m.alloc("y", 128)
    k = m.alloc("k", 16, tagged=False)
    k.write(0, np.array([x.base, y.base], dtype=np.uint64).tobytes())
    hooks = hooks_of(m) if hooks_of else None
    c = Cluster(COPY, {"wgsize": 32, "vgpr": 4}, m, [k], hooks=hooks)
    return c.run(), m, y, hooks


def test_a_cluster_runs_a_kernel_text():
    machines, _, y, _ = _copy_run()
    assert np.array_equal(np.frombuffer(y.data, np.uint32), np.arange(32))
    assert machines[0].stats["ins"] == 8 and machines[0].stats["store_bytes"] == 128


def test_the_gemm_checker_records_a_store_of_memory_bytes():
    from thin.emu import gemm

    def hooks(m):
        p = gemm.Problem({"y": gemm.rowmajor_d(128)}, 32, fail_fast=False)
        return gemm.GemmHooks(p, m)

    _, _, _, h = _copy_run(hooks)
    assert h.prob.errors and "y" in h.prob.errors[0]


def test_a_machine_runs_a_workgroup():
    from thin.emu import Memory, load_kernel

    m = Memory()
    machine = load_kernel("\tv_add_nc_u32 v1, v0, 5\n\ts_endpgm\n", 4, 0, 0, m)
    (w,) = machine.run_wg(0, (0, 0), 1)
    assert w.done and list(w.V[1][:4]) == [5, 6, 7, 8]


def test_errors_map_to_their_classes():
    from thin.emu import Cluster, EmuLimit, Memory

    m = Memory()
    k = m.alloc("k", 16, tagged=False)
    text = "\tv_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20\n\ts_endpgm\n"
    with pytest.raises(EmuLimit, match="instruction not emulated: v_swmmac_f32_16x16x32_f16"):
        Cluster(text, {"wgsize": 32, "vgpr": 24}, m, [k]).run()


def test_pc_byte_layout_survives_the_decoded_python_roundtrip():
    from thin.emu import Memory, Program, load_program, read_program

    p = read_program(
        "s_getpc_b64 s[12:13]\ns_mov_b32 s4, 0x12345678\ns_getpc_b64 s[14:15]\ns_endpgm"
    )
    assert p.pc_layout == (((0, 4), (4, 12), (12, 16), (16, 20)), 20)
    again = Program.of(p.encoded())
    assert again.pc_layout == p.pc_layout
    wave = load_program(again, 2, 0, 0, Memory()).run_wg(0, (0, 0), 1)[0]
    assert wave.S[12:16] == [4, 0, 16, 0]


def test_pc_decoded_programs_refuse_missing_or_invalid_byte_layouts():
    from thin.emu import EmuLimit, Memory, Program, load_program, read_program

    encoded = read_program("s_getpc_b64 s[12:13]\ns_endpgm").encoded()
    encoded.pop("pc_layout")
    with pytest.raises(EmuLimit, match="explicit byte-address layout"):
        load_program(Program.of(encoded), 2, 0, 0, Memory()).run_wg(0, (0, 0), 1)
    encoded["pc_layout"] = {"ranges": [[0, 8], [4, 8]], "end": 8}
    with pytest.raises(ValueError, match="ranges overlap"):
        Program.of(encoded)
