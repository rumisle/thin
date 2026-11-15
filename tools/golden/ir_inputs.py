"""Shared IR verifier input constructors.

Only programs are constructed here; expected verification outcomes live in the
recorded golden data and are evaluated separately by the tests and generator.
"""

from thin import ir
from thin.ir import stream
from thin.target import PHYS


def _base_program(t):
    p = ir.Program(t)
    for name, file, words in (
        ("a", "v", 1),
        ("b", "v", 1),
        ("p", "v", 2),
        ("s", "s", 1),
        ("s2", "s", 2),
    ):
        p.value(file, words, ir.B32, name=name)
    return p


def _instructions(t, name):
    """Construct the instruction sequence for the named verifier rule."""
    p = _base_program(t)
    a, b, pair, s, _s2 = (v[:] if v.n > 1 else v[0] for v in p.values)
    make = ir.Instr
    add = t.form("v_add_nc_u32", 3)
    load = t.form("global_load_b32", 3)
    ds = t.form("ds_load_b32", 2)
    scalar_move = t.form("s_mov_b32", 2)
    branch = t.form("s_cbranch_scc1", 1)
    dpp = t.form("v_mov_b32_dpp", 2)
    dot = t.form("v_dot2_f32_bf16", 4)
    own = t.pseudo
    off, scc, exec_lo = PHYS["off"], PHYS["scc"], PHYS["exec_lo"]

    def dpp_fields(**extra):
        """Construct dpp fields inputs."""
        return {"dpp": "row_shl:3", "row_mask": 15, "bank_mask": 15, **extra}

    programs = {
        "valid_branch_loop": [
            make(scalar_move, (s, ir.Imm(17))),
            stream.label("repeat"),
            make(add, (b, a, ir.Imm(23))),
            make(branch, (ir.BlockRef("repeat"),)),
            make(t.form("s_endpgm", 0), ()),
        ],
        "missing_add_operand": [make(add, (b, a))],
        "scalar_destination": [make(add, (s, b, a))],
        "wide_add_destination": [make(add, (pair, b, a))],
        "load_address_widths": [
            make(load, (b, pair, off)),
            make(load, (a, a, off)),
        ],
        "register_range_overflow": [make(load, (b, ir.VReg(p.values[0], 0, 2), off))],
        "named_vector_destination": [make(add, (exec_lo, b, a))],
        "off_address": [make(load, (b, off, off))],
        "immediate_destination": [make(add, (ir.Imm(19), b, a))],
        "immediate_source": [make(add, (b, a, ir.Imm(19)))],
        "block_source": [make(add, (b, ir.BlockRef("later"), a))],
        "unsupported_modifier": [make(add, (b, a, a), mods=b"\x00\x02\x00")],
        "zero_modifiers": [make(add, (b, a, a), mods=b"\x00\x00\x00")],
        "unresolved_branch": [make(branch, (ir.BlockRef("nowhere"),))],
        "forward_branch": [
            make(branch, (ir.BlockRef("later"),)),
            make(add, (b, b, a)),
            stream.label("later"),
        ],
        "unknown_add_field": [make(add, (b, a, a), {"offset": 9})],
        "unknown_ds_field": [make(ds, (b, a), {"clamp": False})],
        "unsigned_offset_overflow": [make(ds, (b, a), {"offset": 65536})],
        "negative_unsigned_offset": [make(ds, (b, a), {"offset": -1})],
        "boolean_integer_offset": [make(ds, (b, a), {"offset": True})],
        "signed_offset_underflow": [make(load, (b, pair, off), {"offset": -8388609})],
        "float_integer_offset": [make(ds, (b, a), {"offset": 2.0})],
        "integer_flag": [make(dpp, (b, a), dpp_fields(bound_ctrl=1))],
        "oversized_bits": [make(dot, (b, a, b, a), {"op_sel": (0, 0, 0, 1)})],
        "nonbinary_bits": [make(dot, (b, a, b, a), {"op_sel": (0, 2)})],
        "empty_bits": [make(dot, (b, a, b, a), {"op_sel": ()})],
        "invalid_wait_counter": [make(own("ir.wait_all"), (), {"counter": "vm"})],
        "wait_counter_text": [
            make(own("ir.wait_all"), (), {"counter": "store", "said": "independent"})
        ],
        "missing_wait_count": [make(own("ir.wait_left"), (), {"counter": "ds"})],
        "wait_count_overflow": [make(own("ir.wait_left"), (), {"counter": "store", "left": 256})],
        "missing_wait_counter": [make(own("ir.wait_all"), (), {})],
        "integer_label_name": [make(own("ir.label"), (), {"name": 7})],
        "text_region_tuple": [
            make(own("ir.wait_regs"), (b,), {"counter": "store", "regions": "bad"})
        ],
        "named_regions": [
            make(own("ir.wait_regs"), (b, s), {"counter": "ds", "regions": ("left", "right")})
        ],
        "empty_variadic_wait": [make(own("ir.uniform"), (), {"why": "independent"})],
        "mixed_wait_registers": [make(own("ir.uniform"), (s, scc), {"why": "mixed"})],
        "immediate_wait_operand": [
            make(own("ir.assume"), (ir.Imm(9),), {"lo": 2, "hi": 7, "why": "range"})
        ],
        "extra_assume_operand": [
            make(own("ir.assume"), (b, a), {"lo": 2, "hi": 7, "why": "range"})
        ],
        "assume_bound_overflow": [
            make(own("ir.assume"), (b,), {"lo": 2, "hi": 4294967296, "why": "range"})
        ],
        "label_with_operand": [make(own("ir.label"), (b,), {"name": "other"})],
        "nonbarrier_fence": [make(add, (b, a, b), attrs={0: ir.Fence()})],
        "nonmemory_places": [make(scalar_move, (s, ir.Imm(3)), attrs={0: ir.Places(("region",))})],
        "empty_places": [make(scalar_move, (s, ir.Imm(3)), attrs={0: ir.Places(())})],
        "nonbranch_waves": [make(add, (b, a, b), attrs={0: ir.WavesTaken(frozenset({1, 3}))})],
        "nonmemory_lds": [
            make(scalar_move, (s, ir.Imm(3)), attrs={0: ir.LdsAccess(region=None, what="scratch")})
        ],
        "nonatomic_published": [make(add, (b, a, b), attrs={0: ir.Published("visible")})],
        "printable_wait_marker": [make(own("ir.wait_all"), (), {"counter": "store"})],
        "first_instruction_error": [
            make(branch, (ir.BlockRef("nowhere"),)),
            make(add, (s, b, a)),
            make(t.form("s_endpgm", 0), ()),
        ],
        "operand_error_precedence": [make(add, (s, b, a), {"offset": 11})],
    }
    return p, programs[name]


def verify_program(t, name):
    """Instantiate the program for a verifier rule."""
    program, instructions = _instructions(t, name)
    program.body.extend(instructions)
    return program
