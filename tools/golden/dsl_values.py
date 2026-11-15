"""DSL value, expression and instruction-maker snippets.

Explicit programs exercise type rules, rewrites, instruction forms and refusals.
Each runs on a fresh gfx1201 Builder; streams and diagnostics come from thin.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import random

from tools.golden import encode

BASE = "crates/thin-dsl/tests/data/dsl/"
FILES = ("values.json", "expr.json", "maker.json")
OUTPUTS = tuple(BASE + name for name in FILES)

NOTATION = (
    "Each case is a DSL snippet (`src`) run on a fresh Builder for gfx1201, with `b` the "
    "Builder, `V`, `S` its files and every name of the language imported. `built` is the "
    "stream the Builder made, before any pass: one line per instruction, `<form name> "
    "<operands> {fields}; <attributes>`. A register operand is written as VReg repr "
    "(`<file>:<value name>[<index in group>]x<dwords>` and `+<lo>` when lo != 0); a named "
    "register by its name; an immediate as its value, with the writer's spelling in quotes "
    "when it has one (`16'0x10'`); a block as `@label`. Source modifiers are written `-x`, "
    "`|x|`. Fields are in stored order. Attributes in a fixed order: comment, tag, abl, "
    "alias_ok, staged, wait_uses, in_region, places, published, lds=(region, what, field, "
    "offset: none or var, ordered), m0_set, waves_taken, fence. `values` lists the "
    "program's values in id order: `id file name n= align= [index=] [pin=] [same_as=]`. "
    "`records` holds the Program's lane regions and loops, the deferred records' kinds, "
    "each file's entered / pinned groups and lanes-covered names, and the wave index. "
    "`error` is the refusal (exception class and message) raised while the snippet ran. "
    "`render` (the whole pipeline's output) is the final.s; `render_error` a refusal of "
    "the pipeline's passes."
)


# ---------------------------------------------------------------------------
# Declaration templates


def decl(file, ty, names, at=None, **extra):
    """One line declaring values side by side, pinned upward from ``at``."""
    parts = []
    for k, name in enumerate(names):
        args = [f'name="{name}"']
        if at is not None:
            args.append(f"at={at + k * extra.get('step', 1)}")
        parts.append(f"{file}.{ty}({', '.join(args)})")
    return f"{', '.join(names)} = {', '.join(parts)}"


def src(*rows):
    """A program from its lines (a nested tuple contributes several)."""
    out = []
    for row in rows:
        out.extend(row if isinstance(row, tuple) else (row,))
    return "\n".join(out)


# Shared declaration blocks of the expression programs.
LANES = decl("V", "u32", ("u", "w"), at=3)
SCAL = decl("S", "u32", ("k0", "k1"), at=1)
FLTS = decl("V", "f32", ("fa", "fb", "fc"), at=5)
SGN = decl("V", "i32", ("sg", "sh"), at=8)
INTS = (LANES, SCAL)


def named(rows):
    """``name = File.ty(expr)`` statements: (name, file+type, expression)."""
    return tuple(f'{n} = {ft}({e}, name="{n}")' for n, ft, e in rows)


# Seeded, behavior-preserving constant choices: inline integer constants of
# the VALU (1..64), dword literals (past 64, under 2^31) and the magic-number
# divisors whose quotients need no fix-up step versus those that do.
_RNG = random.Random(0x7A1E5)
INLINE = sorted(_RNG.sample(range(9, 64), 6))
LITERAL = [_RNG.randrange(0x1000, 0x7FFF0000) for _ in range(4)]
SIMPLE_DIVISORS = (5, 11)
FIXUP_DIVISORS = (19,)


# ---------------------------------------------------------------------------
# values.json: value declarations, element access, casts and their refusals

VALUES = [
    (
        "widths_alignment_mix",
        "every value type's width, element count and register alignment",
        "built",
        src(
            'g = S.ptr(name="g"); e = V.i32(name="e"); lm = S.mask(name="lm")',
            'quad = V.u32(n=4, name="quad"); tri = V.f32(n=3, name="tri")',
            'wide = S.u64(name="wide"); two = V.f32x2(name="two"); one = V.f32(name="one")',
            'oct = S.i32(n=8, name="oct"); V.f32(n=2); hexa = S.u32(n=6, name="hexa")',
            "shape = [(x.type.name, x.width, len(x)) for x in (g, e, lm, quad, tri, wide, two,"
            " one, oct, hexa)]",
        ),
    ),
    (
        "name_counters_by_file",
        "a repeated name counts per base and per file; unnamed values are t#N",
        "built",
        src(
            'S.u32(name="idx"); V.u32(name="idx"); V.f32()',
            'V.u32(name="idx"); S.u32(); S.u32(name="idx"); V.u32(name="idx.hi")',
        ),
    ),
    (
        "initial_values_each_element",
        "an initial value is one statement per element; a mask starts from a compare",
        "built",
        src(
            'ones = V.f32(1.0, n=4, name="ones")',
            'lim = S.u32(40, name="lim")',
            'inside = S.mask(lim <= V.u32(name="lane", at=6), name="inside")',
        ),
    ),
    (
        "negative_index_slice_halves",
        "negative index, element of a slice, open slices and a pointer's .hi/.lo",
        "built",
        src(
            'buf = V.f32(n=6, name="buf", at=8)',
            'addr = S.ptr(name="addr", at=6)',
            'dst = V.f32(n=3, name="dst")',
            "dst[2] = buf[-2]",
            "dst[0] = buf[2:5][2]",
            "dst[1] = buf[1:][0]",
            'top = S.u32(addr.hi, name="top")',
            'bot = S.u32(addr.lo, name="bot")',
        ),
    ),
    (
        "same_as_register_share",
        "a pinned value and one sharing its register (same_as) written in place",
        "built",
        src(
            'base = V.u32(name="base", at=5)',
            'alias = V.u32(name="alias", same_as=base)',
            "alias[...] = base ^ 9",
        ),
    ),
    (
        "reinterpret_and_convert_chain",
        "f32()/i32()/u32() reinterpret bits without an instruction; cvt_* convert",
        "built",
        src(
            'bits = V.u32(name="bits", at=1)',
            'h = V.f32(f32(bits) - 4.0, name="h")',
            'cnt = V.i32(cvt_i32(h), name="cnt")',
            'logical = V.u32(u32(cnt) >> 3, name="logical")',
            'arith = V.i32(i32(logical) >> 2, name="arith")',
            'raw = V.u32(u32(arith), name="raw")',
        ),
    ),
    (
        "reinterpret_tree_into_f32",
        "an integer tree reinterpreted into an f32 destination's statement",
        "built",
        src(
            'word = V.u32(name="word", at=2)',
            'dstf = V.f32(name="dstf")',
            "dstf[...] = f32(word ^ 0x80)",
        ),
    ),
    (
        "pair_view_of_elements",
        "f32x2() views two even-aligned elements, read and written; .lo/.hi halves",
        "built",
        src(
            'buf = V.f32(n=6, name="buf", at=4)',
            'pr = V.f32x2(f32x2(buf[4:6]), name="pr")',
            "f32x2(buf[0:2])[...] = pr",
            'prod = V.f32(pr.hi * pr.lo, name="prod")',
        ),
    ),
    (
        "lanes_covered_declared",
        "V.lanes_covered names the values whose every lane is written",
        "built",
        src(
            'src0 = V.u32(name="src0", at=7)',
            'acc = V.u32(name="acc")',
            'V.lanes_covered("acc")',
            "acc[...] = src0 - 2",
        ),
    ),
    (
        "index_at_count",
        "an element index equal to the count is refused",
        "error",
        'S.u32(n=2, name="pair2")[2]',
    ),
    (
        "index_before_start",
        "a negative index past the first element is refused",
        "error",
        'V.u32(n=3, name="row")[-4]',
    ),
    ("slice_with_stride", "a slice with a step", "error", "V.u32(n=6)[1:5:2]"),
    ("slice_of_nothing", "an empty slice", "error", "S.u32(n=4)[2:2]"),
    (
        "float_index",
        "an index that is neither an int nor a slice",
        "error",
        "V.f32(n=2)[1.0]",
    ),
    (
        "halves_of_scalar_u32",
        ".lo of a single dword",
        "error",
        'S.u32(name="cnt").lo',
    ),
    (
        "halves_of_f32",
        ".hi of a single f32 dword",
        "error",
        "V.f32().hi",
    ),
    (
        "halves_of_wave32_mask",
        ".hi of a wave32 lane mask (one SGPR)",
        "error",
        'S.mask(name="lanes").hi',
    ),
    (
        "vector_mask",
        "a lane mask declared in the v file",
        "error",
        'V.mask(name="lanes")',
    ),
    (
        "scalar_pair",
        "an f32x2 declared in the s file",
        "error",
        'S.f32x2(name="duo")',
    ),
    ("zero_elements", "n=0", "error", "S.f32(n=0)"),
    ("bool_element_count", "n given as a bool", "error", "V.f32(n=True)"),
    (
        "name_with_dash",
        "a name outside letters, digits, _ and .",
        "error",
        'S.u32(name="k-1")',
    ),
    (
        "same_as_a_slice",
        "same_as names part of a value",
        "error",
        src('base = V.f32(n=3, name="base")', 'V.f32(name="alias", same_as=base[1:3])'),
    ),
    (
        "same_as_vector_from_scalar",
        "same_as names a value of the other file",
        "error",
        src('vv = V.u32(name="vv")', 'S.u32(name="ss", same_as=vv)'),
    ),
    (
        "pair_view_odd_start",
        "f32x2() of elements starting at an odd register",
        "error",
        src('buf = V.f32(n=6, name="buf", at=4)', "f32x2(buf[3:5])"),
    ),
    (
        "pair_view_of_sgprs",
        "f32x2() of a scalar value",
        "error",
        src('sc = S.f32(n=2, name="sc")', "f32x2(sc)"),
    ),
    (
        "pair_view_one_element",
        "f32x2() of one element",
        "error",
        src('buf = V.f32(n=6, name="buf")', "f32x2(buf[2:3])"),
    ),
    (
        "reinterpret_u64_as_i32",
        "a reinterpret cast between different widths",
        "error",
        src('q = V.u64(name="q")', "i32(q)"),
    ),
    (
        "reinterpret_mask_as_f32",
        "a reinterpret cast of a lane mask",
        "error",
        src('lm = S.mask(name="lm")', "f32(lm)"),
    ),
    (
        "reinterpret_pair_as_u32",
        "a dword cast of an f32x2 (its halves are .lo/.hi)",
        "error",
        src('duo = V.f32x2(name="duo")', "u32(duo)"),
    ),
    (
        "operands_of_two_builders",
        "one expression over the files of two Builders",
        "error",
        src(
            "other = Builder()",
            'V.f32(name="mine", at=2) - other.v.f32(name="theirs", at=2)',
        ),
    ),
    (
        "scalar_number_before_placement",
        "a register's physical number read before regalloc places it",
        "error",
        src('k = S.u32(name="k")', "S.phys(as_reg(k))"),
    ),
]


# ---------------------------------------------------------------------------
# expr.json: expression trees into instructions, rewrites and type refusals

i0, i1, i2, i3, i4, i5 = INLINE
L0, L1, L2, L3 = LITERAL
D5, D11 = SIMPLE_DIVISORS
(D19,) = FIXUP_DIVISORS

EXPR = [
    (
        "vadd_operand_kinds",
        "v_add_nc_u32 of VGPR, SGPR either side, inline either side, a literal",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("sum_sv", "V.u32", "k1 + w"),
                    ("sum_vv", "V.u32", "w + u"),
                    ("sum_iv", "V.u32", f"{i1} + u"),
                    ("sum_vs", "V.u32", "u + k0"),
                    ("sum_lit", "V.u32", f"w + {hex(L0)}"),
                    ("sum_vi", "V.u32", f"w + {i4}"),
                ]
            ),
        ),
    ),
    (
        "vsub_reversed_forms",
        "v_sub_nc_u32 / v_subrev_nc_u32 chosen by which side is the constant or SGPR",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("d_ks", "V.u32", "k1 - u"),
                    ("d_vk", "V.u32", "w - k0"),
                    ("d_iv", "V.u32", f"{i5} - w"),
                    ("d_vv", "V.u32", "u - w"),
                    ("d_vi", "V.u32", f"u - {i0}"),
                    ("d_lit", "V.u32", f"w - {L1}"),
                ]
            ),
        ),
    ),
    (
        "vmul_and_power_of_two",
        "v_mul_lo_u32 operand kinds; a power of two becomes a shift",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("m_vs", "V.u32", "w * k1"),
                    ("m_pow", "V.u32", "u * 64"),
                    ("m_iv", "V.u32", f"{i2} * w"),
                    ("m_lit", "V.u32", "u * 0x10001"),
                    ("m_vv", "V.u32", "u * w"),
                ]
            ),
        ),
    ),
    (
        "vbitwise_operand_kinds",
        "v_not/v_xor/v_and/v_or with VGPR, literal and SGPR operands",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("b_not", "V.u32", "~w"),
                    ("b_xor", "V.u32", "u ^ w"),
                    ("b_and", "V.u32", "w & 0xABCDE"),
                    ("b_or", "V.u32", "k0 | u"),
                ]
            ),
        ),
    ),
    (
        "vshift_kinds",
        "logical/arithmetic shifts by VGPR, SGPR or constant; a constant shifted (e64)",
        "built",
        src(
            INTS,
            SGN,
            named(
                [
                    ("sh_r", "V.u32", "w >> 7"),
                    ("sh_lv", "V.u32", "u << w"),
                    ("sh_as", "V.i32", "sg >> i32(k1)"),
                    ("sh_ai", "V.i32", "sh >> 9"),
                    ("sh_rs", "V.u32", "w >> k0"),
                    ("sh_cv", "V.u32", "5 << u"),
                ]
            ),
        ),
    ),
    (
        "identity_rewrites",
        "x+0, 0+x, x*1, x|0, x&~0, x^0, x<<0 (and x-0, x>>0) become moves",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("id_mul", "V.u32", "1 * u"),
                    ("id_or", "V.u32", "0 | u"),
                    ("id_and", "V.u32", "0xFFFFFFFF & u"),
                    ("id_xor", "V.u32", "0 ^ u"),
                    ("id_shl", "V.u32", "u << 0"),
                    ("id_addr", "V.u32", "u + 0"),
                    ("id_addl", "V.u32", "0 + u"),
                    ("id_sub", "V.u32", "u - 0"),
                    ("id_shr", "V.u32", "u >> 0"),
                ]
            ),
        ),
    ),
    (
        "fused_shift_add_or",
        "x*2^k + y and (x<<k) + y fuse into v_lshl_add_u32; (x<<k) | y into v_lshl_or_b32",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("la_mul", "V.u32", "w + u * 8"),
                    ("la_shl", "V.u32", "(w << 1) + u"),
                    ("la_ks", "V.u32", "u + (w << k1)"),
                    ("lo_or", "V.u32", "u | (w << 4)"),
                    ("la_sk", "V.u32", "k0 + w * 2"),
                ]
            ),
        ),
    ),
    (
        "field_extract_fold_mask",
        "(x>>s)&mask becomes v_bfe_u32 (not past bit 31); constants fold; dead low bits drop",
        "built",
        src(
            INTS,
            SGN,
            named(
                [
                    ("bf", "V.u32", "(w >> 9) & 0x3F"),
                    ("fold", "V.u32", "(u - 20) + 5"),
                    ("low", "V.u32", "(w + (u << 6)) & 0x3F"),
                    ("tail", "V.u32", "(u >> 28) & 0xFF"),
                    ("bfs", "V.i32", "(sg >> 12) & 0x7"),
                ]
            ),
        ),
    ),
    (
        "salu_arith_and_lshl_add",
        "SALU arithmetic; s_lshlN_add_u32 for N = 1..4 only, other shifts apart",
        "built",
        src(
            INTS,
            decl("S", "i32", ("sj",), at=4),
            named(
                [
                    ("sc_madd", "S.u32", "k0 + k1 * 6"),
                    ("sc_sub", "S.u32", "k0 - k1"),
                    ("sc_shl", "S.u32", "k1 << 7"),
                    ("sc_shr", "S.u32", "k0 >> k1"),
                    ("sc_ashr", "S.i32", "sj >> 3"),
                    ("sc_not", "S.u32", "~k1"),
                    ("sc_l1", "S.u32", "k1 * 2 + k0"),
                    ("sc_l4", "S.u32", "k0 + k1 * 16"),
                    ("sc_l6", "S.u32", "(k0 << 6) + k1"),
                ]
            ),
        ),
    ),
    (
        "salu_literal_staging",
        "an SALU op reads one literal: a second is staged; s_bfe_u32 packs offset/width",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("pick", "S.u32", "select(k1 < 12, 0x5555, 0x6666)"),
                    ("lin", "S.u32", f"k0 * 300 + {L2}"),
                    ("fld", "S.u32", "(k1 >> 11) & 0x7F"),
                ]
            ),
        ),
    ),
    (
        "scalar_tree_into_vgpr",
        "a scalar-only tree into a VGPR computes on the SALU then moves; constants",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("vk", "V.u32", "k1 ^ 3"),
                    ("vc", "V.u32", "42"),
                    ("vl", "V.u32", "0xDEADBEEF"),
                    ("sl", "S.u32", "123456"),
                    ("vm", "V.u32", "k0"),
                ]
            ),
        ),
    ),
    (
        "float_arith_constants",
        "f32 + - * with inline floats, a literal, an int constant as a float, 0.0",
        "built",
        src(
            FLTS,
            "",
            named(
                [
                    ("fsum", "V.f32", "fa + fc"),
                    ("fdiff", "V.f32", "fb - 4.0"),
                    ("frev", "V.f32", "0.5 - fb"),
                    ("flit", "V.f32", "fc * 0.1"),
                    ("fint", "V.f32", "fb * 4"),
                    ("fzero", "V.f32", "0.0"),
                    ("fone", "V.f32", "1.0"),
                    ("fneg", "V.f32", "fa * -2.0"),
                ]
            ),
        ),
    ),
    (
        "float_no_implicit_fma",
        "a*b + c stays a multiply and an add; fma() is written explicitly",
        "built",
        src(
            FLTS,
            "",
            named([("unf", "V.f32", "fc + fa * fb"), ("fused", "V.f32", "fma(fc, fb, -1.0)")]),
        ),
    ),
    (
        "float_source_modifiers",
        "neg/abs as source modifiers of an op, alone as sign-bit ops, double negation",
        "built",
        src(
            FLTS,
            "",
            named(
                [
                    ("mm", "V.f32", "abs(fb) * -fc"),
                    ("ng", "V.f32", "-fb"),
                    ("ab", "V.f32", "abs(fc)"),
                    ("nab", "V.f32", "-abs(fb)"),
                    ("abn", "V.f32", "abs(-fc)"),
                    ("dbl", "V.f32", "-(-fa) * fb"),
                    ("fm", "V.f32", "fma(abs(fa), -fb, fc)"),
                ]
            ),
        ),
    ),
    (
        "float_transcendentals",
        "exp2, log2, rcp, rsq, sqrt each one instruction",
        "built",
        src(
            FLTS,
            "",
            named(
                [
                    ("t0", "V.f32", "sqrt(fa) - exp2(fb)"),
                    ("t1", "V.f32", "rcp(log2(fc))"),
                    ("t2", "V.f32", "rsq(fa)"),
                ]
            ),
        ),
    ),
    (
        "numeric_conversions",
        "cvt_f32 of u32/i32/SGPR, cvt_u32 and cvt_i32 of f32",
        "built",
        src(
            INTS,
            decl("V", "f32", ("fa", "fb"), at=5),
            decl("V", "i32", ("sg",), at=8),
            "",
            named(
                [
                    ("cs", "V.f32", "cvt_f32(k1)"),
                    ("ci", "V.i32", "cvt_i32(fb)"),
                    ("cn", "V.f32", "cvt_f32(sg)"),
                    ("cu", "V.u32", "cvt_u32(fa)"),
                    ("cw", "V.f32", "cvt_f32(w)"),
                ]
            ),
        ),
    ),
    (
        "min_max_and_three_way",
        "min/max by type and file; min3/max3/med3 with constants and SGPRs",
        "built",
        src(
            INTS,
            decl("V", "f32", ("fa", "fb"), at=5),
            SGN,
            "",
            named(
                [
                    ("lo_f", "V.f32", "min(fb, fa)"),
                    ("hi_f", "V.f32", "max(fa, -2.0)"),
                    ("lo_u", "V.u32", "min(u, w)"),
                    ("hi_i", "V.i32", "max(sg, sh)"),
                    ("lo_s", "S.u32", "min(k0, k1)"),
                    ("t_max", "V.u32", f"max3(u, {i3}, w)"),
                    ("t_min", "V.i32", "min3(sh, sg, sh)"),
                    ("t_med", "V.u32", "med3(k1, u, w)"),
                ]
            ),
        ),
    ),
    (
        "bf16_halves_as_f32",
        "bf16_hi / bf16_lo read a word's halves as f32 by shift and mask",
        "built",
        src(INTS, "", 'pb = V.f32(bf16_hi(u) * bf16_lo(w), name="pb")'),
    ),
    (
        "multiply_family_both_files",
        "mul24, mad24, mulhi, mul_wide on the VALU and on the SALU",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("sw", "S.u64", "mul_wide(k1, k0)"),
                    ("vh", "V.u32", f"mulhi(w, {hex(L3)})"),
                    ("v24", "V.u32", "mul24(w, 300)"),
                    ("s24", "S.u32", "mul24(k0, k1)"),
                    ("vm24", "V.u32", f"mad24(w, u, {i0})"),
                    ("sm24", "S.u32", "mad24(k1, k0, 3)"),
                    ("vhh", "V.u32", "mulhi(u, w)"),
                    ("sh_", "S.u32", "mulhi(k1, k0)"),
                    ("vw", "V.u64", "mul_wide(w, u)"),
                ]
            ),
        ),
    ),
    (
        "vector_divide_by_constants",
        "// and % by powers of two, magic numbers with and without the fix-up, by 1",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("q32", "V.u32", "w // 32"),
                    ("r4", "V.u32", "w % 4"),
                    ("q5", "V.u32", f"u // {D5}"),
                    ("q19", "V.u32", f"u // {D19}"),
                    ("r12", "V.u32", "w % 12"),
                    ("q1", "V.u32", "u // 1"),
                    ("r1", "V.u32", "u % 1"),
                ]
            ),
        ),
    ),
    (
        "scalar_divide_by_constants",
        "SALU // and % by constants: magic, fix-up, remainder, power of two",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("sq64", "S.u32", "k1 // 64"),
                    ("sq11", "S.u32", f"k0 // {D11}"),
                    ("sq19", "S.u32", f"k1 // {D19}"),
                    ("sr10", "S.u32", "k0 % 10"),
                ]
            ),
        ),
    ),
    (
        "divide_by_divisor_kernel_arg",
        "// and % by a divisor() argument on both files (64-bit magic shift)",
        "built",
        src(
            "e = b.entry(Args(n=u32, d=u32.within(2, 1 << 12).divisor()))",
            "n, d = e.load()",
            'rem = S.u32(n % d, name="rem")',
            'quo = S.u32(n // d, name="quo")',
            'vrem = V.u32(e.tid % d, name="vrem")',
            'vquo = V.u32(e.tid // d, name="vquo")',
        ),
    ),
    (
        "sixty_four_bit_add_sub",
        "u64 + u32 / i32 (sign extended), u64 - u32 and - constant, u64 + u64, + 0",
        "built",
        src(
            INTS,
            'base = S.ptr(name="base", at=4)',
            'qv = V.u64(name="qv", at=6)',
            'vi = V.i32(name="vi", at=8)',
            'si = S.i32(name="si", at=6)',
            named(
                [
                    ("w_sv", "V.u64", "base + w"),
                    ("w_vi", "V.u64", "qv + vi"),
                    ("w_ss", "S.u64", "base + k1"),
                    ("w_si", "S.u64", "base + si"),
                    ("w_sub", "V.u64", "qv - u"),
                    ("w_subc", "S.u64", "base - 40"),
                    ("w_vv", "V.u64", "qv + qv"),
                    ("w_rev", "S.u64", "k0 + base"),
                    ("w_id", "V.u64", "qv + 0"),
                ]
            ),
        ),
    ),
    (
        "select_kinds",
        "select into VCC / an SGPR mask / SCC, literal arms, modified arms, 64-bit, "
        "a scalar condition broadcast to lanes",
        "built",
        src(
            INTS,
            decl("V", "f32", ("fa", "fb"), at=5),
            'lm = S.mask(name="lm", at=4)',
            'base = S.u64(name="base", at=6)',
            named(
                [
                    ("sl_v", "V.u32", "select(w >= 20, u, 7)"),
                    ("sl_m", "V.u32", f"select(lm, {L1}, w)"),
                    ("sl_s", "S.u32", "select(k1 < 9, 9, k1)"),
                    ("sl_n", "V.f32", "select(~(fb > 2.0), fb, 1.0)"),
                    ("sl_mod", "V.f32", "select(fa < fb, abs(fa), -fb)"),
                    ("sl_64", "S.u64", "select(k0 != k1, 0, base)"),
                    ("sl_b", "V.u32", "select(k0 == 4, u, w)"),
                ]
            ),
        ),
    ),
    (
        "compares_to_lane_masks",
        "compares and their & | ^ ~ written into SGPR masks; VCC read; a scalar compare",
        "built",
        src(
            INTS,
            decl("V", "f32", ("fa", "fb"), at=5),
            named(
                [
                    ("ma", "S.mask", "u >= k1"),
                    ("mb", "S.mask", "(w == 0) | (u >= k1)"),
                    ("mc", "S.mask", "~mb"),
                    ("md", "S.mask", "ma & mb"),
                    ("me", "S.mask", "(fa > fb) ^ md"),
                    ("mf", "S.mask", "k1 != 7"),
                    ("mg", "S.mask", "9 >= w"),
                    ("mh", "S.mask", "O.VCC"),
                    ("mi", "S.mask", "~(fb < fa)"),
                    ("mj", "S.mask", "fa == fb"),
                ]
            ),
        ),
    ),
    (
        "compare_operands_swapped_for_vcc",
        "a VOPC compare puts a constant or SGPR first, mirroring the operator",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("pa", "V.u32", "select(w <= 30, u, w)"),
                    ("pb", "V.u32", "select(u > k1, u, w)"),
                    ("pc", "V.u32", "select(u != w, w, u)"),
                    ("pd", "V.u32", "select(k0 <= k1, w, u)"),
                ]
            ),
        ),
    ),
    (
        "mask_counts_and_ranks",
        "popcount / first / rank of masks, EXEC and integers",
        "built",
        src(
            INTS,
            'lm = S.mask(name="lm", at=6)',
            named(
                [
                    ("ex", "S.u32", "popcount(O.EXEC)"),
                    ("rk", "V.u32", "rank(lm)"),
                    ("pv", "V.u32", "popcount(w)"),
                    ("fm", "S.u32", "first(lm)"),
                    ("pk", "S.u32", "popcount(k1)"),
                    ("re", "V.u32", "rank(O.EXEC, 11)"),
                    ("pm", "S.u32", "popcount(lm)"),
                ]
            ),
        ),
    ),
    (
        "lane_reads_and_writes",
        "first_lane, lane_read by constant / SGPR, lane_write by constant / SGPR (m0)",
        "built",
        src(
            INTS,
            "",
            'r_s = S.u32(lane_read(w, k1), name="r_s")',
            'r_c = S.u32(lane_read(u, 17), name="r_c")',
            'f_v = S.u32(first_lane(w), name="f_v")',
            'f_s = S.u32(first_lane(k0), name="f_s")',
            "u[...] = lane_write(u, k0, k1)",
            "w[...] = lane_write(w, 30, k0)",
            f"u[...] = lane_write(u, k1, {L0})",
        ),
    ),
    (
        "lane_shuffles",
        "shuffle by a lane tree (ds_bpermute), an SGPR source moved to a VGPR",
        "built",
        src(
            INTS,
            "",
            'sa = V.u32(shuffle(k1, u), name="sa")',
            'sb = V.u32(shuffle(w, w + 2), name="sb")',
        ),
    ),
    (
        "cross_lane_reductions",
        "across() over 4, 8 and 32 lanes: DPP steps, permlanex16 for the last half",
        "built",
        src(
            decl("V", "f32", ("fa", "fb"), at=5),
            "",
            'ra = V.f32(across(fb, "add", 32), name="ra")',
            'rb = V.f32(across(fa, "min", 4), name="rb")',
            'rc = V.f32(across(fa, "max", 8), name="rc")',
        ),
    ),
    (
        "row_prefix_scan",
        "scan() over a row of 16: four row_shr DPP steps",
        "built",
        src(decl("V", "f32", ("fa",), at=9), 'pre = V.f32(scan(fa, "max", 16), name="pre")'),
    ),
    (
        "raw_ops_inside_trees",
        "op() by mnemonic or dotted name in a tree, its result type (ty=) and operands",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("o1", "V.u32", "op('v_add3_u32', w, k0, u) << 3"),
                    ("o2", "S.u32", "op('s_bfe_u32', k1, 0x30008)"),
                    ("o3", "V.u32", "op('v_add3_u32', u, 9, k1)"),
                    ("o4", "V.f32", "op('v_min_num_f32', f32(w), -1.0)"),
                    ("o5", "V.u32", "op('v.mul.lo.u32', w, 6, ty=u32)"),
                    ("o6", "V.u32", "op('v_mad_u32_u24', u, k1, w, ty=u32)"),
                ]
            ),
        ),
    ),
    (
        "shared_tree_reemitted",
        "a Python tree used in two statements is emitted twice",
        "built",
        src(
            INTS,
            "",
            "tr = (w << 2) ^ u",
            'one = V.u32(tr - 3, name="one")',
            'two = V.u32(tr, name="two")',
        ),
    ),
    (
        "augmented_statements",
        "element statements and every augmented assignment on both files",
        "built",
        src(
            INTS,
            "",
            'cell = V.u32(1, n=3, name="cell")',
            "cell[2] = w - cell[2]",
            "cell[1] ^= u",
            "cell[1] -= 6",
            "cell[1] += k1",
            "cell[1] |= 12",
            "cell[1] &= w",
            "cell[1] *= 5",
            "cell[1] >>= 3",
            "cell[1] <<= k0",
            'tot = S.u32(2, name="tot")',
            "tot *= k1",
            "tot -= 1",
        ),
    ),
    (
        "comment_spelling",
        "comments spell constants >= 4096 in hex, unnamed values t#N, halves, elements",
        "built",
        src(
            INTS,
            'gp = S.ptr(name="gp", at=4)',
            'vec = V.u32(n=3, name="vec")',
            "vec[2] = u - 8192",
            "tmp = V.u32()",
            "tmp[...] = w + 100000",
            'hi = S.u32(gp.hi ^ 2, name="hi")',
            'arr = S.u32(n=3, name="arr")',
            "arr[1] = k1",
            "vec[0] = w + 4095",
        ),
    ),
    (
        "packed_mul_refused",
        "pair() builds an f32x2; packed f32 arithmetic is absent on gfx1201",
        "error",
        src(
            FLTS,
            "",
            'pp = V.f32x2(pair(fa, fb * -1.0), name="pp")',
            'sq = V.f32x2(pp * pp, name="sq")',
        ),
    ),
    (
        "packed_fma_refused",
        "fma of three f32x2 is absent on gfx1201",
        "error",
        src(
            'pa, pb = V.f32x2(name="pa"), V.f32x2(name="pb")',
            'V.f32x2(fma(pa, pb, pa), name="pf")',
        ),
    ),
    (
        "packed_negate",
        "paired scalars and packed negation, then a packed add absent on the target",
        "error",
        src(
            FLTS,
            'pa = V.f32x2(pair(fa + fb, fc * 0.5), name="pa")',
            'V.f32x2(-pa, name="pn")',
            'V.f32x2(pa + pa, name="pq")',
        ),
    ),
    (
        "pair_swap_through_temp",
        "a pair statement reading the half it overwrites goes through a temporary",
        "built",
        src(FLTS, "", 'pp = V.f32x2(name="pp", at=10)', "pp[...] = pair(pp.hi, pp.lo * 0.5)"),
    ),
    (
        "storage_format_unpacks",
        "unpack f16 (either half), e4m3fn and e5m2 bytes to f32",
        "built",
        src(
            INTS,
            "",
            named(
                [
                    ("ub", "V.f32", 'unpack(w, "e5m2")'),
                    ("uh1", "V.f32", 'unpack(u, "f16", at=1)'),
                    ("uf", "V.f32", 'unpack(u, "e4m3fn")'),
                    ("uh0", "V.f32", 'unpack(w, "f16")'),
                ]
            ),
        ),
    ),
    (
        "fp8_pair_unpack_pack_low",
        "unpack2 of e5m2 / e4m3fn pairs; pack2 into the low word of a dword",
        "built",
        src(
            INTS,
            "",
            'pq = V.f32x2(unpack2(w, "e5m2"), name="pq")',
            'pf = V.f32x2(unpack2(u, "e4m3fn"), name="pf")',
            'out = V.u32(7, name="out")',
            'out[...] = pack2(pf.hi, pq.lo, "e5m2", into=out)',
        ),
    ),
    (
        "fp8_pack_high_word",
        "pack2(at=1) writes the high word (op_sel)",
        "built",
        src(
            INTS,
            "",
            'pq = V.f32x2(unpack2(u, "e5m2"), name="pq")',
            'out = V.u32(u, name="out")',
            'out[...] = pack2(pq.hi, pq.lo, "e4m3fn", into=out, at=1)',
        ),
    ),
    (
        "fp8_unpack_byte_index",
        "unpack of an fp8 byte other than the first",
        "error",
        src(INTS, 'V.f32(unpack(w, "e5m2", at=1), name="bad")'),
    ),
    (
        "float_literals_staged",
        "non-inline float constants: one literal per VOP3, the second staged",
        "built",
        src(
            FLTS,
            "",
            named(
                [
                    ("ff", "V.f32", "fma(fa, 10.0, 0.75)"),
                    ("fs", "V.f32", "fb - 7.0"),
                    ("ft", "V.f32", "fc * 1e-20"),
                ]
            ),
        ),
    ),
    (
        "i32_minus_u32",
        "an i32 and a u32 in one operation",
        "error",
        src(INTS, SGN, "V.i32(sg - w)"),
    ),
    (
        "lane_tree_into_sgpr",
        "a tree with a VGPR leaf into an SGPR",
        "error",
        src(INTS, "S.u32(k1 * u)"),
    ),
    (
        "compare_as_python_bool",
        "a compare used as a Python truth value",
        "error",
        src(INTS, "flag = not (w == k1)"),
    ),
    (
        "compare_into_u32_value",
        "a condition as a u32 value",
        "error",
        src(INTS, "V.u32(u >= w)"),
    ),
    (
        "f32_or",
        "a bitwise operator on f32",
        "error",
        src(FLTS, "V.f32(fc | fa)"),
    ),
    (
        "remainder_by_zero",
        "% by the constant 0",
        "error",
        src(INTS, "V.u32(u % 0)"),
    ),
    (
        "i32_remainder",
        "% of an i32",
        "error",
        src(decl("V", "i32", ("sg",), at=2), "V.i32(sg % 5)"),
    ),
    (
        "remainder_of_constant",
        "% of a constant leaf (Python computes it)",
        "error",
        src(INTS, "V.u32(w + u32(40) % 6)"),
    ),
    (
        "dotted_op_untyped",
        "a dotted op() without ty=",
        "error",
        src(INTS, 'op("v.mul.hi.u32", w, k0)'),
    ),
    (
        "pack_f16_pair_absent",
        "pack2 to f16 needs v.cvt.pk.f16.f32, absent on gfx1201",
        "error",
        src(FLTS, 'V.u32(pack2(fc * 2.0, fa, "f16"))'),
    ),
    (
        "median_of_three_floats",
        "med3 of f32 is v_med3_num_f32",
        "built",
        src(FLTS, "V.f32(med3(fc, fa, fb))"),
    ),
    (
        "wave_scan_32",
        "scan over 32 lanes: row steps, a readlane and a masked high row",
        "built",
        src(FLTS, 'V.f32(scan(fb, "add", 32))'),
    ),
    (
        "scan_64_lanes_wave32",
        "scan over 64 lanes on a 32-lane target",
        "error",
        src(FLTS, 'V.f32(scan(fc, "min", 64))'),
    ),
    (
        "packed_sub_absent",
        "f32x2 subtraction (packed add) is absent on gfx1201",
        "error",
        src('pr = V.f32x2(name="pr")', "V.f32x2(pr - pr)"),
    ),
    (
        "vector_value_in_formula",
        "a value of several elements as a formula leaf",
        "error",
        src('vec = V.u32(n=3, name="vec")', "V.u32(vec - 1)"),
    ),
    (
        "statement_into_four_dwords",
        "a statement into a value of several elements",
        "error",
        src('blk = S.u32(n=4, name="blk")', "blk[...] = 7"),
    ),
    (
        "false_as_operand",
        "a Python bool as a constant leaf",
        "error",
        src(INTS, "V.u32(w * False)"),
    ),
    (
        "tuple_as_operand",
        "a tuple as an expression leaf",
        "error",
        src(INTS, "V.u32(u - (2, 3))"),
    ),
    (
        "constant_of_129_bits",
        "an integer constant past 128 bits",
        "error",
        src(INTS, "V.u32(w ^ (1 << 128))"),
    ),
    (
        "u64_plus_large_constant",
        "a u64 constant past 32 bits split into its halves",
        "built",
        src('base = S.u64(name="base", at=2)', "S.u64(base + (5 << 32))"),
    ),
    (
        "u64_constant_half_split",
        "negative, 2^32 multiples, all-but-one ones, subtraction: each half's constant",
        "built",
        src(
            'base = S.u64(name="base", at=4)',
            named(
                [
                    ("hneg", "S.u64", "base + (-7)"),
                    ("hmul", "S.u64", "base + 0x300000000"),
                    ("hall", "S.u64", "base + (2**64 - 2)"),
                    ("hsub", "S.u64", "base - (7 << 32)"),
                ]
            ),
        ),
    ),
    (
        "u64_vector_constant_halves",
        "VALU u64 constants: inline halves, negative, a staged high literal",
        "built",
        src(
            'qv = V.u64(name="qv", at=4)',
            named(
                [
                    ("vneg", "V.u64", "qv + (-12)"),
                    ("vbig", "V.u64", "qv - (1 << 44)"),
                    ("vhi", "V.u64", "qv + (3 << 32)"),
                ]
            ),
        ),
    ),
    (
        "u64_constant_above_range",
        "a u64 constant of 2^64 or more",
        "error",
        src('base = S.u64(name="base", at=2)', "S.u64(base + 0x10000000000000007)"),
    ),
    (
        "u64_constant_below_range",
        "a u64 constant under -2^63",
        "error",
        src('base = S.u64(name="base", at=2)', "S.u64(base + (-(2**63) - 9))"),
    ),
    (
        "signed_zero_constants",
        "-0.0 is the sign bit (a literal), not the inline 0",
        "built",
        src(
            decl("V", "f32", ("fb",), at=3),
            named(
                [
                    ("nz", "S.f32", "-0.0"),
                    ("pz", "V.f32", "0.0"),
                    ("mz", "V.f32", "fb * -0.0"),
                    ("vz", "V.f32", "-0.0"),
                ]
            ),
        ),
    ),
    (
        "u64_times_constant",
        "a u64 multiplied",
        "error",
        src('qv = V.u64(name="qv", at=2)', "V.u64(qv * 4)"),
    ),
    (
        "vector_u32_minus_u64",
        "a u32 minus a u64",
        "error",
        src(INTS, 'qv = V.u64(name="qv", at=6)', "V.u64(w - qv)"),
    ),
    (
        "float_constant_scalar_u32",
        "a float constant in a u32 tree",
        "error",
        src(INTS, "S.u32(k0 * 0.25)"),
    ),
    (
        "u32_tree_into_f32",
        "a u32 tree into an f32 value",
        "error",
        src(INTS, "V.f32(w - 2)"),
    ),
    (
        "mask_from_integer",
        "a mask from an integer tree",
        "error",
        src(INTS, "S.mask(k1 + 1)"),
    ),
    (
        "conditions_multiplied",
        "arithmetic on conditions",
        "error",
        src(INTS, "S.mask((u > 4) * (w == 1))"),
    ),
    (
        "condition_compared",
        "a compare of two conditions",
        "error",
        src(INTS, "(u == 2) >= (k0 != w)"),
    ),
    (
        "invert_f32_tree",
        "~ of an f32 tree",
        "error",
        src(FLTS, "~(fa + fb)"),
    ),
    (
        "negate_scalar_u32",
        "unary minus of an integer",
        "error",
        src(INTS, "-k0"),
    ),
    (
        "f32_plus_pair",
        "an f32 and an f32x2 in one operation (no broadcast)",
        "error",
        src(FLTS, 'pr = V.f32x2(name="pr")', "V.f32x2(fc + pr)"),
    ),
    (
        "pair_xor",
        "a bitwise operator on f32x2",
        "error",
        src('pr = V.f32x2(name="pr")', "V.f32x2(pr ^ pr)"),
    ),
    (
        "pair_equality",
        "a compare of f32x2",
        "error",
        src('pr = V.f32x2(name="pr")', "select(pr == pr, 3, 4)"),
    ),
    (
        "select_on_float",
        "select() whose first operand is not a condition",
        "error",
        src(FLTS, "select(fa, fb, fc)"),
    ),
    (
        "select_of_pairs",
        "select() between f32x2 values",
        "error",
        src('pr = V.f32x2(name="pr")', "select(pr.hi > pr.lo, pr, pr)"),
    ),
    (
        "scalar_select_vector_compare",
        "a per-lane condition selecting into an SGPR",
        "error",
        src(INTS, "S.u32(select(w != k0, k1, 8))"),
    ),
    (
        "scalar_med3",
        "a three-way integer op on SGPRs only",
        "error",
        src(INTS, "S.u32(med3(k0, 6, k1))"),
    ),
    (
        "maximum_of_masks",
        "max of lane masks",
        "error",
        src('lm = S.mask(name="lm")', "max(lm, lm)"),
    ),
    (
        "first_of_vector",
        "first() of an integer value",
        "error",
        src(INTS, "first(u)"),
    ),
    (
        "popcount_of_float_tree",
        "popcount of an f32",
        "error",
        src(FLTS, "popcount(fb * 2.0)"),
    ),
    (
        "raw_op_unknown_type",
        "a raw mnemonic whose result type is not a DSL type, without ty=",
        "error",
        src(INTS, 'op("v_cvt_f32_ubyte2", w)'),
    ),
    (
        "unpack_bf16_high",
        "unpack() asked for bf16",
        "error",
        src(INTS, 'unpack(w, "bf16", at=1)'),
    ),
    (
        "unpack_unknown_format",
        "unpack() of a format it does not convert",
        "error",
        src(INTS, 'unpack(u, "e3m2")'),
    ),
    (
        "unpack_negative_index",
        "unpack() of an f16 at a half outside 0..1",
        "error",
        src(INTS, 'unpack(u, "f16", at=-1)'),
    ),
    (
        "pack_f16_with_half",
        "pack2 to f16 given a half (it writes a whole dword)",
        "error",
        src(FLTS, 'dst = V.u32(name="dst")', 'pack2(fb, fc, "f16", into=dst, at=1)'),
    ),
    (
        "pack_fp8_without_target",
        "pack2 to an fp8 format without into=",
        "error",
        src(FLTS, 'pack2(fc, fa, "e5m2")'),
    ),
    (
        "pack_fp8_into_other",
        "pack2(into=x) written into another value",
        "error",
        src(
            FLTS,
            'dst = V.u32(name="dst")',
            'other = V.u32(name="other")',
            'other[...] = pack2(fa, fb, "e5m2", into=dst, at=1)',
        ),
    ),
    (
        "lane_write_into_other",
        "lane_write(x, ..) written into another value",
        "error",
        src(INTS, "w[...] = lane_write(u, k1, 9)"),
    ),
    (
        "fma_scalar_with_pairs",
        "fma mixing f32 and f32x2",
        "error",
        src(FLTS, 'pr = V.f32x2(name="pr")', "fma(fa, pr, pr)"),
    ),
    (
        "across_one_lane",
        "across() over a lane count that is not a power of two from 2",
        "error",
        src(FLTS, 'across(fb, "add", 12)'),
    ),
    (
        "across_full_wave64",
        "across() over 64 lanes on a 32-lane target",
        "error",
        src(FLTS, 'V.f32(across(fc, "min", 64))'),
    ),
    (
        "across_xor",
        "across() with an operator other than max/min/add",
        "error",
        src(FLTS, 'across(fa, "xor", 8)'),
    ),
    (
        "scan_quad",
        "scan() over fewer than 16 lanes",
        "error",
        src(FLTS, 'scan(fb, "max", 4)'),
    ),
    (
        "convert_float_tree_to_f32",
        "cvt_f32 of an f32",
        "error",
        src(FLTS, "cvt_f32(fa * fb)"),
    ),
    (
        "mad24_of_floats",
        "a 24-bit integer multiply of f32",
        "error",
        src(FLTS, "mad24(fb, fc, 1)"),
    ),
    (
        "pack_bf16_pair_absent",
        "pk_bf16 needs v.cvt.pk.bf16.f32, absent on gfx1201",
        "error",
        src(FLTS, "V.u32(pk_bf16(fc, fc * 0.5))"),
    ),
    (
        "round_bf16_absent",
        "bf16() rounding needs v.cvt.pk.bf16.f32, absent on gfx1201",
        "error",
        src(FLTS, "V.f32(bf16(fa + fb))"),
    ),
    (
        "fp8_pair_high_word",
        "unpack2 of a pair other than the first",
        "error",
        src(INTS, 'V.f32x2(unpack2(u, "e5m2", at=2))'),
    ),
    (
        "scalar_u64_ordered_compare",
        "an ordered compare of scalar u64 values",
        "error",
        src(
            'pa = S.u64(name="pa", at=2)',
            'pb = S.u64(name="pb", at=4)',
            "S.u32(select(pa >= pb, 4, 5))",
        ),
    ),
    (
        "mask_of_lane_select",
        "a mask from a select (a value, not a condition)",
        "error",
        src(INTS, "S.mask(select(u != w, u, w))"),
    ),
]


# ---------------------------------------------------------------------------
# maker.json: O.op / family instructions and their refusals

MAKER_INTS = (LANES, SCAL)
GP = 'gp = S.ptr(name="gp", at=6)'
RSRC = 'rsrc = S.u32(n=4, name="rsrc", at=8)'

MAKER = [
    (
        "raw_mnemonic_dotted_comment",
        "O.op by mnemonic and by dotted name, a comment, inline and literal operands",
        "built",
        src(
            MAKER_INTS,
            "",
            'acc = V.u32(name="acc")',
            'b.emit(O.op("v_xor_b32", acc, w, u))',
            'b.emit(O.op("v.add.u32", acc, acc, 40, comment="acc + 40"))',
            'b.emit(O.op("v_mul_lo_u32", acc, 0x12345, acc))',
            'b.emit(O.op("v_mad_u32_u24", acc, u, 12, acc))',
        ),
    ),
    (
        "vcc_exec_named_operands",
        "VCC and EXEC as O names and by text",
        "built",
        src(
            MAKER_INTS,
            "",
            'b.emit(O.op("v_cmp_ne_u32", O.VCC, 0, w))',
            'save = S.mask(name="save")',
            'b.emit(O.op("s_and_saveexec_b32", save, O.VCC))',
            'b.emit(O.op("s_or_b32", O.EXEC, O.EXEC, save))',
            'b.emit(O.op("s_mov_b32", "exec", save))',
        ),
    ),
    (
        "immediate_spellings_dead_write",
        "integer/float immediates as text and numbers; the pipeline refuses a dead write",
        "render_error",
        src(
            MAKER_INTS,
            "",
            'b.emit(O.op("s_mov_b32", k1, "0x40"))',
            'b.emit(O.op("s_mov_b32", k1, -16))',
            'b.emit(O.op("v_mov_b32", w, "-2.0"))',
            'b.emit(O.op("s_mov_b32", k1, 64))',
            'b.emit(O.op("v_mov_b32", w, 4.0))',
        ),
    ),
    (
        "branch_to_text_label",
        "a branch's label given as text, then placed",
        "built",
        src(
            MAKER_INTS,
            "",
            'b.emit(O.op("s_cmp_lg_u32", k1, 7))',
            'b.emit(O.op("s_cbranch_scc0", ".Lskip"))',
            'b.emit(O.op("s_nop", 3))',
            'b.put_label(".Lskip")',
        ),
    ),
    (
        "dpp_fields_nested_modifiers",
        "DPP fields on an op; neg/abs modifiers, nested either way",
        "built",
        src(
            MAKER_INTS,
            "",
            decl("V", "f32", ("fa", "fb"), at=5),
            'r = V.f32(1.0, name="r")',
            'b.emit(O.op("v_fma_f32", r, O.abs_(fa), O.neg(fb), r))',
            'b.emit(O.op("v_mul_f32_dpp", r, fb, r, dpp="quad_perm:[3,2,1,0]", row_mask=0xF,'
            " bank_mask=0xF))",
            'b.emit(O.op("v_fma_f32", r, O.abs_(O.neg(fb)), O.neg(O.abs_(fa)), r))',
        ),
    ),
    (
        "memory_family_forms_in_flight",
        "s_load/global/buffer/ds families by width, bytes, signedness, offsets; the "
        "pipeline refuses a load over a register still in flight",
        "render_error",
        src(
            MAKER_INTS,
            "",
            GP,
            RSRC,
            'c1, c4, c16 = S.u32(name="c1"), S.u32(n=4, name="c4"), S.u32(n=16, name="c16")',
            'v2, v3, b8 = V.u32(n=2, name="v2"), V.u32(n=3, name="v3"), V.u32(name="b8")',
            "b.emit(O.s_load(c4, gp, 0x40))",
            "b.emit(O.s_load(c1, gp, k0))",
            "b.emit(O.s_load(c16, gp))",
            "b.emit(O.global_load(v3, u, gp, offset=-12))",
            "b.emit(O.global_load(b8, u, gp, bytes=1))",
            "b.emit(O.global_load(v2, w))",
            "b.emit(O.buffer_load(b8, w, rsrc, offset=32, bytes=2, signed=True))",
            "b.emit(O.buffer_load(v2, None, rsrc, k1))",
            "b.wait(c1, c4, c16, v2, v3, b8)",
            "b.emit(O.global_store(w, v3[1:3], gp, offset=128))",
            "b.emit(O.global_store(u, b8, gp, bytes=2))",
            "b.emit(O.buffer_store(v2[0], w, rsrc, k0, offset=12))",
            "b.emit(O.ds_write(u, b8, 8, bytes=1))",
            "b.emit(O.ds_read(v3, w, 512))",
        ),
    ),
    (
        "buffer_offsets_staged_in_flight",
        "buffer offsets past the immediate staged as soffset (mov, and add to an "
        "soffset); the pipeline refuses the second load over the in-flight first",
        "render_error",
        src(
            MAKER_INTS,
            "",
            RSRC,
            'got = V.u32(name="got")',
            "b.emit(O.buffer_load(got, u, rsrc, k1, offset=1 << 24))",
            "b.emit(O.buffer_load(got, w, rsrc, offset=8388608))",
        ),
    ),
    (
        "placement_publication_tags",
        "O.place names, O.published, an op's tag= and abl=",
        "built",
        src(
            MAKER_INTS,
            "",
            GP,
            'b.emit(O.op("global_load_b32", u, w, gp, tag=("lap", 3)))',
            "b.wait(u)",
            'b.emit(O.place(O.op("global_store_b32", w, u, gp), "first", "second"))',
            'b.emit(O.op("s_nop", 1, abl=("y", "z")))',
            'b.emit(O.published(O.op("global_atomic_xor_b32", w, u, gp), "one wave owns it"))',
        ),
    ),
    (
        "too_many_operands",
        "a known mnemonic with an operand count no form has",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_mul_lo_u32", u, w, k0, 3))'),
    ),
    (
        "pre_gfx12_mnemonic",
        "a mnemonic gfx1201 does not have",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("s_add_u32", k0, k0, k1))'),
    ),
    (
        "dotted_name_absent",
        "a dotted op name the target lacks",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v.twiddle.b32", w, u, w))'),
    ),
    (
        "three_sgprs_one_valu",
        "three SGPR reads of one VALU op (the constant bus)",
        "error",
        src(
            MAKER_INTS,
            'k2 = S.u32(name="k2", at=4)',
            'b.emit(O.op("v_med3_u32", u, k1, k2, k0))',
        ),
    ),
    (
        "two_sgprs_and_literal",
        "two SGPRs and an integer literal on the constant bus",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_add3_u32", w, k0, 0x4321, k1))'),
    ),
    (
        "salu_literal_pair",
        "an SALU op with two literals",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("s_and_b32", k0, 0x1234, 0x5678))'),
    ),
    (
        "scalar_global_address",
        "an SGPR as a global access's VGPR address",
        "error",
        src(MAKER_INTS, GP, 'b.emit(O.op("global_load_u16", u, k1, gp))'),
    ),
    (
        "scalar_lds_address",
        "an SGPR as an LDS access's address",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("ds_store_b32", k0, w))'),
    ),
    (
        "lds_offset_at_64k",
        "an LDS offset of 65536",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("ds_store_b32", u, w, offset=65536))'),
    ),
    (
        "global_offset_below_range",
        "a global offset under -2^23",
        "error",
        src(MAKER_INTS, GP, 'b.emit(O.op("global_store_b32", u, w, gp, offset=-8388609))'),
    ),
    (
        "buffer_offen_pair",
        "offen with a two-VGPR vaddr",
        "error",
        src(
            MAKER_INTS,
            RSRC,
            'two = V.u32(n=2, name="two", at=10)',
            'b.emit(O.op("buffer_store_b32", w, two, rsrc, k0, offen=True))',
        ),
    ),
    (
        "unknown_op_field",
        "a field no form has",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_add_nc_u32", u, w, u, glc=1))'),
    ),
    (
        "store_temporal_hint",
        "a th= choice on a raw store",
        "built",
        src(MAKER_INTS, GP, 'b.emit(O.op("global_store_b32", w, u, gp, th="TH_STORE_NT"))'),
    ),
    (
        "store_temporal_hint_unknown",
        "an unknown th= choice is recorded, then refused by the pipeline",
        "render_error",
        src(MAKER_INTS, GP, 'b.emit(O.op("global_store_b32", w, u, gp, th="TH_STORE_ZZ"))'),
    ),
    (
        "family_store_cache_fields",
        "th= and scope= on a family store",
        "built",
        src(
            MAKER_INTS,
            GP,
            'b.emit(O.global_store(u, w, gp, th="TH_STORE_HT", scope="SCOPE_DEV"))',
        ),
    ),
    (
        "operand_none_last",
        "None as an operand",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_and_b32", u, w, None))'),
    ),
    (
        "operand_false",
        "a bool as an operand",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("s_mov_b32", k0, False))'),
    ),
    (
        "single_sgpr_in_64bit_slot",
        "one SGPR where the form takes a pair",
        "error",
        src(
            MAKER_INTS,
            'pr = S.u64(name="pr", at=4)',
            'b.emit(O.op("s_lshl_b64", pr, k0, 3))',
        ),
    ),
    (
        "abs_on_integer_slot",
        "a source modifier is kept by the front end and refused by the integer-slot verifier",
        "render_error",
        src(MAKER_INTS, 'b.emit(O.op("v_sub_nc_u32", w, u, O.abs_(w)))'),
    ),
    (
        "family_store_six_dwords",
        "a family access of more dwords than any form",
        "error",
        src(
            MAKER_INTS,
            GP,
            'six = V.u32(n=6, name="six")',
            "b.emit(O.global_store(u, six, gp))",
        ),
    ),
    (
        "family_three_bytes",
        "bytes= other than 1 or 2",
        "error",
        src(MAKER_INTS, GP, "b.emit(O.global_store(w, u, gp, bytes=3))"),
    ),
    (
        "family_byte_into_quad",
        "bytes= with a destination wider than a dword",
        "error",
        src(
            MAKER_INTS,
            RSRC,
            'quad = V.u32(n=4, name="quad")',
            "b.emit(O.buffer_load(quad, w, rsrc, bytes=1))",
        ),
    ),
    (
        "family_signed_dword",
        "signed= without bytes=",
        "error",
        src(MAKER_INTS, RSRC, "b.emit(O.buffer_load(u, w, rsrc, signed=True))"),
    ),
    (
        "scalar_load_six_dwords",
        "an s_load of a dword count no form has",
        "error",
        src(MAKER_INTS, GP, 'six = S.u32(n=6, name="six")', "b.emit(O.s_load(six, gp, 16))"),
    ),
    (
        "buffer_offset_at_2_23",
        "a buffer offset of 2^23 (one past the range)",
        "error",
        src(MAKER_INTS, "O.buffer_off(1 << 23)"),
    ),
    (
        "buffer_offset_negative",
        "a negative buffer offset",
        "error",
        src(MAKER_INTS, "O.buffer_off(-4)"),
    ),
    (
        "scalar_store_pair_absent",
        "a scalar store, absent on gfx1201",
        "error",
        src(
            MAKER_INTS,
            GP,
            'two = S.u32(n=2, name="two", at=10)',
            "b.emit(O.s_store(two, gp, 16))",
        ),
    ),
    (
        "published_without_reason",
        "O.published with an empty reason",
        "error",
        src(MAKER_INTS, GP, 'O.published(O.op("global_store_b32", u, w, gp), "")'),
    ),
    (
        "float_literal_on_bus",
        "a float literal counts on the constant bus",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_fma_f32", V.f32(name="fd"), k0, -3.7, k1))'),
    ),
    (
        "vcc_on_bus",
        "a named register (VCC) counts on the constant bus",
        "error",
        src(MAKER_INTS, 'b.emit(O.op("v_add3_u32", u, O.VCC, k0, k1))'),
    ),
]

PLANS = {"values.json": VALUES, "expr.json": EXPR, "maker.json": MAKER}


# ---------------------------------------------------------------------------
# Evaluation


def refusal(error):
    """A snippet's refusal: a Diag's own message, else the exception's first argument."""
    message = error.msg if hasattr(error, "msg") else error.args[0] if error.args else ""
    return {"class": type(error).__name__, "message": message}


def render_refusal(error):
    """A pipeline refusal: the first line of its text, the snippet named kernel.py."""
    first = str(error).split("\n")[0].replace("<string>", "kernel.py")
    return {"class": type(error).__name__, "message": first}


def evaluate(name, source):
    """Run one snippet on a fresh gfx1201 Builder, then the whole pipeline."""
    from tests import notation
    from thin import codegen

    compile(source, "kernel.py", "exec")
    case = {"name": name, "src": source}
    try:
        b = notation.run_snippet(source)
    except Exception as error:  # refusals are recorded as the case's result
        case["error"] = refusal(error)
        return case
    case["built"] = notation.built(b)
    case["values"] = notation.values(b)
    records = notation.records(b)
    if records:
        case["records"] = records
    try:
        case["render"] = codegen.render(b)
    except Exception as error:  # a pipeline refusal is the case's render result
        case["render_error"] = render_refusal(error)
    return case


def outcome(case):
    """The case's classification: refused while built, by the pipeline, or rendered."""
    if "error" in case:
        return "error"
    return "render_error" if "render_error" in case else "built"


@contextlib.contextmanager
def no_test_data():
    """Refuse any open of checked-in test data while the generator runs."""
    real_open, real_io_open = builtins.open, io.open

    def guarded(opener):
        """Construct guarded inputs."""

        def inner(file, *args, **kwargs):
            """Construct inner inputs."""
            if "tests/data" in str(file).replace("\\", "/"):
                raise PermissionError(f"golden generator opened test data: {file}")
            return opener(file, *args, **kwargs)

        return inner

    builtins.open, io.open = guarded(real_open), guarded(real_io_open)
    try:
        yield
    finally:
        builtins.open, io.open = real_open, real_io_open


def documents():
    """Every file's document; a plan that misses its targeted outcome is an error."""
    docs = {}
    for file, plans in PLANS.items():
        cases = []
        for name, _target, expect, source in plans:
            case = evaluate(name, source)
            if outcome(case) != expect:
                raise AssertionError(f"{file}:{name}: {outcome(case)}, planned {expect}")
            cases.append(case)
        docs[file] = {"notation": NOTATION, "cases": cases}
    return docs


def generate():
    """Evaluate every authored snippet; no golden file is opened."""
    with no_test_data():
        docs = documents()
    return {BASE + file: encode(doc) for file, doc in docs.items()}
