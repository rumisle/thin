"""DSL entry/argument-block and LDS/address-splitting programs, evaluated through thin.

Every program below is composed here from explicit templates and loops; outputs
(built streams, values, records, renders and refusals) come from running each
program on a fresh gfx1201 Builder and the full pipeline. ``magic.json`` is a
numeric table and stays recorded, so it is not owned here.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import pathlib
import random

from tools.golden import encode

BASE = "crates/thin-dsl/tests/data/dsl/"
ENTRY = BASE + "entry.json"
LDS = BASE + "lds.json"
OUTPUTS = (ENTRY, LDS)

NOTATION = (
    "Generated DSL programs for gfx1201. Each case's `src` runs on a fresh Builder `b`, "
    "with `V` and `S` its vector and scalar files and the whole language in scope. "
    "`built` lists the Builder's own stream before any pass, one instruction per line as "
    "`<form> <operands> {fields} ; <attributes>`: a register is its VReg repr "
    "(`<file>:<value>[<group index>]x<dwords>`, `+<lo>` for a nonzero lo), a physical "
    "register its name, an immediate its value with any written spelling quoted after it "
    "(`32'0x20'`), a block `@label`, a negated or absolute source `-x` or `|x|`; fields in "
    "stored order; attributes in the fixed order comment, tag, abl, alias_ok, staged, "
    "wait_uses, in_region, places, published, lds=(region, what, field, offset none|var, "
    "ordered), m0_set, waves_taken, fence. `values` gives each value in id order as "
    "`id file name n= align=` then any `index=`, `pin=`, `same_as=`. `records` holds the "
    "lane regions, loops, deferred kinds, each file's entered/pinned/lanes-covered groups, "
    "needs, the LDS table and the wave index, when present. `error` is the class and "
    "message of a refusal while the program runs; otherwise `render` is the pipeline's "
    "final assembly lines, or `render_error` its refusal (class and the first line of its "
    "text, the program's file written kernel.py)."
)
ARGS_NOTATION = (
    "`args` cases evaluate one declaration expression with the language in scope: an "
    "argument block's `size`, `own()` (arguments without a library) and each argument's "
    "record without the fields left at their defaults, or the expression's refusal."
)

# Record fields that test_dsl.py leaves out when they hold the API's default.
ARG_DEFAULTS = dict(
    lo=None, hi=None, divisor=False, multiple=1, owned=None, library=None, mem=None, ranks=None
)


# ---------------------------------------------------------------------------
# Evaluation (tests/test_dsl.py conventions)


def _message(error) -> str:
    """A Diag's own message, else the exception's first argument (as test_dsl reads it)."""
    return (
        getattr(error, "msg", None)
        if hasattr(error, "msg")
        else (error.args[0] if error.args else "")
    )


def snippet_case(name: str, src: str) -> dict:
    """Run, then render on a fresh Builder exactly as test_snippet/test_render do."""
    from tests import notation
    from thin import codegen

    compile(src, "kernel.py", "exec")
    case = {"name": name, "src": src}
    try:
        builder = notation.run_snippet(src)
    except Exception as error:  # refusals are intentional cases
        case["error"] = {"class": type(error).__name__, "message": _message(error)}
        return case
    case["built"] = notation.built(builder)
    case["values"] = notation.values(builder)
    records = notation.records(builder)
    if records:
        case["records"] = records
    try:
        case["render"] = codegen.render(notation.run_snippet(src))
    except Exception as error:
        first = str(error).split("\n")[0].replace("<string>", "kernel.py")
        case["render_error"] = {"class": type(error).__name__, "message": first}
    return case


def argument_case(name: str, src: str) -> dict:
    """Evaluate one declaration expression as test_argument_block does."""
    from thin import dsl

    scope = {n: getattr(dsl, n) for n in dsl.__all__}
    case = {"name": name, "src": src}
    try:
        block = eval(src, scope)
    except Exception as error:
        case["error"] = {"class": type(error).__name__, "message": _message(error)}
        return case
    case["size"] = block.size
    case["own"] = block.own()
    case["args"] = [
        {k: v for k, v in block[n].items() if ARG_DEFAULTS.get(k, ...) != v} for n in block.names()
    ]
    return case


def lines(*parts: str) -> str:
    """Compose a source program from independently authored statement lines."""
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# entry.json: kernel entries, argument loads and rank pointers


def entry_plans() -> list[tuple[str, str]]:
    """New entry/load programs and boundary/refusal perturbations."""
    plans = [
        ("entry_empty_block", "kern = b.entry(Args())"),
        (
            "entry_tid_copy_one_dim",
            lines(
                "kern = b.entry(wgsize=128)",
                "assert kern.wgid_y is None and kern.wg == 1",
                'lid = V.u32(kern.tid, name="lid")',
            ),
        ),
        (
            "entry_two_dims_tile_index",
            lines(
                "kern = b.entry(wg=2)",
                'tile = S.u32(kern.wgid_y * 16 + kern.wgid, name="tile")',
            ),
        ),
        (
            "entry_one_dim_row_scale",
            lines(
                "kern = b.entry(Args(), wg=1)",
                'row = S.u32(kern.wgid * 5, name="row")',
            ),
        ),
    ]
    # Merged argument loads: a run of words wider than one load, then a pointer.
    words = 12
    plans.append(
        (
            "merge_words_then_pointer",
            lines(
                f'kern = b.entry(Args(**{{f"w{{i}}": u32 for i in range({words})}}, base=ptr))',
                "ws = kern.load()",
                f'acc = S.u32(ws[{words - 1}], name="acc")',
                f"for w in ws[:{words - 1}]:",
                "    acc ^= w",
            ),
        )
    )
    plans.append(
        (
            "merge_pointer_then_signed_words",
            lines(
                'kern = b.entry(Args(base=ptr, **{f"k{i}": i32 for i in range(9)}, scale=f32))',
                "got = kern.load()",
                'mix = S.i32(got[1] - got[9], name="mix")',
            ),
        )
    )
    plans += [
        (
            "load_subset_then_reversed_pair",
            lines(
                "kern = b.entry(Args(cols=u32, src=ptr, ld=u32, rows=u32))",
                '(ld,) = kern.load("ld")',
                'span = S.u32(ld * 3, name="span")',
                '(rows, src) = kern.load("rows", "src")',
                "span -= rows",
            ),
        ),
        (
            "reload_after_guarded_load",
            lines(
                "kern = b.entry(Args(lim=u32, alt=u32, gain=u32))",
                '(lim,) = kern.load("lim")',
                'pick = S.u32(lim, name="pick")',
                "with b.when(lim != 7):",
                '    (alt,) = kern.load("alt")',
                "    pick ^= alt",
                '(alt,) = kern.load("alt")',
                "pick[...] = pick | alt",
            ),
        ),
        (
            "library_argument_leads_block",
            lines(
                'kern = b.entry(Args(tally=lib_arg(gens, "tally"), out=ptr, cnt=u32))',
                "out, cnt = kern.load()",
                '(tl,) = kern.load("tally")',
                'more = S.u32(cnt << 2, name="more")',
            ),
        ),
        (
            "peers_rank_by_loaded_index",
            lines(
                "kern = b.entry(Args(rk=u32, ring=peers(3)))",
                "rk, ring = kern.load()",
                'far = kern.load_rank("ring", rk)',
            ),
        ),
        (
            "mapped_rank_by_masked_index",
            lines(
                "kern = b.entry(Args(win=mapped(4), sel=u32))",
                '(sel,) = kern.load("sel")',
                'pick = S.u32(sel & 3, name="pick")',
                'peer = kern.load_rank("win", pick)',
            ),
        ),
        (
            "mapped_rank_by_pinned_index_after_full_load",
            lines(
                "kern = b.entry(Args(cnt=u32, win=mapped(3)))",
                "cnt, win = kern.load()",
                'who = S.u32(name="who", at=12)',
                'other = kern.load_rank("win", who)',
                'left = S.u32(cnt - 2, name="left")',
            ),
        ),
        (
            "preload_hint_ignored_two_dims",
            lines(
                "kern = b.entry(Args(gain=f32, lo=u32, hi=u32, dst=ptr), preload=2, wg=2)",
                "gain, lo, hi, dst = kern.load()",
                'width = S.u32(kern.wgid_y - (hi - lo), name="width")',
            ),
        ),
        (
            "preload_hint_unused_argument",
            lines(
                "kern = b.entry(Args(n=u32), wg=2, preload=8)",
                "(n,) = kern.load()",
            ),
        ),
    ]
    for wg in (0, 4):
        plans.append((f"refuse_wg_{wg}", f"b.entry(wg={wg})"))
    plans += [
        (
            "refuse_load_unknown_after_known",
            lines("kern = b.entry(Args(src=ptr, n=u32))", 'kern.load("n", "cnt")'),
        ),
        (
            "refuse_load_from_empty_block",
            lines("kern = b.entry(Args())", 'kern.load("x")'),
        ),
        (
            "refuse_load_rank_of_scalar",
            lines(
                "kern = b.entry(Args(n=u32, ring=peers(2)))",
                'kern.load_rank("n", kern.wgid)',
            ),
        ),
        (
            "refuse_load_rank_of_owned_pointer",
            lines(
                "kern = b.entry(Args(buf=kept, ring=peers(2)))",
                'kern.load_rank("buf", kern.wgid)',
            ),
        ),
        ("refuse_second_entry", lines("b.entry(wg=2)", "b.entry()")),
    ]
    return plans


def argument_plans() -> list[tuple[str, str]]:
    """New argument declaration products and invalid-field perturbations."""
    plans = [
        (
            "four_pointers_then_ranged_count",
            "Args(src=ptr, dst=ptr, aux=ptr, bias=ptr, cnt=u32.within(2, 1 << 20))",
        ),
        ("signed_range_then_wide", "Args(off=i32.within(-64, 64), w=u64)"),
        ("i32_pads_before_pointer", "Args(k=i32, base=ptr)"),
        ("wide_values_around_scalars", "Args(h=u64, a=f32, c=i32, e=u32, g=u64)"),
        ("placed_middle_then_follower", 'Args(at={"gamma": 32}, out=ptr, gamma=f32, delta=u32)'),
        ("placed_first_then_pointer", 'Args(at={"lr": 16}, lr=f32, w=ptr)'),
        ("placed_out_of_order", 'Args(at={"a": 8, "b": 0}, a=u32, b=u32)'),
        (
            "divisor_first_then_scalars",
            "Args(stride=u32.within(3, 1000).divisor(), idx=u32, out=ptr)",
        ),
        ("divisor_at_upper_bound", "Args(lead=u32.within(1, 1 << 31).divisor())"),
        ("multiple_then_pointer", "Args(len_=u32.within(8, 512).multiple_of(8), out=ptr)"),
        ("signed_multiple", "Args(q=i32.within(-30, 30).multiple_of(6))"),
        (
            "multiple_and_divisor_either_order",
            "Args(p=u32.within(8, 512).multiple_of(8).divisor(), "
            "r=u32.within(6, 96).divisor().multiple_of(3))",
        ),
        (
            "owned_kinds_with_uncached",
            "Args(g=gens.uncached, k=kept.uncached, s=scratch, c=carried, kk=kept)",
        ),
        ("rank_groups_at_bounds", "Args(r=peers(8), m=mapped(2), z=u32)"),
        ("library_pointer_first", 'Args(cl=lib_arg(kept, "clock"), out=ptr)'),
        (
            "library_scalar_and_uncached",
            'Args(v=u32, sc=lib_arg(u32, "sum"), kc=lib_arg(kept.uncached, "ring"))',
        ),
        ("empty_argument_block", "Args(at={})"),
    ]
    # Seeded random blocks of plain kinds: offsets/padding come from the native layout.
    rng = random.Random(0x4D454D)
    kinds = ("u32", "i32", "f32", "u64", "ptr")
    for n in (3, 6):
        names = rng.sample("abcdefghjkmnpqrstuvwxyz", n)
        body = ", ".join(f"{name}={rng.choice(kinds)}" for name in names)
        plans.append((f"seeded_plain_block_{n}", f"Args({body})"))
    refusals = [
        ("refuse_kind_bool", "Args(flag=bool)"),
        ("refuse_kind_string", 'Args(h="f16")'),
        *(
            (f"refuse_range_on_{kind}", f"Args(v=({kind}).within(0, 9))")
            for kind in ("u64", "ptr", "f32")
        ),
        ("refuse_at_unknown_names", 'Args(at={"nope": 8, "zz": 0}, a=u32)'),
        ("refuse_at_misaligned_ptr", 'Args(at={"q": 12}, p=u32, q=ptr)'),
        ("refuse_at_misaligned_u64", 'Args(at={"w": 4}, v=u32, w=u64)'),
        ("refuse_at_misaligned_u32", 'Args(at={"v": 3}, v=u32)'),
        ("refuse_overlap_placed_into_wide", 'Args(at={"tail": 12}, head=u64, mid=u64, tail=u32)'),
        ("refuse_overlap_same_start", 'Args(at={"b": 0}, a=u32, b=u32)'),
        ("refuse_divisor_lo_zero", "u32.within(0, 10).divisor()"),
        ("refuse_divisor_hi_past", "u32.within(1, (1 << 31) + 1).divisor()"),
        ("refuse_divisor_signed", "i32.within(2, 100).divisor()"),
        ("refuse_multiple_none_in_range", "u32.within(9, 15).multiple_of(16)"),
        *(
            (f"refuse_multiple_{label}", f"u32.within(0, 64).multiple_of({m})")
            for label, m in (("zero", "0"), ("negative", "-4"), ("bool", "True"))
        ),
        ("refuse_multiple_restated", "u32.within(0, 64).multiple_of(8).multiple_of(2)"),
        ("refuse_range_reversed", "i32.within(0, -1)"),
        *((f"refuse_mapped_{n}", f"mapped({n})") for n in (0, 9)),
        *((f"refuse_peers_{n}", f"peers({n})") for n in (1, 16)),
        ("refuse_uncached_scratch_in_block", "Args(tmp=scratch.uncached)"),
        ("refuse_uncached_mapped", "mapped(3).uncached"),
        ("refuse_uncached_twice", "carried.uncached.uncached"),
    ]
    return plans + refusals


# ---------------------------------------------------------------------------
# lds.json: layouts, regions, addresses and address splitting


def lds_plans() -> list[tuple[str, str]]:
    """New layouts, address formulas and memory-access programs."""
    plans = [
        (
            "table_budget_union_parts",
            lines(
                "L = b.lds(budget=48 * 1024)",
                'Hs = L.region("Hs", (32, 48), "f16", pitch=112)',
                "with L.union():",
                '    Acc = L.region("Acc", (16, 16), f32)',
                '    Wt = L.region("Wt", (128, 32), "u8")',
                '    Ix = L.region("Ix", (40,), u32)',
                'St = L.region("St", (4, 8, 64), "bf16")',
                's0, s1, s2, s3 = St.parts("s0", "s1", "s2", "s3")',
            ),
        ),
        (
            "table_groups_alignment_placement",
            lines(
                "L = b.lds()",
                'with L.struct(name="Hdr", align=16):',
                '    Cnt = L.region("Cnt", nbytes=6)',
                '    Flg = L.region("Flg", (2,), "u16", align=8)',
                'Pad = L.region("Pad", (7,), "i8")',
                'with L.union(name="Pool", align=32):',
                '    Ring = L.region("Ring", (3, 5), "b32", pitch={0: 24})',
                '    with L.struct(name="Pair"):',
                '        Lo = L.region("Lo", (3,), "u64")',
                '        Hi = L.region("Hi", (9,), "e4m3")',
                'Tail = L.region("Tail", nbytes=12, at=512)',
                'Cube = L.region("Cube", (3, 2, 5), "u16", pitch={-2: 16, 0: 48})',
            ),
        ),
        (
            "pointer_moves_and_far_constant",
            lines(
                "L = b.lds()",
                'Tile = L.region("Tile", (32, 96), "u32", pitch=400)',
                'r, c = V.u32(name="r", at=3), V.u32(name="c", at=5)',
                'acc = V.u32(n=4, name="acc")',
                'cur = Tile.pointer(r, c, name="cur")',
                "b.emit(O.ds_read(acc[0], cur.at(0, 3)))",
                "b.emit(O.ds_read(acc[1], cur.at(2, -1)))",
                "b.emit(O.ds_read(acc[2], Tile.at(r | 1, c + 2)))",
                "b.emit(O.ds_read(acc[3], Tile + 99996))",
                "b.wait(acc)",
                'tot = V.u32(acc[0] ^ acc[1] ^ acc[2] ^ acc[3], name="tot")',
                "b.emit(O.ds_write(cur.at(1, 1), tot))",
            ),
        ),
    ]
    # Address split: each shape lands in its own destination, then one wait.
    shapes = [
        "Q.at(v, 1)",
        "Q + v * 4 + 28",
        "Q + (v + 6) * 4",
        "Q + (v - 5) * 4 + 8",
        "Q + (127 - v) * 8",
        "Q + k",
        "(Q + v) - 20",
        "Q + v",
        "Q + v * 3 + 6",
        "Q + (v + 1) * 4 - 4",
        "Q.at(1, 1) + v",
    ]
    body = [
        "L = b.lds()",
        'Q = L.region("Q", (32, 4), "u32")',
        'v = V.u32(name="v", at=2)',
        'k = V.i32(name="k", at=9)',
        f'got = V.u32(n={len(shapes) + 1}, name="got")',
    ]
    body += [f"b.emit(O.ds_read(got[{i}], {shape}))" for i, shape in enumerate(shapes)]
    body += [
        f"b.emit(O.ds_read(got[{len(shapes)}], Q + 8, 36))",
        "b.wait(got)",
        'b.emit(O.ds_write(Q.at(5, 3), got[0], ordered="lane zero of wave one stores it"))',
    ]
    plans.append(("split_shapes_one_destination_each", lines(*body)))
    body = [
        "L = b.lds()",
        'Q = L.region("Q", (32, 4), "u32")',
        'v = V.u32(name="v", at=2)',
        'k = V.i32(name="k", at=9)',
        'got = V.u32(n=4, name="got")',
    ]
    for i, extra in enumerate(("v + 65532", "v + 65536", "k + 4", "65535 + v - 3")):
        body.append(f"b.emit(O.ds_read(got[{i}], Q + {extra}))")
    plans.append(("split_offset_field_limit", lines(*body)))
    plans.append(
        (
            "store_reads_load_in_flight",
            lines(
                "L = b.lds()",
                'Q = L.region("Q", (32, 4), "u32")',
                'v = V.u32(name="v", at=2)',
                'got = V.u32(name="got")',
                "b.emit(O.ds_read(got, Q + v))",
                "b.emit(O.ds_write(Q + v + 4, got))",
            ),
        )
    )
    plans.append(
        (
            "reload_same_destination_in_flight",
            lines(
                "L = b.lds()",
                'Bk = L.region("Bk", (16, 16), "u16")',
                'v = V.u32(name="v", at=6)',
                'h = V.u32(name="h")',
                "b.emit(O.ds_read(h, Bk.at(v, 4), bytes=2))",
                "b.emit(O.ds_read(h, Bk + 64, bytes=2))",
                "b.wait(h)",
            ),
        )
    )
    plans.append(
        (
            "scalar_offsets_into_vector_address",
            lines(
                "L = b.lds()",
                'P = L.region("P", (8, 4), f32)',
                'wv = S.u32(name="wv", at=6)',
                'lo, hi = V.f32(name="lo", at=2), V.f32(name="hi", at=3)',
                'o = V.u32(P.offset(wv, 2), name="o")',
                'r = P.pointer(wv + 1, 3, name="r")',
                "b.emit(O.ds_write(P + o, hi))",
                "b.emit(O.ds_write(P + o + 4, lo))",
                "b.emit(O.ds_write(r.at(2, -3), lo))",
            ),
        )
    )
    plans.append(
        (
            "swizzle_shifted_key",
            lines(
                "L = b.lds()",
                (
                    'Ring = L.region("Ring", (128, 16), "b64", '
                    "swizzle=Xor(by=0, into=1, mask=15, shift=2))"
                ),
                't = V.u32(name="t", at=1)',
                'q = Ring.pointer(t >> 2, t & 3, name="q")',
                'pair = V.u32(n=2, name="pair")',
                "b.emit(O.ds_read(pair, q.at(64, 0)))",
                'slot = V.u32(Ring.stored(t >> 4, t & 15)[1], name="slot")',
                'fixed = V.u32(Ring.offset(9, 6), name="fixed")',
                'raw = V.u32(Ring.offset(9, 6, swizzled=False), name="raw")',
            ),
        )
    )
    plans.append(
        (
            "swizzle_plain_key_wide_rows",
            lines(
                "L = b.lds()",
                'Sw = L.region("Sw", (64, 4), "b128", swizzle=Xor(by=0, into=1, mask=3))',
                'ln = V.u32(name="ln", at=7)',
                'p = Sw.pointer(ln >> 2, ln & 3, name="p")',
                'quad = V.u32(n=4, name="quad")',
                "b.emit(O.ds_read(quad, p.at(4, 0)))",
                'pos = V.u32(Sw.stored(ln, ln >> 1)[1], name="pos")',
            ),
        )
    )
    plans.append(
        (
            "parent_pointer_lands_in_part",
            lines(
                "L = b.lds()",
                'KV = L.region("KV", (3, 8, 8), "u32")',
                'Ks, Vs, Ws = KV.parts("Ks", "Vs", "Ws")',
                't = V.u32(name="t", at=2)',
                'p = KV.pointer(0, t, 0, name="p")',
                'x, y = V.u32(name="x"), V.u32(name="y")',
                "b.emit(O.ds_read(x, p.at(2, 1, 3)))",
                "b.emit(O.ds_read(y, Vs.at(t, 5)))",
                "b.wait(Vs, Ws)",
                'z = V.u32(x + y, name="z")',
            ),
        )
    )
    plans.append(
        (
            "part_wait_misses_register",
            lines(
                "L = b.lds()",
                'AB = L.region("AB", (2, 32), f32)',
                'Pa, Pb = AB.parts("Pa", "Pb")',
                'ln = V.u32(name="ln", at=1)',
                'e = V.f32(name="e")',
                "b.emit(O.ds_read(e, AB.pointer(1, ln, name='q').at(0, 0)))",
                "b.emit(O.ds_read(e, Pa.at(ln)))",
                "b.wait(Pb)",
            ),
        )
    )
    plans.append(
        (
            "global_and_buffer_split",
            lines(
                'src = S.ptr(name="src", at=2)',
                'rsrc = S.u32(n=4, name="rsrc", at=8)',
                'i, j = V.u32(name="i", at=4), V.u32(name="j", at=5)',
                'lhs, rhs = V.u32(name="lhs"), V.u32(name="rhs")',
                "b.emit(O.global_load(lhs, src + mul24(j, 768) + i * 8 + 4))",
                "b.emit(O.buffer_load(rhs, i * 16 + 48, rsrc))",
                "b.wait(lhs, rhs)",
                "lhs ^= rhs",
                "b.emit(O.global_store(src + i * 8 + 4096, lhs))",
                "b.emit(O.global_store(src + 8388607, lhs))",
                "b.emit(O.global_store(src + 8388608, lhs))",
                "b.emit(O.global_store(i * 8, lhs, src, offset=-16))",
                "b.emit(O.global_store(src, lhs))",
            ),
        )
    )
    plans.append(
        (
            "scalar_split_distinct_destinations",
            lines(
                'tab = S.ptr(name="tab", at=10)',
                'sel = S.u32(name="sel", at=5)',
                'u, v, w, z = S.u32(name="u"), S.u32(name="v"), S.u32(name="w"), S.u32(name="z")',
                "b.emit(O.s_load(u, tab + sel * 16 + 32))",
                "b.emit(O.s_load(v, tab + 124))",
                "b.emit(O.s_load(w, tab + sel))",
                "b.emit(O.s_load(z, tab, sel + 12))",
                "b.wait(u, v, w, z)",
                'mix = S.u32(u + v + w + z, name="mix")',
            ),
        )
    )
    plans.append(
        (
            "scalar_reload_in_flight",
            lines(
                'tab = S.ptr(name="tab", at=10)',
                'sel = S.u32(name="sel", at=5)',
                'u = S.u32(name="u")',
                "b.emit(O.s_load(u, tab + sel * 4))",
                "b.emit(O.s_load(u, tab, 64))",
            ),
        )
    )
    # Layout refusals: one perturbation each of an otherwise valid declaration.
    plans += [
        ("refuse_layout_twice", lines("b.lds(budget=1024)", "b.lds(budget=1024)")),
        *((f"refuse_budget_{n}", f"b.lds(budget={n})") for n in (0, 65537)),
        (
            "refuse_past_budget",
            lines(
                "L = b.lds(budget=256)",
                'L.region("Lead", (16,), "u32")',
                'L.region("Rest", nbytes=200)',
                "L.leaves()",
                "L.size",
            ),
        ),
        (
            "refuse_past_target",
            lines("L = b.lds()", 'L.region("Huge", (128, 256), f32)', "L.size"),
        ),
        *(
            (
                f"refuse_final_after_{what}",
                lines(
                    "L = b.lds()",
                    'Z = L.region("Z", (4,), "u16")',
                    read,
                    'L.region("Late", nbytes=4)',
                ),
            )
            for what, read in (("size", "L.size"), ("base", "Z.base"))
        ),
        (
            "refuse_duplicate_in_union",
            lines(
                "L = b.lds()",
                "with L.union():",
                '    L.region("Dup", (8,), "u32")',
                '    L.region("Dup", (4,), "u64")',
            ),
        ),
        (
            "refuse_overlap_placed",
            lines(
                "L = b.lds()",
                'L.region("Lft", nbytes=32, at=64)',
                'L.region("Rgt", nbytes=16, at=80)',
                "L.size",
            ),
        ),
        ("refuse_shape_zero_dim", lines("L = b.lds()", 'L.region("Z3", (3, 0, 2), "f16")')),
        ("refuse_no_size", lines("L = b.lds()", 'L.region("Bare", elem="u16")')),
        ("refuse_pitch_one_dim", lines("L = b.lds()", 'L.region("Vec", (16,), "f16", pitch=64)')),
        *(
            (
                f"refuse_pitch_{label}",
                lines("L = b.lds()", f'L.region("Rows", (2, 8), f32, pitch={pitch})'),
            )
            for label, pitch in (("short", 28), ("unaligned", 34))
        ),
        (
            "refuse_nbytes_below_shape",
            lines("L = b.lds()", 'L.region("Nb", (8, 2), "bf16", nbytes=16)'),
        ),
        *(
            (f"refuse_element_type_{label}", lines("L = b.lds()", f'L.region("El", (2,), {t})'))
            for label, t in (("f32x2", "f32x2"), ("pair", "pair"))
        ),
        *(
            (
                f"refuse_element_name_{label}",
                lines("L = b.lds()", f'L.region("El", (2,), "{t}")'),
            )
            for label, t in (("fp8", "fp8"), ("b16", "b16"))
        ),
        (
            "refuse_swizzle_one_dim",
            lines(
                "L = b.lds()", 'L.region("Sw1", (32,), "b128", swizzle=Xor(by=0, into=0, mask=3))'
            ),
        ),
        (
            "refuse_swizzle_same_dims",
            lines("L = b.lds()", 'L.region("Sw3", (4, 4, 4), "u32", swizzle=Xor(2, 2, 3))'),
        ),
        *(
            (
                f"refuse_swizzle_mask_{mask}",
                lines(
                    "L = b.lds()",
                    f'L.region("Swm", (8, 12), "u32", swizzle=Xor(by=0, into=1, mask={mask}))',
                ),
            )
            for mask in (7, 5)
        ),
        *(
            (
                f"refuse_parts_{count}_names",
                lines(
                    "L = b.lds()",
                    'Pr = L.region("Pr", (3, 2), "u16")',
                    "Pr.parts(" + ", ".join(f'"n{i}"' for i in range(count)) + ")",
                ),
            )
            for count in (2, 4)
        ),
        (
            "refuse_parts_along_swizzle_key",
            lines(
                "L = b.lds()",
                'Sk = L.region("Sk", (4, 4), "b64", swizzle=Xor(by=0, into=1, mask=3))',
                'Sk.parts("r0", "r1", "r2", "r3")',
            ),
        ),
        (
            "refuse_too_many_indices",
            lines("L = b.lds()", 'Ln = L.region("Ln", (6,), "u32")', "Ln.offset(2, 1)"),
        ),
        (
            "refuse_pointer_move_by_formula",
            lines(
                "L = b.lds()",
                'Mv = L.region("Mv", (4, 16), "u16")',
                'y = V.u32(name="y", at=3)',
                'm = Mv.pointer(1, y, name="m")',
                "m.at(0, y + 1)",
            ),
        ),
        (
            "refuse_pointer_move_breaks_swizzle",
            lines(
                "L = b.lds()",
                'Bs = L.region("Bs", (32, 4), "b64", swizzle=Xor(by=0, into=1, mask=3))',
                'y = V.u32(name="y", at=3)',
                'm = Bs.pointer(y, 1, name="m")',
                "m.at(0, 2)",
            ),
        ),
        (
            "refuse_pointer_subarray_element_swizzled",
            lines(
                "L = b.lds()",
                'Bs = L.region("Bs", (32, 4), "b64", swizzle=Xor(by=0, into=1, mask=3))',
                'y = V.u32(name="y", at=3)',
                'm = Bs.pointer(y, name="m")',
                "m.at(4, 3)",
            ),
        ),
        (
            "refuse_address_plus_address",
            lines(
                "L = b.lds()",
                'R1 = L.region("R1", (4,), "u32")',
                'R2 = L.region("R2", (4,), "u32")',
                "R1.at(2) + R2",
            ),
        ),
        (
            "refuse_register_address_with_layout",
            lines(
                "L = b.lds()",
                'L.region("Rg", nbytes=64)',
                'off, val = V.u32(name="off", at=1), V.u32(name="val", at=2)',
                "b.emit(O.ds_write(off, val))",
            ),
        ),
        (
            "refuse_region_of_other_builder",
            lines(
                "other = Builder()",
                'Ob = other.lds().region("Ob", (4,), "u32")',
                'val = V.u32(name="val", at=2)',
                "b.emit(O.ds_write(Ob.at(1), val))",
            ),
        ),
        (
            "refuse_bool_byte_offset",
            lines(
                "L = b.lds()",
                'Bo = L.region("Bo", (4,), "u32")',
                'val = V.u32(name="val", at=2)',
                "b.emit(O.ds_write(Bo + False, val))",
            ),
        ),
        (
            "refuse_vector_term_in_scalar_address",
            lines(
                'tab = S.ptr(name="tab", at=4)',
                'ln = V.u32(name="ln", at=3)',
                'h = S.u32(name="h")',
                "b.emit(O.s_load(h, tab + ln * 4))",
            ),
        ),
        *(
            (
                f"refuse_global_offset_{label}",
                lines(
                    'src = S.ptr(name="src", at=2)',
                    'q = V.u32(name="q", at=5)',
                    f"b.emit(O.global_load(q, q, src, offset={off}))",
                ),
            )
            for label, off in (("below", -8388609), ("above", 8388608))
        ),
        (
            "refuse_global_pointer_minus",
            lines(
                'src = S.ptr(name="src", at=2)',
                'q = V.u32(name="q", at=5)',
                "b.emit(O.global_store(src - 256, q))",
            ),
        ),
    ]
    return plans


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _no_fixture_reads():
    """Refuse any open of test data while generating (fixtures are never inputs)."""
    root = (pathlib.Path(__file__).resolve().parents[2]).as_posix()
    originals = (builtins.open, io.open, pathlib.Path.open)

    def guard(path):
        """Construct guard inputs."""
        text = str(path)
        if "tests/data/" in text and (text.startswith(root) or not text.startswith("/")):
            raise PermissionError(f"dsl_memory generator must not read {text}")

    def open_(file, *a, **k):
        """Construct open  inputs."""
        if isinstance(file, (str, pathlib.PurePath)):
            guard(file)
        return originals[0](file, *a, **k)

    def path_open(self, *a, **k):
        """Construct path open inputs."""
        guard(self)
        return originals[2](self, *a, **k)

    builtins.open, io.open, pathlib.Path.open = open_, open_, path_open
    try:
        yield
    finally:
        builtins.open, io.open, pathlib.Path.open = originals


def _unique(plans):
    names = [name for name, _ in plans]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate case names: {sorted(dupes)}")
    return plans


EXPECTED_FAILURES = {
    "refuse_wg_0": "error",
    "refuse_wg_4": "error",
    "refuse_load_unknown_after_known": "error",
    "refuse_load_from_empty_block": "error",
    "refuse_load_rank_of_scalar": "error",
    "refuse_load_rank_of_owned_pointer": "error",
    "refuse_second_entry": "error",
    "refuse_kind_bool": "error",
    "refuse_kind_string": "error",
    "refuse_range_on_u64": "error",
    "refuse_range_on_ptr": "error",
    "refuse_range_on_f32": "error",
    "refuse_at_unknown_names": "error",
    "refuse_at_misaligned_ptr": "error",
    "refuse_at_misaligned_u64": "error",
    "refuse_at_misaligned_u32": "error",
    "refuse_overlap_placed_into_wide": "error",
    "refuse_overlap_same_start": "error",
    "refuse_divisor_lo_zero": "error",
    "refuse_divisor_hi_past": "error",
    "refuse_divisor_signed": "error",
    "refuse_multiple_none_in_range": "error",
    "refuse_multiple_zero": "error",
    "refuse_multiple_negative": "error",
    "refuse_multiple_bool": "error",
    "refuse_multiple_restated": "error",
    "refuse_range_reversed": "error",
    "refuse_mapped_0": "error",
    "refuse_mapped_9": "error",
    "refuse_peers_1": "error",
    "refuse_peers_16": "error",
    "refuse_uncached_scratch_in_block": "error",
    "refuse_uncached_mapped": "error",
    "refuse_uncached_twice": "error",
    "store_reads_load_in_flight": "render_error",
    "reload_same_destination_in_flight": "render_error",
    "part_wait_misses_register": "render_error",
    "scalar_reload_in_flight": "render_error",
    "refuse_layout_twice": "error",
    "refuse_budget_0": "error",
    "refuse_budget_65537": "error",
    "refuse_past_budget": "error",
    "refuse_past_target": "error",
    "refuse_final_after_size": "error",
    "refuse_final_after_base": "error",
    "refuse_duplicate_in_union": "error",
    "refuse_overlap_placed": "error",
    "refuse_shape_zero_dim": "error",
    "refuse_no_size": "error",
    "refuse_pitch_one_dim": "error",
    "refuse_pitch_short": "error",
    "refuse_pitch_unaligned": "error",
    "refuse_nbytes_below_shape": "error",
    "refuse_element_type_f32x2": "error",
    "refuse_element_type_pair": "error",
    "refuse_element_name_fp8": "error",
    "refuse_element_name_b16": "error",
    "refuse_swizzle_one_dim": "error",
    "refuse_swizzle_same_dims": "error",
    "refuse_swizzle_mask_7": "error",
    "refuse_swizzle_mask_5": "error",
    "refuse_parts_2_names": "error",
    "refuse_parts_4_names": "error",
    "refuse_parts_along_swizzle_key": "error",
    "refuse_too_many_indices": "error",
    "refuse_pointer_move_by_formula": "error",
    "refuse_pointer_move_breaks_swizzle": "error",
    "refuse_pointer_subarray_element_swizzled": "error",
    "refuse_address_plus_address": "error",
    "refuse_register_address_with_layout": "error",
    "refuse_region_of_other_builder": "error",
    "refuse_bool_byte_offset": "error",
    "refuse_vector_term_in_scalar_address": "error",
    "refuse_global_offset_below": "error",
    "refuse_global_offset_above": "error",
    "refuse_global_pointer_minus": "error",
}


def _check_outcomes(cases, default):
    for case in cases:
        stage = next((key for key in ("error", "render_error", "render") if key in case), "block")
        expected = EXPECTED_FAILURES.get(case["name"], default)
        if stage != expected:
            raise AssertionError(f"{case['name']}: planned {expected}, got {stage}")


def generate() -> dict[str, bytes]:
    """Evaluate and validate all authored entry and LDS plans."""
    with _no_fixture_reads():
        entry = {
            "notation": NOTATION,
            "args_notation": ARGS_NOTATION,
            "cases": [snippet_case(n, s) for n, s in _unique(entry_plans())],
            "args": [argument_case(n, s) for n, s in _unique(argument_plans())],
        }
        lds = {
            "notation": NOTATION,
            "cases": [snippet_case(n, s) for n, s in _unique(lds_plans())],
        }
    _check_outcomes(entry["cases"], "render")
    _check_outcomes(entry["args"], "block")
    _check_outcomes(lds["cases"], "render")
    return {ENTRY: encode(entry), LDS: encode(lds)}
