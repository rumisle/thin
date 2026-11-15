"""Unit tests of the language beyond the golden data."""

import dataclasses
import itertools

import numpy as np
import pytest

from tests import notation


def built(src):
    return notation.built(notation.run_snippet(src))


def test_identity_statements_emit_nothing():
    src = 'x = V.u32(name="x", at=0)\nx[...] = x\nx += 0\nx -= 0\nx <<= 0\nx |= 0\n'
    assert built(src) == []


def test_a_tree_written_twice_lowers_twice():
    src = 'x, y = V.u32(name="x", at=0), V.u32(name="y", at=1)\nt = x + 3\ny[...] = t\ny[...] = t\n'
    lines = built(src)
    assert len(lines) == 2 and lines[0] == lines[1]


def test_statement_comment_on_first_instruction_and_locs_everywhere():
    b = notation.run_snippet(
        'x, y = V.u32(name="x", at=0), V.u32(name="y", at=1)\ny[...] = (x // 7) % 5\n'
    )
    body = list(b.ins)
    assert len(body) > 2
    assert "comment" in notation.built_line(b, body[0])
    assert all(i.loc for i in body)


def test_loop_passes_make_the_same_groups():
    src = (
        "import dataclasses\n"
        "b.options = dataclasses.replace(b.options, audit=True)\n"
        'n = S.u32(name="n", at=0)\nacc = V.u32(0, name="acc")\n'
        "for lp in b.loop(n):\n"
        '    u = V.u32(lp.i * 3, name="u")\n'
        "    acc += u\n"
    )
    b = notation.run_snippet(src)
    names = [v.name for v in b.values]
    assert names.count("u#1") == 1 and "u#2" not in names
    assert len(b.program.loops) == 1


def test_audit_refuses_an_iteration_dependent_body():
    from thin.ir import ScheduleError

    src = (
        "import dataclasses\n"
        "b.options = dataclasses.replace(b.options, audit=True)\n"
        'n = S.u32(name="n", at=0)\nacc = V.u32(0, name="acc")\nk = [0]\n'
        "for lp in b.loop(n):\n"
        "    k[0] += 1\n"
        "    acc += k[0]\n"
    )
    with pytest.raises(ScheduleError, match="has no steady state"):
        notation.run_snippet(src)


@pytest.mark.parametrize("peel", [1, 2, 3])
def test_peeled_loops_record_k_plus_two_starts(peel):
    src = (
        'n = S.u32(name="n", at=0)\nacc = V.u32(0, name="acc")\n'
        f"for lp in b.loop(n, peel={peel}):\n    acc += 1\n"
    )
    b = notation.run_snippet(src)
    ((kind, record),) = b.deferred
    assert kind == "peel" and len(record["starts"]) == peel + 2 and record["max"] == peel


def test_arms_are_closed_after_every_construct():
    src = (
        's = S.u32(name="s", at=0)\n'
        "with b.when(s == 0):\n    b.nop(0)\n"
        "with b.elif_(s < 4):\n    b.nop(1)\n"
        "with b.otherwise():\n    b.nop(2)\n"
        "b.comment('after')\n"
    )
    lines = built(src)
    assert lines[-1] == "ir.comment {text=; after}"
    assert sum(line.startswith("ir.label") for line in lines) == 3


def test_issued_counts_the_current_path():
    src = (
        's = S.u32(name="s", at=0)\nd = S.u32(name="d", at=1)\np = S.ptr(name="p", at=2)\n'
        "assert b.issued('km') == 0\n"
        "with b.when(s == 0):\n"
        "    b.emit(O.s_load(d, p))\n"
        "    assert b.issued('km') == 1\n"
        "with b.otherwise():\n"
        "    assert b.issued('km') == 0\n"
    )
    notation.run_snippet(src)


@pytest.mark.parametrize(
    "src, cls",
    [
        ('f = V.f32(name="f")\nf & 1', "ExprError"),
        ('p = S.ptr(name="p")\np * 2', "ExprError"),
        ('x = V.u32(name="x")\nx // 0', "ExprError"),
        ('x = V.i32(name="x")\nx // 3', "ExprError"),
        ('x = V.u32(name="x")\nbool(x < 1)', "ExprError"),
        ('m = S.mask(name="m")\nm < 1', "ExprError"),
    ],
)
def test_typing_refusals(src, cls):
    with pytest.raises(Exception) as caught:
        notation.run_snippet(src)
    assert type(caught.value).__name__ == cls


def test_swizzle_is_its_own_inverse():
    b = notation.run_snippet(
        'L = b.lds()\nT = L.region("T", (8, 16), u32, swizzle=Xor(by=0, into=1, mask=7))\n'
    )
    t = b.layout.regions["T"]
    for i in range(8):
        for j in range(16):
            once = t.stored(i, j)
            assert t.stored(*once) == (i, j)


def test_offsets_match_an_index_calculation():
    b = notation.run_snippet('L = b.lds()\nA = L.region("A", (3, 5, 7), u32, pitch={-2: 32})\n')
    a = b.layout.regions["A"]
    strides = a.strides
    assert strides == (5 * 32, 32, 4)
    for idx in [(0, 0, 0), (1, 2, 3), (2, 4, 6)]:
        assert a.offset(*idx) == sum(i * s for i, s in zip(idx, strides, strict=True))


def test_argument_offsets_follow_alignment():
    from thin.dsl import Args, ptr, u32, u64

    block = Args(a=u32, b=ptr, c=u32, d=u64, e=u32)
    assert [block[n]["offset"] for n in block.names()] == [0, 8, 16, 24, 32]
    assert block.size == 36


def test_options_are_read_at_construction():
    from thin import codegen

    with codegen.using(audit=True):
        b = notation.run_snippet("")
    assert b.options.audit
    assert dataclasses.replace(b.options, audit=False).audit is False


def test_program_of_a_kernel_snapshots_its_stream():
    from thin.ir import program

    b = notation.run_snippet('x = V.u32(name="x", at=0)\nwith b.lanes(x < 3):\n    x += 1\n')
    snap = program(b)
    assert len(snap.rows) == len(b.ins) and snap.wave_classes() == [None]


def test_ablate_is_a_live_set():
    src = (
        'x, y = V.u32(name="x", at=0), V.u32(name="y", at=1)\n'
        'b.emit(O.op("v_mov_b32", y, x, abl=["epi"]))\n'
        'b.ablate.add("epi")\n'
        'b.emit(O.op("v_mov_b32", y, x, abl=["epi"]))\n'
        "assert b.ablated(['epi'])\n"
        'b.ablate.discard("epi")\n'
        "assert not b.ablated(['epi'])\n"
        'b.emit(O.op("v_mov_b32", y, x, abl=["epi"]))\n'
        'b.ablate = {"other"}\n'
        'b.emit(O.op("v_mov_b32", y, x, abl=["epi"]))\n'
    )
    assert len(built(src)) == 3


# --- Argument blocks ---------------------------------------------

_KINDS = {
    "u32": ("u32", 1, None),
    "i32": ("i32", 1, None),
    "f32": ("f32", 1, None),
    "ptr": ("ptr", 2, None),
    "u64": ("u64", 2, None),
    "div": ("u32.within(1, 99).divisor()", 1, "divisor"),
    "map2": ("mapped(2)", 2, 2),
    "map3": ("mapped(3)", 2, 3),
    "kept": ("kept", 2, None),
}


def _bytes(kind) -> int:
    _, dwords, extra = _KINDS[kind]
    if extra == "divisor":
        return 16
    return 4 * dwords * ((extra + 1) if isinstance(extra, int) else 1)


def _layout(names, kinds, at):
    """offsets, or the refusal's start."""
    running, out = 0, {}
    for n, kind in zip(names, kinds, strict=True):
        size = _bytes(kind)
        align = min(size, 8)
        off = at.get(n, -(-running // align) * align)
        if off % align:
            return f"argument {n}: at byte {off}, a "
        out[n] = (off, size)
        running = off + size
    if set(at) - set(names):
        return "at= names no argument"
    placed = sorted(out.items(), key=lambda kv: kv[1][0])
    for (a, (a0, an)), (b2, (b0, _)) in itertools.pairwise(placed):
        if b0 < a0 + an:
            return f"arguments {a} and {b2} overlap"
    return out


def _chunks(layout):
    """the s_load pieces (offset, bytes) of every chunk."""
    order = sorted(layout.items(), key=lambda kv: kv[1][0])
    runs, run = [], []
    for name, (off, _size) in order:
        if run and layout[run[-1]][0] + layout[run[-1]][1] != off:
            runs.append(run)
            run = []
        run.append(name)
    runs.append(run)
    pieces = []
    for run in runs:
        j = 0
        while j < len(run):
            start = layout[run[j]][0]
            ends = [layout[n][0] + layout[n][1] for n in run[j:]]
            size = next((c for c in (64, 32, 16, 8, 4) if start + c in ends), layout[run[j]][1])
            end = start + size
            k = j
            while k < len(run) and layout[run[k]][0] + layout[run[k]][1] <= end:
                k += 1
            at = 0
            while at < size:
                piece = next(c for c in (64, 32, 16, 8, 4) if at + c <= size)
                pieces.append((start + at, piece))
                at += piece
            j = k
    return pieces


def _random_block(rng):
    n = int(rng.integers(1, 9))
    names = [f"a{i}" for i in range(n)]
    kinds = [str(rng.choice(list(_KINDS))) for _ in names]
    at = {}
    if rng.random() < 0.5:
        n_at = int(rng.integers(1, 3))
        for name in rng.choice(names, size=min(n_at, n), replace=False):
            at[str(name)] = int(rng.integers(0, 20)) * 4
    if rng.random() < 0.05:
        at["nobody"] = 0
    return names, kinds, at


def test_random_argument_blocks_lay_out_and_load_as_specified():
    """random blocks of kinds and at= placements: offsets, sizes and
    refusals; the loads are greedy chunks, a divisor and a mapped
    argument whole."""
    from thin import dsl

    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    rng = np.random.default_rng(11)
    refused = placed = 0
    for _ in range(200):
        names, kinds, at = _random_block(rng)
        src = ", ".join(f"{n}={_KINDS[k][0]}" for n, k in zip(names, kinds, strict=True))
        want = _layout(names, kinds, at)
        try:
            block = eval(f"Args(at={at!r}, {src})", scope)
        except ValueError as e:
            assert isinstance(want, str), (src, at, e)
            assert str(e).startswith(want), (src, at, str(e), want)
            refused += 1
            continue
        assert isinstance(want, dict), (src, at, want)
        assert {n: (block[n]["offset"], block[n]["bytes"]) for n in names} == want
        b = notation.run_snippet(f"k = b.entry(Args(at={at!r}, {src}), wgsize=32)\nk.load()")
        loads = [line for line in notation.built(b) if line.startswith("s_load_")]
        got = [(int(line.split(", ")[2].split("'")[0]), _width(line)) for line in loads]
        assert sorted(got) == sorted(_chunks(want)), (src, at)
        placed += 1
    assert refused > 10 and placed > 100


def _width(line: str) -> int:
    return int(line.split()[0].removeprefix("s_load_b")) // 8


# --- Layouts ----------------------------------------------------------

_ELEMS = {"u8": 1, "f16": 2, "f32": 4, "u64": 8, "b128": 16}


def _random_tree(rng, depth, names):
    """A random group: ("struct" | "union", align, at, members); a region is
    ("region", name, shape, elem, align, at, pitch)."""
    members = []
    for _ in range(int(rng.integers(1, 4))):
        if depth > 0 and rng.random() < 0.3:
            kind = str(rng.choice(["struct", "union"]))
            align = [None, 4, 64][int(rng.integers(0, 3))]
            members.append((kind, align, None, _random_tree(rng, depth - 1, names)))
        else:
            name = f"r{next(names)}"
            dims = int(rng.integers(1, 3))
            shape = tuple(int(v) for v in rng.integers(1, 6, dims))
            elem = str(rng.choice(list(_ELEMS)))
            align = [None, 4, 8, 32][int(rng.integers(0, 4))]
            at = int(rng.integers(0, 16)) * 16 if rng.random() < 0.1 else None
            pitch = None
            if dims == 2 and rng.random() < 0.3:
                pitch = shape[1] * _ELEMS[elem] + _ELEMS[elem] * int(rng.integers(0, 3))
            members.append(("region", name, shape, elem, align, at, pitch))
    return members


def _place(members, start, union, out, unions, path):
    """C's rules: returns the group's end; out: name -> (base, size,
    the (union, arm) pairs above it); `unions` numbers the unions."""
    end, cursor = start, start
    uid = next(unions) if union else None
    for k, m in enumerate(members):
        here = start if union else cursor
        branch = (*path, (uid, k)) if union else path
        if m[0] == "region":
            _, name, shape, elem, align, at, pitch = m
            e = _ELEMS[elem]
            strides = [e]
            for d in range(len(shape) - 1, 0, -1):
                strides.insert(0, shape[d] * strides[0])
            if pitch is not None:
                strides[0] = pitch
            size = shape[0] * strides[0]
            a = align or 16
            base = at if at is not None else -(-here // a) * a
            out[name] = (base, size, branch)
            m_end = base + size
        else:
            kind, align, at, sub = m
            a = align or max([_align_of(x) for x in sub] or [1])
            base = at if at is not None else -(-here // a) * a
            m_end = _place(sub, base, kind == "union", out, unions, branch)
        end = max(end, m_end)
        if not union:
            cursor = max(cursor, m_end)
    return end


def _align_of(m) -> int:
    if m[0] == "region":
        return m[4] or 16
    return m[1] or max([_align_of(x) for x in m[3]] or [1])


def _overlaps(out) -> bool:
    items = list(out.values())
    for i, (a0, an, ap) in enumerate(items):
        for b0, bn, bp in items[i + 1 :]:
            if a0 < b0 + bn and b0 < a0 + an:
                common = 0
                while common < min(len(ap), len(bp)) and ap[common] == bp[common]:
                    common += 1
                apart = common < min(len(ap), len(bp)) and ap[common][0] == bp[common][0]
                if not apart:
                    return True  # not in two arms of one union
    return False


def _declare(layout, members):
    regions = {}
    for m in members:
        if m[0] == "region":
            _, name, shape, elem, align, at, pitch = m
            regions[name] = layout.region(name, shape, elem, align=align, at=at, pitch=pitch)
        else:
            kind, align, at, sub = m
            with getattr(layout, kind)(align=align, at=at):
                regions.update(_declare(layout, sub))
    return regions


def test_random_layouts_place_as_cs_rules():
    """random trees of structs, unions and regions with align, at
    and pitch: every region's base and the end equal a reference of
    rules, and the overlap refusal fires exactly for overlapping non-arms."""
    import itertools

    from thin import dsl
    from thin.target import targeting

    rng = np.random.default_rng(5)
    refused = placed = 0
    for _ in range(150):
        tree = _random_tree(rng, 2, itertools.count())
        want: dict = {}
        end = _place(tree, 0, False, want, itertools.count(), ())
        with targeting("gfx1201"):
            b = dsl.Builder()
            layout = b.lds()
            regions = _declare(layout, tree)
            if _overlaps(want):
                with pytest.raises(Exception, match="overlap and are not arms of one union"):
                    layout.size  # noqa: B018 - freezes
                refused += 1
                continue
            assert layout.size == end
            assert {n: r.base for n, r in regions.items()} == {n: v[0] for n, v in want.items()}
            placed += 1
    assert placed > 100 and refused > 3, (placed, refused)


def test_a_wait_on_two_counters_says_its_comment_once():
    from thin import codegen

    b = notation.run_snippet(
        'src = S.ptr(name="src", at=2)\nidx = V.u32(name="idx", at=1)\n'
        'lo, cnt = V.f32(name="lo"), S.u32(name="cnt")\n'
        "b.emit(O.s_load(cnt, src, offset=12))\nb.emit(O.global_load(lo, idx, src))\n"
        'b.wait(lo, cnt, "lo and cnt")\nlo[...] = lo + f32(cnt)\n'
    )
    waits = [x for x in codegen.render(b) if x.lstrip().startswith("s_wait_")]
    assert [x.split("//")[1].strip() for x in waits] == ["lo and cnt", "lo and cnt"]


@pytest.mark.parametrize(
    "src, message",
    [
        (
            'a, c = S.f32(name="a", at=2), S.f32(name="c", at=3)\nS.f32(a + c)',
            "a + c: of scalar operands, but the DSL computes it on the VALU (f32 arithmetic, "
            "conversions and lane functions are VALU ops): write it into a V value, and one "
            "lane of that into an SGPR with first_lane(x)",
        ),
        (
            'a, c = S.u32(name="a", at=2), S.u32(name="c", at=3)\nS.f32(cvt_f32(a))',
            "cvt_f32(a): of scalar operands, but the DSL computes it on the VALU (f32 "
            "arithmetic, conversions and lane functions are VALU ops): write it into a V "
            "value, and one lane of that into an SGPR with first_lane(x)",
        ),
        (
            'a, c = V.f32(name="a", at=2), V.f32(name="c", at=3)\nV.u32(~(a + c))',
            "~(a + c): an integer's bits or a condition",
        ),
        (
            'a = V.u32(name="a", at=2)\nV.u32(abs(a))',
            "|a|: a float's (an i32's: max(x, 0 - x))",
        ),
        (
            'a = V.i32(name="a", at=2)\nV.i32((a + 1) % 3)',
            "(a + 1) % 3: a u32's (cast: u32(x))",
        ),
    ],
)
def test_expression_refusals_name_what_was_written(src, message):
    from thin.ir import ExprError

    with pytest.raises(ExprError) as e:
        notation.run_snippet(src)
    assert e.value.msg == message


@pytest.mark.parametrize(
    "src, message",
    [
        (
            'k = S.u32(name="k", at=1)\nb.emit(O.op("s_getreg_b32", k, "hwreg(HW_REG_MODE)"))',
            "s_getreg_b32: operand 1 'hwreg(HW_REG_MODE)': not a register's name, a label or a "
            "number's text (an immediate such as hwreg(..) is given as its encoded int)",
        ),
        (
            'u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)\n'
            'b.emit(O.op("v_mul_lo_u32", u, w, w, 3))',
            "`v_mul_lo_u32 v:u#1, v:w#1, v:w#1, 3`: no gfx1201 form has this mnemonic and "
            "operand count (the target's operand table)",
        ),
    ],
)
def test_maker_refusals_name_the_operands(src, message):
    with pytest.raises(ValueError) as e:
        notation.run_snippet(src)
    assert str(e.value) == message


def test_a_barrier_without_a_comment_is_named_by_its_instruction():
    from thin import codegen
    from thin.ir import MemError

    b = notation.run_snippet(
        'k = b.entry(Args(), wgsize=64)\nT = b.lds().region("T", (64,), "u32")\n'
        "b.emit(O.ds_write(T.at(k.tid), k.tid))\nb.barrier()\n"
        'y = V.u32(name="y")\nb.emit(O.ds_read(y, T.at(63 - k.tid)))\nb.wait(y)\n'
    )
    with pytest.raises(MemError, match=r"in flight at the barrier \(instruction 3\), and"):
        codegen.render(b)


def test_wait_on_use_waits_for_a_vector_load_at_its_first_read():
    from thin import codegen

    b = notation.run_snippet(
        'src = S.ptr(name="src", at=2)\nidx = V.u32(name="idx", at=1)\n'
        'lo, hi = V.f32(name="lo"), V.f32(name="hi")\n'
        "b.emit(O.global_load(lo, idx, src))\nb.emit(O.global_load(hi, idx, src, offset=4))\n"
        'b.wait_on_use([lo, hi], "at use")\nlo[...] = lo * 2.0\nhi[...] = hi + lo\n'
    )
    lines = [x.split("//")[0].strip() for x in codegen.render(b) if "s_delay" not in x]
    assert lines[2:] == [
        "s_wait_loadcnt 1",
        "v_mul_f32 v0, 2.0, v0",
        "s_wait_loadcnt 0",
        "v_add_f32 v2, v2, v0",
    ]


def test_a_loop_count_that_is_an_expression_is_refused():
    from thin.ir import FlowError

    src = 'n = S.u32(name="n", at=0)\nfor it in b.loop(n - 1):\n    pass\n'
    with pytest.raises(FlowError) as e:
        notation.run_snippet(src)
    assert e.value.msg == (
        "loop: n - 1: a count or a start is an int or a scalar u32 value (an expression "
        "is written into one first: S.u32(n - 1))"
    )


def test_the_race_check_reads_an_lds_access_s_offset_field():
    """Wave 0 stores T[1][tid] (its offset field 256) and wave 1 loads
    T[0][63 - tid]: apart, not a race. Wave 0 stores T[0][tid] + 128 (field
    128) where wave 1 loads T[0][tid]: a race the address registers alone
    ([0, 128) and [128, 256)) do not show."""
    from thin import codegen, dsl
    from thin.ir import MemError

    def compile_of(store_at, load_at):
        def build():
            b = dsl.Builder()
            k = b.entry(dsl.Args(), wgsize=64)
            T = b.lds().region("T", (2, 64), "u32")  # noqa: N806
            b.emit(dsl.O.ds_write(store_at(T, k.tid), k.tid))
            y = b.v.u32(name="y")
            b.emit(dsl.O.ds_read(y, load_at(T, k.tid)))
            b.wait(y)
            return b

        return codegen.compile_kernel(codegen.derive(build), "two", wgsize=64, max_grid=(1,))

    compile_of(lambda T, t: T.at(1, t), lambda T, t: T.at(0, 63 - t))  # noqa: N803
    with pytest.raises(MemError, match=r"in wave 0 and .* in wave 1 may touch the same bytes"):
        compile_of(lambda T, t: T.at(0, t) + 128, lambda T, t: T.at(0, t))  # noqa: N803
