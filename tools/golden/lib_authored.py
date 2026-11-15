"""Library product and hand-off protocol programs.

Streams, events and refusal diagnostics come from executing the programs
through thin and inspecting the settled native Builder instructions.
"""

from __future__ import annotations

import builtins
import contextlib
import io
from pathlib import Path

from tools.golden import encode

BASE = "tests/data/lib/"
OUTPUTS = (BASE + "gfx1201.json", BASE + "traces.json")


def _src(*lines):
    """One program from its lines."""
    return "\n".join(lines)


def _hand_off(per, words, wgsize, *, at, gname="gw", box=None, kernel=None, more=""):
    """A generation declaration, optionally a slot box, the entry and both begins.

    ``box`` is the Slots keyword text (without ``wgsize``); ``more`` adds
    entry arguments after the library's own.
    """
    lines = [f'Gw = xwg.Gens(name="{gname}", per="{per}", words={words}, wgsize={wgsize})']
    blocks = "**Gw.args()"
    if box is not None:
        lines.append(f"Box = xwg.Slots({box}, wgsize={wgsize})")
        blocks += ", **Box.args()"
    lines.append(f"kn = b.entry(Args({blocks}{more}), wgsize={kernel or wgsize})")
    lines.append(f"gen = Gw.begin(b, kn, at={at})")
    if box is not None:
        lines.append("box = Box.begin(b, kn, gen)")
    return _src(*lines)


# ----------------------------------------------------------- shared programs

GENS_MEMBER = _src(
    _hand_off("member", 6, 64, at="kn.wgid"),
    "far = gen.tag(ahead=2)",
    "near = gen.tag()",
    "low12 = gen.tag_bits(12, 20)",
    'mix = S.u32(far ^ near, name="mix")',
)

GENS_PARTY = _src(
    _hand_off("party", 3, 128, at="2"),
    'early = S.u32(gen.tag(ahead=1), name="early")',
    "gen.advance()",
)

LAPS_SUM = _src(
    'Lp = Laps(["fetch", "fold", "spill"], wgsize=96, keep="sum")',
    "kn = b.entry(Args(**Lp.args()), wgsize=96)",
    "run = Lp.begin(b, kn)",
    'run.lap("fetch")',
    'run.lap("fold")',
    "run.record()",
    'run.lap("fetch")',
    'run.lap("spill")',
    "run.end()",
)

SLOTS_PUT = _src(
    _hand_off(
        "member", 3, 64, at="kn.wgid", box='name="box", slots=3, pieces=32, piece=16, tag="dword"'
    ),
    "pc = box.piece()",
    "pc[0] = kn.tid",
    "pc[1] = 9",
    "pc[2] = kn.wgid",
    "box.put(2, kn.tid >> 1, pc)",
)

COUNTS = _src(
    'Cn = xwg.Counts(name="tally", wgsize=128)',
    'total = S.u32(6, name="total")',
    "kn = b.entry(Args(**Cn.args()), wgsize=128)",
    "ct = Cn.begin(b, kn)",
    "got = ct.arrive(3, total)",
    "with b.when(got.last == 1):",
    "    ct.reset(3, got)",
    "gone = ct.leave(1, total)",
)

_COUNTS_ONLY = _src(
    'Cn = xwg.Counts(name="tally", wgsize=64)',
    "kn = b.entry(Args(**Cn.args()), wgsize=64)",
    "ct = Cn.begin(b, kn)",
)


def _poll_refusal(per, box, call, *, at="kn.wgid", wgsize=64):
    """A poll whose arguments the library refuses (or, at a limit, accepts)."""
    return _src(_hand_off(per, 2, wgsize, at=at, box=box), call)


# ----------------------------------------------------------- gfx1201 products


def product_plans():
    """(name, program) pairs whose products and renders are recorded."""
    plans = [
        ("gens_member_six_words_two_waves", GENS_MEMBER),
        (
            "gens_member_wgsize192",
            _src(
                _hand_off("member", 2, 192, at="kn.wgid + 1"),
                'stamp = V.u32(gen.tag(), name="stamp")',
            ),
        ),
        ("gens_party_advance", GENS_PARTY),
        (
            "gens_tag_ablated",
            _src(
                'b.ablate = {"xwg_tag"}',
                _hand_off("party", 2, 64, at="1"),
                "whole = gen.tag(ahead=2)",
                "wide = gen.tag_bits(16, 0)",
                'both = S.u32(whole + wide, name="both")',
            ),
        ),
        (
            "gens_tags_cached",
            _src(
                _hand_off("party", 1, 32, at="0"),
                "first8 = gen.tag_bits(8, 4)",
                "again8 = gen.tag_bits(8, 4)",
                "assert first8 is again8",
                "next1 = gen.tag(ahead=1)",
                "assert next1 is not first8",
                "assert gen.tag(ahead=1) is next1",
            ),
        ),
        (
            "laps_off",
            _src(
                (
                    'Lp = Laps(["load", "math", "store"], '
                    'wgsize=128, keep="last", records=8, '
                    "on=False)"
                ),
                "kn = b.entry(Args(**Lp.args()), wgsize=128)",
                "run = Lp.begin(b, kn)",
                'acc = V.u32(kn.tid * 3, name="acc")',
                'for step in ("store", "load", "math"):',
                "    run.lap(step)",
                "run.record()",
                'run.lap("math")',
                "run.end()",
            ),
        ),
        ("laps_on_sum", LAPS_SUM),
        (
            "laps_off_unknown_lap",
            _src(
                'Lp = Laps(["fetch", "fold"], wgsize=32, on=False)',
                "kn = b.entry(Args(**Lp.args()), wgsize=32)",
                "run = Lp.begin(b, kn)",
                'run.lap("fold")',
                'run.lap("flush")',
            ),
        ),
        ("slots_put_dword", SLOTS_PUT),
        ("counts", COUNTS),
        (
            "comm_put",
            _src(
                "kn = b.entry(Args(peers=mapped(3)), wgsize=64)",
                '(boxes,) = kn.load("peers")',
                "wd = comm.word(b)",
                "wd[0] = kn.tid",
                "wd[1] = kn.wgid",
                'stamp = S.u32(11, name="stamp")',
                'at = V.u32(kn.tid * 16, name="at")',
                "comm.put(b, comm.tagged(wd, stamp), boxes, at, offset=64)",
            ),
        ),
        (
            "folds_refused",
            _src(
                'p, q = V.u32(name="p"), V.u32(name="q")',
                's = V.u32(name="s")',
                "s[...] = bf16_sum(b, [q, p])",
            ),
        ),
        (
            "unbounded_poll_needs",
            _src(
                _hand_off("party", 2, 64, at="1", box='slots=2, pieces=32, piece=8, tag="dword"'),
                "got = box.poll([(1, kn.tid >> 1, None), (0, kn.tid >> 1, kn.tid < 32)], None)",
            ),
        ),
        (
            "advance_member",
            _src(
                _hand_off("member", 5, 96, at="kn.wgid"),
                "t = gen.tag()",
                "gen.advance()",
            ),
        ),
        ("wgsize_mismatch", _hand_off("member", 1, 256, at="kn.wgid", kernel=160)),
        (
            "wgsize_mismatch_counts",
            _src(
                "Cn = xwg.Counts(wgsize=32)",
                "kn = b.entry(Args(**Cn.args()), wgsize=64)",
                "ct = Cn.begin(b, kn)",
            ),
        ),
        (
            "bits_tag_with_party",
            _hand_off("party", 4, 64, at="3", box='slots=1, pieces=16, piece=4, tag=("low", 9)'),
        ),
        (
            "publish_not_flag",
            _src(
                _hand_off(
                    "member",
                    2,
                    64,
                    at="kn.wgid",
                    box='name="box", slots=2, pieces=64, piece=8, tag=("high", 6)',
                ),
                "box.publish(1, lanes=kn.tid < 8)",
            ),
        ),
        (
            "peek_not_flag",
            _src(
                _hand_off(
                    "member", 2, 64, at="kn.wgid", box='slots=2, pieces=64, piece=16, tag="dword"'
                ),
                "box.peek([(1, kn.tid, None)])",
            ),
        ),
        (
            "load_not_flag",
            _src(
                _hand_off(
                    "member", 1, 64, at="kn.wgid", box='slots=3, pieces=64, piece=4, tag=("sign",)'
                ),
                "box.load([(0, kn.tid, kn.tid > 3)])",
            ),
        ),
        (
            "quad_lanes_not_callable",
            _src(
                _hand_off(
                    "member", 2, 64, at="kn.wgid", box='slots=2, pieces=8, piece=64, tag="dword"'
                ),
                "pc = box.piece()",
                "box.put(1, kn.lane >> 2, pc, lanes=kn.lane >= 20)",
            ),
        ),
        (
            "callable_lanes_not_quad",
            _src(
                _hand_off(
                    "member", 2, 64, at="kn.wgid", box='slots=2, pieces=64, piece=16, tag="flag"'
                ),
                "box.peek([(0, kn.tid, lambda quad: quad > 1)])",
            ),
        ),
    ]
    # Poll argument refusals, each on a different slot shape, and both spin limits accepted.
    flag8 = 'slots=3, pieces=64, piece=8, tag="flag"'
    word16 = 'slots=2, pieces=64, piece=16, tag="dword"'
    high3 = 'name="box", slots=2, pieces=64, piece=8, tag=("high", 3)'
    quads = 'slots=2, pieces=8, piece=64, tag="dword"'
    read = "[(1, kn.tid, None)]"
    plans += [
        (
            "all_arrived_with_spin",
            _poll_refusal("member", flag8, f"box.poll({read}, 3, all_arrived=True)"),
        ),
        (
            "on_timeout_unbounded",
            _poll_refusal("member", word16, f"box.poll({read}, None, on_timeout=lambda m: None)"),
        ),
        (
            "spin_too_small",
            _poll_refusal("member", high3, f'box.poll({read}, 1, on_timeout="leave")'),
        ),
        (
            "spin_lowest_accepted",
            _poll_refusal("member", high3, f'got, came = box.poll({read}, 2, on_timeout="leave")'),
        ),
        (
            "spin_bool",
            _poll_refusal("member", flag8, f'box.poll({read}, False, on_timeout="leave")'),
        ),
        (
            "spin_too_big",
            _poll_refusal("member", word16, f'box.poll({read}, 2**30, on_timeout="leave")'),
        ),
        (
            "spin_highest_accepted",
            _poll_refusal(
                "member", word16, f'got, came = box.poll({read}, 2**30 - 1, on_timeout="leave")'
            ),
        ),
        ("bound_without_timeout", _poll_refusal("member", high3, f"box.poll({read}, 12)")),
        (
            "timeout_word",
            _poll_refusal("member", flag8, f'box.poll({read}, 5, on_timeout="retry")'),
        ),
        (
            "steal_with_party",
            _poll_refusal(
                "party",
                quads,
                "box.poll([(0, kn.lane >> 2, None)], 9, on_timeout=lambda stale: None)",
                at="0",
            ),
        ),
        (
            "tag_first_made_in_an_arm",
            _src(
                _hand_off("party", 2, 64, at="1"),
                'sel = S.u32(name="sel", at=24)',
                "with b.when(sel != 0):",
                '    inner = S.u32(gen.tag(ahead=1) ^ 5, name="inner")',
                'outer = S.u32(gen.tag(ahead=1) + 3, name="outer")',
                "with b.when(sel == 7):",
                '    again = S.u32(gen.tag(ahead=1) - 1, name="again")',
            ),
        ),
        (
            "tag_first_made_in_a_lane_region",
            _src(
                _hand_off("member", 2, 64, at="kn.wgid"),
                "with b.lanes(kn.lane < 5):",
                '    inside = V.u32(gen.tag_bits(4, 8) | kn.tid, name="inside")',
                'after = V.u32(gen.tag_bits(4, 8) + kn.tid, name="after")',
                "with b.lanes(kn.lane > 9):",
                '    reused = V.u32(gen.tag_bits(4, 8) - kn.tid, name="reused")',
            ),
        ),
    ]
    return plans


# ----------------------------------------------------------- protocol traces


def trace_plans():
    """(name, program) pairs whose settled synchronization events are recorded."""
    quad_steal = _src(
        _hand_off(
            "member",
            2,
            64,
            at="kn.wgid",
            box='name="quads", slots=3, pieces=8, piece=64, tag="dword"',
        ),
        "pc = box.piece()",
        "pc[0] = kn.wgid",
        "pc[1] = 3",
        "pc[2] = 1",
        "pc[3] = 0",
        "box.put(kn.wgid, kn.lane >> 2, pc)",
        "def rescue(stale):",
        "    with b.lanes(stale[1]):",
        "        alt = box.piece()",
        "        alt[0] = 99",
        "        alt[1] = 0",
        "        alt[2] = 2",
        "        alt[3] = 0",
        "        box.put(2, kn.lane >> 2, alt)",
        "reads = [(0, kn.lane >> 2, None), (2, kn.lane >> 2, lambda quad: quad < 4)]",
        "got = box.poll(reads, 16, on_timeout=rescue, sleep=0)",
    )
    bits_leave = _src(
        _hand_off(
            "member", 1, 64, at="kn.wgid", box='slots=2, pieces=64, piece=8, tag=("low", 10)'
        ),
        "pc = box.piece()",
        "pc[0] = kn.tid + 7",
        "pc[1] = 0x400",
        "box.put(kn.wgid & 1, kn.tid, pc)",
        "reads = [(1, kn.tid, None), (0, kn.tid, kn.tid >= 16), (1, kn.tid ^ 1, None)]",
        'vals, came = box.poll(reads, 24, on_timeout="leave")',
    )
    flag_peek = _src(
        _hand_off(
            "member",
            2,
            64,
            at="kn.wgid",
            box='name="box", slots=2, pieces=64, piece=16, tag="flag"',
        ),
        "pc = box.piece()",
        "pc[0] = kn.tid",
        "pc[1] = 1",
        "pc[2] = 2",
        "pc[3] = kn.wgid",
        "box.put(kn.wgid, kn.tid, pc, lanes=kn.tid < 48)",
        "box.publish(kn.wgid)",
        'f0, f1 = V.u32(name="f0"), V.u32(name="f1")',
        "reads = [(0, kn.tid, None), (1, kn.tid, kn.tid < 32)]",
        "seen = box.peek(reads, into=[f0, f1])",
        "ready = box.fresh(seen)",
        "with b.when(ready == 1):",
        "    got = box.load(reads)",
    )
    flag_poll = _src(
        _hand_off("member", 3, 64, at="kn.wgid", box='slots=3, pieces=64, piece=4, tag="flag"'),
        "reads = [(2, kn.tid, kn.tid >= 8), (0, kn.tid, None)]",
        'vals, came = box.poll(reads, 6, on_timeout="leave", sleep=2)',
    )
    word8 = 'slots=2, pieces=64, piece=8, tag="dword"'
    unbounded = _src(
        _hand_off("member", 2, 64, at="kn.wgid", box=word8),
        "got = box.poll([(0, kn.tid, None), (1, kn.tid, None)], None)",
    )
    all_arrived = _src(
        _hand_off("member", 2, 64, at="kn.wgid", box=word8),
        "got = box.poll([(1, kn.tid, kn.tid < 40)], None, all_arrived=True)",
    )
    exchange = _src(
        _hand_off("member", 1, 64, at="kn.wgid", more=", feed=ptr, ring=mapped(3), me=u32"),
        'feed, ring, me = kn.load("feed", "ring", "me")',
        "w0, w1 = comm.word(b), comm.word(b)",
        'b.emit(O.global_load(comm.payload(w0), V.u32(kn.tid * 16, name="lo"), feed))',
        'b.emit(O.global_load(comm.payload(w1), V.u32(kn.tid * 16 + 8, name="hi"), feed))',
    )
    plans = [
        ("laps_sum", LAPS_SUM),
        (
            "laps_last",
            _src(
                'Lp = Laps(["warm", "run"], wgsize=64, keep="last", records=3)',
                "kn = b.entry(Args(**Lp.args()), wgsize=64)",
                "run = Lp.begin(b, kn)",
                'run.lap("warm")',
                'run.lap("run")',
                "run.record()",
                'run.lap("run")',
                "run.end()",
            ),
        ),
        (
            "laps_first",
            _src(
                'Lp = Laps(["copy"], wgsize=128, keep="first", records=2)',
                "kn = b.entry(Args(**Lp.args()), wgsize=128)",
                "run = Lp.begin(b, kn)",
                'run.lap("copy")',
                "run.record()",
                'run.lap("copy")',
                "run.end()",
            ),
        ),
        ("gens_member", GENS_MEMBER),
        ("gens_party", GENS_PARTY),
        ("slots_put_dword_tag", SLOTS_PUT),
        ("slots_put_dword_tag_no_coherent", _src('b.ablate = {"xwg_coherent"}', SLOTS_PUT)),
        ("slots_quad_put_poll_steal", quad_steal),
        ("slots_bits_poll_leave", bits_leave),
        ("slots_flag_publish_peek", flag_peek),
        ("slots_flag_poll", flag_poll),
        ("slots_unbounded", unbounded),
        ("slots_unbounded_all_arrived", all_arrived),
        ("counts", COUNTS),
        ("counts_ablated", _src('b.ablate = {"xwg_takeback", "xwg_release", "xwg_reset"}', COUNTS)),
        ("counts_arrivals_past_the_field", _src(_COUNTS_ONLY, "got = ct.arrive(2, 0x10000)")),
        ("counts_arrivals_at_the_field", _src(_COUNTS_ONLY, "got = ct.arrive(2, 0xFFFF)")),
        ("counts_leave_of_none", _src(_COUNTS_ONLY, "gone = ct.leave(1, 0)")),
        (
            "counts_arrive_issue_no_release",
            _src(
                _COUNTS_ONLY,
                "pend = ct.arrive_issue(2, release=False)",
                'busy = V.u32(kn.tid * 5 + 2, name="busy")',
                "got = ct.arrived(pend, 9)",
                "with b.when(got.last == 1):",
                "    ct.reset(2, got)",
            ),
        ),
        (
            "comm_ll_all",
            _src(
                exchange,
                "cov = [None, kn.tid < 16]",
                "got = comm.ll_all(b, ring, me, 2048, gen.g1, gen.tag(), 1, [w0, w1], cov)",
            ),
        ),
        (
            "comm_ll_all_past_the_slot",
            _src(
                exchange,
                "got = comm.ll_all(b, ring, me, 48, gen.g1, gen.tag(), 2, [w0, w1], [None, None])",
            ),
        ),
        (
            "laps_unknown_lap",
            _src(
                'Lp = Laps(["warm", "hot"], wgsize=64, keep="first")',
                "kn = b.entry(Args(**Lp.args()), wgsize=64)",
                "run = Lp.begin(b, kn)",
                'run.lap("hot")',
                'run.lap("cold")',
            ),
        ),
    ]
    return plans


# ----------------------------------------------------------- evaluation


def products():
    """Each product program built, lowered and rendered by thin."""
    from tools.golden.snippets import evaluate

    return {
        "notation": (
            "Library programs authored for this file, built on gfx1201: the Builder's "
            "statements, values and records, then the final rendering (or the refusal)."
        ),
        "cases": [evaluate(name, src, library=True) for name, src in product_plans()],
    }


def traces():
    """Each trace program's settled native events, or its refusal."""
    from tests import notation
    from tools.golden.protocol import events
    from tools.golden.snippets import error_record

    cases = []
    for name, src in trace_plans():
        case = {"name": name, "src": src}
        try:
            builder = notation.run_snippet(src, library=True)
            case["events"] = events(builder)
            if builder.needs:
                case["needs"] = builder.needs
        except Exception as error:  # the library's own refusals are recorded
            case["error"] = error_record(error)
        cases.append(case)
    return {
        "notation": (
            "Settled native Builder events of library hand-off, clock, count and "
            "communication programs on gfx1201; source positions are not recorded."
        ),
        "cases": cases,
    }


@contextlib.contextmanager
def _no_fixture_reads():
    """Refuse any open of checked-in test data while generating."""

    def refuse(target):
        """Construct refuse inputs."""
        text = Path(target).as_posix() if isinstance(target, (str, Path)) else ""
        if "tests/data/" in text:
            raise AssertionError(f"lib_authored generation opened checked-in data: {text}")

    real_open, real_io_open, real_path_open = builtins.open, io.open, Path.open

    def guarded_open(file, *args, **kwargs):
        """Construct guarded open inputs."""
        refuse(file)
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self, *args, **kwargs):
        """Construct guarded path open inputs."""
        refuse(self)
        return real_path_open(self, *args, **kwargs)

    builtins.open = io.open = guarded_open
    Path.open = guarded_path_open
    try:
        yield
    finally:
        builtins.open, Path.open = real_open, real_path_open
        io.open = real_io_open


def generate():
    """Evaluate the authored programs; never opens checked-in golden data."""
    with _no_fixture_reads():
        docs = (products(), traces())
    return {path: encode(doc) for path, doc in zip(OUTPUTS, docs, strict=True)}
