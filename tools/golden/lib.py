"""Library boundary declarations and synthetic generation/clock protocols."""

from __future__ import annotations

import numpy as np

from thin.dsl import Args
from thin.lib import Laps, xwg
from tools.golden import encode

BASE = "tests/data/lib/"
OUTPUTS = tuple(
    BASE + name
    for name in (
        "declarations.json",
        "tags.json",
        "counts.json",
        "laps_decode.json",
        "gfx1201.json",
        "traces.json",
    )
)
DEFAULTS = dict(
    lo=None, hi=None, divisor=False, multiple=1, owned=None, library=None, mem=None, ranks=None
)


def argument_records(decl):
    """Serialize actual argument records, omitting only the API defaults."""
    args = Args(**decl.args())
    return [
        {k: v for k, v in args[name].items() if DEFAULTS.get(k, ...) != v} for name in args.names()
    ]


def declaration_case(call, kwargs, workgroups=None):
    """Evaluate a constructor, including exact refusal classes/messages."""
    case = {"call": call}
    if kwargs is not None:
        case["kw"] = kwargs
    if workgroups is not None:
        case["workgroups"] = workgroups
    try:
        if call.startswith("Laps(['a'], wgsize=64, on=False)"):
            Laps(["a"], wgsize=64, on=False).decode(b"", workgroups=1)
        elif call.startswith("Laps(['a'], 64, keep='sum').decode"):
            Laps(["a"], 64, keep="sum").decode(np.zeros(252, np.uint8), workgroups=1)
        elif call == "Counts(wgsize=64).sizes(0)":
            xwg.Counts(wgsize=64).sizes(0)
        else:
            kw = dict(kwargs)
            if isinstance(kw.get("tag"), list):
                kw["tag"] = tuple(kw["tag"])
            decl = {"Laps": Laps, "Gens": xwg.Gens, "Slots": xwg.Slots, "Counts": xwg.Counts}[call](
                **kw
            )
            if call == "Laps":
                case.update(
                    waves=decl.waves,
                    rows=decl.rows,
                    args=argument_records(decl),
                    sizes=decl.sizes(workgroups),
                )
            elif call == "Gens":
                case.update(waves=decl.waves, args=argument_records(decl), sizes=decl.sizes())
            elif call == "Counts":
                case.update(args=argument_records(decl))
                case["sizes(3)"] = decl.sizes(3)
            else:
                for field in ("kind", "stride", "flags_at", "lane_dwords"):
                    case[field] = getattr(decl, field)
                case.update(bytes=decl.bytes(), args=argument_records(decl), sizes=decl.sizes())
                if decl.kind == "bits":
                    case.update(
                        bits=decl.bits, at=decl.at, mask=hex(decl.mask), keep=hex(decl.keep)
                    )
    except Exception as error:  # generator records refusals rather than guessing them
        case["error"] = {
            "class": type(error).__name__,
            "message": getattr(error, "msg", error.args[0] if error.args else ""),
        }
    return case


def declarations():
    """Construct valid declarations and single-field perturbations at each limit."""
    laps = []
    plans = [
        ({"names": ["a", "b"], "wgsize": 64, "keep": "sum"}, 3),
        ({"names": ["a", "b"], "wgsize": 256, "keep": "last", "records": 8}, 2),
        ({"names": ["x"], "wgsize": 128, "keep": "first"}, 1),
        ({"names": ["x"], "wgsize": 64, "on": False}, 4),
        ({"names": [f"l{i}" for i in range(23)], "wgsize": 64, "keep": "sum"}, 1),
    ]
    for kwargs, groups in plans:
        laps.append(declaration_case("Laps", kwargs, groups))
    base = {"names": ["a"], "wgsize": 64}
    for delta in (
        {"keep": "max"},
        {"records": 0},
        {"records": 65},
        {"wgsize": 48},
        {"wgsize": 16},
        {"names": []},
        {"names": ["a"] * 2},
        {"names": [f"l{i}" for i in range(56)], "keep": "sum"},
    ):
        laps.append(declaration_case("Laps", {**base, **delta}))
    for call in (
        "Laps(['a'], wgsize=64, on=False).decode(b'', workgroups=1)",
        "Laps(['a'], 64, keep='sum').decode(<a uint8 array of 252 zero bytes>, workgroups=1)",
    ):
        laps.append(declaration_case(call, None))
    gens = []
    for kw in (
        dict(per="member", words=4, wgsize=64),
        dict(per="member", words=3, wgsize=256),
        dict(per="party", words=2, wgsize=256),
        dict(name="g2", per="party", words=1, wgsize=1024),
    ):
        gens.append(declaration_case("Gens", kw))
    base = dict(per="member", words=1, wgsize=64)
    for delta in (dict(per="wave"), dict(words=0), *(dict(wgsize=n) for n in (16, 2048, 100))):
        gens.append(declaration_case("Gens", {**base, **delta}))
    slots = []
    valid = (
        dict(slots=4, pieces=64, piece=8, tag="dword", wgsize=64),
        dict(slots=4, pieces=16, piece=64, tag="dword", wgsize=64),
        dict(slots=3, pieces=10, piece=16, tag="flag", wgsize=128),
        dict(slots=16, pieces=1, piece=8, tag="flag", wgsize=64, stride=8),
    )
    for kw in valid:
        slots.append(declaration_case("Slots", kw))
    for tag, piece, extra in (
        (["sign"], 4, {}),
        (["high", 14], 4, {}),
        (["low", 1], 16, {"uncached": True}),
    ):
        slots.append(
            declaration_case(
                "Slots", dict(slots=2, pieces=64, piece=piece, tag=tag, wgsize=64, **extra)
            )
        )
    slots.append(
        declaration_case(
            "Slots",
            dict(name="s2", slots=5, pieces=3, piece=4, tag=["low", 16], wgsize=64, stride=12),
        )
    )
    base = dict(slots=1, pieces=1, piece=4, tag="dword", wgsize=64)
    for delta in (
        {},
        dict(piece=64, tag="flag"),
        dict(piece=32),
        *(dict(tag=["low", n]) for n in (20, 0)),
        dict(tag=["mid", 3]),
        dict(tag=["sign", 1]),
        dict(tag="bits"),
        dict(slots=0, piece=8),
        dict(pieces=0, piece=8),
        dict(piece=8, wgsize=48),
    ):
        slots.append(declaration_case("Slots", {**base, **delta}))
    for stride in (16, 36):
        slots.append(
            declaration_case(
                "Slots", dict(slots=2, pieces=4, piece=8, tag="dword", wgsize=64, stride=stride)
            )
        )
    counts = [
        declaration_case("Counts", dict(wgsize=64)),
        declaration_case("Counts(wgsize=64).sizes(0)", None),
        declaration_case("Counts", dict(wgsize=65)),
    ]
    return {
        "notation": (
            "Constructor products and invalid single-field boundaries, "
            "evaluated through the library API."
        ),
        "laps": laps,
        "gens": gens,
        "slots": slots,
        "counts": counts,
    }


def tags():
    """Five launches straddling the generation-word wrap."""
    cases = []
    for launch in range(5):
        g = (xwg.GEN_START + launch) % 2**32
        g1 = (g + 1) % 2**32
        case = dict(
            launch=launch, g=g, g1=g1, tag=xwg.full_tag(g1), tag_ahead1=xwg.full_tag(g1, ahead=1)
        )
        for bits, at in ((1, 31), (14, 18), (1, 0), (16, 0)):
            case[f"tag_bits({bits},{at})"] = hex(xwg.bits_tag(g1, bits, at))
        cases.append(case)
    return {"notation": "Generation and tag helper results across the 32-bit wrap.", "cases": cases}


def counts():
    """Atomic arrival/leave/reset schedules, including early and late leaves."""
    arrive = lambda who: ("arrive", who)  # noqa: E731 - local schedule notation
    reset = lambda who: ("reset", who)  # noqa: E731
    leave = lambda who: ("leave", who)  # noqa: E731
    basic = [arrive(i) for i in range(3)]
    plans = [
        (3, [*basic, reset(2)]),
        (3, [basic[0], leave(0), *basic[1:], reset(2)]),
        (3, [*basic, leave(0), reset(2)]),
        (3, [*basic, reset(2), leave(0)]),
        (2, [arrive(1), arrive(0), leave(1), reset(0)]),
    ]
    cases = []
    for n, plan in plans:
        word, seen, trace = 0, {}, []
        for op, who in plan:
            step = dict(who=who, op=op)
            old = word
            if op == "arrive":
                word = (word + (1 << 16)) % 2**32
                last, left = xwg.arrival_fields(old, n)
                seen[who] = old
                step["returns"] = dict(old=old, last=last, left=left)
            elif op == "leave":
                word = (word + 1) % 2**32
                mine = xwg.leave_mine(old, n)
                step.update(returns=dict(old=old, mine=mine), taken_back=bool(mine))
                if mine:
                    word = (word - 1) % 2**32
            else:
                word = (word + xwg.reset_amount(seen[who])) % 2**32
            step["word"] = word
            trace.append(step)
        cases.append(dict(n=n, trace=trace, end_word=word))
    return {
        "notation": "Arrival/leave/reset schedules evaluated with library count helpers.",
        "cases": cases,
    }


def laps_decode():
    """Synthetic headers and lap samples, including both clock and ring wraps."""
    cases = []

    def case(name, kwargs, groups):
        """Construct case inputs."""
        c = dict(name=name, kw=kwargs, workgroups=groups, words={})
        cases.append(c)
        return c["words"]

    def header(words, at, rounds, core_start, real_start, core_end, real_end, sums=()):
        """Construct header inputs."""
        values = [rounds]
        for value in (core_start, real_start, core_end, real_end):
            values.extend((value % 2**32, value // 2**32))
        values.extend(sums)
        words.update({str(at + i): value for i, value in enumerate(values)})

    words = case("sum_two_workgroups", dict(names=["a", "b"], wgsize=32, keep="sum"), 2)
    header(words, 0, 11, 1000, 50000, 11000, 260000, (1500, 700))
    header(words, 32, 11, 2000, 2**32 - 256, 12000, 2**32 - 256 + 190000, (1900, 650))
    words = case("last_wrapped", dict(names=["a", "b"], wgsize=32, keep="last", records=4), 1)
    header(words, 0, 6, 0, 0, 1000, 15000)
    for row, values in enumerate(((900, 0, 500, 700), (980, 0, 590, 750)), 1):
        words.update({str(row * 32 + i): v for i, v in enumerate(values)})
    words = case("first_never_taken", dict(names=["a", "b"], wgsize=64, keep="first", records=4), 1)
    for wave in range(2):
        base = wave * 96
        header(words, base, 3, 0, 0, 1000, 2000 + 100 * wave)
        words.update({str(base + 32 + i): v for i, v in enumerate((100 + wave, 400 + wave, 0))})
    words = case(
        "phase_across_the_low_word_wrap", dict(names=["a"], wgsize=32, keep="last", records=32), 1
    )
    header(words, 0, 2, 0, 0, 1, 0)
    words.update({str(32 + i): v for i, v in enumerate((2**32 - 16, 16))})
    for c in cases:
        declaration = Laps(**c["kw"])
        words = np.zeros(
            c["workgroups"] * declaration.waves * declaration.rows * 32, dtype=np.uint32
        )
        for at, value in c["words"].items():
            words[int(at)] = value
        timeline = declaration.decode(words.view(np.uint8), c["workgroups"])
        c.update(
            rounds=timeline.rounds,
            clock_mhz=[
                [None if m is None else round(m, 6) for m in row] for row in timeline.clock_mhz
            ],
            laps=timeline.laps,
            summary=timeline.summary(),
        )
        if c["kw"]["keep"] != "sum":
            c["phases"] = timeline.phases()
    return {
        "notation": "Synthetic counter/header and lap samples decoded by thin.lib.Laps.",
        "cases": cases,
    }


def generate():
    """Evaluate host declarations and protocols without opening checked-in golden data."""
    return {
        path: encode(doc)
        for path, doc in zip(
            OUTPUTS,
            (declarations(), tags(), counts(), laps_decode(), snippets(), traces()),
            strict=True,
        )
    }


def snippets():
    """Build and render newly authored target-specific library programs."""
    from tools.golden.lib_authored import products

    return products()


def traces():
    """Record native events from newly authored library protocol programs."""
    from tools.golden.lib_authored import traces as authored_traces

    return authored_traces()
