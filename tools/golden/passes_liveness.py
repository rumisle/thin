"""Fresh LDS layouts and assembly listings for native liveness/race evaluation.

The listing language is the input contract of ``Listing``.  Every address,
branch, barrier, fence and byte map below is authored explicitly; native
``lds::prove`` and ``lds::races`` produce all assertions used by the generator.
"""

from __future__ import annotations


def _layout(extra=None):
    lines = [
        "; lds: 64 B of a 4096 B budget",
        ";   offset     end   size  region           shape                strides",
        ";        0      64     64  union",
        ";        0      64     64    + P            [16] u32             4",
        ";        0      64     64    | Q            [16] u32             4",
    ]
    if extra:
        lines[0] = "; lds: 128 B of a 4096 B budget"
        lines.append(f";       64     128     64  {extra:<16} [16] u32             4")
    return lines


def _line(text, comment=None):
    return "\t" + text + (" " * max(2, 64 - len(text))) + ("// " + comment if comment else "")


def _lane_addr():
    return [
        _line("v_and_b32 v6, 15, v0", "slot = lane & 15"),
        _line("s_delay_alu 1", "hazard: instid0(VALU_DEP_1)"),
        _line("v_lshlrev_b32 v7, 2, v6", "adr = slot * 4"),
    ]


def _reverse_addr():
    return [
        _line("v_xor_b32 v8, 15, v0", "reverse = 15 xor lane"),
        _line("s_delay_alu 1", "hazard: instid0(VALU_DEP_1)"),
        _line("v_lshlrev_b32 v9, 2, v8", "adr = reverse * 4"),
    ]


def _store(region, data="v4", offset=None):
    suffix = f" offset:{offset}" if offset else ""
    return _line(f"ds_store_b32 v7, {data}{suffix}", f"{region} + slot * 4")


def _load(region, dest="v5", offset=None):
    suffix = f" offset:{offset}" if offset else ""
    return _line(f"ds_load_b32 {dest}, v9{suffix}", f"{region} + (15 xor lane) * 4")


def _wait(comment=None):
    return _line("s_wait_dscnt 0", comment)


def _bar(comment, fence=False):
    prefix = "[lds fence] " if fence else ""
    return [_line("s_barrier_signal -3"), _line("s_barrier_wait 65535", prefix + comment)]


def _region_sequence(region, barrier_label=None):
    lines = [*_lane_addr(), _store(region)]
    if barrier_label:
        lines += [_wait(barrier_label), *_bar(barrier_label)]
    lines += [*_reverse_addr(), _load(region)]
    return lines


def _loop_prefix(trips=5):
    return [
        _line(f"s_mov_b32 s0, {trips}", f"rounds = {trips}"),
        ".Lrounds_4:",
        _line("s_cmp_eq_u32 s0, 0", "rounds == 0"),
        "s_cbranch_scc1 .Lrounds_end_7",
    ]


def _loop_suffix():
    return [
        _line("s_sub_co_u32 s0, s0, 1", "rounds = rounds - 1"),
        "s_branch .Lrounds_4",
        ".Lrounds_end_7:",
    ]


def lives_plans():
    """Author LDS epoch-liveness programs over named regions."""
    plans = []

    def add(name, source, body, extra=None, refuses=False):
        """Construct add inputs."""
        listing = [*_layout(extra), *body]
        plans.append(
            {
                "name": name,
                "source": source,
                "listing": listing,
                "table_lines": 6 if extra else 5,
                "refuses": refuses,
            }
        )

    add(
        "two union members occupy separate barrier epochs",
        [
            "store P[slot]",
            "publish P and synchronize",
            "read P[reverse]",
            "synchronize P readers",
            "store Q[slot]",
            "publish Q",
        ],
        [
            *_region_sequence("P", "publish P"),
            _wait("readers complete"),
            *_bar("P readers"),
            *_lane_addr(),
            _store("Q", "v5"),
            _wait("publish Q"),
            *_bar("publish Q"),
        ],
    )
    add(
        "union members overlap without a barrier after the read",
        ["store P[slot]", "publish P", "read P[reverse]", "store Q[slot] before a new epoch"],
        [
            *_region_sequence("P", "publish P"),
            _wait(),
            _lane_addr(),
            _store("Q", "v5"),
            _wait("publish Q"),
            *_bar("publish Q"),
        ],
        refuses=True,
    )
    add(
        "distant accesses still share the entry epoch",
        ["store Q then read Q", "store P then read P", "no epoch boundary"],
        [
            *_region_sequence("Q"),
            _wait(),
            *_lane_addr(),
            _store("P", "v5"),
            *_reverse_addr(),
            _load("P", "v10"),
            _wait(),
        ],
        refuses=True,
    )
    add(
        "unretired read extends its region through the next barrier",
        ["publish P writes", "read P[reverse]", "barrier before read completes", "store Q"],
        [
            *_region_sequence("P", "P stores visible"),
            *_bar("P read outstanding"),
            *_lane_addr(),
            _store("Q", "v11"),
            _wait("Q complete"),
            *_bar("Q visible"),
        ],
        refuses=True,
    )

    loop_common = [
        *_region_sequence("P", "P stores"),
        _wait("P readers"),
        *_bar("P readers"),
        *_lane_addr(),
        _store("Q", "v5"),
        _wait("Q stores"),
        *_bar("Q stores"),
        *_reverse_addr(),
        _load("Q", "v10"),
    ]
    add(
        "loop back edge joins union lifetimes without an LDS fence",
        ["repeat: publish/read P; publish/read Q; ordinary round barrier"],
        [*_loop_prefix(5), *loop_common, _wait("Q read"), *_bar("round complete"), *_loop_suffix()],
        refuses=True,
    )
    add(
        "LDS fence at loop boundary ends both current member lifetimes",
        ["repeat: publish/read P; publish/read Q; waited LDS fence"],
        [
            *_loop_prefix(5),
            *loop_common,
            _wait("Q read"),
            *_bar("round fence", fence=True),
            *_loop_suffix(),
        ],
    )
    add(
        "LDS fence rejects an outstanding DS read",
        ["repeat: publish/read P; publish/read Q; fence without final wait"],
        [*_loop_prefix(5), *loop_common, *_bar("round fence", fence=True), *_loop_suffix()],
        refuses=True,
    )

    r_body = [
        *_lane_addr(),
        _store("R", offset=64),
        *_bar("R store outstanding"),
        *_reverse_addr(),
        _load("R", offset=64),
        _wait(),
    ]
    add(
        "store outstanding at barrier followed by a peer read is refused",
        ["store R[slot]", "barrier with no wait", "read R[reverse]"],
        r_body,
        extra="R",
        refuses=True,
    )
    add(
        "producer ordered annotation explains the same barrier protocol",
        ["ds_write R[slot] (ordered: 'consumer sequence established')", "barrier then read R"],
        r_body,
        extra="R",
    )

    atomic_prefix = [_line("s_mov_b32 s0, 0"), ".Latomic_4:", *_bar("atomic slot"), *_lane_addr()]
    atomic_suffix = [
        _line("s_add_co_u32 s0, s0, 2"),
        _line("s_cmp_lt_u32 s0, 6"),
        "s_cbranch_scc1 .Latomic_4:",
        _wait("atomics retire"),
        *_bar("atomics visible"),
    ]
    # Correct the branch spelling to the matching fresh label.
    atomic_suffix[2] = "s_cbranch_scc1 .Latomic_4"
    add(
        "commuting returnless LDS atomics may remain in flight across barriers",
        ["repeat three times: barrier; commuting atomic R[slot]", "wait and publish final R"],
        [*atomic_prefix, _line("ds_add_u32 v7, v4 offset:64", "R + slot * 4"), *atomic_suffix],
        extra="R",
    )
    add(
        "a read of commuting atomic bytes makes the loop barrier unsafe",
        [
            "repeat three times: barrier; read R; then commuting atomic R",
            "wait and publish final R",
        ],
        [
            *atomic_prefix,
            _line("ds_load_b32 v5, v7 offset:64", "R + slot * 4"),
            _wait(),
            *_lane_addr(),
            _line("ds_add_u32 v7, v4 offset:64", "R + slot * 4"),
            *atomic_suffix,
        ],
        extra="R",
        refuses=True,
    )

    add(
        "wave exclusive branch creates unmatched barrier counts",
        ["only wave zero enters", "barrier inside that arm", "barrier after join"],
        [
            _line("s_cmp_lg_u32 s0, 0"),
            "s_cbranch_scc1 .Lwaves_11",
            *_bar("wave zero private"),
            ".Lwaves_11:",
            *_bar("all waves"),
        ],
        refuses=True,
    )
    add(
        "uniform scalar predicate may skip a common barrier",
        ["when p is zero, skip the common barrier", "store D afterwards"],
        [
            _line("s_cmp_eq_u32 s1, 0", "p == 0"),
            "s_cbranch_scc0 .Luniform_12",
            *_bar("zero p only"),
            ".Luniform_12:",
            *_lane_addr(),
            _store("D", offset=64),
            _wait(),
        ],
        extra="D",
    )
    add(
        "wave derived scalar predicates remain wave dependent",
        ["compute alternating wave group", "conditionally barrier", "common barrier after join"],
        [
            _line("s_xor_b32 s1, s0, 2", "wave group xor 2"),
            _line("s_cmp_eq_u32 s1, 2"),
            "s_cbranch_scc0 .Lderived_13",
            *_bar("selected waves"),
            ".Lderived_13:",
            *_bar("all waves"),
        ],
        refuses=True,
    )
    add(
        "scalar assigned under wave branch taints a later branch",
        ["flag=0", "wave zero sets flag", "flag controls a barrier", "common barrier"],
        [
            _line("s_mov_b32 s1, 0", "flag = 0"),
            _line("s_cmp_eq_u32 s0, 0", "wave == 0"),
            "s_cbranch_scc0 .Lflag_skip_14",
            _line("s_mov_b32 s1, 1", "flag = 1"),
            ".Lflag_skip_14:",
            _line("s_cmp_eq_u32 s1, 1"),
            "s_cbranch_scc0 .Lflag_join_14",
            *_bar("flagged wave"),
            ".Lflag_join_14:",
            *_bar("all waves"),
        ],
        refuses=True,
    )
    add(
        "loop nested in a wave arm conflicts with the peer barrier count",
        ["selected waves loop twice with barriers", "other waves execute one barrier"],
        [
            _line("s_cmp_eq_u32 s0, 2"),
            "s_cbranch_scc0 .Lwave_else_15",
            _line("s_mov_b32 s0, 0"),
            ".Linner_15:",
            *_bar("selected repeated"),
            _line("s_add_co_u32 s0, s0, 2"),
            _line("s_cmp_lt_u32 s0, 4"),
            "s_cbranch_scc1 .Linner_15",
            "s_branch .Lwave_join_15",
            ".Lwave_else_15:",
            *_bar("other waves once"),
            ".Lwave_join_15:",
        ],
        refuses=True,
    )
    add(
        "paired wave arms form one epoch and expose overlapping union members",
        ["wave zero: barrier A then P", "other waves: paired barrier B then Q"],
        [
            _line("s_cmp_eq_u32 s0, 0"),
            "s_cbranch_scc0 .Lpaired_else_16",
            *_bar("barrier alpha"),
            *_lane_addr(),
            _store("P"),
            _wait(),
            "s_branch .Lpaired_join_16",
            ".Lpaired_else_16:",
            *_bar("barrier beta"),
            *_lane_addr(),
            _store("Q"),
            _wait(),
            ".Lpaired_join_16:",
        ],
        refuses=True,
    )

    def wave_block(number, region, last_wait):
        """Construct wave block inputs."""
        return [
            _line(f"s_cmp_lg_u32 s0, {number}"),
            f"s_cbranch_scc1 .Lwaves_{number}{region}",
            *_region_sequence(region),
            *([_wait()] if last_wait else []),
            f".Lwaves_{number}{region}:",
        ]

    add(
        "wave order mode allows separated same-wave regions to share storage",
        ["wave 2 only: complete P accesses", "wave 2 only: complete Q accesses"],
        [*wave_block(2, "P", True), *wave_block(2, "Q", True)],
    )
    add(
        "wave order mode retains an outstanding first member at the peer accesses",
        ["wave 3 only: P read outstanding", "wave 3 only: Q accesses", "final wait"],
        [*wave_block(3, "P", False), *wave_block(3, "Q", True), _wait()],
        refuses=True,
    )
    return plans


def _race_layout():
    return [
        "; lds: 256 B of a 4096 B budget",
        ";   offset     end   size  region           shape                strides",
        ";        0     256    256  Z                [64] u32             4",
    ]


def races_plans():
    """Author cross-wave race and ordering programs."""
    plans = []

    def add(name, source, body, byte_maps, refuses=False):
        """Construct add inputs."""
        plans.append(
            {
                "name": name,
                "source": source,
                "listing": [*_race_layout(), *body],
                "wgsize": 96,
                "wave_lanes": 32,
                "per_wave_bytes": byte_maps,
                "refuses": refuses,
            }
        )

    same = [
        _line("v_and_b32 v10, 15, v0", "slot = lane & 15"),
        _line("s_delay_alu 1"),
        _line("v_lshlrev_b32 v11, 2, v10", "adr = slot * 4"),
        _line("ds_store_b32 v11, v4", "Z + slot * 4"),
        _wait(),
    ]
    add(
        "same byte window written independently by all three waves",
        ["slot = lane modulo 16", "store Z[slot]", "wait"],
        same,
        [{"3": [0, 64]} for _ in range(3)],
        refuses=True,
    )
    private = [*_lane_addr(), _store("Z"), _wait(), *_reverse_addr(), _load("Z"), _wait()]
    add(
        "disjoint per-wave store and load windows are race free",
        ["store private Z lane window", "read same private window"],
        private,
        [{str(3): [w * 64, (w + 1) * 64], str(8): [w * 64, (w + 1) * 64]} for w in range(3)],
    )
    reverse = [
        *_lane_addr(),
        _store("Z"),
        _wait(),
        _line("v_xor_b32 v12, 31, v0", "peer = 31 xor lane"),
        _line("s_delay_alu 1"),
        _line("v_lshlrev_b32 v13, 2, v12"),
        _line("ds_load_b32 v14, v13", "Z + peer * 4"),
        _wait(),
    ]
    cross = [{"3": [w * 64, (w + 1) * 64], "8": [(2 - w) * 64, (3 - w) * 64]} for w in range(3)]
    add(
        "reversed peer reads race with another wave write before any barrier",
        ["write own Z window", "read reversed Z window without barrier"],
        reverse,
        cross,
        refuses=True,
    )
    # The inserted pair shifts the load from index 8 to index 10.
    ordered_cross = [{("10" if k == "8" else k): v for k, v in c.items()} for c in cross]
    ordered = [*reverse[:5], *_bar("all writes visible"), *reverse[5:]]
    add(
        "common barrier orders the reversed peer reads after all writes",
        ["write own Z window", "publish all writes", "read reversed Z window"],
        ordered,
        ordered_cross,
    )
    return plans
