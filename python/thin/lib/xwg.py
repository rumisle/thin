"""Workgroup hand-offs within one launch.

``Gens`` tracks generations, ``Slots`` carries tagged payloads between
workgroups, and ``Counts`` tracks arrivals. Each declaration contributes
its own library argument. Host helpers expose the same tag/count arithmetic
for unit tests and buffer initialization.
"""

from __future__ import annotations

from thin.dsl import (
    LGKM,
    VM,
    O,
    carried,
    first_lane,
    gens,
    kept,
    lib_arg,
    popcount,
    select,
)
from thin.lib import _hooks

__all__ = [
    "ABLATIONS",
    "GEN_START",
    "Arrival",
    "Counts",
    "CountsBody",
    "Generation",
    "Gens",
    "Slots",
    "SlotsBody",
    "arrival_fields",
    "bits_tag",
    "full_tag",
    "leave_mine",
    "reset_amount",
]

#: A generation word's first contents: three launches cross the wrap.
GEN_START = 2**32 - 3
_M32 = 0xFFFF_FFFF
_LINE = 128
_ARRIVAL = 1 << 16
_RESET_BASE = 0xFFFF_0000
_LEAVES = 0xFFFF

#: The ablation classes of the protocol's mutants.
ABLATIONS = (
    "xwg_coherent",
    "xwg_tag",
    "xwg_need",
    "xwg_release",
    "xwg_reset",
    "xwg_takeback",
    "xwg_fresh",
)


def full_tag(g1: int, ahead: int = 0) -> int:
    """The full tag of generation g1: 2 (g1 + ahead) + 1 mod 2^32."""
    return ((g1 << 1) + 2 * ahead + 1) & _M32


def bits_tag(g1: int, k: int, at: int) -> int:
    """The k-bit tag of generation g1 at bit `at`."""
    return (((g1 % (1 << k)) ^ 1) << at) & _M32


def _wgsize(wgsize) -> None:
    g = _hooks.granule()
    if not (isinstance(wgsize, int) and wgsize % g == 0 and g <= wgsize <= 1024):
        raise ValueError(f"wgsize {wgsize}: whole waves, {g} .. 1024")


def _kernel(k, wgsize) -> None:
    if k.wgsize is not None and k.wgsize != wgsize:
        raise ValueError(f"xwg: declared for wgsize {wgsize}, the kernel's is {k.wgsize}")


def _pol(b, cache: dict) -> dict:
    """A slot access's cache fields: the coherent ones unless ablated (`xwg_coherent`)."""
    return {} if b.ablated(["xwg_coherent"]) else cache


# --------------------------------------------------------------- generations


class Gens:
    """Generation words: one a member, or one a party."""

    def __init__(self, name="xwg_gen", *, per, words, wgsize):
        """Declares `words` members' or parties' generation words."""
        if per not in ("member", "party"):
            raise ValueError(f"per {per!r}: one of ('member', 'party')")
        if words < 1:
            raise ValueError(f"words {words}: at least 1")
        _wgsize(wgsize)
        self.name, self.per, self.words, self.wgsize = name, per, words, wgsize
        self.waves = wgsize // _hooks.granule() if per == "member" else 1

    def args(self) -> dict:
        """The argument it adds: generation memory."""
        return {self.name: lib_arg(gens, "xwg")}

    def sizes(self) -> dict:
        """Its own memory's bytes at the bind."""
        return {self.name: 4 * self.words * self.waves}

    def begin(self, b, k, at) -> Generation:
        """Reads the member's or party's word."""
        return Generation(self, b, k, at)


class Generation:
    """A body's generation: g, g1 and the tags."""

    def __init__(self, decl: Gens, b, k, at):
        """Emits the word's load, g1 and (per member) its store-back."""
        _kernel(k, decl.wgsize)
        self.decl, self.b, self.k = decl, b, k
        self.S, self.V = b.s, b.v
        (self.base,) = k.load(decl.name)
        self.lane = k.lane
        if decl.waves > 1:
            at = at * decl.waves + k.wave
        self.off = self.S.u32(at * 4, name="xwg_off")
        self.g = self.S.u32(name="xwg_g")
        b.emit(O.s_load(self.g, self.base + self.off, comment="the generation word"))
        b.wait(self.g, comment="the generation")
        self.g1 = self.S.u32(self.g + 1, name="xwg_g1")
        self._tags: dict = {}
        if decl.per == "member":
            why = "per-member generations: every workgroup of every launch advances its own word"
            b.need("same_grid", why, arg=decl.name)
            self._store_back()

    def _store_back(self) -> None:
        b = self.b
        with b.lanes(self.lane == 0):
            ga = self.V.u32(self.off, name="xwg_ga")
            gv = self.V.u32(self.g1, name="xwg_g1v")
            b.emit(O.global_store(ga, gv, self.base, comment="g1 stored back: the next launch's g"))

    def advance(self) -> None:
        """A party's completer advances its word."""
        if self.decl.per != "party":
            raise ValueError('advance(): a per="member" generation is advanced where it is read')
        self._store_back()

    def _tag(self, key, make, name):
        """A tag made once where every later use can read it.

        A tag made inside an arm or a lane region is written only on the
        paths through it, so it is reused only inside the same ones.
        """
        arms, regions = self.b._n.scope()
        made = self._tags.get(key)
        if made is not None:
            value, (at_arms, at_regions) = made
            if arms[: len(at_arms)] == at_arms and regions[: len(at_regions)] == at_regions:
                return value
        value = self.S.u32(make(), name=name)
        self._tags[key] = (value, (arms, regions))
        return value

    def tag(self, ahead=0):
        """The full tag."""

        def make():
            return 1 if self.b.ablated(["xwg_tag"]) else (self.g1 << 1) + (2 * ahead + 1)

        return self._tag(("full", ahead), make, "xwg_tag")

    def tag_bits(self, k, at):
        """The k-bit tag at bit `at`."""

        def make():
            low = (1 << k) - 1
            return 1 << at if self.b.ablated(["xwg_tag"]) else ((self.g1 & low) ^ 1) << at

        return self._tag(("bits", k, at), make, "xwg_tagk")


# --------------------------------------------------------------- slots

_TAGS_TEXT = 'tag {t!r}: "dword", "flag", ("sign",), ("high", k) or ("low", k)'


def _tag_form(tag):
    """(kind, k, at) of a tag form, or the refusal."""
    if tag in ("dword", "flag"):
        return tag, None, None
    if isinstance(tag, tuple) and tag == ("sign",):
        return "bits", 1, 31
    if isinstance(tag, tuple) and len(tag) == 2 and tag[0] in ("high", "low"):
        k = tag[1]
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(_TAGS_TEXT.format(t=tag))
        if not 1 <= k <= 16:
            raise ValueError(f"tag {tag}: 1 .. 16 bits")
        return "bits", k, (32 - k if tag[0] == "high" else 0)
    raise ValueError(_TAGS_TEXT.format(t=tag))


def _round_up(x: int, to: int) -> int:
    return -(-x // to) * to


class Slots:
    """Tagged slots in carried memory."""

    def __init__(
        self, name="xwg_slots", *, slots, pieces, piece, tag, wgsize, stride=None, uncached=False
    ):
        """Declares `slots` slots of `pieces` pieces of `piece` bytes."""
        if piece not in (4, 8, 16, 64):
            raise ValueError(f"piece {piece}: one of (4, 8, 16, 64) bytes")
        self.kind, self.bits, self.at = _tag_form(tag)
        if piece == 4 and self.kind == "dword":
            raise ValueError(
                'piece 4 with tag "dword": nothing left for a payload - a bits tag, or piece 8'
            )
        if piece == 64 and self.kind == "flag":
            raise ValueError('tag "flag" with piece 64: a flag needs no quad; piece 4, 8 or 16')
        if slots < 1 or pieces < 1:
            raise ValueError(f"slots {slots}, pieces {pieces}: at least 1")
        _wgsize(wgsize)
        need = pieces * piece
        if stride is None:
            stride = _round_up(need, _LINE)
        elif stride < need or stride % piece:
            raise ValueError(
                f"stride {stride}: at least the slot's {need} bytes, "
                f"a multiple of the piece's {piece}"
            )
        self.name, self.slots, self.pieces, self.piece = name, slots, pieces, piece
        self.tag, self.wgsize, self.stride, self.uncached = tag, wgsize, stride, uncached
        self.lane_dwords = 4 if piece == 64 else piece // 4
        self.flags_at = _round_up(slots * stride, _LINE)
        if self.kind == "bits":
            self.mask = ((1 << self.bits) - 1) << self.at
            self.keep = ~self.mask & _M32

    def bytes(self) -> int:
        """The slots' memory: the data, then a line per slot's flag (`"flag"`)."""
        return self.flags_at + (_LINE * self.slots if self.kind == "flag" else 0)

    def args(self) -> dict:
        """The argument it adds: carried memory (uncached with `uncached`)."""
        kind = carried.uncached if self.uncached else carried
        return {self.name: lib_arg(kind, "xwg")}

    def sizes(self) -> dict:
        """Its own memory's bytes at the bind."""
        return {self.name: self.bytes()}

    def begin(self, b, k, g: Generation) -> SlotsBody:
        """The body's slots, with the tag of this launch."""
        return SlotsBody(self, b, k, g)


class _Read:
    """One read of a poll, peek or load: its slot, piece and lanes."""

    def __init__(self, xs, slot, i, lanes):
        self.slot, self.i = slot, i
        self.cond = xs._lanes(lanes)


class SlotsBody:
    """A body's slots: piece, put, publish, poll, peek, fresh, load."""

    def __init__(self, decl: Slots, b, k, g: Generation):
        """Emits the slots' load and their tag."""
        _kernel(k, decl.wgsize)
        if decl.kind == "bits" and g.decl.per == "party":
            raise ValueError(
                f'{decl.name}: a bits tag with per="party": a party\'s member may skip a launch, '
                "and a k-bit tag tells only the last launch from this one"
            )
        self.decl, self.b, self.k, self.g = decl, b, k, g
        self.S, self.V = b.s, b.v
        (self.base,) = k.load(decl.name)
        self.lane = k.lane
        self.t = g.tag_bits(decl.bits, decl.at) if decl.kind == "bits" else g.tag()

    # --- lanes and offsets

    def _lanes(self, lanes):
        """A call's lanes as a region condition, or None."""
        d = self.decl
        if d.piece == 64:
            if lanes is None:
                return None
            if not callable(lanes):
                raise ValueError(
                    f"{d.name}: a 64-byte piece's lanes as a function of the quad index "
                    "(lane >> 2), so that the four lanes of a quad agree"
                )
            return lanes(self.lane >> 2)
        if callable(lanes):
            raise ValueError(f"{d.name}: lanes a mask or a condition (piece {d.piece})")
        return lanes

    def _region(self, cond, *more):
        conds = [c for c in (cond, *more) if c is not None]
        return self.b.lanes(*conds)

    def _offset(self, slot, i):
        d = self.decl
        off = slot * d.stride + i * d.piece
        if d.piece == 64:
            off = off + (self.lane & 3) * 16
        return self.V.u32(off, name="xwg_off")

    # --- writing

    def piece(self):
        """A piece's registers; a "dword" tag's last dword is the tag now."""
        d = self.decl
        w = self.V.u32(n=d.lane_dwords, name="xwg_w")
        if d.kind == "dword" and d.piece != 64:
            w[d.lane_dwords - 1] = self.t
        return w

    def put(self, slot, i, w, lanes=None) -> None:
        """Stores a piece, its tag merged, with the coherent policy."""
        d, b = self.decl, self.b
        cond = self._lanes(lanes)
        with self._region(cond):
            last = d.lane_dwords - 1
            if d.kind == "bits":
                w[last] = (w[last] & d.keep) | self.t
            elif d.kind == "dword" and d.piece == 64:
                w[3] = select((self.lane & 3) == 3, self.t, w[3])
            off = self._offset(slot, i)
            store = O.global_store(
                off, w, self.base, **_pol(b, _hooks.coherent()), comment="xwg: a piece"
            )
            b.emit(store)

    def publish(self, slot, lanes=None) -> None:
        """Releases the slot's data, then stores its flag."""
        d, b = self.decl, self.b
        if d.kind != "flag":
            raise ValueError(
                f'{d.name}: publish() is the "flag" tag\'s (its tag rides in the piece)'
            )
        cond = self._lanes(lanes)
        b.wait(
            VM,
            comment="xwg: the slot's data acked before its flag (the release)",
            abl=["xwg_release"],
        )
        with self._region(cond, self.lane == 0):
            fo = self.V.u32(d.flags_at + slot * _LINE, name="xwg_fo")
            ft = self.V.u32(self.t, name="xwg_ft")
            flag = O.global_store(
                fo, ft, self.base, **_pol(b, _hooks.coherent()), comment="xwg: the slot's flag"
            )
            b.emit(flag)

    # --- reading

    def _stale(self, v, flag=None):
        """A read's stale mask, under the current lanes."""
        d, t = self.decl, self.t
        if d.kind == "flag":
            return self.S.mask(flag != t, name="xwg_stale")
        last = d.lane_dwords - 1
        if d.kind == "bits":
            v[last] = v[last] ^ t
            return self.S.mask((v[last] & d.mask) != 0, name="xwg_stale")
        if d.piece == 64:
            return self.S.mask(((self.lane & 3) == 3) & (v[3] != t), name="xwg_stale")
        return self.S.mask(v[last] != t, name="xwg_stale")

    def _missing(self, masks):
        if len(masks) == 1:
            return masks[0]
        joined = masks[0]
        for m in masks[1:]:
            joined = joined | m
        return self.S.mask(joined, name="xwg_missing")

    def _load_pieces(self, reads, offs, comment):
        b, d = self.b, self.decl
        vals = [self.V.u32(n=d.lane_dwords, name="xwg_v") for _ in reads]
        for r, off, v in zip(reads, offs, vals, strict=True):
            with self._region(r.cond):
                load = O.global_load(
                    v, off, self.base, **_pol(b, _hooks.coherent()), comment=comment
                )
                b.emit(load)
        b.wait(*vals, comment="xwg: the flagged pieces")
        return vals

    def poll(self, reads, spin, on_timeout=None, sleep=1, name="xwg_poll", all_arrived=False):
        """Waits until every read's piece is this launch's, with a bound and a fallback."""
        d, b = self.decl, self.b
        steal = callable(on_timeout)
        if all_arrived and spin is not None:
            raise ValueError(f"{name}: all_arrived is an unbounded poll's (spin None)")
        if spin is None:
            if on_timeout is not None:
                raise ValueError(f"{name}: on_timeout with spin None (no bound to pass)")
            if not all_arrived and not b.ablated(["xwg_need"]):
                b.need("co_resident", f"{name}: an unbounded poll (xwg, spin None)")
        else:
            if not isinstance(spin, int) or isinstance(spin, bool) or not 2 <= spin < 1 << 30:
                raise ValueError(f"{name}: spin {spin!r}: 2 .. 2^30 - 1 rounds, or None")
            if on_timeout is None:
                raise ValueError(
                    f'{name}: a bound without on_timeout - a steal (a function) or "leave"'
                )
            if on_timeout != "leave" and not steal:
                raise ValueError(f'{name}: on_timeout {on_timeout!r}: a function or "leave"')
            if steal and self.g.decl.per == "party":
                raise ValueError(
                    f'{name}: a steal with per="party": the completer would advance the '
                    "party's word before the stolen member reads it"
                )
        flag = d.kind == "flag"
        rs = [_Read(self, *r) for r in reads]
        offs = [self._offset(r.slot, r.i) for r in rs]
        fos = [self.V.u32(d.flags_at + r.slot * _LINE, name="xwg_fo") for r in rs] if flag else []
        fs = [self.V.u32(name="xwg_f") for _ in rs] if flag else []
        vals = [] if flag else [self.V.u32(n=d.lane_dwords, name="xwg_v") for _ in rs]
        polls = self.S.u32(0, name="xwg_polls")
        if steal:
            b.wait(
                LGKM, comment="xwg: the scalar loads in, before a loop whose steal may read them"
            )
        polled = fs if flag else vals
        for lp in b.loop(name=name):
            for r, reg, at in zip(rs, polled, fos if flag else offs, strict=True):
                with self._region(r.cond):
                    b.emit(O.global_load(reg, at, self.base, **_pol(b, _hooks.coherent())))
            b.wait(*polled, comment="xwg: the round")
            masks = []
            for r, reg in zip(rs, polled, strict=True):
                with self._region(r.cond):
                    masks.append(self._stale(None, reg) if flag else self._stale(reg))
            missing = self._missing(masks)
            lp.break_when(popcount(missing) == 0, comment="xwg: every piece this launch's")
            if spin is not None:
                b.emit(
                    O.op(
                        _hooks.hook("scalar_add"),
                        polls,
                        polls,
                        1,
                        comment=f"[spin] {name}: the rounds",
                    )
                )
                if steal:
                    with b.when(polls >= spin, name=f"{name}_steal"):
                        on_timeout(self._spread(masks) if d.piece == 64 else masks)
                        polls[...] = spin - 2
                else:
                    lp.break_when(polls >= spin, comment="xwg: timed out: leave")
            if sleep:
                b.emit(O.op(_hooks.hook("sleep"), sleep))
        if flag:
            vals = self._load_pieces(rs, offs, "xwg: a flagged piece")
        if on_timeout == "leave":
            return vals, self.S.u32(select(polls < spin, 1, 0), name="xwg_in")
        return vals

    def _spread(self, masks):
        """Each stale mask spread over its quads, from lane 4q + 3."""
        b = self.b
        shifted = self.S.mask(name="xwg_shifted")
        out = []
        for m in masks:
            quads = self.S.mask(m, name="xwg_quads")
            for s in (1, 2, 3):
                comment = f"xwg: the stale quads, lane - {s}"
                b.emit(O.op(_hooks.hook("mask_shr"), shifted, m, s, comment=comment))
                quads[...] = quads | shifted
            out.append(quads)
        return out

    def peek(self, reads, name="xwg_peek", into=None):
        """Issues the flags' loads, not waited for."""
        d, b = self.decl, self.b
        if d.kind != "flag":
            raise ValueError(
                f'{d.name}: peek() is the "flag" tag\'s (a readiness word apart from the data)'
            )
        out = []
        for j, (slot, i, lanes) in enumerate(reads):
            r = _Read(self, slot, i, lanes)
            fo = self.V.u32(d.flags_at + slot * _LINE, name="xwg_fo")
            f = into[j] if into is not None else self.V.u32(name=f"{name}_f")
            with self._region(r.cond):
                comment = "xwg: a flag, peeked"
                b.emit(
                    O.global_load(f, fo, self.base, **_pol(b, _hooks.coherent()), comment=comment)
                )
            out.append((f, r.cond))
        return out

    def fresh(self, peeked):
        """1 when every peeked flag is this launch's, else 0."""
        b = self.b
        b.wait(*[f for f, _ in peeked], comment="xwg: the peeked flags")
        masks = []
        for f, cond in peeked:
            with self._region(cond):
                masks.append(self.S.mask(f != self.t, name="xwg_stale"))
        missing = self._missing(masks)
        if b.ablated(["xwg_fresh"]):
            return self.S.u32(1, name="xwg_fresh")
        return self.S.u32(select(popcount(missing) == 0, 1, 0), name="xwg_fresh")

    def load(self, reads):
        """Loads flagged pieces whose flags were fresh."""
        d, b = self.decl, self.b
        if d.kind != "flag":
            raise ValueError(f'{d.name}: load() is the "flag" tag\'s (other tags come with poll)')
        vals = []
        for slot, i, lanes in reads:
            r = _Read(self, slot, i, lanes)
            v = self.V.u32(n=d.lane_dwords, name="xwg_v")
            with self._region(r.cond):
                off = self._offset(slot, i)
                load = O.global_load(
                    v, off, self.base, **_pol(b, _hooks.coherent()), comment="xwg: a flagged piece"
                )
                b.emit(load)
            vals.append(v)
        b.wait(*vals, comment="xwg: the flagged pieces")
        return vals


# --------------------------------------------------------------- counts


def arrival_fields(was: int, n: int) -> tuple[int, int]:
    """(last, left) of an arrival that saw the word `was`."""
    return int((was >> 16) == n - 1), was & _LEAVES


def leave_mine(was: int, n: int) -> int:
    """Whether a leave that saw `was` takes the job."""
    a = was >> 16
    return int(a == n or a == 0)


def reset_amount(old: int) -> int:
    """The reset's add, from what the last arrival saw."""
    return (_RESET_BASE - old) & _M32


def _count_n(n, what: str) -> None:
    """Refuses a known count past a count word's 16-bit fields.

    A count in a register is the caller's to keep in 1.. 65535.
    """
    if isinstance(n, int) and not 1 <= n <= _LEAVES:
        raise ValueError(f"{what}: n {n}: 1 .. {_LEAVES} (a count word's 16-bit field)")


class Arrival:
    """An arrival's scalars: last, left and word."""

    def __init__(self, last, left, old):
        """Holds the three scalars."""
        self.last, self.left, self.old = last, left, old


class Counts:
    """Count words in kept memory."""

    def __init__(self, name="xwg_counts", *, wgsize):
        """Declares the counts."""
        _wgsize(wgsize)
        self.name, self.wgsize = name, wgsize

    def args(self) -> dict:
        """The argument it adds: kept memory."""
        return {self.name: lib_arg(kept, "xwg")}

    def sizes(self, counts) -> dict:
        """Its own memory's bytes for `counts` counts, 256 bytes apart."""
        if counts < 1:
            raise ValueError(f"counts {counts}: at least 1")
        return {self.name: 256 * counts}

    def begin(self, b, k) -> CountsBody:
        """The body's counts."""
        return CountsBody(self, b, k)


class CountsBody:
    """A body's counts: arrive, leave, reset, by lane 0's atomics."""

    def __init__(self, decl: Counts, b, k):
        """Loads the counts' base."""
        _kernel(k, decl.wgsize)
        self.decl, self.b, self.k = decl, b, k
        self.S, self.V = b.s, b.v
        (self.base,) = k.load(decl.name)
        self.lane = k.lane
        self.V.lanes_covered("xwg_old")

    def _release(self, what: str) -> None:
        self.b.wait(
            VM,
            comment=f"xwg: the wave's stores acked before its {what} (the release)",
            abl=["xwg_release"],
        )

    def _atomic(self, at, add, comment, old=None, abl=None) -> None:
        # Every workgroup's atomics on a count meet at the device's
        # coherence point: the coherent scope.
        b = self.b
        op, scope = _hooks.hook("returning_atomic_add"), _hooks.coherent()
        with b.lanes(self.lane == 0):
            ca = self.V.u32(256 * at, name="xwg_ca")
            cd = self.V.u32(add, name="xwg_cd")
            if old is None:
                ins = O.op(op, ca, cd, self.base, **scope, comment=comment)
            else:
                fields = {**_hooks.returning(), **scope}
                ins = O.op(op, old, ca, cd, self.base, **fields, comment=comment)
            b.emit(ins, abl=abl)

    def arrive_issue(self, at, release=True):
        """Issues an arrival, returning word's VGPR not waited for."""
        if release:
            self._release("arrival")
        old = self.V.u32(name="xwg_old")
        self._atomic(at, _ARRIVAL, "xwg: an arrival", old=old)
        return old

    def arrived(self, old, n) -> Arrival:
        """The arrival's scalars once its old word is in."""
        _count_n(n, "arrive")
        self.b.wait(old, comment="xwg: the count's return")
        was = self.S.u32(first_lane(old), name="xwg_was")
        last = self.S.u32(select((was >> 16) == n - 1, 1, 0), name="xwg_last")
        left = self.S.u32(was & _LEAVES, name="xwg_left")
        return Arrival(last, left, was)

    def arrive(self, at, n) -> Arrival:
        """An arrival on count `at` of a launch with `n` arrivals."""
        return self.arrived(self.arrive_issue(at), n)

    def reset(self, at, r: Arrival) -> None:
        """The last arrival takes back what it saw and itself."""
        self._atomic(at, _RESET_BASE - r.old, "xwg: the count reset", abl=["xwg_reset"])

    def leave(self, at, n):
        """A waiter's leave once its bound passed; 1 when the job is its own."""
        _count_n(n, "leave")
        b = self.b
        self._release("leave")
        old = self.V.u32(name="xwg_old")
        self._atomic(at, 1, "xwg: a leave", old=old)
        b.wait(old, comment="xwg: the count's return")
        was = self.S.u32(first_lane(old), name="xwg_was")
        a = self.S.u32(was >> 16, name="xwg_arrived")
        mine = self.S.u32(select(a == n, 1, select(a == 0, 1, 0)), name="xwg_mine")
        with b.when(mine == 1, name="xwg_back"):
            self._atomic(at, _M32, "xwg: the leave taken back", abl=["xwg_takeback"])
        return mine
