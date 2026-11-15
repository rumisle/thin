"""The race: hygiene, laying out, judging, timed rounds, the result."""

from __future__ import annotations

import concurrent.futures
import hashlib
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from metron import _device, _memory
from metron._memory import GUARD, MIN_ROUNDS, PROBE, Copies, Pools
from metron._nouns import Call, State
from metron.stats import median_ci, ratio_verdict, settled

__all__ = ["ArmResult", "Race", "Verdict", "race"]


class Verdict(NamedTuple):
    """An arm against another."""

    ratio: float
    """The median of the per-round ratios."""
    ci: tuple
    """(low, high) of that median."""
    verdict: str
    """tie, faster, slower or undecided."""


@dataclass
class ArmResult:
    """One arm's judgement and times."""

    us: float | None = None
    """The median of its adjusted times, None when not timed."""
    ci: tuple | None = None
    """(low, high) of that median, or None."""
    times: list = field(default_factory=list)
    """Its adjusted time in each round, aligned across arms."""
    wrong: str | None = None
    """Why the judge refused it, or None."""
    checked: bool = True
    """Whether the check judged it."""
    fingerprint: str = ""
    """What identifies its code."""
    read: dict = field(default_factory=dict)
    """name -> bytes of its owned buffers read back."""
    ranks_us: list | None = None
    """Each rank's median raw time minus the prelude, or None."""


@dataclass
class Race:
    """A race's result."""

    point: str
    state: dict
    arms: dict
    status: str
    rounds: int
    conditions: dict
    flags: list
    device: dict
    validators: dict
    phases_s: dict
    wall_s: float
    reference: str = ""

    def __post_init__(self):
        if not self.reference and self.arms:
            self.reference = next(iter(self.arms))

    def _within(self) -> float:
        return self.conditions.get("within", 0.02)

    def versus(self, a: str, b: str) -> Verdict:
        """A against b from their per-round ratios; both must be timed."""
        ra, rb = self.arms[a], self.arms[b]
        if ra.us is None or rb.us is None:
            why = ra.wrong if ra.us is None else rb.wrong
            raise ValueError(f"{a} against {b}: both timed ({why})")
        return Verdict(*ratio_verdict(ra.times, rb.times, self._within()))

    @property
    def verdicts(self) -> dict:
        """Name -> versus(name, reference) for every timed arm but the baseline."""
        ref = self.reference
        if self.arms[ref].us is None:
            return {}
        return {
            n: self.versus(n, ref) for n, a in self.arms.items() if n != ref and a.us is not None
        }

    def ranked(self) -> list:
        """The timed arms by `us`, fastest first (ties keep arm order)."""
        timed = [(a.us, i, n) for i, (n, a) in enumerate(self.arms.items()) if a.us is not None]
        return [n for _, _, n in sorted(timed)]

    def save(self, path) -> None:
        """Appends the record's lines."""
        from metron import record

        record.save(self, path)


def _refuse(arms, point, state) -> None:
    names = [a.name for a in arms]
    twice = sorted({n for n in names if names.count(n) > 1})
    # A warm name is an input of every rank.
    unknown = sorted({n for ins in point.ranks()[0] for n in set(state.warm) - set(ins)})
    if twice or not arms or unknown:
        say = twice if twice else "none"
        raise ValueError(
            f"race at {point.key}: arms named once, one at least ({say}); "
            f"warm {unknown}: of the inputs"
        )


class Rank:
    """One rank's memory, stream and graphs."""

    def __init__(self, be, r, inputs, outputs, warm, n, arms, most):
        """Lays out the rank's copies, pools and flags (4 * most bytes)."""
        self.be, self.r = be, r
        self.inputs, self.outputs, self.warm = inputs, outputs, warm
        self.stream = be.stream(r)
        with be.on(r):
            self.pools = Pools(be, r, arms)
            self.flags = be.alloc(r, 4 * most)
        self.lay_out(n)

    def lay_out(self, n) -> None:
        """The copies at n (a warm input has one), made on the rank's stream."""
        self.n = n
        self.copies = {}
        with self.be.on(self.r):
            for name, t in self.inputs.items():
                self.copies[name] = Copies(
                    self.be, self.r, name, t, "in", 1 if name in self.warm else n
                )
            for name, t in self.outputs.items():
                self.copies[name] = Copies(self.be, self.r, name, t, "out", n)

    def restore(self) -> None:
        """Every copy from its original; the pools fresh with kept.

        On the rank's stream, so they are ordered before its next launches.
        """
        with self.be.on(self.r):
            for c in self.copies.values():
                c.restore()
            self.pools.fresh(kept=True)

    def differing(self, c: Copies, ref) -> list[int]:
        """The copies of c whose stride bytes differ from ref's (the copy compare)."""
        import torch

        with self.be.on(self.r):  # the flags zeroed before the compare, not after
            self.be.differ(self.r, c.buf, ref, c.stride, c.n, self.flags)
        self.be.sync(self.r)
        flags = self.flags[: 4 * c.n].view(torch.int32).cpu()
        return [i for i in range(c.n) if int(flags[i])]


class Pass:
    """A plain pass: a rank's calls, launched one by one by its thread."""

    __slots__ = ("calls", "n")

    def __init__(self, calls):
        self.calls, self.n = list(calls), len(calls)


class _Layout:
    """Every rank of a race, its arms and its check memo.

    plain: the plain path, where every graph is a Pass.
    """

    def __init__(self, be, arms, point, state, n, most, plain=False):
        self.be, self.arms, self.point, self.state = be, arms, point, state
        self.plain = plain
        ins, outs = point.ranks()
        self.group = len(ins) > 1
        self.ranks = [
            Rank(be, r, ins[r], outs[r], state.warm, n, arms, most) for r in range(len(ins))
        ]
        self.memo: dict = {}
        self.check_calls = 0
        self.check_s = 0.0

    def sync(self) -> None:
        for rk in self.ranks:
            self.be.sync(rk.r)

    def release(self, graphs) -> None:
        """Destroys one graph a rank, each after its rank's stream."""
        for rk in self.ranks:
            if not isinstance(graphs[rk.r], Pass):
                self.be.release(rk.r, graphs[rk.r])

    def make(self, rk: Rank, thunks) -> object:
        """A rank's graph of the thunks, or its pass on the plain path."""
        return Pass(thunks) if self.plain else self.be.capture(rk.r, thunks)

    def replay(self, rk: Rank, g) -> None:
        """One replay of a rank's graph, or one pass."""
        if isinstance(g, Pass):
            self.be.run(rk.r, g.calls)
        else:
            self.be.replay(rk.r, g)

    def timed(self, rk: Rank, g) -> tuple:
        """(us per call, host us per call or None) of an untimed and a timed run."""
        if isinstance(g, Pass):
            return self.be.timed_pass(rk.r, g.calls)
        return self.be.timed(rk.r, g), None

    def restore(self) -> None:
        for rk in self.ranks:
            rk.restore()
        self.sync()

    def fresh(self) -> None:
        for rk in self.ranks:
            with self.be.on(rk.r):
                rk.pools.fresh()
        self.sync()

    def call(self, arm, rk: Rank, i: int) -> Call:
        """The Call of copy i on a rank."""
        views = {name: c.view(i) for name, c in rk.copies.items()}
        owned = {}
        for name, (_, kind) in arm.owned.items():
            if kind == "mapped":
                owned[name] = [q.pools.view(arm.name, name) for q in self.ranks]
            else:
                owned[name] = rk.pools.view(arm.name, name)

        def peers(name):
            return [q.copies[name].view(i) for q in self.ranks]

        return Call(views, owned, rk.stream, rk.r, peers)

    def thunks(self, arm, rk: Rank, count: int) -> list:
        """The calls of a graph or a pass.

        Each Call is made here, so a pass's timed loop launches and builds no
        views.
        """
        cold = self.state.code == "cold"
        out = []
        for i in range(count):
            if cold:
                out.append(lambda rk=rk: self.be.invalidate(rk.r, rk.stream))
            c = self.call(arm, rk, i)
            out.append(lambda c=c: arm.launch(c))
        return out

    def capture(self, arm, count: int) -> list:
        """A graph of `count` calls per rank, captured rank by rank."""
        return [self.make(rk, self.thunks(arm, rk, count)) for rk in self.ranks]

    def null_graphs(self, count: int, prelude: bool) -> list:
        out = []
        for rk in self.ranks:
            ts = []
            for _ in range(count):
                if prelude:
                    ts.append(lambda rk=rk: self.be.invalidate(rk.r, rk.stream))
                ts.append(lambda rk=rk: self.be.empty(rk.r, rk.stream))
            out.append(self.make(rk, ts))
        return out

    def check(self, inputs, outputs):
        """The point's check, timed (phase `check`)."""
        t = time.perf_counter()
        self.check_calls += 1
        try:
            return self.point.check(inputs, outputs)
        finally:
            self.check_s += time.perf_counter() - t

    def judge(self, arm) -> str | None:
        """checks, in order, after the arm's replay: the first failure."""
        suffix = (lambda r: f" rank {r}") if self.group else (lambda r: "")
        for rk in self.ranks:
            for name in rk.inputs:
                c = rk.copies[name]
                d = len(rk.differing(c, c.original))
                if d:
                    return f"input:{name}{suffix(rk.r)}: {d} of {c.n} copies changed"
        gathered: set = set()
        for rk in self.ranks:
            for name in rk.outputs:
                c = rk.copies[name]
                diff = rk.differing(c, c.row(0).clone())
                gathered |= set(diff)
                bad = [i for i in [0, *diff] if bool((c.row(i)[c.nbytes :] != GUARD).any())]
                count = (c.n - len(diff) + len(bad) - 1) if 0 in bad else len(bad)
                if count:
                    where = f"bounds:{name}{suffix(rk.r)}"
                    return f"{where}: the gap after {count} of {c.n} copies written"
        if arm.checked and self.point.check is not None:
            for i in [0, *sorted(gathered)]:
                why = self._check_copy(i)
                if why is not None:
                    return f"check: copy {i}: {why}"
        for rk in self.ranks:
            why = rk.pools.check(arm.name)
            if why is not None:
                return why + suffix(rk.r)
        return None

    def _check_copy(self, i: int):
        digest = hashlib.sha256()
        outs = []
        for rk in self.ranks:
            host = {}
            for name in rk.outputs:
                v = rk.copies[name].view(i).cpu().clone()
                digest.update(
                    rk.copies[name].row(i)[: rk.copies[name].nbytes].cpu().numpy().tobytes()
                )
                host[name] = v
            outs.append(host)
        key = digest.hexdigest()
        if key not in self.memo:
            # The check takes host tensors, inputs as outputs (a point may be
            # made of device tensors).
            ins = [{n: _host(v) for n, v in i.items()} for i in self.point.ranks()[0]]
            if self.group:
                self.memo[key] = self.check(list(ins), outs)
            else:
                self.memo[key] = self.check(ins[0], outs[0])
        return self.memo[key]


def _host(v):
    """A tensor on the host (itself if it is there), anything else as it is."""
    to = getattr(v, "cpu", None)
    return to() if callable(to) else v


def _median3(f) -> float:
    return statistics.median(f() for _ in range(3))


def race(arms, point, state=None, within=0.02, budget_s=5.0, seed=0) -> Race:
    """A race of arms at a point.

    Judges every arm, then times the right and unchecked ones in shuffled rounds
    until each verdict against the baseline is reached.
    """
    from metron import record

    state = state or State()
    arms = list(arms)
    _refuse(arms, point, state)
    wall, phases = time.perf_counter(), {}
    flags = _memory.hygiene(point.ranks()[0])
    be = _device.backend()
    world = point.world
    #: graphs or plain launches; on the graph path the graph calls alive bound
    # every graph of the race, refused before any allocation.
    archs = [be.rank_arch(r) for r in range(world)]
    plain = not _memory.graph_replays_ok(hip_version=be.hip_version(), archs=archs)
    if plain:
        most = _memory.PLAIN_MOST
    else:
        cold = state.code == "cold"
        most = _memory.most(_memory.graph_calls(), world, len(arms), cold, point.key)
    pool = concurrent.futures.ThreadPoolExecutor(world) if world > 1 else None
    alive: list = []
    try:
        t = time.perf_counter()
        if world > 1:
            be.enable_peers(world)
        ins, outs = point.ranks()
        rotated = max(
            sum(_memory._nbytes(v) for n, v in i.items() if n not in state.warm)
            + sum(_memory._nbytes(v) for v in o.values())
            for i, o in zip(ins, outs, strict=True)
        )
        cold_copies = _memory.copies(rotated, most, getattr(be, "cold", _memory.COLD))
        lay = _Layout(be, arms, point, state, cold_copies, most, plain)
        lay.sync()
        phases["lay_out"] = time.perf_counter() - t

        def each(fn):
            """fn(rank) on every rank, one host thread a rank in a group."""
            if pool is None:
                return [fn(rk) for rk in lay.ranks]
            return list(pool.map(fn, lay.ranks))

        def timed(graphs) -> list:
            """Per rank: (us per call, host us per call or None)."""
            return each(lambda rk: lay.timed(rk, graphs[rk.r]))

        t = time.perf_counter()
        results, kept, n_calls = {}, {}, None
        for k, arm in enumerate(arms):
            lay.restore()
            for rk in lay.ranks:
                with be.on(rk.r):
                    arm.launch(lay.call(arm, rk, 0))
            lay.sync()
            if k == 0:
                n_calls = _size(lay, arm, most, timed)
            graphs = lay.capture(arm, n_calls)
            lay.restore()
            each(lambda rk, graphs=graphs: lay.replay(rk, graphs[rk.r]))
            lay.sync()
            wrong = lay.judge(arm)
            read = {
                name: bytes(lay.ranks[0].pools.view(arm.name, name).cpu().numpy())
                for name in arm.read
            }
            results[arm.name] = ArmResult(
                wrong=wrong, checked=arm.checked, fingerprint=arm.fingerprint, read=read
            )
            if not (arm.checked and wrong):
                kept[arm.name] = graphs
                alive.append(graphs)
            else:
                lay.release(graphs)  # a wrong arm's graphs
        phases["judge"] = time.perf_counter() - t
        if lay.check_calls:
            phases["check"] = lay.check_s
        rounds, prelude, empty_us, status, prelude_flag = 0, 0.0, None, "untimed", None
        bound: list = []
        if kept:
            t = time.perf_counter()
            rounds, prelude, empty_us, status, prelude_flag, bound = _rounds(
                lay, kept, results, n_calls, state, within, budget_s, seed, timed, alive
            )
            phases["rounds"] = time.perf_counter() - t
        if prelude_flag:  # after the hygiene flags, before the after-flag
            flags.append(prelude_flag)
        flags += bound  # the plain path's launch-bound arms
        t = time.perf_counter()
        changed = 0
        for rk in lay.ranks:
            for name in rk.inputs:
                c = rk.copies[name]
                changed += len(rk.differing(c, c.original))
        if any(rk.pools.kept_changed() for rk in lay.ranks):
            flags.append("kept:changed in the rounds")
        phases["after"] = time.perf_counter() - t
        r0 = lay.ranks[0]
        conditions = {
            "world": world,
            "copies": r0.n,
            "cold_copies": cold_copies,
            "calls": n_calls,
            "bytes": {n: c.nbytes for n, c in r0.copies.items()},
            "rotated": [n for n, c in r0.copies.items() if c.n > 1],
            "empty_us": empty_us,
            "prelude_us": prelude,
            "within": within,
            "budget_s": budget_s,
            "seed": seed,
            "rounds_changed_inputs": changed,
        }
        dev = be.describe() if hasattr(be, "describe") else record.device()
        return Race(
            point=point.key,
            state=state.said(),
            arms=results,
            status=status,
            rounds=rounds,
            conditions=conditions,
            flags=flags,
            device=dev,
            validators=record.validators(dev, "plain" if plain else "graph"),
            phases_s={k: round(v, 3) for k, v in phases.items()},
            wall_s=round(time.perf_counter() - wall, 3),
            reference=arms[0].name,
        )
    finally:
        for graphs in alive:  # the kept and null graphs, at the race's end, even raising
            lay.release(graphs)
        if pool is not None:
            pool.shutdown(wait=True)


def _size(lay, arm, most, timed) -> int:
    """Size the baseline arm: the probe, the fitted copies, the sweep and N."""
    n = lay.ranks[0].n
    k = min(PROBE, n)
    probe = lay.capture(arm, k)
    lay.restore()
    call_us = _median3(lambda: max(us for us, _ in timed(probe)))
    lay.release(probe)
    f = _memory.fitted(n, call_us)
    if f < n:
        for rk in lay.ranks:
            rk.lay_out(f)
        lay.restore()
        n = f
    sweep = lay.capture(arm, n)
    lay.restore()
    sweep_us = _median3(lambda: max(us for us, _ in timed(sweep))) * n
    lay.release(sweep)
    return _memory.calls(n, sweep_us, most)


def entries(names, cold: bool) -> list:
    """The rounds' entries.

    The kept arms in arm order, the empty kernel and, under cold code, the prelude.
    """
    return [*names, "(empty)", *(["(prelude)"] if cold else [])]


def stats_of(raw: dict, ranks_raw: dict, reference: str, cold: bool) -> tuple:
    """(prelude_us, empty_us, adjusted times, ranks' medians, prelude flag) of raw times.

    The prelude is median((prelude)) - median((empty)) when cold and above noise:
    the low end of median_ci over the per-round differences above 0. Else
    it is 0.0, nothing is taken off, and the flag says the estimate.
    """
    empty = statistics.median(raw["(empty)"])
    prelude, flag = 0.0, None
    if cold:
        estimate = statistics.median(raw["(prelude)"]) - empty
        diffs = [p - e for p, e in zip(raw["(prelude)"], raw["(empty)"], strict=True)]
        if median_ci(diffs)[0] > 0:
            prelude = estimate
        else:
            flag = f"prelude:within noise ({estimate:.4f} us)"
    adjusted = {n: [x - prelude for x in xs] for n, xs in raw.items() if not n.startswith("(")}
    ranks_us = {n: [statistics.median(xs) - prelude for xs in per] for n, per in ranks_raw.items()}
    return prelude, empty, adjusted, ranks_us, flag


def launch_bound(arm: str, raw: list, host_us: list, within: float) -> str | None:
    """Flag of an arm paced by its launches, else None.

    Its median raw time within (1 + within) of the median host time to launch one call.
    """
    t, h = statistics.median(raw), statistics.median(host_us)
    if t <= (1 + within) * h:
        return f"launch-bound:{arm} ({t:.3f} us a call, {h:.3f} us to launch one)"
    return None


def _rounds(lay, kept, results, n_calls, state, within, budget_s, seed, timed, alive) -> tuple:
    cold = state.code == "cold"
    graphs = dict(kept)
    graphs["(empty)"] = lay.null_graphs(n_calls, False)
    alive.append(graphs["(empty)"])
    if cold:
        graphs["(prelude)"] = lay.null_graphs(n_calls, True)
        alive.append(graphs["(prelude)"])
    order = entries(list(kept), cold)
    rng = random.Random(seed)
    world = len(lay.ranks)
    raw = {e: [] for e in order}
    host = {e: [] for e in order}
    ranks_raw = {e: [[] for _ in range(world)] for e in order}
    start, rounds, reference = time.perf_counter(), 0, next(iter(results))
    while True:
        this = list(order)
        rng.shuffle(this)
        for e in this:
            lay.fresh()
            per = timed(graphs[e])
            slowest = max(range(world), key=lambda r: per[r][0])
            raw[e].append(per[slowest][0])
            host[e].append(per[slowest][1])  # the host_us of the round's rank
            for r, (us, _) in enumerate(per):
                ranks_raw[e][r].append(us)
        rounds += 1
        if rounds < MIN_ROUNDS:
            continue
        _, _, adjusted, _, _ = stats_of(raw, ranks_raw, reference, cold)
        done = settled(adjusted, reference, within)
        if done or time.perf_counter() - start > budget_s:
            break
    prelude, empty, adjusted, ranks_us, flag = stats_of(raw, ranks_raw, reference, cold)
    for name, times in adjusted.items():
        low, mid, high = median_ci(times)
        res = results[name]
        res.us, res.ci, res.times = mid, (low, high), times
        res.ranks_us = ranks_us[name]
    bound = []
    if lay.plain:
        for name in kept:
            f = launch_bound(name, raw[name], host[name], within)
            if f:
                bound.append(f)
    return rounds, prelude, empty, "decided" if done else "undecided", flag, bound
