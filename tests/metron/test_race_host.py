"""A race end to end on the host backend: torch arms on CPU
tensors and copies spanning 1 MiB. The graph path is forced on; budget
tests count stubbed captures. The plain path's
tests are in test_plain_host.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def graphs_on(monkeypatch):
    monkeypatch.setenv("METRON_GRAPHS", "1")


def _point(n=4096):
    import metron

    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(n, generator=g), torch.randn(n, generator=g)

    def check(ins, outs):
        bad = int((outs["z"] != ins["x"] + ins["y"]).sum())
        return None if not bad else f"{bad} wrong"

    return metron.Point("add n=4096", {"x": x, "y": y}, {"z": torch.empty(n, device="meta")}, check)


def _race(arms, **kw):
    import metron
    from metron import _device

    with _device.using(_device.HostBackend(cold=1 << 20)):
        return metron.race(arms, _point(), budget_s=0.5, **kw)


def _add(a):
    torch.add(a.x, a.y, out=a.z)


def test_a_race_judges_then_times_the_right_arms(tmp_path):
    import metron

    def wrong(a):
        torch.sub(a.x, a.y, out=a.z)

    r = _race([metron.torch_arm("ref", _add), metron.torch_arm("bad", wrong)])
    assert r.reference == "ref"
    assert r.arms["bad"].wrong.endswith("wrong") and r.arms["bad"].us is None
    assert r.arms["ref"].wrong is None and r.arms["ref"].us > 0
    assert r.rounds >= 1 and "bad" not in r.verdicts
    path = tmp_path / "r.jsonl"
    r.save(path)
    [(_validators, recs)] = metron.read(path)
    assert {rec["arm"] for rec in recs} == {"ref", "bad"}


def _fake_clock(seed):
    """A host backend whose replays take 10 us times seeded lognormal noise
    (sigma 1 %), the same law for every graph: an exchangeable A/A clock. The
    calls still run, so judging is real; only the times are fixed by the seed."""
    import math
    import random

    from metron import _device

    rng = random.Random(seed)

    class Clock(_device.HostBackend):
        def timed(self, rank, graph):
            self.replay(rank, graph)
            return 10.0 * math.exp(rng.gauss(0.0, 0.01))

    return Clock(cold=1 << 20)


def test_an_a_a_race_is_a_tie_on_a_fake_clock():
    """The same call twice, timed by an exchangeable clock: a tie.

    The host clock's noise made this flaky: the race stops at the first
    decided look, so an A/A race on real timings can say faster or slower
    (see the next test). A seeded clock makes the verdict a fixed result.
    """
    import metron
    from metron import _device

    with _device.using(_fake_clock(seed=0)):
        r = metron.race(
            [metron.torch_arm("a", _add), metron.torch_arm("b", _add)], _point(), budget_s=30
        )
    assert r.arms["a"].wrong is None and r.arms["b"].wrong is None
    assert r.verdicts["b"].verdict == "tie" and r.rounds >= 6


def test_the_sequential_rule_rarely_splits_an_a_a_race():
    """rule on exchangeable times (sigma 1 %), 100 seeded races:
    stopping at the first settled round from MIN_ROUNDS on, at most 3 say
    faster or slower (1 does at these seeds; 11 % at sigma 3 %)."""
    import math
    import random

    from metron._memory import MIN_ROUNDS
    from metron.stats import ratio_verdict, settled

    split = 0
    for seed in range(100):
        rng = random.Random(seed)
        t = {"a": [], "b": []}
        for r in range(1, 300):
            for xs in t.values():
                xs.append(10.0 * math.exp(rng.gauss(0.0, 0.01)))
            if r >= MIN_ROUNDS and settled(t, "a", 0.02):
                break
        split += ratio_verdict(t["b"], t["a"], 0.02)[2] in ("faster", "slower")
    assert split <= 3


def test_a_ranks_restores_compares_and_direct_calls_run_on_its_stream(monkeypatch):
    """Found on gfx1201: restores on the default stream raced the rank's replay and
    poisoned its outputs after it wrote them (both arms of an A/A matmul judged
    wrong). Every write of a rank's memory runs inside the backend's on(rank)."""
    import metron
    from metron import _device, _memory

    class Ordered(_device.HostBackend):
        depth = 0

        def on(self, rank):
            import contextlib

            @contextlib.contextmanager
            def block():
                Ordered.depth += 1
                try:
                    yield
                finally:
                    Ordered.depth -= 1

            return block()

        def differ(self, *args):
            assert Ordered.depth, "the compare outside on(rank)"
            return super().differ(*args)

    restore = _memory.Copies.restore

    def checked(self):
        assert Ordered.depth, "a restore outside on(rank)"
        return restore(self)

    monkeypatch.setattr(_memory.Copies, "restore", checked)
    seen = []

    def arm(a):
        seen.append(Ordered.depth or "captured")
        _add(a)

    with _device.using(Ordered(cold=1 << 20)):
        r = metron.race([metron.torch_arm("ref", arm)], _point(), budget_s=0.2)
    assert r.arms["ref"].wrong is None
    assert seen[0] == 1  # the judge's direct call


class _Counting:
    """A host backend recording every capture and release."""

    def __new__(cls, **kw):
        from metron import _device

        class Counting(_device.HostBackend):
            def __init__(self, **kw):
                # the stack graph replays crash on, the graph path forced
                super().__init__(arch="gfx1201", hip=70152802, **kw)
                self.events, self.allocs = [], 0

            def alloc(self, rank, nbytes):
                self.allocs += 1
                return super().alloc(rank, nbytes)

            def capture(self, rank, thunks):
                g = super().capture(rank, thunks)
                self.events.append(("capture", len(thunks), self.alive))
                return g

            def release(self, rank, graph):
                n = graph.n if graph.calls is not None else 0
                super().release(rank, graph)
                self.events.append(("release", n, self.alive))

        return Counting(**kw)


@pytest.mark.parametrize("cold", [False, True])
def test_the_graphs_alive_never_exceed_the_budget(monkeypatch, cold):
    """on a stubbed capture: every capture and release counted; the alive
    launches stay within graph_calls(); the probe and the sweep are released before
    the baseline's graph is captured; nothing is left alive at the end."""
    import metron
    from metron import _device

    monkeypatch.setenv("METRON_GRAPH_CALLS", "300")
    be = _Counting(cold=1 << 30)  # most binds: tiny rows, many copies wanted
    with _device.using(be):
        r = metron.race(
            [metron.torch_arm("a", _add), metron.torch_arm("b", _add)],
            _point(), state=metron.State(code="cold" if cold else "warm"), budget_s=0.3,
        )  # fmt: skip
    assert r.arms["a"].wrong is None and r.rounds >= 1
    assert be.peak_alive <= 300, be.events
    assert be.alive == 0
    kinds = [e[0] for e in be.events]
    assert kinds[:4] == ["capture", "release", "capture", "release"]  # probe, sweep
    m = metron.most(300, 1, 2, cold)
    assert r.conditions["copies"] <= m and r.conditions["calls"] <= m


def test_a_race_past_the_budget_is_refused_before_any_allocation(monkeypatch):
    import metron
    from metron import _device

    monkeypatch.setenv("METRON_GRAPH_CALLS", "1")
    be = _Counting(cold=1 << 20)
    with (
        _device.using(be),
        pytest.raises(ValueError, match=r"race at add n=4096: 2 arms .* holds 1"),
    ):
        metron.race([metron.torch_arm("a", _add), metron.torch_arm("b", _add)], _point())
    assert be.allocs == 0 and not be.events


def test_a_malformed_budget_is_refused(monkeypatch):
    import metron
    from metron import _device

    monkeypatch.setenv("METRON_GRAPH_CALLS", "x")
    with (
        _device.using(_Counting(cold=1 << 20)),
        pytest.raises(ValueError, match="METRON_GRAPH_CALLS='x'"),
    ):
        metron.race([metron.torch_arm("a", _add)], _point())


def test_a_wrong_arms_graphs_are_released_once_judged(monkeypatch):
    import metron
    from metron import _device

    def wrong(a):
        torch.sub(a.x, a.y, out=a.z)

    be = _Counting(cold=1 << 20)
    with _device.using(be):
        r = metron.race([metron.torch_arm("ref", _add), metron.torch_arm("bad", wrong)], _point(),
                        budget_s=0.2)  # fmt: skip
    assert r.arms["bad"].wrong
    n = r.conditions["calls"]
    # after the baseline's graph (kept), the bad arm's graph is captured and released
    caps = [i for i, e in enumerate(be.events) if e[0] == "capture" and e[1] == n]
    assert be.events[caps[1] + 1] == ("release", n, n)
    assert be.alive == 0
