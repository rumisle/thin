"""The plain path on the host backend: a stack whose
graph replays crash (HIP major 7 with gfx1201) captures nothing; every graph is
a pass launched by the rank's thread."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

ROCM7 = 70152802


def _point(n=4096):
    import metron

    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(n, generator=g), torch.randn(n, generator=g)

    def check(ins, outs):
        bad = int((outs["z"] != ins["x"] + ins["y"]).sum())
        return None if not bad else f"{bad} wrong"

    return metron.Point("add n=4096", {"x": x, "y": y}, {"z": torch.empty(n, device="meta")}, check)


def _add(a):
    torch.add(a.x, a.y, out=a.z)


def _wrong(a):
    torch.sub(a.x, a.y, out=a.z)


@pytest.fixture
def plain(monkeypatch):
    from metron import _device, _memory

    monkeypatch.delenv("METRON_GRAPHS", raising=False)
    # Host sweeps last microseconds, so calls() would fill REPLAY_US with ~15k
    # calls a pass; a smaller PLAIN_MOST keeps these tests fast (read at the race).
    monkeypatch.setattr(_memory, "PLAIN_MOST", 512)
    be = _device.HostBackend(cold=1 << 20, arch="gfx1201", hip=ROCM7)
    with _device.using(be):
        yield be


def test_the_stack_rule_picks_the_plain_path_and_captures_nothing(plain, tmp_path):
    import metron
    from metron import _memory

    state = metron.State(code="cold")
    arms = [metron.torch_arm("ref", _add), metron.torch_arm("b", _add)]
    r = metron.race(arms, _point(), state, budget_s=0.05)
    assert plain.captured == 0 and plain.peak_alive == 0
    assert r.validators["launches"] == "plain"
    c = r.conditions
    assert c["cold_copies"] == _memory.copies(3 * 4 * 4096, _memory.PLAIN_MOST, 1 << 20)
    assert 2 <= c["copies"] <= c["cold_copies"] and c["calls"] % c["copies"] == 0
    assert c["calls"] <= _memory.PLAIN_MOST
    assert r.rounds >= 6 and r.arms["b"].us is not None
    assert r.conditions["prelude_us"] >= 0
    path = tmp_path / "r.jsonl"
    r.save(path)
    [(validators, recs)] = metron.read(path)
    assert validators["launches"] == "plain" and len(recs) == 2


def test_a_planted_fault_is_judged_wrong_as_on_the_graph_path(plain):
    import metron

    r = metron.race([metron.torch_arm("ref", _add), metron.torch_arm("bad", _wrong)], _point(),
                    budget_s=0.05)  # fmt: skip
    assert r.arms["bad"].wrong.endswith("wrong") and r.arms["bad"].us is None
    assert plain.captured == 0


def test_an_arm_paced_by_its_launches_is_flagged(plain):
    """On the host every pass's device time is its launch time: launch-bound."""
    import metron

    r = metron.race([metron.torch_arm("ref", _add)], _point(), budget_s=0.05)
    bound = [f for f in r.flags if f.startswith("launch-bound:ref (")]
    assert len(bound) == 1 and bound[0].endswith(" us to launch one)")


def test_metron_graphs_decides_when_set(monkeypatch):
    import metron
    from metron import _device

    arms = [metron.torch_arm("ref", _add)]
    be = _device.HostBackend(cold=1 << 20, arch="gfx1201", hip=ROCM7)
    monkeypatch.setenv("METRON_GRAPHS", "1")
    with _device.using(be):
        r = metron.race(arms, _point(), budget_s=0.05)
    assert r.validators["launches"] == "graph" and be.captured > 0
    assert not any(f.startswith("launch-bound:") for f in r.flags)
    monkeypatch.setenv("METRON_GRAPHS", "x")
    with _device.using(be), pytest.raises(ValueError, match=r"METRON_GRAPHS='x': 1 \(graphs\)"):
        metron.race(arms, _point(), budget_s=0.05)
