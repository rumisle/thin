"""metron on an RX 9070 XT.

On ROCm 7.x with gfx1201 graph replays crash after about 2^14 launches a
process, so graph_replays_ok() is false and races take the plain
path: nothing is captured. Each test still runs in a process of its own.
"""

from __future__ import annotations

import hashlib

import pytest

# Each race runs in a process of its own.
pytestmark = [pytest.mark.gfx1201, pytest.mark.isolated]
torch = pytest.importorskip("torch")
pytest.importorskip("thin")

N = 1 << 23
"""vadd's n: 96 MiB a call, 22 copies to span COLD: graphs of a few hundred calls."""


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _vadd():
    import thin
    from examples.vadd import build_vadd

    return thin.compile(build_vadd, BLOCK=256, n=(1, (1 << 30) - 1))


def _kernarg(k, n, **swap):
    """kernarg(a) of vadd at n, pointer arguments a's tensors (or swap's names)."""
    host = {t: torch.empty(1) for t in ("x", "y", "z")}
    b = k.bind((1,), {**host, "n": n})

    def pack(a):
        return b.kernarg(lambda name: getattr(a, swap.get(name, name)).data_ptr())

    return pack


# --- the A/A race -------------------------------------------------------------------


def test_an_a_a_race_of_one_matmul_against_itself(tmp_path):
    import metron
    from metron import _memory, record

    n = 1024
    a, b = torch.randn(n, n), torch.randn(n, n)
    want = a @ b

    def check(ins, outs):
        return None if torch.allclose(outs["c"], want, rtol=1e-4, atol=1e-3) else "wrong"

    p = metron.Point(
        "matmul 1024", {"a": a, "b": b}, {"c": torch.empty(n, n, device="meta")}, check
    )

    def mm(x):
        torch.matmul(x.a, x.b, out=x.c)

    r = metron.race([metron.torch_arm("a", mm), metron.torch_arm("b", mm)], p)
    c = r.conditions
    assert r.arms["a"].wrong is None and r.arms["b"].wrong is None
    assert r.rounds >= 6
    ab, ba = r.versus("b", "a"), r.versus("a", "b")
    assert abs(ab.ratio - 1) < 0.03, ab
    assert abs(ab.ratio * ba.ratio - 1) < 1e-3
    rotated = sum(c["bytes"][name] for name in c["rotated"])
    m = _memory.PLAIN_MOST
    assert r.validators["launches"] == "plain"
    assert c["copies"] * rotated >= _memory.COLD or c["copies"] == m
    assert c["calls"] % c["copies"] == 0 and c["calls"] <= m
    assert c["empty_us"] > 0 and c["prelude_us"] == 0
    assert c["rounds_changed_inputs"] == 0 and not r.flags
    path = tmp_path / "r.jsonl"
    r.save(path)
    [(validators, recs)] = metron.read(path)
    assert validators == record.current()
    assert [x["arm"] for x in recs] == ["a", "b"]
    assert recs[0]["verdict"] is None


# --- planted faults among compiler-made arms ------------------------------------------


def gen_vadd_strided(BLOCK: int = 256):  # noqa: N803 - a build parameter
    """z = x + y over n, `iters` strides of `step` lanes: any grid covers n."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        x, y, z, n, step, iters = k.load()
        i = V.u32(k.wgid * BLOCK + k.tid, name="i")
        for _ in b.loop(iters, name="r"):
            with b.lanes(i < n):
                off = V.u32(i * 4, name="off")
                a, c = V.f32(name="a"), V.f32(name="c")
                b.emit(O.global_load(a, off, x))
                b.emit(O.global_load(c, off, y))
                b.wait(a, c, "x, y landed")
                s = V.f32(a + c, name="s")
                b.emit(O.global_store(off, s, z))
            i[...] = i + step

    args = Args(
        x=ptr, y=ptr, z=ptr, n=u32.within(1, 1 << 30), step=u32.within(1, 1 << 30),
        iters=u32.within(1, 1 << 20),
    )  # fmt: skip
    return KernelDef(f"vadd_strided_b{BLOCK}", args, body, wgsize=BLOCK, max_grid=(65536,))


def _cooperative_arm(name):
    """vadd as one cooperative launch of the co-resident bound's workgroups."""
    import metron
    import thin
    from metron import _device

    k = thin.compile(gen_vadd_strided)
    cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    grid = cus * k.manifest.resident
    step = grid * 256
    host = {t: torch.empty(1) for t in ("x", "y", "z")}
    b = k.bind((grid,), {**host, "n": N, "step": step, "iters": -(-N // step)})

    def launch(a):
        p = _device.program(k.co, k.name, (256, 1))
        _device.launch(p, (grid,), b.kernarg(lambda t: getattr(a, t).data_ptr()), a.stream,
                       cooperative=True)  # fmt: skip

    return metron.Arm(name, launch, hashlib.sha256(k.co).hexdigest()[:16])


def test_planted_faults_are_caught_and_the_right_arms_timed():
    import metron
    import metron.thin as mt
    from examples.vadd_op import VAdd, add

    p = mt.point(VAdd(n=N))
    k = _vadd()
    full = -(-N // 256)
    arms = [
        mt.impl(p, add, name="ref"),
        metron.co_arm("short", k.co, k.name, (full - 1,), _kernarg(k, N)),
        metron.co_arm("past", k.co, k.name, (full + 1,), _kernarg(k, N + 64)),
        _cooperative_arm("coop"),
        metron.co_arm("into_x", k.co, k.name, (full,), _kernarg(k, N, z="x")),
        metron.co_arm("unchecked", k.co, k.name, (full - 1,), _kernarg(k, N), checked=False),
    ]
    r = metron.race(arms, p)
    w = {n: a.wrong for n, a in r.arms.items()}
    assert w["ref"] is None and w["coop"] is None
    assert w["short"].startswith("check: copy 0: "), w["short"]
    assert w["past"].startswith("bounds:z: the gap after "), w["past"]
    assert w["into_x"].startswith("input:x: "), w["into_x"]
    assert w["unchecked"] is None or r.arms["unchecked"].checked is False
    timed = {n for n, a in r.arms.items() if a.us is not None}
    assert timed == {"ref", "coop", "unchecked"}
    assert set(r.verdicts) == {"coop", "unchecked"}


# --- cold code, a warm input, the cold point ----------------------------------------


def test_cold_code_and_a_warm_input():
    import metron
    import metron.thin as mt
    from examples.vadd_op import VAdd, add

    p = mt.point(VAdd(n=N))
    state = metron.State(warm={"x"}, code="cold")
    r = metron.race([mt.impl(p, add)], p, state=state, budget_s=3)
    assert r.conditions["rotated"] == ["y", "z"]
    assert r.state == {"warm": ["x"], "code": "cold"}
    # A prelude within noise is 0 and flagged; on gfx1201 the flag is expected.
    noisy = [f for f in r.flags if f.startswith("prelude:within noise (")]
    assert r.conditions["prelude_us"] >= 0
    assert (r.conditions["prelude_us"] > 0) != bool(noisy), (r.conditions["prelude_us"], r.flags)


def _note(key, value) -> None:
    """Adds a measurement to.cache/hw/plain_path.json (the tests run in children)."""
    import json
    from pathlib import Path

    path = Path(__file__).parents[2] / ".cache" / "hw" / "plain_path.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[key] = value
    path.write_text(json.dumps(doc, indent=1))


def _captures(monkeypatch):
    """A count of the GPU backend's graph captures (the plain path makes none)."""
    from metron import _device

    be = _device.backend()
    count = [0]
    real = be.capture

    def capture(rank, thunks):
        count[0] += 1
        return real(rank, thunks)

    monkeypatch.setattr(be, "capture", capture)
    return count


def test_this_stack_takes_the_plain_path(monkeypatch):
    """ROCm 7.x with gfx1201 replays no graphs; METRON_GRAPHS=x is refused."""
    import metron
    from metron import _device, record

    monkeypatch.delenv("METRON_GRAPHS", raising=False)
    assert _device.hip_version() // 10**7 == 7
    assert metron.graph_replays_ok() is False
    assert record.current()["launches"] == "plain"
    monkeypatch.setenv("METRON_GRAPHS", "x")
    with pytest.raises(ValueError, match="METRON_GRAPHS='x'"):
        metron.graph_replays_ok()


@pytest.mark.slow("gfx1201-plain")
def test_a_point_of_128_bytes_under_cold_code(monkeypatch):
    """two arms, one rank, cold code, on the plain path: cold_copies =
    PLAIN_MOST; 2 <= copies <= cold_copies (fitted to the call time; call_us is
    not reported, so the bound alone); calls a multiple of copies, <= most;
    through judging and >= 6 rounds, nothing captured."""
    import metron
    from metron import _memory

    monkeypatch.delenv("METRON_GRAPHS", raising=False)
    caps = _captures(monkeypatch)
    x = torch.arange(32, dtype=torch.float32)
    p = metron.Point("tiny", {"x": x}, {"z": torch.empty(32, device="meta")}, lambda i, o: None)

    def inc(a):
        torch.add(a.x, 1, out=a.z)

    arms = [metron.torch_arm("t", inc), metron.torch_arm("u", inc)]
    r = metron.race(arms, p, state=metron.State(code="cold"), budget_s=2)
    m = _memory.PLAIN_MOST
    c = r.conditions
    assert c["cold_copies"] == m and 2 <= c["copies"] <= c["cold_copies"], c
    assert c["calls"] % c["copies"] == 0 and c["calls"] <= m, c
    assert r.rounds >= 6 and caps[0] == 0
    _note("128 B cold", {k: c[k] for k in ("copies", "cold_copies", "calls", "prelude_us",
                                          "empty_us")} | {"rounds": r.rounds, "flags": r.flags,
                                                          "us": r.arms["t"].us})  # fmt: skip


def _matmul_point(n=256):
    import metron

    a, b = torch.randn(n, n), torch.randn(n, n)
    want = a @ b

    def check(ins, outs):
        return None if torch.allclose(outs["c"], want, rtol=1e-4, atol=1e-3) else "wrong"

    return metron.Point(
        f"matmul {n}", {"a": a, "b": b}, {"c": torch.empty(n, n, device="meta")}, check
    )


def _mm(x):
    torch.matmul(x.a, x.b, out=x.c)


@pytest.mark.slow("gfx1201-plain")
def test_the_plain_path_holds_a_small_matmul_race_and_20_more(monkeypatch):
    """an A/A race of a 256x256 f32 matmul (cold_copies = copies(rotated,
    PLAIN_MOST) = 2731, copies <= it), which crashed the graph path within about 3 rounds,
    runs through judging and >= 6 rounds within 3 % of 1; then 20 more such
    races in this process; nothing is captured."""
    import metron

    monkeypatch.delenv("METRON_GRAPHS", raising=False)
    caps = _captures(monkeypatch)
    p = _matmul_point()
    arms = [metron.torch_arm("a", _mm), metron.torch_arm("b", _mm)]
    r = metron.race(arms, p)
    c = r.conditions
    assert c["cold_copies"] == 2731 and 2 <= c["copies"] <= 2731, c
    assert r.arms["a"].wrong is None and r.arms["b"].wrong is None
    assert r.rounds >= 6 and abs(r.versus("b", "a").ratio - 1) < 0.03, r.verdicts
    v = r.verdicts["b"]
    _note("256x256 A/A", {"status": r.status, "rounds": r.rounds, "calls": r.conditions["calls"],
                          "a_us": r.arms["a"].us, "b_us": r.arms["b"].us, "ratio": v.ratio,
                          "verdict": v.verdict, "flags": r.flags})  # fmt: skip
    ratios = []
    for _ in range(20):
        q = metron.race(arms, p, budget_s=1)
        assert q.rounds >= 6
        ratios.append(round(q.versus("b", "a").ratio, 4))
    assert caps[0] == 0
    _note("20 more A/A ratios", ratios)


def test_the_empty_kernel_as_an_arm_is_launch_bound(monkeypatch):
    """an arm that launches only the empty kernel is paced by the host."""
    import metron
    from metron import _device

    monkeypatch.delenv("METRON_GRAPHS", raising=False)
    be = _device.backend()
    p = _matmul_point(64)

    def empty(a):
        be.empty(a.rank, a.stream)

    arms = [metron.torch_arm("mm", _mm), metron.Arm("empty", empty, "empty", checked=False)]
    r = metron.race(arms, p, budget_s=1)
    bound = [f for f in r.flags if f.startswith("launch-bound:empty (")]
    assert bound, r.flags
    _note("empty as an arm", {"flags": r.flags, "empty_us": r.conditions["empty_us"]})


# --- a co_arm beside the adapter's arm of the same kernel ------------------------------


def test_a_co_arm_beside_the_adapters_arm_of_the_same_kernel():
    import metron
    import metron.thin as mt
    from examples.vadd_op import VAdd, add

    p = mt.point(VAdd(n=N))
    k = _vadd()
    co = metron.co_arm("co", k.co, k.name, (-(-N // 256),), _kernarg(k, N))
    r = metron.race([mt.impl(p, add, BLOCK=256), co], p, budget_s=3)
    assert all(a.wrong is None for a in r.arms.values())
    assert co.fingerprint == hashlib.sha256(k.co).hexdigest()[:16]


# --- owned memory ------------------------------------------------------------------------


def _owned_arm(name, kind, leave):
    """z = x + y with a torch arm, writing its owned buffer `b` and restoring it
    to zero unless `leave`."""
    import metron

    def launch(a):
        torch.add(a.x, a.y, out=a.z)
        buf = a.owned["b"]
        buf.fill_(7)
        if not leave:
            buf.zero_()

    return metron.Arm(name, launch, name, owned={"b": (4096, kind)})


@pytest.mark.parametrize("kind", ["kept", "kept.uncached"])
def test_kept_memory_left_as_found_is_right_and_changed_is_reported(kind):
    import metron
    import metron.thin as mt
    from examples.vadd_op import VAdd

    p = mt.point(VAdd(n=N))
    r = metron.race([_owned_arm("tidy", kind, False), _owned_arm("messy", kind, True)], p,
                    budget_s=3)  # fmt: skip
    assert r.arms["tidy"].wrong is None
    assert (r.arms["messy"].wrong or "").startswith("kept:b: not left as found")
    assert not any(f.startswith("kept:changed") for f in r.flags)


@pytest.mark.parametrize("kind", ["scratch", "carried.uncached", "gens.uncached"])
def test_other_owned_kinds_and_their_uncached_pools(kind):
    import metron
    import metron.thin as mt
    from examples.vadd_op import VAdd

    p = mt.point(VAdd(n=N))
    r = metron.race([_owned_arm("own", kind, True)], p, budget_s=3)
    assert r.arms["own"].wrong is None


def test_laps_read_back_and_decoded():
    """vadd built with laps: its records read back after the race and decoded."""
    import metron
    import metron.thin as mt
    from examples.vadd import build_vadd
    from examples.vadd_op import VAdd, add

    n = 1 << 16
    p = mt.point(VAdd(n=n))
    arm = mt.impl(p, add, BLOCK=256, build={build_vadd: {"laps": True}}, read=("laps",))
    r = metron.race([arm], p, budget_s=2)
    assert r.arms[arm.name].wrong is None
    t = mt.laps(arm, r)
    assert t.rounds == [[1] * 8] * (n // 256)
    assert 300 < t.clock < 5000, t.summary()  # pooled: a wave's own spans are a few ticks
    phases = t.phases()
    assert len(phases["load"]) == len(phases["store"]) == n // 32
    assert all(0 < c < 1 << 24 for c in phases["load"] + phases["store"])
    assert "laps (first)" in t.summary()


def test_a_race_aborted_by_a_refused_launch_leaves_the_process_usable():
    """a cooperative arm whose grid HIP refuses (720) aborts the race
    with the layer's RuntimeError; then torch synchronizes and an A/A race runs."""
    import metron
    import metron.thin as mt
    import thin
    from examples.vadd_op import VAdd, add
    from metron import _device

    k = thin.compile(gen_vadd_strided)
    host = {t: torch.empty(1) for t in ("x", "y", "z")}
    grid = 1 << 16  # far past what is co-resident
    b = k.bind((grid,), {**host, "n": N, "step": grid * 256, "iters": 1})

    def launch(a):
        p = _device.program(k.co, k.name, (256, 1))
        _device.launch(p, (grid,), b.kernarg(lambda t: getattr(a, t).data_ptr()), a.stream,
                       cooperative=True)  # fmt: skip

    p = mt.point(VAdd(n=N))
    with pytest.raises(RuntimeError, match=r"cooperative launch: .*\(720\)"):
        metron.race([mt.impl(p, add), metron.Arm("big", launch, "big")], p, budget_s=2)
    torch.cuda.synchronize()
    r = metron.race([mt.impl(p, add, name="a"), mt.impl(p, add, name="b")], p, budget_s=2)
    assert r.arms["a"].wrong is None and r.arms["b"].wrong is None
