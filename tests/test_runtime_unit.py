"""Unit tests of the runtime beyond its vectors."""

from __future__ import annotations

import os
import random
import signal
import stat
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Launch arithmetic -------------------------------------------------------------


def _divides(d: int, n: int) -> bool:
    from thin.runtime.arith import magic

    mul, shift = magic(d)
    hi = (n * mul) >> 32
    return ((hi + n) & 0xFFFFFFFF) >> shift == n // d


def test_magic_divides():
    rng = random.Random(7)
    divisors = list(range(1, 1 << 9)) + [rng.randrange(1, 1 << 16) for _ in range(300)]
    for p in range(1, 32):
        divisors += [(1 << p) - 1, 1 << p, (1 << p) + 1]
    for d in sorted(set(x for x in divisors if 1 <= x <= 1 << 31)):
        top = ((1 << 31) - 1) // d * d
        ns = {
            0,
            d - 1,
            d,
            d + 1,
            2 * d - 1,
            2 * d,
            top,
            top - 1,
            (1 << 31) - 1,
            rng.randrange(1 << 31),
        }
        assert all(_divides(d, n) for n in ns if 0 <= n < 1 << 31), d


def test_arith_cdiv_refuses_zero_and_the_root_does_not():
    import thin
    from thin.runtime.arith import cdiv
    from thin.runtime.errors import LaunchError

    with pytest.raises(LaunchError, match="division by 0"):
        cdiv(1, 0)
    with pytest.raises(ZeroDivisionError):
        thin.cdiv(1, 0)
    assert thin.cdiv(-7, 2) == -3


# --- The store --------------------------------------------------------------------


def _addr(tag="x"):
    from thin.runtime import store

    return store.address("result", tag)


def test_publish_then_lookup(tmp_path):
    from thin.runtime import store

    a = _addr()
    assert store.publish(tmp_path, a, {"v": b"1"}) == "published"
    path = store.path_of(tmp_path, a)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o444
    assert path == os.path.join(tmp_path, "result", a[7:9], a[7:])
    with open(path, "rb") as f:
        assert f.read().startswith(store.MAGIC)
    assert os.listdir(tmp_path / "tmp") == []
    assert store.lookup(tmp_path, a) == {"v": b"1"}


def test_raced_equal_and_collision(tmp_path):
    from thin.runtime import store
    from thin.runtime.errors import Collision

    a = _addr()
    store.publish(tmp_path, a, {"v": b"1"})
    assert store.publish(tmp_path, a, {"v": b"1"}) == "raced equal"
    with pytest.raises(Collision, match="holds other bytes"):
        store.publish(tmp_path, a, {"v": b"2"})
    assert not issubclass(Collision, ValueError)


def _broken(path):
    os.chmod(path, 0o644)
    with open(path, "r+b") as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))


def test_a_broken_entry_is_a_miss_never_repaired_and_evicted(tmp_path):
    from thin.runtime import store

    a = _addr()
    store.publish(tmp_path, a, {"v": b"1"})
    _broken(store.path_of(tmp_path, a))
    assert store.lookup(tmp_path, a) is store.Miss.CORRUPT
    got = store.get(tmp_path, a, lambda: {"v": b"1"}, wait=5, sources={})
    assert got == {"v": b"1"}
    assert store.lookup(tmp_path, a) is store.Miss.CORRUPT
    assert store.evict(tmp_path, 0) == 0
    assert store.evict(tmp_path, 0, digests=True) == 1
    store.get(tmp_path, a, lambda: {"v": b"1"}, wait=5, sources={})
    assert store.lookup(tmp_path, a) == {"v": b"1"}


def test_get_computes_once_then_finds(tmp_path):
    from thin.runtime import store

    calls = []
    a = _addr("once")

    def compute():
        calls.append(1)
        return {"v": b"9"}

    for _ in range(3):
        assert store.get(tmp_path, a, compute, wait=5, sources={}) == {"v": b"9"}
    assert calls == [1]
    with pytest.raises(TypeError):
        store.get(tmp_path, a, compute)  # wait and sources are required


def test_get_compute_raising_stores_nothing_and_a_missing_member(tmp_path):
    from thin.runtime import store

    a = _addr("raise")

    def bad():
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        store.get(tmp_path, a, bad, wait=5, sources={})
    assert store.lookup(tmp_path, a) is store.Miss.ABSENT
    with pytest.raises(KeyError, match="returned no"):
        store.get(tmp_path, a, lambda: {"v": b""}, ("v", "w"), wait=5, sources={})


def test_get_does_not_store_when_a_source_changed(tmp_path):
    from thin.runtime import store

    src = tmp_path / "gen.py"
    src.write_text("a")
    a = _addr("src")
    sources = {str(src): "0" * 64}
    assert store.get(tmp_path / "s", a, lambda: {"v": b"1"}, wait=5, sources=sources) == {"v": b"1"}
    assert store.lookup(tmp_path / "s", a) is store.Miss.ABSENT


def test_an_unwritable_store_returns(tmp_path):
    from thin.runtime import store

    root = tmp_path / "ro"
    root.mkdir()
    os.chmod(root, 0o555)
    try:
        got = store.get(root, _addr("ro"), lambda: {"v": b"1"}, wait=5, sources={})
    finally:
        os.chmod(root, 0o755)
    assert got == {"v": b"1"}


def test_lookup_writes_nothing(tmp_path):
    from thin.runtime import store

    a = _addr()
    store.publish(tmp_path, a, {"v": b"1"})

    def snapshot():
        out = {}
        for folder, _, names in os.walk(tmp_path):
            for n in names:
                s = os.stat(os.path.join(folder, n))
                out[os.path.join(folder, n)] = (s.st_mtime_ns, s.st_size, s.st_mode, s.st_ino)
        return out

    before = snapshot()
    store.lookup(tmp_path, a)
    store.lookup(tmp_path, _addr("absent"))
    assert snapshot() == before


def test_report_is_a_copy():
    from thin.runtime import store

    r = store.report()
    r["x"] = 1
    assert "x" not in store.report()


def test_evict_by_date_and_its_command_line(tmp_path):
    from thin.runtime import store

    store.publish(tmp_path, _addr("a"), {"v": b"1"})
    store.publish(tmp_path, _addr("b"), {"v": b"2"})
    out = subprocess.run(
        [sys.executable, "-m", "thin.store", "evict", str(tmp_path), "--before", "2999-01-01"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "2 removed"
    assert store.lookup(tmp_path, _addr("a")) is store.Miss.ABSENT


_WORKER = textwrap.dedent(
    """
    import sys, time
    from thin.runtime import store
    root, tag = sys.argv[1], sys.argv[2]
    def compute():
        time.sleep(0.2)
        return {"v": b"same"}
    store.get(root, store.address("result", tag), compute, wait=30, sources={})
    r = store.report()
    print("computed" if any(k.endswith(".computed") for k in r) else "hit")
    """
)


def test_sixteen_processes_compute_once(tmp_path):
    tag = f"many-{os.getpid()}-{random.randrange(1 << 30)}"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(tmp_path), tag], stdout=subprocess.PIPE, text=True
        )
        for _ in range(16)
    ]
    outs = sorted(p.communicate()[0].strip() for p in procs)
    assert outs.count("computed") == 1
    assert outs.count("hit") == 15


_HOLDER = textwrap.dedent(
    """
    import sys, time
    from thin.runtime import store
    root, tag = sys.argv[1], sys.argv[2]
    def compute():
        print("holding", flush=True)
        time.sleep(60)
    store.get(root, store.address("result", tag), compute, wait=30, sources={})
    """
)


def test_a_killed_holder_leaves_nothing(tmp_path):
    from thin.runtime import store

    tag = f"killed-{os.getpid()}-{random.randrange(1 << 30)}"
    p = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(tmp_path), tag], stdout=subprocess.PIPE, text=True
    )
    assert p.stdout.readline().strip() == "holding"
    p.send_signal(signal.SIGKILL)
    p.wait()
    a = store.address("result", tag)
    assert store.get(tmp_path, a, lambda: {"v": b"1"}, wait=30, sources={}) == {"v": b"1"}
    assert store.report().get("result.computed", 0) >= 1


# --- compile and binding -------------------------------------------------------------


def gen_small(BLOCK: int = 64, n: tuple = (1, 100), ablate=frozenset()):  # noqa: N803
    """A small kernel: stores nothing, reads its argument block."""
    from thin import KernelDef, Refused
    from thin.dsl import Args, ptr, u32

    def host(bind, x, n):
        if n % 2:
            raise Refused(f"small: n = {n}: an even count")
        return bind(grid=1, args={"x": x, "n": n})

    return KernelDef(
        "small", Args(x=ptr, n=u32.within(*n)), lambda b, k: None, wgsize=BLOCK, host=host
    )


@pytest.fixture
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def test_one_kernel_per_address(no_store):
    import thin

    a = thin.compile(gen_small)
    assert thin.compile(gen_small, BLOCK=64) is a
    assert thin.compile(gen_small, BLOCK=32) is not a
    assert thin.compile(gen_small, n=(1, 50)).address != a.address
    s1 = thin.compile(gen_small, ablate={"a", "b"})
    assert thin.compile(gen_small, ablate={"b", "a"}) is s1
    assert s1.address != thin.compile(gen_small, ablate=("a", "b")).address


def test_compile_refusals(no_store):
    import thin
    from thin.opspec import Field

    with pytest.raises(TypeError, match="a run-time field's value"):
        thin.compile(gen_small, n=Field(5, 1, 9))
    with pytest.raises(TypeError, match="unexpected keyword argument 'W'"):
        thin.compile(gen_small, W=1)
    with pytest.raises(TypeError, match=r"a generator returns thin\.KernelDef"):
        thin.declare(lambda: 3)


def test_the_host_and_refused(no_store):
    import numpy as np

    import thin
    from thin import Refused

    k = thin.compile(gen_small)
    x = np.zeros(4, np.float32)
    assert k(x=x, n=4).grid == (1, 1, 1)
    with pytest.raises(Refused, match="an even count"):
        k(x=x, n=3)
    d = k.definition
    assert d.accepts(n=3) == "small: n = 3: an even count"
    assert d.accepts(n=4) is None


def gen_keywords(BLOCK: int = 32, **extra):  # noqa: N803
    """A generator taking keywords: each is a build parameter of its own."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    def host(bind, x, over=None):
        return bind(grid=0 if extra.get("bad_grid") else 1, args={"x": x}, overread=over)

    return KernelDef("kw", Args(x=ptr), lambda b, k: None, wgsize=BLOCK, host=host)


def test_keywords_are_build_parameters_and_accepts_checks_as_bind_does(no_store):
    import thin

    plain = thin.compile(gen_keywords)
    assert plain.params == {"BLOCK": 32}
    k = thin.compile(gen_keywords, flavour="a")
    assert k.params == {"BLOCK": 32, "flavour": "a"} and k.address != plain.address
    d = k.definition
    assert d.accepts() is None
    assert d.accepts(over={"x": 0}) == (
        "kw: overread {'x': 0}: a pointer argument's bytes past its end, a positive int"
    )
    bad = thin.compile(gen_keywords, bad_grid=True).definition
    assert bad.accepts() == "kw: grid 0: one to three positive ints"


def gen_default_grid():
    from thin import KernelDef
    from thin.dsl import Args, ptr

    return KernelDef("dg", Args(x=ptr), lambda b, k: None, wgsize=32, grid=(3,), max_grid=(4,))


def test_a_definitions_grid_and_build_reach_the_manifest(no_store):
    import numpy as np

    import thin

    k = thin.compile(gen_default_grid)
    assert k.manifest.grid == [3]
    assert k.bind(args={"x": np.zeros(4, np.float32)}).grid == (3, 1, 1)
    assert thin.compile(gen_small, BLOCK=32).manifest.build == {
        "BLOCK": "32",
        "ablate": "set()",
        "n": "(1, 100)",
    }


def test_a_buffer_is_an_array_or_a_tensor_and_contiguous(no_store):
    import numpy as np

    import thin
    from thin.runtime.bound import is_buffer
    from thin.runtime.errors import LaunchError

    assert is_buffer(np.zeros(4)) and not is_buffer(np.float32(1)) and not is_buffer(np.dtype("u4"))
    k = thin.compile(gen_small)
    x = np.zeros((4, 4), np.float32)
    with pytest.raises(LaunchError, match="argument x is not contiguous"):
        k.bind(1, {"x": x.T, "n": 2})
    with pytest.raises(LaunchError, match="a pointer: a buffer, not UInt32DType"):
        k.bind(1, {"x": np.dtype("u4"), "n": 2})
    assert k.bind(1, {"x": x, "n": 2}).grid == (1, 1, 1)


def test_accepts_stand_in_and_declare_render_nothing(no_store):
    import thin
    from thin import KernelDef
    from thin.dsl import Args, ptr
    from thin.runtime import store

    def reads(bind, x):
        return bind(grid=len(x), args={"x": x})

    d = KernelDef("r", Args(x=ptr), lambda b, k: None, wgsize=32, host=reads)
    with pytest.raises(TypeError, match="the host reads pointer argument x, not given"):
        d.accepts()
    before = store.report()
    thin.declare(gen_small, BLOCK=16).accepts(n=2)
    assert store.report() == before


def test_overriding_leaves_another_generator(no_store):
    import thin
    from thin.runtime.params import bind_params

    with thin.overriding({gen_small: {"BLOCK": 8}}):
        assert bind_params(gen_small, {})["BLOCK"] == 8
        assert bind_params(gen_other, {})["BLOCK"] == 1
        with (
            pytest.raises(TypeError, match="unexpected keyword"),
            thin.overriding({gen_small: {"W": 1}}),
        ):
            thin.declare(gen_small)


def gen_other(BLOCK: int = 1):  # noqa: N803
    """Another generator overriding leaves alone."""
    raise AssertionError("never run")


def test_a_compile_through_the_store_hits_in_another_process(tmp_path):
    code = textwrap.dedent(
        f"""
        import sys; sys.path.insert(0, {ROOT!r})
        import thin
        from examples.vadd import build_vadd
        from thin.runtime import store
        thin.use_store({str(tmp_path)!r})
        k = thin.compile(build_vadd)
        print(k.address, sorted(x for x in store.report() if x.startswith("render.")))
        """
    )
    a = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    b = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert a.split()[0] == b.split()[0]
    assert "render.computed" in a and "render.computed" not in b and "render.hit" in b


# --- Own memory and emulate on gfx1201 -----------------------------------------


def test_held_table_newest_first_contents():
    from thin.runtime.bound import HeldTable, _first

    t = HeldTable()
    a = t.lookup("k", 8, _first("kept", 8))
    assert t.lookup("k", 4, _first("kept", 4)) is a
    big = t.lookup("k", 16, _first("kept", 16))
    assert big is not a and t.buffers["k"] == [a, big]
    gens = t.lookup("g", 8, _first("gens", 8))
    assert gens == ((1 << 32) - 3).to_bytes(4, "little") * 2
    first = t.lookup("f", 4, _first("kept", b"\x01\x02"))
    assert first == b"\x01\x02\x00\x00"


def test_same_grid_refused_on_a_second_grid():
    from thin.runtime.bound import HeldTable
    from thin.runtime.errors import LaunchError

    t = HeldTable()
    t.same_grid("k", "key", "c", (2, 1, 1), "why")
    t.same_grid("k", "key", "c", (2, 1, 1), "why")
    with pytest.raises(
        LaunchError, match=r"grid \(3, 1, 1\) on its held memory c, first bound with \(2, 1, 1\)"
    ):
        t.same_grid("k", "key", "c", (3, 1, 1), "why")


def gen_own(kind="kept", change=False):
    """out[t] = m[t] for 64 words of own memory m, then m[t] += 1 - except
    kept, which a `change` launch leaves changed and otherwise restores."""
    from thin import KernelDef
    from thin.dsl import Args, O, carried, gens, kept, ptr, scratch

    own = {"kept": kept, "carried": carried, "gens": gens, "scratch": scratch}[kind]

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        out, m = k.load()
        off = V.u32(k.tid * 4, name="off")
        v = V.u32(name="v")
        b.emit(O.global_load(v, off, m))
        b.wait(v, "m landed")
        b.emit(O.global_store(off, v, out))
        if kind != "scratch":
            w = V.u32(v + 1 if kind != "kept" or change else v, name="w")
            b.emit(O.global_store(off, w, m))

    return KernelDef(f"own_{kind}", Args(out=ptr, m=own), body, wgsize=64, max_grid=(1,))


def _own_call(k, m):
    import numpy as np

    out = np.zeros(64, np.uint32)
    bound = k.bind(grid=1, args={"out": out, "m": m})
    bound.emulate()
    return out, bound.owned["m"]


def test_scratch_arrives_ff_and_is_read_back_owned(no_store):
    import numpy as np

    import thin

    k = thin.compile(gen_own, kind="scratch")
    out, owned = _own_call(k, 256)
    assert (out == 0xFFFF_FFFF).all()
    assert isinstance(owned, np.ndarray) and owned.dtype == np.uint8 and owned.size == 256


def test_an_overread_of_own_memory_is_refused(no_store):
    import numpy as np

    import thin
    from thin.runtime.errors import LaunchError

    k = thin.compile(gen_own, kind="scratch")
    out = np.zeros(64, np.uint32)
    with pytest.raises(LaunchError, match="overread of m: the kernel's own memory"):
        k.bind(grid=1, args={"out": out, "m": 256}, overread={"m": 4})


def test_carried_is_held_across_calls_in_one_buffer(no_store):
    import thin

    k = thin.compile(gen_own, kind="carried")
    for launch in range(3):
        out, owned = _own_call(k, 256)
        assert (out == launch).all(), launch
        assert (owned.view("<u4") == launch + 1).all()


def test_gens_start_at_2_32_minus_3_and_a_third_launch_crosses_the_wrap(no_store):
    import thin

    k = thin.compile(gen_own, kind="gens")
    seen = [int(_own_call(k, 256)[0][0]) for _ in range(4)]
    assert seen == [2**32 - 3, 2**32 - 2, 2**32 - 1, 0]


def test_kept_is_zero_held_and_must_be_left_as_found(no_store):
    import thin
    from thin.runtime.errors import DeviceFault

    k = thin.compile(gen_own, kind="kept")
    out, owned = _own_call(k, 256)
    assert (out == 0).all() and (owned == 0).all()
    changed = thin.compile(gen_own, kind="kept", change=True)
    import numpy as np

    out = np.full(64, 5, np.uint32)
    with pytest.raises(DeviceFault, match="kept memory m: 64 of 256 bytes not left") as e:
        changed.bind(grid=1, args={"out": out, "m": 256}).emulate()
    assert e.value.cause == "kept:m"
    assert (out == 5).all()  # the fault came before any write-back


def test_kept_given_first_bytes_is_held_per_bytes(no_store):
    import thin

    k = thin.compile(gen_own, kind="kept")
    first = bytes([7, 0, 0, 0]) * 64
    out, _ = _own_call(k, first)
    assert (out == 7).all()
    out, _ = _own_call(k, bytes([9, 0, 0, 0]) * 64)
    assert (out == 9).all()
    out, _ = _own_call(k, 256)
    assert (out == 0).all()


def test_an_owned_argument_takes_its_size_not_a_buffer(no_store):
    import numpy as np

    import thin
    from thin.runtime.errors import LaunchError

    k = thin.compile(gen_own, kind="carried")
    with pytest.raises(LaunchError, match="is the kernel's own carried memory: give its size"):
        k.bind(grid=1, args={"out": np.zeros(64, np.uint32), "m": np.zeros(64, np.uint32)})


@pytest.mark.parametrize("n", [1, 255, 256, 1000])
def test_emulate_runs_vadd(no_store, n):
    """vadd on the emulated device at n = 1, 255, 256, 1000."""
    import numpy as np

    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd)
    x = np.arange(n, dtype=np.float32)
    out = k(x=x, y=2 * x, z=np.zeros(n, np.float32), n=n).emulate()
    assert np.array_equal(out["z"], 3 * x)


def test_emulate_too_small_a_grid_leaves_the_rest(no_store):
    import numpy as np

    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd)
    x = np.arange(600, dtype=np.float32)
    z = np.full(600, -1, np.float32)
    out = k.bind(grid=2, args={"x": x, "y": x, "z": z, "n": 600}).emulate()
    assert np.array_equal(out["z"][:512], 2 * x[:512]) and (out["z"][512:] == -1).all()


def test_emulate_co_resident_runs_and_is_refused_past_a_residency(no_store):
    import numpy as np

    import thin
    from examples.vadd import build_vadd
    from thin.runtime.errors import LaunchError

    k = thin.compile(build_vadd)
    x = np.arange(512, dtype=np.float32)
    args = {"x": x, "y": x, "z": np.zeros(512, np.float32), "n": 512}
    out = k.bind(grid=2, args=args, co_resident=True).emulate()
    assert np.array_equal(out["z"], 2 * x)
    with pytest.raises(LaunchError, match="co-resident, 2 workgroups; sched resident 1"):
        k.bind(grid=2, args=args, co_resident=True).emulate({"resident": 1})


def test_emulate_a_read_past_a_buffer_faults(no_store):
    """a read past a buffer faults naming the instruction."""
    import numpy as np

    import thin
    from examples.vadd import build_vadd
    from thin.runtime.errors import DeviceFault

    k = thin.compile(build_vadd)
    x = np.arange(4, dtype=np.float32)
    with pytest.raises(DeviceFault, match="global_load_b32") as e:
        k.bind(grid=1, args={"x": x, "y": x, "z": np.zeros(8, np.float32), "n": 8}).emulate()
    assert e.value.cause == "fault"
    args = {"x": x, "y": x, "z": np.zeros(8, np.float32), "n": 8}
    out = k.bind(grid=1, args=args, overread={"x": 16, "y": 16}).emulate()
    assert np.array_equal(out["z"][:4], 2 * x) and len(out["x"]) == 4


def test_buffer_and_made_by_buffer():
    import numpy as np

    from thin.runtime.bound import buffer, made_by_buffer

    like = np.zeros(2, np.float32)
    b = buffer(8, like)
    assert b.tobytes() == b"\xff" * 8 and made_by_buffer(b) and not made_by_buffer(like)
    with pytest.raises(ValueError, match=r"buffer\(0\): bytes"):
        buffer(0, like)
    with pytest.raises(TypeError, match=r"buffer\(like=int\)"):
        buffer(8, 3)


# --- The child environment and HIP's discovery -----------------------------------


def test_env_child_and_hold(monkeypatch):
    from thin.runtime import env

    monkeypatch.setenv("NM_GPU_EXCLUSIVE", "1")
    monkeypatch.setenv("SECRET_THING", "x")
    c = env.child(env.HASH_SEED)
    assert (
        c["NM_GPU_EXCLUSIVE"] == "1" and c["PYTHONHASHSEED"] == "12345" and "SECRET_THING" not in c
    )
    assert env.gpu_hold() == "exclusive"
    monkeypatch.setenv("NM_GPU_EXCLUSIVE", "0")
    assert env.gpu_hold() == "shared"
    with pytest.raises(ValueError, match="only the named constants"):
        env.child({"PYTHONHASHSEED": "12345"})


def test_hip_found_says_what_it_found():
    from thin.runtime import hip

    text = hip.found()
    assert text == "no HIP" or text.startswith("no GPU, HIP ") or ", HIP " in text


gpu = pytest.mark.gfx1201


@gpu
def test_vadd_runs_on_gfx1201(stored):  # pragma: no cover - hardware
    import torch

    import thin
    from examples.vadd import build_vadd

    k = thin.compile(build_vadd)
    x, y = torch.arange(1000.0, device="cuda"), torch.ones(1000, device="cuda")
    z = torch.empty(1000, device="cuda")
    k(x=x, y=y, z=z, n=1000).run()
    torch.cuda.synchronize()
    assert torch.equal(z, x + y)
    assert k(x=x, y=y, z=z, n=1000).time() > 0


# --- The files an address is made from --------------------------------------------------


def _vadd_address():
    """vadd's hashed packages and versions, made here, outside any trace."""
    import inspect

    import thin
    from examples.vadd import build_vadd
    from thin.runtime import packages

    return packages.packages_for([thin.__file__, inspect.getsourcefile(build_vadd)])


def test_the_render_reads_nothing_outside_its_address():
    """load the generator's file, render its KernelDef and code object; nothing else."""
    from thin.runtime import packages

    gen = os.path.join(ROOT, "examples", "vadd.py")
    code = textwrap.dedent(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("vadd", {gen!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from thin import codegen
        codegen.code_object(module.build_vadd().compile().text)
        """
    )
    assert packages.outside(code, *_vadd_address()) == set()


def test_the_examples_read_nothing_outside_their_folder():
    """examples/ is a folder of its own (no __init__.py), so an example's address
    hashes it and not its siblings at the repo root (tests, with tests/data, and
    tools): importing every example and rendering reads nothing else."""
    from thin.runtime import packages

    hashed, versions = _vadd_address()
    assert os.path.join(ROOT, "examples") in hashed and ROOT not in hashed
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {ROOT!r})
        import examples.gfx942.gemm_bf16, examples.gfx950.gemm_bf16, examples.vadd, examples.vadd_op
        from thin import codegen
        from thin.target import targeting
        codegen.code_object(examples.vadd.build_vadd().compile().text)
        with targeting("gfx942"):
            codegen.code_object(examples.gfx942.gemm_bf16.build_gemm_bf16(2048).compile().text)
        with targeting("gfx950"):
            codegen.code_object(examples.gfx950.gemm_bf16.build_gemm_bf16(1024).compile().text)
        """
    )
    assert packages.outside(code, hashed, versions) == set()


def test_a_trace_that_reads_metadata_fails():
    """making an address inside the trace reads distributions' metadata, and it counts."""
    from thin.runtime import packages

    code = f"import sys; sys.path.insert(0, {ROOT!r})\nfrom thin.runtime import packages\n"
    code += "packages.distributions()\n"
    found = packages.outside(code, *_vadd_address())
    assert any(".dist-info" in p for p in found)


def _loose(path):
    """(hashed tops, files relative to their folder) of a loose generator."""
    from thin.runtime import packages

    hashed, versions = packages.packages_for([str(path)])
    assert versions == ()
    return hashed, sorted(rel for top in hashed for rel, _ in packages.files_of(top))


def test_a_lone_script_is_its_folder_s_modules_and_packages(tmp_path):
    import importlib.util

    from thin.runtime import packages

    gen = tmp_path / "gen.py"
    gen.write_text("x = 1\n")
    (tmp_path / "helper.py").write_text("y = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "table.json").write_text("[]")
    (tmp_path / "notes.txt").write_text("unrelated")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.py").write_text("not importable from here")
    (tmp_path / "not-a-module.py").write_text("")
    hashed, files = _loose(gen)
    assert hashed == (str(tmp_path),)
    assert files == ["gen.py", "helper.py", "pkg/__init__.py", "pkg/table.json"]

    def digest():
        return packages.packages_digest(hashed, ())

    before = digest()
    (tmp_path / "notes.txt").write_text("changed")
    (tmp_path / "data" / "x.py").write_text("changed")
    assert digest() == before
    for changed in (gen, tmp_path / "helper.py", tmp_path / "pkg" / "table.json"):
        changed.write_text(changed.read_text() + " ")
        assert digest() != before
        before = digest()
    with open(importlib.util.cache_from_source(str(gen)), "rb") as f:
        assert int.from_bytes(f.read(8)[4:8], "little") == 0b11


def test_a_loose_digest_is_the_same_in_any_folder(tmp_path):
    from thin.runtime import packages

    digests = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "gen.py").write_text("x = 1\n")
        hashed, _ = _loose(tmp_path / name / "gen.py")
        digests.append(packages.packages_digest(hashed, ()))
    assert digests[0] == digests[1]


def test_a_module_in_a_loose_package_hashes_the_whole_package(tmp_path):
    (tmp_path / "top" / "sub").mkdir(parents=True)
    for p in ("top/__init__.py", "top/sub/__init__.py", "top/sub/gen.py", "top/util.py"):
        (tmp_path / p).write_text("")
    (tmp_path / "sibling.py").write_text("")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "big.bin").write_bytes(b"0" * 4096)
    hashed, files = _loose(tmp_path / "top" / "sub" / "gen.py")
    assert hashed == (str(tmp_path),)
    assert files == [
        "sibling.py",
        "top/__init__.py",
        "top/sub/__init__.py",
        "top/sub/gen.py",
        "top/util.py",
    ]


def test_a_script_named_as_no_module_is_still_hashed(tmp_path):
    gen = tmp_path / "my-gen.py"
    gen.write_text("")
    (tmp_path / "helper.py").write_text("")
    hashed, files = _loose(gen)
    assert hashed == tuple(sorted((str(tmp_path), str(gen))))
    assert files == ["helper.py", "my-gen.py"]


def _socket_file(path) -> None:
    """A Unix socket at ``path``, bound by its name from its folder: a socket's
    path may be 104 bytes on macOS, which a temporary folder there can pass."""
    import socket

    here = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        s = socket.socket(socket.AF_UNIX)
        s.bind(os.path.basename(path))
        s.close()
    finally:
        os.chdir(here)


def test_special_files_beside_a_script_are_not_read(tmp_path):
    """A socket or FIFO (e.g. a session bus in /tmp) has no bytes; opening a FIFO would block."""
    gen = tmp_path / "gen.py"
    gen.write_text("")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    for folder in (tmp_path, tmp_path / "pkg"):
        os.mkfifo(folder / "fifo.py")
        _socket_file(folder / "bus.py")
        os.symlink(folder / "missing", folder / "broken.py")
    _, files = _loose(gen)
    assert files == ["gen.py", "pkg/__init__.py"]


def test_a_huge_unrelated_tree_beside_a_script_is_not_walked(tmp_path, monkeypatch):
    """A script in a home folder or /tmp: nothing below a folder without __init__.py is read."""
    from thin.runtime import packages

    gen = tmp_path / "gen.py"
    gen.write_text("")
    for i in range(20):
        d = tmp_path / "tree" / f"d{i}" / "deeper"
        d.mkdir(parents=True)
        for j in range(50):
            (d / f"f{j}.py").write_text("")
    with open(tmp_path / "huge.bin", "wb") as f:
        f.truncate(1 << 40)  # sparse: a terabyte that costs nothing on disk
    walked = []
    real_walk = os.walk
    monkeypatch.setattr(
        packages.os, "walk", lambda p, *a, **k: walked.append(p) or real_walk(p, *a, **k)
    )
    _, files = _loose(gen)
    assert files == ["gen.py"]
    assert walked == []


def test_a_script_in_a_busy_folder_compiles_through_the_store(tmp_path):
    """A generator beside a socket and a FIFO, compiled in a fresh process."""
    import shutil

    folder = tmp_path / "busy"
    folder.mkdir()
    shutil.copy(os.path.join(ROOT, "examples", "vadd.py"), folder / "gen.py")
    os.mkfifo(folder / "fifo")
    _socket_file(folder / "bus")
    code = (
        f"import sys; sys.path.insert(0, {str(folder)!r})\n"
        "import thin, gen\n"
        "print(thin.compile(gen.build_vadd).address)\n"
    )
    env = {**os.environ, "THIN_STORE": str(tmp_path / "store")}
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


def test_an_unknown_top_is_refused():
    from thin.runtime import packages

    with pytest.raises(ValueError, match="'no_such_top_xyz': no such package"):
        packages.files_of("no_such_top_xyz")


# --- The emulated device's queue ----------------------------------------------


class _Recorder:
    """The emulator's memory with a run that records the launches it is given."""

    target, wave = "test", 64

    def __init__(self):
        from thin.runtime.emulated import Emulator

        self._memory = Emulator()
        self.buffers = self._memory.buffers
        self.runs = []

    def alloc(self, nbytes, fill, meta, rank=0):
        return self._memory.alloc(nbytes, fill, meta, rank)

    def read(self, k, offset, n):
        return self._memory.read(k, offset, n)

    def write(self, k, offset, data):
        self._memory.write(k, offset, data)

    def run(self, launches, stall_rounds, interleaving, staged=None):
        self.runs.append([x[5] for x in launches])


def _queued_group(world, **interleaving):
    from thin.runtime import emulated
    from thin.runtime.device import Program

    shared = emulated._Shared(world, 20000, interleaving, _Recorder())
    devices = [emulated.EmulatedDevice(shared, r) for r in range(world)]
    progs = [Program(d, "k", (64, 1), 0, "code") for d in devices]
    return shared, devices, progs


def test_a_groups_ranks_queue_until_a_wait_then_run_together():
    from thin.runtime import dispatch

    shared, (_, b), (pa, pb) = _queued_group(2)
    dispatch(pa, (1, 1, 1), b"", {}, "k rank 0")
    dispatch(pb, (1, 1, 1), b"", {}, "k rank 1")
    assert shared.emu.runs == []
    b.wait()
    assert shared.emu.runs == [["k rank 0", "k rank 1"]]


def test_a_groups_run_refusals():
    from thin.runtime import dispatch
    from thin.runtime.errors import LaunchError, NotEmulated

    _, (a, _b), (pa, pb) = _queued_group(2, resident=2)
    with pytest.raises(
        LaunchError, match=r"a cooperative launch of 3 workgroups; the group's residency 2"
    ):
        dispatch(pa, (3, 1, 1), b"", {}, "w", cooperative=True)
    dispatch(pa, (1, 1, 1), b"", {}, "one")
    dispatch(pb, (2, 1, 1), b"", {}, "two")
    with pytest.raises(LaunchError, match="one kernel and one grid a group's run"):
        a.wait()
    dispatch(pa, (1, 1, 2), b"", {}, "z")
    with pytest.raises(NotEmulated, match="z: a grid of 2 in z on the emulator"):
        a.wait()


def test_the_bank_warning_text():
    """the machines' bank reports summed, most conflict cycles first."""
    import warnings
    from types import SimpleNamespace

    from thin.runtime.emulated import BankConflicts, _bank_warning

    machines = [
        SimpleNamespace(banks={3: (2, 8, 4), 5: (2, 4, 0)}),
        SimpleNamespace(banks={3: (2, 8, 4), 7: (1, 3, 1)}),
    ]
    texts = {3: ("ds_load_b32 v1, v2", 12, "the row"), 5: ("ds_store_b32 v2, v1", 9, ""),
             7: ("ds_load_b64 v[4:5], v2", 15, "")}  # fmt: skip
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _bank_warning("k", machines, texts, "gfx1201")
    (w,) = seen
    assert w.category is BankConflicts
    assert str(w.message) == "\n".join(
        [
            "k: LDS bank conflicts (the gfx1201 bank model) at 2 of 3 ds instructions, "
            "9 of 23 cycles",
            "  .s line 12: `ds_load_b32 v1, v2`  // the row: 4 executions, 4 cycles an execution, "
            "2 of them conflicts",
            "  .s line 15: `ds_load_b64 v[4:5], v2`: 1 executions, 3 cycles an execution, "
            "1 of them conflicts",
        ]
    )


def test_gfx1201_has_no_bank_model_so_no_run_warns(no_store):
    """gfx1201's LDS bank rules are not yet measured, so a
    conflicting LDS pattern reports nothing and warns nothing."""
    import warnings

    import numpy as np

    import thin

    k = thin.compile(gen_lds_conflicts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = np.zeros(64, np.uint32)
        k.bind(grid=1, args={"out": out}).emulate()
    assert out.tolist() == list(range(64))


def gen_lds_conflicts():
    """Every lane writes and reads the LDS at a 32-word stride (one bank, were there a model)."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        (out,) = k.load()
        T = b.lds().region("T", (64 * 32,), "u32")  # noqa: N806
        v = V.u32(k.tid, name="v")
        b.emit(O.ds_write(T.at(k.tid * 32), v))
        b.wait(T)
        b.barrier("T written")
        r = V.u32(name="r")
        b.emit(O.ds_read(r, T.at(k.tid * 32)))
        b.wait(r)
        b.emit(O.global_store(V.u32(k.tid * 4, name="o"), r, out))

    return KernelDef("lds_conflicts", Args(out=ptr), body, wgsize=64, max_grid=(1,))


def gen_stale():
    """WG 0 stores the data and then the flag, not waiting between; WG 1 reads the
    flag and then the data: nothing orders the data before the flag."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        d, f, o = k.load()
        off = V.u32(k.tid * 4, name="off")
        with b.when(k.wgid == 0):
            b.emit(O.global_store(off, V.u32(7, name="seven"), d))
            b.emit(O.global_store(V.u32(0, name="z"), V.u32(1, name="one"), f, scope="SCOPE_DEV"))
        with b.otherwise():
            g = V.u32(name="g")
            b.emit(O.global_load(g, V.u32(0, name="z2"), f, scope="SCOPE_DEV"))
            b.wait(g, "the flag")
            x = V.u32(name="x")
            b.emit(O.global_load(x, off, d))
            b.wait(x, "the data")
            b.emit(O.global_store(off, x, o))

    return KernelDef("stale", Args(d=ptr, f=ptr, o=ptr), body, wgsize=32, max_grid=(2,))


def test_a_stale_read_the_flight_model_served_is_reported(no_store):
    import numpy as np

    import thin
    from thin.runtime.emulated import StaleReads

    k = thin.compile(gen_stale)
    d, f, o = np.zeros(32, np.uint32), np.zeros(1, np.uint32), np.zeros(32, np.uint32)
    with pytest.warns(
        StaleReads, match=r"stale: 1 stale reads served\n  wave 0 of WG 1 .* \(storecnt;"
    ):
        k.bind((2,), {"d": d, "f": f, "o": o}).emulate()
    assert o.tolist() == [0] * 32  # the previous data, served


def test_a_hip_error_raised_is_cleared_from_hips_sticky_error(monkeypatch):
    """A refusal HIP returned is raised once: its sticky last error is cleared, so
    the tensor library's next call does not report it as its own (found on
    gfx1201: a refused cooperative launch failed torch's next synchronize)."""
    from thin.runtime import hip
    from thin.runtime.errors import LaunchError

    class Fake:
        cleared = 0

        def hipGetErrorString(self, code):  # noqa: N802 - HIP's name
            return b"too many blocks in cooperative launch"

        def hipGetLastError(self):  # noqa: N802 - HIP's name
            Fake.cleared += 1
            return 0

    monkeypatch.setattr(hip, "_LIB", [Fake()])
    with pytest.raises(LaunchError, match=r"k: cooperative launch: too many blocks .* \(720\)"):
        hip._check(720, "k: cooperative launch")
    with pytest.raises(LaunchError, match="hipMemset: HIP error 1"):
        hip._call(1, "hipMemset")
    assert Fake.cleared == 2
    hip._check(0, "fine")
    assert Fake.cleared == 2


# --- Regression checks: options and captures in the render's address ---


def _address(gen, **params):
    """The address compile would use, without rendering."""
    from thin._options import output_pairs
    from thin.runtime import kernel
    from thin.runtime.params import bind_params, held_pairs
    from thin.target import target

    identity, digest, _ = kernel._sources(gen)
    pairs = held_pairs(bind_params(gen, params))
    return kernel.render_address(identity, pairs, output_pairs(), target().name, digest)


def test_vadd_under_using_through_one_store(tmp_path):
    """defaults, then using(safe_waits=True), then using(peak_report=True), one
    process, one store: two Kernels, the second waiting with every count 0, and
    the third the first itself."""
    import re

    import thin
    from examples.vadd import build_vadd
    from thin.runtime import kernel

    before = thin.use_store(str(tmp_path))
    kernel.forget()
    try:
        plain = thin.compile(build_vadd)
        with thin.codegen.using(safe_waits=True):
            safe = thin.compile(build_vadd)
        with thin.codegen.using(peak_report=True):
            same = thin.compile(build_vadd)
    finally:
        thin.use_store(before)
    assert safe is not plain and safe.address != plain.address
    assert same is plain
    counts = re.findall(r"s_wait_\w+cnt (\d+)", safe.asm)
    assert counts and set(counts) == {"0"}


def test_nested_using_gives_the_innermost_records_address():
    from examples.vadd import build_vadd
    from thin import codegen

    with codegen.using(safe_waits=True), codegen.using(regions=False):
        nested = _address(build_vadd)
    with codegen.using(safe_waits=True, regions=False):
        flat = _address(build_vadd)
    assert nested == flat != _address(build_vadd)


def test_every_output_option_flipped_moves_the_address_and_no_other_does():
    import dataclasses

    from examples.vadd import build_vadd
    from thin import codegen

    base = _address(build_vadd)
    for f in dataclasses.fields(codegen.Options):
        flipped = (not f.default) if isinstance(f.default, bool) else f.default + 1
        with codegen.using(**{f.name: flipped}):
            moved = _address(build_vadd) != base
        assert moved == f.metadata["output"], f.name


def _closure_factory(w):
    """Two closures of one factory, told apart by their captured W."""
    from examples.vadd import build_vadd

    def gen(BLOCK: int = 256):  # noqa: N803 - a build parameter
        d = build_vadd(BLOCK=BLOCK)
        d.meta["W"] = w
        return d

    return gen


def test_two_closures_of_one_factory_are_two_kernels():
    import thin

    before = thin.use_store(None)
    try:
        a, b = thin.compile(_closure_factory(64)), thin.compile(_closure_factory(128))
        again = thin.compile(_closure_factory(64))
    finally:
        thin.use_store(before)
    assert a is not b and a.address != b.address
    assert again.address == a.address


def test_a_rebound_cell_is_seen_at_the_next_compile():
    from thin.runtime import generator_identity

    w = 1

    def gen():
        return w

    first = generator_identity(gen)
    w = 2
    assert generator_identity(gen) != first
    assert dict(generator_identity(gen)[3]) == {"w": "2"}


def _identity_in_a_child(folder):
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import gen_mod; "
        "from thin.runtime import generator_identity; print(repr(generator_identity(gen_mod.gen)))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(folder)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_helpers_edit_changes_the_address(tmp_path):
    """A generator capturing a helper of another file: editing the helper's file
    (not the generator's) gives another identity, in the next process."""
    (tmp_path / "helper_mod.py").write_text("def scale(x):\n    return 2 * x\n")
    (tmp_path / "gen_mod.py").write_text(
        "from helper_mod import scale\n\n"
        "def factory():\n    f = scale\n    def gen():\n        return f(1)\n    return gen\n\n"
        "gen = factory()\n"
    )
    before = _identity_in_a_child(tmp_path)
    assert before == _identity_in_a_child(tmp_path)
    (tmp_path / "helper_mod.py").write_text("def scale(x):\n    return 3 * x\n")
    after = _identity_in_a_child(tmp_path)
    assert after != before and "helper_mod" not in before.split("'gen_mod")[0]


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("", "~/.cache/thin-store"),
        ("off", None),
        ("~/stores/x", "~/stores/x"),
        ("/tmp/s", "/tmp/s"),
    ],
)
def test_thin_store_sets_the_default_store(value, want):
    """THIN_STORE, read once when a store is first needed."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "THIN_STORE"}
    if value:
        env["THIN_STORE"] = value
    code = "import thin; print(repr(thin.use_store(None)))"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    want = repr(os.path.expanduser(want) if want else None)
    assert done.stdout.strip() == want, done.stderr


def test_thin_store_is_read_once():
    import os
    import subprocess
    import sys

    code = (
        "import os, thin; from thin.runtime import kernel; first = kernel._store(); "
        "os.environ['THIN_STORE'] = '/elsewhere'; print(kernel._store() == first)"
    )
    env = {**os.environ, "THIN_STORE": "/first"}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert done.stdout.strip() == "True", done.stderr


def test_the_packages_digest_is_computed_once_a_process_for_its_tops(monkeypatch, tmp_path):
    """a second generator under the same tops reuses the digest; one in a folder of
    its own is under other tops."""
    import importlib.util
    import shutil

    from thin.runtime import kernel, packages

    shutil.copy(os.path.join(ROOT, "examples", "vadd.py"), tmp_path / "own.py")
    spec = importlib.util.spec_from_file_location("own", tmp_path / "own.py")
    own = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(own)
    calls = []
    real = packages.packages_digest
    monkeypatch.setattr(packages, "packages_digest", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(kernel, "_SOURCES", {})
    monkeypatch.setattr(kernel, "_DIGESTS", {})
    kernel._sources(_closure_factory(1))
    kernel._sources(_closure_factory(2))
    assert len(calls) == 1
    kernel._sources(own.build_vadd)
    assert len(calls) == 2


class _FakeHip:
    """HIP's calls stubbed: `fail` names the one that returns its error code."""

    def __init__(self, fail=None, code=1):
        self.fail, self.code, self.cleared = fail, code, 0

    def __getattr__(self, name):
        if name == "hipGetLastError":

            def clear():
                self.cleared += 1
                return 0

            return clear
        if name == "hipGetErrorString":
            return lambda code: b"stubbed"
        return lambda *a: self.code if name == self.fail else 0


_HIP_PATHS = {
    "hipModuleLoadData": lambda hip: hip.module(b"co", "k", "k"),
    "hipModuleGetFunction": lambda hip: hip.module(b"co", "k", "k"),
    "hipModuleLaunchKernel": lambda hip: hip.enqueue(None, (1, 1, 1), (32, 1), b"\0" * 4, 0, "k"),
    "hipModuleLaunchCooperativeKernel": lambda hip: hip.enqueue(
        None, (1, 1, 1), (32, 1), b"\0" * 4, 0, "k", cooperative=True
    ),
    "hipExtMallocWithFlags": lambda hip: hip.alloc(16),
    "hipMemset": lambda hip: hip.alloc(16),
    "hipDeviceEnablePeerAccess": lambda hip: hip.enable_peers([0, 1]),
    "hipIpcGetMemHandle": lambda hip: hip.ipc_handle(1),
    "hipIpcOpenMemHandle": lambda hip: hip.open_ipc(b"\0" * 64),
    "hipIpcCloseMemHandle": lambda hip: hip.close_ipc(1),
}


@pytest.mark.parametrize("call", sorted(_HIP_PATHS))
def test_every_launch_error_clears_hips_last_error_once(monkeypatch, call):
    """Each LaunchError path calls hipGetLastError exactly
    once, after the failing call and before raising."""
    from thin.runtime import hip
    from thin.runtime.errors import LaunchError

    fake = _FakeHip(fail=call)
    monkeypatch.setattr(hip, "_LIB", [fake])
    with pytest.raises(LaunchError):
        _HIP_PATHS[call](hip)
    assert fake.cleared == 1


def test_a_successful_hip_call_leaves_the_last_error_alone(monkeypatch):
    from thin.runtime import hip

    fake = _FakeHip()
    monkeypatch.setattr(hip, "_LIB", [fake])
    for name, path in _HIP_PATHS.items():
        if name != "hipDeviceEnablePeerAccess":
            path(hip)
    assert fake.cleared == 0
    fake = _FakeHip(fail="hipDeviceEnablePeerAccess", code=704)  # already enabled: accepted
    monkeypatch.setattr(hip, "_LIB", [fake])
    hip.enable_peers([0, 1])
    assert fake.cleared == 2  # once for each ordered pair it accepted


def test_a_staged_copy_takes_its_bytes_at_its_ranks_first_launch_only(monkeypatch):
    """A later launch of the rank finds what the earlier ones left in the copy."""
    import thin.emu
    from thin.runtime.emulated import Emulator

    seen = {}

    class Fake:
        def __init__(self, *args, **kw):
            self.texts = []

        def run(self):
            seen["staged"] = self.staged
            return []

    monkeypatch.setattr(thin.emu, "Cluster", Fake)
    emu = Emulator()
    k = emu.alloc(4, 0, {"staged": b"\x01\x02\x03\x04"}, rank=0)
    kernarg = emu.alloc(8, 0, {"tagged": False})
    code = ("", 32, 1, 0, 0)
    launches = [(0, "k", code, (1, 1, 1), kernarg, f"k launch {n}") for n in range(2)]
    emu.run(launches, 100, {})
    assert seen["staged"] == [(0, 0, k, 0, b"\x01\x02\x03\x04")]


def test_gpus_keeps_an_unreadable_device_at_its_index(monkeypatch):
    import ctypes

    from thin.runtime import hip

    class Fake:
        def hipGetDeviceCount(self, p):  # noqa: N802 - HIP's name
            ctypes.cast(p, ctypes.POINTER(ctypes.c_int))[0] = 2
            return 0

        def hipRuntimeGetVersion(self, p):  # noqa: N802 - HIP's name
            ctypes.cast(p, ctypes.POINTER(ctypes.c_int))[0] = 70100000
            return 0

        def hipGetDevicePropertiesR0600(self, buf, i):  # noqa: N802 - HIP's name
            if i == 0:
                return 1  # unreadable
            ctypes.memmove(ctypes.addressof(buf) + hip._ARCH_AT, b"gfx1201:sramecc-\0", 17)
            return 0

    monkeypatch.setattr(hip, "runtime", lambda: "fake")
    monkeypatch.setattr(hip.ctypes, "CDLL", lambda _: Fake())
    assert hip.gpus() == (("unknown", "gfx1201"), 70100000)


def test_a_co_resident_bound_launches_cooperatively_on_the_emulator(no_store, monkeypatch):
    import numpy as np

    import thin
    from examples.vadd import build_vadd
    from thin.runtime import emulated

    seen = []
    run = emulated.EmulatedDevice.run

    def spy(self, code, key, grid, block, kernarg, what, cooperative):
        seen.append(cooperative)
        return run(self, code, key, grid, block, kernarg, what, cooperative)

    monkeypatch.setattr(emulated.EmulatedDevice, "run", spy)
    k = thin.compile(build_vadd)
    x = np.arange(512, dtype=np.float32)
    args = {"x": x, "y": x, "z": np.zeros(512, np.float32), "n": 512}
    k.bind(grid=2, args=args, co_resident=True).emulate()
    k.bind(grid=2, args=args).emulate()
    assert seen == [True, False]


def test_a_copy_behind_a_groups_queued_launch_lands_after_it():
    from thin.runtime import dispatch

    shared, (a, b), (pa, pb) = _queued_group(2)
    src = shared.emu.alloc(4, 7, {})
    dst = shared.emu.alloc(4, 0, {})
    dispatch(pa, (1, 1, 1), b"", {}, "k rank 0")
    a.copy_bytes(dst, 0, src, 0, 4)
    assert shared.emu.read(dst, 0, 4) == b"\0\0\0\0"  # behind the launch
    dispatch(pb, (1, 1, 1), b"", {}, "k rank 1")
    b.wait()
    assert shared.emu.runs == [["k rank 0", "k rank 1"]]
    assert shared.emu.read(dst, 0, 4) == b"\x07" * 4
