"""metron's edges: reserved names, checks, owned keys, records, programs.

The invalidate, the cooperative launch and program() are launch paths on the
GPU, marked hardware."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# Launch paths on the GPU: the gfx1201 hardware suite, each in its own process.
hardware = [pytest.mark.gfx1201, pytest.mark.isolated]


def _cpu_point(names=("x",), n=64, key="p"):
    import metron

    ins = {name: torch.arange(n, dtype=torch.float32) for name in names}
    return metron.Point(key, ins, {"z": torch.empty(n, device="meta")})


def test_rank_and_peers_are_reserved_names():
    for name in ("rank", "peers"):
        with pytest.raises(ValueError, match=rf"reserved \['{name}'\]"):
            _cpu_point((name,))


def test_the_check_is_given_host_inputs():
    """A race's judge hands the check host tensors of the point's inputs."""
    import metron
    from metron import _device

    seen = []
    x = torch.arange(256, dtype=torch.float32)

    def check(i, o):
        seen.append(i["x"])
        return None

    p = metron.Point("p", {"x": x}, {"z": torch.empty(256, device="meta")}, check)
    arm = metron.Arm("a", lambda a: torch.add(a.x, a.x, out=a.z), "a")
    with _device.using(_device.HostBackend(cold=1 << 16)):
        metron.race([arm], p, budget_s=0.1)
    assert seen and seen[0].device.type == "cpu" and torch.equal(seen[0], x)


def test_warm_names_are_inputs_of_every_rank():
    import metron
    from metron import _race

    p = metron.Point(
        "g",
        [{"x": torch.zeros(8), "w": torch.zeros(8)}, {"x": torch.zeros(8)}],
        [{"z": torch.empty(8, device="meta")}] * 2,
    )
    arm = metron.Arm("a", lambda a: None, "f")
    with pytest.raises(ValueError, match=r"warm \['w'\]: of the inputs"):
        _race._refuse([arm], p, metron.State(warm={"w"}))  # rank 1 has no w


def test_the_graph_bound_counts_calls_not_launches():
    from metron import _memory

    # A call of three launches counts as one: three times the budget's launches alive.
    budget = 4096
    m = _memory.most(budget, 1, 1, False)
    n = _memory.calls(2, 0.001, m)
    assert n <= m and 3 * (n + n) > budget


def test_sizing_comes_from_the_first_arm_alone(monkeypatch):
    """An unchecked first arm still sizes every arm's graphs."""
    import metron
    from metron import _device, _race

    sized = []
    real = _race._size

    def spy(lay, arm, most, timed):
        sized.append(arm.name)
        return real(lay, arm, most, timed)

    monkeypatch.setattr(_race, "_size", spy)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, generator=g)

    def check(i, o):
        return None if torch.equal(o["z"], i["x"] * 2) else "wrong"

    p = metron.Point("p", {"x": x}, {"z": torch.empty(256, device="meta")}, check)
    ref = metron.Arm("ref", lambda a: torch.add(a.x, a.x, out=a.z), "r", checked=False)
    other = metron.Arm("b", lambda a: torch.mul(a.x, 2, out=a.z), "b")
    with _device.using(_device.HostBackend(cold=1 << 16)):
        metron.race([ref, other], p, budget_s=0.2)
    assert sized == ["ref"]


def test_hygiene_reads_four_slices():
    from metron import _memory

    x = torch.zeros(65536, dtype=torch.float32)
    x[32868] = 1  # hygiene.json one_off_outside_slices: off every slice
    assert _memory.hygiene({"x": x}) == ["constant:x"]


def test_two_calls_of_one_kernel_own_their_memory_apart():
    import metron.thin as mn

    pytest.importorskip("thin")
    kernel = SimpleNamespace(name="k", args={"w": {"kind": "ptr", "owned": "scratch", "mem": None}})
    one = SimpleNamespace(kernel=kernel, args={"w": 64}, overread={})
    other = SimpleNamespace(kernel=kernel, args={"w": 128}, overread={})
    owned = {}
    mn._plan("arm", 0, [one, other], {}, {}, owned, {})
    assert owned == {"k#0.w": (64, "scratch"), "k#1.w": (128, "scratch")}


@hardware[0]
@hardware[1]
def test_the_invalidate_reach_is_assumed():  # pragma: no cover - hardware
    from metron import _device

    be = _device.GpuBackend()
    be.invalidate(0, None)  # cus * per_cu workgroups; nothing checks every unit was reached
    be.sync(0)


@hardware[0]
@hardware[1]
def test_the_cooperative_block_is_the_sole_parameter():  # pragma: no cover - hardware
    from metron import _device, _kernels

    desc, co = _kernels.kept(_device.GpuBackend().arch, "empty")
    prog = _device.program(co, desc["symbol"], tuple(desc["block"]))
    _device.launch(prog, (1, 1, 1), bytes(desc["kernarg_bytes"]), None, cooperative=True)


def test_version_covers_the_kept_kernels_of_every_target():
    from pathlib import Path

    import metron
    from metron import record

    root = Path(metron.__file__).parent
    names = {p.relative_to(root).as_posix() for p in record._measuring(root)}
    assert {"kernels/gfx1201/differ.co", "kernels/gfx1201/empty.json"} <= names


def test_read_skips_malformed_lines(tmp_path):
    import metron

    path = tmp_path / "r.jsonl"
    path.write_text("\nnot json\n[1]\n" + json.dumps({"schema": "metron/5"}) + "\n")
    assert metron.read(path) == []


def test_the_time_is_utc():
    import re

    import metron
    from metron import record

    r = metron.Race("p", {}, {"a": metron.ArmResult()}, "untimed", 0, {}, [], {}, {}, {}, 0.0)
    stamp = record.lines(r)[0]["time"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", stamp), stamp


@hardware[0]
@hardware[1]
def test_a_program_per_block_over_one_module():  # pragma: no cover - hardware
    from metron import _device, _kernels

    desc, co = _kernels.kept(_device.GpuBackend().arch, "empty")
    a = _device.program(co, desc["symbol"], (64, 1))
    b = _device.program(co, desc["symbol"], (128, 1))
    assert (a.block, b.block) == ((64, 1), (128, 1))
    assert (a.module, a.fn) == (b.module, b.fn)
    assert _device.program(co, desc["symbol"], (64, 1)) is a


_METRON_PATHS = {
    "hipModuleLoadData": lambda d: d.program(b"co", "k", (32, 1)),
    "hipModuleGetFunction": lambda d: d.program(b"co2", "k", (32, 1)),
    "hipModuleLaunchKernel": lambda d: d.launch(
        d.Program(None, None, "k", (32, 1)), (1,), b"\0", 0
    ),
    "hipModuleLaunchCooperativeKernel": lambda d: d.launch(
        d.Program(None, None, "k", (32, 1)), (1,), b"\0", 0, cooperative=True
    ),
    "hipDeviceEnablePeerAccess": lambda d: d.peers(2),
}


@pytest.mark.parametrize("call", sorted(_METRON_PATHS))
def test_every_device_error_clears_hips_last_error_once(monkeypatch, call):
    """Each failure path calls hipGetLastError once."""
    from metron import _device

    class Fake:
        cleared = 0

        def __getattr__(self, name):
            if name == "hipGetLastError":
                return lambda: setattr(Fake, "cleared", Fake.cleared + 1) or 0
            if name == "hipGetErrorString":
                return lambda code: b"stubbed"
            return lambda *a: 1 if name == call else 0

    monkeypatch.setattr(_device, "_LIB", [Fake()])
    monkeypatch.setattr(_device, "_PROGRAMS", {})
    with pytest.raises(RuntimeError):
        _METRON_PATHS[call](_device)
    assert Fake.cleared == 1
