"""Check cooperative residency bounds and launch refusals on gfx1201."""

from __future__ import annotations

import json
import os

import pytest

from tests.hw._coresident import gen_arrive, patched

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")

SPIN = 4000


def _bound(k):
    cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return cus, cus * k.manifest.resident


def _arrive(k, n, cooperative, spin=SPIN):
    """How many of n workgroups saw all n arrive."""
    from thin.runtime import hip

    counter = hip.tensor(256, torch.cuda.current_device())  # uncached: every load sees the L2
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    k.bind((n,), {"counter": counter, "out": out, "spin": spin}, co_resident=cooperative).run()
    torch.cuda.synchronize()
    assert int(counter.view(torch.int32)[0]) == n
    return int((out == n).sum())


@pytest.mark.parametrize("wgsize", [64, 256, 1024])
def test_the_co_resident_bound_runs_cooperatively(wgsize):
    k = patched(gen_arrive(wgsize))
    _, bound = _bound(k)
    assert _arrive(k, bound, cooperative=True) == bound


@pytest.mark.parametrize("wgsize", [64, 1024])
def test_one_past_the_bound_is_refused_before_launching(wgsize):
    from thin.runtime.errors import LaunchError

    k = patched(gen_arrive(wgsize))
    _, bound = _bound(k)
    with pytest.raises(
        LaunchError, match=rf"co-resident, {bound + 1} workgroups; the GPU holds {bound}"
    ):
        _arrive(k, bound + 1, cooperative=True)


def test_hip_itself_refuses_a_cooperative_launch_past_what_it_holds():
    """Past the bound, straight through HIP: an error, not a hang."""
    from thin.runtime import gpu, hip
    from thin.runtime.errors import LaunchError

    k = patched(gen_arrive(64))
    _, bound = _bound(k)
    fn = gpu._function(k, torch.cuda.current_device())
    counter = hip.tensor(256, torch.cuda.current_device())
    out = torch.zeros(4 * bound, dtype=torch.int32, device="cuda")
    addrs = {"counter": counter.data_ptr(), "out": out.data_ptr()}
    b = k.bind((4 * bound,), {"counter": counter, "out": out, "spin": SPIN})
    kernarg = b.kernarg(addrs.__getitem__)
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(LaunchError, match="cooperative launch"):
        hip.enqueue(fn, (4 * bound, 1, 1), (64, 1), kernarg, stream, "past", cooperative=True)
    # The refusal is reported once: HIP's sticky error is cleared, so the tensor
    # library's next call does not raise it as its own.
    torch.cuda.synchronize()
    assert int(torch.ones(4, device="cuda").sum()) == 4


def test_how_many_workgroups_the_gpu_really_holds(tmp_path):
    """whether exactly the bound's workgroups are resident at once, launched
    plainly (a bisection over wgsize 32 to 1024 found the bound exact: 2048 waves)."""
    report = {}
    for wgsize in (64, 256, 1024):
        k = patched(gen_arrive(wgsize))
        cus, bound = _bound(k)
        # The bound holds and one more does not: what the GPU holds is the bound exactly.
        held = {n: _arrive(k, n, cooperative=False, spin=SPIN) == n for n in (bound, bound + 1)}
        held = bound if held[bound] and not held[bound + 1] else (0 if not held[bound] else -1)
        report[wgsize] = {"cus": cus, "resident": k.manifest.resident, "bound": bound, "held": held}
    path = os.environ.get("THIN_HW_REPORT_DIR", str(tmp_path))
    with open(os.path.join(path, "coresident.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    for r in report.values():
        assert r["held"] == r["bound"], r  # measured: exact (2048 waves) at every wgsize


def _last_error():
    import ctypes

    from thin.runtime import hip

    fn = hip.lib().hipGetLastError
    fn.restype, fn.argtypes = ctypes.c_int, []
    return fn()


def test_a_refused_cooperative_launch_leaves_no_last_error():
    """after the LaunchError, hipGetLastError is success and torch runs."""
    from thin.runtime import gpu, hip
    from thin.runtime.errors import LaunchError

    k = patched(gen_arrive(64))
    _, bound = _bound(k)
    fn = gpu._function(k, torch.cuda.current_device())
    counter = hip.tensor(256, torch.cuda.current_device())
    out = torch.zeros(4 * bound, dtype=torch.int32, device="cuda")
    b = k.bind((4 * bound,), {"counter": counter, "out": out, "spin": SPIN})
    kernarg = b.kernarg({"counter": counter.data_ptr(), "out": out.data_ptr()}.__getitem__)
    with pytest.raises(LaunchError, match=r"\(720\)"):
        hip.enqueue(fn, (4 * bound, 1, 1), (64, 1), kernarg, 0, "past", cooperative=True)
    assert _last_error() == 0
    torch.cuda.synchronize()
    assert int(torch.ones(4, device="cuda").sum()) == 4


def test_another_targets_code_object_refused_leaves_no_last_error():
    """a gfx1201 code object loaded on another GPU (this host's integrated
    gfx1036) is refused, and leaves no last error."""
    from thin.runtime import hip
    from thin.runtime.errors import LaunchError

    archs, _ = hip.gpus()
    other = next((i for i, a in enumerate(archs) if a != "gfx1201"), None)
    if other is None:
        pytest.skip("no GPU of another target here")
    k = patched(gen_arrive(64))
    before = torch.cuda.current_device()
    torch.cuda.set_device(other)
    try:
        with pytest.raises(LaunchError, match="load"):
            hip.module(k.co, k.name, k.name)
        assert _last_error() == 0
        torch.cuda.synchronize()
    finally:
        torch.cuda.set_device(before)
    assert int(torch.ones(4, device="cuda").sum()) == 4
