"""Running a Bound on the GPU through torch and HIP.

Covered by the gfx1201 hardware tests: the tests that run it are marked
for gfx1201 hardware.
"""

from __future__ import annotations

from thin.runtime.errors import LaunchError

__all__ = ["run", "time_bound"]

_FUNCTIONS: dict = {}
"""(kernel address, GPU) → (module, function); loaded once, never unloaded."""


def _function(kernel, device: int):
    from thin.runtime import hip

    key = (kernel.address, device)
    found = _FUNCTIONS.get(key)
    if found is None:
        found = hip.module(kernel.co, kernel.name, kernel.name)
        _FUNCTIONS[key] = found
    return found[1]


def _first(kind, given, size, device, uncached):
    import torch

    from thin.runtime import hip

    if uncached:
        t = hip.tensor(size, device)
    else:
        t = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{device}")
    if kind == "gens":
        t.view(torch.int32).fill_(-3)
    elif isinstance(given, (bytes, bytearray)):
        t[: len(given)].copy_(torch.frombuffer(bytearray(given), dtype=torch.uint8))
    return t


def _held(b, rec, device, capturing):
    from thin.runtime.bound import PROCESS_HELD

    k, name, owned, given = b.kernel.name, rec["name"], rec["owned"], b.args[rec["name"]]
    size = len(given) if isinstance(given, (bytes, bytearray)) else given
    held = b.group.team.held if b.group is not None else PROCESS_HELD
    where = b.group.rank if b.group is not None else device
    key = (b.kernel.address, name, where) + (
        (bytes(given),) if isinstance(given, (bytes, bytearray)) else ()
    )
    why = ((b.kernel.manifest.needs or {}).get("same_grid") or {}).get(name)
    if why is not None:
        held.same_grid(k, (*key, "grid"), name, b.grid, why)
    found = held.buffers.get(key)
    if capturing and not (found and len(found[-1]) >= size):
        raise LaunchError(
            f"{k}: its {owned} memory {name} first allocated inside a graph capture: call the "
            "kernel once before "
            "the capture"
        )
    uncached = rec.get("mem") == "uncached"
    return held.lookup(key, size, lambda n: _first(owned, given, n, device, uncached))


def _addresses(b, device, stream, capturing):
    """{name: address or every rank's}, the padded copies, the owned tensors."""
    import torch

    k, addrs, padded, owned = b.kernel.name, {}, [], {}
    for name, rec in b.kernel.args.items():
        if rec["kind"] != "ptr":
            continue
        v = b.args[name]
        if rec["owned"] == "scratch":
            owned[name] = torch.empty(v, dtype=torch.uint8, device=f"cuda:{device}")
            addrs[name] = owned[name].data_ptr()
        elif rec["owned"] == "mapped":
            if b.group is None:
                raise LaunchError(
                    f"{k}: argument {name} is a group's mapped memory, but the call is no group "
                    "op's (a group op's "
                    "call has its rank's handle as its group)"
                )
            if capturing and not b.group.team.holds(b.kernel, name, v):
                raise LaunchError(
                    f"{k}: its mapped memory {name} first allocated inside a graph capture: call "
                    "the kernel once "
                    "before the capture"
                )
            bufs = b.group.team.mapped(b.kernel, name, v)
            addrs[name] = [x if isinstance(x, int) else x.data_ptr() for x in bufs]
        elif rec["owned"]:
            owned[name] = _held(b, rec, device, capturing)
            addrs[name] = owned[name].data_ptr()
        elif rec.get("ranks") is not None:
            peers = b.group.team.peers_of(v) if b.group is not None else None
            if peers is None:
                raise LaunchError(
                    f"{k}: argument {name} is every rank's copy of a group tensor (peers): bind "
                    "it to the op's - "
                    f"the group's (g.tensor) - not {type(v).__name__}"
                )
            addrs[name] = [x if isinstance(x, int) else x.data_ptr() for x in peers]
        else:
            addrs[name] = _plain(b, name, v, padded)
    return addrs, padded, owned


def _plain(b, name, v, padded) -> int:
    import torch

    over = b.overread.get(name, 0)
    if not over:
        return v.data_ptr()
    storage = v.untyped_storage().nbytes() - v.storage_offset() * v.element_size()
    if storage - v.nbytes >= over:
        return v.data_ptr()
    flat = v.reshape(-1).view(torch.uint8)
    copy = torch.empty(v.nbytes + over, dtype=torch.uint8, device=v.device)
    copy[: v.nbytes].copy_(flat)
    copy[v.nbytes :].fill_(0xFF)
    padded.append((v, copy))
    return copy.data_ptr()


def run(b) -> None:
    """Launches a Bound on the current torch stream."""
    import torch

    from thin.runtime import hip

    k = b.kernel.name
    for name, rec in b.kernel.args.items():
        v = b.args.get(name)
        plain = rec["kind"] == "ptr" and not rec["owned"] and rec.get("ranks") is None
        on_gpu = type(v).__module__.split(".")[0] == "torch" and getattr(v, "is_cuda", False)
        if plain and not on_gpu:
            raise TypeError(f"{k}: argument {name}: a torch tensor on the GPU to run")
    device = torch.cuda.current_device()
    if b.group is not None and b.group.device != device:
        raise LaunchError(
            f"{k}: rank {b.group.rank}'s call runs on its GPU {b.group.device}, the current one "
            f"is {device}"
        )
    stream = torch.cuda.current_stream()
    capturing = torch.cuda.is_current_stream_capturing()
    addrs, padded, owned = _addresses(b, device, stream, capturing)
    b.owned = owned
    if b.co_resident:
        workgroups = b.grid[0] * b.grid[1] * b.grid[2]
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        resident = b.kernel.manifest.resident
        if workgroups > cus * resident:
            raise LaunchError(
                f"{k}: co-resident, {workgroups} workgroups; the GPU holds {cus * resident} "
                f"({resident} a CU)"
            )
    fn = _function(b.kernel, device)
    kernarg = b.kernarg(lambda n: addrs[n])
    block = (b.kernel.manifest.wgsize, 1)
    hip.enqueue(fn, b.grid, block, kernarg, stream.cuda_stream, k, cooperative=b.co_resident)
    for v, copy in padded:
        v.reshape(-1).view(torch.uint8).copy_(copy[: v.nbytes])


def time_bound(b, iters: int = 20) -> float:
    """One run, then the mean microseconds of ``iters`` runs between two events."""
    import torch

    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise LaunchError(f"{b.kernel.name}: time(iters={iters!r}): a positive int of runs")
    run(b)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run(b)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters
