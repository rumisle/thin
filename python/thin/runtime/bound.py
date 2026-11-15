"""Binding and running a call.

Bound and its kernarg, accepts, own memory, emulate and the joint run, and
buffers; run and time on the GPU are in ``gpu``.
"""

from __future__ import annotations

import operator
import weakref
from dataclasses import dataclass, field

from thin.runtime.args import check, pack
from thin.runtime.errors import DeviceFault, LaunchError

__all__ = [
    "Bound",
    "HeldTable",
    "accepts",
    "bind",
    "buffer",
    "is_buffer",
    "joint_run",
    "made_by_buffer",
]


# --- Buffers -----------------------------------------------------------------


def _is_tensor(v) -> bool:
    """A torch tensor (torch is not imported to ask)."""
    return any(c.__name__ == "Tensor" and c.__module__.startswith("torch") for c in type(v).__mro__)


def is_buffer(v) -> bool:
    """A numpy array or a torch tensor."""
    if _is_tensor(v):
        return True
    module = type(v).__module__ or ""
    return module.split(".")[0] == "numpy" and type(v).__name__ == "ndarray"


def _contiguous(k, args: dict, recs: dict) -> None:
    """Refuses a buffer argument whose elements are not contiguous.

    The kernel addresses it as one run of bytes, and its result is written
    back as one.
    """
    for n, v in args.items():
        r = recs.get(n)
        if r is None or r["kind"] != "ptr" or r.get("owned") or not is_buffer(v):
            continue
        ok = v.is_contiguous() if _is_tensor(v) else v.flags["C_CONTIGUOUS"]
        if not ok:
            raise LaunchError(f"{k}: argument {n} is not contiguous (pass a contiguous copy)")


def host_bytes(v) -> bytes:
    """A buffer's bytes on the host."""
    if type(v).__module__.split(".")[0] == "torch":
        return v.detach().contiguous().cpu().view(-1).numpy().view("u1").tobytes()
    import numpy as np

    return np.ascontiguousarray(v).view(np.uint8).tobytes()


def write_back(v, data: bytes) -> None:
    """Writes bytes into a buffer: numpy in place, torch copied in."""
    if type(v).__module__.split(".")[0] == "torch":
        import numpy as np
        import torch

        src = torch.from_numpy(np.frombuffer(bytearray(data), dtype=np.uint8))
        v.view(-1).view(torch.uint8).copy_(src.to(v.device))
        return
    v.reshape(-1).view("u1")[:] = memoryview(data).cast("B")


_MADE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def buffer(nbytes, like):
    """Memory a binding passes from one kernel to the next."""
    if type(nbytes) is not int or nbytes < 1:
        raise ValueError(f"buffer({nbytes!r}): bytes, an int of 1 or more")
    if not is_buffer(like):
        raise TypeError(f"buffer(like={type(like).__name__}): an op's tensor, numpy or torch")
    if type(like).__module__.split(".")[0] == "torch":
        import torch

        made = torch.empty(nbytes, dtype=torch.uint8, device=like.device)
    else:
        import numpy as np

        made = np.full(nbytes, 0xFF, dtype=np.uint8)
    _MADE[id(made)] = made
    return made


def made_by_buffer(v) -> bool:
    """Whether ``buffer`` made v (identity)."""
    return _MADE.get(id(v)) is v


# --- Bind ------------------------------------------------------------------------


@dataclass(eq=False)
class Bound:
    """One call, not yet made."""

    kernel: object
    grid: tuple
    args: dict
    co_resident: bool = False
    overread: dict = field(default_factory=dict)
    group: object = None
    """The rank's handle for a group op's call, else None."""

    def __post_init__(self):
        self.owned = {}
        """The last call's own memory."""

    def kernarg(self, address) -> bytes:
        """The packed kernarg given ``address(name)`` for every pointer argument."""
        rank = getattr(self.group, "rank", 0) if self.group is not None else 0
        recs = list(self.kernel.args.values())

        def addr(name):
            a = address(name)
            rec = self.kernel.args[name]
            if rec.get("ranks") is not None and isinstance(a, (list, tuple)):
                return [*a, a[rank]]
            return a

        return pack(recs, self.kernel.manifest.kernarg_bytes, self.args, addr)

    def emulate(self, sched=None):
        """Runs on the emulated device; returns the caller's arrays by name."""
        return emulate(self, sched)

    def run(self):
        """Launches on the GPU on the current torch stream."""
        from thin.runtime.gpu import run

        return run(self)

    def time(self, iters=20) -> float:
        """Mean microseconds of ``iters`` warm runs."""
        from thin.runtime.gpu import time_bound

        return time_bound(self, iters)


def _whole(m):
    """An int (numpy's too) as an int; None for a bool or anything else."""
    if isinstance(m, bool):
        return None
    try:
        return operator.index(m)
    except TypeError:
        return None


def _grid(k, grid, max_grid) -> tuple:
    """The grid as three positive ints: an int or one to three ints."""
    members = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
    g = tuple(_whole(m) for m in members)
    if not (1 <= len(g) <= 3 and all(m is not None and m >= 1 for m in g)):
        raise LaunchError(f"{k}: grid {grid!r}: one to three positive ints")
    g = g + (1,) * (3 - len(g))
    if max_grid:
        # A dimension max_grid does not name is bounded by 1.
        bound = tuple(max_grid) + (1,) * (3 - len(max_grid))
        if any(m > b for m, b in zip(g, bound, strict=False)):
            raise LaunchError(
                f"{k}: grid {g} past the kernel's max_grid {tuple(max_grid)} "
                "(its ranges are proven for that)"
            )
    return g


def _overread(k, recs: dict, overread) -> dict:
    """The overread given: each a pointer argument's bytes past its end, a positive int."""
    overread = dict(overread or {})
    bad = {
        n: v
        for n, v in overread.items()
        if n not in recs or recs[n]["kind"] != "ptr" or type(v) is not int or v < 1
    }
    if bad:
        raise LaunchError(
            f"{k}: overread {bad}: a pointer argument's bytes past its end, a positive int"
        )
    # The kernel's own memory is sized by the kernel: nothing to read past.
    own = sorted(n for n in overread if recs[n].get("owned"))
    if own:
        raise LaunchError(f"{k}: overread of {', '.join(own)}: the kernel's own memory")
    return overread


def bind(kernel, grid=None, args=None, *, co_resident=False, overread=None) -> Bound:
    """Validate the launch contract and construct the Bound."""
    k = kernel.name
    args = dict(args or {})
    recs = kernel.args
    check(k, list(recs.values()), args, is_buffer)
    _contiguous(k, args, recs)
    overread = _overread(k, recs, overread)
    why = (kernel.manifest.needs or {}).get("co_resident")
    if why is not None and not co_resident:
        raise LaunchError(
            f"{k}: needs co_resident ({why}): bind(..., co_resident=True) - a cooperative launch"
        )
    if grid is None:
        grid = kernel.manifest.grid
        if not grid:
            raise TypeError(f"{k}: no grid given and none in the manifest")
    return Bound(
        kernel, _grid(k, grid, kernel.manifest.max_grid), args, bool(co_resident), overread
    )


# --- accepts ---------------------------------------------------------------------


class _StandIn:
    """A pointer argument not given: any read of it is refused."""

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def _refuse(self, *a, **kw):
        raise TypeError(
            f"accepts: the host reads pointer argument {object.__getattribute__(self, '_name')}, "
            "not given - "
            "give it (a host that reads a buffer is asked with it)"
        )

    __len__ = __getitem__ = __int__ = __bool__ = __index__ = __iter__ = _refuse

    def __getattr__(self, name):
        self._refuse()


def _records(d) -> list[dict]:
    """The definition's argument records."""
    return [{"name": n, **dict(d.args[n])} for n in d.args.names()]


def accepts(d, given):
    """Runs the host with a checking bind; returns None or the refusal's message."""
    import inspect

    if d.host is None:
        raise TypeError(f"{d.symbol}: no host function (KernelDef(host=)) to ask")
    recs = _records(d)
    params = inspect.signature(d.host).parameters
    given = dict(given)
    for r in recs:
        if (
            r["kind"] == "ptr"
            and not r.get("owned")
            and r["name"] in params
            and r["name"] not in given
        ):
            given[r["name"]] = _StandIn(r["name"])

    def checking(grid=None, args=None, *, co_resident=False, overread=None):
        check(d.symbol, recs, dict(args or {}), lambda v: is_buffer(v) or isinstance(v, _StandIn))
        _overread(d.symbol, {r["name"]: r for r in recs}, overread)
        if grid is not None:
            _grid(d.symbol, grid, d.max_grid)

    try:
        d.host(checking, **given)
    except LaunchError as e:
        return str(e)
    return None


# --- Own memory and the joint run -----------------------------------------------


class HeldTable:
    """Held memory (kept, carried, gens): per key its buffers, the newest last."""

    def __init__(self):
        self.buffers: dict = {}
        self.grids: dict = {}

    def lookup(self, key, size: int, first):
        """The newest buffer of at least ``size``, else a new one from ``first(size)``."""
        found = self.buffers.setdefault(key, [])
        if found and len(found[-1]) >= size:
            return found[-1]
        made = first(size)
        found.append(made)
        return made

    def same_grid(self, k, key, name, grid, why) -> None:
        """Records the first grid of a same_grid argument's memory; refuses another."""
        first = self.grids.setdefault(key, grid)
        if first != grid:
            raise LaunchError(
                f"{k}: grid {grid} on its held memory {name}, first bound with {first}: {why} - "
                "every call on "
                "it binds the same grid"
            )


PROCESS_HELD = HeldTable()
"""The process's held table."""


def _first(kind: str, given):
    def make(size: int) -> bytearray:
        if kind == "gens":
            return bytearray(((1 << 32) - 3).to_bytes(4, "little") * (size // 4))
        out = bytearray(size)
        if isinstance(given, (bytes, bytearray)):
            out[: len(given)] = given
        return out

    return make


def emulate(b: Bound, sched=None) -> dict:
    """One Bound on an emulated device, rank 0, the process's held table."""
    from thin.runtime.emulated import EmulatedDevice, group

    k = b.kernel.name
    if b.group is not None:
        raise LaunchError(
            f"{k}: a group's call runs with its ranks' - thin.group.emulate(bounds) or thin.check"
        )
    workgroups = b.grid[0] * b.grid[1] * b.grid[2]
    if b.co_resident and sched and "resident" in sched and workgroups > sched["resident"]:
        raise LaunchError(
            f"{k}: co-resident, {workgroups} workgroups; sched resident {sched['resident']}"
        )
    device = group(1, **sched)[0] if sched else EmulatedDevice()
    try:
        joint_run([b], [device], PROCESS_HELD, lambda _: "emulated", ())
    finally:
        if not device.closed:
            device.close()
    return {
        n: v
        for n, v in b.args.items()
        if b.kernel.args[n]["kind"] == "ptr" and not b.kernel.args[n]["owned"]
    }


def _rank(b) -> int:
    return 0 if b.group is None else b.group.rank


def _copy_index(peers, v) -> int:
    for q, c in enumerate(peers):
        if c is v:
            return q
    return next(
        q for q, c in enumerate(peers) if hasattr(c, "data_ptr") and c.data_ptr() == v.data_ptr()
    )


class _Joint:
    """One joint run's buffers: made once each, before any launch."""

    def __init__(self, bounds, devices, held, where, staged):
        self.device_of: dict = {}
        for b, d in zip(bounds, devices, strict=True):
            self.device_of.setdefault(_rank(b), d)
        self.held, self.where = held, where
        self.staged = {id(c) for c in staged}
        self.made: dict = {}
        self.copies: dict = {}
        self.mapped: dict = {}

    def copy(self, copy, rank):
        """The device buffer of a group tensor's copy, on its rank's device, ``ipc``."""
        found = self.copies.get(id(copy))
        if found is None:
            data = host_bytes(copy)
            meta = {"ipc": True}
            if id(copy) in self.staged:
                meta["staged"] = data  # 0xFF until its rank's launches start
                data = bytes([0xFF]) * len(data)
            found = (copy, rank, self.device_of[rank].tensor("u8", data, meta=meta))
            self.copies[id(copy)] = found
        return found[2]

    def mapped_buffers(self, b, name, size) -> list:
        """Every rank's device buffer of a mapped argument, one a rank per run."""
        key = (b.kernel.address, name)
        found = self.mapped.get(key)
        if found is None:
            host = b.group.team.mapped(b.kernel, name, size)
            device = [
                self.device_of[q].tensor("u8", bytes(h), meta={"ipc": True})
                for q, h in enumerate(host)
            ]
            found = (host, device)
            self.mapped[key] = found
        return found[1]

    def own(self, b, device, rec):
        """(device buffer, address or addresses, held key and host buffer or None)."""
        k, name, owned = b.kernel.name, rec["name"], rec["owned"]
        given = b.args[name]
        size = len(given) if isinstance(given, (bytes, bytearray)) else given
        if owned == "scratch":
            t = device.tensor("u8", nbytes=size, fill=0xFF)
            return t, t.ptr, None
        if owned == "mapped":
            if b.group is None:
                raise LaunchError(
                    f"{k}: argument {name} is a group's mapped memory, but the call is no group "
                    "op's (a group op's call has its rank's handle as its group)"
                )
            bufs = self.mapped_buffers(b, name, size)
            return bufs[_rank(b)], [t.ptr for t in bufs], None
        key = (b.kernel.address, name, self.where(b))
        key += (bytes(given),) if isinstance(given, (bytes, bytearray)) else ()
        why = ((b.kernel.manifest.needs or {}).get("same_grid") or {}).get(name)
        if why is not None:
            self.held.same_grid(k, (*key, "grid"), name, b.grid, why)
        host = self.held.lookup(key, size, _first(owned, given))
        if id(host) not in self.made:
            meta = {"uncached": True} if rec.get("mem") == "uncached" else None
            self.made[id(host)] = device.tensor("u8", bytes(host), meta=meta)
        t = self.made[id(host)]
        return t, t.ptr, (key, host)

    def plain(self, b, device, name, rec):
        """(device buffer or None, address or addresses, the caller's value or None)."""
        v = b.args[name]
        team = getattr(b.group, "team", None)
        peers = team.peers_of(v) if team is not None else None
        if rec.get("ranks") is not None:
            if peers is None:
                raise LaunchError(
                    f"{b.kernel.name}: argument {name} is every rank's copy of a group tensor "
                    f"(peers): bind it to the op's - the group's (g.tensor) - not "
                    f"{type(v).__name__}"
                )
            bufs = [self.copy(c, q) for q, c in enumerate(peers)]
            return bufs[_rank(b)], [t.ptr for t in bufs], None
        if peers is not None:
            q = _copy_index(peers, v)
            t = self.copy(peers[q], q)
            return (t if q == _rank(b) else None), t.ptr, None
        over = b.overread.get(name, 0)
        t = device.tensor("u8", host_bytes(v) + bytes([0xFF]) * over)
        return t, t.ptr, v


def joint_run(bounds, devices, held, where, staged) -> None:
    """Runs Bounds together, each on its rank's device.

    Every buffer is made before any launch; each Bound is dispatched with
    the device addresses (every rank's for a ranked argument) and the
    buffers of its own device; every device waits; then the caller's
    arrays, ``owned``, mapped memory, group-tensor copies and held memory
    take what the run left.
    """
    from thin.runtime.device import dispatch, program

    joint = _Joint(bounds, devices, held, where, staged)
    plans = []
    for b, device in zip(bounds, devices, strict=True):
        prog = program(device, b.kernel.co, b.kernel.asm, b.kernel.name)
        bufs, addrs, holds, plain = {}, {}, {}, {}
        for name, rec in b.kernel.args.items():
            if rec["kind"] != "ptr":
                continue
            if rec["owned"]:
                t, addrs[name], hold = joint.own(b, device, rec)
                if hold:
                    holds[name] = (rec, hold)
            else:
                t, addrs[name], v = joint.plain(b, device, name, rec)
                if v is not None:
                    plain[name] = v
            if t is not None:
                bufs[name] = t
        plans.append((b, device, prog, bufs, addrs, holds, plain))
    for b, _, prog, bufs, addrs, _, _ in plans:
        kernarg = b.kernarg(lambda n, addrs=addrs: addrs[n])
        dispatch(prog, b.grid, kernarg, bufs, cooperative=b.co_resident)
    for d in dict.fromkeys(devices):
        d.wait()
    # Kept memory is checked first: a fault leaves every buffer as it was.
    for b, device, _, bufs, _, holds, _ in plans:
        _check_kept(b, device, bufs, holds)
    for b, device, _, bufs, _, holds, plain in plans:
        for name, v in plain.items():
            write_back(v, device.read(bufs[name])[: len(host_bytes(v))])
        _settle(b, device, bufs, holds)
    for host, device_bufs in joint.mapped.values():
        for h, t in zip(host, device_bufs, strict=True):
            h[:] = t._device.read(t)[: len(h)]
    for copy, _, t in joint.copies.values():
        write_back(copy, t._device.read(t)[: len(host_bytes(copy))])


def _check_kept(b, device, bufs, holds) -> None:
    """A DeviceFault if the launch changed a kept buffer."""
    for name, (rec, (_, host)) in sorted(holds.items(), key=lambda x: x[1][1][0]):
        if rec["owned"] != "kept":
            continue
        after = device.read(bufs[name])[: len(host)]
        changed = sum(x != y for x, y in zip(after, host, strict=True))
        if changed:
            raise DeviceFault(
                f"{b.kernel.name}: its kept memory {name}: {changed} of {len(host)} bytes not "
                "left as the launch found them (every launch leaves kept memory as it found "
                "it)",
                f"kept:{name}",
            )


def _settle(b, device, bufs, holds) -> None:
    """``b.owned`` and the carried and generation buffers' new bytes.

    Mapped memory is the group's, not the kernel's own: left out, as on the GPU.
    """
    import numpy as np

    b.owned = {
        n: np.frombuffer(device.read(t), dtype=np.uint8).copy()
        for n, t in bufs.items()
        if b.kernel.args[n]["owned"] not in (None, False, "mapped")
    }
    for name, (rec, (_, host)) in holds.items():
        if rec["owned"] != "kept":
            host[:] = device.read(bufs[name])[: len(host)]
