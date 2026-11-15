"""The compiler adapter.

An op becomes a point, an implementation at tunables an arm, and a kernel's
own clock is decoded.

The only module besides the maker that imports the compiler: it reads the
op spec and its implementations, the runtime's bound calls and the kernel library's laps.
"""

from __future__ import annotations

# metron[thin]: without thin installed, importing this module raises ImportError.
import thin
from metron._memory import GAP
from metron._nouns import Arm, Point
from thin.runtime.bound import made_by_buffer

__all__ = ["impl", "impls", "laps", "point"]


def _torch_dtype(t):
    import torch

    return getattr(torch, t.dtype.torch)


def _host_tensor(array, t):
    """A host tensor of the dtype's torch dtype in the storage shape, over the storage bits."""
    import torch

    tt = torch.from_numpy(array.copy())
    want = _torch_dtype(t)
    return tt if tt.dtype == want else tt.view(want)


def _storage_back(tensor, t):
    """A host tensor as the numpy storage array of its spec tensor."""
    import numpy as np

    name, shape = t.storage()
    raw = tensor.contiguous().view(-1).view(_byte_view(tensor.element_size())).numpy()
    return raw.view(np.dtype(name)).reshape(shape)


def _byte_view(size):
    import torch

    return {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size]


def _group_world(op) -> int | None:
    field = op.group_field()
    return None if field is None else op.field_values()[field]


def point(op, seed=0, dist=None) -> Point:
    """The point of an op at (seed, dist)."""
    import torch

    sig = op.signature()
    inout = sorted(n for n, t in sig.items() if t.io == "inout")
    if inout:
        raise NotImplementedError(f"metron: tensors written in place {inout} ({op.key()})")
    dist = dict(dist or {})
    made = op.inputs(seed, **dist)
    world = _group_world(op)
    per_rank = made if world is not None else [made]
    hosts, ins, outs = [], [], []
    for arrays in per_rank:
        host = {n: arrays[n] for n, t in sig.items() if t.io == "in"}
        host.update({n: t.poisoned() for n, t in sig.items() if t.io == "out"})
        hosts.append(host)
        ins.append({n: _host_tensor(host[n], sig[n]) for n, t in sig.items() if t.io == "in"})
        outs.append(
            {
                n: torch.empty(t.storage()[1], dtype=_torch_dtype(t), device="meta")
                for n, t in sig.items()
                if t.io == "out"
            }
        )

    def check(inputs, outputs):
        outputs = outputs if world is not None else [outputs]
        back = [{n: _storage_back(v, sig[n]) for n, v in o.items()} for o in outputs]
        mine = [{n: h[n] for n, t in sig.items() if t.io == "in"} for h in hosts]
        if world is None:
            return op.check(mine[0], back[0])
        return op.check(mine, back)

    key = f"{op.key()} seed {seed}" + (f" dist {dist!r}" if dist else "")
    meta = {"op": op, "hosts": hosts if world is not None else hosts[0]}
    if world is None:
        return Point(key, ins[0], outs[0], check, meta)
    return Point(key, ins, outs, check, meta)


def _label(record, chosen, name, build) -> str:
    if name is not None:
        return name
    label = f"{record.name} {chosen!r}"
    if build:
        label += f" { ({g.__name__: v for g, v in build.items()})!r}"
    return label


def impl(p, fn, *, name=None, build=None, checked=True, read=(), **tunables) -> Arm:
    """An arm of an implementation at tunables, on the point's own tensors."""
    from thin.opspec import group, lowered

    op = p.meta["op"]
    hosts = p.meta["hosts"]
    hosts = list(hosts) if isinstance(hosts, list) else [hosts]
    world = _group_world(op)
    handles = group.emulated(world) if world is not None else [None]
    sig = op.signature()
    for r, h in enumerate(handles):
        for n, t in sig.items():
            if t.shared and h is not None:
                hosts[r] = {**hosts[r], n: h.tensor(f"{op.key()}.{n}", t.dtype.name, t.shape)}
    lowered_ranks = []
    with thin.overriding(build or {}):
        for host, h in zip(hosts, handles, strict=True):
            lowered_ranks.append(lowered(op.with_tensors(host, h), fn, **tunables))
    record, chosen, _ = lowered_ranks[0]
    label = _label(record, chosen, name, build)
    owned, first, plans = {}, {}, []
    for r, (_, _, bounds) in enumerate(lowered_ranks):
        plan = _plan(label, r, bounds, hosts[r], sig, owned, first)
        plans.append(plan)

    def launch(a):
        from metron import _device

        for b, kinds in plans[a.rank]:
            k = b.kernel

            def address(n, kinds=kinds, a=a):
                what, key = kinds[n]
                if what == "owned":
                    v = a.owned[key]
                    return [x.data_ptr() for x in v] if isinstance(v, list) else v.data_ptr()
                if what == "peers":
                    return [x.data_ptr() for x in a.peers(key)]
                return getattr(a, key).data_ptr() if what == "tensor" else a.owned[key].data_ptr()

            prog = _device.program(k.co, k.manifest.symbol, (k.manifest.wgsize, 1))
            _device.launch(prog, b.grid, b.kernarg(address), a.stream, cooperative=b.co_resident)

    rank0 = lowered_ranks[0][2]
    return Arm(
        label,
        launch,
        fingerprint=" ".join(b.kernel.address for b in rank0),
        owned={k: v for k, v in owned.items()},
        checked=checked,
        read=tuple(read),
        meta={"bounds": rank0},
    )


def _plan(label, r, bounds, host, sig, owned, first) -> list:
    """Each bound call with, per pointer argument, what it takes."""
    out, buffers = [], {}
    names = [b.kernel.name for b in bounds]
    for i, b in enumerate(bounds):
        k = b.kernel
        # Each bound call's own memory: a kernel called twice has two (#i).
        call = k.name if names.count(k.name) == 1 else f"{k.name}#{names[:i].count(k.name)}"
        over = {n: v for n, v in (b.overread or {}).items() if v > GAP}
        if over:
            raise ValueError(
                f"metron: {label}: {k.name} overreads {over} bytes, past the gap of {GAP}"
            )
        kinds = {}
        for n, rec in k.args.items():
            if rec["kind"] != "ptr":
                continue
            v = b.args[n]
            if rec.get("owned"):
                key = n if len(bounds) == 1 else f"{call}.{n}"
                kind = rec["owned"] + (f".{rec['mem']}" if rec.get("mem") else "")
                entry = (v, kind)
                if key in first and first[key] != entry:
                    raise ValueError(
                        f"metron: {label}: rank {r} binds {key} to {entry}, rank 0 to {first[key]}"
                    )
                first.setdefault(key, entry)
                owned[key] = entry
                kinds[n] = ("owned", key)
                continue
            name = next((t for t, hv in host.items() if hv is v), None)
            if name is not None:
                kinds[n] = ("peers" if rec.get("ranks") else "tensor", name)
            elif made_by_buffer(v):
                if id(v) not in buffers:
                    buffers[id(v)] = f"buffer{len(buffers)}"
                key = buffers[id(v)]
                owned[key] = (int(v.nbytes), "scratch")
                kinds[n] = ("owned", key)
            else:
                raise ValueError(
                    f"metron: {label}: {k.name}'s argument {n} is no tensor of the op "
                    "(nor a buffer buffer() made)"
                )
        out.append((b, kinds))
    return out


def impls(p) -> list[Arm]:
    """An arm of every covering (implementation, tunables) pair, in declaration order."""
    from thin.opspec import candidates

    op = p.meta["op"]
    return [impl(p, record.fn, **chosen) for record, chosen, _ in candidates(op)]


def laps(arm, r):
    """The decoded laps of the arm's first kernel that takes laps."""
    import numpy as np

    bounds = arm.meta["bounds"]
    names = [b.kernel.name for b in bounds]
    for i, b in enumerate(bounds):
        decl = b.kernel.definition.meta.get("laps") if b.kernel.definition else None
        if decl is None or not getattr(decl, "on", False):
            continue
        n = b.kernel.name
        call = n if names.count(n) == 1 else f"{n}#{names[:i].count(n)}"
        key = "laps" if len(bounds) == 1 else f"{call}.laps"
        read = r.arms[arm.name].read
        if key not in read:
            raise ValueError(f"{arm.name}: {key} not read back - impl(..., read=({key!r},))")
        workgroups = b.grid[0] * b.grid[1] * b.grid[2]
        return decl.decode(np.frombuffer(read[key], np.uint8), workgroups)
    raise ValueError(
        f"{arm.name}: no kernel of it takes laps (build={{generator: {{'laps': True}}}})"
    )
