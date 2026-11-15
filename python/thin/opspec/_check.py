"""check: running an op at seeds, dist points and schedules."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from thin.opspec._impl import lowered

__all__ = ["SCHEDS", "Failure", "check"]

SCHEDS = (
    None,
    {"order": "reverse"},
    {"order": 1},
    {"resident": 1},
    {"order": "reverse", "resident": 1},
    {"jitter": 1},
    {"wave_jitter": 1},
)
"""The default schedules."""

_PKG = "thin"


@dataclass(frozen=True, eq=False)
class Failure:
    """One failed run: its scenario and why."""

    op: object
    """check's op."""
    impl: str
    """The implementation's name."""
    tunables: dict = field(default_factory=dict)
    """The pair's tunables."""
    seed: int = 0
    """The seed."""
    dist: dict = field(default_factory=dict)
    """The dist point."""
    sched: object = None
    """The schedule, or None."""
    device: str = "emulated"
    """The device check ran on."""
    why: str = ""
    """What the check said, or the fault."""

    def rerun(self) -> str:
        """The call that runs this scenario alone."""
        parts = [f"impl={self.impl.rsplit('/', 1)[-1]!r}"]
        parts += [f"{n}={v!r}" for n, v in self.tunables.items()]
        dists = {n: [v] for n, v in self.dist.items()}
        parts += [
            f"seeds=({self.seed!r},)",
            f"dists={dists!r}",
            f"scheds=({self.sched!r},)",
            f"device={self.device!r}",
        ]
        return f"{_PKG}.check(op, {', '.join(parts)})"

    def __str__(self) -> str:
        return f"{self.op.key()} {self.impl} {self.tunables!r}: {self.why}\n  rerun: {self.rerun()}"


def _points(spec, dists) -> list[dict]:
    table = dict(spec.dists if dists is None else dists)
    if dists is not None:
        unknown = sorted(n for n in dists if n not in spec.dists)
        if unknown:
            raise ValueError(
                f"{spec.name()}: dists {unknown}: its data parameters are {sorted(spec.dists)}"
            )
    empty = sorted(n for n, vs in table.items() if not list(vs))
    if empty:
        raise ValueError(f"{spec.name()}: dists {empty} list no values")
    names = list(table)
    points = [dict(zip(names, vs, strict=True)) for vs in itertools.product(*table.values())]
    return points or [{}]


def _skipped(sched, bounds) -> bool:
    """A residency below a co-resident Bound's workgroups skips the run."""
    resident = (sched or {}).get("resident")
    return bool(resident) and any(
        b.co_resident and resident < b.grid[0] * b.grid[1] * b.grid[2] for b in bounds
    )


# The GPU's copies and synchronisation, apart so a replay may stand in for them.


def to_device(array, tensor):
    """A copy of a numpy array on the current GPU as a torch tensor of the dtype (same bits)."""
    import torch

    t = torch.from_numpy(array.copy()).cuda()
    return t.view(getattr(torch, tensor.dtype.torch))


def to_host(value, tensor):
    """A numpy array of the storage from a tensor on the GPU or the host."""
    if (type(value).__module__ or "").split(".")[0] != "torch":
        return value
    import torch

    numpy, _ = tensor.storage()
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[value.element_size()]
    return value.detach().cpu().view(bits).numpy().view(numpy)


def wait(devices=None) -> None:
    """Waits for the GPUs (all of them, or the given ones)."""
    import torch

    for d in devices if devices is not None else [None]:
        torch.cuda.synchronize(d)


def _tensors(op, inputs, gpu: bool) -> dict:
    out = {}
    for n, t in op.signature().items():
        array = inputs[n].copy() if n in inputs else t.poisoned()
        out[n] = to_device(array, t) if gpu else array
    return out


def _why(op, inputs, outputs, run):
    from thin.runtime.errors import DeviceFault

    try:
        run()
        return op.check(inputs, outputs())
    except DeviceFault as e:
        return f"fault: {e}"


def check(op, seeds=(0, 1, 2), device="emulated", scheds=SCHEDS, dists=None, **lower_args):
    """Runs op at every (seed, dist point, schedule) and returns the Failures."""
    if device not in ("emulated", "gpu"):
        raise ValueError(f"check(device={device!r}): 'emulated' or 'gpu'")
    spec = type(op)
    base = spec(**op.field_values())
    points = _points(spec, dists)
    if spec.group_field() is not None:
        return _group_check(base, seeds, device, scheds, points, lower_args)
    gpu = device != "emulated"
    runs = [None] if gpu else list(scheds)
    failures = []
    for seed, point, sched in itertools.product(seeds, points, runs):
        inputs = base.inputs(seed, **point)
        tensors = _tensors(base, inputs, gpu)
        impl, tunables, bounds = lowered(base.with_tensors(tensors), **lower_args)
        if not gpu and _skipped(sched, bounds):
            continue
        sig = base.signature()

        def run(bounds=bounds, sched=sched):
            for b in bounds:
                if gpu:
                    b.run()
                else:
                    b.emulate(sched)
            if gpu:
                wait()

        def outputs(tensors=tensors, sig=sig):
            return {n: to_host(tensors[n], t) for n, t in sig.items() if t.io != "in"}

        why = _why(base, inputs, outputs, run)
        if why is not None:
            failures.append(Failure(base, impl.name, tunables, seed, point, sched, device, why))
    return failures


def _group_tensors(op, handle, rank_inputs, gpu: bool) -> tuple[dict, list]:
    """A rank's tensors and its staged copies."""
    tensors, staged = {}, []
    for n, t in op.signature().items():
        values = rank_inputs[n] if n in rank_inputs else t.poisoned()
        if t.shared:
            copy = handle.tensor(f"{op.key()}.{n}", t.dtype, t.shape)
            _fill(copy, values)
            tensors[n] = copy
            if t.io == "in":
                staged.append(copy)
        elif gpu:
            import torch

            with torch.cuda.device(handle.device):
                tensors[n] = to_device(values.copy(), t)
        else:
            tensors[n] = values.copy()
    return tensors, staged


def _fill(copy, values) -> None:
    if (type(copy).__module__ or "").split(".")[0] == "torch":
        import torch

        src = torch.from_numpy(values.copy()).to(copy.device)
        copy.view(-1).view(torch.uint8).copy_(src.view(-1).view(torch.uint8))
    else:
        copy[...] = values


def _group_check(op, seeds, device, scheds, points, lower_args) -> list:
    from thin.opspec import group as groups

    gpu = device != "emulated"
    world = op.field_values()[op.group_field()]
    runs = [None] if gpu else list(scheds)
    if not gpu and tuple(scheds) == tuple(SCHEDS):  # the default schedules
        runs += [groups.slow(world), groups.slow(world, resident=1)]
    handles = groups.local(world) if gpu else groups.emulated(world)
    failures = []
    try:
        for seed, point, sched in itertools.product(seeds, points, runs):
            why, impl, tunables = _group_run(op, handles, seed, point, sched, gpu, lower_args)
            if why is not None:
                failures.append(Failure(op, impl.name, tunables, seed, point, sched, device, why))
    finally:
        handles[0].close()
    return failures


def _group_run(op, handles, seed, point, sched, gpu, lower_args):
    from thin.opspec import group as groups

    world = len(handles)
    inputs = op.inputs(seed, **point)
    if len(inputs) != world:
        raise ValueError(
            f"{op.name()}: inputs gives {len(inputs)} ranks' inputs, its world is {world}"
        )
    ranks, staged = [], []
    for r, h in enumerate(handles):
        tensors, mine = _group_tensors(op, h, inputs[r], gpu)
        ranks.append(tensors)
        staged += mine
    lowerings = [
        lowered(op.with_tensors(t, h), **lower_args) for t, h in zip(ranks, handles, strict=True)
    ]
    counts = [len(x[2]) for x in lowerings]
    if len(set(counts)) > 1:
        raise ValueError(f"{op.key()}: the ranks lower to {counts} calls: one count")
    chosen = [(x[0].name, x[1]) for x in lowerings]
    if any(c != chosen[0] for c in chosen[1:]):
        raise ValueError(f"{op.key()}: the ranks lower to {chosen}: one implementation")
    impl, tunables, _ = lowerings[0]
    every = [b for x in lowerings for b in x[2]]
    if gpu or not _skipped(sched, every):
        sig = op.signature()

        def run():
            for k in range(counts[0]):
                calls = [x[2][k] for x in lowerings]
                if gpu:
                    _gpu_round(calls)
                else:
                    groups.emulate(calls + calls, sched, staged)
            if gpu:
                wait(list(range(world)))

        def outputs():
            return [{n: to_host(t[n], s) for n, s in sig.items() if s.io != "in"} for t in ranks]

        return _why(op, inputs, outputs, run), impl, tunables
    return None, impl, tunables


def _gpu_round(calls) -> None:
    import torch

    for b in calls:
        with torch.cuda.device(b.group.device):
            b.run()
