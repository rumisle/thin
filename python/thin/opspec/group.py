"""Groups of ranks and the memory they hold.

A handle (``Group``) is one rank's view of a group; the handles of one group
share a team: its world, the runtime's held table for the group's Bounds,
its mapped memory and its group tensors. A handle satisfies the runtime's
RankGroup protocol: ``rank``, ``device`` and ``team``.

The three kinds differ only in where memory lives: host buffers and numpy
arrays (``emulated``), uncached allocations on each GPU of one process
(``local``), or one process a rank exchanging IPC handles (``ipc``, in
``thin.opspec._gpu``).
"""

from __future__ import annotations

from thin.opspec._tensor import dtype_of, shape_of, storage_of

__all__ = ["Group", "Team", "emulate", "emulated", "ipc", "local", "slow"]


def check_world(world) -> None:
    """A world is an int, not a bool, of 1 or more."""
    if not isinstance(world, int) or isinstance(world, bool) or world < 1:
        raise ValueError(f"a group of {world!r} ranks: a positive int")


class EmulatedMemory:
    """An emulated group's memory: zeroed host bytes and numpy arrays."""

    kind = "emulated"

    def __init__(self, world: int):
        self.world = world

    def device(self, rank: int):
        """Every rank's device is the emulated tag."""
        return "emulated"

    def tensors(self, name: str, dtype, storage) -> list:
        """Every rank's zeroed copy of a group tensor."""
        import numpy as np

        numpy, shape = storage
        return [np.zeros(shape, dtype=numpy) for _ in range(self.world)]

    def buffers(self, nbytes: int, label: str) -> list:
        """Every rank's zeroed buffer of nbytes."""
        return [bytearray(nbytes) for _ in range(self.world)]

    def close(self) -> None:
        """Host memory is freed when its last reference is released."""


class Team:
    """What the handles of one group share: the world and three tables."""

    def __init__(self, memory):
        self.memory = memory
        """The kind's memory."""
        self.world = memory.world
        """The group's size."""
        self._clear()

    def _clear(self) -> None:
        from thin.runtime.bound import HeldTable

        self.held = HeldTable()
        """The runtime's held table of the group's Bounds, keyed by rank."""
        self._tensors: dict = {}
        self._mapped: dict = {}

    @property
    def kind(self) -> str:
        """``emulated``, ``local`` or ``ipc``."""
        return self.memory.kind

    def tensor(self, rank: int, name: str, dtype, shape):
        """Rank's copy of the group tensor name, every rank's made at the first ask."""
        d = dtype_of(dtype)
        shape = shape_of(shape)
        storage = storage_of(d, shape)
        ident = (d.name, tuple(int(s) for s in shape))
        held = self._tensors.get(name)
        if held is None:
            held = (ident, self.memory.tensors(name, d, storage))
            self._tensors[name] = held
        elif held[0] != ident:
            raise ValueError(f"the group's tensor {name}: {held[0]}, asked as {ident}")
        return held[1][rank]

    def tensor_names(self) -> dict:
        """Name → every rank's copy, for each group tensor made."""
        return {name: copies for name, (_, copies) in self._tensors.items()}

    def peers_of(self, v):
        """Every rank's copy of the group tensor v is a copy of, or None.

        A torch view of a copy's whole storage (the same data pointer, device,
        shape and dtype) counts as the copy.
        """
        torch = (type(v).__module__ or "").split(".")[0] == "torch"
        for _, copies in self._tensors.values():
            for c in copies:
                if c is v:
                    return list(copies)
                same = torch and (type(c).__module__ or "").startswith("torch")
                if same and (c.data_ptr(), c.device, c.shape, c.dtype) == (
                    v.data_ptr(),
                    v.device,
                    v.shape,
                    v.dtype,
                ):
                    return list(copies)
        return None

    def holds(self, kernel, name: str, nbytes: int) -> bool:
        """Whether the newest allocation of (kernel address, name) exists and is at least nbytes."""
        found = self._mapped.get((kernel.address, name))
        return bool(found) and found[-1][0] >= nbytes

    def mapped(self, kernel, name: str, nbytes: int) -> list:
        """Every rank's buffer of a mapped argument: the newest if large enough, else new."""
        allocations = self._mapped.setdefault((kernel.address, name), [])
        if not allocations or allocations[-1][0] < nbytes:
            label = f"{kernel.name}:{name}"
            allocations.append((nbytes, self.memory.buffers(nbytes, label)))
        return list(allocations[-1][1])

    def close(self) -> None:
        """Empties the three tables and frees what the kind allocated."""
        try:
            self.memory.close()
        finally:
            self._clear()


class Group:
    """One rank's handle on a group."""

    def __init__(self, team: Team, rank: int):
        self.team = team
        """The team the group's handles share."""
        self.rank = rank
        """This handle's rank."""

    @property
    def world(self) -> int:
        """The group's size."""
        return self.team.world

    @property
    def device(self):
        """``"emulated"``, or the rank's GPU index."""
        return self.team.memory.device(self.rank)

    def tensor(self, name: str, dtype, shape):
        """The handle's rank's copy of the group tensor name."""
        return self.team.tensor(self.rank, name, dtype, shape)

    def close(self) -> None:
        """Frees the group's memory, for every handle of the group."""
        self.team.close()

    def __repr__(self) -> str:
        return f"Group({self.team.kind}, rank {self.rank} of {self.world})"


def handles(memory) -> tuple:
    """Every rank's handle on one new team over memory."""
    team = Team(memory)
    return tuple(Group(team, r) for r in range(memory.world))


def emulated(world) -> tuple:
    """``world`` handles on the emulator, rank r at [r]."""
    check_world(world)
    return handles(EmulatedMemory(world))


def local(world) -> tuple:
    """``world`` handles in one process, GPU r as rank r."""
    from thin.opspec import _gpu

    check_world(world)
    return handles(_gpu.LocalMemory(world))


def ipc(world, rank, allgather):
    """The process's one handle, one process a rank, on the current GPU."""
    from thin.opspec import _gpu

    check_world(world)
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world:
        raise ValueError(f"rank {rank!r} of a group of {world}")
    return Group(Team(_gpu.IpcMemory(world, rank, allgather)), rank)


def slow(world, resident=None) -> dict:
    """The group's schedule in which the last rank starts 40 turns late, 16 times slower."""
    sched = {"skew": [0] * (world - 1) + [40], "speed": [1] * (world - 1) + [1 / 16]}
    if resident is not None:
        sched["resident"] = resident
    return sched


def _refuse(bounds, sched) -> Team:
    """refusals, in order; returns the team."""
    from thin.runtime.errors import LaunchError

    teams = {id(getattr(b.group, "team", None)) for b in bounds}
    first = bounds[0].group if bounds else None
    if not bounds or None in (b.group for b in bounds) or len(teams) != 1:
        raise LaunchError("emulate: the Bounds of one emulated group (thin.group.emulated)")
    if first.team.kind != "emulated":
        raise LaunchError("emulate: the Bounds of one emulated group (thin.group.emulated)")
    from thin.opspec._impl import arg_fact

    # Each kernel's mapped memory is sized alike in every Bound of it.
    heads: dict = {}
    for b in bounds:
        head = heads.setdefault(id(b.kernel), b)
        if head is b:
            continue
        mapped = [n for n, r in head.kernel.args.items() if arg_fact(r, "owned") == "mapped"]
        for n in mapped:
            if b.args[n] != head.args[n]:
                raise LaunchError(
                    f"{b.kernel.name}: rank {b.group.rank} binds its mapped {n} to {b.args[n]} "
                    f"bytes, rank {head.group.rank} to {head.args[n]}: a group's memory is sized "
                    "alike on every rank"
                )
    resident = (sched or {}).get("resident")
    for b in bounds:
        workgroups = b.grid[0] * b.grid[1] * b.grid[2]
        if b.co_resident and resident and workgroups > resident:
            raise LaunchError(
                f"{b.kernel.name}: co-resident, {workgroups} workgroups; sched resident {resident}"
            )
    return first.team


def emulate(bounds, sched=None, staged=()) -> None:
    """Runs the Bounds of one emulated group together, rank r's on device r."""
    from thin.runtime import emulated as emu
    from thin.runtime.bound import _co_target, joint_run

    bounds = list(bounds)
    team = _refuse(bounds, sched)
    target = _co_target(bounds[0].kernel.co) if bounds else None
    devices = emu.group(team.world, target=target, **(sched or {}))
    try:
        on = [devices[b.group.rank] for b in bounds]
        joint_run(bounds, on, team.held, lambda b: b.group.rank, tuple(staged))
    finally:
        for d in devices:
            if not d.closed:
                d.close()
