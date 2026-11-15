"""The emulated device and its groups.

The device owns allocation (tagged buffers ``b<k>``, no guard), loading
with its refusals, and the queue a group runs at a wait. The machine that
runs a group's queue is the emulator's: ``Emulator``
holds its memory and runs a queue as one cluster run. The emulator loads
on first use, so importing the runtime loads no compiler.
"""

from __future__ import annotations

import warnings

from thin.runtime.device import Device
from thin.runtime.errors import DeviceFault, LaunchError, NotEmulated

__all__ = ["INTERLEAVING", "BankConflicts", "EmulatedDevice", "Emulator", "group"]

INTERLEAVING = [
    "skew",
    "speed",
    "jitter",
    "wave_jitter",
    "quantum",
    "resident",
    "order",
    "wg_speed",
    "lockstep",
]
"""The Cluster's interleaving keys."""


class BankConflicts(UserWarning):
    """LDS bank conflicts under the emulator's bank model."""


class StaleReads(UserWarning):
    """Reads the flight model served a previous value.

    The program read memory another workgroup was still writing, with nothing
    ordering the two.
    """


class Emulator:
    """The emulator as the device sees it: its target, wave width, memory and run."""

    def __init__(self):
        from thin import _native
        from thin.emu import Memory

        self.target = _native.target_names()[0]
        """The emulator's target name: the registry's default target."""
        self.wave = _native.target_info(self.target)[0]
        """The emulated wave width (the target's)."""
        self.memory = Memory()
        self.buffers: list = []
        self.staged: list = []
        self.stale_seen = 0

    def alloc(self, nbytes: int, fill: int, meta, rank=0) -> int:
        """A buffer ``b<k>`` (tagged unless meta says ``tagged: False``); returns its index."""
        meta = dict(meta or {})
        name = meta.pop("name", f"b{len(self.buffers)}")
        staged = meta.pop("staged", None)
        tagged = meta.pop("tagged", True)
        b = self.memory.alloc(name, nbytes, tagged, fill, **meta)
        self.buffers.append(b)
        if staged is not None:
            self.staged.append((rank, len(self.buffers) - 1, bytes(staged)))
        return len(self.buffers) - 1

    def read(self, k: int, offset: int, n: int) -> bytes:
        """N bytes of buffer k."""
        return self.buffers[k].read(offset, n)

    def write(self, k: int, offset: int, data) -> None:
        """Writes bytes into buffer k."""
        self.buffers[k].write(offset, data)

    def run(self, launches, stall_rounds, interleaving, staged=None) -> None:
        """Runs a group's queued launches in one emulator run."""
        from thin.emu import Cluster, EmuError, EmuLimit

        whats = "; ".join(x[5] for x in launches)
        (asm, wgsize, vgpr, agpr, lds), grid = launches[0][2], launches[0][3]
        world = max(x[0] or 0 for x in launches) + 1
        kernargs = [
            [self.buffers[x[4]] for x in launches if (x[0] or 0) == r] for r in range(world)
        ]
        man = {"wgsize": wgsize, "vgpr": max(vgpr, 1), "agpr": agpr, "lds": lds}
        options = {k: v for k, v in interleaving.items() if k in INTERLEAVING}
        try:
            c = Cluster(
                asm, man, self.memory, kernargs, grid[:2], stall_rounds=stall_rounds, **options
            )
            # A staged copy takes its bytes as its rank's first launch starts:
            # later launches find what the earlier ones left in it.
            c.staged = [
                (r, 0, k, 0, data) for r, k, data in self.staged if r < world and kernargs[r]
            ]
            machines = c.run()
        except EmuLimit as e:
            raise NotEmulated(f"{whats}: {e}") from None
        except EmuError as e:
            raise DeviceFault(f"{whats}: {e}", e.cause) from None
        _bank_warning(whats, machines, c.texts, self.target)
        events = self.memory.flight.events
        new, self.stale_seen = events[self.stale_seen :], len(events)
        if new:
            shown = "\n  ".join(new[:8])
            more = f"\n  ({len(new) - 8} more)" if len(new) > 8 else ""
            text = f"{whats}: {len(new)} stale reads served\n  {shown}{more}"
            warnings.warn(text, StaleReads, stacklevel=3)


def _bank_warning(whats, machines, texts, target) -> None:
    """Report BankConflicts when LDS instructions have bank conflicts."""
    total: dict = {}
    for m in machines:
        for pc, (e, cyc, conf) in m.banks.items():
            t = total.setdefault(pc, [0, 0, 0])
            t[0], t[1], t[2] = t[0] + e, t[1] + cyc, t[2] + conf
    hot = sorted((v[2], pc) for pc, v in total.items() if v[2])
    if not hot:
        return
    cycles = sum(v[1] for v in total.values())
    conflicts = sum(v[2] for v in total.values())
    lines = [
        f"{whats}: LDS bank conflicts (the {target} bank model) at {len(hot)} of "
        f"{len(total)} ds instructions, {conflicts:g} of {cycles:g} cycles"
    ]
    for _, pc in sorted(hot, key=lambda x: (-x[0], x[1]))[:8]:
        e, cyc, conf = total[pc]
        text, line, comment = texts[pc]
        note = f"  // {comment}" if comment else ""
        lines.append(
            f"  .s line {line}: `{text}`{note}: {e:g} executions, {cyc / e:g} cycles an "
            f"execution, {conf / e:g} of them conflicts"
        )
    if len(hot) > 8:
        lines.append(f"  ({len(hot) - 8} more)")
    warnings.warn("\n".join(lines), BankConflicts, stacklevel=4)


class _Shared:
    """A group's one memory and one queue."""

    def __init__(self, world, stall_rounds, interleaving, emulator=None):
        self.emu = emulator or Emulator()
        self.world, self.stall_rounds, self.interleaving = world, stall_rounds, interleaving
        self.queue: list = []
        self.after: list = []
        """Copies and fills enqueued behind queued launches, done once those run."""
        self.devices: list[EmulatedDevice] = []


class EmulatedDevice(Device):
    """A device over the emulator's memory: guard 0."""

    guard = 0

    def __init__(self, shared=None, rank=None):
        super().__init__(rank)
        self.shared = shared or _Shared(1, 20000, {})
        self.shared.devices.append(self)

    def alloc_bytes(self, n, fill, meta):
        """A tagged buffer ``b<k>`` of the emulator's memory, at its address."""
        k = self.shared.emu.alloc(n, fill, meta, self.rank or 0)
        return k, self.shared.emu.buffers[k].base

    def write_bytes(self, allocation, offset, data):
        """Writes into the buffer at once."""
        self.shared.emu.write(allocation, offset, data)

    def read_bytes(self, allocation, offset, n):
        """Reads the buffer (the launches before it have run)."""
        return self.shared.emu.read(allocation, offset, n)

    def copy_bytes(self, dst, dst_offset, src, src_offset, n):
        """A copy, after the launches enqueued before it."""
        emu = self.shared.emu
        self._after(lambda: emu.write(dst, dst_offset, emu.read(src, src_offset, n)))

    def fill_bytes(self, allocation, offset, n, byte):
        """A fill, after the launches enqueued before it."""
        emu = self.shared.emu
        self._after(lambda: emu.write(allocation, offset, bytes([byte]) * n))

    def _after(self, op):
        """Does op once the launches queued before it ran.

        At once when none is queued (a group of one runs its launches at
        once); else when the group's queue next runs, in order.
        """
        if self.shared.queue:
            self.shared.after.append(op)
        else:
            op()

    def free(self):
        """The emulator's memory lives as long as the group."""

    def load(self, image, code_object, asm, block, key):
        """refusals, in order; the code is the .s with its counts."""
        emu, obj = self.shared.emu, code_object
        if obj.target != emu.target:
            raise LaunchError(
                f"{key}: a code object for {obj.target}, the emulated device is {emu.target}"
            )
        meta = (obj.metadata.get("amdhsa.kernels") or [{}])[0]
        block = tuple(block)
        if block[0] % emu.wave or block[1] != 1:
            raise LaunchError(f"{key}: block {block} on the emulated device: whole waves, one row")
        _same_kernel(key, asm, obj)
        _encodes(key, asm, obj)
        agpr = meta.get(".agpr_count", 0)
        return (asm, block[0], meta.get(".vgpr_count", 0) - agpr, agpr, obj.lds)

    def run(self, code, key, grid, block, kernarg, what, cooperative):
        """Queues the launch; a group of one runs it at once."""
        resident = self.shared.interleaving.get("resident")
        workgroups = grid[0] * grid[1] * grid[2]
        if cooperative and resident is not None and workgroups > resident:
            raise LaunchError(
                f"{what}: a cooperative launch of {workgroups} workgroups; the group's residency "
                f"{resident}"
            )
        k = self.shared.emu.alloc(
            len(kernarg), 0, {"name": f"{what} rank {self.rank}#kernarg", "tagged": False}
        )
        self.shared.emu.write(k, 0, kernarg)
        self.shared.queue.append((self.rank, key, code, grid, k, what))
        if self.shared.world == 1:
            self._drain()

    def _drain(self):
        queue, self.shared.queue = self.shared.queue, []
        if not queue:
            return
        whats = "; ".join(x[5] for x in queue)
        rank0, key, code, grid = queue[0][:4]
        for rank, key2, code2, grid2, _, _ in queue[1:]:
            if (key2, grid2) != (key, grid):
                raise LaunchError(
                    f"{whats}: one kernel and one grid a group's run, not {key2} at {grid2} "
                    f"beside {key} at {grid}"
                )
            if code2 != code:
                raise LaunchError(
                    f"{whats}: rank {rank} loaded other code under {key} than rank {rank0}: one "
                    "program a group's run"
                )
        if grid[2] != 1:
            raise NotEmulated(f"{whats}: a grid of {grid[2]} in z on the emulator")
        self.shared.emu.run(queue, self.shared.stall_rounds, self.shared.interleaving)
        after, self.shared.after = self.shared.after, []
        for op in after:
            op()

    def sync(self, what):
        """Runs the group's queue."""
        self._drain()


def _same_kernel(key, asm, obj) -> None:
    """The .s names the code object's kernel and user SGPR count."""
    name, count = None, 2
    seen = False
    for line in asm.splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == ".amdhsa_kernel" and name is None:
            name = words[1]
        elif len(words) >= 2 and words[0] == ".amdhsa_user_sgpr_count" and not seen:
            count, seen = int(words[1], 0), True
    rsrc2 = int.from_bytes(obj.descriptor[52:56], "little")
    theirs = (rsrc2 >> 1) & 0x1F
    if name != obj.symbol or count != theirs:
        raise LaunchError(
            f"{key}: the .s is not this code object's (.amdhsa_kernel {name}, "
            f".amdhsa_user_sgpr_count {count}; "
            f"the code object's {obj.symbol}, {theirs})"
        )


_ENCODED: dict = {}


def _encodes(key, asm, obj) -> None:
    """The .s encodes to exactly the code object's .text."""
    from thin import codegen, formats

    text = _ENCODED.get(asm)
    if text is None:
        try:
            text = formats.read(codegen.code_object(asm)).text
        except Exception as e:
            raise LaunchError(f"{key}: the .s does not encode: {e}") from None
        _ENCODED[asm] = text
    if text != obj.text:
        raise LaunchError(f"{key}: the .s does not encode to the code object's .text")


def group(world, stall_rounds=20000, **interleaving) -> list[EmulatedDevice]:
    """``world`` emulated devices on one memory and one queue."""
    if type(world) is not int or world < 1:
        raise LaunchError(f"a group of {world!r} ranks: a positive int")
    bad = sorted(set(interleaving) - set(INTERLEAVING))
    if bad:
        raise LaunchError(f"a group's interleaving {bad}: the Cluster's are {INTERLEAVING}")
    if type(stall_rounds) is not int or stall_rounds < 1:
        raise LaunchError(f"a group's stall_rounds {stall_rounds!r}: a positive int")
    shared = _Shared(world, stall_rounds, interleaving)
    return [EmulatedDevice(shared, rank) for rank in range(world)]
