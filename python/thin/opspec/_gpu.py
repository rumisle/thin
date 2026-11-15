"""The GPU kinds' memory.

Local: one process, GPU r as rank r. IPC: one process a rank.

Both use the runtime's zeroed uncached allocations; the
IPC kind exchanges each allocation's record through the caller's allgather.
Covered by the gfx1201 hardware tests; the exchange's record and its refusal are
pure and tested with a fake allgather.
"""

from __future__ import annotations

__all__ = ["IpcMemory", "LocalMemory", "exchange", "record"]


def _as_tensor(buf, dtype, shape):
    """A u8 uncached tensor viewed as the dtype's torch type in the storage shape."""
    import torch

    return buf.view(getattr(torch, dtype.torch)).reshape(shape)


def _nbytes(dtype, storage) -> int:
    import numpy as np

    numpy, shape = storage
    return int(np.dtype(numpy).itemsize * int(np.prod(shape, dtype=np.int64)))


class LocalMemory:
    """Rank r's memory is uncached allocations on GPU r; peer access between all."""

    kind = "local"

    def __init__(self, world: int):
        import torch

        from thin.runtime import hip
        from thin.runtime.errors import LaunchError

        count = torch.cuda.device_count()
        if count < world:
            raise LaunchError(f"a local group of {world}: {count} GPUs here")
        self.world = world
        hip.enable_peers(list(range(world)))
        self._owned: list = []

    def device(self, rank: int) -> int:
        """Rank r is GPU r."""
        return rank

    def tensors(self, name: str, dtype, storage) -> list:
        """Every rank's zeroed copy, a torch tensor in GPU r's uncached memory."""
        from thin.runtime import hip

        n = _nbytes(dtype, storage)
        out = [_as_tensor(hip.tensor(n, r)[:n], dtype, storage[1]) for r in range(self.world)]
        self._owned.append(out)
        return out

    def buffers(self, nbytes: int, label: str) -> list:
        """Every rank's zeroed uncached buffer of nbytes, on its GPU."""
        from thin.runtime import hip

        out = [hip.tensor(nbytes, r)[:nbytes] for r in range(self.world)]
        self._owned.append(out)
        return out

    def close(self) -> None:
        """Drops the allocations; each is freed with its last reference."""
        self._owned.clear()


def record(handle: bytes, nbytes: int, label: str) -> bytes:
    """One rank's exchange record: the 64-byte IPC handle, the size (8 bytes LE), the label."""
    return bytes(handle) + nbytes.to_bytes(8, "little") + label.encode()


def _fact(rec: bytes) -> tuple[int, str]:
    return int.from_bytes(rec[64:72], "little"), rec[72:].decode(errors="replace")


def exchange(world: int, rank: int, allgather, handle: bytes, nbytes: int, label: str) -> list:
    """Every rank's record of one allocation; refuses ranks that allocate differently."""
    from thin.runtime.errors import LaunchError

    mine = record(handle, nbytes, label)
    got = [bytes(r) for r in allgather(mine)]
    if len(got) != world or any(r[64:] != mine[64:] for r in got):
        every = [_fact(r) for r in got]
        raise LaunchError(
            f"the ranks allocate differently - rank {rank}: {nbytes} bytes of {label}; every "
            f"rank's: {every} (every rank makes the same calls in the same order, sized alike)"
        )
    return got


class IpcMemory:
    """An IPC rank's memory: its own allocations, exchanged; the others' opened."""

    kind = "ipc"

    def __init__(self, world: int, rank: int, allgather):
        import torch

        self.world, self.rank, self.allgather = world, rank, allgather
        self.gpu = torch.cuda.current_device()
        self._owned: list = []
        self._opened: list[int] = []

    def device(self, rank: int) -> int:
        """The process's GPU, for every rank."""
        return self.gpu

    def _shared(self, nbytes: int, label: str) -> list:
        from thin.runtime import hip

        own = hip.tensor(nbytes, self.gpu)[:nbytes]
        self._owned.append(own)
        records = exchange(
            self.world, self.rank, self.allgather, hip.ipc_handle(own.data_ptr()), nbytes, label
        )
        out: list = []
        for q, rec in enumerate(records):
            if q == self.rank:
                out.append(own)
                continue
            address = hip.open_ipc(rec[:64])
            self._opened.append(address)
            out.append(address)
        return out

    def tensors(self, name: str, dtype, storage) -> list:
        """Every rank's copy: this rank's a torch tensor, the others' addresses."""
        out = self._shared(_nbytes(dtype, storage), f"tensor {name}")
        out[self.rank] = _as_tensor(out[self.rank], dtype, storage[1])
        return out

    def buffers(self, nbytes: int, label: str) -> list:
        """Every rank's buffer: this rank's a tensor, the others' addresses."""
        return self._shared(nbytes, label)

    def close(self) -> None:
        """Closes the opened handles and frees the own buffers."""
        from thin.runtime import hip

        opened, self._opened = self._opened, []
        for address in opened:
            hip.close_ipc(address)
        self._owned.clear()
