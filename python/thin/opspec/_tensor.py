"""Tensors: as a signature declares them, their storage, given to an op.

Nothing here imports numpy, torch or the formats module until a tensor is
made or checked: importing the definitions stays light.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

__all__ = ["In", "InOut", "Out", "Tensor", "dtype_of", "place_of", "storage_of"]


def dtype_of(dtype):
    """The formats dtype record of a dtype or its name; ValueError for an unknown name."""
    from thin import formats

    if isinstance(dtype, formats.Dtype):
        return dtype
    found = formats.DTYPES.get(dtype) if isinstance(dtype, str) else None
    if found is None:
        raise ValueError(f"dtype {dtype!r}: one of {sorted(formats.DTYPES)}")
    return found


def _size_ok(size) -> bool:
    if isinstance(size, bool):
        return False
    try:
        operator.index(size)
    except TypeError:
        return False
    return True


def shape_of(shape) -> tuple:
    """The sizes, validated one by one; returned as given."""
    shape = tuple(shape)
    if not all(_size_ok(s) for s in shape):
        raise ValueError(f"shape {shape!r}: the sizes one by one, ints - In(dtype, rows, cols)")
    return shape


def storage_of(dtype, shape) -> tuple[str, tuple]:
    """(numpy dtype name, shape): the bits a numpy array of such a tensor holds."""
    d = dtype_of(dtype)
    sizes = tuple(operator.index(s) for s in shape)
    if d.width >= 8:
        return d.numpy, sizes
    per = 8 // d.width
    if not sizes or sizes[-1] % per:
        raise ValueError(
            f"a {d.name} tensor's last axis whole bytes ({per} a byte), not {tuple(shape)!r}"
        )
    return d.numpy, (*sizes[:-1], sizes[-1] // per)


@dataclass(frozen=True)
class Tensor:
    """A signature's tensor: io, dtype, shape (the sizes as given) and shared."""

    io: str
    """``"in"``, ``"out"`` or ``"inout"``."""
    dtype: object
    """Its dtype."""
    shape: tuple
    """Its sizes, as given."""
    shared: bool = False
    """Whether it is a group's tensor, one copy a rank."""

    def storage(self) -> tuple[str, tuple]:
        """(numpy dtype name, shape) of the storage."""
        return storage_of(self.dtype, self.shape)

    def poisoned(self):
        """A numpy array of the storage with every byte 0xFF."""
        import numpy as np

        name, shape = self.storage()
        out = np.empty(shape, dtype=name)
        out.reshape(-1).view(np.uint8)[:] = 0xFF
        return out


def _make(io, dtype, shape, shared) -> Tensor:
    return Tensor(io, dtype_of(dtype), shape_of(shape), bool(shared))


def In(dtype, *shape, shared=False) -> Tensor:  # noqa: N802 - the name
    """An input tensor."""
    return _make("in", dtype, shape, shared)


def Out(dtype, *shape, shared=False) -> Tensor:  # noqa: N802 - the name
    """An output tensor."""
    return _make("out", dtype, shape, shared)


def InOut(dtype, *shape, shared=False) -> Tensor:  # noqa: N802 - the name
    """A tensor read and written."""
    return _make("inout", dtype, shape, shared)


def _module(v) -> str:
    return (type(v).__module__ or "").split(".")[0]


def place_of(v) -> str:
    """``"host"`` for a numpy array, the torch device's text for a torch tensor."""
    return "host" if _module(v) == "numpy" else str(v.device)


def check_given(spec, name, t: Tensor, v) -> None:
    """A tensor given for signature tensor t: kind and dtype, shape, contiguity."""
    want, shape = t.storage()
    module = _module(v)
    if module == "numpy" and hasattr(v, "dtype") and hasattr(v, "flags"):
        if v.dtype.name != want:
            raise TypeError(
                f"{spec}: tensor {name}: {t.dtype.name} - a numpy array of {want}, not "
                f"{v.dtype.name}"
            )
        contiguous = v.flags.c_contiguous
    elif module == "torch" and hasattr(v, "is_contiguous"):
        got = str(v.dtype).removeprefix("torch.")
        if got != t.dtype.torch:
            raise TypeError(
                f"{spec}: tensor {name}: {t.dtype.name} - a torch tensor of {t.dtype.torch}, "
                f"not {got}"
            )
        contiguous = v.is_contiguous()
    else:
        raise TypeError(
            f"{spec}: tensor {name}: a numpy array or a torch tensor, not {type(v).__name__}"
        )
    if tuple(v.shape) != shape:
        raise ValueError(f"{spec}: tensor {name}: shape {shape}, not {tuple(v.shape)}")
    if not contiguous:
        raise ValueError(f"{spec}: tensor {name}: contiguous")
