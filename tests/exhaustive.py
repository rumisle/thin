"""The work of the exhaustive encode checks (tests/test_formats_exhaustive.py).

All 2^32 f32 patterns are split into pieces of ``PIECE`` patterns made
where they are used, so memory stays a few pieces whatever the input.
"""

from __future__ import annotations

import numpy as np

PIECE = 1 << 24
PIECES = (1 << 32) // PIECE


def host_patterns(piece: int) -> np.ndarray:
    """The f32 values of one piece, on the host."""
    start = piece * PIECE
    return np.arange(start, start + PIECE, dtype=np.uint32).view(np.float32)


def device_patterns(torch, piece: int, device):
    """The bits of one piece as int32, on a device."""
    start = piece * PIECE
    wide = torch.arange(start, start + PIECE, dtype=torch.int64, device=device)
    return wide.to(torch.int32)


def first_bad_index(torch, piece: int, device) -> int | None:
    """The first pattern of one piece whose encode-table index, as the
    device path computes it, is not its top 16 bits then a sticky bit."""
    from thin import formats

    bits = device_patterns(torch, piece, device)
    got = formats._table_index(bits)
    unsigned = bits.long() & 0xFFFFFFFF
    want = (unsigned >> 16) * 2 + ((unsigned % 65536) > 0).long()
    bad = torch.nonzero(got != want)
    return None if bad.numel() == 0 else piece * PIECE + int(bad[0, 0])


def first_bad_cast(torch, cases, piece: int, device) -> list[int | None]:
    """Per option set (dtype, kwargs), the first pattern of one piece whose
    device encode differs from the definition, the numpy path."""
    from thin import formats

    host = host_patterns(piece)
    on_device = device_patterns(torch, piece, device).view(torch.float32)
    found = []
    for name, kwargs in cases:
        dtype = formats.DTYPES[name]
        want = torch.from_numpy(formats.encode(host, dtype, **kwargs).view(np.int16))
        got = formats.encode(on_device, dtype, **kwargs).view(torch.int16)
        bad = torch.nonzero(got != want.to(device))
        found.append(None if bad.numel() == 0 else piece * PIECE + int(bad[0, 0]))
    return found
