"""laps: a kernel's own clock.

Waves read the core clock at named points (laps), keep each round's laps
and store their records into scratch memory the declaration adds; the host
decodes them (``Laps.decode``) into a ``Timeline``.

The core clock is the shader clock the wave runs at, the target's
``core_clock`` (gfx1201: SHADER_CYCLES); the constant clock beside it, the
target's ``real_clock`` at ``real_clock_mhz`` (gfx1201: the 100 MHz
REALTIME), gives its frequency. Measured on an RX 9070 XT over a 3.9 s
span: 3377 MHz against the driver's 3380 MHz sclk, the constant clock
agreeing with the host's to 1 in 10^5; the card boosts past its rated
2970 MHz when lightly loaded.
"""

from __future__ import annotations

from thin.dsl import O, lane_write, lib_arg, scratch, select
from thin.lib import _hooks

__all__ = ["Laps", "LapsBody", "Timeline"]

_KEEPS = ("last", "first", "sum")
_HEADER = 9
"""The header row's own words; the summed laps follow them."""


def _row() -> int:
    """A record row's words: one per lane of a wave."""
    return _hooks.granule()


def _check(names, wgsize, keep, records) -> None:
    """The declaration's refusals, in order."""
    if keep not in _KEEPS:
        raise ValueError(f"keep {keep!r}: one of {_KEEPS}")
    row = _row()
    if not (isinstance(records, int) and 1 <= records <= row):
        raise ValueError(f"records {records}: 1 to {row} (a VGPR's lanes)")
    if not (isinstance(wgsize, int) and wgsize > 0 and wgsize % _hooks.granule() == 0):
        raise ValueError(f"wgsize {wgsize}: whole waves")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"laps {names}: one or more, each once")
    if keep == "sum" and _HEADER + len(names) > row:
        raise ValueError(
            f"{len(names)} laps: at most {row - _HEADER} summed (the header row's lanes)"
        )


class _Clock:
    """A clock read as the target's hook spells it: ``pair OP [IMM]`` or ``halves OP LO HI``."""

    def __init__(self, hook: str):
        shape, op, *imms = _hooks.hook(hook).split()
        self.pair = shape == "pair"
        self.op, self.imms = op, [int(i) for i in imms]


class Laps:
    """A declaration of named laps and their records."""

    def __init__(self, names, wgsize, keep="last", records=None, on=True):
        """Declares the laps a round takes, in order."""
        names = list(names)
        records = _row() if records is None else records
        _check(names, wgsize, keep, records)
        self.names, self.wgsize, self.keep, self.records, self.on = (
            names,
            wgsize,
            keep,
            records,
            on,
        )
        self.waves = wgsize // _hooks.granule()
        self.rows = 1 if keep == "sum" else 1 + len(names)

    def args(self) -> dict:
        """The argument it adds (none when off)."""
        return {"laps": lib_arg(scratch, "laps")} if self.on else {}

    def sizes(self, workgroups) -> dict:
        """The records' bytes for `workgroups` workgroups (none when off)."""
        return {"laps": workgroups * self.waves * self.rows * 4 * _row()} if self.on else {}

    def begin(self, b, k):
        """Starts the clock in a body; the off object when off."""
        return LapsBody(self, b, k) if self.on else _Off(self.names)

    def decode(self, data, workgroups) -> Timeline:
        """The host decode of the records."""
        if not self.on:
            raise ValueError("laps off: nothing recorded")
        return Timeline(self, _words(data), workgroups)


class _Off:
    """Off laps: every call does nothing but check its lap's name."""

    def __init__(self, names):
        self.names = names

    def lap(self, name) -> None:
        """Nothing, for a declared lap."""
        if name not in self.names:
            raise ValueError(f"lap {name!r}: not one of {self.names}")

    def record(self) -> None:
        """Nothing."""

    def end(self, wg=None) -> None:
        """Nothing."""


class LapsBody:
    """A body's laps: lap, record, end."""

    def __init__(self, decl: Laps, b, k):
        """Reads the clocks at begin and makes the round's registers."""
        self.decl, self.b, self.k = decl, b, k
        self.S, self.V = b.s, b.v
        self.real, self.core = _Clock("real_clock"), _Clock("core_clock")
        self.real0 = self._clock(self.real, "laps_real0")
        self.core0 = self._clock(self.core, "laps_core0")
        self.clocks = {n: self.S.u64(0, name=f"lap_{n}") for n in decl.names}
        self.r = self.S.u32(0, name="laps_r")
        self.taken: list = []
        if decl.keep == "sum":
            self.sums = {n: self.S.u32(0, name=f"laps_sum_{n}") for n in decl.names}
            self.prev = self.S.u32(name="laps_prev")
            self._wait([(self.core0, self.core)], "laps: the begin's clock")
            self.prev[...] = self.core0.lo
        else:
            self.recs = {n: self.V.u32(0, name=f"laps_rec_{n}") for n in decl.names}

    def _clock(self, clock, name):
        reg = self.S.u64(name=name)
        self._read(clock, reg, wide=True)
        return reg

    def _read(self, clock: _Clock, reg, wide: bool) -> None:
        """A clock read into a 64-bit scalar pair.

        An asynchronous read is waited for by a register wait (``_wait``).
        A read by halves is synchronous; ``wide`` reads the high half on
        both sides of the low one and takes a low half read across a carry
        as 0 (a time between the two high reads), so the pair is one
        clock. A lap needs only the low half.
        """
        b = self.b
        if clock.pair:
            b.emit(O.op(clock.op, reg, *clock.imms))
            return
        lo, hi = clock.imms
        if not wide:
            b.emit(O.op(clock.op, reg.lo, lo))
            return
        first = self.S.u32(name="laps_hi")
        b.emit(O.op(clock.op, first, hi))
        b.emit(O.op(clock.op, reg.lo, lo))
        b.emit(O.op(clock.op, reg.hi, hi))
        reg.lo[...] = select(first == reg.hi, reg.lo, 0)

    def _wait(self, reads, comment) -> None:
        """Waits for the asynchronous reads among ``reads``, (register, clock) pairs."""
        regs = [r for r, clock in reads if clock.pair]
        if regs:
            self.b.wait(*regs, comment=comment)

    def lap(self, name) -> None:
        """Reads the core clock into the lap's clock, not waited for."""
        if name not in self.clocks:
            raise ValueError(f"lap {name!r}: not one of {self.decl.names}")
        self._read(self.core, self.clocks[name], wide=False)
        if name not in self.taken:
            self.taken.append(name)

    def record(self) -> None:
        """Keeps the round's laps, then zeroes them."""
        d, b = self.decl, self.b
        if self.taken:
            taken = [self.clocks[n] for n in d.names if n in self.taken]
            self._wait([(c, self.core) for c in taken], "laps: the round's clocks")
        if d.keep == "last":
            slot = self.S.u32(self.r % d.records, name="laps_slot")
        for n in d.names:
            c = self.clocks[n].lo
            if d.keep == "sum":
                self.sums[n][...] = self.sums[n] + select(c != 0, c - self.prev, 0)
                self.prev[...] = select(c != 0, c, self.prev)
            elif d.keep == "last":
                self.recs[n][...] = lane_write(self.recs[n], slot, c)
            else:
                with b.when(self.r < d.records):
                    self.recs[n][...] = lane_write(self.recs[n], self.r, c)
        for n in d.names:
            self.clocks[n][...] = 0
        self.r[...] = self.r + 1

    def end(self, wg=None) -> None:
        """Records the last round and stores the records."""
        d, b, k = self.decl, self.b, self.k
        self.record()
        real1 = self._clock(self.real, "laps_real1")
        core1 = self._clock(self.core, "laps_core1")
        ends = [
            (real1, self.real),
            (core1, self.core),
            (self.real0, self.real),
            (self.core0, self.core),
        ]
        self._wait(ends, "laps: the clocks at begin and end")
        head = self.V.u32(0, name="laps_head")
        words = [self.r]
        for c in (self.real0, self.core0, real1, core1):
            words += [c.lo, c.hi]
        if d.keep == "sum":
            words += [self.sums[n] for n in d.names]
        for lane, word in enumerate(words):
            head[...] = lane_write(head, lane, word)
        (laps,) = k.load("laps")
        flat = k.wgid if wg is None else wg
        row0 = self.S.u32((flat * d.waves + k.wave) * d.rows, name="laps_row0")
        lane4 = self.V.u32(k.lane * 4, name="laps_lane4")
        rows = [head] + ([] if d.keep == "sum" else [self.recs[n] for n in d.names])
        for j, row in enumerate(rows):
            at = self.S.ptr(laps + (row0 + j) * (4 * _row()), name="laps_at")
            b.emit(O.global_store(lane4, row, at))


# --------------------------------------------------------------- the host decode


def _words(data):
    """The records as little-endian u32 words, from their bytes.

    ``data`` is the scratch buffer as any contiguous array or tensor (its
    bytes are viewed, whatever its element type) or a bytes-like object.
    """
    import numpy as np

    if hasattr(data, "detach"):
        data = data.detach().cpu().contiguous().numpy()
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(data, np.uint8)
    else:
        raw = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    return raw[: raw.size // 4 * 4].view("<u4")


def _percentile(values, q):
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class Timeline:
    """The decoded records: rounds, clocks and laps per workgroup and wave."""

    def __init__(self, decl: Laps, words, workgroups):
        """Reads the records of `workgroups` workgroups."""
        want = workgroups * decl.waves * decl.rows * _row()
        if words.size < want:
            raise ValueError(
                f"{words.size * 4} bytes of records, {workgroups} workgroups need {want * 4}"
            )
        self.decl, self.workgroups = decl, workgroups
        rows = words[:want].reshape(workgroups, decl.waves, decl.rows, _row()).astype(int)
        self._rows = rows
        self.rounds = [
            [int(rows[g, w, 0, 0]) for w in range(decl.waves)] for g in range(workgroups)
        ]
        self.clock_mhz = [
            [self._mhz(rows[g, w, 0]) for w in range(decl.waves)] for g in range(workgroups)
        ]
        # Each wave's span in ticks of the constant clock: a clock is good to
        # one tick in its span (a 1 us span, 100 ticks at 100 MHz, to 1%).
        self.clock_ticks = [
            [self._ticks(rows[g, w, 0]) for w in range(decl.waves)] for g in range(workgroups)
        ]
        self.clock = self._pooled()
        """The core clock in MHz over every wave's span together (None if none
        has one): good to one tick in the spans' total, where a wave's own
        clock is good to one in its span."""
        self.laps = {}
        for j, name in enumerate(decl.names):
            if decl.keep == "sum":
                per = [
                    [int(rows[g, w, 0, _HEADER + j]) for w in range(decl.waves)]
                    for g in range(workgroups)
                ]
            else:
                per = [
                    [self._kept(rows[g, w, 1 + j], self.rounds[g][w]) for w in range(decl.waves)]
                    for g in range(workgroups)
                ]
            self.laps[name] = per

    def _pooled(self):
        pairs = [
            (m * t, t)
            for row_m, row_t in zip(self.clock_mhz, self.clock_ticks, strict=True)
            for m, t in zip(row_m, row_t, strict=True)
            if m is not None
        ]
        ticks = sum(t for _, t in pairs)
        return sum(c for c, _ in pairs) / ticks if ticks else None

    @staticmethod
    def _ticks(head) -> int:
        return ((int(head[5]) | int(head[6]) << 32) - (int(head[1]) | int(head[2]) << 32)) % 2**64

    @staticmethod
    def _mhz(head) -> float:
        def wide(lo):
            return int(head[lo]) | int(head[lo + 1]) << 32

        core = (wide(7) - wide(3)) % 2**64
        real = (wide(5) - wide(1)) % 2**64
        if not core or not real:
            return None  # an empty span measures no clock
        return core / real * int(_hooks.hook("real_clock_mhz"))

    def _kept(self, row, rounds) -> list:
        records = self.decl.records
        kept = min(rounds, records)
        if self.decl.keep == "last":
            return [int(row[i % records]) for i in range(rounds - kept, rounds)]
        return [int(row[i]) for i in range(kept)]

    def phases(self) -> dict:
        """Each lap's phases over every workgroup, wave and kept round."""
        d = self.decl
        if d.keep == "sum":
            raise ValueError("keep sum: the totals are the laps (Timeline.laps)")
        out = {n: [] for n in d.names}
        for g in range(self.workgroups):
            for w in range(d.waves):
                prev = None
                kept = len(self.laps[d.names[0]][g][w])
                for i in range(kept):
                    for n in d.names:
                        c = self.laps[n][g][w][i]
                        if c == 0:
                            continue
                        if prev is not None:
                            out[n].append((c - prev) % 2**32)
                        prev = c
        return out

    def summary(self) -> str:
        """The text summary: each lap's median and spread, in clocks and ns.

        A wave whose clock spans are empty measures no clock; a summed lap's
        totals are its clocks mod 2^32. The clock is the pooled one
        (``clock``), the waves' own ranging beside it; when a wave's span is
        under 100 ticks of the constant clock its own clock is off by up to
        one tick in the span, which the text gives as "short spans: ±P%".
        """
        import numpy as np

        d = self.decl
        every = [m for row in self.clock_mhz for m in row]
        known = [m for m in every if m is not None]
        rounds = [r for row in self.rounds for r in row]
        waves = f"{len(every)} wave{'s' if len(every) != 1 else ''}"
        mhz = self.clock
        if mhz is None:
            clock = "clock unknown (no wave's clock span)"
        else:
            unknown = f", {len(every) - len(known)} unknown" if len(known) < len(every) else ""
            spans = [
                t
                for row_m, row_t in zip(self.clock_mhz, self.clock_ticks, strict=True)
                for m, t in zip(row_m, row_t, strict=True)
                if m is not None
            ]
            worst = 100 / min(spans)
            short = f", short spans: ±{worst:.2g}%" if worst > 1 else ""
            span = f"waves {min(known):.0f}-{max(known):.0f}{unknown}{short}"
            clock = f"clock {mhz:.0f} MHz ({span})"
        lines = [
            f"laps ({d.keep}): {waves}, {int(np.median(rounds))} rounds a wave (median); {clock}"
        ]
        values = (
            {n: [v for row in self.laps[n] for v in row] for n in d.names}
            if d.keep == "sum"
            else self.phases()
        )
        for n in d.names:
            vs = values[n]
            # A summed lap never taken totals 0 on every wave.
            if not vs or (d.keep == "sum" and not any(vs)):
                taken = d.keep != "sum" and any(c for row in self.laps[n] for w in row for c in w)
                what = "taken, with no lap before it to measure from" if taken else "never taken"
                lines.append(f"  {n:12s} {what}")
                continue
            p10, p50, p90 = (_percentile(vs, q) for q in (10, 50, 90))
            ns = f"({p50 / mhz * 1000:10.1f} ns)" if mhz else f"({'?':>10} ns)"
            lines.append(f"  {n:12s} {p50:12.0f} clk {ns}  p10 {p10:.0f}  p90 {p90:.0f}")
        return "\n".join(lines)
