# 7. Measuring

The earlier parts quoted times from `Bound.time()`. This part says what
that number is, how much it moves, how to compare two kernels fairly, and
how to look inside a kernel with its own clock.

```python
import numpy as np
import thin
from thin import cdiv
from examples.vadd import build_vadd
```

`examples/vadd.py` is part 1's kernel as a generator with options: the
workgroup size `BLOCK`, and `laps`, which we will use below. Its `host`
function makes the call from the arrays alone: `kern(x=, y=, z=, n=)`
computes the grid itself.

## What `time` measures

`bound.time(iters=20)` launches the call once (to warm up: the code
object loaded, the caches filled), then `iters` times back to back between
two GPU events, and returns the mean microseconds of one launch. It is
the time from the start of the first launch to the end of the last,
divided by `iters`: the kernels themselves, and the gaps between them.

For a tiny kernel the gaps are all there is:

```python gpu
import torch

kern = thin.compile(build_vadd)
for n in (1 << 10, 1 << 16, 1 << 20, 1 << 24):
    x, y, z = (torch.randn(n, device="cuda") for _ in range(3))
    us = kern(x=x, y=y, z=z, n=n).time()
    print(f"n = 2^{n.bit_length() - 1:2d}: {us:8.1f} us, {12 * n / us / 1e3:7.1f} GB/s")
```

```text
n = 2^10:     13.3 us,     0.9 GB/s
n = 2^16:     14.0 us,    56.3 GB/s
n = 2^20:     11.9 us,  1059.5 GB/s
n = 2^24:    328.0 us,   613.9 GB/s
```

Up to a million elements the time hardly depends on the size: it is the
cost of a launch. The bandwidth of the 2^20 run is even past what the
memory delivers: its 12 MiB stay in the GPU's cache from one launch to the
next. Measure kernels at the sizes they will run at, with data that does
not fit in the cache unless it will in use, and do not read bandwidth off
a kernel that runs for a few microseconds.

## How much it moves

Measure the same call several times:

```python gpu
n = 1 << 24
x, y, z = (torch.randn(n, device="cuda") for _ in range(3))
call = kern(x=x, y=y, z=z, n=n)
runs = sorted(call.time() for _ in range(7))
print(" ".join(f"{t:.1f}" for t in runs))
print(f"spread {100 * (runs[-1] - runs[0]) / runs[0]:.1f}%")
```

```text
328.5 328.5 328.7 328.7 328.9 328.9 329.1
spread 0.2%
```

Here the spread is small: a long, simple kernel on an idle GPU. It is
not always: a GPU's clock moves with its temperature and power, and other
work on the machine shares it. Report a median of several measurements,
and say what else ran.

## Comparing two kernels

To compare two variants, alternate them, so that a drift of the clock
affects both alike, and compare medians. Here are two workgroup sizes
of the same kernel:

```python gpu
calls = {b: thin.compile(build_vadd, BLOCK=b)(x=x, y=y, z=z, n=n) for b in (64, 256)}
times = {b: [] for b in calls}
for _ in range(5):
    for b, c in calls.items():
        times[b].append(c.time())
med = {b: float(np.median(t)) for b, t in times.items()}
for b, t in med.items():
    print(f"BLOCK {b:3d}: {t:7.1f} us")
print(f"ratio {med[64] / med[256]:.2f}")
```

```text
BLOCK  64:   328.3 us
BLOCK 256:   328.6 us
ratio 1.00
```

The two are the same, within the noise: this kernel's time is the
memory's, whatever the workgroup size. A difference of a few percent
between two medians is within the noise of one machine; a factor, like
the steps of part 6, is not. The `metron`
harness in `python/metron` does this properly: randomized rounds,
restored inputs between runs, a correctness check of each candidate.

## Inside a kernel: laps

`Bound.time` sees a launch as a whole. To see where the time goes inside
one, the kernel can read its own clock. `thin.lib.Laps` declares named
points ("laps"); each wave reads the shader clock at each, and stores its
readings in a buffer the declaration adds to the arguments. The host
decodes them. `build_vadd(laps=True)` takes three laps: `start` before
the loads, `load` when both have landed, `store` after the store.

The emulator models the clock, so a run there shows the mechanism (its
cycle counts are the model's, not a GPU's):

```python
kern = thin.compile(build_vadd, laps=True)
n = 4096
x, y = np.ones(n, np.float32), np.ones(n, np.float32)
z = np.zeros(n, np.float32)
call = kern(x=x, y=y, z=z, n=n)
call.emulate()
clock = kern.definition.meta["laps"]
print(clock.decode(call.owned["laps"], call.grid[0]).summary())
```

```text
laps (first): 128 waves, 1 rounds a wave (median); clock 2000 MHz (waves 2000-2000, short spans: ±4%)
  start        taken, with no lap before it to measure from
  load                  100 clk (      50.0 ns)  p10 100  p90 100
  store                 100 clk (      50.0 ns)  p10 100  p90 100
```

On the GPU, the same summary in real cycles:

```python gpu
n = 1 << 24
x, y, z = (torch.randn(n, device="cuda") for _ in range(3))
call = kern(x=x, y=y, z=z, n=n)
call.run()
torch.cuda.synchronize()
print(clock.decode(call.owned["laps"].cpu().numpy(), call.grid[0]).summary())
```

```text
laps (first): 524288 waves, 1 rounds a wave (median); clock 3355 MHz (waves 3-6158532, short spans: ±6.2%)
  start        taken, with no lap before it to measure from
  load                  631 clk (     188.0 ns)  p10 259  p90 801
  store                  27 clk (       8.0 ns)  p10 27  p90 29
```

Each lap is the time since the one before, per wave (`start`, the first,
has none before it): `load` is from the
loads' issue to their data, `store` the add and the store's issue. The
median and the 10th and 90th percentiles over every wave show how much
the waves differ: memory latency is not one number, and the slowest
waves set the kernel's time. The clock line says how fast the shader
clock ran, measured against the GPU's constant 100 MHz clock: over every
wave's span together, which is good to a fraction of a percent. One wave's
own span is a few hundred nanoseconds, a handful of ticks of the 100 MHz
clock, so the per-wave estimates in parentheses range widely; the summary
says how short the spans were.

Laps cost a few instructions each and change the kernel they measure, a
little: compare a kernel's `time()` with and without them before trusting
small differences. With `laps=False` the same generator builds the kernel
without any of it.
