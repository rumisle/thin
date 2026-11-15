# 4. Shared memory

The lanes of a workgroup can share data through the *LDS* (local data
share): 64 KiB of fast memory per workgroup, which every wave of the
workgroup reads and writes. It is how a tile of data loaded once is
used by many lanes. This part builds a matrix transpose through the LDS,
meets the two rules that come with sharing (waves run apart, and an LDS
access takes time to land), and measures what the layout of the LDS
costs.

```python
import numpy as np
import thin
from thin import KernelDef
from thin.dsl import Args, O, cvt_f32, ptr

TILE = 32
```

## A transpose

Each workgroup of 256 lanes (8 waves) transposes one 32 × 32 tile. Lane
`t` handles column `c = t % 32` of rows `r, r + 8, r + 16, r + 24`, where
`r = t // 32` is its wave. The lanes read the tile row by row from
global memory (each wave reads 32 consecutive floats: one 128-byte line)
and store it into the LDS; then they read it back column by column and
write it, again row by row, to the transposed position.

```python
def transpose(n):
    def body(b, k):
        V = b.v
        src, dst = k.load()
        T = b.lds().region("T", (TILE, TILE), "f32")
        c = V.u32(k.tid & 31, name="c")
        r = V.u32(k.tid >> 5, name="r")
        tx, ty = k.wgid, k.wgid_y  # the tile's column and row
        x = V.f32(n=4, name="x")
        at = V.u32(((ty * TILE + r) * n + tx * TILE + c) * 4, name="at")
        for j in range(4):
            b.emit(O.global_load(x[j], at, src, offset=j * 8 * n * 4))
        for j in range(4):
            b.wait(x[j])
            b.emit(O.ds_write(T.at(r + 8 * j, c), x[j]))
        b.wait(T)
        b.barrier()
        y = V.f32(n=4, name="y")
        for j in range(4):
            b.emit(O.ds_read(y[j], T.at(c, r + 8 * j)))
        at2 = V.u32(((tx * TILE + r) * n + ty * TILE + c) * 4, name="at2")
        for j in range(4):
            b.wait(y[j])
            b.emit(O.global_store(at2, y[j], dst, offset=j * 8 * n * 4))

    g = n // TILE
    return KernelDef("transpose", Args(src=ptr, dst=ptr), body, wgsize=256, wg=2, max_grid=(g, g))
```

`b.lds()` makes the kernel's LDS *layout* and `region("T", (32, 32),
"f32")` declares a region in it: a 32 × 32 array of floats. An LDS
address is always a region's: `T.at(r, c)` is element `(r, c)` of `T`,
and thin computes its byte offset (`r * 128 + c * 4`). `wg=2` gives the
kernel two workgroup ids, `k.wgid` and `k.wgid_y`, for a 2-D grid of
tiles.

```python
n = 64
a = np.arange(n * n, dtype=np.float32).reshape(n, n)
out = np.zeros_like(a)
kern = thin.compile(transpose, n=n)
kern.bind(grid=(n // TILE, n // TILE), args={"src": a, "dst": out}).emulate()
print(np.array_equal(out, a.T))
```

```text
True
```

## Waves run apart

The two lines before the column reads are what make the sharing work.
The 8 waves of a workgroup are 8 programs running at their own pace:
when wave 1 reads column `c` of the tile, the rows it reads were written
by all 8 waves, and some of them may not have got there yet.
`b.barrier()` makes every wave of the workgroup wait until all have
reached it. Leave it out, and thin refuses the kernel:

```python
def transpose_nobarrier(n):
    def body(b, k):
        V = b.v
        src, dst = k.load()
        T = b.lds().region("T", (TILE, TILE), "f32")
        c = V.u32(k.tid & 31, name="c")
        r = V.u32(k.tid >> 5, name="r")
        x = V.f32(name="x")
        b.emit(O.global_load(x, V.u32((r * n + c) * 4, name="at"), src))
        b.wait(x)
        b.emit(O.ds_write(T.at(r, c), x))
        y = V.f32(name="y")
        b.emit(O.ds_read(y, T.at(c, r)))
        b.wait(y)
        b.emit(O.global_store(V.u32((r * n + c) * 4, name="at2"), y, dst))

    return KernelDef("transpose", Args(src=ptr, dst=ptr), body, wgsize=256, max_grid=(1,))


thin.compile(transpose_nobarrier, n=32)
```

```error
MemError: `ds_store_b32 T + (r * 128) + (c * 4)` (instruction 13) in wave 0 and `ds_load_b32 T + (c * 128) + (r * 4)` (instruction 17) in wave 1 may touch the same bytes ([0, 128) and [4, 3976)) in the epoch 'entry', a write among them: the waves run apart there, either may go first (a barrier between them, or each wave's bytes its own; a block only some waves run: b.waves([...]) - a branch on the wave index is not read as one)
```

The compiler knows which bytes each wave touches (they follow from
`k.tid`, the workgroup size and the layout), and wave 0's store and wave
1's load overlap with nothing between them to order them: a *race*.

A barrier orders the waves, but not the LDS operations they have issued:
an LDS store, like a load from memory, lands some time after it is
issued. A wave may pass the barrier while its store is still on its way.
So the wait goes before the barrier: `b.wait(T)` waits for every LDS
operation on `T`. Without it:

```python
def transpose_nowait(n):
    def body(b, k):
        V = b.v
        src, dst = k.load()
        T = b.lds().region("T", (TILE, TILE), "f32")
        c = V.u32(k.tid & 31, name="c")
        r = V.u32(k.tid >> 5, name="r")
        x = V.f32(name="x")
        b.emit(O.global_load(x, V.u32((r * n + c) * 4, name="at"), src))
        b.wait(x)
        b.emit(O.ds_write(T.at(r, c), x))
        b.barrier()
        y = V.f32(name="y")
        b.emit(O.ds_read(y, T.at(c, r)))
        b.wait(y)
        b.emit(O.global_store(V.u32((r * n + c) * 4, name="at2"), y, dst))

    return KernelDef("transpose", Args(src=ptr, dst=ptr), body, wgsize=256, max_grid=(1,))


thin.compile(transpose_nowait, n=32)
```

```error
MemError: `ds_store_b32 T + (r * 128) + (c * 4)` (instruction 13) is in flight at the barrier (instruction 15), and `ds_load_b32 T + (c * 128) + (r * 4)` (instruction 19) accesses its bytes after it: another wave may run it before the op lands (wait before the barrier: b.wait(T))
```

## A race only a run can show

The compiler can prove the absence of a race only where it knows the
addresses. When they come from data, it cannot. Here each lane stores
its value at a position read from a permutation, then reads its own
position back:

```python
def scatter(barrier):
    def body(b, k):
        V = b.v
        src, perm, dst = k.load()
        T = b.lds().region("T", (256,), "f32")
        off = V.u32(k.tid * 4, name="off")
        x, p = V.f32(name="x"), V.u32(name="p")
        b.emit(O.global_load(x, off, src))
        b.emit(O.global_load(p, off, perm))
        b.wait(x, p)
        b.assume(p, lo=0, hi=255, why="perm is a permutation of 0 .. 255")
        b.emit(O.ds_write(T + p * 4, x, ordered="perm is a permutation: one write an element"))
        if barrier:
            b.wait(T)
            b.barrier()
        y = V.f32(name="y")
        b.emit(O.ds_read(y, T.at(k.tid)))
        b.wait(y)
        b.emit(O.global_store(off, y, dst))

    return KernelDef("scatter", Args(src=ptr, perm=ptr, dst=ptr), body, wgsize=256, max_grid=(1,))
```

Two claims tell the compiler what it cannot see: `b.assume` that `p` is
in range (so the store stays inside `T`), and `ordered=` that no two
lanes store to the same element. With those, it compiles both versions.
The emulator then runs them, and it tracks, byte by byte, which wave
wrote and read what since the last barrier:

```python
x = np.arange(256, dtype=np.float32)
perm = np.random.default_rng(0).permutation(256).astype(np.uint32)
z = np.zeros_like(x)
call = thin.compile(scatter, barrier=False).bind(grid=1, args={"src": x, "perm": perm, "dst": z})
call.emulate()
```

```error
DeviceFault: scatter: wave 1 ds_store_b32 v4, v2: writes LDS byte 20 that wave 0's `ds_load_b32 v0, v2` read, since the last barrier (none between the read and the write): the write may land before the read (a WAR race)
```

Wave 0 got ahead and read its element (bytes 20 to 23, element 5)
before wave 1 had stored it: wave 0 read whatever was there before. The
message names the instruction and both waves. With the barrier:

```python
want = np.zeros_like(x)
want[perm] = x
call = thin.compile(scatter, barrier=True).bind(grid=1, args={"src": x, "perm": perm, "dst": z})
call.emulate()
print(np.array_equal(z, want))
```

```text
True
```

The emulator runs one interleaving of the waves per call, so a race may
take some schedules to show (`emulate(sched=...)` picks others); but
when it shows, it is reported where it happens.

## Banks

The LDS is divided into *banks*, each serving one 4-byte word per cycle;
consecutive words are in consecutive banks. When the 32 lanes of one
instruction touch words in different banks they are served at once;
words in the same bank are served one after another, a *bank conflict*.

The transpose reads columns: lane `c` reads `T[c][r]`, and the rows are
32 words apart, so all 32 lanes hit the same bank. The usual fix is to
pad each row by one word: rows are then 33 words apart, and the 32 lanes
of a column read fall in 32 different banks. thin's emulator has no bank
model for gfx1201, so we measure. This kernel does nothing but read
columns of a 256 × 32 region, 64 times over:

```python
def columns(pad):
    def body(b, k):
        V = b.v
        (out,) = k.load()
        T = b.lds().region("T", (256, 32), "f32", pitch=4 * (32 + pad))
        lane = V.u32(k.tid, name="lane")
        for j in range(32):
            b.emit(O.ds_write(T.at(lane, j), V.f32(cvt_f32(lane + j), name="w")))
        b.wait(T)
        b.barrier()
        acc = V.f32(0.0, name="acc")
        v = V.f32(n=32, name="v")
        p = T.pointer(lane, 0, name="p")  # lane's row: one address for 32 reads
        for it in b.loop(64):
            for j in range(32):
                b.emit(O.ds_read(v[j], p.at(0, j)))
            b.wait(v)
            for j in range(32):
                acc += v[j]
        b.emit(O.global_store(V.u32((k.wgid * 256 + k.tid) * 4, name="o"), acc, out))

    return KernelDef("columns", Args(out=ptr), body, wgsize=256, max_grid=(1 << 12,))
```

`pitch=` sets a row's bytes: 128 (32 words) or 132 (33). `T.pointer` makes
a value holding one element's address, and `p.at(0, j)` is element `j`
further along the row: the offset `4 j` goes into the instruction, so the
32 reads share one address register. On the emulator both versions
compute the same sums:

```python
for pad in (0, 1):
    out = np.zeros(256, np.float32)
    thin.compile(columns, pad=pad).bind(grid=1, args={"out": out}).emulate()
    print(pad, out[:2], [64 * sum(lane + j for j in range(32)) for lane in (0, 1)])
```

```text
0 [31744. 33792.] [31744, 33792]
1 [31744. 33792.] [31744, 33792]
```

On the GPU, 4096 workgroups:

```python gpu
import torch

for pad in (0, 1):
    og = torch.zeros(4096 * 256, device="cuda")
    us = thin.compile(columns, pad=pad).bind(grid=4096, args={"out": og}).time()
    print(f"pitch {4 * (32 + pad)} B: {us:.1f} us")
```

```text
pitch 128 B: 19681.9 us
pitch 132 B: 646.9 us
```

One word of padding a row, and the reads are many times faster. The
transpose above has the same column read,
but it reads each element once, and its time goes to global memory: its
LDS layout is not where its time goes. In the matrix multiply of part 6,
where each LDS element is read many times, it is.
