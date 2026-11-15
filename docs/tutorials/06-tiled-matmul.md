# 6. A tiled matrix multiply

`C = A · B` for 4096 × 4096 matrices, A and B of f16, C of f32, built from
part 5's WMMA tile. We go in four steps, measuring each: one wave a tile
reading straight from memory; a workgroup tile staged through the LDS; an
LDS layout without bank conflicts; and double buffering, which overlaps
the loads of one step with the arithmetic of the last.

As in part 5, B is given transposed (`bt`, N × K): every fragment is then
16 contiguous bytes. Every size is a build parameter, so each kernel is
compiled for its matrices and every offset is a constant.

```python
import numpy as np
import thin
from thin import KernelDef
from thin.dsl import Args, O, ptr

WMMA = "v_wmma_f32_16x16x16_f16"


def lane_rh(b, k):
    """(r, h) of the lane: it holds row (A) or column (B, C) r, half h of K."""
    V = b.v
    return V.u32(k.lane & 15, name="r"), V.u32(k.lane >> 4, name="h")


def store_tile(b, c, d, row, col, r, h, n):
    """Stores the 16 x 16 tile d at C[row:, col:] (C has n columns)."""
    at = b.v.u32(((row + 8 * h) * n + col + r) * 4, name="c_at")
    for e in range(8):
        b.emit(O.global_store(at, d[e], c, offset=e * n * 4))


def check(kern, grid, M, N, K):
    """Runs kern on the emulator on small integers: its C is exact."""
    rng = np.random.default_rng(0)
    A = rng.integers(-3, 4, (M, K)).astype(np.float16)
    B = rng.integers(-3, 4, (K, N)).astype(np.float16)
    C = np.zeros((M, N), np.float32)
    args = {"a": A, "bt": np.ascontiguousarray(B.T), "c": C}
    kern.bind(grid=grid, args=args).emulate()
    return np.array_equal(C, A.astype(np.float32) @ B.astype(np.float32))
```

## Step 1: a wave a tile

Each workgroup is one wave and computes one 16 × 16 tile of C: for each 16
of K it loads its fragments of A and B from memory and accumulates one
WMMA.

```python
def matmul_global(M, N, K):
    def body(b, k):
        V = b.v
        a, bt, c = k.load()
        r, h = lane_rh(b, k)
        row, col = k.wgid_y * 16, k.wgid * 16
        a_at = V.u32(((row + r) * K + 8 * h) * 2, name="a_at")
        b_at = V.u32(((col + r) * K + 8 * h) * 2, name="b_at")
        d = V.f32(0.0, n=8, name="d")
        fa, fb = V.u32(n=4, name="fa"), V.u32(n=4, name="fb")
        for _ in b.loop(K // 16):
            b.emit(O.global_load(fa, a_at, a))
            b.emit(O.global_load(fb, b_at, bt))
            b.wait(fa, fb)
            b.emit(O.op(WMMA, d, fa, fb, d))
            a_at += 32  # the next 16 of K: 32 bytes along the row
            b_at += 32
        store_tile(b, c, d, row, col, r, h, N)

    grid = (N // 16, M // 16)
    return KernelDef(
        "matmul_global", Args(a=ptr, bt=ptr, c=ptr), body, wgsize=32, wg=2, max_grid=grid
    )


print(check(thin.compile(matmul_global, M=32, N=32, K=64), (2, 2), 32, 32, 64))
```

```text
True
```

On the GPU, against torch's own matrix multiply:

```python gpu
import torch

M = N = K = 4096
A = torch.randn(M, K, device="cuda").half()
B = torch.randn(K, N, device="cuda").half()
bt = B.t().contiguous()
want = A.float() @ B.float()


def measure(name, gen, grid, **params):
    C = torch.zeros(M, N, device="cuda")
    kern = thin.compile(gen, M=M, N=N, K=K, **params)
    us = kern.bind(grid=grid, args={"a": A, "bt": bt, "c": C}).time(10)
    err = (C - want).abs().max().item()
    print(f"{name:12s} {us:8.1f} us {2 * M * N * K / us / 1e6:6.1f} TFLOP/s  max error {err:.4f}")


measure("global", matmul_global, (N // 16, M // 16))
```

```text
global        19607.9 us    7.0 TFLOP/s  max error 0.0027
```

The error is the rounding of f32 sums of 4096 terms: every version below
gives the same one, since they all add in the same order. The speed is
poor, for a simple reason: every fragment is read from memory, and each
is used once. A row of A is read again for each of the 256 tiles along
it.

## Step 2: a workgroup tile through the LDS

Data loaded once should be used many times. Let a workgroup of 4 waves
compute a 64 × 64 tile of C. For each step of 32 along K, it copies a
64 × 32 tile of A and of bt into the LDS (each of 128 lanes moves 32
bytes of each); then each wave computes its 32 × 32 quarter of C as
2 × 2 WMMA tiles, reading its fragments from the LDS. Each fragment of A
it reads serves two WMMAs, and each 16-byte piece copied from memory is
read by two waves.

```python
BM = BN = 64  # a workgroup's tile of C
BK = 32  # the K a step stages


class Tiles:
    """The LDS stages of A and bt, and the lane's places in them."""

    def __init__(self, b, k, stages, pad, K):
        V = b.v
        L = b.lds()
        pitch = {1: BK * 2 + pad}  # a row's bytes: 64, plus pad
        self.As = L.region("As", (stages, BM, BK), "f16", pitch=pitch)
        self.Bs = L.region("Bs", (stages, BN, BK), "f16", pitch=pitch)
        t = k.tid
        # The copy: 64 rows of 64 bytes are 256 pieces of 16 bytes, two a
        # lane: rows t // 4 and t // 4 + 32, piece t % 4.
        self.put_a = self.As.pointer(0, t >> 2, (t & 3) * 8, name="put_a")
        self.put_b = self.Bs.pointer(0, t >> 2, (t & 3) * 8, name="put_b")
        row, col = k.wgid_y * BM, k.wgid * BN
        self.a_at = V.u32(((row + (t >> 2)) * K + (t & 3) * 8) * 2, name="a_at")
        self.b_at = V.u32(((col + (t >> 2)) * K + (t & 3) * 8) * 2, name="b_at")
        self.ga, self.gb = V.u32(n=8, name="ga"), V.u32(n=8, name="gb")
        self.K = K
        # The fragments: wave w computes quarter (w // 2, w % 2) of the tile.
        self.r, self.h = lane_rh(b, k)
        self.wr, self.wc = k.wave >> 1, k.wave & 1
        self.get_a = self.As.pointer(0, self.wr * 32 + self.r, 8 * self.h, name="get_a")
        self.get_b = self.Bs.pointer(0, self.wc * 32 + self.r, 8 * self.h, name="get_b")

    def fetch(self, b, a, bt):
        """Issues the loads of the next step's pieces."""
        for j in range(2):
            o = j * 32 * self.K * 2  # 32 rows on
            b.emit(O.global_load(self.ga[4 * j : 4 * j + 4], self.a_at, a, offset=o))
            b.emit(O.global_load(self.gb[4 * j : 4 * j + 4], self.b_at, bt, offset=o))
        self.a_at += BK * 2
        self.b_at += BK * 2

    def put(self, b, s):
        """Stores the fetched pieces into stage s."""
        b.wait(self.ga, self.gb)
        for j in range(2):
            b.emit(O.ds_write(self.put_a.at(s, 32 * j, 0), self.ga[4 * j : 4 * j + 4]))
            b.emit(O.ds_write(self.put_b.at(s, 32 * j, 0), self.gb[4 * j : 4 * j + 4]))

    def compute(self, b, s, acc, fa, fb):
        """The step's WMMAs from stage s: two 16s of K, 2 x 2 tiles each."""
        for kk in range(BK // 16):
            for i in range(2):
                b.emit(O.ds_read(fa[i], self.get_a.at(s, 16 * i, 16 * kk)))
                b.emit(O.ds_read(fb[i], self.get_b.at(s, 16 * i, 16 * kk)))
            b.wait(fa, fb)
            for i in range(2):
                for j in range(2):
                    b.emit(O.op(WMMA, acc[i][j], fa[i], fb[j], acc[i][j]))

    def store(self, b, k, c, acc, N):
        row, col = k.wgid_y * BM + self.wr * 32, k.wgid * BN + self.wc * 32
        for i in range(2):
            for j in range(2):
                store_tile(b, c, acc[i][j], row + 16 * i, col + 16 * j, self.r, self.h, N)


def registers(b):
    V = b.v
    acc = [[V.f32(0.0, n=8, name=f"acc{i}{j}") for j in range(2)] for i in range(2)]
    fa = [V.u32(n=4, name=f"fa{i}") for i in range(2)]
    fb = [V.u32(n=4, name=f"fb{j}") for j in range(2)]
    return acc, fa, fb
```

A few things to notice. The region `As` has a dimension for the stages
(one, until step 4). `self.As.pointer(0, t >> 2, (t & 3) * 8)` is a value
holding the address of the lane's first element; `put_a.at(s, 32 * j,
0)` moves it by constant indices, which become the instruction's offset
field: one address register serves every stage and both pieces. `k.wave`
is the wave's index in the workgroup, a scalar, so `wr` and `wc` are
scalars too, and `get_a`'s address mixes them with the lane's `r` and `h`.

The kernel itself is short:

```python
def matmul_lds(M, N, K, pad=0):
    def body(b, k):
        a, bt, c = k.load()
        t = Tiles(b, k, 1, pad, K)
        acc, fa, fb = registers(b)
        for _ in b.loop(K // BK):
            t.fetch(b, a, bt)
            t.put(b, 0)
            b.wait(t.As, t.Bs)
            b.barrier()  # the tiles are in the LDS: compute
            t.compute(b, 0, acc, fa, fb)
            b.barrier()  # every wave is done with them: the next step may overwrite
        t.store(b, k, c, acc, N)

    grid = (N // BN, M // BM)
    return KernelDef(
        "matmul_lds", Args(a=ptr, bt=ptr, c=ptr), body, wgsize=128, wg=2, max_grid=grid
    )


print(check(thin.compile(matmul_lds, M=64, N=64, K=128), (1, 1), 64, 64, 128))
```

```text
True
```

Two barriers a step: one so that no wave reads the tiles before every
wave has stored its pieces, one so that no wave overwrites them while
another still reads. Leave out the second and the compiler refuses the
kernel, as in part 4. Here is one step's computation, as compiled:

```python
kern = thin.compile(matmul_lds, M=64, N=64, K=128)
body = kern.asm.splitlines()
start = next(i for i, x in enumerate(body) if "ds_load_b128" in x)
print("\n".join(body[start : start + 16]))
```

```asm
	ds_load_b128 v[40:43], v5                                    // As: get_a + 0
	ds_load_b128 v[44:47], v0                                    // Bs: get_b + 0
	ds_load_b128 v[48:51], v5 offset:1024                        // As: get_a + 1024
	ds_load_b128 v[52:55], v0 offset:1024                        // Bs: get_b + 1024
	s_wait_dscnt 0
	v_wmma_f32_16x16x16_f16 v[8:15], v[40:43], v[44:47], v[8:15]
	v_wmma_f32_16x16x16_f16 v[16:23], v[40:43], v[52:55], v[16:23]
	v_wmma_f32_16x16x16_f16 v[24:31], v[48:51], v[44:47], v[24:31]
	v_wmma_f32_16x16x16_f16 v[32:39], v[48:51], v[52:55], v[32:39]
	ds_load_b128 v[40:43], v5 offset:32                          // As: get_a + 32
	ds_load_b128 v[44:47], v0 offset:32                          // Bs: get_b + 32
	ds_load_b128 v[48:51], v5 offset:1056                        // As: get_a + 1056
	ds_load_b128 v[52:55], v0 offset:1056                        // Bs: get_b + 1056
	s_wait_dscnt 0
	v_wmma_f32_16x16x16_f16 v[8:15], v[40:43], v[44:47], v[8:15]
	v_wmma_f32_16x16x16_f16 v[16:23], v[40:43], v[52:55], v[16:23]
```

The four fragment reads go out together (the A ones read the same
address register, `get_a`, at offsets 0 and 1024: rows 16 apart), one
wait, four WMMAs.

```python gpu
measure("lds", matmul_lds, (N // BN, M // BM))
```

```text
lds            3267.2 us   42.1 TFLOP/s  max error 0.0027
```

## Step 3: the banks

The fragment reads are 16 bytes a lane, and the 16 lanes of a half read
16 rows of `As`, 64 bytes apart: part 4's column read again, with the
same bank conflicts. Pad each row by 16 bytes (one fragment) so that
consecutive rows start in different banks:

```python gpu
measure("lds, pad", matmul_lds, (N // BN, M // BM), pad=16)
```

```text
lds, pad       3178.9 us   43.2 TFLOP/s  max error 0.0027
```

Hardly better: in this kernel the waves spend most of their time
waiting, at the barriers, for the tile's loads from memory. Each step
loads, waits, computes, and only then starts the next step's loads. The
conflicts will matter once that waiting is gone.

## Step 4: double buffering

Give the LDS two stages. While the waves compute from one, the next
step's tiles load from memory and are stored into the other: the loads'
latency overlaps the WMMAs, and one barrier a step is enough.

```python
def matmul_double(M, N, K, pad=0):
    def body(b, k):
        a, bt, c = k.load()
        t = Tiles(b, k, 2, pad, K)
        acc, fa, fb = registers(b)

        def step(s, prefetch):
            if prefetch:
                t.fetch(b, a, bt)  # the next step's tiles: in flight while
            t.compute(b, s, acc, fa, fb)  # this step computes from stage s
            if prefetch:
                t.put(b, 1 - s)  # into the other stage
                b.wait(t.As, t.Bs)
            b.barrier()

        t.fetch(b, a, bt)  # the first step's tiles
        t.put(b, 0)
        b.wait(t.As, t.Bs)
        b.barrier()
        for _ in b.loop(K // (2 * BK) - 1):  # two steps a trip: stages 0, 1
            step(0, True)
            step(1, True)
        step(0, True)  # the last two steps: the last prefetches nothing
        step(1, False)
        t.store(b, k, c, acc, N)

    grid = (N // BN, M // BM)
    return KernelDef(
        "matmul_double", Args(a=ptr, bt=ptr, c=ptr), body, wgsize=128, wg=2, max_grid=grid
    )


print(check(thin.compile(matmul_double, M=64, N=64, K=128), (1, 1), 64, 64, 128))
print(check(thin.compile(matmul_double, M=64, N=128, K=256, pad=16), (2, 1), 64, 128, 256))
```

```text
True
True
```

Why one barrier is enough: step `t` reads stage `s` and writes stage
`1 - s`. The last reads of stage `1 - s` were in step `t - 1`, before the
barrier that ended it; the writes of stage `s` that step `t` reads were
in step `t - 1` too, before the same barrier. The compiler checks this:
it reads the barriers as epochs and proves that no wave's access in an
epoch overlaps another wave's write in it.

The loop body is two steps, so that each stage's index is a constant
(an offset field), and the last two steps are written out after the
loop, so that nothing is fetched past the end of K.

```python gpu
measure("double", matmul_double, (N // BN, M // BM))
measure("double, pad", matmul_double, (N // BN, M // BM), pad=16)
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
torch.matmul(A, B)
t0.record()
for _ in range(10):
    torch.matmul(A, B)
t1.record()
t1.synchronize()
us = t0.elapsed_time(t1) * 100
print(f"{'torch':12s} {us:8.1f} us {2 * M * N * K / us / 1e6:6.1f} TFLOP/s")
```

```text
double         3144.9 us   43.7 TFLOP/s  max error 0.0027
double, pad    1616.7 us   85.0 TFLOP/s  max error 0.0027
torch          1102.4 us  124.7 TFLOP/s
```

With the loads overlapped, the waves wait less at the barriers, and the
bank conflicts are now what holds them up: padding nearly doubles the
speed. From here a faster kernel would compute larger tiles a wave
(more WMMAs per fragment read), keep more steps in flight, and order
the workgroups so that neighbours share their tiles in the cache. Each
is a change to the code above, and part 7 shows how to measure where
the time goes.
