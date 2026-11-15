# 5. Matrix cores

gfx1201 has instructions that multiply small matrices: one
`v_wmma_f32_16x16x16_f16` computes `D = A·B + C` for a 16 × 16 tile, with A
and B of f16 (half precision) and C and D of f32, in one wave. That is
4096 multiply-adds in one instruction, where a `v_fma_f32` does 32 (one
a lane). This part computes one tile with it and checks it against numpy;
part 6 builds a whole matrix multiply out of tiles.

```python
import numpy as np
import thin
from thin import KernelDef
from thin.dsl import Args, O, ptr


def lines(kern):
    out = kern.asm.splitlines()
    start = out.index(f"{kern.name}:") + 1
    return out[start : out.index("\ts_endpgm", start) + 1]
```

## Fragments

A WMMA's operands are spread over the wave's 32 lanes: each lane holds a
*fragment* of each matrix, in its own registers. For lane `l`, write
`l = 16h + r`, so `r = l % 16` and `h = l // 16` (the lane's position in
its 16-lane half, and which half):

- **A** (16 × 16 f16): lane `l` holds row `r`, elements `k = 8h .. 8h + 7`.
  Eight f16 are 16 bytes: 4 VGPRs, two values each.
- **B** (16 × 16 f16): lane `l` holds column `r`, the same `k`.
- **C, D** (16 × 16 f32): lane `l` holds column `r`, rows `8h .. 8h + 7`:
  8 VGPRs, one value each.

Lanes 0 to 15 hold the first half of K of every row of A (and column of
B), lanes 16 to 31 the second half.

A row of A is contiguous in memory (row-major), but a column of B is
not. So we store B transposed, `bt = B.T` (16 × 16, row-major): column `r`
of B is row `r` of `bt`, and both fragments are 16 contiguous bytes, one
load each.

## One tile

```python
WMMA = "v_wmma_f32_16x16x16_f16"


def wmma_tile():
    def body(b, k):
        V = b.v
        a, bt, c = k.load()
        r = V.u32(k.lane & 15, name="r")
        h = V.u32(k.lane >> 4, name="h")
        at = V.u32((r * 16 + 8 * h) * 2, name="at")  # row r, k = 8h: its byte
        fa = V.u32(n=4, name="fa")
        fb = V.u32(n=4, name="fb")
        b.emit(O.global_load(fa, at, a))  # A's fragment, 16 bytes
        b.emit(O.global_load(fb, at, bt))  # B's, from row r of bt
        b.wait(fa, fb)
        d = V.f32(n=8, name="d")
        b.emit(O.op(WMMA, d, fa, fb, 0))  # D = A·B + 0
        out = V.u32(((8 * h) * 16 + r) * 4, name="out")  # D[8h][r]
        for e in range(8):
            b.emit(O.global_store(out, d[e], c, offset=e * 16 * 4))  # D[8h + e][r]

    return KernelDef("wmma_tile", Args(a=ptr, bt=ptr, c=ptr), body, wgsize=32, max_grid=(1,))


kern = thin.compile(wmma_tile)
print("\n".join(lines(kern)))
```

```asm
	s_load_b128 s[4:7], s[0:1], 0x0                              // a, bt
	s_load_b64 s[2:3], s[0:1], 0x10                              // c
	v_and_b32 v1, 15, v0                                         // r = tid & 15
	v_lshrrev_b32 v2, 4, v0                                      // h = tid >> 4
	s_delay_alu 2                                                // hazard: instid0(VALU_DEP_2)
	v_lshlrev_b32 v0, 4, v1                                      // at = ((r * 16) + (8 * h)) * 2
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v3, v2, 3, v0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshlrev_b32 v0, 1, v3
	s_wait_kmcnt 0                                               // kernel arguments
	global_load_b128 v[4:7], v0, s[4:5]
	global_load_b128 v[8:11], v0, s[6:7]
	s_wait_loadcnt 0
	v_wmma_f32_16x16x16_f16 v[12:19], v[4:7], v[8:11], 0
	v_lshlrev_b32 v0, 3, v2                                      // out = (((8 * h) * 16) + r) * 4
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v2, v0, 4, v1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshlrev_b32 v0, 2, v2
	global_store_b32 v0, v12, s[2:3]
	global_store_b32 v0, v13, s[2:3] offset:64
	global_store_b32 v0, v14, s[2:3] offset:128
	global_store_b32 v0, v15, s[2:3] offset:192
	global_store_b32 v0, v16, s[2:3] offset:256
	global_store_b32 v0, v17, s[2:3] offset:320
	global_store_b32 v0, v18, s[2:3] offset:384
	global_store_b32 v0, v19, s[2:3] offset:448
	s_endpgm
```

`V.u32(n=4)` is a value of four registers: `global_load` sees its width and
makes a 16-byte `global_load_b128`. The WMMA is emitted like any other
instruction, by its mnemonic; `0` for C is the inline constant: no
accumulator to start from. Each lane then stores its 8 elements of D,
column `r`, rows `8h` to `8h + 7`, 64 bytes apart (a row of C is 16
floats).

## Checking against numpy

Small integers first: their products and sums are exact in f32, so the
result must match to the bit.

```python
rng = np.random.default_rng(0)
A = rng.integers(-4, 5, (16, 16)).astype(np.float16)
B = rng.integers(-4, 5, (16, 16)).astype(np.float16)
C = np.zeros((16, 16), np.float32)
kern.bind(grid=1, args={"a": A, "bt": np.ascontiguousarray(B.T), "c": C}).emulate()
want = A.astype(np.float32) @ B.astype(np.float32)
print(np.array_equal(C, want))
```

```text
True
```

With random values the products are still exact (an f16 product fits in
an f32), but the 16 sums round, and the hardware adds in its own order:

```python
A = rng.standard_normal((16, 16)).astype(np.float16)
B = rng.standard_normal((16, 16)).astype(np.float16)
kern.bind(grid=1, args={"a": A, "bt": np.ascontiguousarray(B.T), "c": C}).emulate()
want = A.astype(np.float64) @ B.astype(np.float64)
print(np.abs(C - want).max() < 1e-5, np.array_equal(C, want.astype(np.float32)))
```

```text
True False
```

The emulator's WMMA reproduces the GPU's rounding, measured on the
hardware; on the GPU the same call gives the same bits:

```python gpu
import torch

C_emu = C.copy()
tensors = {
    "a": torch.from_numpy(A).cuda(),
    "bt": torch.from_numpy(np.ascontiguousarray(B.T)).cuda(),
}
tensors["c"] = torch.zeros(16, 16, device="cuda")
kern.bind(grid=1, args=tensors).run()
torch.cuda.synchronize()
print(np.array_equal(tensors["c"].cpu().numpy(), C_emu))
```

```text
True
```

## Accumulating

A WMMA's C can be the D of the one before: `d = A₀·B₀ + d` then
`d = A₁·B₁ + d` accumulates two products, which is how a longer K is
summed 16 at a time. The hardware forwards the accumulator from one WMMA
to the next; reading a WMMA's D as an A or B operand of the next one is
different, and needs a wait the compiler inserts:

```python
def chain():
    def body(b, k):
        V = b.v
        fa, fb = V.u32(n=4, name="fa", at=8), V.u32(n=4, name="fb", at=12)
        d = V.f32(n=8, name="d", at=16)
        h = V.u32(n=4, name="h", at=24)
        b.emit(O.op(WMMA, d, fa, fb, d))  # accumulate into d
        b.emit(O.op(WMMA, d, fa, fb, d))  # and again
        b.emit(O.op("v_wmma_f16_16x16x16_f16", h, fa, fb, 0))  # an f16 result
        b.emit(O.op("v_wmma_f16_16x16x16_f16", fa, h, fb, fa))  # read as A

    return KernelDef("chain", Args(), body, wgsize=32, max_grid=(1,))


print("\n".join(lines(thin.compile(chain))))
```

```asm
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[12:15], v[16:23]
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[12:15], v[16:23]
	v_wmma_f16_16x16x16_f16 v[24:27], v[8:11], v[12:15], 0
	v_nop                                                        // hazard
	v_wmma_f16_16x16x16_f16 v[8:11], v[24:27], v[12:15], v[8:11]
	s_endpgm
```

`v_nop // hazard`: the second f16 WMMA reads `h` as its A operand, and
the ISA requires a cycle between the two. (`at=` pins the inputs to
registers here, so that this fragment of a kernel compiles on its own.)
