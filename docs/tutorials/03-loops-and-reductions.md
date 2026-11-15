# 3. Loops and reductions

A row sum: `out[r] = x[r, 0] + x[r, 1] + ... + x[r, cols - 1]`. Each row
is summed by one wave. The wave's 32 lanes walk along the row 32 columns
at a time, each lane keeping its own partial sum, and at the end the 32
partial sums are added across the lanes. On the way we meet loops, waits
inside loops, and operations between lanes.

```python
import numpy as np
import thin
from thin import KernelDef
from thin.dsl import Args, O, across, ptr, u32


def lines(kern):
    out = kern.asm.splitlines()
    start = out.index(f"{kern.name}:") + 1
    return out[start : out.index("\ts_endpgm", start) + 1]
```

## A counted loop

The row length is an argument. To keep the loop simple we require it to
be a multiple of 32, and say so in its kind: `bind` refuses any other
value.

```python
COLS = u32.within(32, 1 << 17).multiple_of(32)


def rowsum():
    def body(b, k):
        V, S = b.v, b.s
        x, out, cols = k.load()
        trips = S.u32(cols >> 5, name="trips")  # cols / 32
        off = V.u32((k.wgid * cols + k.tid) * 4, name="off")
        acc = V.f32(0.0, name="acc")
        v = V.f32(name="v")
        for it in b.loop(trips):
            b.emit(O.global_load(v, off, x))
            b.wait(v)
            acc += v
            off += 32 * 4
        total = V.f32(across(acc, "add", 32), name="total")
        with b.lanes(k.tid == 0):
            b.emit(O.global_store(V.u32(k.wgid * 4, name="o"), total, out))

    args = Args(x=ptr, out=ptr, cols=COLS)
    return KernelDef("rowsum", args, body, wgsize=32, max_grid=(1 << 12,))
```

`for it in b.loop(trips):` is a Python loop that runs its body *once*:
the body is written once and becomes the loop's instructions; the GPU
runs them `trips` times. `it.i` would be the trip's index; this kernel
does not need it, because each trip moves `off` along instead. The
workgroup is one wave, 32 lanes: `k.tid` is the lane, and `k.wgid` the
row.

`max_grid=(1 << 12,)` says the kernel is launched with at most 4096
workgroups. With that and the range of `cols`, the compiler proves that
`off` never passes 2^32: give it a larger bound and it refuses the
kernel, because a 32-bit offset that may wrap reads the wrong memory.

```python
kern = thin.compile(rowsum)
body = lines(kern)
print("\n".join(body[: body.index(".Lloop_end_2:") + 1]))
```

```asm
	s_mov_b32 s2, ttmp9                                          // wgid#1: ttmp9
	s_load_b128 s[4:7], s[0:1], 0x0                              // x, out
	s_load_b32 s3, s[0:1], 0x10                                  // cols
	s_wait_kmcnt 0                                               // kernel arguments
	s_lshr_b32 s0, s3, 5                                         // trips = cols >> 5
	s_mul_i32 s1, s2, s3                                         // off = ((wgid * cols) + tid) * 4
	v_add_nc_u32 v1, s1, v0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshlrev_b32 v2, 2, v1
	v_mov_b32 v1, 0                                              // acc = 0.0
	s_mov_b32 s1, 0
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_cmp_lt_u32 s1, s0
	s_cbranch_scc0 .Lloop_end_2
.Lloop_1:
	global_load_b32 v3, v2, s[4:5]
	s_wait_loadcnt 0
	v_add_f32 v1, v1, v3                                         // acc = acc + v
	v_add_nc_u32 v2, 128, v2                                     // off = off + 128
	s_add_co_u32 s1, s1, 1
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_cmp_lt_u32 s1, s0
	s_cbranch_scc1 .Lloop_1
.Lloop_end_2:
```

The loop is the block from `s_mov_b32 s0, 0` to `.Lloop_end_2`. The
index starts at 0; since `trips` is only known when the kernel runs, a
compare and a branch skip the loop when it is 0. The body is ours: the
load, its wait, the add, the step. Then the index is incremented,
compared again, and `s_cbranch_scc1` jumps back while it is below
`trips`. These are scalar instructions: the loop is the same for every
lane of the wave, so it is run once per wave, not once per lane.

## Checking it

```python
rows, cols = 6, 96
rng = np.random.default_rng(1)
x = rng.standard_normal((rows, cols)).astype(np.float32)
out = np.zeros(rows, np.float32)
kern.bind(grid=rows, args={"x": x, "out": out, "cols": cols}).emulate()
print(np.abs(out - x.sum(axis=1)).max())
print(np.allclose(out, x.sum(axis=1), rtol=1e-5, atol=1e-5))
```

```text
1.9073486e-06
True
```

The sums agree, but not to the last bit: the kernel adds in a different
order than numpy (32 partial sums, then a tree across them), and float
addition is not associative. That is expected of any parallel sum.

## Adding across lanes

`across(acc, "add", 32)` adds the 32 lanes' values and gives the sum to
every lane. Lanes cannot read each other's registers directly; the
reduction is built from instructions that move values between lanes:

```python
print("\n".join(body[body.index(".Lloop_end_2:") + 1 :]))
```

```asm
	v_add_f32_dpp v2, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf // total = across(acc, add, 32)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32_dpp v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32_dpp v2, v2, v2 row_half_mirror row_mask:0xf bank_mask:0xf
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32_dpp v2, v2, v2 row_mirror row_mask:0xf bank_mask:0xf
	s_mov_b32 s0, 1985229328
	s_mov_b32 s1, 4275878552
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_permlanex16_b32 v1, v2, s0, s1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32 v2, v2, v1
	v_cmp_eq_u32 vcc_lo, 0, v0                                   // tid == 0
	s_mov_b32 exec_lo, vcc_lo
	s_lshl_b32 s0, s2, 2                                         // o = wgid * 4
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	v_mov_b32 v0, s0
	global_store_b32 v0, v2, s[6:7]
	s_mov_b32 exec_lo, -1
	s_endpgm
```

`v_add_f32_dpp` is an add whose first operand comes from another lane
(*DPP*, data-parallel primitives): `quad_perm:[1,0,3,2]` swaps
neighbours, so after the first add every lane holds its pair's sum;
`[2,3,0,1]` swaps pairs within each group of four; `row_half_mirror` and
`row_mirror` reverse groups of 8 and 16. After four steps each 16-lane
row holds its sum. `v_permlanex16_b32` then reads the other row's value,
and a last add gives every lane the total. Lane 0 stores it.

## The wait inside the loop

Each trip loads, waits, adds. While it waits the wave has nothing to do:
a load from memory takes hundreds of cycles, and only one is in flight
at a time. Better to ask for the next trip's data before adding this
trip's: issue the load for trip `i + 1`, then add trip `i`'s.

```python
def rowsum_ahead(depth):
    def body(b, k):
        V, S = b.v, b.s
        x, out, cols = k.load()
        off = V.u32((k.wgid * cols + k.tid) * 4, name="off")
        acc = V.f32(0.0, name="acc")
        v = V.f32(n=depth, name="v")
        for j in range(depth):  # the first depth trips' loads
            b.emit(O.global_load(v[j], off, x, offset=128 * j))
        steady = S.u32(cols // (32 * depth) - 1, name="steady")
        for it in b.loop(steady):
            for j in range(depth):
                b.wait(v[j])
                acc += v[j]
                b.emit(O.global_load(v[j], off, x, offset=128 * (j + depth)))
            off += 128 * depth
        for j in range(depth):  # the last depth trips
            b.wait(v[j])
            acc += v[j]
        total = V.f32(across(acc, "add", 32), name="total")
        with b.lanes(k.tid == 0):
            b.emit(O.global_store(V.u32(k.wgid * 4, name="o"), total, out))

    cols = u32.within(32 * depth, 1 << 17).multiple_of(32 * depth)
    args = Args(x=ptr, out=ptr, cols=cols)
    return KernelDef("rowsum", args, body, wgsize=32, max_grid=(1 << 12,))
```

`depth` loads are in flight at all times. The loop body handles `depth`
trips, so the loop runs `cols / (32 depth) - 1` times between a prologue
that issues the first loads and an epilogue that adds the last ones.
Here is its loop with `depth=2`:

```python
k2 = thin.compile(rowsum_ahead, depth=2)
body2 = lines(k2)
print("\n".join(body2[body2.index(".Lloop_1:") : body2.index(".Lloop_end_2:") + 1]))
```

```asm
.Lloop_1:
	s_wait_loadcnt 1
	v_add_f32 v1, v1, v4                                         // acc = acc + v[0]
	global_load_b32 v4, v2, s[4:5] offset:256
	s_wait_loadcnt 1
	v_add_f32 v1, v1, v5                                         // acc = acc + v[1]
	global_load_b32 v5, v2, s[4:5] offset:384
	v_add_nc_u32 v2, 256, v2                                     // off = off + 256
	s_add_co_u32 s0, s0, 1
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_cmp_lt_u32 s0, s1
	s_cbranch_scc1 .Lloop_1
.Lloop_end_2:
```

Look at the waits. At the top of the body, two loads are in flight, and
`v[0]`'s is the older one: `s_wait_loadcnt 1` lets the newer one stay
in flight. Then the body issues a load into `v[0]`, so when it waits for
`v[1]`, `v[0]`'s new load may still be outstanding: `s_wait_loadcnt 1`
again. We wrote `b.wait(v[j])`; the compiler counted, across the loop's
back edge, how many loads are allowed to remain.

```python
rows, cols = 6, 128
x = rng.standard_normal((rows, cols)).astype(np.float32)
for depth in (1, 2, 4):
    out = np.zeros(rows, np.float32)
    k = thin.compile(rowsum_ahead, depth=depth)
    k.bind(grid=rows, args={"x": x, "out": out, "cols": cols}).emulate()
    print(depth, np.allclose(out, x.sum(axis=1), rtol=1e-5, atol=1e-5))
```

```text
1 True
2 True
4 True
```

## Measuring

On the GPU, 256 rows of 131072 floats (128 MiB):

```python gpu
import torch

rows, cols = 256, 1 << 17
xg = torch.randn(rows, cols, device="cuda")
want = xg.sum(dim=1)
for name, k in [("wait each trip", kern)] + [
    (f"{d} ahead", thin.compile(rowsum_ahead, depth=d)) for d in (1, 2, 4, 8)
]:
    outg = torch.zeros(rows, device="cuda")
    call = k.bind(grid=rows, args={"x": xg, "out": outg, "cols": cols})
    us = call.time()
    ok = torch.allclose(outg, want, rtol=1e-4, atol=1e-3)
    print(f"{name:15s} {us:8.1f} us {rows * cols * 4 / us / 1e3:7.1f} GB/s {ok}")
```

```text
wait each trip     718.9 us   186.7 GB/s True
1 ahead            715.7 us   187.5 GB/s True
2 ahead            499.0 us   269.0 GB/s True
4 ahead            262.1 us   512.0 GB/s True
8 ahead            218.2 us   615.1 GB/s True
```

Every version reads the same bytes; what changes is how many loads each
wave keeps in flight. 256 rows are 256 waves, far fewer than the GPU can
run at once, so each wave's own waiting decides the time. With one load
in flight, a wave spends nearly all its time waiting for it; `1 ahead`
overlaps it with a single add, which is hardly better. With 8 in flight
per wave the same 256 waves keep the memory about as busy as it gets.
With many more rows, many more waves would be in flight at once, each
waiting for its own load, and the difference shrinks: the GPU hides one
wave's latency behind the others'. Part 7 returns to this.
