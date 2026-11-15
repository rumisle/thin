# 1. Vector add

We start with the smallest kernel that does real work: `z[i] = x[i] + y[i]`
for a million floats. We will write it, compile it, run it on the CPU
emulator and on the GPU, and then read every instruction the compiler
produced.

## The machine, briefly

A GPU kernel is one program that many *lanes* run at once, each on its
own data. On gfx1201 (an RDNA 4 GPU, such as the RX 9070 XT) lanes come in
*waves* of 32: the 32 lanes of a wave run each instruction together, in
lockstep. Waves are grouped into *workgroups*, and a launch runs a *grid*
of workgroups.

Each lane has its own registers, the **VGPRs** (vector registers: one
32-bit value per lane). Each wave also has registers it shares across its
lanes, the **SGPRs** (scalar registers: one value per wave). Work that is
the same for every lane of a wave, such as computing a buffer's address,
goes in SGPRs; per-lane work goes in VGPRs.

## Writing the kernel

A thin kernel is a Python function. It is given a *Builder* `b`, which
collects the instructions, and the kernel's *entry* `k`, which holds what
the hardware gives every lane when the kernel starts: the arguments, the
workgroup's id and the lane's id within its workgroup.

```python
import numpy as np
import thin
from thin import KernelDef, cdiv
from thin.dsl import Args, O, ptr, u32

WG = 256  # lanes per workgroup


def vadd():
    def body(b, k):
        V = b.v  # the vector register file
        x, y, z, n = k.load()  # the arguments
        i = V.u32(k.wgid * WG + k.tid, name="i")  # this lane's element
        with b.lanes(i < n):  # only the lanes with an element
            off = V.u32(i * 4, name="off")  # its byte offset
            a = V.f32(name="a")
            c = V.f32(name="c")
            b.emit(O.global_load(a, off, x))  # a = x[i]
            b.emit(O.global_load(c, off, y))  # c = y[i]
            b.wait(a, c)  # until both have arrived
            a += c
            b.emit(O.global_store(off, a, z))  # z[i] = a

    args = Args(x=ptr, y=ptr, z=ptr, n=u32)
    return KernelDef("vadd", args, body, wgsize=WG)
```

Read it top to bottom:

- `Args(x=ptr, y=ptr, z=ptr, n=u32)` declares the arguments: three
  buffers and a count.
- `k.load()` loads the arguments. They are the same for every lane, so
  they land in SGPRs.
- `V.u32(...)` makes a value in the vector file: one u32 per lane. Its
  argument is an expression; thin turns it into instructions.
- `with b.lanes(i < n):` switches off the lanes whose `i` is past the end,
  for the instructions inside the block. The last workgroup may have
  lanes with nothing to do.
- `O.global_load(a, off, x)` makes a load instruction: `a` gets the four
  bytes at `x + off`. `b.emit` adds it to the kernel.
- A load takes time to arrive. `b.wait(a, c)` says that the next
  instructions need `a` and `c`; the compiler works out how long to wait.

`vadd` is a *generator*: it returns the kernel's definition, a
`KernelDef`. Nothing is compiled yet.

## Compiling

`thin.compile` runs the generator and compiles the kernel:

```python
kern = thin.compile(vadd)
print(kern.name, kern.manifest.vgpr, kern.manifest.sgpr)
```

```text
vadd 3 10
```

The kernel uses 3 VGPRs and 10 SGPRs. `kern.asm` is the whole assembly
file: a header, the instructions, then the data the GPU driver needs to
launch it. The instructions are the interesting part. A small helper
prints just those; we will use it throughout:

```python
def listing(kern):
    lines = kern.asm.splitlines()
    start = lines.index(f"{kern.name}:") + 1
    end = lines.index("\ts_endpgm", start) + 1
    print("\n".join(lines[start:end]))


listing(kern)
```

```asm
	s_mov_b32 s2, ttmp9                                          // wgid#1: ttmp9
	s_load_b128 s[4:7], s[0:1], 0x0                              // x, y
	s_load_b64 s[8:9], s[0:1], 0x10                              // z
	s_load_b32 s3, s[0:1], 0x18                                  // n
	v_lshl_add_u32 v1, s2, 8, v0                                 // i = (wgid * 256) + tid
	s_wait_kmcnt 0                                               // kernel arguments
	v_cmp_gt_u32 vcc_lo, s3, v1                                  // i < n
	s_mov_b32 exec_lo, vcc_lo
	v_lshlrev_b32 v0, 2, v1                                      // off = i * 4
	global_load_b32 v1, v0, s[4:5]
	global_load_b32 v2, v0, s[6:7]
	s_wait_loadcnt 0
	v_add_f32 v1, v1, v2                                         // a = a + c
	global_store_b32 v0, v1, s[8:9]
	s_mov_b32 exec_lo, -1
	s_endpgm
```

Sixteen instructions. Before reading them, let us check that they work.

## Running on the emulator

thin has an emulator: it runs the compiled instructions on the CPU, lane
by lane, as the GPU would. Here is a launch over 1000 elements:

```python
n = 1000
rng = np.random.default_rng(0)
x = rng.standard_normal(n).astype(np.float32)
y = rng.standard_normal(n).astype(np.float32)
z = np.zeros(n, np.float32)

call = kern.bind(grid=cdiv(n, WG), args={"x": x, "y": y, "z": z, "n": n})
call.emulate()
print(np.array_equal(z, x + y))
```

```text
True
```

`kern.bind` makes one call: the grid (`cdiv(n, WG)`, 4 workgroups of 256
lanes, 1024 lanes for 1000 elements) and the arguments by name. Numpy
arrays are copied to the emulator's memory and back: `z` now holds the
sums, bit for bit what numpy computes.

thin also checks things the GPU would not tell you about. Take
away the wait:

```python
def vadd_nowait():
    def body(b, k):
        V = b.v
        x, y, z, n = k.load()
        i = V.u32(k.wgid * WG + k.tid, name="i")
        with b.lanes(i < n):
            off = V.u32(i * 4, name="off")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            b.emit(O.global_load(c, off, y))
            a += c
            b.emit(O.global_store(off, a, z))

    return KernelDef("vadd_nowait", Args(x=ptr, y=ptr, z=ptr, n=u32), body, wgsize=WG)


thin.compile(vadd_nowait)
```

```error
ScheduleError: v_add_f32 v:a#1, v:a#1, v:c#1 (after the entry) touches a register while the load on it may be in flight on some path: it needs loadcnt(0) - add a wait in the schedule
```

This one did not even reach the emulator: the compiler refused it. `a`
is read by the add while its load may still be on its way, and thin does
not guess where you want to wait. A wait is not free (the lane stalls
until the data arrives), so where it goes is the author's decision; the
compiler only checks that there is one and computes its count.

## Running on the GPU

On a machine with a gfx1201 GPU and PyTorch, the same call runs on the
GPU with torch tensors as buffers:

```python gpu
import torch

n = 1 << 24
xg = torch.randn(n, device="cuda")
yg = torch.randn(n, device="cuda")
zg = torch.zeros(n, device="cuda")
gpu_call = kern.bind(grid=cdiv(n, WG), args={"x": xg, "y": yg, "z": zg, "n": n})
gpu_call.run()
torch.cuda.synchronize()
print(torch.equal(zg, xg + yg))
```

```text
True
```

`run` launches on the current torch stream. `time` runs the call
repeatedly and returns the mean time of one run:

```python gpu
us = gpu_call.time()
print(f"{us:.1f} us, {3 * 4 * n / us / 1e3:.1f} GB/s")
```

```text
327.9 us, 614.1 GB/s
```

The kernel reads 8 bytes and writes 4 per element. At this size it moves
data about as fast as the card's memory allows: there is nothing in the
kernel to speed up. Part 7 looks at measuring more closely.

## Reading the instructions

Here is the listing again, line by line.

```asm quote
s_mov_b32 s2, ttmp9                  // wgid#1: ttmp9
```

The hardware puts the workgroup id in a special register, `ttmp9`. The
kernel reads `k.wgid`, so thin copies it into an ordinary SGPR. `s_`
instructions are *scalar*: they run once per wave, on SGPRs.

```asm quote
s_load_b128 s[4:7], s[0:1], 0x0      // x, y
s_load_b64 s[8:9], s[0:1], 0x10      // z
s_load_b32 s3, s[0:1], 0x18          // n
```

The arguments sit in memory; `s[0:1]` holds their address when the kernel
starts. Each pointer is 8 bytes, so `x` and `y` (bytes 0 to 15) come in
one 16-byte load, `z` in an 8-byte one, `n` in a 4-byte one. Scalar
loads take time too; nothing waits for them yet.

```asm quote
v_lshl_add_u32 v1, s2, 8, v0         // i = (wgid * 256) + tid
```

`i = k.wgid * 256 + k.tid` in one instruction: multiplying by 256 is a
shift left by 8, and the GPU has a shift-then-add. `v0` is `tid`, filled
by the hardware. `v_` instructions are *vector*: each lane computes its
own `i`. The comment is the statement this instruction came from.

```asm quote
s_wait_kmcnt 0                       // kernel arguments
v_cmp_gt_u32 vcc_lo, s3, v1          // i < n
s_mov_b32 exec_lo, vcc_lo
```

The compare reads `n`, the first instruction to read an argument, so the
wait for the argument loads goes right before it: the index was computed
while they arrived. `v_cmp_gt_u32` sets one bit per lane in `vcc_lo`
(`n > i`, which is `i < n`). Writing that to `exec_lo`, the *execution
mask*, switches off every lane whose bit is 0. This is `b.lanes(i < n)`.

```asm quote
v_lshlrev_b32 v0, 2, v1              // off = i * 4
```

`off = i * 4` is a shift by 2. It is placed in `v0`: `tid` is no longer
needed, so its register is free. Every register number in the listing is
the compiler's choice: you name values, it places them.

```asm quote
global_load_b32 v1, v0, s[4:5]
global_load_b32 v2, v0, s[6:7]
s_wait_loadcnt 0
```

The two loads go out back to back: `a` into `v1` (`i` is dead too), `c`
into `v2`. Loads complete in order, and the loads counter tells how many
are still in flight; `s_wait_loadcnt 0` waits until none is. That is
what `b.wait(a, c)` asked for: `c` was the last one issued.

```asm quote
v_add_f32 v1, v1, v2                 // a = a + c
global_store_b32 v0, v1, s[8:9]
s_mov_b32 exec_lo, -1
s_endpgm
```

The add, in place; the store of `a` to `z + off`; every lane switched on
again at the end of the `with` block; the end of the program.

Nothing in the listing was added or moved by the compiler beyond what
the code asked for: every instruction is one of ours, in our order, with
registers filled in and the waits counted. The next part looks at exactly
what the compiler does.
