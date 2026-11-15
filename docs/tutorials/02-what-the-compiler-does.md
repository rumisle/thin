# 2. What the compiler does, and what it doesn't

In part 1 every instruction of the listing was one we wrote. This part
makes that precise. The compiler does four things to the instructions it
is given: it chooses registers, counts waits, pads hazards and checks the
result. It never reorders, fuses, removes or adds work. We will see each
of the four, and see that an edit to the kernel changes exactly the lines
it touches.

We start from part 1's kernel, with a build parameter: `axpy(scale)`
computes `z = scale * x + y`, or `x + y` without a scale.

```python
import difflib

import numpy as np
import thin
from thin import KernelDef, cdiv
from thin.dsl import Args, O, cvt_f32, ptr, u32

WG = 256


def axpy(scale=None, early=False):
    def body(b, k):
        V = b.v
        x, y, z, n = k.load()
        i = V.u32(k.wgid * WG + k.tid, name="i")
        with b.lanes(i < n):
            off = V.u32(i * 4, name="off")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            if early:
                b.wait(a)
            b.emit(O.global_load(c, off, y))
            b.wait(a, c)
            if scale is not None:
                a[...] = a * scale
            a += c
            b.emit(O.global_store(off, a, z))

    return KernelDef("axpy", Args(x=ptr, y=ptr, z=ptr, n=u32), body, wgsize=WG)


def lines(kern):
    out = kern.asm.splitlines()
    start = out.index(f"{kern.name}:") + 1
    return out[start : out.index("\ts_endpgm", start) + 1]


def diff(before, after):
    for line in difflib.unified_diff(lines(before), lines(after), lineterm="", n=1):
        if not line.startswith(("---", "+++")):
            print(line)
```

`thin.compile(axpy, scale=2.0)` compiles the generator with that
parameter: each set of parameters is its own kernel.

## An edit changes the lines it touches

Add the scale and compare the two listings:

```python
plain = thin.compile(axpy)
scaled = thin.compile(axpy, scale=2.0)
diff(plain, scaled)
```

```text
@@ -12,2 +12,4 @@
 	s_wait_loadcnt 0
+	v_mul_f32 v1, 2.0, v1                                        // a = a * 2.0
+	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
 	v_add_f32 v1, v1, v2                                         // a = a + c
```

One statement, `a[...] = a * scale`, became one instruction,
`v_mul_f32 v1, 2.0, v1`, in the place we wrote it. Everything else is
unchanged, down to the register numbers. One more line is new, and it is
not ours: the add reads the multiply's result right after the multiply
produced it, and gfx1201 asks the program to say so.

## Hazards

The lines `s_delay_alu instid0(VALU_DEP_1)` are *delay hints*. A vector
instruction takes several cycles to produce its result. The hardware does
not track this for every pair of instructions on its own: it relies on
the program to say "the next instruction depends on the one 1 back".
Leave the hint out and the add may read `v1` before the multiply wrote
it. Such dependencies, and the wait states the ISA requires in a few
other cases (a WMMA's result read as a matrix operand, a scalar register
read by a vector instruction right after a scalar write), are *hazards*.
After the registers are placed, the compiler finds every one and inserts
the instruction that covers it, marked `// hazard`. It never moves an
instruction to hide one: it only pads.

## Registers

The names in the code (`i`, `off`, `a`, `c`) are *values*; the compiler
gives each a register for as long as it is *live*, from its write to its
last read. Values that are never live at the same time share registers:
that is why `off` takes `v0` (`tid` is dead by then) and `a` takes `v1`
(`i` is dead once `off` is computed).

Keep `i` alive to the end, and it can no longer give up `v1`:

```python
def axpy_i():
    def body(b, k):
        V = b.v
        x, y, z, n = k.load()
        i = V.u32(k.wgid * WG + k.tid, name="i")
        with b.lanes(i < n):
            off = V.u32(i * 4, name="off")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            b.emit(O.global_load(c, off, y))
            b.wait(a, c)
            a += c
            a += cvt_f32(i)  # reads i, after the loads
            b.emit(O.global_store(off, a, z))

    return KernelDef("axpy", Args(x=ptr, y=ptr, z=ptr, n=u32), body, wgsize=WG)


diff(plain, thin.compile(axpy_i))
```

```text
@@ -9,7 +9,10 @@
 	v_lshlrev_b32 v0, 2, v1                                      // off = i * 4
-	global_load_b32 v1, v0, s[4:5]
-	global_load_b32 v2, v0, s[6:7]
+	global_load_b32 v2, v0, s[4:5]
+	global_load_b32 v3, v0, s[6:7]
 	s_wait_loadcnt 0
-	v_add_f32 v1, v1, v2                                         // a = a + c
-	global_store_b32 v0, v1, s[8:9]
+	v_add_f32 v2, v2, v3                                         // a = a + c
+	v_cvt_f32_u32 v3, v1                                         // a = a + cvt_f32(i)
+	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
+	v_add_f32 v2, v2, v3
+	global_store_b32 v0, v2, s[8:9]
 	s_mov_b32 exec_lo, -1
```

`a` and `c` moved to `v2` and `v3`, and the kernel now uses one more
VGPR. Placement is the one change that can ripple beyond the edited
lines: a value that lives longer holds its register longer. It is still
deterministic: the same code always gets the same registers.

The compiler never spills a value to memory to make room. When the
values live at one point do not fit in the file (256 VGPRs, fewer when
the workgroup is large), it refuses the kernel and lists what is live at
the peak. Freeing registers is the author's job, and the message says
where.

## Waits

`b.wait(a, c)` asks for both loads to have landed. Loads of one kind
complete in the order they were issued, and the hardware counts those
still in flight: `s_wait_loadcnt n` waits until at most `n` are left. The
compiler computes each `n`. Ask for `a` before issuing the second load
(`early=True`):

```python
diff(plain, thin.compile(axpy, early=True))
```

```text
@@ -10,2 +10,3 @@
 	global_load_b32 v1, v0, s[4:5]
+	s_wait_loadcnt 0
 	global_load_b32 v2, v0, s[6:7]
```

`a` was the only load in flight, so its wait is `s_wait_loadcnt 0`, and
the second wait is now for `c` alone. The edit moved where the program
waits, and the compiler kept it there: it would be wrong to move it
(the second load now starts only after the first arrives, which is what
we wrote) and it would be wrong to drop it.

## What it does not do

- **It does not reorder.** The instructions come out in the order the
  code emitted them, so the code decides what overlaps with what: issuing
  both loads before waiting overlaps their latency, and waiting in
  between does not.
- **It does not optimize.** No common subexpressions are merged, no dead
  code is removed, `a * b + c` is not fused into an `fma` (that would
  change its rounding). What thin rewrites is a short list of exact
  identities, such as `i * 4` into a shift (the reference lists them:
  [3.3 Rewrites](../lang.md#33-rewrites)).
- **It does not guess.** A missing wait, a value read before it is
  written, a register file that overflows, a vector value assigned to a
  scalar register: each is refused, with the line that caused it.

The second point means a kernel can be read off its source. A statement
is a few instructions, in place, and the comment on the first one says
which statement it was:

```python
print("\n".join(lines(scaled)[11:16]))
```

```asm
	s_wait_loadcnt 0
	v_mul_f32 v1, 2.0, v1                                        // a = a * 2.0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32 v1, v1, v2                                         // a = a + c
	global_store_b32 v0, v1, s[8:9]
```
