# The thin language

A kernel is a Python function that makes a `Builder`, makes values in its
register files and emits instructions through it: an instruction written
out (`b.emit(O.op("v_add_nc_u32", z, x, y))`), or a statement over values
(`z[...] = x + y`) that the Builder lowers to instructions as it runs.
The compiled kernel is what was emitted, in the order it was emitted.
Compilation fills in only what is mechanical: the register of every value
(placement by liveness, [1.4](#14-placement)), the count of every wait
the author asked for (`s_wait_loadcnt` and the others,
[5](#5-waits-and-hazards)), the hazard instructions between dependent
instructions (`s_delay_alu`, `s_wait_alu`, `v_nop`, `s_nop`), a final
`s_endpgm` where the last instruction falls through, and the code
object's descriptor and metadata. It then runs its checks
([7](#7-checks)); a kernel that fails one is refused, not repaired.
Nothing is reordered. Four passes delete or fold what the Builder itself
emitted and nothing needs: workgroup-id moves no instruction reads, an
exit from a lane region that the next region's entry overwrites (left as
a comment), M0 writes that repeat the value M0 holds, and the copies of
a peeled loop past its steady state ([4.2](#42-loops)).

The target is gfx1201 (RDNA 4). Instruction names, encodings and
semantics are AMD's: see the *RDNA 4 Instruction Set Architecture*
reference guide.

| Chapter | What it covers |
|---|---|
| [1. Values](#1-values) | register files, types, elements and halves, pins, placement |
| [2. Instructions](#2-instructions) | `O.op`, operands, fields, modifiers, the memory families, refusals |
| [3. Expressions](#3-expressions) | statements, the operator table, division, floats, conversions, masks, lanes, 64-bit |
| [4. Control flow](#4-control-flow) | branches, loops, exits, lane regions, wave blocks |
| [5. Waits and hazards](#5-waits-and-hazards) | `b.wait`, the counters, what the Builder counts, hazards, delay hints |
| [6. Entry and arguments](#6-entry-and-arguments) | `Args`, argument kinds, the entry registers, loading arguments, workgroup ids |
| [7. Checks](#7-checks) | everything compilation refuses, and why |
| [8. A whole kernel](#8-a-whole-kernel) | vector add, every line explained |
| [9. Memory](#9-memory) | LDS layouts, addresses as formulas, swizzles, the LDS proof, global and buffer addressing |
| [10. Matrix cores and the rest](#10-matrix-cores-and-the-rest) | WMMA, claims, ablation, labels, the Builder's other members |
| [11. Compiling, running, emulating](#11-compiling-running-emulating) | `KernelDef`, `compile`, `bind`, `run`, `emulate`, the store, the emulator's reports, limits |

## How to read the examples

Every example is run by `tests/test_docs.py`, and the block under it is
what it produced.

- An example starts with a fresh `Builder` named `b` for gfx1201, its
  files `V = b.v` and `S = b.s`, and every name of `thin.dsl` imported.
- Its ```` ```asm ```` block is the kernel's instructions as compilation
  renders them, with the Builder's own comments after `//`. A kernel's
  header, descriptor and final `s_endpgm` are left out
  ([chapter 11](#11-compiling-running-emulating) shows a whole file).
- Its ```` ```error ```` block is the refusal it raised, `Class: message`,
  whether while the example ran or when it was compiled. A location
  `kernel.py:N` is line N of the example.
- Its ```` ```text ```` block is what it printed.
- Inputs are pinned: `V.u32(name="x", at=3)` is the value the hardware
  holds in v3 when the kernel starts, so its number is fixed. Every other
  register number is the placement's choice, and may differ in a kernel
  with other values around.

# 1. Values

A value is a group of 32-bit registers in one file, with a type, made by
the file. The author names values, never registers: compilation places
each value in registers, by liveness.

## 1.1 Files and types

A Builder has two files: `b.v`, the VGPRs (one 32-bit register per lane
of a wave: 256 of them on gfx1201), and `b.s`, the SGPRs (one per wave:
106). Each is a `RegFile`; `Builder(vgprs=, sgprs=)` makes smaller files,
and `F.limit` is a file's size. A value is a `TypedValue`, its type a
`DslType`: `u32`, `f32` and the others below are its instances, compared
by identity.

| Type | Dwords | Files | What it is |
|---|---|---|---|
| `u32` | 1 | v, s | unsigned 32-bit integer |
| `i32` | 1 | v, s | signed 32-bit integer |
| `f32` | 1 | v, s | IEEE single |
| `u64` | 2 | v, s | unsigned 64-bit integer |
| `ptr` | 2 | v, s | a 64-bit address (a `u64` that names memory) |
| `f32x2` | 2 | v | two f32, an even-aligned VGPR pair |
| `mask` | 1 | s | a lane mask: one bit per lane (wave32: one SGPR, its type `mask32`) |

`F.u32(init=None, *, n=1, name=None, at=None, same_as=None)` makes a
value of `n` elements in file `F`; `F.i32`, `F.f32`, `F.u64`, `F.ptr`,
`F.f32x2` likewise, and `S.mask(init=None, *, name=None, at=None,
same_as=None)`. `init`, when given, is written at once (a statement,
[3.1](#31-statements)), into every element.

```python
x = V.u32(name="x", at=4)
y = V.u32(x + 1, name="y")
ones = V.f32(1.0, n=2, name="ones")
print(repr(x), x.type, x.width, len(ones), ones.width)
```

```text
v:x#1 u32 1 2 2
```

A value's `repr` is `<file>:<name>#<k>`: names repeat freely, and `k`
counts the values of a name. A name is letters, digits, `_` and `.`; a
value made without one is `t`. The comments of the `.s` show a value by
its name alone. `x.file`, `x.type`, `x.n` (elements), `x.width` (dwords)
and `x.name` describe it.

A mask is made in the s file only, an `f32x2` in the v file only:

```python
V.mask(name="lanes")
```

```error
ValueError: a mask in the v file: a lane mask is SGPRs (S.mask())
```

## 1.2 Elements, slices and halves

`x[j]` is element `j` (negative from the end), `x[a:b]` the elements `a`
to `b - 1`, a value of their own type; `len(x)` and `for e in x` work as
for a list. Slices are contiguous: a step is refused.

```python
q = V.u32(n=4, name="q", at=8)
q[1] = q[0] + q[3]
q[2:4][1] = q[1] - 1
```

```asm
	v_add_nc_u32 v9, v8, v11                                     // q[1] = q[0] + q[3]
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_subrev_nc_u32 v11, 1, v9                                   // q[2:4][1] = q[1] - 1
```

`.lo` and `.hi` are the halves of a two-dword value (`u64`, `ptr`,
`f32x2`) as `u32`, or `f32` for an `f32x2`:

```python
p = S.ptr(name="p", at=2)
lo = S.u32(p.lo + 64, name="lo")
hi = S.u32(p.hi, name="hi")
```

```asm
	s_add_co_u32 s0, s2, 64                                      // lo = p.lo + 64
	s_mov_b32 s1, s3                                             // hi = p.hi
```

A divisor argument ([6.2](#62-argument-kinds)) has `.mul` and `.shr`,
its magic multiplier and shift. `x.sub` is a view's dword offset into its
value, `x.part` the way it was taken (`[1:3]`, `.hi`).

## 1.3 Pins and aliases

`at=r` pins a value: it is the value the hardware holds in register `r`
(and the ones after it) when the kernel starts. Pins are how inputs enter
a kernel; the entry pins its own ([6.3](#63-the-entry)). Two pins may
name one register only if their values are never live together: a pinned
value is live from the entry to its last read.

```python
x = V.u32(name="x", at=0)
y = V.u32(name="y", at=0)
z = V.u32(x + y, name="z")
```

```error
AssignError: v: no room for y#1[0] (1 dwords, align 1; occupied from `v_add_nc_u32 v:z#1, v:x#1, v:y#1` to `v_add_nc_u32 v:z#1, v:x#1, v:y#1`): pinned at 0, which a pin before it holds. The peak: 3 dwords live at `v_add_nc_u32 v:z#1, v:x#1, v:y#1` (instruction 0); there: z#1[0] 1, y#1[0] 1, x#1[0] 1
```

`same_as=v` puts a value on the registers of another whole value of the
same file: the two names are one storage, and a write through either is
seen through both.

```python
a = V.u32(name="a", at=2)
alias = V.u32(name="alias", same_as=a)
alias[...] = a + 1
b.emit(O.op("v_mov_b32", a, alias))
```

```asm
	v_add_nc_u32 v2, 1, v2                                       // alias = a + 1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_mov_b32 v2, v2
```

## 1.4 Placement

Every value that is not pinned is placed: compilation gives it registers
for its live range, from each write to the last read reached from it, and
values whose ranges do not overlap share registers. Placement
guarantees:

- A read sees the last write to its value on every path to it; no other
  value is placed over a value while it is live.
- A value occupies consecutive registers, from a multiple of 2 (a value
  of two or three dwords) or of 4 (four or more), as wide operands
  require.
- A value read where no path from the entry wrote it is refused:

```python
r = V.u32(name="r")
s = V.u32(r + 1, name="s")
```

```error
AssignError: v:r#1[0] is read where no path from the entry wrote it (first use: `v_add_nc_u32 v:s#1, 1, v:r#1`, instruction 0): write it before (a register the hardware fills at entry is pinned: RegFile.entry, RegFile.pin)
```

Liveness is per lane. A VGPR written under a partial EXEC (in a lane
region, [4.4](#44-lane-regions)) keeps its other lanes, so the value
before the write stays live to it:

```python
k = b.entry(wgsize=64)
x = V.u32(name="x", at=1)
y = V.u32(5, name="y")
with b.lanes(k.tid < 10):
    y[...] = x + 1
b.emit(O.op("v_mov_b32", x, y))
```

```asm
	v_mov_b32 v2, 5                                              // y = 5
	v_cmp_gt_u32 vcc_lo, 10, v0                                  // tid < 10
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32 v2, 1, v1                                       // y = x + 1
	s_mov_b32 exec_lo, -1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_mov_b32 v1, v2
```

Here `y = 5` must survive in lanes 10 and up, so `y` cannot take `tid`'s
v0. `V.lanes_covered(name)` declares that the partial writes of the
values named `name` cover every lane their reads use: such a write ends
the old value. Lanes a region leaves off are not otherwise tracked: a
value written only inside a region and read outside it holds, in the
other lanes, whatever its register held.

When values do not fit, placement refuses and names the peak (here in
a file of 8 VGPRs):

```python
b = Builder(vgprs=8)
V = b.v
x = V.u32(name="x", at=0)
w = V.u32(n=8, name="w")
for e in w:
    e[...] = x + 1
b.emit(O.op("v_mov_b32", x, w[7]))
```

```error
AssignError: v: no room for w#1[0] (8 dwords, align 4; occupied from `v_add_nc_u32 v:w#1+0, 1, v:x#1` to `v_mov_b32 v:x#1, v:w#1+7`): every aligned place in 8 is taken there. The peak: 9 dwords live at `v_add_nc_u32 v:w#1+7, 1, v:x#1` (instruction 7); there: w#1[0] 8, x#1[0] 1
```

`F.number(r)` and `F.phys(r)` read a register's number, the second only
once the placement has made it final; `F.used()` is the file's high-water
mark. These, `F.alloc`, `F.pin`, `F.same_as`, `F.reg`, `F.value_of`,
`F.apply`, `F.entry`, `F.vals`, `F.values`, `F.entered`, `F.pinned` and
`F.lanes` are the file's lower-level members, for libraries and tools:
they declare groups of raw values, pin or alias them by name, and read
the placement.

# 2. Instructions

`O.op(name, *operands, **fields)` makes one instruction; `b.emit(i)`
appends it to the kernel. `name` is a gfx1201 mnemonic
(`"v_add_nc_u32"`) or thin's dotted op name (`"v.add.u32"`), which is
the mnemonic with dots; where gfx1201 renamed an op, the dotted name is
the older one (`v.add.u32` is `v_add_nc_u32`, `s.add.u32` is
`s_add_co_u32`, `v.max.f32` is `v_max_num_f32`). Operands are in the
ISA's order, destinations first.

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k = S.u32(name="k", at=2)
b.emit(O.op("v_add3_u32", u, u, w, k))
b.emit(O.op("v.add.u32", w, u, 7, comment="w = u + 7"))
```

```asm
	v_add3_u32 v3, v3, v4, s2
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_nc_u32 v4, v3, 7                                       // w = u + 7
```

## 2.1 Operands

| Operand | Written as |
|---|---|
| a value or a view of one | `x`, `x[2]`, `p.lo`, `q[0:4]` |
| a named register | `O.VCC`, `O.EXEC`, `O.SCC`, `O.M0`, `O.NULL`, `O.VCC_LO`, `O.VCC_HI`, `O.EXEC_LO`, `O.EXEC_HI`, `O.phys(name)`, or the name as a string (`"vcc"`) |
| an integer | `7`, `-16`, `0x40` |
| a float | `1.0`, `0.1` |
| a constant's text | `"0x40"`, `"-2.0"`: the value, printed as written |
| a label | `".Lskip"` (a branch's target) |
| a source modifier | `O.neg(x)`, `O.abs_(x)`, nested in either order |

An operand's width must be its slot's: a 64-bit slot takes a two-dword
value. On wave32, `O.VCC` and `O.EXEC` name `vcc_lo` and `exec_lo`.

```python
f, g = V.f32(name="f", at=1), V.f32(name="g", at=2)
b.emit(O.op("v_fma_f32", f, O.neg(f), O.abs_(g), 1.0))
b.emit(O.op("v_cmp_gt_f32", O.VCC, 0.5, f))
b.emit(O.op("v_cndmask_b32", g, 0, g, O.VCC))
```

```asm
	v_fma_f32 v1, -v1, |v2|, 1.0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cmp_gt_f32 vcc_lo, 0.5, v1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cndmask_b32 v2, 0, v2, vcc_lo
```

## 2.2 Fields

Keyword arguments are the instruction's fields, spelled as in the ISA:
`offset`, `offset0`, `offset1`, `th`, `scope`, `offen`, `idxen`, `clamp`,
`omod`, `op_sel`, `op_sel_hi`, `neg_lo`, `neg_hi`, `dpp`, `row_mask`,
`bank_mask`, `bound_ctrl`. A field set to `None`, `False` or `""` is
left out. A field the form does not have is refused:

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
b.emit(O.op("v_add_nc_u32", u, u, w, clamp=True))
```

```error
Diag: kernel.py:2: `v_add_nc_u32`: no field clamp (it takes none)
```

Four keywords are not fields: `comment=` (the text after `//`), `tag=`
(a tuple naming a memory op for a wait, [5.1](#51-waiting)), `abl=` (its
ablation classes, [10.3](#103-ablation)), and `stage=`, the families'
own. `O.place(i, *names)` records names an instruction writes for
`b.wait(tags=, written=True)`, and `O.published(i, why)` marks a store
as a publication, with its reason, for libraries' checks.

## 2.3 Memory instructions

`O.op` makes any memory instruction by its mnemonic. The families below
pick the mnemonic from the data's width, take an address as a formula
([9.5](#95-global-and-buffer-addresses)), and check offsets:

| Family | Makes |
|---|---|
| `O.s_load(dst, base, offset=0)` | `s_load_b32` .. `s_load_b512` (1, 2, 4, 8 or 16 dwords) |
| `O.s_store(data, base, offset=0)` | `s_store_b32` |
| `O.global_load(dst, addr, base=None, *, offset=0, bytes=None, signed=False)` | `global_load_b32` .. `_b128`, `_u8`, `_i8`, `_u16`, `_i16` |
| `O.global_store(addr, data, base=None, *, offset=0, bytes=None)` | `global_store_b32` .. `_b128`, `_b8`, `_b16` |
| `O.buffer_load(dst, voffset, desc, soffset=0, *, offset=0, bytes=None, signed=False)` | `buffer_load_b32` .. |
| `O.buffer_store(data, voffset, desc, soffset=0, *, offset=0, bytes=None)` | `buffer_store_b32` .. |
| `O.ds_read(dst, addr, offset=0, *, bytes=None, signed=False, ordered="")` | `ds_load_b32` .. `_b128`, `_u8`, `_i8`, `_u16`, `_i16` |
| `O.ds_write(addr, data, offset=0, *, bytes=None, ordered="")` | `ds_store_b32` .. `_b128`, `_b8`, `_b16` |

`bytes=1` or `2` makes a sub-dword access of one dword, `signed=True` its
sign-extending load. Every family also takes `comment=` and `tag=`; the
global, buffer and scalar ones take `th=` and `scope=` (the temporal
hint `TH_*` and scope `SCOPE_*` of the ISA).

```python
gp = S.ptr(name="gp", at=6)
i = V.u32(name="i", at=1)
d = V.u32(n=4, name="d", at=4)
h = V.u32(name="h", at=8)
b.emit(O.global_load(d, i, gp, offset=16))
b.emit(O.global_load(h, i, gp, bytes=2, signed=True))
b.wait(d, h)
b.emit(O.global_store(i, d[1:3], gp, th="TH_STORE_NT", scope="SCOPE_DEV"))
```

```asm
	global_load_b128 v[4:7], v1, s[6:7] offset:16
	global_load_i16 v8, v1, s[6:7]
	s_wait_loadcnt 0
	global_store_b64 v1, v[5:6], s[6:7] th:TH_STORE_NT scope:SCOPE_DEV
```

`O.buffer_off(off)` is the text of a buffer instruction's `offset:` field
(`""` for 0), checked against its range.

## 2.4 Control and special instructions

Branches, `s_endpgm`, `s_nop`, `s_sleep`, `s_barrier_*`, clock reads and
EXEC writes are instructions like any other:

```python
k = S.u32(name="k", at=1)
b.emit(O.op("s_cmp_lg_u32", k, 7))
b.emit(O.op("s_cbranch_scc0", ".Lskip"))
b.emit(O.op("s_mov_b32", O.M0, k))
b.put_label(".Lskip")
b.emit(O.op("s_endpgm"))
```

```asm
	s_cmp_lg_u32 s1, 7
	s_cbranch_scc0 .Lskip
	s_mov_b32 m0, s1
.Lskip:
	s_endpgm
```

The Builder has its own spellings for some: `b.nop(n)` is `s_nop n`,
`b.barrier(comment="", fence=False)` the target's barrier sequence
(`fence=True` marks it as an LDS fence, [9.4](#94-what-is-proved-about-lds)),
`b.put_label(name)` places a label and `b.comment(text)` a comment line.

## 2.5 What an instruction is refused for

The maker refuses, at the `O.op` call, an instruction gfx1201 has no form
for:

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
b.emit(O.op("v_mul_lo_u32", u, w, w, 3))
```

```error
ValueError: `v_mul_lo_u32 v:u#1, v:w#1, v:w#1, 3`: no gfx1201 form has this mnemonic and operand count (the target's operand table)
```

an operand of the wrong file or width for its slot, a `None` or `bool`
operand, a VALU instruction reading more than two SGPRs or literals
(gfx1201's constant bus), two literals in one SALU instruction, and an
offset outside its field:

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k0, k1, k2 = S.u32(name="k0", at=0), S.u32(name="k1", at=1), S.u32(name="k2", at=2)
b.emit(O.op("v_add3_u32", u, k0, k1, k2))
```

```error
ValueError: v_add3_u32: reads k0#1, k1#1, k2#1: a VALU op reads 2 SGPRs or literals at most on gfx1201 (the constant bus) - move the others into VGPRs first (v_mov_b32), or write it as an expression, which moves them
```

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
b.emit(O.op("ds_store_b32", u, w, offset=65536))
```

```error
ValueError: ds_store_b32: LDS offset 65536 out of range (offset 0..65535, offset0 / offset1 0..255; unsigned): put the rest into the address register (a second base: addr + 65536 k, or addr - 16 for a negative one)
```

A modifier on a slot that takes none and an unknown `th` or `scope` are
refused when the kernel is compiled, with the line that made them.

# 3. Expressions

An expression is a Python expression over values and constants: a `Tree`,
which emits nothing. A statement writes a tree into a value; the Builder
lowers it then, instruction by instruction, into the destination and
temporaries of its own. The comment on a statement's first instruction is
the statement as written.

## 3.1 Statements

`F.T(tree)` makes a value and writes the tree into it; `x[...] = tree`
writes into an existing value or view, `x[j] = tree` into its element `j`,
and `x += tree` (any of `+ - * & | ^ << >> // %`) is `x[...] = x + tree`.
A statement writes one element: a value of several is written element by
element.

```python
x = V.u32(name="x", at=1)
y = V.u32(x << 2, name="y")
y[...] = y * x
y += 2
```

```asm
	v_lshlrev_b32 v0, 2, v1                                      // y = x << 2
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_mul_lo_u32 v0, v0, v1                                      // y = y * x
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_nc_u32 v0, 2, v0                                       // y = y + 2
```

A tree held in a Python variable is a formula, not a value: each
statement that uses it lowers it again.

```python
x = V.u32(name="x", at=1)
t = (x << 2) | 1
a = V.u32(t, name="a")
c = V.u32(t + 4, name="c")
```

```asm
	v_lshl_or_b32 v0, v1, 2, 1                                   // a = (x << 2) | 1
	v_lshl_or_b32 v2, v1, 2, 1                                   // c = ((x << 2) | 1) + 4
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_nc_u32 v1, 4, v2
```

A tree is computed in the file of its leaves: integer arithmetic over
scalar leaves only on the SALU, anything else on the VALU (f32
arithmetic, conversions and lane functions are VALU operations). A scalar result written into a
VGPR value is moved there; a vector result is never moved into an SGPR,
because which lane's value is meant must be said (`first_lane(x)`,
`lane_read(x, k)`):

```python
s = S.u32(name="s", at=4)
r = V.u32(s + 1, name="r")
```

```asm
	s_add_co_u32 s0, s4, 1                                       // r = s + 1
	v_mov_b32 v0, s0
```

```python
v = V.u32(name="v", at=1)
S.u32(v + 1)
```

```error
ExprError: v + 1: a vector value into an SGPR (v_readfirstlane is never implied: first_lane(x))
```

Types do not mix. `u32(x)`, `i32(x)`, `f32(x)` read a one-dword value's
bits as another type and emit nothing of their own; `f32x2(x[2:4])` reads
two elements of an f32 value as a pair. A bool is not a constant: a
condition becomes a number through `select`.

```python
x = V.i32(name="x", at=1)
y = V.u32(name="y", at=2)
V.u32(x + y)
```

```error
ExprError: x + y: i32 with u32 (types do not mix: cast one - u32(x), i32(x), f32(x) reinterpret, no instruction)
```

## 3.2 The operator table

<!-- tests/test_docs.py checks this table: each row's expression, of a and
c of the column's file and type, written into a value of that type (a
compare: into S.mask), renders these instructions, hazards aside. -->

| Expression | `V` u32 | `V` i32 | `V` f32 | `S` u32 | `S` i32 |
|---|---|---|---|---|---|
| `a + c` | `v_add_nc_u32` | `v_add_nc_u32` | `v_add_f32` | `s_add_co_u32` | `s_add_co_u32` |
| `a - c` | `v_sub_nc_u32` | `v_sub_nc_u32` | `v_sub_f32` | `s_sub_co_u32` | `s_sub_co_u32` |
| `a * c` | `v_mul_lo_u32` | `v_mul_lo_u32` | `v_mul_f32` | `s_mul_i32` | `s_mul_i32` |
| `a & c` | `v_and_b32` | `v_and_b32` | refused | `s_and_b32` | `s_and_b32` |
| `a \| c` | `v_or_b32` | `v_or_b32` | refused | `s_or_b32` | `s_or_b32` |
| `a ^ c` | `v_xor_b32` | `v_xor_b32` | refused | `s_xor_b32` | `s_xor_b32` |
| `a << c` | `v_lshlrev_b32` | `v_lshlrev_b32` | refused | `s_lshl_b32` | `s_lshl_b32` |
| `a >> c` | `v_lshrrev_b32` | `v_ashrrev_i32` | refused | `s_lshr_b32` | `s_ashr_i32` |
| `~a` | `v_not_b32` | `v_not_b32` | refused | `s_not_b32` | `s_not_b32` |
| `-a` | refused | refused | `v_xor_b32` | refused | refused |
| `abs(a)` | refused | refused | `v_and_b32` | refused | refused |
| `min(a, c)` | `v_min_u32` | `v_min_i32` | `v_min_num_f32` | `s_min_u32` | `s_min_i32` |
| `max(a, c)` | `v_max_u32` | `v_max_i32` | `v_max_num_f32` | `s_max_u32` | `s_max_i32` |
| `select(a < c, a, c)` | `v_cmp_lt_u32`, `v_cndmask_b32` | `v_cmp_lt_i32`, `v_cndmask_b32` | `v_cmp_lt_f32`, `v_cndmask_b32` | `s_cmp_lt_u32`, `s_cselect_b32` | `s_cmp_lt_i32`, `s_cselect_b32` |
| `a < c` | `v_cmp_lt_u32_e64` | `v_cmp_lt_i32_e64` | `v_cmp_lt_f32_e64` | `s_cmp_lt_u32`, `s_cselect_b32` | `s_cmp_lt_i32`, `s_cselect_b32` |
| `a <= c` | `v_cmp_le_u32_e64` | `v_cmp_le_i32_e64` | `v_cmp_le_f32_e64` | `s_cmp_le_u32`, `s_cselect_b32` | `s_cmp_le_i32`, `s_cselect_b32` |
| `a == c` | `v_cmp_eq_u32_e64` | `v_cmp_eq_i32_e64` | `v_cmp_eq_f32_e64` | `s_cmp_eq_u32`, `s_cselect_b32` | `s_cmp_eq_i32`, `s_cselect_b32` |
| `a != c` | `v_cmp_ne_u32_e64` | `v_cmp_ne_i32_e64` | `v_cmp_neq_f32_e64` | `s_cmp_lg_u32`, `s_cselect_b32` | `s_cmp_lg_i32`, `s_cselect_b32` |
| `~(a < c)` | `v_cmp_ge_u32_e64` | `v_cmp_ge_i32_e64` | `v_cmp_nlt_f32_e64` | `s_cmp_ge_u32`, `s_cselect_b32` | `s_cmp_ge_i32`, `s_cselect_b32` |

`>` and `>=` are `<` and `<=` with the operands exchanged. `~` of a
compare is its exact complement: for f32 the unordered compare
(`~(a < c)` is `v_cmp_nlt_f32`, true when either operand is a NaN). A
compare written into a mask is a lane mask, one bit per lane
([3.7](#37-masks-and-select)); a scalar compare sets SCC, which
`s_cselect_b32` turns into a mask of every lane or none.

Every operator takes a constant on either side. A VALU instruction reads
an SGPR or a constant in its first source only: the lowering puts it
there, takes the reversed form (`v_subrev_nc_u32`, `v_lshlrev_b32`)
where that changes the operand order, and moves a constant into a VGPR
where no form takes it (`d3`):

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k = S.u32(name="k", at=1)
d0 = V.u32(w - k, name="d0")
d1 = V.u32(62 - w, name="d1")
d2 = V.u32(5 << u, name="d2")
d3 = V.u32(select(u > 3, u, 100), name="d3")
```

```asm
	v_subrev_nc_u32 v0, s1, v4                                   // d0 = w - k
	v_sub_nc_u32 v1, 62, v4                                      // d1 = 62 - w
	v_lshlrev_b32_e64 v2, v3, 5                                  // d2 = 5 << u
	v_mov_b32 v4, 100                                            // d3 = select(u > 3, u, 100)
	v_cmp_lt_u32 vcc_lo, 3, v3
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cndmask_b32 v5, v4, v3, vcc_lo
```

A shift by a value takes the low 5 bits of the amount, as the
instruction does. Python's `bool` operators do not apply to trees: `a <
b < c`, `not c` and `if c:` raise, because a tree is no Python bool.

```python
a, c = V.u32(name="a", at=1), V.u32(name="c", at=2)
if a < c:
    pass
```

```error
ExprError: a < c: an expression is not a Python bool (a branch is b.when(c); a chain of compares is (a < b) & (b < c); a value is select(c, 1, 0))
```

## 3.3 Rewrites

The Builder lowers a few patterns to one instruction, and drops identities:

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k = S.u32(name="k", at=1)
la = V.u32(w + u * 8, name="la")
lo = V.u32(u | (w << 4), name="lo")
bf = V.u32((w >> 9) & 0x3F, name="bf")
m = V.u32(u * 64, name="m")
same = V.u32(u + 0, name="same")
sk = S.u32(k * 4 + k, name="sk")
```

```asm
	v_lshl_add_u32 v0, v3, 3, v4                                 // la = w + (u * 8)
	v_lshl_or_b32 v1, v4, 4, v3                                  // lo = u | (w << 4)
	v_bfe_u32 v2, v4, 9, 6                                       // bf = (w >> 9) & 63
	v_lshlrev_b32 v4, 6, v3                                      // m = u * 64
	v_mov_b32 v5, v3                                             // same = u + 0
	s_lshl2_add_u32 s0, s1, s1                                   // sk = (k * 4) + k
```

These are the rewrites, all exact: `x * 2^k` is a shift; `a + b * 2^k`
and `a + (b << k)` are `v_lshl_add_u32` (SALU: `s_lshl1_add_u32` ..
`s_lshl4_add_u32` for k of 1 to 4); `a | (b << k)` is `v_lshl_or_b32`;
`(x >> s) & (2^w - 1)` is a bit-field extract (`v_bfe_u32`, `s_bfe_u32`;
`v_bfe_i32` for an i32); constants of `(x - c1) + c2` fold; `x + 0`,
`x - 0`, `x * 1`, `x | 0`, `x ^ 0`, `x & ~0`, `x << 0`, `x >> 0` are `x`.
Nothing else is rewritten: in particular, `a * b + c` of floats is a
multiply then an add, never a fused `v_fma_f32` (`fma(a, b, c)` is
written).

## 3.4 Integer multiply and divide

`a * b` keeps the low 32 bits (`v_mul_lo_u32`). `mulhi(a, b)` is the
high 32 bits of the u32 product, `mul_wide(a, b)` the whole 64-bit
product as a u64, `mul24(a, b)` and `mad24(a, b, c)` the product of the
low 24 bits (`v_mul_u32_u24`, `v_mad_u32_u24`; on the SALU a full
`s_mul_i32`). A 24-bit operand that may pass 24 bits is refused at
compile time ([7.5](#75-ranges)).

`x // d` and `x % d` take a u32 `x` and a positive constant, or a divisor
argument ([6.2](#62-argument-kinds)). A power of two is a shift or a
mask; another constant is a multiply by its magic number, `magic(d)`,
the triple `(m, s, short)` of the multiplier, the shift and whether the
short sequence is exact:

```python
u = V.u32(name="u", at=3)
q5 = V.u32(u // 5, name="q5")
r12 = V.u32(u % 12, name="r12")
```

```asm
	v_mul_hi_u32 v0, v3, 3435973837                              // q5 = u // 5
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshrrev_b32 v0, 2, v0
	v_mul_hi_u32 v1, v3, 2863311531                              // r12 = u % 12
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshrrev_b32 v1, 3, v1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_mul_lo_u32 v2, v1, 12
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_sub_nc_u32 v1, v3, v2
```

```python
print(magic(5), magic(7), magic(12))
```

```text
(3435973837, 2, True) (613566757, 3, False) (2863311531, 3, True)
```

A divisor argument carries its multiplier and shift, computed at bind
([6.2](#62-argument-kinds)): the division is the long sequence, of
registers. Division of an i32, or by zero, is refused:

```python
u = V.u32(name="u", at=3)
V.u32(u % 0)
```

```error
ExprError: u % 0: by a positive constant or a divisor argument (u32.within(lo, hi).divisor())
```

## 3.5 Floats

f32 arithmetic is IEEE single with round to nearest even, denormals kept,
on the VALU only: an f32 tree with scalar leaves still computes into a
VGPR value. `fma(a, b, c)` is `v_fma_f32` (one rounding); `sqrt`, `rcp`,
`rsq`, `exp2`, `log2` are the hardware's transcendentals (`v_sqrt_f32`,
`v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32`), with the ISA's
precision, not correctly rounded; `min`, `max`, `min3`, `max3`, `med3`
are the `_num_` forms, which return the number when one operand is a NaN.

```python
fa, fb, fc = V.f32(name="fa", at=5), V.f32(name="fb", at=6), V.f32(name="fc", at=7)
unf = V.f32(fc + fa * fb, name="unf")
fused = V.f32(fma(fc, fb, -1.0), name="fused")
mods = V.f32(abs(fb) * -fc, name="mods")
t = V.f32(rcp(sqrt(fa)), name="t")
```

```asm
	v_mul_f32 v0, v5, v6                                         // unf = fc + (fa * fb)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32 v1, v7, v0
	v_fma_f32 v0, v7, v6, -1.0                                   // fused = fma(fc, fb, -1.0)
	v_mul_f32_e64 v2, |v6|, -v7                                  // mods = |fb| * -fc
	v_sqrt_f32 v3, v5                                            // t = rcp(sqrt(fa))
	s_delay_alu 5                                                // hazard: instid0(TRANS32_DEP_1)
	v_rcp_f32 v4, v3
```

`-x` and `abs(x)` are source modifiers on an operand of a float
instruction, and a sign-bit `v_xor_b32` / `v_and_b32` alone. A float
constant is an inline constant where the ISA has one (0.5, 1.0, 2.0,
4.0 and their negatives, 0.0), else a 32-bit literal; `-0.0` is the
literal `0x80000000`. An instruction takes one literal: a second is
moved into a register first.

## 3.6 Conversions and storage formats

| Function | Converts |
|---|---|
| `cvt_f32(x)` | a u32 or i32 to f32 (`v_cvt_f32_u32`, `v_cvt_f32_i32`) |
| `cvt_u32(x)`, `cvt_i32(x)` | an f32 to u32 or i32, rounding toward zero, saturating (`v_cvt_u32_f32`, `v_cvt_i32_f32`) |
| `unpack(w, fmt, at=0)` | element `at` of the dword `w` in a storage format to f32: `"f16"` (`at` 0 or 1), `"e4m3fn"`, `"e5m2"` (`at` 0) |
| `unpack2(w, fmt, at=0)` | two fp8 elements of `w` to an f32x2 (`v_cvt_pk_f32_fp8`, `v_cvt_pk_f32_bf8`) |
| `pack2(a, b, fmt, into=w, at=0)` | two f32 to two fp8 elements, word `at` of `w`, its other bits kept: written into `w` itself |
| `bf16_lo(w)`, `bf16_hi(w)` | the low or high bf16 of a dword to f32 (a shift or a mask: exact) |
| `pk_bf16(a, b)`, `bf16(x)` | two f32 to a bf16 pair, and an f32 rounded through bf16: gfx1201 has no `v_cvt_pk_bf16_f32`, so these are refused |

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
h1 = V.f32(unpack(u, "f16", at=1), name="h1")
pq = V.f32x2(unpack2(w, "e4m3fn"), name="pq")
out = V.u32(7, name="out")
out[...] = pack2(pq.hi, pq.lo, "e5m2", into=out)
hi = V.f32(bf16_hi(u), name="hi")
```

```asm
	v_lshrrev_b32 v0, 16, v3                                     // h1 = unpack(u, f16, at=1)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cvt_f32_f16 v1, v0
	v_cvt_pk_f32_fp8 v[6:7], v4                                  // pq = unpack2(w, e4m3fn)
	v_mov_b32 v0, 7                                              // out = 7
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cvt_pk_bf8_f32 v0, v7, v6                                  // out = pack2(pq.hi, pq.lo, e5m2, into=out)
	v_and_b32 v2, 4294901760, v3                                 // hi = bf16_hi(u)
```

## 3.7 Masks and select

A compare is a condition. Written into `S.mask(...)` it is a lane mask
(each active lane's bit is its compare); combined with `&`, `|`, `^`
and `~`, conditions and masks make masks, `~m` being the active lanes
not in `m`:

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k = S.u32(name="k", at=2)
ma = S.mask(u >= k, name="ma")
mb = S.mask((w == 0) | ma, name="mb")
mc = S.mask(~mb, name="mc")
```

```asm
	v_cmp_ge_u32_e64 s0, v3, s2                                  // ma = u >= k
	v_cmp_eq_u32_e64 s1, v4, 0                                   // mb = (w == 0) | ma
	s_or_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_and_not1_b32 s0, exec_lo, s2                               // mc = ~mb
```

`select(c, a, b)` is `a` in each lane where the condition or mask `c`
holds and `b` elsewhere; of scalar operands and a scalar compare it is
`s_cselect_b32` (`_b64` for a u64). `popcount(m)` counts a mask's lanes,
`first(m)` is the lowest lane in it (`s_ctz_i32_b32`), `rank(m, base=0)`
is, in each lane, `base` plus the lanes of `m` below it (`v_mbcnt_lo`).
`popcount(x)` of a u32 counts its bits.

```python
m = S.mask(name="m", at=4)
r = V.u32(popcount(m) + rank(m), name="r")
f = S.u32(first(m), name="f")
```

```asm
	s_bcnt1_i32_b32 s0, s4                                       // r = popcount(m) + rank(m, 0)
	v_mbcnt_lo_u32_b32 v0, s4, 0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_nc_u32 v1, s0, v0
	s_ctz_i32_b32 s0, s4                                         // f = first(m)
```

`any(c)` and `all(c)` of a lane condition are wave-uniform conditions,
for branches ([4.1](#41-branches)). The DSL's `min`, `max`, `any` and
`all` are Python's own when no argument is an expression.

## 3.8 Lane operations

| Function | Is |
|---|---|
| `first_lane(x)` | `x` in the lowest active lane, a scalar (`v_readfirstlane_b32`) |
| `lane_read(x, k)` | `x` in lane `k` (a scalar or a constant), a scalar (`v_readlane_b32`) |
| `lane_write(x, k, s)` | `x` with lane `k` set to the scalar `s` (`v_writelane_b32`), written into `x` itself |
| `shuffle(x, j)` | in each lane, `x` of lane `j` (`ds_bpermute_b32`; the Builder waits for it) |
| `across(x, op, lanes)` | `op` (`"add"`, `"min"`, `"max"`) over each aligned group of `lanes` lanes (2 .. 32), in every lane of the group |
| `scan(x, op, lanes)` | the inclusive prefix `op` over each group of `lanes` lanes (16 or 32) |

```python
fa = V.f32(name="fa", at=5)
ra = V.f32(across(fa, "add", 8), name="ra")
k = S.u32(name="k", at=1)
r = S.f32(lane_read(fa, k), name="r")
```

```asm
	v_add_f32_dpp v0, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf // ra = across(fa, add, 8)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32_dpp v0, v0, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_f32_dpp v0, v0, v0 row_half_mirror row_mask:0xf bank_mask:0xf
	v_readlane_b32 s0, v5, s1                                    // r = lane_read(fa, k)
```

64 lanes are refused on a 32-lane wave.

## 3.9 64-bit arithmetic

A u64 or ptr takes `+` and `-` of another u64, a u32 (zero-extended), an
i32 (sign-extended) or a constant in -2^63 .. 2^64 - 1, as an add and a
carry add; it compares with the six compares on the VALU, with `==` and
`!=` on the SALU. Anything else is done on its halves.

```python
base = S.ptr(name="base", at=4)
w = V.u32(name="w", at=3)
i = V.i32(name="i", at=8)
a = V.u64(base + w, name="a")
c = S.u64(base - 40, name="c")
d = V.u64(a + i, name="d")
```

```asm
	v_mov_b32 v0, s5                                             // a = base + w
	v_add_co_u32 v2, vcc_lo, s4, v3
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_co_ci_u32 v3, vcc_lo, 0, v0, vcc_lo
	s_sub_co_u32 s0, s4, 40                                      // c = base - 40
	s_sub_co_ci_u32 s1, s5, 0
	v_ashrrev_i32 v0, 31, v8                                     // d = a + i
	s_delay_alu 3                                                // hazard: instid0(VALU_DEP_3)
	v_add_co_u32 v4, vcc_lo, v2, v8
	s_wait_alu depctr_va_vcc(0)                                  // hazard
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_co_ci_u32 v5, vcc_lo, v3, v0, vcc_lo
```

`pair(a, b)` makes an f32x2 of two f32 trees; `-p` negates both halves.
gfx1201 has no packed f32 arithmetic: `+`, `-`, `*` and `fma` of f32x2
are refused.

## 3.10 Any instruction as a node

`op(name, *args, ty=None, **fields)` is an instruction as a tree node,
its first operand the destination the statement gives it. Its result's
type is the mnemonic's (`v_add3_u32`: u32), or `ty=` for a dotted name or
a mnemonic without one.

```python
u, w = V.u32(name="u", at=3), V.u32(name="w", at=4)
k = S.u32(name="k", at=1)
o1 = V.u32(op("v_add3_u32", w, k, u) << 3, name="o1")
o2 = V.u32(op("v.mul.lo.u32", w, 6, ty=u32), name="o2")
```

```asm
	v_add3_u32 v0, v4, s1, v3                                    // o1 = v_add3_u32(w, k, u) << 3
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshlrev_b32 v1, 3, v0
	v_mul_lo_u32 v0, v4, 6                                       // o2 = v.mul.lo.u32(w, 6)
```

# 4. Control flow

Control flow is written with context managers and Python `for` loops
over the Builder's constructs. A branch condition must be wave-uniform:
the whole wave takes one side. A condition that varies by lane is a lane
region instead, which runs its block in every wave with EXEC narrowed.

## 4.1 Branches

`with b.when(c):` runs its block when `c` holds, `b.unless(c)` when it
does not; `b.elif_(c)` and `b.otherwise()` continue the chain right
after an arm. A condition is a scalar compare, `&` and `|` of scalar
compares (branching on each in turn, short-circuit), `any(c)` or
`all(c)` of a lane condition, or an `s_cmp*` / `s_bitcmp*` instruction.

```python
sel = S.u32(name="sel", at=3)
out = S.u32(name="out")
with b.when(sel < 2):
    out[...] = 100
with b.elif_((sel == 5) | (sel > 9)):
    out[...] = 200
with b.otherwise():
    out[...] = sel + 7
out += sel
```

```asm
	s_cmp_lt_u32 s3, 2                                           // sel < 2
	s_cbranch_scc0 .Lif_1
	s_mov_b32 s0, 100                                            // out = 100
	s_branch .Lif_end_2
.Lif_1:
	s_cmp_eq_u32 s3, 5                                           // sel == 5
	s_cbranch_scc1 .Lj_4
	s_cmp_gt_u32 s3, 9                                           // sel > 9
	s_cbranch_scc0 .Lif_3
.Lj_4:
	s_mov_b32 s0, 200                                            // out = 200
	s_branch .Lif_end_2
.Lif_3:
	s_add_co_u32 s0, s3, 7                                       // out = sel + 7
.Lif_end_2:
	s_add_co_u32 s0, s0, s3                                      // out = out + sel
```

`name=` sets the labels' stem, `end=` the join label, `comment=` the
branch's comment. A comment, a label or an instruction between two arms
closes the chain. A lane condition is refused:

```python
col = V.u32(name="col", at=2)
with b.when(col >= 6):
    pass
```

```error
ExprError: col >= 6: a lane condition in a branch (wave-uniform): any(c) or all(c), or an exec region (b.lanes)
```

```python
col = V.u32(name="col", at=2)
r = S.u32(0, name="r")
with b.when(any(col >= 6)):
    r[...] = 1
r += 3
```

```asm
	s_mov_b32 s0, 0                                              // r = 0
	v_cmp_le_u32 vcc_lo, 6, v2                                   // any(col >= 6)
	s_cmp_lg_u32 vcc_lo, 0
	s_cbranch_scc0 .Lif_1
	s_mov_b32 s0, 1                                              // r = 1
.Lif_1:
	s_add_co_u32 s0, s0, 3                                       // r = r + 3
```

`b.exit_when(c, to="L_end")` branches to the label `to` when `c` holds;
the label is placed by `b.put_label(to, from_branches=True)`.

## 4.2 Loops

`for it in b.loop(...)` writes a loop; its body is the `for` block, run
once by Python (twice under the audit option, [7.8](#78-the-audit)).

| Call | Loop |
|---|---|
| `b.loop(n)` | counted: `it.i` from `start=0` to `n - 1` (an SGPR, `i=` to give one); `n` an int or a scalar u32 |
| `b.loop(until=c)` | the body, then again while `c` is false |
| `b.loop()` | forever, left by `it.break_when(c)` |

```python
trips = S.u32(name="trips", at=5)
total = V.u32(3, name="total")
for it in b.loop(trips, name="row"):
    total += it.i * 3
```

```asm
	v_mov_b32 v0, 3                                              // total = 3
	s_mov_b32 s0, 0
	s_cmp_lt_u32 s0, s5
	s_cbranch_scc0 .Lrow_end_2
.Lrow_1:
	s_mul_i32 s1, s0, 3                                          // total = total + (row.i * 3)
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	v_add_nc_u32 v0, s1, v0
	s_add_co_u32 s0, s0, 1
	s_wait_alu depctr_sa_sdst(0)                                 // hazard
	s_cmp_lt_u32 s0, s5
	s_cbranch_scc1 .Lrow_1
.Lrow_end_2:
```

A loop with a count not known at compile time tests it before the first
trip; a constant count does not. `it.break_when(c, comment=None)` leaves
the loop when `c` holds, to its end label (`end=` names it). A Python
`break` is refused: the loop's back edge would never be emitted.

```python
lim = S.u32(name="lim", at=1)
v = S.u32(1, name="v")
for it in b.loop(until=v > 90):
    v += 11
    it.break_when(v >= lim)
```

```asm
	s_mov_b32 s0, 1                                              // v = 1
.Luntil_1:
	s_add_co_u32 s0, s0, 11                                      // v = v + 11
	s_cmp_ge_u32 s0, s1                                          // v >= lim
	s_cbranch_scc1 .Luntil_end_2
	s_cmp_gt_u32 s0, 90                                          // v > 90
	s_cbranch_scc0 .Luntil_1
.Luntil_end_2:
```

`peel=k` emits the body up to `k + 2` times and keeps the copies until
two in a row are the same, waits included: the first trips of a
software-pipelined loop, whose waits differ from the steady state's,
stay apart from the loop. A body that never repeats is refused.
`conservative=True` lets the waits pass use, for the loop, the larger of
its entry's counts and its steady state's.

## 4.3 Exits and labels

`b.put_label(name)` places a label, `b.new_label(stem)` makes a fresh
name `.L<stem>_<k>`. A branch instruction emitted with `O.op` names its
label as a string. Labels are the only way control enters a point other
than its predecessor.

## 4.4 Lane regions

`with b.lanes(*conds, save=None, skip=False, comment=""):` runs its block
with EXEC narrowed to the lanes where every condition holds. A condition
is a lane compare, a mask, a constant mask, or a compare instruction
into VCC or SCC (the last one); a wave-uniform compare (SCC) may follow
one mask at most.

```python
col, val = V.u32(name="col", at=2), V.u32(name="val", at=3)
keep = S.mask(name="keep", at=6)
with b.lanes(col < 30, keep):
    val += 6
with b.lanes(val != 1):
    val += 4
```

```asm
	v_cmp_lt_u32_e64 s0, v2, 30                                  // col < 30
	s_and_b32 exec_lo, s0, s6
	v_add_nc_u32 v3, 6, v3                                       // val = val + 6
	s_mov_b32 exec_lo, -1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cmp_ne_u32 vcc_lo, 1, v3                                   // val != 1
	s_mov_b32 exec_lo, vcc_lo
	s_delay_alu 2                                                // hazard: instid0(VALU_DEP_2)
	v_add_nc_u32 v3, 4, v3                                       // val = val + 4
	s_mov_b32 exec_lo, -1
```

A region outside any other is entered by writing EXEC and left by
setting every lane; it requires every lane on when it is entered (a
partial wave must use `save=`), checked at compile time. A nested region,
or one with `save=True` or `save=m` (a mask value), saves EXEC with
`s_and_saveexec_b32` and restores it at the end. `skip=True` branches
over the block when no lane is on.

```python
col, val = V.u32(name="col", at=2), V.u32(name="val", at=3)
with b.lanes(col >= 8):
    val -= 1
    with b.lanes(val == 0, skip=True):
        val[...] = col + 1
```

```asm
	v_cmp_le_u32 vcc_lo, 8, v2                                   // col >= 8
	s_mov_b32 exec_lo, vcc_lo
	v_subrev_nc_u32 v3, 1, v3                                    // val = val - 1
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_cmp_eq_u32 vcc_lo, 0, v3                                   // val == 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .Llanes_none_1
	v_add_nc_u32 v3, 1, v2                                       // val = col + 1
.Llanes_none_1:
	s_mov_b32 exec_lo, s0
	s_mov_b32 exec_lo, -1
```

Inside a region, `b.narrow(c)` narrows EXEC further, to the region's
end. An instruction that writes EXEC, or a branch out of the region, is
refused: the region owns EXEC. `with b.raw_exec(why):` lets the block
write EXEC itself.

```python
col = V.u32(name="col", at=2)
with b.lanes(col >= 4):
    b.emit(O.op("s_or_b32", O.EXEC, O.EXEC, 1))
```

```error
ScheduleError: `s_or_b32 exec_lo, exec_lo, 1` writes exec inside the lane region 1 (vcc): name it (Builder.raw_exec(why)) or make it a region
```

## 4.5 Wave blocks

`b.wave_index(reg, nwaves, by_wave=False)` names the SGPR that holds the
wave's index in its workgroup (of `nwaves`). `with b.waves(ws):` runs its
block in the waves of the contiguous list `ws` only, by a branch on that
register. With `by_wave=True` the checks treat each class of waves apart
(each takes its own path through the wave blocks), which the LDS race
check needs to prove that waves running apart touch their own bytes.

```python
widx = S.u32(name="widx", at=1)
b.wave_index(widx, 8)
acc = V.u32(name="acc", at=0)
with b.waves([4, 5, 6, 7]):
    acc[...] = acc + 9
```

```asm
	s_cmp_lt_u32 s1, 4
	s_cbranch_scc1 .Lwaves_1
	v_add_nc_u32 v0, 9, v0                                       // acc = acc + 9
.Lwaves_1:
```

# 5. Waits and hazards

A memory instruction issues and returns; its result lands later. gfx1201
counts what is in flight on four counters, `load` (vector-memory loads),
`store` (vector-memory stores), `ds` (LDS) and `km` (scalar memory), and
`s_wait_loadcnt n` (and `s_wait_storecnt`, `s_wait_dscnt`,
`s_wait_kmcnt`) waits until at most `n` of its kind are left. The author
says what must have landed; the Builder works out each count.

## 5.1 Waiting

`b.wait(*what, left=None, tags=(), written=False, comment="", abl=())`
asks, at that point, for what it names to have landed:

| Argument | Waits for |
|---|---|
| a value or view | every load that writes it |
| an LDS region | every LDS op on it ([9.1](#91-layouts)) |
| `O.VM`, `O.LGKM`, or a counter's name (`"load"`, `"store"`, `"ds"`, `"km"`, `"vm"`, `"lgkm"`, `"smem"`) | everything on those counters; `left=n`: all but `n` |
| `tags=[t, ...]` | the vector-memory ops emitted with `tag=t`; `written=True`: and the registers or placed names they write |
| a trailing string | nothing: it is the wait's comment |

```python
src = S.ptr(name="src", at=2)
idx = V.u32(name="idx", at=1)
lo, hi = V.f32(name="lo"), V.f32(name="hi")
b.emit(O.global_load(hi, idx, src, offset=32))
b.emit(O.global_load(lo, idx, src))
b.wait(hi, "hi landed")
hi[...] = hi + 0.5
b.wait(lo)
lo[...] = lo * hi
```

```asm
	global_load_b32 v0, v1, s[2:3] offset:32
	global_load_b32 v2, v1, s[2:3]
	s_wait_loadcnt 1                                             // hi landed
	v_add_f32 v0, 0.5, v0                                        // hi = hi + 0.5
	s_wait_loadcnt 0
	v_mul_f32 v2, v2, v0                                         // lo = lo * hi
```

Loads of one counter land in order, so the count is the number of later
loads that may still be in flight: `hi` has landed once one load is
left. The count is the largest that is enough on every path to the
wait, loops included: a load issued in a loop body and waited for at the
top of the next trip is counted over the back edge.

```python
src = S.ptr(name="src", at=2)
idx = V.u32(name="idx", at=1)
trips = S.u32(name="trips", at=5)
acc = V.f32(0.0, name="acc")
x = V.f32(name="x")
b.emit(O.global_load(x, idx, src))
for it in b.loop(trips):
    b.wait(x, "this trip's x")
    acc += x
    b.emit(O.global_load(x, idx, src, offset=4))
```

```asm
	v_mov_b32 v0, 0                                              // acc = 0.0
	global_load_b32 v2, v1, s[2:3]
	s_mov_b32 s0, 0
	s_cmp_lt_u32 s0, s5
	s_cbranch_scc0 .Lloop_end_2
.Lloop_1:
	s_wait_loadcnt 0                                             // this trip's x
	v_add_f32 v0, v0, v2                                         // acc = acc + x
	global_load_b32 v2, v1, s[2:3] offset:4
	s_add_co_u32 s0, s0, 1
	s_cmp_lt_u32 s0, s5
	s_cbranch_scc1 .Lloop_1
.Lloop_end_2:
```

A wait for a value no load writes is refused, as is one for a tag never
issued, and a wait with nothing to wait for:

```python
t = V.u32(5, name="t")
b.wait(t)
```

```error
ScheduleError: wait: v:t#1[0]x1 was written by no load
```

## 5.2 What the Builder waits for itself

Some waits the Builder places without being asked, each with its
comment:

- **Kernel arguments**, loaded by `k.load` ([6.4](#64-loading-arguments)),
  are waited for at their first read (`// kernel arguments`), and before
  a loop that has not read them yet. `b.land_arguments(why)` waits there
  for every argument loaded and not yet read.
- **`shuffle`** waits for its `ds_bpermute_b32` (`// the shuffle`).
- **`b.wait_on_use(regs)`** waits for each of `regs` at its first read,
  as for an argument.

## 5.3 What is refused

Every register an instruction reads or writes must have landed: the
waits pass refuses an instruction that touches a register a load may
still be writing, on any path, and names the count it needs.

```python
src = S.ptr(name="src", at=2)
idx = V.u32(name="idx", at=1)
lo = V.f32(name="lo")
b.emit(O.global_load(lo, idx, src))
lo[...] = lo * 2.0
```

```error
ScheduleError: v_mul_f32 v:lo#1, 2.0, v:lo#1 (after the entry) touches a register while the load on it may be in flight on some path: it needs loadcnt(0) - add a wait in the schedule
```

This is a check, not a repair: the wait is the author's, where the
author wants it, because where a wait goes decides what overlaps.

## 5.4 Hazards

The hazard pass inserts what gfx1201 needs between dependent
instructions that its hardware does not interlock, after placement (the
registers decide what depends on what). Each is commented `// hazard`:

| Instruction | Placed |
|---|---|
| `s_delay_alu instid0(...)` | before a VALU instruction that reads the result of a recent VALU (`VALU_DEP_n`, the producer `n` instructions back) or transcendental (`TRANS32_DEP_n`) instruction: a delay hint, which the hardware needs for correct timing only |
| `s_wait_alu depctr_*(0)` | before an instruction that reads an SGPR, VCC or SCC a recent SALU or VALU instruction wrote (`depctr_sa_sdst`, `depctr_va_vcc`, `depctr_va_sdst`, ...) |
| `v_nop`, `s_nop n` | between a WMMA and an instruction that reads its result as a matrix operand, and the ISA's other wait-state rules |

```python
fa = V.f32(name="fa", at=5)
t = V.f32(rcp(fa) + 1.0, name="t")
p = V.u64(name="p", at=2)
w = V.u32(name="w", at=4)
a = V.u64(p + w, name="a")
c = V.u64(a + w, name="c")
```

```asm
	v_rcp_f32 v0, v5                                             // t = rcp(fa) + 1.0
	s_delay_alu 5                                                // hazard: instid0(TRANS32_DEP_1)
	v_add_f32 v1, 1.0, v0
	v_add_co_u32 v6, vcc_lo, v2, v4                              // a = p + w
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_co_ci_u32 v7, vcc_lo, 0, v3, vcc_lo
	s_delay_alu 2                                                // hazard: instid0(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v6, v4                              // c = a + w
	s_wait_alu depctr_va_vcc(0)                                  // hazard
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_add_co_ci_u32 v3, vcc_lo, 0, v7, vcc_lo
```

`b.hazards` is the number of hazard instructions the pipeline placed.
The Builder never moves an instruction to hide a hazard: the author's
order is kept, and the pass only pads it.

# 6. Entry and arguments

A kernel's inputs are its argument block in memory, which the hardware
points to, and the registers it fills at entry: the block's address, the
workgroup ids and each lane's id in its workgroup.

## 6.1 The argument block

`Args(at=None, **kinds)` lays out the block in keyword order, each
argument aligned to its size (at most 8); `at={name: offset}` places
arguments at byte offsets of the author's.

```python
a = Args(x=ptr, n=u32, scale=f32, y=ptr)
print(a.size, a.names(), a["y"]["offset"], a["scale"]["offset"])
```

```text
24 ['x', 'n', 'scale', 'y'] 16 12
```

`a.size` is the block's end, `a.names()` every name in order,
`a.own()` the names a library did not add, `a[name]` an argument's
record (`offset`, `bytes`, `kind`, `lo`, `hi`, `divisor`, `multiple`,
`owned`, `library`, `mem`, `ranks`).

## 6.2 Argument kinds

| Kind | The argument |
|---|---|
| `ptr` | a buffer: bind gives an array or tensor, the kernel gets its address |
| `u32`, `i32`, `u64`, `f32` | a number |
| `T.within(lo, hi)` | a number in `lo .. hi`: `bind` refuses others, and the range checks rely on it ([7.5](#75-ranges)) |
| `k.multiple_of(m)` | of a ranged kind: a multiple of `m` |
| `u32.within(lo, hi).divisor()` | a u32 the kernel divides by: 3 dwords, the divisor, its magic multiplier (`.mul`) and shift (`.shr`), computed by `bind` |
| `scratch` | memory of the kernel's own, of a size given at bind, any bytes at entry |
| `kept` | own memory, zero at first, which every launch must leave as it found it (checked) |
| `carried` | own memory, zero at first, kept from one launch to the next |
| `gens` | carried generation words, starting at 2^32 - 3 |
| `kept.uncached`, `carried.uncached`, `gens.uncached` | the same, every access bypassing the L2 |
| `mapped(w)` | a group's carried memory, one buffer per rank of `w` (2 .. 8): the kernel gets every rank's address |
| `peers(w)` | every rank's copy of a group-shared tensor |
| `lib_arg(kind, library)` | an argument a library adds and loads itself |

A kind is a `Kind`; `T.kind` is a type as one. Own memory is given at
bind as its size in bytes (`kept` also as its first bytes); the runtime
allocates it and holds it between launches, by the kernel's identity.

## 6.3 The entry

`b.entry(args=None, *, wg=1, preload=0, wgsize=None)` makes the
kernel's entry, once, and returns the `Entry` `k`:

| Member | Is |
|---|---|
| `k.karg` | the block's address, `s[0:1]` at entry |
| `k.tid` | the lane's id in its workgroup, v0 at entry |
| `k.wgid`, `k.wgid_y` | the workgroup ids x and y (`wg=2`), read from `ttmp9` and `ttmp7` into SGPRs of their own |
| `k.lane` | the lane's index in its wave, `tid & 31` (`tid` itself when `wgsize` is 32) |
| `k.wave` | the wave's index in its workgroup, `first_lane(tid) >> 5`, a scalar (0 when `wgsize` is 32) |
| `k.wgsize`, `k.wg`, `k.preload` | as given; `preload` dwords of arguments the hardware would load into SGPRs (gfx1201: none) |
| `k.entry_moves` | the values the workgroup-id moves write, which compilation deletes when nothing reads them |

```python
k = b.entry(Args(x=ptr, n=u32), wgsize=256)
x, n = k.load()
i = V.u32(k.wgid * 256 + k.tid, name="i")
w = V.u32(k.lane + k.wave, name="w")
```

```asm
	s_mov_b32 s2, ttmp9                                          // wgid#1: ttmp9
	s_load_b64 s[4:5], s[0:1], 0x0                               // x
	s_load_b32 s3, s[0:1], 0x8                                   // n
	v_lshl_add_u32 v1, s2, 8, v0                                 // i = (wgid * 256) + tid
	v_and_b32 v2, 31, v0                                         // lane = tid & 31
	v_readfirstlane_b32 s0, v0                                   // wave = first_lane(tid) >> 5
	s_lshr_b32 s1, s0, 5
	s_delay_alu 2                                                // hazard: instid0(VALU_DEP_2)
	v_add_nc_u32 v0, s1, v2                                      // w = lane + wave
```

## 6.4 Loading arguments

`k.load(*names, comment="")` loads the named arguments (every argument,
none named) into SGPR values of their kinds, and returns them in order: a
ptr as a ptr, a divisor as its three dwords. Arguments next to one
another load with one `s_load` of the widest form that covers them; the
wait is placed at their first read ([5.2](#52-what-the-builder-waits-for-itself)).

```python
k = b.entry(Args(cols=u32, src=ptr, ld=u32, rows=u32))
ld, rows = k.load("ld", "rows")
span = S.u32(ld * 3 - rows, name="span")
```

```asm
	s_load_b64 s[2:3], s[0:1], 0x10                              // ld, rows
	s_wait_kmcnt 0                                               // kernel arguments
	s_mul_i32 s0, s2, 3                                          // span = (ld * 3) - rows
	s_sub_co_u32 s1, s0, s3
```

`k.load_rank(name, d)` loads rank `d`'s address of a `mapped` or `peers`
argument. A name not in the block is refused:

```python
k = b.entry(Args(src=ptr, n=u32))
k.load("n", "cnt")
```

```error
ScheduleError: load('cnt'): no such argument (the block: src, n)
```

## 6.5 Grids

A launch runs a grid of workgroups of `wgsize` lanes, in waves of 32.
The grid is given at bind ([11.2](#112-binding-and-running)); the
`KernelDef`'s `max_grid` is the largest grid the kernel is compiled for,
which bounds `k.wgid` for the range checks.

# 7. Checks

Compilation checks the kernel and refuses it with the first failure. Some
refusals come as the Builder runs (the maker's, [2.5](#25-what-an-instruction-is-refused-for);
the expressions', [3](#3-expressions); control flow's, [4](#4-control-flow)),
the others from the passes. All are `ValueError`s: `ExprError`,
`FlowError`, `MemError`, `AssignError` and `RangeError` are kinds of
`ScheduleError`, itself a `Diag`, which carries the author's source line.

## 7.1 Dead writes

A write whose value is overwritten on every path before any read is
refused: it does nothing, and is usually a mistake.

```python
w = V.u32(name="w", at=4)
b.emit(O.op("v_mov_b32", w, 1))
b.emit(O.op("v_mov_b32", w, 2))
```

```error
ScheduleError: `v_mov_b32 v4, 1` (kernel.py:2): its value is written over on every path before a read (`v_mov_b32 v4, 2` (kernel.py:3) writes it again) (dead write): drop the write, or b.discard(reg, why) where it is not read; an ablation variant that switched the reader off - b.ablate
```

`b.discard(reg, why)` says a register's contents are not read (a write
kept on purpose). The check is off under ablation
([10.3](#103-ablation)) and with the `keep_dead` option.

## 7.2 SCC

An instruction that reads SCC (`s_cselect_*`, `s_cbranch_scc*`,
`s_add_co_ci_u32`) must read the SCC of the instruction meant for it: a
select or branch reads a compare's, a carry add an add's or a sub's.
SALU arithmetic writes SCC too, so an instruction between the two
breaks the pair:

```python
sel = S.u32(name="sel", at=1)
b.emit(O.op("s_cmp_eq_u32", sel, 2))
x = S.u32(sel + 1, name="x")
r = S.u32(name="r")
b.emit(O.op("s_cselect_b32", r, 1, x))
```

```error
ScheduleError: `s_cselect_b32 s1, 1, s0` (kernel.py:5): SCC was last written by s_add_co_u32 (arith), not a compare
```

## 7.3 Lane regions

A region without a save register must be entered with every lane on: it
sets EXEC, and sets every lane on at its end ([4.4](#44-lane-regions)). A
region entered where EXEC may be partial is refused:

```python
col = V.u32(name="col", at=2)
b.emit(O.op("s_mov_b32", O.EXEC, 0xFF))
with b.lanes(col < 9):
    col += 1
```

```error
ScheduleError: `s_mov_b32 exec_lo, vcc_lo` (instruction 2) enters a lane region without a save register where exec is 0xff: every lane must be on (else pass save=)
```

## 7.4 Aliases

Placement shares registers between values that are never live together.
A value written on some paths only, then read, would read whatever value
held its registers on the others. That is refused:

```python
sel = S.u32(name="sel", at=1)
t = S.u32(sel * 3, name="t")
r = S.u32(name="r")
with b.when(sel == 2):
    r[...] = t + 1
out = S.u32(r + t, name="out")
```

```error
ScheduleError: register alias: a value read through another name than it was written through (in a derived file: a value read where some path did not write it)
  `s_add_co_u32 s2, s1, s0` (instruction 5) reads s1 through `r#1`, but on a path to it (through .Lif_1) `r#1` is not written: s1 holds `sel#1` there, written by the hardware at entry. A derived file gives a value's registers to others before its first write: write `r#1` on every path to the read (all its dwords; a default before the branch), or - when that path never runs (the read's branch taken only when the write's was: one condition) - say so around the read: `with b.alias_ok("r#1", why=..)`
```

`with b.alias_ok(*names, over=(), why=""):` says the path never runs
(the read's branch is taken only when the write's was).

## 7.5 Ranges

With a `max_grid` ([6.5](#65-grids)), compilation bounds every value
from the arguments' ranges, the grid and the workgroup size, and refuses
a memory address whose 32-bit offset may wrap, an LDS address outside
its region, and a 24-bit multiply operand that may pass 24 bits. Without
one, a wrap it cannot exclude is counted as unproven in the summary.

```python
k = b.entry(Args(out=ptr), wgsize=64)
(out,) = k.load()
i = V.u32(k.wgid * 64 + k.tid, name="i")
o = V.u32(i * 4, name="o")
b.emit(O.global_store(o, i, out))
compile_kernel(b, "k", wgsize=64, max_grid=(1 << 26,))
```

```error
RangeError: `global_store_b32 v:o#1, v:i#1, s:out#1x2` (instruction 6): its address offset may wrap past 32 bits - `v_lshlrev_b32 v:o#1, 2, v:i#1` (instruction 4) computes [0, 4294967295] << 2: 0 .. 17179869180, every u32 value
```

`b.assume(x, lo=0, hi=, why=)` states a range the check cannot derive.

## 7.6 LDS

The LDS proof ([9.4](#94-what-is-proved-about-lds)) refuses regions that
share bytes and are live together, an LDS op still in flight at a
barrier whose bytes an op after it touches, and accesses of two waves to
the same bytes, a write among them, with no barrier between:

```python
k = b.entry(Args(), wgsize=64)
L = b.lds()
T = L.region("T", (64,), "u32")
x = V.u32(k.tid * 3, name="x")
b.emit(O.ds_write(T.at(k.tid), x))
b.barrier()
y = V.u32(name="y")
b.emit(O.ds_read(y, T.at(63 - k.tid)))
b.wait(y)
```

```error
MemError: `ds_store_b32 T + tid * 4` (instruction 2) is in flight at the barrier (instruction 4), and `ds_load_b32 T + (63 - tid) * 4` (instruction 8) accesses its bytes after it: another wave may run it before the op lands (wait before the barrier: b.wait(T))
```

## 7.7 Placement

Placement refuses a value read before any write ([1.4](#14-placement)),
pins that overlap while live, and a kernel whose registers do not fit:
the file's size, or the budget that the workgroup size sets (a
workgroup's waves share the four SIMDs of one WGP, each with 1536 VGPRs a
lane: a workgroup of 1024 lanes puts 8 waves on each SIMD, at 192 VGPRs a
wave at most).

## 7.8 The audit

The `audit` option (`codegen.using(audit=True)`, [11.1](#111-kerneldef-and-compile))
emits each loop body twice and refuses a body whose second pass differs
from its first, waits included: the loop's steady state must be its
first trip. It then builds the kernel a second time with the first
placement, and refuses a second run that differs.

## 7.9 Hazards and the verifier

After the hazard pass, a check confirms that no hazard is left
(`Assertion`, a bug of thin's if it fires), and the IR verifier checks
every instruction's form, fields and operands (`Diag`, with the line
that made it).

# 8. A whole kernel

Vector add: `z[i] = x[i] + y[i]` for `i < n`, 256 lanes a workgroup.

```python
k = b.entry(Args(x=ptr, y=ptr, z=ptr, n=u32), wgsize=256)
x, y, z, n = k.load()
i = V.u32(k.wgid * 256 + k.tid, name="i")
with b.lanes(i < n):
    off = V.u32(i * 4, name="off")
    a, c = V.f32(name="a"), V.f32(name="c")
    b.emit(O.global_load(a, off, x))
    b.emit(O.global_load(c, off, y))
    b.wait(a, c, "x, y landed")
    a += c
    b.emit(O.global_store(off, a, z))
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
	s_wait_loadcnt 0                                             // x, y landed
	v_add_f32 v1, v1, v2                                         // a = a + c
	global_store_b32 v0, v1, s[8:9]
	s_mov_b32 exec_lo, -1
```

Line by line:

1. `b.entry(Args(...), wgsize=256)` declares the block (three pointers at
   bytes 0, 8 and 16, `n` at 24) and the entry registers: the block's
   address in `s[0:1]`, the lane's id `tid` in v0, the workgroup id in
   `ttmp9`. Reading `k.wgid` moves it into an SGPR of its own
   (`s_mov_b32 s2, ttmp9`).
2. `k.load()` loads the four arguments: `x` and `y` with one
   `s_load_b128`, `z` and `n` with one load each. The wait for them
   (`s_wait_kmcnt 0`) is placed before their first read, the compare: the
   index is computed while they load.
3. `i = V.u32(k.wgid * 256 + k.tid)` is one `v_lshl_add_u32`: `* 256` is a
   shift by 8, fused with the add ([3.3](#33-rewrites)); `wgid` is an
   SGPR, read by the VALU as an operand.
4. `with b.lanes(i < n):` compares into VCC and sets EXEC to it: the lanes
   past `n` sit out the block. At its end, `s_mov_b32 exec_lo, -1` sets
   every lane on again.
5. `off = V.u32(i * 4)` is the byte offset, a shift. It takes v0: `tid` is
   no longer live.
6. The two loads issue back to back; `b.wait(a, c)` places one
   `s_wait_loadcnt 0`, since `c` is the last load issued.
7. `a += c` adds in place: the sum takes `a`'s register.
8. `global_store` writes the sum. No wait follows: a wave's stores
   complete after it ends.

Chapter 11 compiles the same kernel into a code object and runs it.

# 9. Memory

## 9.1 Layouts

LDS, the workgroup's shared memory (64 KiB a workgroup on gfx1201), is
declared as a layout of named regions; an LDS access names its region,
and the compiler proves things of every access by it.

`L = b.lds(budget=None)` makes the kernel's `Layout`, once, and
`L.region` a `Region` of it. `L.region(name,
shape=(), elem="u8", *, pitch=None, nbytes=None, align=None, at=None,
swizzle=None)` declares a region of `shape` elements of `elem` (a type,
`u32`, `f32`, ..., or a name: `b8`, `b32`, `b64`, `b128`, `u8`, `i8`,
`u16`, `i16`, `f16`, `bf16`, `e4m3`, `e5m2`, `e2m1x2`, `e8m0`), row-major.
`pitch=` is a row's bytes (a dict `{dim: bytes}` sets any dimension's
stride), `nbytes=` its size, `align=` its alignment, `at=` its byte
offset. Regions follow one another; `with L.union():` makes the regions
declared in it share their start, `with L.struct():` groups regions that
follow one another, each with `name=`, `align=` and `at=`. A layout's
table is printed at the top of the kernel:

```python
L = b.lds(budget=48 * 1024)
Hs = L.region("Hs", (32, 48), "f16", pitch=112)
with L.union():
    Acc = L.region("Acc", (16, 16), f32)
    Wt = L.region("Wt", (128, 32), "u8")
St = L.region("St", (4, 8, 64), "bf16")
s0, s1, s2, s3 = St.parts("s0", "s1", "s2", "s3")
print(L.size, Hs.strides, St.shape, s1.base)
```

```text
11776 (112, 2) (4, 8, 64) 8704
```

`R.parts(*names)` makes each entry of a region's first dimension a region
of its own. `L.size` and `R.base` freeze the layout: no region is
declared after. `L.regions`, `L.leaves()`, `L.table()`, and `R.name`,
`R.shape`, `R.elem`, `R.strides`, `R.size`, `R.align`, `R.swizzle`,
`R.parent` describe it. A layout past its budget, or past the 64 KiB a
workgroup has, is refused.

## 9.2 Addresses

An `Address` is a region plus a byte offset, an int or a u32 formula:
`R + t`, `R.at(*idx)` (the element's offset, `R.offset(*idx)`), and a
`Pointer`, a value holding an element's address (`R.pointer(*idx,
name=None)`), moved by constant indices with `p.at(*d)`. The DS
families split an address into a register and the instruction's
16-bit offset: the constant part goes into the field.

```python
L = b.lds()
Tile = L.region("Tile", (32, 96), "u32", pitch=400)
r, c = V.u32(name="r", at=3), V.u32(name="c", at=5)
acc = V.u32(n=3, name="acc")
cur = Tile.pointer(r, c, name="cur")
b.emit(O.ds_read(acc[0], cur.at(0, 3)))
b.emit(O.ds_read(acc[1], cur.at(2, -1)))
b.emit(O.ds_read(acc[2], Tile.at(r | 1, c + 2)))
b.wait(acc)
tot = V.u32(acc[0] ^ acc[1] ^ acc[2], name="tot")
b.emit(O.ds_write(cur.at(1, 1), tot))
```

```asm
; lds: 12800 B
;   offset     end   size  region           shape                strides
;        0   12800  12800  Tile             [32][96] u32         400 4
	v_mul_u32_u24 v0, 400, v3                                    // cur = mul24(r, 400) + (c * 4)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v1, v5, 2, v0
	ds_load_b32 v6, v1 offset:12                                 // Tile: cur + 12
	ds_load_b32 v7, v1 offset:796                                // Tile: cur + 796
	v_or_b32 v0, 1, v3                                           // adr = mul24(r | 1, 400) + (c * 4)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_mul_u32_u24 v2, 400, v0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v0, v5, 2, v2
	ds_load_b32 v8, v0 offset:8                                  // Tile + (mul24(r | 1, 400) + (c * 4)) + 8
	s_wait_dscnt 0
	v_xor_b32 v0, v6, v7                                         // tot = (acc[0] ^ acc[1]) ^ acc[2]
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_xor_b32 v2, v0, v8
	ds_store_b32 v1, v2 offset:404                               // Tile: cur + 404
```

In a kernel with a layout, an LDS access by a bare register is refused:
every access names its region. `b.wait(R)` waits for every LDS op on `R`
issued on any path; `R.tags()` are the names it answers to.

## 9.3 Swizzles

`swizzle=Xor(by, into, mask, shift=0)` stores element `c` of dimension
`into` at `c ^ ((idx[by] >> shift) & mask)`: rows of a column-read tile
land in different banks. Addresses are written with the logical indices;
the swizzle is applied in the formula. `R.stored(*idx)` is an element's
stored place, `R.offset(*idx, swizzled=False)` its unswizzled offset.

```python
L = b.lds()
Sw = L.region("Sw", (64, 4), "b128", swizzle=Xor(by=0, into=1, mask=3))
ln = V.u32(name="ln", at=7)
quad = V.u32(n=4, name="quad")
b.emit(O.ds_read(quad, Sw.at(ln >> 2, ln & 3)))
```

```asm
; lds: 4096 B
;   offset     end   size  region           shape                strides
;        0    4096   4096  Sw               [64][4] b128         64 16  swizzle [1] ^= [0] & 3
	v_and_b32 v0, 3, v7                                          // adr = ((ln >> 2) * 64) + (((ln & 3) ^ ((ln >> 2) & 3)) * 16)
	v_bfe_u32 v1, v7, 2, 2
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_xor_b32 v2, v0, v1
	v_lshrrev_b32 v0, 2, v7
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshlrev_b32 v1, 6, v0
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v0, v2, 4, v1
	ds_load_b128 v[4:7], v0                                      // Sw + ((ln >> 2) * 64) + (((ln & 3) ^ ((ln >> 2) & 3)) * 16)
```

The mask is one less than a power of two that divides the dimension; a
pointer moves only by multiples that keep the swizzle.

## 9.4 What is proved about LDS

The render's LDS proof reads the stream's barriers as epochs (the code
between two barriers) and proves, for every pair of accesses:

- **Unions.** Two regions that share bytes are never live in one epoch:
  each arm of a union is used in epochs of its own.
  `b.final_report` lists the epochs each arm is used in.
- **Barriers.** A barrier orders the waves only for LDS ops that have
  landed: an op in flight at a barrier, whose bytes an op after it
  touches, is refused (`b.wait(R)` before the barrier). `b.barrier(fence=True)`
  makes a barrier that also orders the LDS, so every LDS op must have
  landed before it.
- **Races.** Within an epoch, two waves' accesses to the same bytes, a
  write among them, are refused, where the per-wave bytes are known
  (from `tid`, the workgroup size and the wave blocks); `ordered=why` on a
  DS family exempts an access whose order the author guarantees.

What the proof cannot bound, such as addresses from loaded data, the
emulator checks at run time ([11.4](#114-emulating)).

```python
k = b.entry(Args(), wgsize=64)
L = b.lds()
with L.union():
    A = L.region("A", (64,), "u32")
    B = L.region("B", (64,), "u32")
x = V.u32(k.tid * 3, name="x")
b.emit(O.ds_write(A.at(k.tid), x))
b.emit(O.ds_write(B.at(k.tid), x))
y = V.u32(name="y")
b.emit(O.ds_read(y, A.at(k.tid)))
b.wait(y)
```

```error
MemError: A [0, 256) and B [0, 256) share bytes and are both live in the epoch 'entry': A accessed 2 `ds_store_b32 A + tid * 4` .. 6 `ds_load_b32 A + tid * 4`, in it at 2 `ds_store_b32 A + tid * 4`; B accessed 4 `ds_store_b32 B + tid * 4` .. 4 `ds_store_b32 B + tid * 4`, in it at 4 `ds_store_b32 B + tid * 4`
```

## 9.5 Global and buffer addresses

A global family takes `addr, base` (a VGPR offset and an SGPR pointer)
or one formula, `ptr + u32 terms`: the families put a constant term into
the 24-bit signed `offset` field, the scalar pointer into the base and
the rest into a VGPR offset.

```python
src = S.ptr(name="src", at=2)
i, j = V.u32(name="i", at=4), V.u32(name="j", at=5)
x = V.u32(name="x")
b.emit(O.global_load(x, src + mul24(j, 768) + i * 8 + 4))
b.wait(x)
b.emit(O.global_store(src + i * 8 + 4096, x))
```

```asm
	v_mul_u32_u24 v0, 768, v5                                    // adr = mul24(j, 768) + (i * 8)
	s_delay_alu 1                                                // hazard: instid0(VALU_DEP_1)
	v_lshl_add_u32 v1, v4, 3, v0
	global_load_b32 v0, v1, s[2:3] offset:4                      // src + (mul24(j, 768) + (i * 8)) + 4
	s_wait_loadcnt 0
	v_lshlrev_b32 v1, 3, v4                                      // adr = i * 8
	global_store_b32 v1, v0, s[2:3] offset:4096                  // src + (i * 8) + 0x1000
```

A scalar load takes `ptr + scalar terms` (the constant into its offset,
a scalar remainder into `soff`); a buffer family takes a VGPR `voffset`
(or a formula), the 4-dword descriptor `desc` and an SGPR `soffset`. A
formula's offset is a u32: the range check ([7.5](#75-ranges)) proves it
does not wrap.

# 10. Matrix cores and the rest

## 10.1 WMMA

gfx1201's matrix instructions compute `D = A·B + C` for one 16 × 16 tile
in one wave: `v_wmma_<D>_16x16x<K>_<A/B>`, with A 16 × K, B K × 16, C and
D 16 × 16. They are emitted with `O.op`, the operands values of the
fragment sizes below.

| Mnemonic | A, B | K | A, B dwords | C, D |
|---|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` | f16 | 16 | 4 | 8 f32 |
| `v_wmma_f32_16x16x16_bf16` | bf16 | 16 | 4 | 8 f32 |
| `v_wmma_f16_16x16x16_f16` | f16 | 16 | 4 | 4 (8 f16) |
| `v_wmma_bf16_16x16x16_bf16` | bf16 | 16 | 4 | 4 (8 bf16) |
| `v_wmma_f32_16x16x16_fp8_fp8`, `_fp8_bf8`, `_bf8_fp8`, `_bf8_bf8` | e4m3 / e5m2 | 16 | 2 | 8 f32 |
| `v_wmma_i32_16x16x16_iu8` | 8-bit integers | 16 | 2 | 8 i32 |
| `v_wmma_i32_16x16x16_iu4` | 4-bit integers | 16 | 1 | 8 i32 |
| `v_wmma_i32_16x16x32_iu4` | 4-bit integers | 32 | 2 | 8 i32 |

The fragments' layout across the wave's 32 lanes, lane `l = 16h + r`
(`r` = `l % 16`, `h` = `l // 16`):

- A: lane `l` holds row `r` of A, elements `k` = `K/2·h` .. `K/2·h + K/2 - 1`,
  packed from the low bits (two f16 a dword).
- B: lane `l` holds column `r` of B, the same `k`.
- C and D: lane `l` holds column `r` of D, rows `8h` .. `8h + 7`, one
  element a dword (f32, i32) or two (f16, bf16).

```python
a = V.u32(n=4, name="a", at=8)
bm = V.u32(n=4, name="bm", at=12)
c = V.f32(n=8, name="c", at=16)
b.emit(O.op("v_wmma_f32_16x16x16_f16", c, a, bm, c))
b.emit(O.op("v_wmma_f32_16x16x16_f16", c, a, bm, c))
h = V.u32(n=4, name="h", at=24)
b.emit(O.op("v_wmma_f16_16x16x16_f16", h, a, bm, 0))
b.emit(O.op("v_wmma_f16_16x16x16_f16", a, h, bm, a))
```

```asm
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[12:15], v[16:23]
	v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[12:15], v[16:23]
	v_wmma_f16_16x16x16_f16 v[24:27], v[8:11], v[12:15], 0
	v_nop                                                        // hazard
	v_wmma_f16_16x16x16_f16 v[8:11], v[24:27], v[12:15], v[8:11]
```

A WMMA whose result is the next one's C needs nothing between them; one
whose result is read as a matrix operand gets a `v_nop` from the hazard
pass. `neg_lo=`, `neg_hi=` negate A and B elements (floats) or make them
signed (integers), as the ISA says. The sparse `v_swmmac_*` forms are
emitted the same way; the emulator does not model them.

## 10.2 Claims

A claim tells the checks and the emulator a fact they cannot derive.
Each takes its reason, is printed in the `.s` as a comment line, and is
checked by the emulator where it runs.

| Claim | Says |
|---|---|
| `b.uniform(*regs, why=)` | these SGPRs (or `"scc"`) hold the same value in every wave |
| `b.undefined(*regs, why=, edge=None)` | these registers hold nothing a read uses (from here; `edge="taken"`: on the next branch's taken edge) |
| `b.assume(x, lo=0, hi=, why=)` | each dword of `x` is in `lo .. hi`, for the range checks |
| `b.discard(reg, why)` | the register's contents are not read: the write before it is no dead write |

```python
col, val = V.u32(name="col", at=2), V.u32(name="val", at=3)
base, flag = S.u32(name="base", at=4), S.u32(name="flag", at=5)
b.uniform(flag, base, why="the flag is broadcast from one wave")
b.assume(col, lo=16, hi=63, why="upper lanes only")
b.discard(col, "the column index is not reused")
```

```asm
; uniform: s5, s4 (the flag is broadcast from one wave)
; assume: v2 in [16, 63] (upper lanes only)
; discard v2: the column index is not reused
```

## 10.3 Ablation

Ablation builds a variant of a kernel with classes of instructions left
out, to measure what they cost. An instruction gets classes by `abl=` (on
`O.op`, `b.emit`, `b.wait`, `b.barrier`) or by a `with
b.ablatable(*classes):` block; `b.ablate` is the set of classes the
build drops, read at each emission. `b.ablated(abl)` says whether an
instruction of those classes would be dropped. A branch is not
ablatable: the instructions to drop are marked, not the block around
them. An ablated kernel skips the dead-write and alias checks, whose
findings it causes.

```python
val = V.u32(name="val", at=3)
b.ablate = {"tail"}
b.emit(O.op("v_mov_b32", val, 7))
with b.ablatable("tail"):
    b.emit(O.op("v_add_nc_u32", val, 5, val))
b.emit(O.op("v_lshlrev_b32", val, 3, val, abl=("tail",)))
```

```asm
	v_mov_b32 v3, 7
```

## 10.4 The Builder's other members

| Member | Is |
|---|---|
| `b.program`, `b.ins`, `b.values` | the kernel's IR, its instruction stream and its values |
| `b.pos`, `b.last`, `b.emitted(start, stop)` | the stream's length (hazard instructions aside), its last instruction, the instructions between two positions |
| `b.issued(counter)` | the ops of a counter (`"load"`, `"store"`, `"ds"`, `"km"`) issued on the current path |
| `b.overlap(r1, r2)` | whether two registers' numbers overlap |
| `b.hold(label, *regs)`, `b.release(label)` | marks registers held: writing a held register is refused (a library's registers across its call) |
| `b.need("co_resident", why)`, `b.need("same_grid", why, arg)` | what the binding must honour: every workgroup resident at once, or a grid the same in every launch for an argument |
| `b.needs` | the needs recorded |
| `b.fork()`, `b.cnt`, `b.path_state` | the wait state of a path, saved at a branch and restored on its other side (for hand-written branches), and an author's records joined at labels |
| `b.steady_state_check(body_fn, stem)` | the audit's steady-state check for a hand-written loop |
| `b.seal()`, `b.settle()`, `b.sealed` | ends emission; closes an open chain; whether the passes have run |
| `b.options`, `b.pass_times`, `b.final_report`, `b.hazards`, `b.waits_placed`, `b.deferred` | the build options, each pass's seconds, the LDS proof's report, the hazard instructions placed, whether the waits pass ran, the deferred transforms' records |
| `b.layout`, `b.entered`, `b.target`, `b.files`, `b.wave_reg`, `b.nwaves`, `b.by_wave` | the layout, the entry, the target, the files by letter, the wave index's register and count |

`as_reg(x)` is the register a value stands for (`Reg`, thin's `VReg`).
`Counter`, `VM` and `LGKM` are the wait counter names `O.VM` and `O.LGKM`
are. `SPIN` is the comment that marks a bounded poll's round counter:
the emulator reports a poll that times out `TIMEOUTS` times in a row as
a deadlock. `Maker` is `O`'s class.

# 11. Compiling, running, emulating

## 11.1 KernelDef and compile

A kernel is given to the runtime as a `KernelDef(symbol, args, body,
wgsize, max_grid=(), host=None, wg=1, preload=0, meta={}, grid=())`,
returned by a generator function: `body(b, k)` is called with a fresh
Builder and its entry. `thin.compile(generator, **params)` calls the
generator with its build parameters and compiles the kernel into a
`Kernel`, through the store: a kernel compiled once (same generator
source, parameters, options and target) is read back, not compiled
again.

```python
import thin
from thin import KernelDef, compile


def vadd():
    def body(b, k):
        V = b.v
        x, y, z, n = k.load()
        i = V.u32(k.wgid * 256 + k.tid, name="i")
        with b.lanes(i < n):
            off = V.u32(i * 4, name="off")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            b.emit(O.global_load(c, off, y))
            b.wait(a, c, "x, y landed")
            a += c
            b.emit(O.global_store(off, a, z))

    args = Args(x=ptr, y=ptr, z=ptr, n=u32.within(1, 1 << 24))
    return KernelDef("vadd", args, body, wgsize=256, max_grid=(1 << 16,))


kern = compile(vadd)
print(kern.name, kern.manifest.vgpr, kern.manifest.sgpr, kern.manifest.kernarg_bytes)
print(kern.asm.splitlines()[0])
print(vadd().compile().summary.splitlines()[0])
```

```text
vadd 3 10 28
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
vadd: 15 instructions, vgpr 3 agpr 0 sgpr 10, 8 WGs per CU
```

`Kernel.asm` is the whole `.s`: the header, the instructions, the
descriptor, the metadata and a last line `; thin.manifest {...}`, the
manifest (`Kernel.manifest`: registers, arguments, LDS, workgroup size,
the build parameters and the instruction mix). `Kernel.co` is the code
object, `Kernel.args` the arguments' records, `Kernel.address` the
render's identity. `KernelDef.compile()` compiles without the store and
returns an `Artifact` (`text`, `manifest`, `summary`); `KernelDef.render(path)`
writes the `.s`. `thin.declare(generator, **params)` is the `KernelDef`
alone, and `with thin.overriding(params):` sets build parameters under
every compile in the block.

Below `KernelDef` are the pipeline's functions: `derive(build)` places a
Builder made by `build()` (running it twice under the audit), and
`place(b, build)` the same for a made one; `compile_kernel` runs every
pass and renders the `.s` and its manifest, and `emit_kernel` writes it:

    compile_kernel(b, symbol, *, wgsize, args=None, lds=0, build=None, grid=(), max_grid=())
    emit_kernel(out, symbol, b, **kwargs)
 `thin.codegen.using(**options)` sets build
options for the kernels made in its block: `audit`, `keep_dead`,
`safe_barrier` (wait for every counter before each barrier),
`safe_waits` (every count 0), `waits_unique`, `regions`, `pressure`,
`peak_report`, `peak_list`, `lds_wave_order`.

## 11.2 Binding and running

`kern.bind(grid=None, args=None, *, co_resident=False, overread=None)`
makes a `Bound`: one call, its arguments checked against their kinds and
ranges. A pointer argument is a numpy array or a torch tensor, a number
an int or a float, own memory its size. `grid` is the number of
workgroups (an int, or a tuple for `wg=2`). `thin.cdiv(a, b)` is `a / b`
rounded up.

| Method | Does |
|---|---|
| `bound.emulate(sched=None)` | runs the call on the emulator; returns the pointer arguments by name, written back |
| `bound.run()` | launches it on the GPU, on the current torch stream |
| `bound.time(iters=20)` | runs it once, then the mean microseconds of `iters` runs between two GPU events |
| `kern.run(grid, args)` | `kern.bind(grid, args).run()` |
| `kern(**given)` | the kernel's own binding, `KernelDef(host=)`: `host(bind, **given)` |

```python
import numpy as np
import thin
from thin import KernelDef, cdiv, compile


def vadd():
    def body(b, k):
        V = b.v
        x, y, z, n = k.load()
        i = V.u32(k.wgid * 64 + k.tid, name="i")
        with b.lanes(i < n):
            off = V.u32(i * 4, name="off")
            a, c = V.f32(name="a"), V.f32(name="c")
            b.emit(O.global_load(a, off, x))
            b.emit(O.global_load(c, off, y))
            b.wait(a, c)
            a += c
            b.emit(O.global_store(off, a, z))

    return KernelDef("vadd", Args(x=ptr, y=ptr, z=ptr, n=u32), body, wgsize=64)


n = 100
x = np.arange(n, dtype=np.float32)
y = np.full(n, 0.5, np.float32)
z = np.zeros(n, np.float32)
out = compile(vadd).bind(grid=cdiv(n, 64), args={"x": x, "y": y, "z": z, "n": n}).emulate()
print(out["z"][:4], out["z"] is z, np.array_equal(z, x + y))
compile(vadd).bind(grid=1, args={"x": x, "y": y, "z": z, "n": -1})
```

```error
[0.5 1.5 2.5 3.5] True True
LaunchError: vadd: argument n = -1: outside its range 0 .. 4294967295
```

## 11.3 The store

`thin.use_store(root)` sets the directory compile reads and writes (None:
no store) and returns the previous one; by default it is `THIN_STORE` or
`~/.cache/thin-store` (`THIN_STORE=off`: none). An entry is addressed by
the generator's source file and captured values, its parameters, the
output options, the target and the installed thin; it holds the `.s` and
the code object, and is checked on reading.

## 11.4 Emulating

The emulator runs the code object's instructions, decoded, on the CPU:
every wave of every workgroup, with registers, EXEC, the counters, LDS
and global memory. It computes what the GPU computes, bit for bit, for
the instructions it models (WMMA included: its summation and rounding
follow measurements of gfx1201), and is a functional model: it counts
instructions and clock reads, but does not predict time.

`Bound.emulate` raises `DeviceFault` (its `cause` says which) where the
kernel's behaviour would depend on something the hardware does not fix:

- a register read or written while a load into it is in flight (a
  missing wait), and a read of a register no instruction wrote;
- a load or store outside its buffer, and a write to `kept` memory not
  undone by the launch's end;
- an LDS read of bytes another wave wrote since the last barrier, and an
  LDS write over bytes another wave's read or write still in flight
  touches: races, at the instruction, with both waves named;
- a claim ([10.2](#102-claims)) that does not hold;
- an access to another rank's memory without the scope its policy
  requires;
- no wave able to make progress (a deadlock; a bounded poll, its round
  counter commented `SPIN`, that times out `TIMEOUTS` times in a row).

It raises `NotEmulated` for an instruction or field it does not model,
which is never the kernel's fault. Reads of global memory another
workgroup was still writing, with nothing ordering the two, are reported
as a `StaleReads` warning. `sched=` sets how the emulator interleaves
workgroups and waves (`order`, `skew`, `speed`, `jitter`, `wave_jitter`,
`quantum`, `resident`, `wg_speed`, `lockstep`): a race shows under some
schedules only, and one call runs one schedule, so a clean run is not a
proof that a kernel has none. gfx1201 has no LDS bank model in thin: the
emulator reports no bank conflicts for it. `thin.emu` is the emulator's
own API (programs, memories, clusters of ranks); its design is in
[dev/emulator.md](dev/emulator.md).

## 11.5 Limits

- gfx1201 is the only target. thin has no LDS DMA, no AGPR authoring
  and no MX-scaled instructions; MX formats exist on the host, in
  `thin.formats`.
- Instruction support is explicit: the encoder refuses a form it cannot
  encode as written (some literal and inline constants of 16- and 64-bit
  operands, some VOP3 clamp combinations), and never rewrites it into
  another.
- The waits pass is conservative: a wait may stay where a scalar read of
  a value it cannot track would have let it go.
- A loop written with labels and branches, not `b.loop`, has no loop
  record: the audit and the steady-state checks do not see it
  (`b.steady_state_check` gives it one).
- The emulator models the operations thin emits and those of shipped
  kernels; numerical paths new to a kernel should be checked on the GPU.
  Some random bf8 WMMA outputs differ from the hardware's.
- A store entry whose header names another address is treated as corrupt
  and may be evicted.
- A lap clock read whose low dword is zero is not recorded (one point in
  2^32); a lap taken twice in a round keeps its last reading.
