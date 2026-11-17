# Architecture

thin compiles instruction-level Python kernels to AMD GPU code objects and
runs the same programs in a CPU emulator. `gfx1201` (RDNA 4), `gfx942`
(CDNA 3 / Instinct MI300X), and `gfx950` (CDNA 4 / Instinct MI350X & MI355X)
are the supported targets.
`metron`, kept in this tree, is a separate timing and correctness harness;
it is not part of thin's PyPI distribution.

## Boundaries

The Rust core owns instruction semantics, analyses, lowering, encoding and
byte formats. Python owns authoring syntax, source locations, array ownership,
HIP resources, launching and reporting. The PyO3 extension converts values
between them; it does not implement compiler decisions. `tools.check_binding`
checks its dependencies and rejects domain branches and loops in the binding.

| crate | responsibility |
|---|---|
| thin-formats | element encodings (OCP & FNUZ FP8/BF8, FP16/BF16/FP4/FP6, E8M0), host MX quantization, layouts, MessagePack, code objects |
| thin-target | target vocabulary, `MatrixForm` shapes (dense, sparse, mixed `F8F6F4`, MX-scaled), form rules, shared assembly reader and assembler driver |
| thin-ir | immutable instructions, snapshots, CFG, dataflow (`v`/`s`/`a` files, `m0`, VMEM-to-LDS DMA), diagnostics and printing |
| thin-passes | pass-count-aware hazards (`s_delay_alu` / `s_nop`), waits (`s_wait_*` / `s_waitcnt`), register allocation, ranges, alias and LDS checks |
| thin-targets | concrete targets (`gfx1201` and parameterized `cdna` family: `gfx942`, `gfx950`), instruction tables, encoder/decoder, hardware hooks |
| thin-dsl | Builder state, typed values, matrix/MFMA/MX-scale/SMFMAC helpers, transpose LDS reads, VMEM-to-LDS DMA, expression lowering, control flow and arguments |
| thin-codegen | compilation pipeline, placement, ABI, manifest and code-object construction |
| thin-lift | physical assembly/code-object lifting, register-value recovery and execution laws |
| thin-emu | decoded machine execution, memory, provenance and experimental GEMM checks |
| thin-py | conversion-only Python extension |

Generic crates do not name GPU instances. The registry supplies the default
target and pairs each target with its hazard model. Wave size, register
resources, memory counters, scope policy, barriers, clocks and bank geometry
come from target data or hooks. Adding a target is described in
[adding-a-target.md](adding-a-target.md).

Production dependencies flow toward shared data types, never toward Python.
Dev-dependencies may use the target registry to exercise generic algorithms
against a real instance. `tests/rust` is an unpublished library of shared test
builders, notation readers and exact golden assertions, not a runtime layer.

## Compilation

1. Python calls the kernel author's body with a Builder and loaded arguments.
   Operators and context managers delegate construction and lowering to Rust.
2. The Builder emits instructions and claims without reordering, assigning
   physical registers, counting waits or inserting hazard instructions.
3. Compilation takes immutable snapshots. Analyses operate on snapshots and
   explicit options; a transformed instruction is a new value, so an earlier
   snapshot never observes a later edit.
4. The pipeline applies deferred transforms, solves placement and register
   assignment, inserts waits and hardware hazard instructions, and audits the
   result before printing assembly and its resource manifest.
5. The target assembler writes code and byte ranges. Code-object construction
   adds the descriptor, ABI metadata and ELF sections.

A host object attached to a row is held by opaque identity; the core captures
only the facts it needs and never calls back into Python. Diagnostics carry
structured text, instruction and source parts; Python renders source locations.
Register pins denote hardware values at entry. Overlapping pins are legal only
when their liveness permits sharing the physical register.

The target's assembly reader is shared by compilation, lifting and emulation.
The independent wait checker is the exception: it reads assembly on its own
and deliberately shares neither compiler logic nor parsing tables.
This gives scheduling tests a check that cannot simply repeat the same bug.

## Encoding and lifting

Each target's encoder and decoder (`gfx1201`, `gfx942`, and `gfx950`, including
128-bit two-qword `ENC_VOP3PX2` instructions on `gfx950`) use checked-in
instruction tables derived from AMD's machine-readable ISA XML. Successfully
decoded bytes must encode back identically; forms that cannot preserve the
input are refused rather than guessed. ELF descriptor and metadata checks
are separate from instruction decoding.

Lifting constructs physical control flow and recovers register-value structure.
PC-relative call resolution retains whole register pairs and arithmetic
correlations, with bounded sets and explicit refusals. On incomplete control
flow, dead-write results are conditional, not whole-program removal proofs.
See [limitations.md](limitations.md) for the bounds and remaining literal cases.

## Emulation

The machine executes decoded canonical operations, not mnemonic strings.
Three layers keep execution independent from any particular checker:

- `tags`: provenance representation, interned records, transfer rules and
  the checker interface; it uses no machine or GEMM modules.
- `machine`: waves, register files, global memory, LDS, waits, scope and
  instruction semantics; it uses tags, never GEMM internals.
- `gemm`: data-driven matrix provenance checks, implementing the checker
  interface using tags only.

Python checkers cannot override native execution. Problems and layouts are
plain data. Tag tables are caller-owned Rust values; the Python binding keeps
process-scoped tables with a scoped fresh-table context for tests.

The emulator is a functional model with selected counters, not a cycle-level
GPU simulator. Its numerical routines, wave32 WMMA model (`gfx1201`, fitted to
RX 9070 XT measurements), wave64 CDNA 3 MFMA / sparse SMFMAC model (`gfx942`,
fitted to Instinct MI300X measurements across AGPR/VGPR accumulators, FNUZ
FP8/BF8, VMEM-to-LDS DMA, GFX9 DPP controls, packed `v_pk_*_f32` ops and
32-bank LDS conflict tracking), and wave64 CDNA 4 model (`gfx950`, verified
offline against AMD ISA XML, LLVM 22, `ml_dtypes`, and the CDNA 4 ISA manual,
with `MatrixForm.provisional = True` pending MI350X/MI355X silicon confirmation)
are tested against ISA specifications, AMD Matrix Calculator (where supported),
LLVM MC/codegen and GPU hardware. A successful run does not prove a kernel
race-free on all hardware schedules. Read [emulator.md](emulator.md) before
treating a diagnostic as a proof.

## Runtime and tooling

Compilation identity includes code, options and captured values. Bound calls
own their arguments and retain allocations while launches or captures need
them. HIP errors are reported at the boundary where they occur. Store entries
are content-addressed and their headers are validated before reuse.

`thin.opspec` declares operations, implementations and checks. Selection uses
the first covering pair in declaration order; a broken implementation contract
raises immediately. `metron` compares candidates with randomized timing rounds
and explicit restore/check phases. On stacks where graph replay is unsafe it
uses host launches instead.

## Tests

- Rust unit and integration tests live with their owning crates. Golden input
  and expected output data is under each crate's `tests/data`.
- Python API, runtime and host protocol tests live in `tests`; Python-only
  golden data is in `tests/data`. Shared core data is read in place.
- Hardware tests (`pytest -m gfx1201` on RDNA 4; `pytest -m gfx942` on CDNA 3 /
  MI300X; `pytest -m gfx950` on CDNA 4 / MI350X & MI355X) compare emitted
  programs, numerical edge cases, WMMA/MFMA/MX-scale/SMFMAC output, FNUZ and OCP
  small-float conversions, transpose LDS reads, VMEM-to-LDS DMA, DPP/VOP3P
  semantics, memory visibility and timing behavior against `thin.emu`.
- Architecture tests enforce target isolation, the emulator's layering and
  the binding boundary. The independent native wait checker validates waits;
  Python delay-hint tests recompute VALU dependencies from assembly. There
  is no separate native hazard checker.
- `tests/budgets.toml` bounds test time and memory.

Every golden case must pass; there is no list of expected failures.

### Golden-data generation

`tools/golden` constructs inputs in code and evaluates thin's APIs to produce
test data. Programs, constructor calls, grammar cases and seeded input plans
live in the generators. Python APIs and stdin-driven native examples supply
artifacts, analysis results and diagnostics. Generators cannot read stored test
data; only the comparison layer reads it after construction.

Run `uv run python -m tools.golden COMPONENT` to write a component's outputs.
Add `--check --report diff.json` to generate a temporary tree and report every
difference without changing stored outputs. `all --authored --check` selects
source/program-bearing files and preserves their numeric-only recorded
sections. Numeric-only files outside that scope remain recorded.

`--coverage` lists every data file without a generator and fails while any
remain; `all --check` fails on the same files. `--authored-coverage` checks the
program inventory separately without exempting recorded files from full
coverage. See `tools/golden/README.md` for component details.
