# Emulator design

`Bound.emulate()` runs a compiled program on the CPU. The same decoded program
can be run through `thin.emu` with explicit memory, execution and checker data.
The purpose is to catch functional mistakes before launching a GPU kernel,
not to predict its elapsed time.

## Execution and memory

The target reader decodes assembly into canonical operations and operands.
A machine holds waves with scalar/vector registers, EXEC and VCC, outstanding
loads and waits. Only active EXEC lanes commit vector results. Reads of
undefined values use poison so an accidental dependency can be diagnosed.

Global buffers retain byte identity, allocation bounds and initialization.
Loads can remain in flight until the appropriate counter wait; premature
register access is a fault. LDS is workgroup-local and tracks conflicting
accesses and barriers. Scope checks require policy bits when IPC memory is
used. Rank clusters model cooperating kernels and their progress; lack of
progress is reported as deadlock rather than silently completing.

Instruction counts, bank counts and supported clock observations are model
outputs, not hardware latency estimates. Scheduling, caches and arbitrary
cross-workgroup interleavings are not simulated exhaustively.

## Decoded JSON

`Program.of` accepts a JSON-shaped dictionary. `ins` is an ordered list of
instructions, each with canonical `op`, `kind`, `operands` and `fields`.
Register operands use `{"reg": [file, first, dwords]}`; literal operands use
`{"lit": [32-bit pattern, display text]}`; labels use `{"label": name}`.
Optional `neg` and `abs` flags describe source modifiers. Program metadata
includes label indices, section ranges, user SGPR count and optional PC byte
layout. Encoding methods return these same plain dictionaries.

Lane-mask registers may be supplied as two dwords; the native reader narrows
them to one dword on a wave32 target. This changes register-mask width, not
the target's execution width. Byte layout is required for PC instructions.

## Numbers

Bit-level conversions and ALU routines explicitly handle NaNs, signed zero,
subnormals, fused rounding and destination halves. A host floating-point
operation is used only where its behavior matches the implemented instruction.
Hardware tests include payload-heavy NaNs, cancellation, midpoints and dense
transcendental sweeps. The WMMA dot model incorporates measured summation and
rounding behavior. Some bf8 random cases still disagree; this is a documented
model limit, not a license to accept a wrong output.

## Provenance

A byte can be untagged, identify a byte of a global allocation, or refer to an
interned accumulator/poison/sum record. Integer byte operations preserve tags
only when they preserve identifiable bytes. Arithmetic normally drops plain
memory provenance. Matrix and supported float rules construct or combine
records describing rows, columns, problem identities and covered K blocks.

The checker interface receives memory and matrix events. The machine itself
knows nothing about matrix layouts or a particular GEMM problem. Caller-owned
tables intern equal records, so comparisons use stable tag identities during
a run. Python's fresh-table context restores the previous tables on exit.

## Experimental GEMM checker — test use only

`thin.emu.gemm` is experimental test tooling, not a supported correctness
certificate. Its API may change. It checks tagged data flow against declared
row-major layouts, dimensions and optional scatter/record mappings. It can
report wrong coordinates, missing or repeated partial sums and invalid stores.

It does not establish numerical accuracy, arbitrary epilogue equivalence,
complete K coverage across every control-flow path or correctness under every
concurrent schedule. A coordinate/provenance match is not an algebraic proof.
The current target has no wired GEMM matrix form for WMMA: asking the checker
to validate that event gives an explicit limit instead of claiming success.
Abstract matrix-form tests exercise the checker independently of execution.

MX scale factors, scale tables, nibble layouts and pending scales are not
implemented. Unsupported machine operations and checker events must produce
an explicit limit. Use ordinary emulation output comparisons and hardware
checks alongside this checker; never treat its silence as proof of correctness.

## Error classes

What a run reports to a kernel writer, and as which error, is in the
language reference, [lang.md 11.4](../lang.md#114-emulating). In
`thin.emu`: `EmuError` is an invalid program or a violated execution
claim, `EmuLimit` behavior the model cannot execute or check, `Deadlock`
no modeled wave making progress, `IpcPolicy` an access lacking the
required policy, `RegisterInFlight` an access before an asynchronous
load has retired.

Assembly-reading errors remain value errors rather than runtime kernel faults.
PC instructions require assembler byte layout; literal-extension words are not
instruction boundaries. Alignment gaps advance to the next decoded row, and
implicit padding nops do not contribute to decoded-instruction statistics.
