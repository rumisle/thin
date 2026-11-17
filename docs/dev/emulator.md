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
transcendental sweeps. The `gfx1201` WMMA dot model incorporates measured summation and rounding
behavior (some bf8 random cases still disagree; this is a documented model
limit, not a license to accept a wrong output). On `gfx942` (measured on an
Instinct MI300X), integer and `f64` DGEMM forms, sequential single-rounded
`f32` SGEMM forms (including NaN payload propagation), `xf32` bitwise mantissa
truncation (`raw & 0xFFFF_E000`) and the 128-bit grouped XDLOP dot model (`f16`,
`bf16`, `xf32`, `fp8`/`bf8`, dense and sparse `v_smfmac_*`, with canonical
`0xFFC0_0000` NaN output) match hardware bit for bit across the hardware suite
(extreme wide-exponent `fp8`/`bf8` randoms remain a documented model limit, as
on `gfx1201`).

On `gfx950` (CDNA 4 / Instinct MI350X & MI355X, verified offline against
`amdgpu_isa_cdna4.xml`, LLVM 22, `ml_dtypes`, and the CDNA 4 ISA manual,
awaiting silicon confirmation via `pytest -m gfx950`), all 68 `MatrixForm`
entries carry `provisional = True`:

- **OCP FP8/BF8 & sub-byte formats**: `fnuz_fp8 = False` selects OCP `e4m3fn`
  (`±448.0`, `0x7F`/`0xFF` NaN) and `e5m2` (`±57344.0`, `0x7C`/`0xFC` `±Inf`) on
  both shared `v_mfma_*_fp8_*` / `v_cvt_*fp8*` ops and new CDNA 4 ops, alongside
  6-bit `e2m3`/`e3m2`, 4-bit `e2m1`, and 8-bit `e8m0` (`0x00 -> 2^-127`,
  `0xFF -> NaN`).
- **Derived CDNA 4 matrix layouts & MX scaling**: AMD's `matrix_calculator.py`
  has no `cdna4` data, so all 24 new CDNA 4 matrix shapes (doubled-K dense/sparse,
  `v_mfma_f32_*_f8f6f4`, and `v_mfma_scale_f32_*_f8f6f4`) are derived from ISA
  manual §7.1.5 and §7.5.1.3. Narrow 8-VGPR operands (`fp8`/`bf8` in `f8f6f4` and
  doubled-K sparse `B`) stack two 4-VGPR K-halves (`v0..v3`, `v4..v7`), whereas
  6-VGPR FP6/BF6 (`192` bits = 32 × 6-bit packed across dword boundaries) and
  4-VGPR FP4 (`128` bits = 32 × 4-bit) pack one contiguous chunk per lane.
  `v_mfma_scale_*` multiplies each 32-element K block by `2^(ea - 127)` and
  `2^(eb - 127)` before 32-element grouped summation (`0xFF` propagates
  `0xFFC0_0000` only to affected rows/columns). The manual also allows inline
  constants as scale operands, but LLVM 22 accepts only VGPRs, so thin does too.
- **Scaled/SR conversions & ISA manual discrepancies**: `v_cvt_scalef32_*`
  models default `FP16_OVFL = 0` (`e4m3fn` overflow -> `NaN` `0x7F`/`0xFF`;
  `e5m2` overflow -> `±Inf` `0x7C`/`0xFC`; `fp6`/`bf6`/`fp4` overflow -> `±max`).
  The saturating `FP16_OVFL = 1` mode is not modeled.
  Stochastic rounding (`v_cvt_sr_*`, `v_cvt_scalef32_sr_*`) implements unbiased
  add-and-truncate of `23 - mb` top random bits per CDNA 4 manual §7.3 (whereas
  VOP3 `V_CVT_SR_FP8_F32` pseudocode writes `23'U(...)` wrap + RNE; probed in
  `tests/hw/test_gfx950_hw.py`). `v_exp_legacy_f32` and `v_log_legacy_f32`
  (present in `amdgpu_isa_cdna4.xml`, LLVM `isGFX7GFX8GFX9`, and manual Table 12,
  though omitted in Chapter 12's body) follow standard GFX9 legacy semantics
  (`log.legacy(±0.0 / denorm) = -FLT_MAX`).
- **Transpose LDS reads, wide DMA & 64-bank LDS**: `ds_read_b64_tr_b{4,8,16}` and
  `ds_read_b96_tr_b6` transpose 16-lane rows into MFMA input order;
  `global_load_lds_dwordx3` and `buffer_load_dwordx3 … lds` step 16 bytes/lane in
  LDS and leave the 4th dword untouched (manual §9.1.9); and LDS models the
  160 KB (`163,840` B), 64-bank geometry.

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
Dense `MatrixForm` shapes on `gfx942` and fixed-element unscaled dense shapes on
`gfx950` (MFMA over AGPR/VGPR accumulators, including inputs staged via
VMEM-to-LDS DMA) wire directly into the checker; forms without a wired GEMM
shape model (such as `gfx1201` WMMA, sparse `v_smfmac_*`, or dynamic/MX-scaled
`*_f8f6f4` forms) raise an explicit `EmuLimit` instead of claiming success.
Abstract matrix-form tests also exercise the checker independently of execution.

MX scale provenance tags, scale tables, nibble provenance layouts and pending
scales are not implemented in `thin.emu.gemm` (`thin-emu`'s machine layer
executes `v_mfma_scale_*` and sub-byte `f8f6f4` numerically, while the
experimental GEMM provenance checker refuses them with `EmuLimit`). Unsupported
machine operations and checker events must produce an explicit limit. Use
ordinary emulation output comparisons and hardware checks alongside this
checker; never treat its silence as proof of correctness.

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
