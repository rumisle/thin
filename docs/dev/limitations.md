# Limits and intentional behavior

The limits a kernel writer meets are in the language reference,
[lang.md 11.5](../lang.md#115-limits); these are the internals'.

## Target and encoding

- The physical decoder refuses literal words in multi-dword operands.

## Analysis and compilation

- Empty IR attributes are absent: putting an empty value removes the attribute.
  Snapshots keep the instruction/operand data they captured, even if the host
  later calls `set_ops` on its stream.
- Register pins mean hardware values at entry, not independent copies. Pins
  on one register share in-flight state; overlapping live assignments are
  refused.
- PC-relative call resolution bounds scalar sets at 64 and address/link sets
  at 4096. Lost correlations, unsupported dependencies, non-numbered pairs
  and over-limit sets produce typed refusals. Branches are may-edges, not a
  proof of runtime path feasibility. Return edges narrow the link pair to
  the named caller; other registers are not a general calling convention.
- Dead-write findings on incomplete CFGs are conditional. A dead register
  destination does not prove all other effects of its instruction removable.

## Emulation and checking

- Numerical behavior follows measured RDNA 4 results, including canonical
  negative quiet NaNs for fp8/bf8 conversion.
- Hand-written decoded JSON must supply byte layout for PC execution. Text
  uses a virtual zero code base. Padding is not a decoded instruction and
  does not count toward instruction statistics.
- Ambiguous wide scalar literals in decoded JSON do not yet preserve signed
  inline versus unsigned literal provenance. PC execution tests cover register
  pairs, not arbitrary literal address arithmetic.
- The GEMM provenance checker is **experimental, test-only** and does not
  certify arbitrary kernels. See [emulator.md](emulator.md) for its scope.

## Runtime, operations and timing

- Operation selection is the first covering pair in declaration order. A
  misdeclared `serves` implementation raises in any lowering. A remote IPC
  rank's GPU is not a local device handle and cannot be queried as one.
- metron's graph bound counts calls, not launches inside an opaque call.
  Sizing uses the baseline arm (arm 0). Hygiene samples four slices, not
  every byte; cache invalidation reaches the dispatcher's domain, not an
  arbitrary device or remote process. Version identity covers metron's own modules and kept
  kernels, not arbitrary adapter or maker code.
- ROCm graph replay can fail cumulatively or crash at teardown on affected
  stacks. Those stacks use host launches. A hardware re-test is opt-in; do
  not infer graph safety from a small successful race.
- Communication polls have no timeout; the group launch contract must
  guarantee progress.
