# Changelog

Notable changes to thin, newest first. Versions follow semantic
versioning; before 1.0 a breaking release raises the minor version.

## Unreleased

- Emulator, gfx1201: every dense float WMMA uses the dot unit model.
  Fixes `v_wmma_f16_16x16x16_f16` after a step overflows (∞ took the wrong
  sign, ∞ − ∞ was missed) and below 2^-40, and the fp8 / bf8 forms on
  denormal inputs, products summed in quads, NaNs decided per step, C's
  `neg_lo` / `neg_hi` (ignored there) and −0 sums (+0).
- Emulator, gfx1201: integer WMMA with `clamp`, saturating after each half
  of k.

## 0.1.0

The first release.

- Targets: gfx1201 (RDNA 4), gfx942 (CDNA 3, Instinct MI300X) and gfx950
  (CDNA 4, Instinct MI350X and MI355X; verified offline, not yet on
  hardware).
- The instruction-level DSL, its compiler and code objects, and the
  emulator with its GEMM checker.
- Licensed under Apache-2.0.
