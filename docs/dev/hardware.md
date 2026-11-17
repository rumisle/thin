# Hardware validation

Hardware suites run per target:

- **RDNA 4 (`gfx1201`)**: validated on an RX 9070 XT (32 WGPs / 64 CUs, 16 GB)
  with ROCm 7.1 (`pytest -m gfx1201`).
- **CDNA 3 (`gfx942`)**: validated on an Instinct MI300X VF (304 CUs across
  8 XCDs in SPX partition, 192 GB HBM3) with ROCm 10.1 (`pytest -m gfx942`, see
  `tests/hw/test_gfx942_hw.py`).
- **CDNA 4 (`gfx950`)**: **awaiting Instinct MI350X / MI355X silicon**.
  Encoding/decoding (including 128-bit `ENC_VOP3PX2` and `EF_AMDGPU_MACH = 0x4F`
  `e_flags = 0x54F`), instruction tables (`amdgpu_isa_cdna4.xml`, 1,242
  instructions), OCP MX formats (`ml_dtypes`), and `Cdna4Model` hazards (`llc
  -mcpu=gfx950`) are verified offline; derived CDNA 4 MFMA/F8F6F4/MX-scale/SMFMAC
  layouts and numerics (`provisional = True`), stochastic rounding edge cases,
  `ds_read_*_tr_*` lane mappings, `dwordx3` LDS DMA skipping dword 3, and 160 KB /
  64-bank LDS behavior have a pre-built, CPU-preflighted probe suite ready in
  `tests/hw/test_gfx950_hw.py` (`pytest -m gfx950`).

Tests explicitly select the matching GPU, enter `targeting(<arch>)`, and skip
cleanly when that GPU is absent.

```sh
uv sync
uv run pytest -m gfx1201   # on RDNA 4
uv run pytest -m gfx942    # on MI300X / CDNA 3
uv run pytest -m gfx950    # on MI350X or MI355X / CDNA 4
```

Rebuild the extension after pulling Rust changes. Tests are held to
`tests/budgets.toml`. Isolated tests run in child processes.
`THIN_HW_REPORT_DIR` retains hazard and co-residency measurement reports;
`THIN_HAZARD_RUNS` controls repetitions of hazard negative controls.

## What is checked

- Code-object loading and symbol lookup, ABI arguments, launch grids and
  vadd outputs including subnormals, signed zero, infinities and NaNs.
- Wait and hazard chains, SGPR round trips, independent wait checks and
  Python delay-hint checks.
  Removing the nop between dependent WMMAs produces wrong bytes on the
  validation GPU; not every scheduling omission is exposed by every run.
- Uncached allocation, visibility between streams, retained allocations,
  capture restrictions and output bounds.
- Cooperative co-residency bounds across workgroup sizes 32 to 1024. HIP's
  reported multiprocessor count is used as the 4-SIMD WGP unit; one past the
  computed bound is refused before launch.
- metron A/A races, planted correctness faults, cold/warm inputs, cooperative
  arms and owned memory. Restore/check operations must use the rank's stream.
- WMMA layout and numerical models, cancellation and C-entry pairs, measured
  vectors and random matrices across supported float forms.
- ALU, conversion, partial-register, LDS/global-memory and atomic
  operations of shipped kernels ([lift.md](lift.md)) with randomized and
  adversarial inputs. NaN payloads, midpoint rounding, cancellation and
  denormals are compared bit for bit.
- Laps readback and decoding with the target's clock hooks.

Measured WMMA cases and randoms are in
`crates/thin-emu/tests/data/emu`. `python -m tests.hw.wmma_probe out.json`
and `python -m tests.hw.wmma_fit out.json` run the standalone probes and fits;
`--randoms out.json.gz` records random matrix outputs. These measurements
bound the model's claims; they do not establish universal numerical agreement.

## Stack caveats

On ROCm 7.1 on the RX 9070 XT, graph replay can fail after cumulative launches
and torch graph teardown can segfault. metron uses host launches on affected
stacks. The opt-in graph re-test (`METRON_GRAPH_RETEST=1`) runs each case in a
fresh process. Successful small captures alone do not establish stack safety.

Timing and memory budgets separate torch's import/allocation cost from the
harness's own cost. Record software versions with new measurements; timings
are observations of this machine, not target-independent promises.
