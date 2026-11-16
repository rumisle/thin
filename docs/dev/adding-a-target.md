# Adding a target

A target is one crate module under `crates/thin-targets/src/<name>/`
plus its machine-readable tables under `crates/thin-targets/data/<name>/`.
No other crate may name it; `tests/test_architecture.py` checks that.
The definition and hooks below supply the hardware facts used by core code.

## 1. The definition (`thin_target::Definition`)

`build()` returns the definition, and `thin_target::Target::build` checks it.

| Item | What it is |
|---|---|
| `name` | The target name (`gfx1201`, `gfx942`), as code objects and the registry spell it. |
| `lds_max` | Maximum LDS bytes available to one workgroup. |
| `fnuz_fp8` | Whether 8-bit float ops use CDNA 3 FNUZ encodings (`e4m3fnuz` / `e5m2fnuz`, `0x80` NaN, unsigned `0x00` zero) instead of OCP FP8/BF8. |
| `counter_names` | Counter names, in the same order as `counters`. |
| `wave` | Lanes per wave, 32 or 64: the width of lane masks, of EXEC and VCC, of the emulator's waves, and of the DSL's masks. |
| `files` | The register files (`v`, `s`, `a`), each with its size, allocation granule and the alignment of a 64-bit or wider range. |
| `forms` | One form per (op, arity, returning), built from the operand table by `FormRules` (below): its slots, fields, flags (including `LDS_MEM` for direct-to-LDS DMA), kind, counter, `dst_sel`, memory class and memory bytes. |
| `encodings` | The mnemonics its encoder has an opcode for. |
| `counters` | The wait counters: name, largest count, wait field, the units it counts (VMEM, DS, SMEM) and when its results land (at issue, or at the wait). The emulator maps them onto its four counters (vector loads, vector stores, LDS, scalar memory). One counter may cover several of them (`gfx942`: `vmcnt` covers loads + stores + VMEM-to-LDS DMA; `lgkmcnt` covers DS + SMEM). |
| `waits` | The wait instructions: mnemonic, count fields, and whether the line spells them bare or by name. |
| `hazards` | The hazard model: the number of wait-state rules, whether the target has its own pass, the rules' window, and the issue hint's mnemonic (`gfx1201`: `s_delay_alu`; `gfx942`: `s_nop` with pass-count-aware MFMA/SMFMAC/DGEMM wait states). |
| `operands` | The maker's operand rules: whether VOP3 takes a literal, the constant-bus limit, the SMEM offset width, and whether a VALU SGPR write needs an SMEM wait. |
| `chip` | Sizes and limits: XCDs (one L2 each), compute units, SIMDs, LDS bytes, VGPRs a wave addresses and per lane, SGPRs, user SGPRs, preload dwords, wave slots, and the named registers holding the workgroup ids (empty: the ids follow the user SGPRs). |
| `spellings` | The `(dotted op, mnemonic)` pairs where the default spelling does not apply. |
| `barrier` | The workgroup barrier sequence. The emulator waits at its last instruction and runs the others as no-ops. |
| `unit_prefixes` | The matrix and scalar-memory mnemonic prefixes. |
| `hooks` | Its `Hooks`. |

`FormRules` takes the target's matrix, scalar-memory, transcendental and
wait prefixes, its barrier, its carry-in and wave-varying ops, and four
functions: the counter of a form, its fields, its sign-modifier operands
and its operand widths.

## 2. The hooks (`thin_target::Hooks` and `Syntax`)

| Hook | What it supplies |
|---|---|
| `Syntax` | Register counts, named scalar registers, `hwreg` ids, bare modifiers, named wait counts, modifier symbols. |
| `mem_of`, `returning` | A mnemonic's memory class; whether an atomic is its returning form. |
| `encode` | One `.s` line to bytes, or a branch with its label. |
| `nop_word`, `e_flags`, `granule`, `allocated_vgprs`, `directives`, `footer`, `resources`, `interface_resources` | The code object: the padding word, the ELF flags, the VGPR/AGPR allocation block (including unified VGPR+AGPR pool rules on CDNA), the `.amdhsa_*` names and the kernel descriptor. |
| `matrix_form` | Per matrix mnemonic: `M`, `N`, `K`, `blocks`, `sparse`, operand types, accumulator register classes (`v` / `a`), and the summation model. |
| `scope_rule` | How memory instructions name their scope and temporal policy: the fields of the scope code, the scope of each code for plain accesses and atomics, when write-backs and invalidates reach the L2, the non-temporal hints, the IPC policy and its text, and the field that makes an atomic return. |
| `lds_banks` | The LDS bank rule per access width, for the bank-conflict report (`gfx1201`: 64 banks; `gfx942`: 32 banks with asymmetric `b128` read/write phase counts; `None`: no report). |
| `lib_hooks` | The library's hooks: the lane granule, the core and constant clock reads and the constant clock's MHz, the coherent and system scopes, the returning atomic add and its hint, the scalar add, the lane-mask shift, the sleep. |
| `icache_invalidate` | The instruction-cache invalidate and its no-ops, and the workgroups per unit that reach every unit. |

## 3. Beside the definition

- **Tables** (`data/<name>/`): `isa_tables.json` (encodings and opcodes,
  from the vendor's machine-readable ISA) and `operand_table.json`.
- **Decoder**: `thin_targets::decoder(name)`, a `Decode` that turns bytes
  into instructions and prints lines the encoder takes back.
- **Hazard model**: `thin_targets::hazards(name)`, a `HazardModel`: the
  wait-state rules as data for the shared engine, and/or the target's own
  pass.
- **Physical forms**: `thin_targets::physical_form`, the form the lifter
  gives a decoded instruction.
- **Registry**: `names()`, `get()`, `MACHINES` (the ELF machine number).
  The first name is the default.
- **metron**: the kept kernels `empty`, `differ` and `icache_inv` under
  `python/metron/src/metron/kernels/<name>/`, built by `metron.maker`.
- **Hardware test plumbing**: register the pytest marker (`@pytest.mark.<name>`)
  and `_HARDWARE_TARGETS` entry in `tests/conftest.py`, add `[<name>]` and
  `[<name>-plain]` budgets in `tests/budgets.toml`, and place the GPU parity
  suite in `tests/hw/test_<name>_hw.py` so `pytest -m <name>` runs cleanly on
  hardware and skips cleanly without it.

## 4. What RDNA 4 (`gfx1201`) and CDNA 3 (`gfx942`) exercise — and remaining `gfx950` items

Supporting `gfx942` (CDNA 3 / Instinct MI300X) alongside `gfx1201` (RDNA 4)
generalized the core across both architecture families:

1. **Wave width & register files**: wave32 (`gfx1201`) and wave64 (`gfx942`),
   64-bit `EXEC`/`VCC`, the `a` (AGPR) register file (`b.a`, `v_accvgpr_read_b32`,
   `v_accvgpr_write_b32`), and unified VGPR+AGPR occupancy accounting.
2. **Matrix units & hazards**: `MatrixForm` carries `M`, `N`, `K`, `blocks`,
   `sparse`, operand types and accumulator files. `thin-emu` executes both
   wave32 RDNA 4 WMMA and all 32 dense wave64 CDNA 3 `v_mfma_*` + 14 sparse 4:2
   `v_smfmac_*` shapes (46 total, with `cbsz`/`abid`/`blgp` broadcasts and `f64`
   `neg`), wires dense `MatrixForm` shapes into the GEMM provenance checker, and
   models pass-count-aware CDNA 3 matrix/DGEMM `s_nop` wait states in
   `thin-passes`. The DSL exposes `b.mfma`, `b.mfma_acc`, `b.smfmac`,
   `b.read_acc`, `b.write_acc`, `O.mfma` and `O.smfmac`.
3. **8-bit floats**: both OCP FP8/BF8 (`e4m3`/`e5m2` on `gfx1201`) and FNUZ
   FP8/BF8 (`e4m3fnuz`/`e5m2fnuz` on `gfx942`, `0x80` NaN, no Inf) in
   `thin-formats`, `thin.formats` and `thin-emu`.
4. **Memory, LDS DMA & bank geometry**: split RDNA 4 counters (`loadcnt` /
   `storecnt` / `dscnt` / `kmcnt`) and combined CDNA counters (`vmcnt` /
   `lgkmcnt`); `sc0` / `sc1` / `nt` scope rules across 8 XCDs; direct
   buffer/global-to-LDS DMA (`buffer_load … lds`, `global_load_lds_*`) with
   `m0` tracking across DSL, IR and `thin-emu`; `ds_read2*` / `ds_write2*`; and
   32-bank vs 64-bank LDS conflict reporting.
5. **Clocks, DPP & VOP3P**: `s_memtime` / `s_memrealtime`, GFX9 DPP controls
   (`row_bcast:15`, `row_bcast:31`, `wave_shl/shr/rol/ror:1`, `row_newbcast`),
   and `v_pk_fma_f32` / `v_pk_mul_f32` / `v_pk_add_f32` / `v_pk_mov_b32`.

Adding `gfx950` (CDNA 4) on top of this foundation primarily requires:
- **MX micro-scaled MFMA (`v_mfma_scale_*`)**: scale operands, block-scale
  provenance tags in `thin-emu`, and CDNA 4 FP4/FP6/OCP-FP8 matrix forms.
- **CDNA 4 LDS transpose reads** (`ds_read_tr*`) and wider direct-to-LDS DMA
  widths (`b128`) in the target tables and emulator.
