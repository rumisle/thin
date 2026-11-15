# Adding a target

A target is one crate module under `crates/thin-targets/src/<name>/`
plus its machine-readable tables under `crates/thin-targets/data/<name>/`.
No other crate may name it; `tests/test_architecture.py` checks that.
The definition and hooks below supply the hardware facts used by core code.

## 1. The definition (`thin_target::Definition`)

`build()` returns the definition, and `thin_target::Target::build` checks it.

| Item | What it is |
|---|---|
| `name` | The target name (`gfx1201`), as code objects and the registry spell it. |
| `lds_max` | Maximum LDS bytes available to one workgroup. |
| `counter_names` | Counter names, in the same order as `counters`. |
| `wave` | Lanes per wave, 32 or 64: the width of lane masks, of EXEC and VCC, of the emulator's waves, and of the DSL's masks. |
| `files` | The register files (`v`, `s`, `a`), each with its size, allocation granule and the alignment of a 64-bit or wider range. |
| `forms` | One form per (op, arity, returning), built from the operand table by `FormRules` (below): its slots, fields, flags, kind, counter, `dst_sel`, memory class and memory bytes. |
| `encodings` | The mnemonics its encoder has an opcode for. |
| `counters` | The wait counters: name, largest count, wait field, the units it counts (VMEM, DS, SMEM) and when its results land (at issue, or at the wait). The emulator maps them onto its four counters (vector loads, vector stores, LDS, scalar memory). One counter may cover several of them. |
| `waits` | The wait instructions: mnemonic, count fields, and whether the line spells them bare or by name. |
| `hazards` | The hazard model: the number of wait-state rules, whether the target has its own pass, the rules' window, and the issue hint's mnemonic (gfx1201: `s_delay_alu`). |
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
| `nop_word`, `e_flags`, `granule`, `directives`, `footer`, `resources`, `interface_resources` | The code object: the padding word, the ELF flags, the VGPR block, the `.amdhsa_*` names and the kernel descriptor. |
| `matrix_form` | Per matrix mnemonic: operand types, K, the summation model. |
| `scope_rule` | How memory instructions name their scope and temporal policy: the fields of the scope code, the scope of each code for plain accesses and atomics, when write-backs and invalidates reach the L2, the non-temporal hints, the IPC policy and its text, and the field that makes an atomic return. |
| `lds_banks` | The LDS bank rule per access width, for the bank-conflict report (none: no report). |
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

## 4. Porting to CDNA (gfx950): known gaps

gfx950 (CDNA4) supplies all of the above as data or hooks. That covers
64-lane waves, an `a` file, its counters (`vmcnt` covering loads and
stores, `lgkmcnt` covering LDS and scalar memory), `s_waitcnt` with named
counts, `sc0` / `sc1` / `nt` as a scope rule, workgroup ids after the user
SGPRs, 8 XCDs, its code object and its hazard rules. These parts would
need core changes:

1. **The field vocabulary.** `thin_target::FIELD_NAMES` is a closed list
   holding gfx1201's fields and thin's own. gfx950's `sc0`, `sc1`, `nt`,
   `lds`, `cbsz`, `abid`, `blgp` and the SDWA selects must be added to it,
   or the list must be built per target.
2. **The matrix unit.** `MatrixForm` gives types and K, not M, N or the
   register layout. The emulator's `wmma` is RDNA4's 32-lane 16×16
   layout. MFMA shapes and layouts need a model, and the hazard engine
   needs a pass-count state, with the pass count supplied by the target.
3. **Three gfx950 features thin does not have.** gfx950 support needs
   LDS DMA (`buffer_load … lds`, `global_load_lds_*`), AGPRs (the `a`
   file's moves and their DSL spelling) and MX scaling (the scaled matrix
   ops, their scale operands and the emulator's scale tags). None is in
   the DSL, the IR or the emulator; each needs all three.
4. **Clocks.** The emulator models the clock reads gfx1201's hooks name
   (`s_getreg_b32` of SHADER_CYCLES, `s_sendmsg_rtn_b64` REALTIME).
   gfx950's `s_memtime` / `s_memrealtime` need modelling.
5. **Ops gfx1201 lacks**: `v_pk_*_f32`, the transposed LDS reads and
   gfx9's DPP controls. Each needs emulator semantics.
6. **Encodings in the shared rules.** `mem_of_encoding` and the slot rules
   know gfx1201's encodings (`ENC_VBUFFER`, `ENC_VGLOBAL`, …). gfx950's
   `ENC_MUBUF` / `ENC_FLAT` / `ENC_DS` need a hook or an entry there.

Nothing in the core assumes gfx1201's wave size, register files, granules,
counters, scope bits, barrier or hazards. Those come from the target
description.
