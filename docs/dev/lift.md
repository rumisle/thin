# Lifting code objects

`thin-lift` reads AMDGPU code objects and recovers an interface, a physical
instruction program and virtual register values. It is a Rust library with
example drivers (below). Its inverse operations retain typed residue for
information a higher-level representation cannot express.

Its inputs are usually *shipped kernels*: code objects produced by other
compilers or libraries rather than by thin. The rest of the tree uses the
term the same way, for instance for the ops the emulator models because
such kernels use them.

## Steps and laws

| Step | Input | Output | Law and limits |
|---|---|---|---|
| `object::read` | ELF64 AMDGPU bytes | Sections, symbols, per-kernel code ranges, descriptors and metadata | Validates structure and bounds. Returns `ObjectError` for malformed or unsupported images; this is a reader, not a general ELF rewriter. |
| `interface::read_interface` | Descriptor, metadata, target | `Interface` and `InterfaceResidue` | Writing the interface into the supplied template plus residue reconstructs the descriptor. The interface consumes only ABI fields; allocation fields, reserved bits, entry offsets and unknown metadata remain in residue. Invalid interface data returns `ObjectError`. |
| target decode | Kernel code bytes | Decoded instructions with original byte offsets | Every successful decode re-encodes identically. Unsupported forms and invalid branch boundaries return typed decode refusals. |
| `schedule::unschedule` | Decoded instructions | Body plus `Schedule` | Reinserting the schedule restores all decoded instructions and their order. Residue retains complete instructions at original indices. Invalid insertion positions, ordering or size overflow return `ScheduleError`. |
| `branches::unrelax_branches` | Body and schedule | Short-branch view plus `BranchRewrites` | Restoring rewrites reproduces the original sequences exactly. Each rewrite retains scratch registers, SCC effects and original instructions. Recognition requires resolved in-range instruction boundaries; unmatched patterns stay explicit. Modified replacements or malformed residue return `RewriteError`. |
| `program::unrender` | Body, interface and target | `Program` plus `Rendering` | Rendering with the residue reproduces physical operands and encoding/layout choices. The program shares pinned backing values per register file so overlapping ranges remain aliased. `ProgramError` carries an optional original offset, a stable class and a detailed cause. |
| `calls::resolve_calls` | Physical `Program` | The same instructions with call/return control-flow edges and transfer findings | Resolution changes control-flow knowledge, not machine instructions. Whole SGPR pairs and matching carry relations preserve relocation invariance. Unknown transfers retain explicit typed reasons instead of guessed targets. |
| `unassign::unassign` | Physical program and schedule | Virtual-value program plus `Assignment` | **Exact law:** `Assignment::pin` restores the physical program, including entry and control flow. **Semantic law:** placing the virtual program with a valid allocation preserves its computation over the represented control flow. Static audits and emulator execution tests exercise this second law. |

Offsets stay in the input address space during unscheduling and branch
recognition. Those views are not relocated executable images. Restoring a
long branch's scratch writes does not prove those writes dead; side effects
remain explicit in its residue.

## Control flow and refusals

Call resolution propagates finite sets of complete address pairs. Scalar
sets are bounded at 64 and address/link sets at 4096. Typed reasons include
unknown entry input, overwritten dependencies, domain limits, widened loop
values, lost correlations, partial pairs, unrelated carry, non-PC-relative
values, non-numbered operands and invalid instruction boundaries.

Branches are may-edges, not a proof that a path is feasible at runtime.
Returns are paired with their named caller; this does not infer a general
calling convention for arbitrary registers. Incomplete control flow remains
marked as such. Dead-write findings on incomplete graphs are conditional,
and a dead destination does not make an instruction's other effects removable.

## Recovering virtual values

Unassignment joins reaching writes for each read dword, then groups the dword
webs an operand needs together. Partial writes, read-modify-write operations,
in-place updates and writes over a still-pending load preserve required value
identity. Named registers remain physical operands.

`Assignment::reasons` explains every value that must keep its register:
hardware/ABI entry input, a read without a reaching write, liveness across a
call boundary, relative register indexing, or unresolved control flow. Free
values can be placed elsewhere; `ProgramError` reports incompatible widths,
register constraints or failed reconstruction instead of discarding residue.

## Tests and examples

Tests compile thin kernels or construct synthetic instruction streams; no
third-party code objects or disassembly fixtures are included. Round trips
check descriptor fields, schedule reinsertion, branch restoration, exact
physical rendering and register pinning. Semantic checks execute paired
programs with the same inputs and compare outputs and preserved state.

Examples accept user-supplied files or directories:

```sh
cargo run --release -p thin-lift --example lift_census -- PATH...
cargo run --release -p thin-lift --example lift_bench -- --top 10 PATH...
python -m tools.lift_call_census --json counts.json PATH...
```

Generated reports describe the supplied inputs. Do not commit code objects,
third-party disassembly or corpus-specific reports as test data.
