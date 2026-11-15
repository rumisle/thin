//! Unassignment laws on thin-produced code objects, never corpus bytes.
//!
//! Exact law: pinning the virtual program with its residue gives back the
//! physical program exactly. Semantic law: thin's own register assignment of the
//! virtual program gives a program that thin's checks accept and that
//! computes the same memory on the emulator.

mod common;

use std::collections::HashMap;

use serde_json::Value;
use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::Memory;
use thin_emu::machine::run::Machine;
use thin_ir::verify::{verify, ValueFacts};
use thin_ir::{Operand, Reg};
use thin_lift::calls::resolve_calls;
use thin_lift::interface::read_interface;
use thin_lift::program::{unrender, EntryValue, Program, Rendering};
use thin_lift::schedule::{unschedule, Schedule};
use thin_lift::unassign::{place, unassign, Assignment, Reason};
use thin_passes::regalloc;
use thin_targets::Instruction;

struct Lifted {
    program: Program,
    rendering: Rendering,
    schedule: Schedule,
    input: Vec<Instruction>,
    bytes: Vec<u8>,
}

fn lift(image: &[u8]) -> Lifted {
    let object = thin_lift::object::read(image).unwrap();
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.unwrap(),
        kernel.metadata.as_ref().unwrap(),
    )
    .unwrap()
    .0;
    let input = thin_targets::decode(object.target, &image[kernel.code.clone()]).unwrap();
    let (body, schedule) = unschedule(&input);
    let target = thin_targets::get(object.target).unwrap();
    let (program, rendering) = unrender(target, &body, &schedule, &interface).unwrap();
    let (program, calls) = resolve_calls(program).unwrap();
    assert!(calls.complete());
    Lifted {
        program,
        rendering,
        schedule,
        input,
        bytes: image[kernel.code.clone()].to_vec(),
    }
}

fn fixture(body: &str) -> Vec<u8> {
    let base = &common::kernels()[0];
    let start = base.find("\ts_mov_b32").unwrap();
    let footer = base.find("\t.section\t.rodata").unwrap();
    thin_codegen::object::code_object(&format!("{}{body}\n{}", &base[..start], &base[footer..]))
        .unwrap()
}

/// Every row's form, operands, fields and modifiers, the entry and the
/// control flow are the same.
fn same_program(a: &Program, b: &Program) {
    let (sa, sb) = (&a.snapshot, &b.snapshot);
    assert_eq!(sa.instrs().len(), sb.instrs().len());
    for (k, (x, y)) in sa.instrs().iter().zip(sb.instrs().iter()).enumerate() {
        assert_eq!(x.form().id, y.form().id, "row {k}");
        assert_eq!(x.ops(), y.ops(), "row {k}");
        assert_eq!(
            format!("{:?}", x.fields()),
            format!("{:?}", y.fields()),
            "row {k}"
        );
        assert_eq!(x.mods(), y.mods(), "row {k}");
        assert_eq!(a.line(k).unwrap(), b.line(k).unwrap());
    }
    assert_eq!(sa.values(), sb.values());
    assert_eq!(sa.entry(), sb.entry());
    assert_eq!(sa.control_flow(), sb.control_flow());
    assert_eq!(a.entry, b.entry);
}

/// The text and bytes a placed program renders to.
fn rendered(l: &Lifted, placed: &Program) -> (Vec<Instruction>, Vec<u8>) {
    let body = l.rendering.render(placed).unwrap();
    let all = l.schedule.insert(&body).unwrap();
    let bytes = thin_targets::encode(thin_targets::default_name(), &all).unwrap();
    (all, bytes)
}

/// The exact law, returning the virtual program and its residue.
fn law1(l: &Lifted) -> (Program, Assignment) {
    let (virt, residue) = unassign(&l.program, &l.schedule).unwrap();
    let pinned = residue.pin(&virt).unwrap();
    same_program(&pinned, &l.program);
    // The schedule goes back in the input order, labels before waits.
    let scheduled = l.program.scheduled(&l.schedule).unwrap();
    let forms: Vec<i32> = (scheduled.instrs().iter())
        .filter(|i| i.form().kind != thin_target::Kind::Label)
        .map(|i| i.form().id)
        .collect();
    let original: Vec<i32> = (l.input.iter())
        .map(|i| {
            thin_targets::physical_form(thin_targets::default_name(), i)
                .unwrap()
                .id
        })
        .collect();
    assert_eq!(forms, original);
    let (text, bytes) = rendered(l, &pinned);
    common::instructions_equal(&text, &l.input);
    assert_eq!(bytes, l.bytes);
    // Pinned values are exactly those with reasons, at their bases.
    for (v, info) in virt.snapshot.values().iter().enumerate() {
        assert_eq!(info.pin.is_some(), !residue.reasons[v].is_empty());
        if let Some(p) = info.pin {
            assert_eq!(p, residue.bases[v]);
        }
    }
    let facts = facts_of(&virt);
    assert_eq!(
        verify(
            virt.snapshot.target(),
            virt.snapshot.instrs(),
            &facts,
            false
        ),
        None
    );
    (virt, residue)
}

fn facts_of(p: &Program) -> HashMap<i32, ValueFacts> {
    (p.snapshot.values().iter().enumerate())
        .map(|(k, v)| {
            (
                k as i32,
                ValueFacts {
                    repr: format!("%{}{k}", v.file),
                    n: v.n,
                    align: v.align,
                    pinned: v.pin.is_some(),
                },
            )
        })
        .collect()
}

/// The semantic law's static half: thin's register assignment of the virtual
/// program, checked, placed and verified; the placed instructions.
fn law2(l: &Lifted, virt: &Program) -> Vec<Instruction> {
    // Register assignment reads the waits, as in thin's pipeline.
    let scheduled = virt.scheduled(&l.schedule).unwrap();
    let s = &scheduled;
    let target = s.target();
    let rules = thin_targets::hazards(target.name()).unwrap().rules();
    let limits: Vec<(char, u32)> = ['v', 's']
        .iter()
        .map(|&f| (f, target.hooks().reg_count(f)))
        .collect();
    let o = regalloc::Options::default();
    let placement = regalloc::assign(s, rules, &limits, o).unwrap();
    regalloc::check(s, rules, &placement.bases, o).unwrap();
    for (v, info) in virt.snapshot.values().iter().enumerate() {
        if let Some(p) = info.pin {
            assert_eq!(placement.bases[v], Some(p), "a pin is kept");
        }
    }
    let placed = place(virt, &placement.bases).unwrap();
    let ps = &placed.snapshot;
    assert_eq!(verify(target, ps.instrs(), &facts_of(&placed), true), None);
    assert!(thin_passes::checks::physical_scc(ps).is_empty());
    assert_eq!(
        thin_passes::alias::check(ps).is_ok(),
        thin_passes::alias::check(&l.program.snapshot).is_ok()
    );
    assert_eq!(
        thin_passes::checks::dead_write_findings(ps).len(),
        thin_passes::checks::dead_write_findings(&l.program.snapshot).len()
    );
    rendered(l, &placed).0
}

/// The values of each register operand of a row.
fn values_at(p: &Program, row: usize) -> Vec<Reg> {
    p.snapshot.instrs()[row]
        .ops()
        .iter()
        .filter_map(Operand::reg)
        .copied()
        .collect()
}

/// The first row whose printed physical line starts with `text`.
fn row(l: &Lifted, text: &str) -> usize {
    (0..l.program.snapshot.instrs().len())
        .find(|&k| l.program.line(k).unwrap().starts_with(text))
        .unwrap_or_else(|| panic!("no row {text}"))
}

#[test]
fn every_compiled_kernel_satisfies_law_1_and_its_reassignment_verifies() {
    for text in common::kernels() {
        let l = lift(&thin_codegen::object::code_object(&text).unwrap());
        let (virt, residue) = law1(&l);
        assert!(residue.bases.len() > 2, "values are recovered");
        law2(&l, &virt);
    }
}

/// The stores' prelude: x's address in s[20:21], each lane's byte offset
/// in v250.
const PRELUDE: &str =
    "s_load_b64 s[20:21], s[0:1], 0x0\ns_wait_kmcnt 0\nv_lshlrev_b32 v250, 2, v0\n";

/// Laws 1 and 2 on a body after the prelude, in the first compiled
/// kernel's frame, with both programs run on the emulator: the same
/// memory, from a different register assignment.
fn laws(body: &str) -> (Lifted, Program, Assignment) {
    let body = format!("{PRELUDE}{body}");
    let l = lift(&fixture(&body));
    let (virt, residue) = law1(&l);
    let placed = law2(&l, &virt);
    let k = kernel_json(common_name());
    let base = k["s"].as_str().unwrap();
    let start = base.find("\ts_mov_b32").unwrap();
    let footer = base.find("\t.section\t.rodata").unwrap();
    let s = format!("{}{body}\n{}", &base[..start], &base[footer..]);
    let symbol = k["symbol"].as_str().unwrap();
    let reassigned = with_body(&s, symbol, &placed);
    assert_ne!(
        reassigned,
        with_body(&s, symbol, &l.input),
        "registers move"
    );
    assert_eq!(run(&k, &reassigned), run(&k, &s));
    (l, virt, residue)
}

fn common_name() -> &'static str {
    "scaled_merge_b128"
}

#[test]
fn a_reused_register_holds_unrelated_values() {
    let (l, virt, residue) = laws(
        "v_mov_b32 v1, 1\nv_add_nc_u32 v2, v1, v1\nv_mov_b32 v1, 2\nv_add_nc_u32 v3, v1, v0\nglobal_store_b32 v250, v2, s[20:21]\nglobal_store_b32 v250, v3, s[20:21] offset:1024\ns_endpgm",
    );
    let first = values_at(&virt, row(&l, "v_mov_b32 v1, 1"))[0];
    let second = values_at(&virt, row(&l, "v_mov_b32 v1, 2"))[0];
    assert_ne!(first.value, second.value);
    assert!(residue.reasons[first.value as usize].is_empty());
    let v0 = values_at(&virt, row(&l, "v_add_nc_u32 v3"))[2];
    assert!(matches!(
        residue.reasons[v0.value as usize][..],
        [Reason::Entry(EntryValue::WorkitemIds(_))]
    ));
}

#[test]
fn tuples_are_one_value_with_their_parts_at_offsets() {
    let (l, virt, residue) = laws(
        "s_load_b128 s[4:7], s[0:1], 0x0\ns_wait_kmcnt 0\nv_mov_b32 v1, s6\nv_mov_b32 v2, s7\nv_lshlrev_b64 v[4:5], 2, v[1:2]\nv_lshlrev_b32 v6, 3, v0\nglobal_store_b64 v6, v[4:5], s[4:5]\ns_endpgm",
    );
    let quad = values_at(&virt, row(&l, "s_load_b128"))[0];
    assert_eq!(virt.snapshot.values()[quad.value as usize].n, 4);
    let store = values_at(&virt, row(&l, "global_store_b64"));
    assert_eq!(store[2].value, quad.value);
    assert_eq!((store[2].lo, store[2].n), (0, 2));
    let pair = values_at(&virt, row(&l, "v_lshlrev_b64"));
    assert_eq!(virt.snapshot.values()[pair[1].value as usize].n, 2);
    assert!(residue.reasons[pair[1].value as usize].is_empty());
}

#[test]
fn partial_writes_join_the_value_they_modify() {
    let (l, virt, _) = laws(
        "v_mov_b32 v1, 0\nv_mov_b32 v2, 0x3c00\nv_lshlrev_b16 v1, 1, v2\nv_writelane_b32 v1, s2, 3\nglobal_load_d16_hi_b16 v1, v250, s[20:21]\ns_wait_loadcnt 0\nv_cmp_gt_u32 vcc_lo, 5, v0\ns_and_saveexec_b32 s8, vcc_lo\nv_mov_b32 v1, 7\ns_mov_b32 exec_lo, s8\nglobal_store_b32 v250, v1, s[20:21] offset:1024\ns_endpgm",
    );
    let at = |text| values_at(&virt, row(&l, text))[0].value;
    let v = at("v_mov_b32 v1, 0");
    for text in [
        "v_lshlrev_b16",
        "v_writelane_b32",
        "global_load_d16_hi_b16",
        "v_mov_b32 v1, 7",
    ] {
        assert_eq!(at(text), v, "{text}");
    }
    assert_eq!(values_at(&virt, row(&l, "global_store_b32"))[1].value, v);
}

#[test]
fn a_value_around_a_loop_is_one_value_and_an_accumulator_stays_in_place() {
    let (l, virt, _) = laws(
        "s_mov_b32 s8, 0\nv_cvt_f32_u32 v3, v0\nv_mov_b32 v1, 0\n.Lloop:\nv_fmac_f32 v1, v3, v3\ns_add_co_u32 s8, s8, 1\ns_cmp_lt_u32 s8, 4\ns_cbranch_scc1 .Lloop\nv_mov_b32 v10, 0x3c003c00\nv_mov_b32 v11, v10\nv_mov_b32 v12, v10\nv_mov_b32 v13, v10\nv_mov_b32 v20, 0\nv_mov_b32 v21, 0\nv_mov_b32 v22, 0\nv_mov_b32 v23, 0\nv_mov_b32 v24, 0\nv_mov_b32 v25, 0\nv_mov_b32 v26, 0\nv_mov_b32 v27, v1\nv_wmma_f32_16x16x16_f16 v[20:27], v[10:13], v[10:13], v[20:27]\nglobal_store_b32 v250, v1, s[20:21]\nglobal_store_b32 v250, v27, s[20:21] offset:1024\ns_endpgm",
    );
    let at = |text| values_at(&virt, row(&l, text));
    assert_eq!(at("s_mov_b32 s8, 0")[0].value, at("s_add_co_u32")[0].value);
    assert_eq!(at("v_mov_b32 v1, 0")[0].value, at("v_fmac_f32")[0].value);
    let wmma = at("v_wmma");
    assert_eq!(wmma[0].value, wmma[3].value, "C = D stays in place");
    assert_eq!(at("global_store_b32 v250, v27")[1].value, wmma[0].value);
}

fn setup(pair: u32, temp: u32, material: u32) -> String {
    let (hi, at) = (pair + 1, material + 4);
    format!("s_getpc_b64 s[{pair}:{hi}]\ns_mov_b32 s{temp}, {at}\ns_add_co_u32 s{pair}, s{pair}, s{temp}\ns_add_co_ci_u32 s{hi}, s{hi}, 0\n")
}

#[test]
fn values_crossing_a_call_stay_pinned_and_callee_temporaries_are_free() {
    let body = format!(
        "v_mov_b32 v5, 9\n{}s_swappc_b64 s[30:31], s[12:13]\nglobal_store_b32 v250, v5, s[20:21]\nglobal_store_b32 v250, v6, s[20:21] offset:1024\ns_endpgm\n{}v_mov_b32 v7, 3\nv_add_nc_u32 v6, v7, v0\ns_setpc_b64 s[30:31]",
        setup(12, 14, 40),
        "s_nop 0\n".repeat(4)
    );
    let (l, virt, residue) = laws(&body);
    let at = |text| values_at(&virt, row(&l, text));
    let crossing = |r: Reg| residue.reasons[r.value as usize].contains(&Reason::CallBoundary);
    assert!(crossing(at("v_mov_b32 v5, 9")[0]), "live across the call");
    assert!(crossing(at("s_swappc_b64")[0]), "the link pair");
    assert!(crossing(at("v_add_nc_u32 v6")[0]), "a result");
    assert!(!crossing(at("v_mov_b32 v7, 3")[0]), "a callee temporary");
    assert!(residue.reasons[at("v_mov_b32 v7, 3")[0].value as usize].is_empty());
    let link = at("s_setpc_b64")[0];
    assert_eq!(link.value, at("s_swappc_b64")[0].value);
}

#[test]
fn a_read_before_any_write_is_pinned() {
    let (l, virt, residue) = laws("v_mov_b32 v1, 1\nglobal_store_b32 v250, v9, s[20:21]\ns_endpgm");
    let v9 = values_at(&virt, row(&l, "global_store_b32"))[1];
    assert_eq!(residue.reasons[v9.value as usize], [Reason::Undefined]);
}

#[test]
fn relative_indexing_pins_its_file() {
    let (l, virt, residue) = laws(
        "s_mov_b32 m0, 1\nv_mov_b32 v1, 5\nv_mov_b32 v2, 6\nv_movrels_b32 v3, v1\ns_mov_b32 s9, 1\nglobal_store_b32 v250, v3, s[20:21]\ns_endpgm",
    );
    let v1 = values_at(&virt, row(&l, "v_mov_b32 v1"))[0];
    assert!(residue.reasons[v1.value as usize].contains(&Reason::RelativeIndexing));
    let s9 = values_at(&virt, row(&l, "s_mov_b32 s9"))[0];
    assert!(residue.reasons[s9.value as usize].is_empty());
}

// The semantic law on the emulator: each compiled kernel and its reassignment run
// on the same inputs and leave the same memory.

fn kernel_json(name: &str) -> Value {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/thin-codegen/tests/data/codegen/kernels.json"
    );
    let all: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    all["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["name"] == name)
        .cloned()
        .unwrap()
}

/// A kernel's `.s` with its body replaced by decoded instructions.
fn with_body(s: &str, symbol: &str, body: &[Instruction]) -> String {
    let start = s.find(&format!("\n{symbol}:\n")).unwrap() + symbol.len() + 3;
    let footer = s.find("\t.section\t.rodata").unwrap();
    let targets: std::collections::BTreeSet<u32> =
        body.iter().filter_map(Instruction::branch_target).collect();
    let mut text = s[..start].to_owned();
    for i in body {
        if targets.contains(&i.offset) {
            text.push_str(&format!("{}:\n", thin_targets::decode::label(i.offset)));
        }
        text.push_str(&format!("\t{i}\n"));
    }
    let end = body.last().map_or(0, |i| i.offset + u32::from(i.size));
    if targets.contains(&end) {
        text.push_str(&format!("{}:\n", thin_targets::decode::label(end)));
    }
    text + &s[footer..]
}

/// Runs a kernel's text on deterministic inputs; every buffer's bytes.
fn run(k: &Value, s: &str) -> Vec<Vec<u8>> {
    let target = thin_targets::get(thin_targets::default_name()).unwrap();
    let program = thin_emu::read_program(s, target).unwrap();
    let mut mem = Memory::default();
    let mut kernarg = Vec::new();
    let mut ids = Vec::new();
    for (n, a) in k["args"].as_array().unwrap().iter().enumerate() {
        let offset = a["offset"].as_u64().unwrap_or(0) as usize;
        kernarg.resize(offset, 0);
        if a["kind"] == "ptr" {
            let size = 1 << 16;
            let b = mem.alloc(&format!("b{n}"), size, true, 0, Default::default());
            let b = b.unwrap();
            for (w, chunk) in mem.bufs[b].data.chunks_mut(4).enumerate() {
                let x = (w as u32 ^ (n as u32) << 20).wrapping_mul(0x9E37_79B1);
                // Small exact floats in the low half, raw bits in the high.
                let x = if w < 4096 {
                    ((x >> 20) as f32 / 64.0).to_bits()
                } else {
                    x
                };
                chunk.copy_from_slice(&x.to_le_bytes());
            }
            kernarg.extend_from_slice(&mem.bufs[b].base.to_le_bytes());
            ids.push(b);
        } else {
            let (lo, hi) = (a["lo"].as_u64().unwrap(), a["hi"].as_u64().unwrap());
            kernarg.extend_from_slice(&(300u64.clamp(lo, hi) as u32).to_le_bytes());
        }
    }
    let kb = mem
        .alloc(
            "kernarg",
            kernarg.len() as u64,
            false,
            0,
            Default::default(),
        )
        .unwrap();
    mem.bufs[kb].data[..kernarg.len()].copy_from_slice(&kernarg);
    let lds = s
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_group_segment_fixed_size "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let machine = Machine::new(program, Facts::of(target), 256, 0, lds).unwrap();
    let grid = match (k["wg"].as_u64(), k["max_grid"][0].as_u64()) {
        (Some(2), _) => (2, 3),
        (_, Some(1)) => (1, 1),
        _ => (2, 1),
    };
    let waves = k["wgsize"].as_u64().unwrap() as usize / 32;
    let mut c = Cluster::new(
        vec![machine],
        vec![vec![mem.bufs[kb].base]],
        grid,
        waves,
        Options::default(),
    );
    c.run(&mut mem, &mut thin_emu::Tables::default()).unwrap();
    ids.iter().map(|&b| mem.bufs[b].data.clone()).collect()
}

#[test]
fn every_compiled_kernel_reassigned_computes_the_same_memory() {
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../crates/thin-codegen/tests/data/codegen/kernels.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut moved = 0;
    for k in doc["kernels"].as_array().unwrap() {
        let name = k["name"].as_str().unwrap();
        let k = kernel_json(name);
        let s = k["s"].as_str().unwrap();
        let l = lift(&thin_codegen::object::code_object(s).unwrap());
        let (virt, _) = unassign(&l.program, &l.schedule).unwrap();
        let body = law2(&l, &virt);
        let symbol = k["symbol"].as_str().unwrap_or(name);
        let original = run(&k, s);
        let decoded = run(&k, &with_body(s, symbol, &l.input));
        assert_eq!(decoded, original, "{name}: the decoded text runs alike");
        let reassigned = with_body(s, symbol, &body);
        assert_eq!(run(&k, &reassigned), original, "{name}");
        moved += usize::from(reassigned != with_body(s, symbol, &l.input));
    }
    assert!(moved > 0, "some kernel's registers are reassigned");
}

/// A fixed short encoding keeps its 16-bit VGPRs below v128: placed first,
/// they stay there while 136 loaded dwords are live around them.
#[test]
fn a_fixed_short_form_keeps_its_16_bit_registers_below_v128() {
    let loads: String = (0..34)
        .map(|k| {
            format!(
                "global_load_b128 v[{}:{}], v250, s[20:21] offset:{}\n",
                4 + 4 * k,
                7 + 4 * k,
                16 * k
            )
        })
        .collect();
    let stores: String = (0..34)
        .map(|k| {
            format!(
                "global_store_b128 v250, v[{}:{}], s[20:21] offset:{}\n",
                4 + 4 * k,
                7 + 4 * k,
                1024 + 16 * k
            )
        })
        .collect();
    let body = format!(
        "{loads}s_wait_loadcnt 0\nv_mov_b32 v2, 0x3c00\nv_mov_b32 v3, 0x4000\nv_mov_b32 v1, 0\nv_max_num_f16 v1, v2, v3\n{stores}global_store_b32 v250, v1, s[20:21] offset:2048\ns_endpgm"
    );
    let (l, virt, _) = laws(&body);
    let shift = row(&l, "v_max_num_f16");
    let form = virt.snapshot.instrs()[shift].form();
    let mnemonic = |i: &&Instruction| i.mnemonic.starts_with("v_max_num_f16");
    assert_eq!(
        l.input.iter().find(mnemonic).unwrap().size,
        4,
        "the short encoding"
    );
    let limits: Vec<_> = (0..3).map(|k| form.slot(k).unwrap().limit).collect();
    assert_eq!(limits, [Some(128); 3], "every field is a 16-bit one");
    // The placement check refuses a base past the limit.
    let s = virt.scheduled(&l.schedule).unwrap();
    let rules = thin_targets::hazards(s.target().name()).unwrap().rules();
    let limits = [('v', 256), ('s', 106)];
    let o = regalloc::Options::default();
    let mut bases = regalloc::assign(&s, rules, &limits, o).unwrap().bases;
    let v = values_at(&virt, shift)[0].value as usize;
    bases[v] = Some(200);
    let refusal = regalloc::check(&s, rules, &bases, o).unwrap_err();
    assert!(format!("{:?}", refusal.msg).contains("register field"));
}

/// Correlated branches: the load is waited for on every path that issues
/// it, but the CFG has a path from the load past the wait to a write of
/// its register. That write joins the load's value; another register
/// would be held for nothing.
#[test]
fn a_write_over_a_load_that_may_be_in_flight_joins_it() {
    let (l, virt, _) = laws(
        "s_cmp_eq_u32 s2, 0\ns_cbranch_scc1 .Lskip\nglobal_load_b32 v1, v250, s[20:21]\n.Lskip:\ns_cmp_eq_u32 s2, 0\ns_cbranch_scc1 .Lend\ns_wait_loadcnt 0\nglobal_store_b32 v250, v1, s[20:21] offset:256\n.Lend:\nv_mov_b32 v1, 7\nglobal_store_b32 v250, v1, s[20:21] offset:512\ns_endpgm",
    );
    let load = values_at(&virt, row(&l, "global_load_b32"))[0].value;
    let overwrite = values_at(&virt, row(&l, "v_mov_b32 v1, 7"))[0].value;
    assert_eq!(load, overwrite);
}
