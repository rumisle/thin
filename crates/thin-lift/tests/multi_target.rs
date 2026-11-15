//! Multi-target lifting laws across every registered target (`thin_targets::names()`).
//!
//! Verifies:
//! 1. Exact law (`law1`): `read -> read_interface -> decode -> unschedule -> unrender ->
//!    resolve_calls -> unassign -> pin -> render -> schedule.insert -> encode` reproduces
//!    both the physical IR `Program` and the `.text` bytes bit-for-bit on every target.
//! 2. Static semantic law (`law2`): `regalloc::assign` on the unassigned virtual program
//!    (including AGPRs where the target has them) satisfies `regalloc::check`, `verify`,
//!    and `thin-passes` checks.
//! 3. Dynamic semantic law: running the original and register-reassigned `differ` kernel
//!    on `thin-emu` produces identical global memory across every target.

use std::collections::HashMap;
use std::path::PathBuf;

use thin_emu::machine::cluster::{Cluster, Options};
use thin_emu::machine::facts::Facts;
use thin_emu::machine::memory::Memory;
use thin_emu::machine::run::Machine;
use thin_ir::verify::{verify, ValueFacts};
use thin_lift::calls::resolve_calls;
use thin_lift::interface::read_interface;
use thin_lift::program::{unrender, Program, Rendering};
use thin_lift::schedule::{unschedule, Schedule};
use thin_lift::unassign::{place, unassign, Assignment};
use thin_passes::regalloc;
use thin_target::{Footer, Target};
use thin_targets::Instruction;

struct Lifted {
    target_name: &'static str,
    target: &'static Target,
    program: Program,
    rendering: Rendering,
    schedule: Schedule,
    input: Vec<Instruction>,
    bytes: Vec<u8>,
}

fn lift(image: &[u8]) -> Lifted {
    let object = thin_lift::object::read(image).expect("read object");
    let kernel = &object.kernels[0];
    let interface = read_interface(
        &kernel.descriptor.expect("descriptor"),
        kernel.metadata.as_ref().expect("metadata"),
    )
    .expect("read_interface")
    .0;
    let input =
        thin_targets::decode(object.target, &image[kernel.code.clone()]).expect("decode text");
    let (body, schedule) = unschedule(&input);
    let target = thin_targets::get(object.target).expect("get target");
    let (program, rendering) =
        unrender(target, &body, &schedule, &interface).expect("unrender program");
    let (program, calls) = resolve_calls(program).expect("resolve calls");
    assert!(calls.complete());
    Lifted {
        target_name: object.target,
        target,
        program,
        rendering,
        schedule,
        input,
        bytes: image[kernel.code.clone()].to_vec(),
    }
}

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

fn rendered(l: &Lifted, placed: &Program) -> (Vec<Instruction>, Vec<u8>) {
    let body = l.rendering.render(placed).expect("render placed");
    let all = l.schedule.insert(&body).expect("insert schedule");
    let bytes = thin_targets::encode(l.target_name, &all).expect("encode instructions");
    (all, bytes)
}

fn law1(l: &Lifted) -> (Program, Assignment) {
    let (virt, residue) = unassign(&l.program, &l.schedule).expect("unassign");
    let pinned = residue.pin(&virt).expect("pin residue");
    same_program(&pinned, &l.program);
    let scheduled = l.program.scheduled(&l.schedule).expect("scheduled");
    let forms: Vec<i32> = (scheduled.instrs().iter())
        .filter(|i| i.form().kind != thin_target::Kind::Label)
        .map(|i| i.form().id)
        .collect();
    let original: Vec<i32> = (l.input.iter())
        .map(|i| {
            thin_targets::physical_form(l.target_name, i)
                .expect("physical form")
                .id
        })
        .collect();
    assert_eq!(forms, original);
    let (text, bytes) = rendered(l, &pinned);
    assert_eq!(text.len(), l.input.len());
    for (a, b) in text.iter().zip(&l.input) {
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.size, b.size);
        assert_eq!(a.mnemonic, b.mnemonic);
        assert_eq!(a.counted, b.counted);
        assert_eq!(a.operands(), b.operands());
        assert_eq!(a.modifiers(), b.modifiers());
        for k in 0..a.operands().len() {
            assert_eq!(a.signs(k), b.signs(k));
        }
    }
    assert_eq!(
        bytes, l.bytes,
        "byte-exact round-trip for {}",
        l.target_name
    );
    assert_eq!(
        verify(l.target, virt.snapshot.instrs(), &facts_of(&virt), false),
        None
    );
    (virt, residue)
}

fn law2(l: &Lifted, virt: &Program) -> Vec<Instruction> {
    let scheduled = virt.scheduled(&l.schedule).expect("scheduled");
    let rules = thin_targets::hazards(l.target_name)
        .expect("hazards")
        .rules();
    let limits: Vec<(char, u32)> = ['v', 's', 'a']
        .iter()
        .filter_map(|&f| {
            let n = l.target.hooks().reg_count(f);
            (n > 0).then_some((f, n))
        })
        .collect();
    let o = regalloc::Options::default();
    let placement = regalloc::assign(&scheduled, rules, &limits, o).expect("regalloc assign");
    regalloc::check(&scheduled, rules, &placement.bases, o).expect("regalloc check");
    for (v, info) in virt.snapshot.values().iter().enumerate() {
        if let Some(p) = info.pin {
            assert_eq!(placement.bases[v], Some(p), "pin preserved");
        }
    }
    let placed = place(virt, &placement.bases).expect("place");
    let ps = &placed.snapshot;
    assert_eq!(
        verify(l.target, ps.instrs(), &facts_of(&placed), true),
        None
    );
    assert!(thin_passes::checks::physical_scc(ps).is_empty());
    assert_eq!(
        thin_passes::alias::check(ps).is_ok(),
        thin_passes::alias::check(&l.program.snapshot).is_ok()
    );
    rendered(l, &placed).0
}

fn metron_kernel_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/metron/src/metron/kernels")
}

fn instructions_to_asm(
    target: &Target,
    symbol: &str,
    insns: &[Instruction],
    footer: &Footer,
) -> String {
    let mut targets: Vec<u32> = insns
        .iter()
        .filter_map(Instruction::branch_target)
        .collect();
    targets.sort_unstable();
    targets.dedup();
    let mut body = String::new();
    let mut next = targets.iter().peekable();
    for i in insns {
        while let Some(&&t) = next.peek() {
            if t > i.offset {
                break;
            }
            body.push_str(&format!("{}:\n", thin_targets::decode::label(t)));
            next.next();
        }
        body.push_str(&format!("\t{i}\n"));
    }
    for &t in next {
        body.push_str(&format!("{}:\n", thin_targets::decode::label(t)));
    }
    format!(
        "\t.amdgcn_target \"amdgcn-amd-amdhsa--{}\"\n\
         \t.amdhsa_code_object_version 6\n\
         \t.text\n\
         \t.protected\t{symbol}\n\
         \t.globl\t{symbol}\n\
         \t.p2align\t8\n\
         \t.type\t{symbol},@function\n\
         {symbol}:\n\
         {body}{}",
        target.name(),
        target.hooks().footer(footer).expect("footer")
    )
}

#[test]
fn metron_kernels_for_every_registered_target_satisfy_laws_1_and_2() {
    assert!(thin_targets::names().len() >= 2, "multi-target registry");
    for &target_name in thin_targets::names() {
        for kernel in ["differ", "icache_inv", "empty"] {
            let path = metron_kernel_dir()
                .join(target_name)
                .join(format!("{kernel}.co"));
            let image =
                std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let l = lift(&image);
            assert_eq!(l.target_name, target_name);
            let (virt, _) = law1(&l);
            law2(&l, &virt);
        }
    }
}

#[test]
fn reassigned_differ_computes_identical_memory_on_emulator_for_every_target() {
    for &target_name in thin_targets::names() {
        let path = metron_kernel_dir().join(target_name).join("differ.co");
        let image = std::fs::read(&path).expect("read differ.co");
        let l = lift(&image);
        let (virt, _) = law1(&l);
        let placed_insns = law2(&l, &virt);

        let footer = Footer {
            name: "metron_differ".into(),
            target_name: target_name.into(),
            lds: 0,
            kernarg: 28,
            vgpr: 80,
            agpr: 0,
            sgpr: 32,
            wgsize: 256,
            wgy: 1,
            preload: 0,
            ends: true,
        };
        let orig_asm = instructions_to_asm(l.target, "metron_differ", &l.input, &footer);
        let reas_asm = instructions_to_asm(l.target, "metron_differ", &placed_insns, &footer);

        let run_differ = |asm: &str| -> Vec<u32> {
            let program = thin_emu::read_program(asm, l.target).expect("read_program");
            let mut mem = Memory::default();
            let stride: u32 = 64;
            let copies: usize = 3;
            let ref_buf: Vec<u8> = (0..stride as u8).collect();
            let mut base_buf = ref_buf.repeat(copies);
            // Mutate copy 1 byte 19 -> expected flags = [0, 1, 0]
            base_buf[stride as usize + 19] ^= 0x55;
            let b_base = mem
                .alloc("base", base_buf.len() as u64, false, 0, Default::default())
                .expect("alloc base");
            mem.bufs[b_base].data[..base_buf.len()].copy_from_slice(&base_buf);
            let b_ref = mem
                .alloc("ref", ref_buf.len() as u64, false, 0, Default::default())
                .expect("alloc ref");
            mem.bufs[b_ref].data[..ref_buf.len()].copy_from_slice(&ref_buf);
            let b_flags = mem
                .alloc("flags", (copies * 4) as u64, true, 0, Default::default())
                .expect("alloc flags");

            let mut kargs = Vec::with_capacity(28);
            kargs.extend_from_slice(&mem.bufs[b_base].base.to_le_bytes());
            kargs.extend_from_slice(&mem.bufs[b_ref].base.to_le_bytes());
            kargs.extend_from_slice(&mem.bufs[b_flags].base.to_le_bytes());
            kargs.extend_from_slice(&stride.to_le_bytes());
            let kb = mem
                .alloc("kernarg", kargs.len() as u64, false, 0, Default::default())
                .expect("alloc kernarg");
            mem.bufs[kb].data[..kargs.len()].copy_from_slice(&kargs);

            let facts = Facts::of(l.target);
            let waves = 256 / facts.w;
            let machine = Machine::new(program, facts, 80, 0, 0).expect("Machine");
            let mut cluster = Cluster::new(
                vec![machine],
                vec![vec![mem.bufs[kb].base]],
                (1, copies as u32),
                waves,
                Options::default(),
            );
            cluster
                .run(&mut mem, &mut thin_emu::Tables::default())
                .expect("emulate differ");
            mem.bufs[b_flags].data[..copies * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let expected = vec![0, 1, 0];
        assert_eq!(
            run_differ(&orig_asm),
            expected,
            "orig differ on {target_name}"
        );
        assert_eq!(
            run_differ(&reas_asm),
            expected,
            "reassigned differ on {target_name}"
        );
    }
}

#[test]
fn agpr_and_mfma_kernel_lifts_unassigns_and_reassigns() {
    let mut tested = 0;
    for &target_name in thin_targets::names() {
        let target = thin_targets::get(target_name).unwrap();
        if target.hooks().reg_count('a') == 0 {
            continue;
        }
        tested += 1;
        let footer = Footer {
            name: "lift_mfma".into(),
            target_name: target_name.into(),
            lds: 0,
            kernarg: 8,
            vgpr: 8,
            agpr: 4,
            sgpr: 8,
            wgsize: 64,
            wgy: 0,
            preload: 0,
            ends: true,
        };
        let asm = format!(
            "\t.amdgcn_target \"amdgcn-amd-amdhsa--{target_name}\"\n\
             \t.amdhsa_code_object_version 6\n\
             \t.text\n\
             \t.protected\tlift_mfma\n\
             \t.globl\tlift_mfma\n\
             \t.p2align\t8\n\
             \t.type\tlift_mfma,@function\n\
             lift_mfma:\n\
             \ts_load_dwordx2 s[4:5], s[0:1], 0x0\n\
             \tv_mov_b32 v2, 0x3c003c00\n\
             \tv_mov_b32 v3, 0x3c003c00\n\
             \tv_mov_b32 v4, 0x3c003c00\n\
             \tv_mov_b32 v5, 0x3c003c00\n\
             \tv_accvgpr_write_b32 a0, 0\n\
             \tv_accvgpr_write_b32 a1, 0\n\
             \tv_accvgpr_write_b32 a2, 0\n\
             \tv_accvgpr_write_b32 a3, 0\n\
             \ts_nop 1\n\
             \tv_mfma_f32_16x16x16_f16 a[0:3], v[2:3], v[4:5], a[0:3]\n\
             \ts_nop 17\n\
             \tv_accvgpr_read_b32 v4, a0\n\
             \tv_accvgpr_read_b32 v5, a1\n\
             \tv_accvgpr_read_b32 v6, a2\n\
             \tv_accvgpr_read_b32 v7, a3\n\
             \tv_lshlrev_b32 v1, 4, v0\n\
             \ts_waitcnt lgkmcnt(0)\n\
             \tglobal_store_dwordx4 v1, v[4:7], s[4:5]\n\
             \ts_endpgm\n\
             {}",
            target.hooks().footer(&footer).unwrap()
        );
        let co = thin_codegen::object::code_object(&asm).expect("build mfma code object");
        let l = lift(&co);
        let (virt, _) = law1(&l);
        // Verify virtual 'a' values were recovered from physical a[0:3]
        assert!(
            virt.snapshot
                .values()
                .iter()
                .any(|v| v.file == 'a' && v.n == 4 && v.pin.is_none()),
            "unassigned virtual 4-dword AGPR tuple on {target_name}"
        );
        law2(&l, &virt);
    }
    assert!(tested >= 1, "at least one registered target has AGPRs");
}
