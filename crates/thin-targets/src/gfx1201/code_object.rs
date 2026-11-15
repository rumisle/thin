//! gfx1201's code-object facts: the footer and its descriptor words
//!

use std::fmt::Write as _;

use thin_target::{Block, Error, Footer, Resources};

/// The block directives gfx1201 takes beyond the common set.
pub static DIRECTIVES: [&str; 5] = [
    "wavefront_size32",
    "workgroup_processor_mode",
    "memory_ordered",
    "forward_progress",
    "round_robin_scheduling",
];

/// A YAML `key: value` line of the metadata, the value at column 17 as
/// LLVM's assembler writes it.
fn entry(out: &mut String, indent: &str, key: &str, value: impl std::fmt::Display) {
    let pad = 16usize.saturating_sub(key.len()).max(1);
    let _ = writeln!(out, "{indent}{key}:{}{value}", " ".repeat(pad));
}

/// The .s text after a kernel's body: the descriptor block and the
/// metadata. Refuses agpr or preloaded arguments, which gfx1201
/// does not have.
pub fn footer(f: &Footer) -> Result<String, Error> {
    if f.agpr != 0 {
        return Err(Error::Value(format!(
            "gfx1201 has no AGPRs (agpr {})",
            f.agpr
        )));
    }
    if f.preload != 0 {
        return Err(Error::Value(format!(
            "gfx1201 preloads no kernel arguments (preload {})",
            f.preload
        )));
    }
    let (name, vgpr) = (&f.name, f.vgpr.max(1));
    let mut s = String::new();
    if !f.ends {
        s += "\ts_endpgm\n";
    }
    s += "\t.section\t.rodata,\"a\",@progbits\n\t.p2align\t6, 0x0\n";
    let _ = writeln!(s, "\t.amdhsa_kernel {name}");
    let block: [(&str, u32); 29] = [
        ("group_segment_fixed_size", f.lds),
        ("private_segment_fixed_size", 0),
        ("kernarg_size", f.kernarg),
        ("user_sgpr_count", 2),
        ("user_sgpr_dispatch_ptr", 0),
        ("user_sgpr_queue_ptr", 0),
        ("user_sgpr_kernarg_segment_ptr", 1),
        ("user_sgpr_dispatch_id", 0),
        ("user_sgpr_private_segment_size", 0),
        ("wavefront_size32", 1),
        ("uses_dynamic_stack", 0),
        ("enable_private_segment", 0),
        ("system_sgpr_workgroup_id_x", 1),
        ("system_sgpr_workgroup_id_y", f.wgy),
        ("system_sgpr_workgroup_id_z", 0),
        ("system_sgpr_workgroup_info", 0),
        ("system_vgpr_workitem_id", 0),
        ("next_free_vgpr", vgpr),
        ("next_free_sgpr", f.sgpr),
        ("reserve_vcc", 1),
        ("float_round_mode_32", 0),
        ("float_round_mode_16_64", 0),
        ("float_denorm_mode_32", 3),
        ("float_denorm_mode_16_64", 3),
        ("fp16_overflow", 0),
        ("workgroup_processor_mode", 1),
        ("memory_ordered", 1),
        ("forward_progress", 1),
        ("round_robin_scheduling", 0),
    ];
    for (key, value) in block {
        let _ = writeln!(s, "\t\t.amdhsa_{key} {value}");
    }
    s += "\t.end_amdhsa_kernel\n\t.text\n.Lfunc_end0:\n";
    let _ = writeln!(s, "\t.size\t{name}, .Lfunc_end0-{name}");
    s += "\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n";
    entry(&mut s, "      - ", ".offset", 0);
    entry(&mut s, "        ", ".size", f.kernarg);
    entry(&mut s, "        ", ".value_kind", "by_value");
    let kernel = "    ";
    entry(&mut s, kernel, ".group_segment_fixed_size", f.lds);
    entry(&mut s, kernel, ".kernarg_segment_align", 8);
    entry(&mut s, kernel, ".kernarg_segment_size", f.kernarg);
    entry(&mut s, kernel, ".language", "OpenCL C");
    s += "    .language_version:\n      - 2\n      - 0\n";
    entry(&mut s, kernel, ".max_flat_workgroup_size", f.wgsize);
    entry(&mut s, kernel, ".name", name);
    entry(&mut s, kernel, ".private_segment_fixed_size", 0);
    entry(&mut s, kernel, ".sgpr_count", f.sgpr + 2);
    entry(&mut s, kernel, ".sgpr_spill_count", 0);
    entry(&mut s, kernel, ".symbol", format!("{name}.kd"));
    entry(&mut s, kernel, ".uniform_work_group_size", 1);
    entry(&mut s, kernel, ".uses_dynamic_stack", false);
    entry(&mut s, kernel, ".vgpr_count", vgpr);
    entry(&mut s, kernel, ".vgpr_spill_count", 0);
    entry(&mut s, kernel, ".wavefront_size", 32);
    entry(&mut s, kernel, ".workgroup_processor_mode", 1);
    entry(&mut s, "", "amdhsa.target", &f.target_name);
    s += "amdhsa.version:\n  - 1\n  - 2\n...\n\t.end_amdgpu_metadata\n";
    Ok(s)
}

/// Entry-interface properties, independent of the compiler's wave32-only
/// register allocation policy. Wave64 descriptors are valid shipped inputs.
pub fn interface_resources(block: &Block) -> Result<Resources, Error> {
    Ok(Resources {
        props: (block.bits("wavefront_size32", 1)? as u32) << 10,
        ..Resources::default()
    })
}

/// gfx1201's own descriptor words from a block.
pub fn resources(block: &Block) -> Result<Resources, Error> {
    let wave32 = block.get("wavefront_size32")?;
    if wave32 != 1 {
        return Err(Error::Asm(format!(
            ".amdhsa_wavefront_size32 {wave32}: gfx1201's kernels are wave32"
        )));
    }
    let (vgpr, sgpr) = (block.get("next_free_vgpr")?, block.get("next_free_sgpr")?);
    let vgpr_blocks = vgpr.max(1).div_ceil(8) - 1;
    if vgpr_blocks >= 64 || sgpr > 106 {
        return Err(Error::Asm(format!(
            "registers past the descriptor's fields: vgpr {vgpr}, sgpr {sgpr}"
        )));
    }
    block.bits("reserve_vcc", 1)?;
    let rsrc1 = vgpr_blocks as u32
        | (block.bits("round_robin_scheduling", 1)? as u32) << 21
        | (block.bits("workgroup_processor_mode", 1)? as u32) << 29
        | (block.bits("memory_ordered", 1)? as u32) << 30
        | (block.bits("forward_progress", 1)? as u32) << 31;
    Ok(Resources {
        rsrc1,
        rsrc3: 0,
        props: interface_resources(block)?.props,
        preload: 0,
        user_sgprs: 0,
    })
}
