//! gfx942's code-object facts: the footer and its descriptor words.

use std::fmt::Write as _;

use thin_target::{Block, Error, Footer, Resources};

/// The block directives gfx942 takes beyond the common set.
pub static DIRECTIVES: [&str; 6] = [
    "accum_offset",
    "dx10_clamp",
    "ieee_mode",
    "tg_split",
    "user_sgpr_kernarg_preload_length",
    "user_sgpr_kernarg_preload_offset",
];

/// A YAML `key: value` line of the metadata, the value at column 17 as
/// LLVM's assembler writes it.
fn entry(out: &mut String, indent: &str, key: &str, value: impl std::fmt::Display) {
    let pad = 16usize.saturating_sub(key.len()).max(1);
    let _ = writeln!(out, "{indent}{key}:{}{value}", " ".repeat(pad));
}

/// The .s text after a kernel's body: the descriptor block and the
/// metadata for gfx942 (CDNA3, wave64, unified VGPR+AGPR file).
pub fn footer(f: &Footer) -> Result<String, Error> {
    if f.preload > 16 {
        return Err(Error::Value(format!(
            "gfx942 preloads at most 16 argument dwords (preload {})",
            f.preload
        )));
    }
    if f.preload * 4 > f.kernarg {
        return Err(Error::Value(format!(
            "preload ({} dwords) exceeds kernarg ({} bytes)",
            f.preload, f.kernarg
        )));
    }
    let vgpr = f.vgpr.max(1);
    if vgpr > 256 || f.agpr > 256 {
        return Err(Error::Value(format!(
            "gfx942 has at most 256 VGPRs and 256 AGPRs (vgpr {}, agpr {})",
            f.vgpr, f.agpr
        )));
    }
    let accum = (vgpr.div_ceil(4) * 4).max(4);
    let next_free_vgpr = if f.agpr == 0 { vgpr } else { accum + f.agpr };
    if next_free_vgpr > 512 {
        return Err(Error::Value(format!(
            "combined VGPR+AGPR allocation {next_free_vgpr} exceeds 512"
        )));
    }
    let name = &f.name;
    let mut s = String::new();
    if !f.ends {
        s += "\ts_endpgm\n";
    }
    s += "\t.section\t.rodata,\"a\",@progbits\n\t.p2align\t6, 0x0\n";
    let _ = writeln!(s, "\t.amdhsa_kernel {name}");
    let block: [(&str, u32); 30] = [
        ("group_segment_fixed_size", f.lds),
        ("private_segment_fixed_size", 0),
        ("kernarg_size", f.kernarg),
        ("user_sgpr_count", 2 + f.preload),
        ("user_sgpr_dispatch_ptr", 0),
        ("user_sgpr_queue_ptr", 0),
        ("user_sgpr_kernarg_segment_ptr", 1),
        ("user_sgpr_dispatch_id", 0),
        ("user_sgpr_private_segment_size", 0),
        ("user_sgpr_kernarg_preload_length", f.preload),
        ("user_sgpr_kernarg_preload_offset", 0),
        ("uses_dynamic_stack", 0),
        ("enable_private_segment", 0),
        ("system_sgpr_workgroup_id_x", 1),
        ("system_sgpr_workgroup_id_y", f.wgy),
        ("system_sgpr_workgroup_id_z", 0),
        ("system_sgpr_workgroup_info", 0),
        ("system_vgpr_workitem_id", 0),
        ("next_free_vgpr", next_free_vgpr),
        ("next_free_sgpr", f.sgpr),
        ("accum_offset", accum),
        ("reserve_vcc", 1),
        ("float_round_mode_32", 0),
        ("float_round_mode_16_64", 0),
        ("float_denorm_mode_32", 3),
        ("float_denorm_mode_16_64", 3),
        ("dx10_clamp", 1),
        ("ieee_mode", 1),
        ("fp16_overflow", 0),
        ("tg_split", 0),
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
    entry(&mut s, kernel, ".agpr_count", f.agpr);
    entry(&mut s, kernel, ".group_segment_fixed_size", f.lds);
    entry(&mut s, kernel, ".kernarg_segment_align", 8);
    entry(&mut s, kernel, ".kernarg_segment_size", f.kernarg);
    entry(&mut s, kernel, ".language", "OpenCL C");
    s += "    .language_version:\n      - 2\n      - 0\n";
    entry(&mut s, kernel, ".max_flat_workgroup_size", f.wgsize);
    entry(&mut s, kernel, ".name", name);
    entry(&mut s, kernel, ".private_segment_fixed_size", 0);
    entry(&mut s, kernel, ".sgpr_count", f.sgpr + 6);
    entry(&mut s, kernel, ".sgpr_spill_count", 0);
    entry(&mut s, kernel, ".symbol", format!("{name}.kd"));
    entry(&mut s, kernel, ".uniform_work_group_size", 1);
    entry(&mut s, kernel, ".uses_dynamic_stack", false);
    entry(&mut s, kernel, ".vgpr_count", next_free_vgpr);
    entry(&mut s, kernel, ".vgpr_spill_count", 0);
    entry(&mut s, kernel, ".wavefront_size", 64);
    entry(&mut s, "", "amdhsa.target", &f.target_name);
    s += "amdhsa.version:\n  - 1\n  - 2\n...\n\t.end_amdgpu_metadata\n";
    Ok(s)
}

/// Entry-interface properties (preload word and preloaded user SGPRs).
pub fn interface_resources(block: &Block) -> Result<Resources, Error> {
    let pre_len = block.bits("user_sgpr_kernarg_preload_length", 7)? as u32;
    let pre_off = block.bits("user_sgpr_kernarg_preload_offset", 9)? as u32;
    let kernarg = block.bits("kernarg_size", 32)?;
    if pre_len > 16 {
        return Err(Error::Asm(format!(
            ".amdhsa_user_sgpr_kernarg_preload_length {pre_len}: at most 16"
        )));
    }
    if pre_len > 0 && u64::from(pre_off + pre_len) * 4 > kernarg {
        return Err(Error::Asm(format!(
            "kernarg preload ({pre_len} dwords at offset {pre_off}) exceeds kernarg_size {kernarg}"
        )));
    }
    Ok(Resources {
        preload: pre_len | (pre_off << 7),
        user_sgprs: pre_len,
        ..Resources::default()
    })
}

/// gfx942's own descriptor words from a block.
pub fn resources(block: &Block) -> Result<Resources, Error> {
    let iface = interface_resources(block)?;
    let (vgpr, sgpr) = (block.get("next_free_vgpr")?, block.get("next_free_sgpr")?);
    let vgpr_blocks = vgpr.max(1).div_ceil(8) - 1;
    if vgpr > 512 || vgpr_blocks >= 64 || sgpr > 102 {
        return Err(Error::Asm(format!(
            "registers past the descriptor's fields: vgpr {vgpr}, sgpr {sgpr}"
        )));
    }
    let accum = block.get("accum_offset")?;
    if !(4..=256).contains(&accum) || accum % 4 != 0 {
        return Err(Error::Asm(format!(
            ".amdhsa_accum_offset {accum}: must be a multiple of 4 in 4..=256"
        )));
    }
    block.bits("reserve_vcc", 1)?;
    let sgpr_blocks = (sgpr + 6).div_ceil(8) - 1;
    let rsrc1 = vgpr_blocks as u32
        | (sgpr_blocks as u32) << 6
        | (block.bits("dx10_clamp", 1)? as u32) << 21
        | (block.bits("ieee_mode", 1)? as u32) << 23;
    let rsrc3 = ((accum / 4 - 1) as u32) | (block.bits("tg_split", 1)? as u32) << 16;
    Ok(Resources {
        rsrc1,
        rsrc3,
        props: iface.props,
        preload: iface.preload,
        user_sgprs: iface.user_sgprs,
    })
}
