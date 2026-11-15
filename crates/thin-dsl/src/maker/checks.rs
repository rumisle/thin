//! Instruction-form validation and constant-bus constraints.

use thin_ir::{Instr, Reg};
use thin_target::{Flags, Mem};

use super::num::{int_text, is_literal32, INLINE_F32};
use super::{value_name, Arg, Ctx, FArg};
use crate::error::{value, Result};

/// A 32-bit literal operand: an int, or a string of decimal or hex
/// digits, past the inline range and not an inline float's bits; or a
/// float that is no inline constant (+0.0, ±0.5, ±1, ±2, ±4, 1/(2π)).
fn literal(a: &Arg) -> Option<String> {
    match a {
        Arg::Int(v) if is_literal32(*v) => Some(v.to_string()),
        Arg::Str(s) => int_text(s).filter(|&v| is_literal32(v)).map(|_| s.clone()),
        Arg::Float(x) => {
            let bits = (*x as f32).to_bits();
            let inline = (*x == 0.0 && x.is_sign_positive()) || INLINE_F32.contains(&bits);
            (!inline).then(|| super::py_float(*x))
        }
        _ => None,
    }
}

fn field<'f>(fields: &'f [(String, FArg)], name: &str) -> Option<&'f FArg> {
    fields.iter().find(|f| f.0 == name).map(|f| &f.1)
}

/// The SGPR operands an instruction of that memory kind takes as VGPRs.
fn sgprs_as_vgprs(mem: Mem, args: &[Arg]) -> Vec<Reg> {
    let regs = args.iter().filter_map(|a| {
        if let Arg::Reg(r, _) = a {
            Some(*r)
        } else {
            None
        }
    });
    match mem {
        Mem::Ds => regs.filter(|r| r.file == 's').collect(),
        Mem::Global | Mem::Flat | Mem::Scratch => {
            regs.filter(|r| r.file == 's' && r.n != 2).collect()
        }
        Mem::Buffer => {
            let resource = args
                .iter()
                .position(|a| matches!(a, Arg::Reg(r, _) if r.file == 's' && r.n == 4));
            let before = &args[..resource.unwrap_or(args.len())];
            before
                .iter()
                .filter_map(|a| {
                    if let Arg::Reg(r, _) = a {
                        Some(*r)
                    } else {
                        None
                    }
                })
                .filter(|r| r.file == 's')
                .collect()
        }
        _ => Vec::new(),
    }
}

/// on the converted operands.
pub(super) fn before_form(
    c: &Ctx<'_>,
    name: &str,
    args: &[Arg],
    fields: &[(String, FArg)],
) -> Result<()> {
    let t = c.t;
    if let Some(mem) = t.hooks().mem_of(name) {
        let bad = sgprs_as_vgprs(mem, args);
        if !bad.is_empty() {
            let regs: Vec<String> = bad.iter().map(|r| c.repr(r)).collect();
            return value(format!(
                "{name}: {} an SGPR where the instruction takes a VGPR (an address / voffset / data: a V value - \
                 V.u32(x) moves it; a buffer's scalar offset is its soffset)",
                regs.join(", ")
            ));
        }
        mem_fields(c, name, mem, args, fields)?;
    }
    if let Some(mf) = t.matrix_form(name) {
        if !mf.sparse && mf.wave == 64 {
            if let (Some(Arg::Reg(rd, _)), Some(Arg::Reg(rc, _))) = (args.first(), args.get(3)) {
                if matches!(rd.file, 'a' | 'v')
                    && matches!(rc.file, 'a' | 'v')
                    && rd.file != rc.file
                {
                    return value(format!(
                        "{name}: dst {} ('{}') and src2 (C) {} ('{}') must use the same register file ('a' or 'v')",
                        c.repr(rd),
                        rd.file,
                        c.repr(rc),
                        rc.file
                    ));
                }
            }
        }
    }
    if !t.definition().operands.vop3_literal {
        vop3_literal(c, name, args)?;
    }
    if t.unit_of(name) == Some(Flags::SALU) {
        let mut lits: Vec<i128> = (args.iter())
            .filter_map(|a| if let Arg::Int(v) = a { Some(*v) } else { None })
            .filter(|v| !(-16..=64).contains(v))
            .map(|v| v & 0xFFFF_FFFF)
            .collect();
        lits.sort_unstable();
        lits.dedup();
        if lits.len() > 1 {
            let l: Vec<String> = lits.iter().map(i128::to_string).collect();
            return value(format!(
                "{name}: two literal operands [{}]: stage one in an SGPR",
                l.join(", ")
            ));
        }
    }
    Ok(())
}

/// The buffer address and the DS and global offsets.
fn mem_fields(
    c: &Ctx<'_>,
    name: &str,
    mem: Mem,
    args: &[Arg],
    fields: &[(String, FArg)],
) -> Result<()> {
    let set = |f: &str| field(fields, f).is_some_and(FArg::set);
    if mem == Mem::Buffer && args.len() == 3 && !set("lds") && name.starts_with("buffer_load_") {
        return value(format!(
            "{name}: 3 operands (the buffer-to-LDS DMA form) requires lds=True"
        ));
    }
    if mem == Mem::Buffer && !(set("offen") && set("idxen")) {
        let resource = args
            .iter()
            .position(|a| matches!(a, Arg::Reg(r, _) if r.file == 's' && r.n == 4));
        if let Some(Arg::Reg(r, _)) = resource.and_then(|k| k.checked_sub(1)).map(|k| &args[k]) {
            if r.n != 1 {
                return value(format!(
                    "{name}: vaddr {} is {} VGPRs - offen takes one (a 64-bit offset: its range past 2^32)",
                    c.repr(r),
                    r.n
                ));
            }
        }
    }
    if mem == Mem::Ds && (set("offset") || set("offset0") || set("offset1")) {
        let pair = set("offset0") || set("offset1");
        let top = if pair { 255 } else { 65535 };
        for f in ["offset", "offset0", "offset1"] {
            let o = field(fields, f).and_then(|v| match v {
                FArg::Int(x) => Some(*x),
                FArg::Str(s) => int_text(s),
                _ => None,
            });
            if let Some(o) = o.filter(|o| !(0..=top).contains(o)) {
                return value(format!(
                    "{name}: LDS offset {o} out of range (offset 0..65535, offset0 / offset1 0..255; unsigned): put \
                     the rest into the address register (a second base: addr + 65536 k, or addr - 16 for a negative \
                     one)"
                ));
            }
        }
    }
    if matches!(mem, Mem::Global | Mem::Scratch) && set("offset") {
        let range = c.t.offset_range(mem).unwrap_or(0..=0);
        if let Some(FArg::Int(o)) = field(fields, "offset") {
            if !range.contains(o) {
                let (lo, hi) = (*range.start(), *range.end());
                return value(format!(
                    "global offset {o} out of range ({lo}..{hi}): put the rest into the address or the scalar \
                     base; the families split a formula address (O.global_load(d, p + t)) themselves"
                ));
            }
        }
    }
    Ok(())
}

/// The VOP3-literal refusal of a target whose VOP3 takes none.
fn vop3_literal(c: &Ctx<'_>, name: &str, args: &[Arg]) -> Result<()> {
    let exempt = ["v_readlane", "v_writelane", "v_readfirstlane"];
    if !name.starts_with("v_") || exempt.iter().any(|e| name.starts_with(e)) {
        return Ok(());
    }
    let srcs = args.get(1..).unwrap_or(&[]);
    let Some((k, lit)) = srcs
        .iter()
        .enumerate()
        .find_map(|(k, a)| literal(a).map(|l| (k, l)))
    else {
        return Ok(());
    };
    let t = c.t;
    let vop3_only = name.ends_with("_e64")
        || !t.encodable(&format!("{name}_e32")) && t.encodable(&format!("{name}_e64"));
    let cmp = name.starts_with("v_cmp");
    let to_vcc = matches!(args.first(), Some(Arg::Phys(p, _)) if p.name.starts_with("vcc"));
    let refused = vop3_only || cmp && !to_vcc || name.starts_with("v_cndmask") || cmp && k == 1;
    if refused {
        return value(format!(
            "{name}: the literal {lit} in a VOP3 encoding, which {} takes none of: stage it in an SGPR first",
            t.name()
        ));
    }
    Ok(())
}

/// The constant bus of a VALU instruction.
pub(super) fn constant_bus(c: &Ctx<'_>, name: &str, i: &Instr, args: &[Arg]) -> Result<()> {
    let mut sgprs: Vec<Reg> = Vec::new();
    for r in i.uses().filter(|r| r.file == 's') {
        if !sgprs.contains(r) {
            sgprs.push(*r);
        }
    }
    // Named scalar sources (vcc, m0, a ttmp) use the bus too; null does not.
    let mut named: Vec<&str> = Vec::new();
    for a in args.get(1..).unwrap_or(&[]) {
        if let Arg::Phys(p, _) = a {
            let scalar = p.space == thin_target::Space::S && p.name != "null";
            if scalar && !named.contains(&p.name) {
                named.push(p.name);
            }
        }
    }
    let lit = args.get(1..).unwrap_or(&[]).iter().find_map(literal);
    let bus = c.t.definition().operands.constant_bus;
    let used = (sgprs.len() + named.len()) as u32 + u32::from(lit.is_some());
    if used <= bus {
        return Ok(());
    }
    let mut what: Vec<String> = Vec::new();
    for r in &sgprs {
        let n = value_name(c, r);
        if !what.contains(&n) {
            what.push(n);
        }
    }
    what.extend(named.iter().map(|n| (*n).to_owned()));
    what.extend(lit);
    let limit = if bus == 1 {
        "one SGPR or literal".to_owned()
    } else {
        format!("{bus} SGPRs or literals at most")
    };
    value(format!(
        "{name}: reads {}: a VALU op reads {limit} on {} (the constant bus) - move the others into VGPRs first \
         (v_mov_b32), or write it as an expression, which moves them",
        what.join(", "),
        c.t.name()
    ))
}
