//! High-level MFMA and SMFMAC instruction making and accumulator rules.

use thin_target::{Elem, MatrixForm, Target};

use super::{op, Arg, Ctx, FArg, Made, Notes};
use crate::error::{value, Result};
use crate::types::Ty;

/// Modifier options for a dense, MX-scaled, or sparse MFMA instruction.
#[derive(Clone, Debug, Default)]
pub struct MfmaOpts {
    /// Source-A broadcast block size or `f8f6f4` A-format code (`cbsz`, 0..=4 dense or 0..=3 sparse).
    pub cbsz: i128,
    /// Source-A broadcast block / index-set selector (`abid`).
    pub abid: i128,
    /// Source-B lane-group broadcast/permutation or `f8f6f4` B-format code (`blgp`, 0..=7 or 0..=4).
    pub blgp: i128,
    /// Optional symbolic A-format name (`"fp8"`, `"bf8"`, `"fp6"`, `"bf6"`, `"fp4"`, or OCP dtype name) on `f8f6f4` forms.
    pub fmt_a: Option<String>,
    /// Optional symbolic B-format name (`"fp8"`, `"bf8"`, `"fp6"`, `"bf6"`, `"fp4"`, or OCP dtype name) on `f8f6f4` forms.
    pub fmt_b: Option<String>,
    /// E8M0 byte selector (`0..=3`) within `scale_src0` on `v_mfma_scale_*`.
    pub sel_a: i128,
    /// E8M0 byte selector (`0..=3`) within `scale_src1` on `v_mfma_scale_*`.
    pub sel_b: i128,
    /// Optional 3-element negation mask (`neg:[a, b, c]`) on binary64 MFMA forms.
    pub neg: Option<Vec<i128>>,
    /// Comment, tag, and ablation notes.
    pub notes: Notes,
}

/// Parses an `f8f6f4` format name (`"fp8"`/`"e4m3fn"`, `"bf8"`/`"e5m2"`, `"fp6"`/`"e2m3"`,
/// `"bf6"`/`"e3m2"`, `"fp4"`/`"e2m1"`) into its hardware 3-bit code `0..=4`.
pub fn parse_f8f6f4_fmt(name: &str) -> Result<i128> {
    match name.trim().to_ascii_lowercase().as_str() {
        "fp8" | "e4m3" | "e4m3fn" => Ok(0),
        "bf8" | "e5m2" => Ok(1),
        "fp6" | "e2m3" | "e2m3fn" => Ok(2),
        "bf6" | "e3m2" | "e3m2fn" => Ok(3),
        "fp4" | "e2m1" | "e2m1fn" => Ok(4),
        _ => value(format!(
            "f8f6f4 format {}: expected one of 'fp8' ('e4m3fn'), 'bf8' ('e5m2'), 'fp6' ('e2m3'), 'bf6' ('e3m2'), 'fp4' ('e2m1')",
            super::py_repr(name)
        )),
    }
}

/// Chooses the accumulator register file (`'a'` or `'v'`) for `t`, honouring
/// explicit `acc` (`"a"` / `"v"`) or `agpr` (`true` -> `'a'`, `false` -> `'v'`),
/// and defaulting to `'a'` when the target provides an accumulator file.
pub fn choose_acc_file(t: &Target, acc: Option<&str>, agpr: Option<bool>) -> Result<char> {
    let from_acc = match acc {
        None => None,
        Some("a") => Some('a'),
        Some("v") => Some('v'),
        Some(other) => {
            return value(format!(
                "acc={}: 'a' (AGPR) or 'v' (VGPR)",
                super::py_repr(other)
            ))
        }
    };
    let from_agpr = agpr.map(|use_a| if use_a { 'a' } else { 'v' });
    let letter = match (from_acc, from_agpr) {
        (Some(x), Some(y)) if x != y => {
            return value(format!(
                "acc='{}' and agpr={} disagree: give one",
                x,
                agpr.unwrap_or(false)
            ))
        }
        (Some(x), _) | (None, Some(x)) => x,
        (None, None) => {
            if t.file('a').is_some() {
                'a'
            } else {
                'v'
            }
        }
    };
    if t.file(letter).is_none() {
        return value(format!(
            "acc='{letter}': {} has no {letter} registers (use acc='v' / agpr=False)",
            t.name()
        ));
    }
    Ok(letter)
}

/// The DSL element type and element count of an accumulator for `mf`.
pub fn acc_shape(mf: &MatrixForm) -> (Ty, u32) {
    let dwords = mf.dwords('d');
    match mf.d {
        Elem::F32 => (Ty::F32, dwords),
        Elem::I32 => (Ty::I32, dwords),
        Elem::F64 => (Ty::U64, dwords / 2),
        _ => (Ty::U32, dwords),
    }
}

/// Resolves any matrix form (dense MFMA, MX-scaled MFMA, sparse SMFMAC, or WMMA)
/// on `t` from a short shape (`"f32_16x16x16_f16"`), a prefixed form (`"mfma_..."`
/// / `"mfma_scale_..."` / `"smfmac_..."`), or a full/dotted mnemonic.
pub fn resolve_matrix_form(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    let candidates = [
        norm.clone(),
        format!("v_{norm}"),
        format!("v_mfma_{norm}"),
        format!("v_mfma_scale_{norm}"),
        format!("v_smfmac_{norm}"),
        format!("v_wmma_{norm}"),
    ];
    for cand in candidates {
        if let Some(spelled) = t.spell(&cand) {
            if let Some(mf) = t.matrix_form(&spelled) {
                return Ok((spelled, mf));
            }
        }
    }
    value(format!(
        "matrix form {}: {} has no such matrix form",
        super::py_repr(form),
        t.name()
    ))
}

/// Resolves an unscaled dense `v_mfma_*` form on `t`.
pub fn resolve_mfma(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    if norm.starts_with("smfmac_") || norm.starts_with("v_smfmac_") {
        return value(format!(
            "mfma({}): sparse 4:2 matrix op — use smfmac",
            super::py_repr(form)
        ));
    }
    if norm.starts_with("scale_")
        || norm.starts_with("mfma_scale_")
        || norm.starts_with("v_mfma_scale_")
    {
        return value(format!(
            "mfma({}): MX-scaled matrix op — use mfma_scale",
            super::py_repr(form)
        ));
    }
    let candidates = [
        if norm.starts_with("v_mfma_") {
            Some(norm.clone())
        } else {
            None
        },
        if norm.starts_with("mfma_") {
            Some(format!("v_{norm}"))
        } else {
            None
        },
        Some(format!("v_mfma_{norm}")),
    ];
    for cand in candidates.into_iter().flatten() {
        if let Some(spelled) = t.spell(&cand) {
            if let Some(mf) = t.matrix_form(&spelled).filter(|m| !m.sparse && !m.scaled) {
                return Ok((spelled, mf));
            }
        }
    }
    value(format!(
        "mfma({}): {} has no such dense MFMA form",
        super::py_repr(form),
        t.name()
    ))
}

/// Resolves an MX-scaled `v_mfma_scale_*` form on `t`.
pub fn resolve_mfma_scale(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    if norm.starts_with("smfmac_") || norm.starts_with("v_smfmac_") {
        return value(format!(
            "mfma_scale({}): sparse 4:2 matrix op — use smfmac",
            super::py_repr(form)
        ));
    }
    let candidates = [
        if norm.starts_with("v_mfma_scale_") {
            Some(norm.clone())
        } else {
            None
        },
        if norm.starts_with("mfma_scale_") {
            Some(format!("v_{norm}"))
        } else {
            None
        },
        if norm.starts_with("scale_") {
            Some(format!("v_mfma_{norm}"))
        } else {
            None
        },
        Some(format!("v_mfma_scale_{norm}")),
    ];
    for cand in candidates.into_iter().flatten() {
        if let Some(spelled) = t.spell(&cand) {
            if let Some(mf) = t.matrix_form(&spelled).filter(|m| m.scaled) {
                return Ok((spelled, mf));
            }
        }
    }
    value(format!(
        "mfma_scale({}): {} has no such MX-scaled MFMA form",
        super::py_repr(form),
        t.name()
    ))
}

/// Resolves a sparse 4:2 `v_smfmac_*` form on `t`.
pub fn resolve_smfmac(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    if norm.starts_with("mfma_") || norm.starts_with("v_mfma_") {
        return value(format!(
            "smfmac({}): dense matrix op — use mfma",
            super::py_repr(form)
        ));
    }
    let candidates = [
        if norm.starts_with("v_smfmac_") {
            Some(norm.clone())
        } else {
            None
        },
        if norm.starts_with("smfmac_") {
            Some(format!("v_{norm}"))
        } else {
            None
        },
        Some(format!("v_smfmac_{norm}")),
    ];
    for cand in candidates.into_iter().flatten() {
        if let Some(spelled) = t.spell(&cand) {
            if let Some(mf) = t.matrix_form(&spelled).filter(|m| m.sparse) {
                return Ok((spelled, mf));
            }
        }
    }
    value(format!(
        "smfmac({}): {} has no such sparse SMFMAC form",
        super::py_repr(form),
        t.name()
    ))
}

fn check_vec_reg(
    c: &Ctx<'_>,
    mnemonic: &str,
    role: &str,
    arg: &Arg,
    expected_dwords: u32,
    allow_const: bool,
    only_v: bool,
) -> Result<()> {
    match arg {
        Arg::Reg(r, m) => {
            if *m != 0 {
                return value(format!(
                    "{mnemonic}: {role} {} carries source modifiers; pass neg=[a,b,c] where supported",
                    c.repr(r)
                ));
            }
            if only_v && r.file != 'v' {
                return value(format!(
                    "{mnemonic}: {role} {} is in the '{}' file; sparse index takes 'v' (VGPR)",
                    c.repr(r),
                    r.file
                ));
            }
            if !matches!(r.file, 'a' | 'v') {
                return value(format!(
                    "{mnemonic}: {role} {} is in the '{}' file; matrix operands take 'a' (AGPR) or 'v' (VGPR)",
                    c.repr(r),
                    r.file
                ));
            }
            if c.t.file(r.file).is_none() {
                return value(format!(
                    "{mnemonic}: {role} {}: {} has no {} registers",
                    c.repr(r),
                    c.t.name(),
                    r.file
                ));
            }
            if r.n != expected_dwords {
                return value(format!(
                    "{mnemonic}: {role} {} is {} dword(s); form takes {expected_dwords}",
                    c.repr(r),
                    r.n
                ));
            }
            Ok(())
        }
        Arg::Int(_) | Arg::Float(_) | Arg::Str(_) if allow_const => Ok(()),
        _ => value(format!(
            "{mnemonic}: {role} must be a {expected_dwords}-dword {} register{}",
            if only_v { "'v'" } else { "'a' or 'v'" },
            if allow_const {
                " (or an inline constant)"
            } else {
                ""
            }
        )),
    }
}

fn resolve_effective_opts(mnemonic: &str, mf: &MatrixForm, opts: &mut MfmaOpts) -> Result<()> {
    let is_f8f6f4 = mf.a == Elem::F8F6F4;
    if let Some(fa) = opts.fmt_a.as_deref() {
        if !is_f8f6f4 {
            return value(format!(
                "{mnemonic}: fixed operand format (fmt_a applies only to f8f6f4 forms)"
            ));
        }
        let code = parse_f8f6f4_fmt(fa)?;
        if opts.cbsz != 0 && opts.cbsz != code {
            return value(format!(
                "{mnemonic}: fmt_a={} (cbsz={code}) and cbsz={} disagree: give one",
                super::py_repr(fa),
                opts.cbsz
            ));
        }
        opts.cbsz = code;
    }
    if let Some(fb) = opts.fmt_b.as_deref() {
        if !is_f8f6f4 {
            return value(format!(
                "{mnemonic}: fixed operand format (fmt_b applies only to f8f6f4 forms)"
            ));
        }
        let code = parse_f8f6f4_fmt(fb)?;
        if opts.blgp != 0 && opts.blgp != code {
            return value(format!(
                "{mnemonic}: fmt_b={} (blgp={code}) and blgp={} disagree: give one",
                super::py_repr(fb),
                opts.blgp
            ));
        }
        opts.blgp = code;
    }
    if !mf.scaled && (opts.sel_a != 0 || opts.sel_b != 0) {
        return value(format!("{mnemonic}: does not take sel_a or sel_b"));
    }
    Ok(())
}

fn build_fields(
    t: &Target,
    mnemonic: &str,
    arity: usize,
    mf: &MatrixForm,
    opts: &MfmaOpts,
) -> Result<Vec<(String, FArg)>> {
    let instr_form = t
        .form(mnemonic, arity, false)
        .ok_or_else(|| crate::error::Error {
            class: crate::error::Class::Value,
            msg: format!("{mnemonic}: missing {arity}-operand form on {}", t.name()),
        })?;
    let has_field = |name: &str| instr_form.fields.iter().any(|f| f.name == name);
    let is_f8f6f4 = mf.a == Elem::F8F6F4;
    let mut fields = Vec::new();

    if mf.scaled {
        if !(0..=3).contains(&opts.sel_a) {
            return value(format!(
                "{mnemonic}: sel_a {} out of range (0..=3)",
                opts.sel_a
            ));
        }
        if !(0..=3).contains(&opts.sel_b) {
            return value(format!(
                "{mnemonic}: sel_b {} out of range (0..=3)",
                opts.sel_b
            ));
        }
        let lo0 = opts.sel_a & 1;
        let lo1 = opts.sel_b & 1;
        if lo0 != 0 || lo1 != 0 {
            fields.push((
                "op_sel".into(),
                FArg::List(vec![FArg::Int(lo0), FArg::Int(lo1), FArg::Int(0)]),
            ));
        }
        let hi0 = (opts.sel_a >> 1) & 1;
        let hi1 = (opts.sel_b >> 1) & 1;
        if hi0 != 0 || hi1 != 0 {
            fields.push((
                "op_sel_hi".into(),
                FArg::List(vec![FArg::Int(hi0), FArg::Int(hi1), FArg::Int(0)]),
            ));
        }
    }

    if is_f8f6f4 {
        if opts.abid != 0 {
            return value(format!("{mnemonic}: does not take abid"));
        }
        if !(0..=4).contains(&opts.cbsz) {
            return value(format!(
                "{mnemonic}: cbsz {} out of range (0..=4 for f8f6f4 format)",
                opts.cbsz
            ));
        }
        if !(0..=4).contains(&opts.blgp) {
            return value(format!(
                "{mnemonic}: blgp {} out of range (0..=4 for f8f6f4 format)",
                opts.blgp
            ));
        }
        if opts.cbsz != 0 {
            fields.push(("cbsz".into(), FArg::Int(opts.cbsz)));
        }
        if opts.blgp != 0 {
            fields.push(("blgp".into(), FArg::Int(opts.blgp)));
        }
    } else {
        if opts.cbsz != 0 || opts.abid != 0 {
            if !mf.cbsz_abid || !has_field("cbsz") {
                return value(format!("{mnemonic}: does not take cbsz or abid"));
            }
            if mf.sparse {
                let max_abid = i128::from(mf.sparse_sets().saturating_sub(1));
                if !(0..=3).contains(&opts.cbsz) {
                    return value(format!(
                        "{mnemonic}: cbsz {} out of range (0..=3)",
                        opts.cbsz
                    ));
                }
                if !(0..=max_abid).contains(&opts.abid) {
                    return value(format!(
                        "{mnemonic}: abid {} out of range (0..={max_abid})",
                        opts.abid
                    ));
                }
            } else {
                let max_cbsz = i128::from(mf.blocks.ilog2()).min(4);
                if !(0..=max_cbsz).contains(&opts.cbsz) {
                    return value(format!(
                        "{mnemonic}: cbsz {} out of range (0..={max_cbsz})",
                        opts.cbsz
                    ));
                }
                let max_abid = (1i128 << opts.cbsz) - 1;
                if !(0..=max_abid).contains(&opts.abid) {
                    return value(format!(
                        "{mnemonic}: abid {} out of range (0..={max_abid} for cbsz={})",
                        opts.abid, opts.cbsz
                    ));
                }
            }
            if opts.cbsz != 0 {
                fields.push(("cbsz".into(), FArg::Int(opts.cbsz)));
            }
            if opts.abid != 0 {
                fields.push(("abid".into(), FArg::Int(opts.abid)));
            }
        }

        if opts.blgp != 0 {
            if !mf.blgp || !has_field("blgp") {
                return value(format!("{mnemonic}: does not take blgp"));
            }
            if !(0..=7).contains(&opts.blgp) {
                return value(format!(
                    "{mnemonic}: blgp {} out of range (0..=7)",
                    opts.blgp
                ));
            }
            fields.push(("blgp".into(), FArg::Int(opts.blgp)));
        }
    }

    if let Some(neg) = &opts.neg {
        if !has_field("neg") {
            return value(format!("{mnemonic}: does not take neg:[a,b,c]"));
        }
        if neg.len() != 3 || neg.iter().any(|&x| x != 0 && x != 1) {
            return value(format!(
                "{mnemonic}: neg takes 3 flags (0/1 or bool) for [src0, src1, src2]"
            ));
        }
        if neg.iter().any(|&x| x != 0) {
            fields.push((
                "neg".into(),
                FArg::List(neg.iter().map(|&x| FArg::Int(x)).collect()),
            ));
        }
    }

    Ok(fields)
}

/// Makes an unscaled dense `v_mfma_*` instruction (`c_in = None` accumulates in-place into `dst`).
pub fn mfma(
    c: &Ctx<'_>,
    form: &str,
    dst: Arg,
    a: Arg,
    b: Arg,
    c_in: Option<Arg>,
    mut opts: MfmaOpts,
) -> Result<Made> {
    let (mnemonic, mf) = resolve_mfma(c.t, form)?;
    resolve_effective_opts(&mnemonic, &mf, &mut opts)?;
    let fields = build_fields(c.t, &mnemonic, 4, &mf, &opts)?;
    let a_dw = mf
        .operand_dwords('a', opts.cbsz as u32, opts.blgp as u32)
        .unwrap_or_else(|| mf.dwords('a'));
    let b_dw = mf
        .operand_dwords('b', opts.cbsz as u32, opts.blgp as u32)
        .unwrap_or_else(|| mf.dwords('b'));
    check_vec_reg(c, &mnemonic, "dst", &dst, mf.dwords('d'), false, false)?;
    check_vec_reg(c, &mnemonic, "src0 (A)", &a, a_dw, false, false)?;
    check_vec_reg(c, &mnemonic, "src1 (B)", &b, b_dw, false, false)?;
    let c_arg = c_in.unwrap_or_else(|| dst.clone());
    check_vec_reg(
        c,
        &mnemonic,
        "src2 (C)",
        &c_arg,
        mf.dwords('c'),
        true,
        false,
    )?;
    if let (Arg::Reg(rd, _), Arg::Reg(rc, _)) = (&dst, &c_arg) {
        if rd.file != rc.file {
            return value(format!(
                "{mnemonic}: dst {} ('{}') and src2 (C) {} ('{}') must use the same register file ('a' or 'v')",
                c.repr(rd),
                rd.file,
                c.repr(rc),
                rc.file
            ));
        }
    }
    op(c, &mnemonic, vec![dst, a, b, c_arg], fields, opts.notes)
}

/// Makes an MX-scaled `v_mfma_scale_*` instruction (`c_in = None` accumulates in-place into `dst`).
#[allow(clippy::too_many_arguments)]
pub fn mfma_scale(
    c: &Ctx<'_>,
    form: &str,
    dst: Arg,
    a: Arg,
    b: Arg,
    scale_a: Arg,
    scale_b: Arg,
    c_in: Option<Arg>,
    mut opts: MfmaOpts,
) -> Result<Made> {
    let (mnemonic, mf) = resolve_mfma_scale(c.t, form)?;
    resolve_effective_opts(&mnemonic, &mf, &mut opts)?;
    let fields = build_fields(c.t, &mnemonic, 6, &mf, &opts)?;
    let a_dw = mf
        .operand_dwords('a', opts.cbsz as u32, opts.blgp as u32)
        .unwrap_or_else(|| mf.dwords('a'));
    let b_dw = mf
        .operand_dwords('b', opts.cbsz as u32, opts.blgp as u32)
        .unwrap_or_else(|| mf.dwords('b'));
    check_vec_reg(c, &mnemonic, "dst", &dst, mf.dwords('d'), false, false)?;
    check_vec_reg(c, &mnemonic, "src0 (A)", &a, a_dw, false, false)?;
    check_vec_reg(c, &mnemonic, "src1 (B)", &b, b_dw, false, false)?;
    let c_arg = c_in.unwrap_or_else(|| dst.clone());
    check_vec_reg(
        c,
        &mnemonic,
        "src2 (C)",
        &c_arg,
        mf.dwords('c'),
        true,
        false,
    )?;
    if let (Arg::Reg(rd, _), Arg::Reg(rc, _)) = (&dst, &c_arg) {
        if rd.file != rc.file {
            return value(format!(
                "{mnemonic}: dst {} ('{}') and src2 (C) {} ('{}') must use the same register file ('a' or 'v')",
                c.repr(rd),
                rd.file,
                c.repr(rc),
                rc.file
            ));
        }
    }
    check_vec_reg(
        c,
        &mnemonic,
        "scale_src0 (scale_a)",
        &scale_a,
        mf.dwords('s'),
        false,
        true,
    )?;
    check_vec_reg(
        c,
        &mnemonic,
        "scale_src1 (scale_b)",
        &scale_b,
        mf.dwords('s'),
        false,
        true,
    )?;
    op(
        c,
        &mnemonic,
        vec![dst, a, b, c_arg, scale_a, scale_b],
        fields,
        opts.notes,
    )
}

/// Makes a sparse 4:2 `v_smfmac_*` instruction (`dst` is read and updated in-place).
pub fn smfmac(
    c: &Ctx<'_>,
    form: &str,
    dst: Arg,
    a: Arg,
    b: Arg,
    idx: Arg,
    mut opts: MfmaOpts,
) -> Result<Made> {
    let (mnemonic, mf) = resolve_smfmac(c.t, form)?;
    resolve_effective_opts(&mnemonic, &mf, &mut opts)?;
    check_vec_reg(
        c,
        &mnemonic,
        "dst (C/D)",
        &dst,
        mf.dwords('d'),
        false,
        false,
    )?;
    check_vec_reg(c, &mnemonic, "src0 (A)", &a, mf.dwords('a'), false, false)?;
    check_vec_reg(c, &mnemonic, "src1 (B)", &b, mf.dwords('b'), false, false)?;
    check_vec_reg(
        c,
        &mnemonic,
        "src2 (idx)",
        &idx,
        mf.dwords('k'),
        false,
        true,
    )?;
    let fields = build_fields(c.t, &mnemonic, 4, &mf, &opts)?;
    op(c, &mnemonic, vec![dst, a, b, idx], fields, opts.notes)
}
