//! High-level MFMA and SMFMAC instruction making and accumulator rules.

use thin_target::{Elem, MatrixForm, Target};

use super::{op, Arg, Ctx, FArg, Made, Notes};
use crate::error::{value, Result};
use crate::types::Ty;

/// Modifier options for a dense or sparse MFMA instruction.
#[derive(Clone, Debug, Default)]
pub struct MfmaOpts {
    /// Source-A broadcast block size (`cbsz`, 0..=4 dense or 0..=3 sparse).
    pub cbsz: i128,
    /// Source-A broadcast block / index-set selector (`abid`).
    pub abid: i128,
    /// Source-B lane-group broadcast/permutation (`blgp`, 0..=7).
    pub blgp: i128,
    /// Optional 3-element negation mask (`neg:[a, b, c]`) on binary64 MFMA forms.
    pub neg: Option<Vec<i128>>,
    /// Comment, tag, and ablation notes.
    pub notes: Notes,
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

/// Resolves any matrix form (dense MFMA, sparse SMFMAC, or WMMA) on `t` from a
/// short shape (`"f32_16x16x16_f16"`), a prefixed form (`"mfma_..."` /
/// `"smfmac_..."`), or a full/dotted mnemonic.
pub fn resolve_matrix_form(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    let candidates = [
        norm.clone(),
        format!("v_{norm}"),
        format!("v_mfma_{norm}"),
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

/// Resolves a dense `v_mfma_*` form on `t`.
pub fn resolve_mfma(t: &'static Target, form: &str) -> Result<(String, MatrixForm)> {
    let norm = form.trim().replace('.', "_");
    if norm.starts_with("smfmac_") || norm.starts_with("v_smfmac_") {
        return value(format!(
            "mfma({}): sparse 4:2 matrix op — use smfmac",
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
            if let Some(mf) = t.matrix_form(&spelled).filter(|m| !m.sparse) {
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

fn build_fields(
    t: &Target,
    mnemonic: &str,
    mf: &MatrixForm,
    opts: &MfmaOpts,
) -> Result<Vec<(String, FArg)>> {
    let instr_form = t
        .form(mnemonic, 4, false)
        .ok_or_else(|| crate::error::Error {
            class: crate::error::Class::Value,
            msg: format!("{mnemonic}: missing 4-operand form on {}", t.name()),
        })?;
    let has_field = |name: &str| instr_form.fields.iter().any(|f| f.name == name);
    let mut fields = Vec::new();

    if opts.cbsz != 0 || opts.abid != 0 {
        if !mf.cbsz_abid || !has_field("cbsz") {
            return value(format!("{mnemonic}: does not take cbsz or abid"));
        }
        if mf.sparse {
            let max_abid = i128::from(mf.a.bits() / 4 - 1);
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

/// Makes a dense `v_mfma_*` instruction (`c_in = None` accumulates in-place into `dst`).
pub fn mfma(
    c: &Ctx<'_>,
    form: &str,
    dst: Arg,
    a: Arg,
    b: Arg,
    c_in: Option<Arg>,
    opts: MfmaOpts,
) -> Result<Made> {
    let (mnemonic, mf) = resolve_mfma(c.t, form)?;
    check_vec_reg(c, &mnemonic, "dst", &dst, mf.dwords('d'), false, false)?;
    check_vec_reg(c, &mnemonic, "src0 (A)", &a, mf.dwords('a'), false, false)?;
    check_vec_reg(c, &mnemonic, "src1 (B)", &b, mf.dwords('b'), false, false)?;
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
    let fields = build_fields(c.t, &mnemonic, &mf, &opts)?;
    op(c, &mnemonic, vec![dst, a, b, c_arg], fields, opts.notes)
}

/// Makes a sparse 4:2 `v_smfmac_*` instruction (`dst` is read and updated in-place).
pub fn smfmac(
    c: &Ctx<'_>,
    form: &str,
    dst: Arg,
    a: Arg,
    b: Arg,
    idx: Arg,
    opts: MfmaOpts,
) -> Result<Made> {
    let (mnemonic, mf) = resolve_smfmac(c.t, form)?;
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
    let fields = build_fields(c.t, &mnemonic, &mf, &opts)?;
    op(c, &mnemonic, vec![dst, a, b, idx], fields, opts.notes)
}
