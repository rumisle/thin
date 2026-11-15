//! High-level Builder methods for matrix accumulators, MFMA/SMFMAC emission,
//! and AGPR/VGPR transfers.

use thin_ir::Reg;

use super::{Builder, Make};
use crate::error::{schedule, value, Result};
use crate::expr::node::{leaf, Leafish, Tree};
use crate::maker::matrix::{
    acc_shape, choose_acc_file, mfma, resolve_matrix_form, resolve_mfma, resolve_smfmac, smfmac,
    MfmaOpts,
};
use crate::maker::{Arg, Ctx, Made};
use crate::types::Ty;
use crate::values::Typed;

/// An accumulator argument from Python: either a `TypedValue` or a raw `VReg`.
#[derive(Clone, Debug)]
pub enum AccIn {
    /// A `TypedValue`.
    Typed(Typed),
    /// A raw `VReg` `(value, file, lo, n)`.
    Reg(Reg),
}

impl Builder {
    /// Wraps a `Reg` of this Builder as a `Typed` with element type `ty`.
    pub fn typed_of_reg(&self, r: Reg, ty: Ty) -> Result<Typed> {
        let v = usize::try_from(r.value)
            .ok()
            .filter(|&v| v < self.values.len() && self.values[v].file == r.file)
            .ok_or_else(|| crate::error::Error {
                class: crate::error::Class::Schedule,
                msg: format!("{}: not a value of this Builder", self.repr(&r)),
            })?;
        if r.lo + r.n > self.values[v].n {
            return value(format!(
                "{}: sub {} + width {} exceeds declared width {}",
                self.repr(&r),
                r.lo,
                r.n,
                self.values[v].n
            ));
        }
        let elem_dw = ty.dwords().max(1);
        let (ty, n) = if r.n % elem_dw == 0 {
            (ty, r.n / elem_dw)
        } else {
            (Ty::U32, r.n)
        };
        let part = match (r.lo, r.n == self.values[v].n) {
            (0, true) => String::new(),
            (lo, _) if n == 1 => format!("[{}]", lo / elem_dw),
            (lo, _) => format!("[{}:{}]", lo / elem_dw, lo / elem_dw + n),
        };
        Ok(Typed {
            builder: self.id,
            file: r.file,
            value: v,
            group: self.values[v].name.clone(),
            ty,
            n,
            sub: r.lo,
            label: None,
            where_: self.where_text.as_str().into(),
            part: part.into(),
            divisor: false,
        })
    }

    fn resolve_acc_in(&self, x: AccIn, default_ty: Ty) -> Result<Typed> {
        match x {
            AccIn::Typed(t) => {
                if t.builder != self.id {
                    return schedule(format!("{}: not a value of this Builder", t.repr()));
                }
                Ok(t)
            }
            AccIn::Reg(r) => self.typed_of_reg(r, default_ty),
        }
    }

    fn copy_typed(&mut self, dst: &Typed, src: &Typed) -> Result<()> {
        if dst.width() != src.width() {
            return value(format!(
                "{} ({} dwords) and {} ({} dwords): widths differ",
                dst.repr(),
                dst.width(),
                src.repr(),
                src.width()
            ));
        }
        let src_view = if src.ty == dst.ty && src.n == dst.n {
            src.clone()
        } else {
            src.retyped(dst.ty)
        };
        for j in 0..dst.n {
            let d_e = if dst.n == 1 {
                dst.clone()
            } else {
                dst.element(i64::from(j))?
            };
            let s_e = if src_view.n == 1 {
                src_view.clone()
            } else {
                src_view.element(i64::from(j))?
            };
            let tree = leaf(Leafish::Value(s_e))?;
            self.assign(&d_e, tree, None)?;
        }
        Ok(())
    }

    /// Allocates (and optionally initializes) a matrix accumulator for `form`
    /// in `'a'` (AGPR) or `'v'` (VGPR).
    pub fn mfma_acc(
        &mut self,
        form: &str,
        acc: Option<&str>,
        agpr: Option<bool>,
        mut m: Make,
        init: Option<Tree>,
    ) -> Result<Typed> {
        let (_, mf) = resolve_matrix_form(self.t, form)?;
        let file = choose_acc_file(self.t, acc, agpr)?;
        let (ty, count) = acc_shape(&mf);
        m.n = Some(i64::from(count));
        m.n_repr = count.to_string();
        m.where_ = self.where_text.clone();
        self.make_init(file, ty, m, init)
    }

    /// Emits a dense `v_mfma_*` instruction and returns the destination `Typed`
    /// accumulator (`'a'` or `'v'`).
    #[allow(clippy::too_many_arguments)]
    pub fn mfma_emit(
        &mut self,
        form: &str,
        a: Arg,
        b: Arg,
        c_arg: Option<Arg>,
        c_acc: Option<AccIn>,
        dst: Option<AccIn>,
        acc: Option<&str>,
        agpr: Option<bool>,
        name: Option<String>,
        opts: MfmaOpts,
        abl: &[String],
    ) -> Result<Typed> {
        let (mnemonic, mf) = resolve_mfma(self.t, form)?;
        let (ty, count) = acc_shape(&mf);
        let c_typed = c_acc.map(|x| self.resolve_acc_in(x, ty)).transpose()?;
        let dst_typed = dst.map(|x| self.resolve_acc_in(x, ty)).transpose()?;

        let (out, c_final) = if let Some(d) = dst_typed {
            if acc.is_some() || agpr.is_some() {
                let expected = choose_acc_file(self.t, acc, agpr)?;
                if d.file != expected {
                    return value(format!(
                        "{mnemonic}: dst {} is in '{}', but acc='{}' was requested",
                        d.repr(),
                        d.file,
                        expected
                    ));
                }
            }
            let cf = c_arg.unwrap_or_else(|| Arg::Reg(d.reg(), 0));
            (d, cf)
        } else if let Some(ct) =
            c_typed.filter(|_| acc.is_none() && agpr.is_none() && name.is_none())
        {
            let reg = ct.reg();
            (ct, Arg::Reg(reg, 0))
        } else {
            let file = choose_acc_file(self.t, acc, agpr)?;
            let d = self.make_value(
                file,
                ty,
                Make {
                    n: Some(i64::from(count)),
                    n_repr: count.to_string(),
                    name,
                    where_: self.where_text.clone(),
                    ..Make::default()
                },
            )?;
            let cf = c_arg.unwrap_or(Arg::Int(0));
            (d, cf)
        };

        let ins = {
            let ctx = Ctx {
                t: self.t,
                values: &self.values,
            };
            match mfma(
                &ctx,
                form,
                Arg::Reg(out.reg(), 0),
                a,
                b,
                Some(c_final),
                opts,
            )? {
                Made::Instr(i) => *i,
                Made::Lds(_) => unreachable!(),
            }
        };
        self.emit(ins, abl)?;
        Ok(out)
    }

    /// Emits a sparse 4:2 `v_smfmac_*` instruction in-place on `c` and returns `c`.
    #[allow(clippy::too_many_arguments)]
    pub fn smfmac_emit(
        &mut self,
        form: &str,
        a: Arg,
        b: Arg,
        idx: Arg,
        c: AccIn,
        acc: Option<&str>,
        agpr: Option<bool>,
        opts: MfmaOpts,
        abl: &[String],
    ) -> Result<Typed> {
        let (mnemonic, mf) = resolve_smfmac(self.t, form)?;
        let (ty, _) = acc_shape(&mf);
        let out = self.resolve_acc_in(c, ty)?;
        if acc.is_some() || agpr.is_some() {
            let expected = choose_acc_file(self.t, acc, agpr)?;
            if out.file != expected {
                return value(format!(
                    "{mnemonic}: accumulator {} is in '{}', but acc='{}' was requested",
                    out.repr(),
                    out.file,
                    expected
                ));
            }
        }
        let ins = {
            let ctx = Ctx {
                t: self.t,
                values: &self.values,
            };
            match smfmac(&ctx, form, Arg::Reg(out.reg(), 0), a, b, idx, opts)? {
                Made::Instr(i) => *i,
                Made::Lds(_) => unreachable!(),
            }
        };
        self.emit(ins, abl)?;
        Ok(out)
    }

    /// Returns a `'v'` (VGPR) `Typed` holding `src`'s dwords, emitting
    /// `v_accvgpr_read_b32` when `src` is in `'a'`, or returning `src` directly
    /// when it is already in `'v'` (unless `copy` or `dst`/`name` is given).
    pub fn read_acc(
        &mut self,
        src: AccIn,
        dst: Option<AccIn>,
        copy: bool,
        name: Option<String>,
    ) -> Result<Typed> {
        let s = self.resolve_acc_in(src, Ty::F32)?;
        if !matches!(s.file, 'a' | 'v') {
            return value(format!(
                "read_acc({}): expected an 'a' (AGPR) or 'v' (VGPR) accumulator",
                s.repr()
            ));
        }
        if let Some(d_in) = dst {
            let d = self.resolve_acc_in(d_in, s.ty)?;
            if d.file != 'v' {
                return value(format!(
                    "read_acc(..., dst={}): dst must be in 'v' (VGPR)",
                    d.repr()
                ));
            }
            self.copy_typed(&d, &s)?;
            return Ok(d);
        }
        if s.file == 'v' && !copy && name.is_none() {
            return Ok(s);
        }
        let d = self.make_value(
            'v',
            s.ty,
            Make {
                n: Some(i64::from(s.n)),
                n_repr: s.n.to_string(),
                name,
                where_: self.where_text.clone(),
                ..Make::default()
            },
        )?;
        self.copy_typed(&d, &s)?;
        Ok(d)
    }

    /// Writes `src` into an `'a'` or `'v'` accumulator (`v_accvgpr_write_b32` /
    /// `v_accvgpr_mov_b32` / `v_mov_b32`), returning `src` unchanged when it is
    /// already in the chosen file (unless `copy` or `dst`/`name` is given).
    pub fn write_acc(
        &mut self,
        src: AccIn,
        dst: Option<AccIn>,
        acc: Option<&str>,
        agpr: Option<bool>,
        copy: bool,
        name: Option<String>,
    ) -> Result<Typed> {
        let s = self.resolve_acc_in(src, Ty::F32)?;
        if let Some(d_in) = dst {
            let d = self.resolve_acc_in(d_in, s.ty)?;
            if !matches!(d.file, 'a' | 'v') {
                return value(format!(
                    "write_acc(..., dst={}): dst must be in 'a' (AGPR) or 'v' (VGPR)",
                    d.repr()
                ));
            }
            if acc.is_some() || agpr.is_some() {
                let expected = choose_acc_file(self.t, acc, agpr)?;
                if d.file != expected {
                    return value(format!(
                        "write_acc: dst {} is in '{}', but acc='{}' was requested",
                        d.repr(),
                        d.file,
                        expected
                    ));
                }
            }
            self.copy_typed(&d, &s)?;
            return Ok(d);
        }
        let file = choose_acc_file(self.t, acc, agpr)?;
        if s.file == file && !copy && name.is_none() {
            return Ok(s);
        }
        let d = self.make_value(
            file,
            s.ty,
            Make {
                n: Some(i64::from(s.n)),
                n_repr: s.n.to_string(),
                name,
                where_: self.where_text.clone(),
                ..Make::default()
            },
        )?;
        self.copy_typed(&d, &s)?;
        Ok(d)
    }
}
