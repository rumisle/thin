//! The encoding moves: VOP and SOP.

use super::lower::Opd;
use crate::builder::Builder;
use crate::error::Result;
use crate::maker::FArg;
use crate::types::Ty;
use crate::values::Typed;

/// VOP's options beside its operands.
#[derive(Clone, Debug, Default)]
pub struct Vop<'a> {
    /// The reversed op (`v.subrev.u32` for `v.sub.u32`).
    pub rev: Option<&'a str>,
    /// Whether the sources commute.
    pub commute: bool,
    /// Whether the op reads VCC (one scalar less on the bus).
    pub vcc_read: bool,
    /// Its fields.
    pub fields: Vec<(String, FArg)>,
}

/// A source's constant-bus key.
#[derive(Clone, Debug, PartialEq)]
enum Key {
    Sgpr(u64, usize, u32),
    Phys(&'static str),
    Lit(i128),
}

fn key(x: &Opd) -> Option<Key> {
    match x {
        Opd::Reg(v, _) if v.file == 's' => Some(Key::Sgpr(v.builder, v.value, v.sub)),
        Opd::Phys(p) => Some(Key::Phys(p.name)),
        Opd::Int(c) if x.literal() => Some(Key::Lit(*c)),
        _ => None,
    }
}

impl Builder {
    /// VOP(op, dst, srcs, t; rev, commute, vcc_read, fields).
    pub fn vop(&mut self, op: &str, dst: &Typed, srcs: Vec<Opd>, t: Ty, o: Vop<'_>) -> Result<()> {
        self.vop_dst(op, Opd::Reg(dst.clone(), 0), srcs, t, o)
    }

    /// VOP into a named register.
    pub fn vop_phys(
        &mut self,
        op: &str,
        dst: &'static thin_target::Phys,
        srcs: Vec<Opd>,
        t: Ty,
        o: Vop<'_>,
    ) -> Result<()> {
        self.vop_dst(op, Opd::Phys(dst), srcs, t, o)
    }

    fn vop_dst(&mut self, op: &str, dst: Opd, srcs: Vec<Opd>, t: Ty, o: Vop<'_>) -> Result<()> {
        let mut name = self.spelled(op)?;
        let encodable = |b: &Builder, n: &str| b.t.encodable(&format!("{n}_e32"));
        let mut short =
            encodable(self, &name) && !srcs.iter().any(Opd::modified) && o.fields.is_empty();
        // The target's constant bus: each distinct scalar register, named
        // register or literal takes a place (VCC an implicit one when the op
        // reads it); one literal value at most. The rest move into VGPRs.
        let operands = self.t.definition().operands;
        let keep = operands.constant_bus.saturating_sub(u32::from(o.vcc_read)) as usize;
        let mut seen: Vec<Key> = Vec::new();
        let mut srcs = srcs;
        for s in srcs.iter_mut() {
            let Some(key) = key(s) else { continue };
            if seen.contains(&key) {
                continue;
            }
            let second_literal =
                matches!(key, Key::Lit(_)) && seen.iter().any(|k| matches!(k, Key::Lit(_)));
            if seen.len() < keep && !second_literal {
                seen.push(key);
                continue;
            }
            *s = self.vgpr_copy_keep(s, t)?;
        }
        if short && srcs.len() == 2 && !srcs[1].vgpr() {
            let (a, b) = (&srcs[0], &srcs[1]);
            if (o.commute || o.rev.is_some()) && (a.vgpr() || b.literal() && !a.literal()) {
                srcs.swap(0, 1);
                if !o.commute {
                    name = self.spelled(o.rev.expect("a reversed op"))?;
                }
            }
            // VOP2's second source is a VGPR; with none to put there, the
            // VOP3 form takes both as they are when the target allows it.
            if !srcs[1].vgpr() {
                let long = operands.vop3_literal && self.t.encodable(&format!("{name}_e64"));
                if srcs[0].vgpr() || long {
                    short = false;
                } else {
                    srcs[1] = self.vgpr_copy_keep(&srcs[1], t)?;
                }
            }
        }
        if !short && encodable(self, &name) {
            name = format!("{name}_e64");
        }
        if !short && !operands.vop3_literal {
            for k in 0..srcs.len() {
                if !srcs[k].literal() {
                    continue;
                }
                let any_sgpr = srcs.iter().any(Opd::sgpr);
                srcs[k] = match !any_sgpr && !o.vcc_read {
                    true => self.sgpr_copy(&srcs[k], t)?,
                    false => self.vgpr_copy(&srcs[k], t)?,
                };
            }
        }
        let mut all = vec![dst];
        all.extend(srcs);
        self.emit_name(&name, &all, &o.fields)
    }

    /// SOP(op, dst, srcs): every literal source after the first moved into
    /// an SGPR temporary.
    pub fn sop(&mut self, op: &str, dst: &Typed, srcs: Vec<Opd>) -> Result<()> {
        let mut srcs = srcs;
        let mut first = true;
        for s in srcs.iter_mut() {
            if !s.literal() {
                continue;
            }
            if first {
                first = false;
                continue;
            }
            *s = self.sgpr_copy(s, Ty::U32)?;
        }
        self.emit_op(op, Some(dst), &srcs, &[])
    }
}
