//! The entry and the argument loads.

use std::collections::BTreeMap;

use crate::args::{Arg, Args};
use crate::builder::{Builder, Make};
use crate::error::{schedule, Result};
use crate::expr::node::{binop, constant, leaf, Leafish, Num, Op};
use crate::maker::families::{scalar, Opts, Val};
use crate::types::Ty;
use crate::values::Typed;

/// A loaded argument's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Loaded {
    /// A value of its type (a divisor carries its `mul` and `shr`).
    Value(Typed),
    /// A mapped or peers argument: per rank its pointer, then its own.
    Ranked(Vec<Typed>, Typed),
}

/// The kernel's entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The argument block.
    pub args: Args,
    /// The argument pointer.
    pub karg: Typed,
    /// The lane's id in the workgroup.
    pub tid: Typed,
    /// The workgroup id x.
    pub wgid: Typed,
    /// The workgroup id y, for wg = 2.
    pub wgid_y: Option<Typed>,
    /// The preloaded dwords.
    pub preload: u32,
    /// The workgroup id dimensions.
    pub wg: u32,
    /// The workgroup's lanes, if given.
    pub wgsize: Option<u32>,
    /// The entry moves: each workgroup-id copy's row and
    /// value, in order; `drop_entry` deletes those nothing reads.
    pub entry_moves: Vec<(u64, Typed)>,
    loaded: BTreeMap<String, (Loaded, Vec<u64>)>,
    lane: Option<Typed>,
    wave: Option<Typed>,
}

/// The lane or the wave: a value, or (wave) the int 0.
#[derive(Clone, Debug, PartialEq)]
pub enum Derived {
    /// A value.
    Value(Typed),
    /// A constant.
    Int(i128),
}

impl Builder {
    fn entry_mut(&mut self) -> &mut Entry {
        self.entered.as_mut().expect("an entry")
    }

    /// `b.entry(args, wg=, preload=, wgsize=)`.
    pub fn entry(&mut self, args: Args, wg: i128, preload: u32, wgsize: Option<u32>) -> Result<()> {
        if self.entered.is_some() {
            return schedule("entry: made once a kernel");
        }
        if !matches!(wg, 1 | 2) {
            return schedule(format!(
                "entry: wg={wg} - workgroup ids x (1) or x and y (2)"
            ));
        }
        let max = preload.min(self.t.definition().chip.preload);
        let mut sorted: Vec<&Arg> = args.args.iter().collect();
        sorted.sort_by_key(|a| a.offset);
        let (mut end, mut dwords) = (0, 0);
        let mut pre: Vec<(String, u32)> = Vec::new();
        for a in sorted {
            let d = (a.bytes / 4) as u32;
            if a.offset != end || dwords + d > max {
                break;
            }
            pre.push((format!("arg.{}", a.name), d));
            end += a.bytes;
            dwords += d;
        }
        let wgids = self.t.definition().chip.wgids;
        let mut groups: Vec<(String, u32)> = vec![("karg".into(), 2)];
        groups.extend(pre.iter().cloned());
        if wgids.is_empty() {
            groups.push(("wgid".into(), 1));
            if wg == 2 {
                groups.push(("wgid_y".into(), 1));
            }
        }
        let loc = self.loc;
        let refs: Vec<(&str, u32)> = groups.iter().map(|(n, w)| (n.as_str(), *w)).collect();
        let values = &mut self.values;
        self.files[1].entry(values, &refs, loc)?;
        let karg = self.typed_of('s', "karg", Ty::Ptr, Some("karg"));
        let mut ids = Vec::new();
        let mut moves = Vec::new();
        for (k, name) in ["wgid", "wgid_y"].iter().enumerate().take(wg as usize) {
            let v = match wgids.get(k) {
                Some(reg) => {
                    let m = Make {
                        n: Some(1),
                        n_repr: "1".into(),
                        name: Some((*name).into()),
                        where_: self.where_text.clone(),
                        ..Make::default()
                    };
                    let v = self.make_value('s', Ty::U32, m)?;
                    let phys = thin_target::phys_named(reg).expect("a workgroup id register");
                    let comment = format!("{}: {reg}", v.group);
                    let i = self.make_plain(
                        "s.mov.b32",
                        vec![
                            crate::maker::Arg::Reg(v.reg(), 0),
                            crate::maker::Arg::Phys(phys, 0),
                        ],
                        &comment,
                    )?;
                    if let Some(row) = self.emit(i, &[])? {
                        moves.push((row, v.clone()));
                    }
                    v
                }
                None => self.typed_of('s', name, Ty::U32, Some(name)),
            };
            ids.push(v);
        }
        let values = &mut self.values;
        self.files[0].entry(values, &[("tid", 1)], loc)?;
        let tid = self.typed_of('v', "tid", Ty::U32, Some("tid"));
        let mut it = ids.into_iter();
        let wgid = it.next().expect("wgid");
        self.entered = Some(Entry {
            args,
            karg,
            tid,
            wgid,
            wgid_y: it.next(),
            preload: dwords,
            wg: wg as u32,
            wgsize,
            entry_moves: moves,
            loaded: BTreeMap::new(),
            lane: None,
            wave: None,
        });
        Ok(())
    }

    fn outside(&self) -> bool {
        self.flow.arms.is_empty() && !self.exec_open()
    }

    /// `k.lane`.
    pub fn entry_lane(&mut self) -> Result<Typed> {
        let k = self.entered.as_ref().expect("an entry").clone();
        let wave = self.t.definition().wave;
        if k.wgsize == Some(wave) {
            return Ok(k.tid);
        }
        if let Some(l) = k.lane {
            return Ok(l);
        }
        let tree = binop(
            Op::And,
            leaf(Leafish::Value(k.tid.clone()))?,
            constant(Num::Int(i128::from(wave) - 1)),
        )?;
        let m = Make {
            n: Some(1),
            n_repr: "1".into(),
            name: Some("lane".into()),
            where_: self.where_text.clone(),
            ..Make::default()
        };
        let v = self.make_value('v', Ty::U32, m)?;
        self.assign(&v, tree, None)?;
        if self.outside() {
            self.entry_mut().lane = Some(v.clone());
        }
        Ok(v)
    }

    /// `k.wave`.
    pub fn entry_wave(&mut self) -> Result<Derived> {
        let k = self.entered.as_ref().expect("an entry").clone();
        let wave = self.t.definition().wave;
        if k.wgsize == Some(wave) {
            return Ok(Derived::Int(0));
        }
        if let Some(w) = k.wave {
            return Ok(Derived::Value(w));
        }
        let fl =
            crate::expr::funcs::lane(Op::FirstLane, vec![leaf(Leafish::Value(k.tid.clone()))?])?;
        let tree = binop(
            Op::Shr,
            fl,
            constant(Num::Int(i128::from(wave.trailing_zeros()))),
        )?;
        let m = Make {
            n: Some(1),
            n_repr: "1".into(),
            name: Some("wave".into()),
            where_: self.where_text.clone(),
            ..Make::default()
        };
        let v = self.make_value('s', Ty::U32, m)?;
        self.assign(&v, tree, None)?;
        if self.outside() {
            self.entry_mut().wave = Some(v.clone());
        }
        if self.wave_reg.is_none() {
            if let Some(size) = k.wgsize {
                self.wave_index(v.reg(), size / wave, false);
            }
        }
        Ok(Derived::Value(v))
    }

    /// `k.load(*names, comment)`.
    pub fn load(&mut self, names: &[String], comment: &str) -> Result<Vec<Loaded>> {
        let k = self.entered.as_ref().expect("an entry").clone();
        let all: Vec<String> = k.args.args.iter().map(|a| a.name.clone()).collect();
        if let Some(bad) = names.iter().find(|n| !all.contains(n)) {
            let block = if all.is_empty() {
                "the block is empty".to_owned()
            } else {
                format!("the block: {}", all.join(", "))
            };
            return schedule(format!(
                "load({}): no such argument ({block})",
                crate::maker::py_repr(bad)
            ));
        }
        let arms = self.arm_ids();
        let issue: Vec<String> = (if names.is_empty() {
            all.clone()
        } else {
            names.to_vec()
        })
        .into_iter()
        .filter(|n| !k.loaded.get(n).is_some_and(|(_, at)| arms.starts_with(at)))
        .collect();
        let mut args: Vec<Arg> = issue
            .iter()
            .filter_map(|n| k.args.get(n).cloned())
            .collect();
        args.sort_by_key(|a| a.offset);
        for chunk in chunks(&args) {
            self.load_chunk(&k, &chunk, comment, &arms)?;
        }
        let wanted = if names.is_empty() {
            k.args.own()
        } else {
            names.to_vec()
        };
        let k = self.entered.as_ref().expect("an entry");
        Ok(wanted.iter().map(|n| k.loaded[n].0.clone()).collect())
    }

    fn load_chunk(&mut self, k: &Entry, chunk: &[Arg], comment: &str, arms: &[u64]) -> Result<()> {
        let start = chunk[0].offset;
        let bytes = chunk.last().map_or(0, |a| a.offset + a.bytes) - start;
        let alone = chunk.len() == 1;
        let plain = alone && !chunk[0].divisor && chunk[0].ranks.is_none();
        let where_ = self.where_text.clone();
        let (whole, values): (Typed, Vec<(String, Loaded)>) = if plain {
            let a = &chunk[0];
            let ty = Ty::named(&a.kind).unwrap_or(Ty::U32);
            let m = Make {
                n: Some(1),
                n_repr: "1".into(),
                name: Some(a.name.clone()),
                where_,
                ..Make::default()
            };
            let v = self.make_value('s', ty, m)?;
            (v.clone(), vec![(a.name.clone(), Loaded::Value(v))])
        } else {
            let name = if alone {
                chunk[0].name.clone()
            } else {
                "args".into()
            };
            let n = (bytes / 4) as i64;
            let m = Make {
                n: Some(n),
                n_repr: n.to_string(),
                name: Some(name),
                where_,
                ..Make::default()
            };
            let v = self.make_value('s', Ty::U32, m)?;
            let vals = chunk
                .iter()
                .map(|a| (a.name.clone(), view(&v, a, start, alone)))
                .collect();
            (v, vals)
        };
        let names: Vec<&str> = chunk.iter().map(|a| a.name.as_str()).collect();
        let said = if comment.is_empty() {
            names.join(", ")
        } else {
            comment.to_owned()
        };
        let mut at = 0u64;
        while at < bytes {
            let piece = [64u64, 32, 16, 8, 4]
                .into_iter()
                .find(|&p| at + p <= bytes)
                .expect("a piece");
            let dst = match piece == bytes {
                true => whole.reg(),
                false => thin_ir::Reg::new(
                    whole.value as i32,
                    's',
                    whole.sub + (at / 4) as u32,
                    (piece / 4) as u32,
                ),
            };
            let o = Opts {
                offset: Val::Int(i128::from(start + at)),
                comment: said.clone(),
                ..Opts::default()
            };
            let c = crate::maker::Ctx {
                t: self.t,
                values: &self.values,
            };
            let f = scalar(&c, false, &dst, Val::Reg(k.karg.reg()), o)?;
            self.emit_family(f, &[])?;
            at += piece;
        }
        self.wait_arguments_on_use(&[whole.reg()], "kernel arguments");
        let e = self.entry_mut();
        for (n, v) in values {
            e.loaded.insert(n, (v, arms.to_vec()));
        }
        Ok(())
    }

    /// `k.load_rank(name, d, comment)`.
    pub fn load_rank(&mut self, name: &str, d: Val, comment: &str) -> Result<Typed> {
        let k = self.entered.as_ref().expect("an entry").clone();
        let a = k.args.get(name).filter(|a| a.ranks.is_some()).cloned();
        let Some(a) = a else {
            return schedule(format!(
                "load_rank({}): a peers or mapped argument's",
                crate::maker::py_repr(name)
            ));
        };
        let m = Make {
            n: Some(1),
            n_repr: "1".into(),
            name: Some(format!("{name}_d")),
            where_: self.where_text.clone(),
            ..Make::default()
        };
        let p = self.make_value('s', Ty::Ptr, m)?;
        let dterm = match d {
            Val::Int(v) => constant(Num::Int(v)),
            Val::Tree(t) => t,
            Val::Reg(r) => leaf(Leafish::Value(self.typed_reg_u32(r)))?,
            _ => constant(Num::Int(0)),
        };
        // The argument's offset plus 8 bytes a rank; an offset of 0 adds
        // nothing, so the comment does not show it.
        let scaled = binop(Op::Mul, dterm, constant(Num::Int(8)))?;
        let off = if a.offset == 0 {
            scaled
        } else {
            binop(Op::Add, constant(Num::Int(i128::from(a.offset))), scaled)?
        };
        let m = Make {
            n: Some(1),
            n_repr: "1".into(),
            name: Some("koff".into()),
            where_: self.where_text.clone(),
            ..Make::default()
        };
        let koff = self.make_value('s', Ty::U32, m)?;
        self.assign(&koff, off, None)?;
        let base = binop(
            Op::Add,
            leaf(Leafish::Value(k.karg.clone()))?,
            leaf(Leafish::Value(koff))?,
        )?;
        let said = if comment.is_empty() {
            name.to_owned()
        } else {
            comment.to_owned()
        };
        let c = crate::maker::Ctx {
            t: self.t,
            values: &self.values,
        };
        let o = Opts {
            comment: said,
            ..Opts::default()
        };
        let f = scalar(&c, false, &p.reg(), Val::Tree(base), o)?;
        self.emit_family(f, &[])?;
        self.wait_arguments_on_use(&[p.reg()], "kernel arguments");
        Ok(p)
    }

    /// A one-dword register as a u32 typed value.
    pub(crate) fn typed_reg_u32(&self, r: thin_ir::Reg) -> Typed {
        let v = &self.values[r.value as usize];
        Typed {
            builder: self.id,
            file: r.file,
            value: r.value as usize,
            group: v.name.clone(),
            ty: Ty::U32,
            n: r.n,
            sub: r.lo,
            label: Some(v.name.split('#').next().unwrap_or("").into()),
            where_: "".into(),
            part: "".into(),
            divisor: false,
        }
    }
}

/// An argument's view of its chunk's value.
fn view(v: &Typed, a: &Arg, start: u64, alone: bool) -> Loaded {
    let sub = v.sub + ((a.offset - start) / 4) as u32;
    let part = if alone {
        String::new()
    } else {
        format!(".{}", a.name)
    };
    let mk = |ty: Ty, sub: u32, part: String| Typed {
        ty,
        n: 1,
        sub,
        label: Some(a.name.as_str().into()),
        part: part.into(),
        divisor: false,
        ..v.clone()
    };
    if let Some(w) = a.ranks {
        let ranks = (0..w)
            .map(|q| mk(Ty::Ptr, sub + 2 * q, format!("{part}[{q}]")))
            .collect();
        return Loaded::Ranked(ranks, mk(Ty::Ptr, sub + 2 * w, format!("{part}.own")));
    }
    let ty = Ty::named(&a.kind).unwrap_or(Ty::U32);
    let mut t = mk(ty, sub, part);
    t.divisor = a.divisor;
    Loaded::Value(t)
}

/// The chunks of arguments in offset order.
fn chunks(args: &[Arg]) -> Vec<Vec<Arg>> {
    let mut out = Vec::new();
    let mut runs: Vec<Vec<Arg>> = Vec::new();
    for a in args {
        match runs.last_mut() {
            Some(run) if run.last().is_some_and(|l| l.offset + l.bytes == a.offset) => {
                run.push(a.clone())
            }
            _ => runs.push(vec![a.clone()]),
        }
    }
    for run in runs {
        let mut j = 0;
        while j < run.len() {
            let s = run[j].offset;
            let size = [64u64, 32, 16, 8, 4]
                .into_iter()
                .find(|&p| run[j..].iter().any(|a| a.offset + a.bytes == s + p))
                .unwrap_or(run[j].bytes);
            let k = run[j..]
                .iter()
                .position(|a| a.offset + a.bytes == s + size)
                .map_or(j, |p| j + p);
            out.push(run[j..=k].to_vec());
            j = k + 1;
        }
    }
    out
}
