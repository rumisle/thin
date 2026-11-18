//! The memory families and buffer-offset staging.

use thin_ir::{Datum, Instr, Reg, Tag};

use super::{op, Arg, Ctx, FArg, Made, Notes, Stage};
use crate::error::{value, Result};
use crate::expr::node::Tree;
use crate::lds::{Address, Dma, DmaKind, Split, SplitKind};

/// The staging placeholder's value number.
pub const LIT: i32 = -1;

/// The staging placeholder register.
pub fn lit() -> Reg {
    Reg::new(LIT, 's', 0, 1)
}

/// A family's operand: a register, an int, a formula, an LDS address,
/// none, or a string.
#[derive(Clone, Debug)]
pub enum Val {
    /// None.
    None,
    /// A register.
    Reg(Reg),
    /// An int.
    Int(i128),
    /// A formula.
    Tree(Tree),
    /// An LDS address.
    Addr(Address),
    /// A string (`"off"`).
    Str(String),
}

impl Val {
    pub(crate) fn arg(&self) -> Arg {
        match self {
            Val::None => Arg::None,
            Val::Reg(r) => Arg::Reg(*r, 0),
            Val::Int(v) => Arg::Int(*v),
            Val::Str(s) => Arg::Str(s.clone()),
            Val::Addr(a) => Arg::Addr(a.clone()),
            Val::Tree(_) => Arg::None,
        }
    }
}

/// What a family made.
#[derive(Clone, Debug)]
pub enum Family {
    /// An instruction, or an LDS access.
    Made(Made),
    /// A split object.
    Split(Box<Split>),
    /// A VMEM-to-LDS DMA object.
    Dma(Box<Dma>),
}

/// A memory op's cache fields: its temporal hint (`th`) and scope, each
/// one of the target's choices, or empty for the default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cache {
    /// The temporal hint, `TH_*`.
    pub th: String,
    /// The scope, `SCOPE_*`.
    pub scope: String,
    /// Boolean cache flags (`sc0`, `sc1`, `nt`, `glc`).
    pub flags: Vec<String>,
}

impl Cache {
    /// The fields it sets.
    pub fn fields(&self) -> Vec<(String, FArg)> {
        let mut out: Vec<(String, FArg)> = [("th", &self.th), ("scope", &self.scope)]
            .into_iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k.to_owned(), FArg::Str(v.clone())))
            .collect();
        out.extend(self.flags.iter().map(|f| (f.clone(), FArg::Bool(true))));
        out
    }
}

/// A family's options.
#[derive(Clone, Debug, Default)]
pub struct Opts {
    /// The immediate offset.
    pub offset: Val,
    /// The cache fields.
    pub cache: Cache,
    /// The comment.
    pub comment: String,
    /// The tag.
    pub tag: Option<Tag>,
    /// `bytes=`: 1, 2, none, or another value by its repr.
    pub bytes: Option<std::result::Result<i128, String>>,
    /// `signed=`.
    pub signed: bool,
    /// The ordered reason (LDS).
    pub ordered: String,
}

impl Default for Val {
    fn default() -> Val {
        Val::Int(0)
    }
}

fn reg_of(v: &Val) -> Option<Reg> {
    if let Val::Reg(r) = v {
        Some(*r)
    } else {
        None
    }
}

/// The width word of a family's data.
fn width(c: &Ctx<'_>, fam: &str, data: &Reg, o: &Opts, load: bool) -> Result<String> {
    let repr = crate::values::reg_repr(c.values, data);
    match &o.bytes {
        Some(Ok(b @ (1 | 2))) => {
            if data.n != 1 {
                return value(format!(
                    "{fam}: bytes={b}: one dword's low bytes, not {repr} ({} dwords)",
                    data.n
                ));
            }
            let bits = b * 8;
            Ok(match (load, o.signed) {
                (true, true) => format!("i{bits}"),
                (true, false) => format!("u{bits}"),
                (false, _) => format!("b{bits}"),
            })
        }
        Some(Ok(b)) => value(format!(
            "{fam}: bytes={b}: 1 or 2 (a whole dword or more: no bytes=)"
        )),
        Some(Err(r)) => value(format!(
            "{fam}: bytes={r}: 1 or 2 (a whole dword or more: no bytes=)"
        )),
        None if o.signed => value(format!(
            "{fam}: signed= is a 1- or 2-byte load's (bytes=1 or 2)"
        )),
        None => match data.n {
            1 => Ok("b32".into()),
            2 => Ok("b64".into()),
            3 => Ok("b96".into()),
            4 => Ok("b128".into()),
            n => value(format!(
                "{fam}: {repr} is {n} dword(s); the forms take 1, 2, 3, 4"
            )),
        },
    }
}

fn notes(o: &Opts) -> Notes {
    Notes {
        comment: o.comment.clone(),
        tag: o.tag.clone(),
        ..Notes::default()
    }
}

fn offset_field(o: &Opts) -> Vec<(String, FArg)> {
    match &o.offset {
        Val::Int(0) | Val::None => Vec::new(),
        Val::Int(v) => vec![("offset".to_owned(), FArg::Int(*v))],
        _ => Vec::new(),
    }
}

/// `s_load(dst, base, offset)` and `s_store(data, base, offset)`.
pub fn scalar(c: &Ctx<'_>, store: bool, data: &Reg, base: Val, o: Opts) -> Result<Family> {
    let fam = if store { "s_store" } else { "s_load" };
    let widths: &[(u32, &str)] = match store {
        false => &[
            (1, "b32"),
            (2, "b64"),
            (4, "b128"),
            (8, "b256"),
            (16, "b512"),
        ],
        true => &[(1, "b32"), (2, "b64"), (4, "b128")],
    };
    let Some((_, w)) = widths.iter().find(|(n, _)| *n == data.n) else {
        let takes: Vec<String> = widths.iter().map(|(n, _)| n.to_string()).collect();
        let repr = crate::values::reg_repr(c.values, data);
        return value(format!(
            "{fam}: {repr} is {} dword(s); the forms take {}",
            data.n,
            takes.join(", ")
        ));
    };
    let name = format!("s.{}.{w}", if store { "store" } else { "load" });
    if matches!(base, Val::Tree(_)) || matches!(o.offset, Val::Tree(_)) {
        return Ok(Family::Split(Box::new(Split::new(
            SplitKind::Scalar,
            name,
            *data,
            base,
            o.offset.clone(),
            o,
        ))));
    }
    let off = match &o.offset {
        Val::Int(v) => Arg::Str(format!("0x{v:x}")),
        v => v.arg(),
    };
    let fields = o.cache.fields();
    op(
        c,
        &name,
        vec![Arg::Reg(*data, 0), base.arg(), off],
        fields,
        notes(&o),
    )
    .map(Family::Made)
}

/// `global_load(dst, addr, base)` and `global_store(addr, data, base)`.
pub fn global(
    c: &Ctx<'_>,
    store: bool,
    data: &Reg,
    addr: Val,
    base: Val,
    o: Opts,
) -> Result<Family> {
    let fam = if store { "global_store" } else { "global_load" };
    let w = width(c, fam, data, &o, !store)?;
    let name = format!("global.{}.{w}", if store { "store" } else { "load" });
    // A bare scalar pointer is the base with a zero offset, split as a sum.
    let pointer = matches!(&addr, Val::Reg(r) if r.file == 's' && r.n == 2);
    let split = matches!(addr, Val::Tree(_))
        || pointer && matches!(base, Val::None)
        || !matches!(base, Val::None) && matches!(addr, Val::Int(_));
    if split {
        let s = Split::new(
            SplitKind::Global {
                store,
                base: reg_of(&base),
            },
            name,
            *data,
            addr,
            o.offset.clone(),
            o,
        );
        return Ok(Family::Split(Box::new(s)));
    }
    let base = match base {
        Val::None => Arg::Str("off".into()),
        b => b.arg(),
    };
    let args = match store {
        true => vec![addr.arg(), Arg::Reg(*data, 0), base],
        false => vec![Arg::Reg(*data, 0), addr.arg(), base],
    };
    let mut fields = offset_field(&o);
    fields.extend(o.cache.fields());
    op(c, &name, args, fields, notes(&o)).map(Family::Made)
}

/// `buffer_load(dst, voffset, desc, soffset)` and `buffer_store(…)`.
pub fn buffer(
    c: &Ctx<'_>,
    store: bool,
    data: &Reg,
    voffset: Val,
    desc: Val,
    soffset: Val,
    o: Opts,
) -> Result<Family> {
    let fam = if store { "buffer_store" } else { "buffer_load" };
    let w = width(c, fam, data, &o, !store)?;
    let name = format!("buffer.{}.{w}", if store { "store" } else { "load" });
    if matches!(voffset, Val::Tree(_) | Val::Int(_)) {
        let kind = SplitKind::Buffer {
            store,
            desc: reg_of(&desc).expect("a resource"),
            soffset: soffset.clone(),
        };
        return Ok(Family::Split(Box::new(Split::new(
            kind,
            name,
            *data,
            voffset,
            o.offset.clone(),
            o,
        ))));
    }
    buffer_op(c, &name, store, data, voffset, desc, soffset, o).map(Family::Made)
}

/// Width word for a VMEM-to-LDS DMA (`1`, `2`, `4`, `12`, or `16` bytes per lane).
fn dma_width(fam: &str, o: &Opts) -> Result<String> {
    match &o.bytes {
        Some(Ok(b @ (1 | 2))) => {
            let bits = b * 8;
            Ok(if o.signed {
                format!("i{bits}")
            } else {
                format!("u{bits}")
            })
        }
        Some(Ok(b @ (4 | 12 | 16))) => {
            if o.signed {
                return value(format!(
                    "{fam}: signed= is a 1- or 2-byte load's (bytes=1 or 2)"
                ));
            }
            Ok(format!("b{}", b * 8))
        }
        Some(Ok(b)) => value(format!("{fam}: bytes={b}: 1, 2, 4, 12, or 16")),
        Some(Err(r)) => value(format!("{fam}: bytes={r}: 1, 2, 4, 12, or 16")),
        None if o.signed => value(format!(
            "{fam}: signed= is a 1- or 2-byte load's (bytes=1 or 2)"
        )),
        None => Ok("b32".into()),
    }
}

/// `global_load(lds, addr, base=...)` into an LDS destination address.
fn global_dma(c: &Ctx<'_>, lds: Address, addr: Val, base: Val, o: Opts) -> Result<Family> {
    let w = dma_width("global_load", &o)?;
    let op_name = format!("global.load.lds.{w}");
    if c.t.spell(&op_name).is_none() {
        return value(format!(
            "global_load into LDS ({op_name}): {} has no global-to-LDS DMA",
            c.t.name()
        ));
    }
    let offset = o.offset.clone();
    Ok(Family::Dma(Box::new(Dma::new(
        DmaKind::Global { base },
        op_name,
        lds,
        addr,
        offset,
        o,
    ))))
}

/// `buffer_load(lds, voffset, desc, soffset=...)` into an LDS destination address.
fn buffer_dma(
    c: &Ctx<'_>,
    lds: Address,
    voffset: Val,
    desc: Val,
    soffset: Val,
    o: Opts,
) -> Result<Family> {
    let w = dma_width("buffer_load", &o)?;
    let op_name = format!("buffer.load.{w}");
    let supported =
        c.t.spell(&op_name)
            .and_then(|m| c.t.form(&m, 3, false))
            .is_some();
    if !supported {
        return value(format!(
            "buffer_load into LDS ({op_name}): {} has no buffer-to-LDS DMA",
            c.t.name()
        ));
    }
    if !matches!(&desc, Val::Reg(r) if r.file == 's' && r.n == 4) {
        return value("buffer_load: descriptor must be a 4-SGPR value");
    }
    let offset = o.offset.clone();
    Ok(Family::Dma(Box::new(Dma::new(
        DmaKind::Buffer { desc, soffset },
        op_name,
        lds,
        voffset,
        offset,
        o,
    ))))
}

/// A memory family by name: `s_load`, `s_store`, `global_load`,
/// `global_store`, `buffer_load`, `buffer_store`, `ds_read`, `ds_read_tr`,
/// `ds_write`; `xs` its operands after the data in the table's order (a
/// scalar's and an LDS access's offset last).
pub fn family(c: &Ctx<'_>, name: &str, data: Val, xs: Vec<Val>, o: Opts) -> Result<Family> {
    let store = name.ends_with("store") || name == "ds_write";
    let at = |k: usize| xs.get(k).cloned().unwrap_or(Val::None);
    match data {
        Val::Addr(lds) => match name {
            "global_load" => global_dma(c, lds, at(0), at(1), o),
            "buffer_load" => buffer_dma(c, lds, at(0), at(1), at(2), o),
            _ => value(format!(
                "{name}: destination/data is an LDS address, which only buffer_load and global_load DMA into"
            )),
        },
        Val::Reg(ref data) => match name {
            "s_load" | "s_store" => scalar(c, store, data, at(0), Opts { offset: at(1), ..o }),
            "global_load" | "global_store" => global(c, store, data, at(0), at(1), o),
            "buffer_load" | "buffer_store" => buffer(c, store, data, at(0), at(1), at(2), o),
            "ds_read_tr" => ds_read_tr(c, data, at(0), at(1), Opts { offset: at(2), ..o }),
            _ => ds(c, store, data, at(0), Opts { offset: at(1), ..o }),
        },
        _ => value(format!("{name}: expected a register or an LDS address")),
    }
}

/// The 3-operand buffer-to-LDS DMA instruction with its offset staging.
pub fn buffer_dma_op(
    c: &Ctx<'_>,
    name: &str,
    voffset: Val,
    desc: Val,
    soffset: Val,
    o: Opts,
) -> Result<Made> {
    let offen = !matches!(voffset, Val::None);
    let off = match &o.offset {
        Val::Int(v) => *v,
        _ => 0,
    };
    let mut n = notes(&o);
    let mut soff = soffset.arg();
    let mut fields: Vec<(String, FArg)> = Vec::new();
    if offen {
        fields.push(("offen".into(), FArg::Bool(true)));
    }
    let range = c.t.offset_range(thin_target::Mem::Buffer).unwrap_or(0..=0);
    if range.contains(&off) {
        if off != 0 {
            fields.push(("offset".into(), FArg::Int(off)));
        }
    } else {
        n.stage = Some(match &soffset {
            Val::Reg(r) => Stage::Add(*r, off),
            Val::Int(s) => Stage::Mov(off + s),
            _ => {
                return value(format!(
                    "{name}: offset {off} past {}: soffset must be a register or a constant",
                    range.end()
                ))
            }
        });
        soff = Arg::Reg(lit(), 0);
    }
    fields.extend(o.cache.fields());
    fields.push(("lds".into(), FArg::Bool(true)));
    let vo = match voffset {
        Val::None => Arg::Str("off".into()),
        v => v.arg(),
    };
    op(c, name, vec![vo, desc.arg(), soff], fields, n)
}

/// The buffer instruction with its offset staging.
#[allow(clippy::too_many_arguments)]
pub fn buffer_op(
    c: &Ctx<'_>,
    name: &str,
    store: bool,
    data: &Reg,
    voffset: Val,
    desc: Val,
    soffset: Val,
    o: Opts,
) -> Result<Made> {
    let offen = !matches!(voffset, Val::None);
    let off = match &o.offset {
        Val::Int(v) => *v,
        _ => 0,
    };
    let mut n = notes(&o);
    let mut soff = soffset.arg();
    let mut fields: Vec<(String, FArg)> = Vec::new();
    if offen {
        fields.push(("offen".into(), FArg::Bool(true)));
    }
    // An offset past the field's range moves into soffset: added to a
    // register soffset, or folded with a constant one.
    let range = c.t.offset_range(thin_target::Mem::Buffer).unwrap_or(0..=0);
    if range.contains(&off) {
        if off != 0 {
            fields.push(("offset".into(), FArg::Int(off)));
        }
    } else {
        n.stage = Some(match &soffset {
            Val::Reg(r) => Stage::Add(*r, off),
            Val::Int(s) => Stage::Mov(off + s),
            _ => {
                return value(format!(
                    "{name}: offset {off} past {}: soffset must be a register or a constant",
                    range.end()
                ))
            }
        });
        soff = Arg::Reg(lit(), 0);
    }
    fields.extend(o.cache.fields());
    let vo = match voffset {
        Val::None => Arg::Str("off".into()),
        v => v.arg(),
    };
    let args = match store {
        true => vec![Arg::Reg(*data, 0), vo, desc.arg(), soff],
        false => vec![Arg::Reg(*data, 0), vo, desc.arg(), soff],
    };
    op(c, name, args, fields, n)
}

/// `ds_read(dst, addr, offset)` and `ds_write(addr, data, offset)`.
pub fn ds(c: &Ctx<'_>, store: bool, data: &Reg, addr: Val, o: Opts) -> Result<Family> {
    let fam = if store { "ds_write" } else { "ds_read" };
    let w = width(c, fam, data, &o, !store)?;
    let name = format!("ds.{}.{w}", if store { "store" } else { "load" });
    let args = match store {
        true => vec![addr.arg(), Arg::Reg(*data, 0)],
        false => vec![Arg::Reg(*data, 0), addr.arg()],
    };
    let mut fields = offset_field(&o);
    if matches!(addr, Val::Addr(_)) && !o.ordered.is_empty() {
        fields.push(("ordered".into(), FArg::Str(o.ordered.clone())));
    }
    op(c, &name, args, fields, notes(&o)).map(Family::Made)
}

/// `ds_read_tr(dst, addr, elem_bits, offset)`: transpose LDS read (`ds.load.b64.tr.b{4,8,16}`
/// or `ds.load.b96.tr.b6`).
pub fn ds_read_tr(c: &Ctx<'_>, data: &Reg, addr: Val, elem_bits: Val, o: Opts) -> Result<Family> {
    let bits = match elem_bits {
        Val::Int(v) => v,
        Val::Str(s) => match s.trim().to_ascii_lowercase().as_str() {
            "b4" | "fp4" | "e2m1" | "e2m1fn" => 4,
            "b6" | "fp6" | "bf6" | "e2m3" | "e2m3fn" | "e3m2" | "e3m2fn" => 6,
            "b8" | "fp8" | "bf8" | "e4m3" | "e4m3fn" | "e5m2" | "i8" | "u8" => 8,
            "b16" | "f16" | "bf16" | "i16" | "u16" => 16,
            _ => {
                return value(format!(
                    "ds_read_tr: element format {}: expected 4, 6, 8, or 16 bits (or 'b4'/'b6'/'b8'/'b16')",
                    super::py_repr(&s)
                ))
            }
        },
        _ => {
            return value(
                "ds_read_tr: element bits must be 4, 6, 8, or 16 (or 'b4'/'b6'/'b8'/'b16')",
            )
        }
    };
    let (name, expected_dwords) = match bits {
        4 => ("ds.load.b64.tr.b4", 2u32),
        6 => ("ds.load.b96.tr.b6", 3u32),
        8 => ("ds.load.b64.tr.b8", 2u32),
        16 => ("ds.load.b64.tr.b16", 2u32),
        _ => {
            return value(format!(
                "ds_read_tr: element bits {bits}: expected 4, 6, 8, or 16"
            ))
        }
    };
    if c.t.spell(name).is_none() {
        return value(format!(
            "ds_read_tr ({name}): {} has no transpose LDS read",
            c.t.name()
        ));
    }
    if data.n != expected_dwords {
        let repr = crate::values::reg_repr(c.values, data);
        return value(format!(
            "ds_read_tr ({name}): {repr} is {} dword(s); {bits}-bit transpose read takes {expected_dwords} dwords",
            data.n
        ));
    }
    let mut fields = offset_field(&o);
    if matches!(addr, Val::Addr(_)) && !o.ordered.is_empty() {
        fields.push(("ordered".into(), FArg::Str(o.ordered.clone())));
    }
    op(
        c,
        name,
        vec![Arg::Reg(*data, 0), addr.arg()],
        fields,
        notes(&o),
    )
    .map(Family::Made)
}

/// The native stage an instruction carries.
pub fn staged_of(i: &Instr) -> Option<Stage> {
    let o = i.attrs().staged.as_ref()?;
    match o.datum()? {
        Datum::Tuple(t) => match t.as_slice() {
            [Datum::Str(k), Datum::Reg(r), Datum::Int(off)] if &**k == "add" => {
                Some(Stage::Add(*r, *off))
            }
            [Datum::Str(k), Datum::Int(off)] if &**k == "mov" => Some(Stage::Mov(*off)),
            _ => None,
        },
        _ => None,
    }
}
