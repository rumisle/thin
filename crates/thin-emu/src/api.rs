//! What a host binding calls: building clusters and machines
//! from host values, and the decoded program's flat rows.

use std::collections::BTreeMap;

use crate::machine::cluster::{Carry, Cluster, Options, Order, Staged};
use crate::machine::facts::Facts;
use crate::machine::memory::Memory;
use crate::machine::run::Machine;
use crate::tags::Tables;
use crate::{Base, EmuError, Field, Ins, Operand, Program};

/// An operand as flat host values: (kind `reg` / `lit` / `label`, the
/// file or the text, first, count, neg, abs).
pub type OperandRow = (String, String, u64, u32, bool, bool);

/// A field as flat host values: (name, kind `true` / `int` / `list`, the
/// integer, the list).
pub type FieldRow = (String, String, i64, Vec<i64>);

/// A decoded instruction as flat host values.
pub type InsRow = (
    String,
    String,
    Vec<OperandRow>,
    Vec<FieldRow>,
    Option<String>,
    String,
    String,
    usize,
);

/// An operand's row.
pub fn operand_row(o: &Operand) -> OperandRow {
    match &o.base {
        Base::Reg(f, a, n) => ("reg".into(), f.clone(), u64::from(*a), *n, o.neg, o.abs),
        Base::Literal(v, t) => ("lit".into(), t.clone(), u64::from(*v), 1, o.neg, false),
        Base::Label(l) => ("label".into(), l.clone(), 0, 0, false, false),
    }
}

/// A field's row.
pub fn field_row(name: &str, f: &Field) -> FieldRow {
    match f {
        Field::True => (name.into(), "true".into(), 1, Vec::new()),
        Field::Int(v) => (name.into(), "int".into(), *v as i64, Vec::new()),
        Field::List(l) => (
            name.into(),
            "list".into(),
            0,
            l.iter().map(|x| *x as i64).collect(),
        ),
    }
}

/// An instruction's row.
pub fn ins_row(i: &Ins) -> InsRow {
    (
        i.canon.clone(),
        i.kind.clone(),
        i.operands.iter().map(operand_row).collect(),
        i.fields.iter().map(|(k, v)| field_row(k, v)).collect(),
        i.matrix.clone(),
        i.text.clone(),
        i.comment.clone(),
        i.line,
    )
}

/// A program from decoded JSON: `{ins: [...],
/// labels: {...}, sections: [...], user_sgprs}`, or a list of instructions.
pub fn program_of(json: &str, w: usize) -> Result<Program, EmuError> {
    let reading = |message| EmuError {
        cause: None,
        message,
    };
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| EmuError {
        cause: None,
        message: format!("a decoded program's JSON: {e}"),
    })?;
    let list = v.get("ins").unwrap_or(&v);
    let ins = list
        .as_array()
        .into_iter()
        .flatten()
        .map(|i| Ins::of(i, w))
        .collect::<Vec<_>>();
    let labels = v
        .get("labels")
        .and_then(|l| l.as_object())
        .into_iter()
        .flatten()
        .map(|(k, i)| (k.clone(), i.as_u64().unwrap_or(0) as usize))
        .collect();
    let sections = v
        .get("sections")
        .and_then(|s| s.as_array())
        .into_iter()
        .flatten()
        .map(|s| {
            let n = |k: usize| s[k].as_u64().unwrap_or(0) as usize;
            (n(0), s[1].as_str().unwrap_or("").to_owned(), n(2))
        })
        .collect();
    let pc_layout = v
        .get("pc_layout")
        .filter(|v| !v.is_null())
        .map(crate::machine::pc::Layout::from_json)
        .transpose()?;
    if let Some(layout) = &pc_layout {
        layout.validate(ins.len())?;
    }
    Ok(Program {
        ins,
        pc_layout,
        labels,
        sections,
        user_sgprs: v.get("user_sgprs").and_then(|u| u.as_u64()).unwrap_or(2) as u32,
        label_lines: Vec::new(),
        workgroup_ids: v
            .get("workgroup_ids")
            .filter(|ids| !ids.is_null())
            .map(|ids| -> Result<[bool; 3], EmuError> {
                let array = ids
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .ok_or_else(|| reading("workgroup_ids: expected three booleans".into()))?;
                let flags = array
                    .iter()
                    .map(|id| {
                        id.as_bool()
                            .ok_or_else(|| reading("workgroup_ids: expected a boolean".into()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok([flags[0], flags[1], flags[2]])
            })
            .transpose()?,
        kernarg_sgpr: v
            .get("kernarg_sgpr")
            .and_then(|n| n.as_u64())
            .map(u32::try_from)
            .transpose()
            .map_err(|_| reading("kernarg_sgpr: outside the u32 range".into()))?,
    })
}

/// The facts of a target.
pub fn facts(target: &thin_target::Target) -> Facts {
    Facts::of(target)
}

/// A dispatch order from host text: `""` (grid), `"reverse"`, or a seed.
pub fn order(text: &str) -> Result<Order, EmuError> {
    match text {
        "" | "none" => Ok(Order::Grid),
        "reverse" => Ok(Order::Reverse),
        seed => seed.parse().map(Order::Shuffle).map_err(|_| EmuError {
            cause: None,
            message: format!("order {seed:?}: none, \"reverse\", an int, or one per rank"),
        }),
    }
}

/// An LDS carry mode by name, `fill` taking its bytes.
pub fn carry(mode: &str, data: Vec<u8>, tags: Vec<i64>) -> Result<Carry, EmuError> {
    Ok(match mode {
        "" | "none" => Carry::None,
        "prev" => Carry::Prev,
        "launch" => Carry::Launch,
        "fill" => Carry::Fill(data, tags),
        "record" => Carry::Record,
        other => {
            return Err(EmuError {
                cause: None,
                message: format!("lds carry {other:?}: prev, launch, fill or record"),
            })
        }
    })
}

/// What a cluster runs: the program and its counts.
pub struct Kernel {
    /// The decoded program.
    pub program: Program,
    /// The target.
    pub target: &'static thin_target::Target,
    /// VGPRs.
    pub nv: usize,
    /// AGPRs.
    pub na: usize,
    /// LDS bytes.
    pub lds: usize,
    /// Lanes per workgroup.
    pub wgsize: usize,
}

/// Builds a cluster: one machine per rank.
pub fn cluster(
    k: &Kernel,
    launches: Vec<Vec<u64>>,
    grid: (u32, u32),
    options: Options,
    sum_records: bool,
    trace: bool,
) -> Result<Cluster, EmuError> {
    let facts = facts(k.target);
    let waves = k.wgsize / facts.w;
    let machines = (0..launches.len())
        .map(|r| {
            let mut m = Machine::new(k.program.clone(), facts.clone(), k.nv, k.na, k.lds)?;
            m.rank = r;
            m.sum_records = sum_records;
            m.trace = trace;
            Ok(m)
        })
        .collect::<Result<Vec<_>, EmuError>>()?;
    Ok(Cluster::new(machines, launches, grid, waves, options))
}

/// Runs a cluster over memory and tables, its staged writes per (rank,
/// launch) applied as each launch starts.
pub fn run(
    c: &mut Cluster,
    mem: &mut Memory,
    tables: &mut Tables,
    staged: BTreeMap<(usize, usize), Vec<Staged>>,
) -> Result<(), EmuError> {
    c.staged = staged;
    c.run(mem, tables)
}
