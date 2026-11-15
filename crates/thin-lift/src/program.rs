//! Physical IR construction, inverse of rendering pinned instructions.
//!
//! Registers in each file share a pinned physical backing value, preserving
//! overlapping ranges. Labels retain original addresses, including targets
//! on removed scheduling instructions. Indirect control flow remains explicit
//! and unresolved until subroutine-call lifting; check verdicts are not a
//! claim that those missing interprocedural edges have been analyzed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use thin_ir::{FieldId, FieldValue, Imm, ImmValue, Instr, Operand, Reg, Snapshot, ValueInfo};
use thin_target::{phys_named, Encoded, Form, Pseudo, Target};
use thin_targets::decode::{MVal, Opd};
use thin_targets::Instruction;

use crate::interface::{Interface, UserValue};
use crate::schedule::Schedule;

/// A refusal to construct or re-render a physical program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramError {
    /// Original instruction offset, when the refusal names an instruction.
    pub offset: Option<u32>,
    /// A stable refusal class, suitable for coverage counts.
    pub class: &'static str,
    /// The precise cause.
    pub detail: String,
}

impl std::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.class, self.detail)
    }
}
impl std::error::Error for ProgramError {}

fn error(offset: Option<u32>, class: &'static str, detail: impl Into<String>) -> ProgramError {
    ProgramError {
        offset,
        class,
        detail: detail.into(),
    }
}

/// A hardware-filled input's meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryValue {
    /// One enabled ABI user value.
    User(UserValue),
    /// User-data dwords reserved beyond the named ABI values.
    UserTail,
    /// Workgroup coordinate, axis 0/1/2.
    WorkgroupId(u8),
    /// Workgroup information after the coordinate SGPRs.
    WorkgroupInfo,
    /// Packed local workitem coordinates, with the descriptor component mode.
    WorkitemIds(u8),
    /// A target-defined named hardware input.
    HardwareNamed(&'static str),
    /// A named input whose initialization depends on a dispatch condition.
    ConditionalHardware {
        /// The hardware register.
        name: &'static str,
        /// The condition from the ISA's initial-state table.
        condition: &'static str,
    },
}

/// A typed binding to a pinned register or a named hardware register.
#[derive(Clone, Debug, PartialEq)]
pub struct EntryBinding {
    /// The input's meaning.
    pub value: EntryValue,
    /// Its exact physical location.
    pub operand: Operand,
}

/// A physical thin IR program, prior to resolving indirect calls.
pub struct Program {
    /// Immutable native IR and its entry subranges.
    pub snapshot: Snapshot,
    /// Entry locations recovered from the Interface and target hardware inputs.
    pub entry: Vec<EntryBinding>,
    /// IR indices of unresolved indirect PC transfers.
    pub indirect: Vec<usize>,
    /// Original address of each IR row (labels included).
    pub addresses: Vec<u32>,
    /// Original byte boundaries, including removed waits, mapped to IR rows.
    pub boundaries: BTreeMap<u32, thin_ir::control::Destination>,
}

#[derive(Clone)]
struct Origin {
    row: usize,
    offset: u32,
    size: u8,
    mnemonic: &'static str,
    form: i32,
}

/// Encoding/layout choices not represented by physical IR.
#[derive(Clone)]
pub struct Rendering {
    origins: Vec<Origin>,
    end: usize,
}

fn physical(file: char, first: u32, count: u32) -> Operand {
    Operand::Reg(Reg::new(
        if file == 'v' { 0 } else { 1 },
        file,
        first,
        count,
    ))
}

fn entry_bindings(interface: &Interface) -> Vec<EntryBinding> {
    let mut out = Vec::new();
    let mut filled = 0;
    for user in &interface.user_sgprs {
        let n = user.value.dwords();
        out.push(EntryBinding {
            value: EntryValue::User(user.value),
            operand: physical('s', user.first, n),
        });
        filled = filled.max(user.first + n);
    }
    if filled < interface.user_sgpr_count {
        out.push(EntryBinding {
            value: EntryValue::UserTail,
            operand: physical('s', filled, interface.user_sgpr_count - filled),
        });
    }
    let mut next = interface.user_sgpr_count;
    for (axis, enabled) in interface.workgroup_ids.iter().enumerate() {
        if *enabled {
            out.push(EntryBinding {
                value: EntryValue::WorkgroupId(axis as u8),
                operand: physical('s', next, 1),
            });
            next += 1;
        }
    }
    if interface.workgroup_info {
        out.push(EntryBinding {
            value: EntryValue::WorkgroupInfo,
            operand: physical('s', next, 1),
        });
    }
    // RDNA4 ISA guide: CS local X/Y/Z are packed in VGPR0.
    out.push(EntryBinding {
        value: EntryValue::WorkitemIds(interface.workitem_id_components),
        operand: physical('v', 0, 1),
    });
    // Table 12: grid X and packed grid Y/Z are also delivered in TTMPs.
    out.push(EntryBinding {
        value: EntryValue::ConditionalHardware {
            name: "ttmp7",
            condition: "GridYZvalid",
        },
        operand: Operand::Phys(phys_named("ttmp7").expect("ttmp7")),
    });
    for name in ["ttmp9", "exec"] {
        out.push(EntryBinding {
            value: EntryValue::HardwareNamed(name),
            operand: Operand::Phys(phys_named(name).expect("hardware input")),
        });
    }
    out
}

fn operand(i: &Instruction, k: usize, o: &Opd) -> Result<Operand, ProgramError> {
    Ok(match *o {
        Opd::V { n, count } => physical('v', n.into(), count.into()),
        Opd::S { n, count } => physical('s', n.into(), count.into()),
        Opd::Named(&(name, _, _)) => Operand::Phys(
            phys_named(name).ok_or_else(|| error(Some(i.offset), "named register", name))?,
        ),
        Opd::Off => Operand::Phys(phys_named("off").expect("off")),
        Opd::Int(n) => Operand::Imm(Imm::int(n.into())),
        Opd::Float(n) => Operand::Imm(Imm {
            value: ImmValue::Float(n.into()),
            text: "".into(),
        }),
        Opd::Hwreg { code, .. } => Operand::Imm(Imm {
            value: ImmValue::Int(code.into()),
            text: i.line().operands[k].text.clone().into(),
        }),
        Opd::Label(t) => Operand::Block(thin_targets::decode::label(t).into()),
    })
}

fn fields(i: &Instruction, f: &Form) -> Result<Vec<(FieldId, FieldValue)>, ProgramError> {
    i.modifiers()
        .iter()
        .map(|m| {
            if m.key == "bound_ctrl" && m.value != MVal::Int(0) {
                return Err(error(
                    Some(i.offset),
                    "rendering field",
                    "physical bound_ctrl:1 is not preserved by the current IR printer",
                ));
            }
            if m.key == "fi" {
                return Err(error(
                    Some(i.offset),
                    "field",
                    "physical DPP fi has no IR field",
                ));
            }
            let id = FieldId::named(m.key);
            if id.is_err() && f.field("dpp").is_some() {
                let value = match m.value {
                    MVal::Int(n) => n.to_string(),
                    MVal::List(items, n) => format!(
                        "[{}]",
                        items[..n as usize]
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    ),
                    MVal::Flag => String::new(),
                    MVal::Symbol(name, _) => name.into(),
                };
                return Ok((
                    FieldId::named("dpp").expect("dpp"),
                    FieldValue::Text(
                        if matches!(m.value, MVal::Flag) {
                            m.key.to_owned()
                        } else {
                            format!("{}:{value}", m.key)
                        }
                        .into(),
                    ),
                ));
            }
            let id = id.map_err(|e| error(Some(i.offset), "field", e.to_string()))?;
            let value = match m.value {
                MVal::Flag => FieldValue::Bool(true),
                MVal::Int(n)
                    if f.field(m.key)
                        .is_some_and(|f| f.kind == thin_target::FieldKind::Flag) =>
                {
                    FieldValue::Bool(n != 0)
                }
                MVal::Int(n) => FieldValue::Int(n.into()),
                MVal::Symbol(name, _) => FieldValue::Text(name.into()),
                MVal::List(items, n) => FieldValue::Ints(
                    items[..n as usize]
                        .iter()
                        .map(|&v| i128::from(v))
                        .collect::<Vec<_>>()
                        .into(),
                ),
            };
            Ok((id, value))
        })
        .collect()
}

/// One decoded instruction as pinned physical IR.
fn instr_of(target: &'static Target, i: &Instruction) -> Result<Instr, ProgramError> {
    let f = thin_targets::physical_form(target.name(), i)
        .map_err(|e| error(Some(i.offset), "form", e.to_string()))?;
    let ops = i
        .operands()
        .iter()
        .enumerate()
        .map(|(k, o)| operand(i, k, o))
        .collect::<Result<Vec<_>, _>>()?;
    let mods: Vec<_> = (0..ops.len())
        .map(|k| {
            let (neg, abs) = i.signs(k);
            u8::from(neg) | u8::from(abs) << 1
        })
        .collect();
    let mods = mods
        .iter()
        .any(|&v| v != 0)
        .then(|| mods.into_boxed_slice());
    Ok(Instr::new(f, ops, fields(i, f)?, mods))
}

/// Inverts rendering of decoded, unscheduled instructions into pinned IR.
/// Every original instruction has a row; labels at removed waits attach to
/// the next surviving row. Rendering the result with its residue reproduces
/// the same decoded instructions and original address space.
pub fn unrender(
    target: &'static Target,
    instructions: &[Instruction],
    schedule: &Schedule,
    interface: &Interface,
) -> Result<(Program, Rendering), ProgramError> {
    if interface.wave_size != target.definition().wave {
        return Err(error(
            None,
            "wave size",
            format!(
                "interface wave{}, IR target wave{}",
                interface.wave_size,
                target.definition().wave
            ),
        ));
    }
    let end = instructions
        .iter()
        .chain(schedule.entries.iter().map(|e| &e.instruction))
        .map(|i| i.offset as usize + usize::from(i.size))
        .max()
        .unwrap_or(0);
    let labels: BTreeSet<_> = instructions
        .iter()
        .filter_map(Instruction::branch_target)
        .collect();
    let mut labels = labels.into_iter().peekable();
    let mut rows = Vec::new();
    let mut addresses = Vec::new();
    let mut origins = Vec::new();
    let mut indirect = Vec::new();
    let label = |t: u32, rows: &mut Vec<Arc<Instr>>, addresses: &mut Vec<u32>| {
        rows.push(Arc::new(Instr::new(
            target.pseudo(Pseudo::Label),
            Vec::new(),
            vec![(
                FieldId::named("name").expect("label name"),
                FieldValue::Text(thin_targets::decode::label(t).into()),
            )],
            None,
        )));
        addresses.push(t);
    };
    for i in instructions {
        while labels.peek().is_some_and(|&t| t <= i.offset) {
            label(labels.next().expect("label"), &mut rows, &mut addresses);
        }
        let instr = instr_of(target, i)?;
        let f = instr.form();
        origins.push(Origin {
            row: rows.len(),
            offset: i.offset,
            size: i.size,
            mnemonic: i.mnemonic,
            form: f.id,
        });
        if matches!(i.mnemonic, "s_setpc_b64" | "s_swappc_b64" | "s_call_b64") {
            indirect.push(rows.len());
        }
        rows.push(Arc::new(instr));
        addresses.push(i.offset);
    }
    for t in labels {
        label(t, &mut rows, &mut addresses);
    }
    let values = ['v', 's']
        .iter()
        .map(|&file| ValueInfo {
            file,
            n: target.hooks().reg_count(file),
            phys: Some(0),
            pin: Some(0),
            align: 1,
            ..ValueInfo::default()
        })
        .collect();
    let entry = entry_bindings(interface);
    let entry_regs = entry
        .iter()
        .filter_map(|b| b.operand.reg().copied())
        .collect();
    let mut boundaries = BTreeMap::new();
    for address in instructions
        .iter()
        .map(|i| i.offset)
        .chain(schedule.entries.iter().map(|e| e.instruction.offset))
        .chain(u32::try_from(end))
    {
        let row = addresses.partition_point(|&a| a < address);
        let destination = if row < rows.len() {
            thin_ir::control::Destination::Instruction(row)
        } else {
            thin_ir::control::Destination::Exit
        };
        boundaries.insert(address, destination);
    }
    let snapshot = Snapshot::new(target, values, rows.into())
        .with_entry(entry_regs)
        .map_err(|e| error(None, "entry", e.to_string()))?;
    Ok((
        Program {
            snapshot,
            entry,
            indirect,
            addresses,
            boundaries,
        },
        Rendering { origins, end },
    ))
}

impl Program {
    /// The program's rows with its schedule's instructions back in place,
    /// for the analyses that read waits (register assignment holds a
    /// load's registers while it may be in flight). Each
    /// scheduling instruction goes after the rows at or before its
    /// original address, so a label on a removed wait reaches the wait;
    /// typed edges to a row keep going to that row (past a wait before
    /// it, which can only hold loads longer). Scheduling instructions name
    /// no registers, so this serves virtual programs too.
    pub fn scheduled(&self, schedule: &Schedule) -> Result<Snapshot, ProgramError> {
        self.scheduled_rows(schedule).map(|(s, _)| s)
    }

    /// [`Program::scheduled`] and, per row of this program, its row there.
    pub fn scheduled_rows(
        &self,
        schedule: &Schedule,
    ) -> Result<(Snapshot, Vec<usize>), ProgramError> {
        let s = &self.snapshot;
        let target = s.target();
        let n = s.instrs().len();
        let mut rows: Vec<Arc<Instr>> = Vec::with_capacity(n + schedule.entries.len());
        let mut new_row = vec![0; n];
        let mut at = 0;
        for e in &schedule.entries {
            let i = instr_of(target, &e.instruction)?;
            if i.ops().iter().any(|o| o.reg().is_some()) {
                return Err(error(
                    Some(e.instruction.offset),
                    "schedule",
                    "a scheduling instruction names a register",
                ));
            }
            let before = self
                .addresses
                .partition_point(|&a| a <= e.instruction.offset);
            while at < before {
                new_row[at] = rows.len();
                rows.push(s.instrs()[at].clone());
                at += 1;
            }
            rows.push(Arc::new(i));
        }
        while at < n {
            new_row[at] = rows.len();
            rows.push(s.instrs()[at].clone());
            at += 1;
        }
        let row = |d: thin_ir::control::Destination| match d {
            thin_ir::control::Destination::Instruction(r) => {
                thin_ir::control::Destination::Instruction(new_row[r])
            }
            exit => exit,
        };
        let control = s.control_flow().map(|c| {
            let mut c = c.clone();
            for t in &mut c.transfers {
                t.instruction = new_row[t.instruction];
                t.kind = match t.kind {
                    thin_ir::control::TransferKind::Call { link, continuation } => {
                        thin_ir::control::TransferKind::Call {
                            link,
                            continuation: row(continuation),
                        }
                    }
                    thin_ir::control::TransferKind::UnresolvedCall { continuation } => {
                        thin_ir::control::TransferKind::UnresolvedCall {
                            continuation: continuation.map(row),
                        }
                    }
                    kind => kind,
                };
                for e in &mut t.edges {
                    e.target = row(e.target);
                    if let thin_ir::control::EdgeKind::Return { caller } = e.kind {
                        e.kind = thin_ir::control::EdgeKind::Return {
                            caller: new_row[caller],
                        };
                    }
                }
            }
            c
        });
        let snapshot = Snapshot::new(target, s.values().to_vec(), rows.into())
            .with_entry(s.entry().to_vec())
            .and_then(|v| v.with_control_flow(control))
            .map_err(|e| error(None, "schedule", e.to_string()))?;
        Ok((snapshot, new_row))
    }

    /// One IR row printed with its pinned physical register spellings.
    pub fn line(&self, row: usize) -> Result<String, thin_ir::Error> {
        let i = self
            .snapshot
            .instrs()
            .get(row)
            .ok_or_else(|| thin_ir::Error::Value("physical IR row no longer exists".into()))?;
        let regs = i
            .ops()
            .iter()
            .filter_map(Operand::reg)
            .map(|r| thin_ir::print::phys_text(r.file, self.snapshot.phys(r), r.n))
            .collect::<Vec<_>>();
        thin_ir::print::line(i, &regs)
    }
}

impl Rendering {
    /// Re-renders IR at its original offsets. The residue stores spelling,
    /// layout and IR-row correspondence, not original operand values.
    pub fn render(&self, program: &Program) -> Result<Vec<Instruction>, ProgramError> {
        let target = program.snapshot.target();
        let mut bytes = vec![0; self.end];
        for origin in &self.origins {
            let i = program
                .snapshot
                .instrs()
                .get(origin.row)
                .ok_or_else(|| error(Some(origin.offset), "layout", "IR row was removed"))?;
            if i.form().id != origin.form {
                return Err(error(
                    Some(origin.offset),
                    "form",
                    "IR opcode changed from its encoding choice",
                ));
            }
            let printed = program
                .line(origin.row)
                .map_err(|e| error(Some(origin.offset), "render", e.to_string()))?;
            let suffix = printed.find(' ').map_or("", |n| &printed[n..]);
            let line = format!("{}{suffix}", origin.mnemonic);
            let encoded = match target
                .hooks()
                .encode(&line)
                .map_err(|e| error(Some(origin.offset), "encode", e))?
            {
                Encoded::Bytes(b) => b,
                Encoded::Branch { .. } => {
                    let Some(Operand::Block(name)) = i.ops().first() else {
                        return Err(error(Some(origin.offset), "branch", "missing label"));
                    };
                    let t = name
                        .strip_prefix(".L")
                        .and_then(|t| t.parse().ok())
                        .ok_or_else(|| {
                            error(
                                Some(origin.offset),
                                "branch",
                                "edited label is not an original address",
                            )
                        })?;
                    let mut branch = Instruction::new(origin.offset, origin.mnemonic);
                    branch.push(Opd::Label(t));
                    thin_targets::encode_one(target.name(), &branch)
                        .map_err(|e| error(Some(origin.offset), "encode", e))?
                }
            };
            if encoded.len() != usize::from(origin.size) {
                return Err(error(
                    Some(origin.offset),
                    "layout",
                    "encoded instruction size changed",
                ));
            }
            bytes[origin.offset as usize..origin.offset as usize + encoded.len()]
                .copy_from_slice(&encoded);
        }
        let decoder = thin_targets::decoder(target.name())
            .map_err(|e| error(None, "decoder", e.to_string()))?;
        self.origins
            .iter()
            .map(|o| {
                decoder
                    .decode_at(&bytes, o.offset as usize)
                    .map_err(|e| error(Some(o.offset), "decode", e.to_string()))
            })
            .collect()
    }
}
