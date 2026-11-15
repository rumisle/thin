//! Unassignment coverage: the exact law on every converted kernel, and how
//! many values, dwords, operands and registers become virtual or stay
//! pinned, per reason.

use std::collections::{BTreeMap, BTreeSet};

use thin_ir::Operand;
use thin_lift::program::{Program, Rendering};
use thin_lift::schedule::Schedule;
use thin_lift::unassign::{unassign, Reason};
use thin_targets::Instruction;

/// The order a value's first reason is counted in for the exclusive split.
fn rank(r: &Reason) -> u8 {
    match r {
        Reason::UnresolvedControlFlow => 0,
        Reason::RelativeIndexing => 1,
        Reason::Entry(_) => 2,
        Reason::CallBoundary => 3,
        Reason::Undefined => 4,
    }
}

fn same(a: &[Instruction], b: &[Instruction]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.offset, x.size, x.mnemonic, x.counted) == (y.offset, y.size, y.mnemonic, y.counted)
                && x.operands() == y.operands()
                && x.modifiers() == y.modifiers()
                && (0..x.operands().len()).all(|k| x.signs(k) == y.signs(k))
        })
}

#[derive(Default)]
pub struct Census {
    counts: BTreeMap<String, u64>,
    examples: BTreeMap<String, String>,
}

impl Census {
    fn add(&mut self, key: impl Into<String>, n: u64) {
        *self.counts.entry(key.into()).or_default() += n;
    }

    pub fn check(
        &mut self,
        program: &Program,
        rendering: &Rendering,
        schedule: &Schedule,
        body: &[Instruction],
        context: &str,
    ) {
        self.add("kernels attempted", 1);
        let (virt, residue) = match unassign(program, schedule) {
            Ok(r) => r,
            Err(e) => {
                let class = format!("refused: {}", e.class);
                self.add(class.clone(), 1);
                self.examples
                    .entry(class)
                    .or_insert(format!("{context}: {e}"));
                return;
            }
        };
        self.add("kernels unassigned", 1);
        let law = residue
            .pin(&virt)
            .map_err(|e| e.to_string())
            .and_then(|pinned| {
                let rows = pinned.snapshot.instrs().iter();
                let equal = rows.zip(program.snapshot.instrs().iter()).all(|(a, b)| {
                    a.form().id == b.form().id && a.ops() == b.ops() && a.mods() == b.mods()
                });
                let same_flow = pinned.snapshot.control_flow() == program.snapshot.control_flow()
                    && pinned.snapshot.entry() == program.snapshot.entry()
                    && pinned.entry == program.entry;
                let rendered = rendering.render(&pinned).map_err(|e| e.to_string())?;
                match (equal && same_flow, same(&rendered, body)) {
                    (true, true) => Ok(()),
                    (false, _) => Err("pinned rows differ".to_owned()),
                    (_, false) => Err("rendered instructions differ".to_owned()),
                }
            });
        match law {
            Ok(()) => self.add("exact law held", 1),
            Err(why) => {
                let class = format!("exact law failed: {why}");
                self.add(class.clone(), 1);
                self.examples.entry(class).or_insert(context.to_owned());
            }
        }
        self.assign(&virt, rendering, schedule, context);
        let s = &virt.snapshot;
        let mut pinned_regs: BTreeSet<(char, u32)> = BTreeSet::new();
        let mut touched: BTreeSet<(char, u32)> = BTreeSet::new();
        for (v, info) in s.values().iter().enumerate() {
            let file = info.file;
            let n = u64::from(info.n);
            let reasons = &residue.reasons[v];
            let regs = (residue.bases[v]..residue.bases[v] + info.n).map(|r| (file, r));
            touched.extend(regs.clone());
            let Some(first) = reasons.iter().min_by_key(|r| rank(r)) else {
                self.add(format!("values virtual {file}"), 1);
                self.add(format!("dwords virtual {file}"), n);
                continue;
            };
            pinned_regs.extend(regs);
            self.add(format!("values pinned {file}"), 1);
            self.add(format!("dwords pinned {file}"), n);
            self.add(format!("values pinned first reason: {}", first.name()), 1);
            self.add(format!("dwords pinned first reason: {}", first.name()), n);
            for r in reasons {
                self.add(format!("values pinned any reason: {}", r.name()), 1);
                if let Reason::Entry(e) = r {
                    self.add(format!("values pinned entry: {e:?}"), 1);
                }
            }
            let class = format!("example pinned: {}", first.name());
            self.examples.entry(class).or_insert_with(|| {
                let at = (0..s.instrs().len()).find(|&k| {
                    s.instrs()[k]
                        .ops()
                        .iter()
                        .any(|o| o.reg().is_some_and(|r| r.value as usize == v))
                });
                let line = at.map_or("(entry only)".to_owned(), |k| {
                    format!(
                        "+{:#x} {}",
                        program.addresses[k],
                        program.line(k).unwrap_or_default()
                    )
                });
                format!("{context}{line}")
            });
        }
        for (file, _) in &touched {
            self.add(format!("registers used {file}"), 1);
        }
        for (file, _) in touched.difference(&pinned_regs) {
            self.add(format!("registers only virtual {file}"), 1);
        }
        for i in s.instrs().iter() {
            for o in i.ops() {
                match o {
                    Operand::Reg(r) => {
                        let pinned = !residue.reasons[r.value as usize].is_empty();
                        let kind = if pinned { "pinned" } else { "virtual" };
                        self.add(format!("operands {kind} {}", r.file), 1);
                    }
                    Operand::Phys(p) => self.add(format!("operands named {}", p.name), 1),
                    _ => {}
                }
            }
        }
    }

    /// The semantic law's static half: thin's register assignment of the virtual
    /// program, its placement check, and the placed program's rendering.
    fn assign(
        &mut self,
        virt: &Program,
        rendering: &Rendering,
        schedule: &Schedule,
        context: &str,
    ) {
        let scheduled = match virt.scheduled(schedule) {
            Ok(s) => s,
            Err(e) => {
                self.add(format!("semantic law refused: {}", e.class), 1);
                return;
            }
        };
        let s = &scheduled;
        let t = s.target();
        let calls = s.control_flow().is_some_and(|c| {
            c.transfers
                .iter()
                .any(|t| matches!(t.kind, thin_ir::control::TransferKind::Call { .. }))
        });
        let kind = if calls { "with calls" } else { "without calls" };
        self.add(format!("semantic law attempted {kind}"), 1);
        let rules = thin_targets::hazards(t.name()).expect("hazards").rules();
        let limits: Vec<(char, u32)> = ['v', 's']
            .iter()
            .map(|&f| (f, t.hooks().reg_count(f)))
            .collect();
        let o = thin_passes::regalloc::Options::default();
        let outcome = thin_passes::regalloc::assign(s, rules, &limits, o)
            .map_err(|d| {
                let text: String = (d.msg.iter())
                    .filter_map(|p| match p {
                        thin_ir::Part::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect();
                let class = if text.contains("which a pin before it holds") {
                    "no room: pin over pin"
                } else if text.contains("no room") {
                    "no room"
                } else {
                    "other"
                };
                (class, text)
            })
            .and_then(|p| {
                thin_passes::regalloc::check(s, rules, &p.bases, o)
                    .map_err(|_| ("placement check", String::new()))?;
                let placed = thin_lift::unassign::place(virt, &p.bases)
                    .map_err(|e| ("place", e.to_string()))?;
                rendering
                    .render(&placed)
                    .map_err(|e| ("render", e.to_string()))?;
                Ok(())
            });
        match outcome {
            Ok(()) => self.add(format!("semantic law assigned and rendered {kind}"), 1),
            Err((class, text)) => {
                let class = format!("semantic law refused {kind}: {class}");
                self.add(class.clone(), 1);
                let short: String = text.chars().take(300).collect();
                self.examples
                    .entry(class)
                    .or_insert(format!("{context}: {short}"));
            }
        }
    }

    pub fn json(&self) -> (serde_json::Value, serde_json::Value) {
        (
            serde_json::to_value(&self.counts).expect("counts"),
            serde_json::to_value(&self.examples).expect("examples"),
        )
    }

    pub fn report(&self) {
        for (k, n) in &self.counts {
            println!("unassign {k}: {n}");
        }
        for (k, e) in &self.examples {
            println!("unassign {k}: {e}");
        }
    }
}
