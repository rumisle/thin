//! Read golden liveness listings back into kernel values and instructions.

use thin_passes::lds;

use crate::adapters::default_target;
use crate::asm;

/// The wave index's value number.
const WAVE: i32 = 2;

/// A `lives.json` listing read back: the layout from its header, the
/// regions' accesses from the DS ops' comments (`G + lane * 4`), LDS
/// fences from their `[lds fence]` comments, and each `.Lwaves` skip
/// branch's waves from the compare before it (`s_cmp_lg_u32 s0, W`: every
/// wave but W).
pub struct Listing {
    /// The layout.
    pub layout: lds::Layout,
    /// The instructions' snapshot.
    pub snapshot: thin_ir::Snapshot,
}

impl Listing {
    /// The listing's kernel for a workgroup of `nwaves` waves, its
    /// ordered accesses read from the source. Its first `table` lines
    /// are the layout's table (all its leading `;` lines when not given).
    ///
    /// The wave index is a value of its own, read as `s0` until the first
    /// instruction that writes `s0`: placement may give its register to a
    /// value made later (a loop's counter), and the proof keys the taint
    /// by value.
    pub fn new(
        lines: &[&str],
        source: &[&str],
        nwaves: u32,
        table: Option<usize>,
    ) -> Result<Listing, String> {
        let t = default_target();
        let table =
            table.unwrap_or_else(|| lines.iter().take_while(|l| l.starts_with(';')).count());
        let layout = Listing::layout(&lines[..table]);
        let ordered = Listing::ordered(source);
        let body = &lines[table..];
        let writes_s0 = |l: &&str| {
            asm::line(t, l.trim_start()).is_ok_and(|i| i.defs().any(|r| r.file == 's' && r.lo == 0))
        };
        let first_write = body.iter().position(writes_s0).unwrap_or(body.len());
        let wave = |tok: &str| match tok {
            "s0" => Some(thin_ir::Reg::new(WAVE, 's', 0, 1)),
            _ => asm::register(tok),
        };
        let mut instrs: Vec<std::sync::Arc<thin_ir::Instr>> = Vec::new();
        for (k, line) in body.iter().enumerate() {
            let mut i = match k < first_write {
                true => asm::line_with(t, line.trim_start(), &wave)?,
                false => asm::line(t, line.trim_start())?,
            };
            let comment = i
                .attrs()
                .comment
                .as_ref()
                .map(|c| c.to_string())
                .unwrap_or_default();
            if i.form().kind == thin_target::Kind::Ds {
                let leaf = comment.split(' ').next().unwrap_or("");
                let key = layout
                    .leaves
                    .iter()
                    .position(|l| l.name == leaf)
                    .ok_or("an access's region")? as u64;
                let why = ordered
                    .iter()
                    .find(|(r, _)| r == leaf)
                    .map_or("", |(_, why)| why.as_str());
                let access = thin_ir::LdsAccess {
                    object: thin_ir::HostObject {
                        handle: std::sync::Arc::new(()),
                        facts: std::sync::Arc::default(),
                    },
                    region: key,
                    field: 0,
                    what: format!("{} {comment}", i.form().op).into(),
                    ordered: why.into(),
                };
                i = i
                    .with_attr(thin_ir::Attr::LdsAccess(access))
                    .map_err(|e| e.to_string())?;
            }
            if comment.starts_with("[lds fence]") {
                i = i
                    .with_attr(thin_ir::Attr::Fence)
                    .map_err(|e| e.to_string())?;
            }
            if line.contains(".Lwaves") && i.form().flags.has(thin_target::Flags::BRANCH) {
                let runs = instrs
                    .last()
                    .and_then(|c| c.ops().get(1))
                    .and_then(|o| match o {
                        thin_ir::Operand::Imm(m) => match m.value {
                            thin_ir::ImmValue::Int(v) => Some(v),
                            thin_ir::ImmValue::Float(_) => None,
                        },
                        _ => None,
                    });
                let taken = (0..i128::from(nwaves))
                    .filter(|&w| Some(w) != runs)
                    .collect();
                i = i
                    .with_attr(thin_ir::Attr::WavesTaken(taken))
                    .map_err(|e| e.to_string())?;
            }
            instrs.push(std::sync::Arc::new(i));
        }
        let mut gather = thin_ir::Gather {
            wave_reg: Some(thin_ir::Reg::new(WAVE, 's', 0, 1)),
            nwaves,
            ..Default::default()
        };
        for (file, n, name) in [('v', 256, "v"), ('s', 106, "s"), ('s', 1, "wave")] {
            gather.values.push(thin_ir::GatherValue {
                file,
                n,
                align: 1,
                name: name.to_owned(),
                register: Some(0),
                ..Default::default()
            });
        }
        for file in ['v', 's'] {
            gather.files.push(thin_ir::GatherFile {
                letter: file,
                placed: true,
                ..Default::default()
            });
        }
        gather.regions = (layout.leaves.iter().enumerate())
            .map(|(k, l)| thin_ir::GatherRegion {
                key: k as u64,
                base: l.base,
                size: l.size,
                name: l.name.clone(),
                on_layout: true,
                ..Default::default()
            })
            .collect();
        let snapshot =
            thin_ir::Snapshot::of_kernel(t, &gather, instrs.into()).map_err(|e| e.to_string())?;
        Ok(Listing { layout, snapshot })
    }

    /// The layout's rows: `; OFFSET END SIZE REGION …`, a union's arms
    /// marked `+` (the first) and `|`.
    fn layout(lines: &[&str]) -> lds::Layout {
        let mut layout = lds::Layout::default();
        let mut open = false;
        for line in lines.iter().filter_map(|l| l.strip_prefix(';')) {
            let words: Vec<&str> = line.split_whitespace().collect();
            let num = |k: usize| words.get(k).and_then(|w| w.parse::<u64>().ok());
            let (Some(base), Some(end)) = (num(0), num(1)) else {
                continue;
            };
            match words.get(3).copied() {
                Some("union") => {
                    layout.unions.push(lds::Union {
                        base,
                        end,
                        arms: Vec::new(),
                    });
                    open = true;
                }
                Some("+" | "|") if open => {
                    let name = words[4].to_owned();
                    let u = layout.unions.last_mut().expect("an open union");
                    u.arms.push((name.clone(), vec![layout.leaves.len()]));
                    layout.leaves.push(lds::Leaf {
                        name,
                        base,
                        size: end - base,
                    });
                }
                Some(name) => {
                    open = false;
                    layout.leaves.push(lds::Leaf {
                        name: name.to_owned(),
                        base,
                        size: end - base,
                    });
                }
                None => {}
            }
        }
        layout
    }

    /// The source's ordered writes: `ds_write X[…] (ordered: 'WHY')`.
    fn ordered(source: &[&str]) -> Vec<(String, String)> {
        (source.iter())
            .filter_map(|l| {
                let (head, why) = l.split_once(" (ordered: '")?;
                let region = head.trim().strip_prefix("ds_write ")?.split('[').next()?;
                Some((region.to_owned(), why.trim_end_matches("')").to_owned()))
            })
            .collect()
    }
}
