//! Forms from the operand table: the shared rules.

use std::collections::HashMap;

use crate::form::{base, mem_bytes, reads_dst, Field, Form, Slot, WaitText};
use crate::vocab::{dst_half, dst_sel, mem_of_encoding, phys_named, Flags, Kind, SlotKind};
use crate::Error;

/// A row of a target's operand table: one (mnemonic, operand count).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// The spelling, `+rtn` appended for a returning atomic keyed apart.
    pub mnemonic: String,
    /// The number of explicit operands.
    pub count: usize,
    /// Each operand's role text: `r`, `w`, `rw` or `-`.
    pub roles: Vec<String>,
    /// Named registers written implicitly.
    pub defs: Vec<String>,
    /// Named registers read implicitly.
    pub uses: Vec<String>,
    /// Each operand's scalar widths in dwords.
    pub sgpr_widths: Vec<Vec<u32>>,
    /// The XML encoding name.
    pub encoding: String,
    /// Each operand's (XML type, size in bits).
    pub slots: Vec<(String, u32)>,
}

/// A wait instruction: its mnemonic, count fields and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitInsn {
    /// The mnemonic.
    pub mnemonic: &'static str,
    /// The field names of its counts, in order.
    pub fields: &'static [&'static str],
    /// How its line spells its counts.
    pub text: WaitText,
}

/// A target's form rules: the hooks the shared rules read.
pub struct FormRules {
    /// Name prefixes of matrix-unit ops.
    pub matrix: &'static [&'static str],
    /// Name prefixes of scalar-memory ops.
    pub smem: &'static [&'static str],
    /// The transcendental ops, by operation name.
    pub trans: &'static [&'static str],
    /// Name prefixes of ops that wait for counters.
    pub waits: &'static [&'static str],
    /// The one op that is the barrier.
    pub barrier: &'static str,
    /// Scalar ops that read SCC as a carry.
    pub carry_in: &'static [&'static str],
    /// Name prefixes of scalar ops whose result differs between waves.
    pub wave_varying: &'static [&'static str],
    /// The counter a form counts on, from (flags, name, first role writes).
    pub counter: fn(Flags, &str, bool) -> Option<usize>,
    /// A form's fields, from (spelling, encoding, nsrc).
    pub fields: fn(&str, &str, usize) -> Result<Vec<Field>, Error>,
    /// The operand positions taking neg/abs.
    pub signs: fn(&str) -> Vec<usize>,
    /// Slot k's widths beyond a vector-memory address's.
    pub widths: fn(&str, usize) -> Vec<u32>,
}

/// The flag when the condition holds, else none.
fn when(on: bool, flag: Flags) -> Flags {
    if on {
        flag
    } else {
        Flags::NONE
    }
}

fn starts(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| name.starts_with(p))
}

/// The slot kind of an XML operand type.
pub fn slot_kind(xml_type: &str) -> Result<SlotKind, Error> {
    use SlotKind::*;
    Ok(match xml_type {
        "OPR_VGPR" | "OPR_VGPR_OR_LDS" | "OPR_SRC_VGPR" => Vgpr,
        "OPR_VGPR_OR_ACCVGPR" | "OPR_SRC_VGPR_OR_ACCVGPR" => Av,
        "OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST" => Avc,
        "OPR_ACCVGPR" | "OPR_SRC_ACCVGPR" => Agpr,
        "OPR_SRC"
        | "OPR_SRC_SIMPLE"
        | "OPR_SRC_NOLIT"
        | "OPR_SRC_NOLDS"
        | "OPR_SRC_VGPR_OR_INLINE" => Vsrc,
        "OPR_SSRC"
        | "OPR_SSRC_NOLIT"
        | "OPR_SSRC_LANESEL"
        | "OPR_SMEM_OFFSET"
        | "OPR_SMEM_OFFSET_NOK"
        | "OPR_SSRC_BARRIER_ID"
        | "OPR_SREG_M0" => Ssrc,
        "OPR_SREG" | "OPR_SREG_NOVCC" | "OPR_SDST" | "OPR_SREG_LITERAL" => Sgpr,
        "OPR_VCC" => Special,
        "OPR_LABEL" => Label,
        "OPR_SIMM4" | "OPR_SIMM5" | "OPR_SIMM8" | "OPR_SIMM16" | "OPR_SIMM24" | "OPR_SIMM32"
        | "OPR_HWREG" | "OPR_SENDMSG" | "OPR_SENDMSG_RTN" | "OPR_WAITCNT" | "OPR_DELAY"
        | "OPR_WAIT_ALU" | "OPR_WAIT_EVENT" | "OPR_WAIT_MEM_DS" | "OPR_CLAUSE" | "OPR_SLEEP"
        | "OPR_VERSION" | "OPR_ATTR" | "OPR_TGT" => Imm,
        other => {
            return Err(Error::Definition(format!(
                "XML operand type {other} has no slot kind"
            )))
        }
    })
}

/// The unit flag of a mnemonic with a target's matrix and scalar-memory
/// prefixes.
pub fn unit_of(mnemonic: &str, matrix: &[&str], smem: &[&str]) -> Option<Flags> {
    Some(if starts(mnemonic, matrix) {
        Flags::MATRIX
    } else if mnemonic.starts_with("v_") {
        Flags::VALU
    } else if starts(mnemonic, &["s_cbranch", "s_branch"]) {
        Flags::BRANCH
    } else if starts(mnemonic, smem) {
        Flags::SMEM
    } else if mnemonic.starts_with("s_") {
        Flags::SALU
    } else if mnemonic.starts_with("ds_") {
        Flags::DS
    } else if starts(
        mnemonic,
        &["buffer_", "global_", "flat_", "scratch_", "tbuffer_"],
    ) {
        Flags::VMEM
    } else {
        return None;
    })
}

impl FormRules {
    /// The flags of a form's op.
    pub fn flags(&self, name: &str) -> Result<Flags, Error> {
        let mut f = unit_of(name, self.matrix, self.smem)
            .ok_or_else(|| Error::Definition(format!("{name}: no unit")))?;
        let has = |s: &str| name.contains(s);
        f |= when(name == "s_branch", Flags::UNCOND);
        f |= when(name == "s_endpgm", Flags::ENDPGM | Flags::UNCOND);
        f |= when(name == "s_nop", Flags::NOP);
        f |= when(name == self.barrier, Flags::BARRIER);
        f |= when(starts(name, self.waits), Flags::WAITCNT);
        let memory = f.any(Flags::VMEM | Flags::DS | Flags::SMEM);
        f |= when(memory && has("atomic"), Flags::ATOMIC);
        f |= when(
            memory && (has("store") || starts(name, &["ds_write", "ds_cmpst"])),
            Flags::STORE,
        );
        f |= when(
            memory && (has("load") || starts(name, &["ds_read"])),
            Flags::LOAD,
        );
        let fp = ["_f32", "_bf16", "_f16", "_f64"].iter().any(|s| has(s));
        let fp_atomic = starts(name, &["buffer_atomic", "global_atomic"]) && fp;
        f |= when(f.has(Flags::ATOMIC) && fp_atomic, Flags::FP_ATOMIC);
        let operation = base(name.strip_suffix("_e32").unwrap_or(name));
        f |= when(
            f.has(Flags::VALU) && self.trans.contains(&operation),
            Flags::TRANS,
        );
        f |= when(name.starts_with("v_dot"), Flags::DOT);
        f |= when(f.has(Flags::VALU) && has("_dpp"), Flags::DPP);
        f |= when(has("d16"), Flags::D16);
        f |= when(has("permlane"), Flags::PERMLANE);
        f |= when(
            starts(name, &["v_readlane", "v_readfirstlane"]),
            Flags::READLANE,
        );
        f |= when(name.starts_with("v_writelane"), Flags::WRITELANE);
        f |= when(starts(name, self.wave_varying), Flags::WAVE_VARYING);
        let not_lds = ["ds_bpermute", "ds_permute", "ds_swizzle", "ds_nop"];
        f |= when(f.has(Flags::DS) && !starts(name, &not_lds), Flags::LDS_MEM);
        f |= when(reads_dst(name), Flags::READS_DST);
        let accumulates = starts(name, &["v_fmac", "v_fma_mix"]);
        f |= when(accumulates, Flags::ACCUMULATE);
        f |= when(has("saveexec"), Flags::SAVEEXEC);
        if name.starts_with("v_cmpx") {
            f |= Flags::CMPX | Flags::CMP;
        } else if name.starts_with("v_cmp") {
            f |= Flags::CMP;
        }
        f |= when(starts(name, &["s_cmp_", "s_bitcmp"]), Flags::SCC_CMP);
        f |= when(name.starts_with("s_cselect"), Flags::SCC_SELECT);
        f |= when(self.carry_in.contains(&name), Flags::SCC_CARRY_IN);
        f |= when(starts(name, &["s_add_", "s_sub_"]), Flags::SCC_CARRY_OUT);
        let swizzles = starts(name, &["ds_bpermute", "ds_permute", "ds_swizzle"]);
        let cross = f.any(Flags::MATRIX | Flags::DPP) || swizzles;
        f |= when(cross, Flags::CROSS_LANE);
        Ok(f)
    }

    /// A form's kind.
    pub fn kind(&self, name: &str, flags: Flags, waits: &[WaitInsn]) -> Kind {
        if waits.iter().any(|w| w.mnemonic == name) {
            return Kind::Wait;
        }
        if name == "s_nop" {
            return Kind::Nop;
        }
        if name == self.barrier {
            return Kind::Barrier;
        }
        [
            (Flags::MATRIX, Kind::Matrix),
            (Flags::VALU, Kind::Valu),
            (Flags::BRANCH, Kind::Branch),
            (Flags::SMEM, Kind::Smem),
            (Flags::DS, Kind::Ds),
            (Flags::VMEM, Kind::Vmem),
        ]
        .iter()
        .find(|(flag, _)| flags.has(*flag))
        .map_or(Kind::Salu, |&(_, kind)| kind)
    }

    /// Every form of an operand table, in row order.
    pub fn forms(&self, rows: &[Row], waits: &[WaitInsn]) -> Result<Vec<Form>, Error> {
        let index: HashMap<(&str, usize), &Row> = rows
            .iter()
            .map(|r| ((r.mnemonic.as_str(), r.count), r))
            .collect();
        rows.iter()
            .enumerate()
            .map(|(id, row)| self.form(id, row, &index, waits))
            .collect()
    }

    fn form(
        &self,
        id: usize,
        row: &Row,
        index: &HashMap<(&str, usize), &Row>,
        waits: &[WaitInsn],
    ) -> Result<Form, Error> {
        let returning = row.mnemonic.ends_with("+rtn");
        let op = row.mnemonic.trim_end_matches("+rtn");
        let writes = row.roles.first().is_some_and(|r| r.contains('w'));
        let flags = self.flags(op)?;
        let kind = self.kind(op, flags, waits);
        let mut form = Form::plain(id as i32, op, kind, flags);
        form.name = row.mnemonic.clone();
        form.base = base(op).to_owned();
        form.counter = (self.counter)(flags, op, writes);
        // Scalar loads return out of order, and so does an op that counts
        // without being memory (a returning message) among them.
        let memory = flags.any(Flags::DS | Flags::VMEM | Flags::SMEM);
        form.unordered = flags.has(Flags::SMEM) || (form.counter.is_some() && !memory);
        form.encoding = row.encoding.clone();
        form.arity = row.count;
        form.returning = returning;
        form.slots = self.slots(op, row, index, writes)?;
        let nsrc = row.count - usize::from(writes);
        form.fields = (self.fields)(op, &row.encoding, nsrc)?;
        form.implicit_defs = registers(&row.defs)?;
        form.implicit_uses = registers(&row.uses)?;
        let memory = flags.any(Flags::DS | Flags::VMEM | Flags::SMEM);
        form.mem_bytes = if memory { mem_bytes(op) } else { 0 };
        form.dpp = op.contains("_dpp");
        form.dst_sel = dst_sel(flags, op);
        let first = form.slots.first().filter(|s| s.role & 2 != 0);
        let vgpr = first.is_some_and(|s| s.kind == SlotKind::Vgpr);
        let bits = row.slots.first().map_or(0, |s| s.1);
        form.dst_half = dst_half(flags, op, bits, vgpr);
        form.mem = mem_of_encoding(&row.encoding);
        form.wait_text = waits
            .iter()
            .find(|w| w.mnemonic == op && row.count == 0)
            .map(|w| w.text);
        Ok(form)
    }

    fn slots(
        &self,
        op: &str,
        row: &Row,
        index: &HashMap<(&str, usize), &Row>,
        writes: bool,
    ) -> Result<Vec<Slot>, Error> {
        let short = ["ENC_VOP1", "ENC_VOP2", "ENC_VOPC"].contains(&row.encoding.as_str());
        let wide = short
            .then(|| {
                index
                    .get(&(format!("{op}_e64").as_str(), row.count))
                    .copied()
            })
            .flatten();
        let vmem = [
            "ENC_VBUFFER",
            "ENC_MUBUF",
            "ENC_VFLAT",
            "ENC_VGLOBAL",
            "ENC_VSCRATCH",
            "ENC_FLAT",
        ];
        let buffer = row.encoding == "ENC_VBUFFER" || row.encoding == "ENC_MUBUF";
        let address = if buffer { 1 } else { usize::from(writes) };
        let signs = (self.signs)(op);
        let mut slots = Vec::with_capacity(row.slots.len());
        for (k, (xml_type, size)) in row.slots.iter().enumerate() {
            let mut kind = slot_kind(xml_type)?;
            if let Some(wide) = wide {
                let other = wide.slots.get(k).map(|(t, _)| slot_kind(t)).transpose()?;
                if kind == SlotKind::Vgpr && other.is_some_and(|o| o != kind) {
                    kind = SlotKind::Vsrc;
                }
            }
            let address_slot = xml_type == "OPR_VGPR"
                && *size == 64
                && starts(&row.encoding, &vmem)
                && k == address;
            let literal = ![
                "_NOLIT",
                "OPR_SRC_SIMPLE",
                "OPR_SRC_VGPR",
                "_ACCVGPR",
                "_LANESEL",
            ]
            .iter()
            .any(|s| xml_type.contains(s));
            slots.push(Slot {
                kind,
                dwords: (size / 32).max(1),
                role: match row.roles.get(k).map(String::as_str) {
                    Some("r") => 1,
                    Some("w") => 2,
                    Some("rw") => 3,
                    _ => 0,
                },
                xml_type: xml_type.clone(),
                sgpr_widths: row.sgpr_widths.get(k).cloned().unwrap_or_default(),
                modifiers: signs.contains(&k),
                literal,
                widths: if address_slot {
                    vec![1, 2]
                } else {
                    (self.widths)(op, k)
                },
                limit: None,
            });
        }
        Ok(slots)
    }
}

/// Named registers by number, sorted, refusing a name outside
fn registers(names: &[String]) -> Result<Vec<usize>, Error> {
    let mut numbers = names
        .iter()
        .map(|n| {
            phys_named(n)
                .map(|p| p.number)
                .ok_or_else(|| Error::Definition(format!("{n} is not a named register")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    numbers.sort_unstable();
    Ok(numbers)
}
