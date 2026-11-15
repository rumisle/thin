//! The budget check, residency, the mix and the notes.

use std::collections::BTreeMap;

use thin_target::Target;

/// What residency says of a kernel: WGs per CU, what limits it, and each
/// limit (`vgpr`, `lds`, `waves`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residency {
    /// Workgroups resident per CU.
    pub resident: u32,
    /// `LDS`, `VGPRs` or `wave slots`.
    pub limiter: &'static str,
    /// By VGPRs and by LDS (none for a kernel that uses none), by wave
    /// slots.
    pub limits: (Option<u32>, Option<u32>, u32),
}

/// The budget check: the over-budget items, `"<name>: over
/// budget:..."` when any.
pub fn check_budget(
    t: &Target,
    name: &str,
    lds: u64,
    vgpr: u32,
    agpr: u32,
    sgpr: u32,
    wgsize: u32,
) -> Result<(), String> {
    let d = t.definition();
    let c = &d.chip;
    let waves = wgsize.div_ceil(d.wave);
    let per_simd = waves.div_ceil(c.simds).max(1);
    let total = t.allocated_vgprs(vgpr, agpr);
    let mut items = Vec::new();
    if vgpr > c.vgprs {
        items.push(format!("{vgpr} arch VGPRs (max {})", c.vgprs));
    }
    if agpr > c.vgprs {
        items.push(format!("{agpr} AGPRs (max {})", c.vgprs));
    }
    let most = c.vgprs_per_lane / per_simd;
    if total > most {
        items.push(format!(
            "{total} VGPRs + AGPRs per wave, {per_simd} waves per SIMD (WG {wgsize}): max {most}"
        ));
    }
    if sgpr > c.sgprs {
        items.push(format!("{sgpr} SGPRs (max {})", c.sgprs));
    }
    if lds > u64::from(d.lds_max) {
        items.push(format!("{lds} B LDS (max {})", d.lds_max));
    }
    match items.is_empty() {
        true => Ok(()),
        false => Err(format!("{name}: over budget: {}", items.join("; "))),
    }
}

/// Residency: the workgroups a CU holds, 0 for a kernel none fits
/// (the budget check refuses those).
pub fn residency(t: &Target, lds: u64, vgpr: u32, agpr: u32, wgsize: u32) -> Residency {
    let d = t.definition();
    let c = &d.chip;
    let waves = wgsize.div_ceil(d.wave).max(1);
    let total = t.allocated_vgprs(vgpr, agpr);
    let by_vgpr = (total > 0).then(|| c.simds * (c.vgprs_per_lane / total) / waves);
    let by_lds = (lds > 0).then(|| (u64::from(c.lds) / lds) as u32);
    let by_waves = c.simds * c.wave_slots / waves;
    let least = [by_vgpr, by_lds]
        .into_iter()
        .flatten()
        .fold(by_waves, u32::min);
    let limiter = match () {
        _ if by_lds == Some(least) => "LDS",
        _ if by_vgpr == Some(least) => "VGPRs",
        _ => "wave slots",
    };
    Residency {
        resident: least,
        limiter,
        limits: (by_vgpr, by_lds, by_waves),
    }
}

/// The mix: counts by kind name, the target's delay hint split out
/// of `salu` into `delay`; `kinds` is each instruction's (kind, mnemonic).
pub fn mix(t: &Target, kinds: &[(&str, &str)]) -> BTreeMap<String, u64> {
    let hint = t.definition().hazards.delay_hint;
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    for &(kind, op) in kinds {
        if kind == "label" || kind == "comment" {
            continue;
        }
        let key = if Some(op) == hint { "delay" } else { kind };
        *out.entry(key.to_owned()).or_default() += 1;
        if key == "delay" {
            out.entry("salu".to_owned()).or_default();
        }
    }
    out
}

/// One instruction as the delay notes read it.
pub struct Line<'a> {
    /// Its kind's name.
    pub kind: &'a str,
    /// Its mnemonic.
    pub op: &'a str,
    /// A label's name.
    pub label: Option<&'a str>,
    /// A branch's first block operand.
    pub target: Option<&'a str>,
    /// A delay hint's operand; 0 for any other instruction.
    pub hint: i128,
    /// Whether it is the target's delay hint.
    pub delay: bool,
}

/// The notes, in order.
pub fn notes(
    t: &Target,
    r: &Residency,
    mix: &BTreeMap<String, u64>,
    lines: &[Line<'_>],
) -> Vec<String> {
    let mut out = Vec::new();
    if r.resident <= 4 {
        let wgs = if r.resident == 1 { "WG" } else { "WGs" };
        let cus = t.definition().chip.cus;
        out.push(format!(
            "{} {wgs} per CU ({}): a grid past {} WGs runs in rounds",
            r.resident,
            r.limiter,
            r.resident * cus
        ));
    }
    let get = |k: &str| mix.get(k).copied().unwrap_or(0);
    let (salu, vector) = (get("salu"), get("valu") + get("matrix"));
    if salu > vector && vector > 0 {
        out.push(format!(
            "SALU {salu} > VALU + MFMA {vector} over the whole kernel: where the hot loop has this mix, the CU's \
             one scalar issue per cycle is the bound once ~4 waves share it (an in-kernel clock says whether it is \
             the loop)"
        ));
    }
    let by = t
        .definition()
        .hazards
        .delay_hint
        .unwrap_or("the delay hint");
    out.extend(delay_notes(lines, by));
    out
}

/// The innermost loops: (label index, branch index, name).
fn loops(lines: &[Line<'_>]) -> Vec<(usize, usize, String)> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut all = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        if let Some(name) = l.label {
            seen.insert(name, k);
        }
        if l.kind == "branch" {
            if let Some(&a) = l.target.and_then(|t| seen.get(t)) {
                all.push((a, k, l.target.unwrap_or("").to_owned()));
            }
        }
    }
    let inner =
        |x: &(usize, usize, String)| !all.iter().any(|y| y != x && x.0 <= y.0 && y.1 <= x.1);
    all.iter().filter(|x| inner(x)).cloned().collect()
}

fn delay_notes(lines: &[Line<'_>], by: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (a, z, name) in loops(lines) {
        let (mut hint, mut valu, mut told, mut near) = (0i128, 0, 0, 0);
        for l in &lines[a..z] {
            if l.delay {
                hint = l.hint;
                continue;
            }
            if l.kind == "valu" || l.kind == "matrix" {
                valu += 1;
                if hint != 0 {
                    told += 1;
                    near += i32::from(hint & 15 == 1);
                }
            }
            if l.kind != "label" && l.kind != "comment" {
                hint = 0;
            }
        }
        if told > 0 {
            let short = name.strip_prefix(".L").unwrap_or(&name);
            out.push(format!(
                "loop {short}: {told} of its {valu} VALU ops wait on a recent VALU op's result ({near} on the op \
                 just before): told by {by}, the other waves issue meanwhile, but ops independent of the few \
                 before them need no wait"
            ));
        }
    }
    out
}
