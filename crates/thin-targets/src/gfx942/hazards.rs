//! gfx942's hazard model: CDNA3 architectural wait-state rules, including
//! pass-count- and family-aware XDL / SMFMA / DGEMM / SMFMAC wait states.

use std::sync::OnceLock;

use thin_ir::{Imm, ImmValue, Instr, Operand, Reg, Snapshot};
use thin_passes::hazards::{Dep, DynamicStates, HazardModel, Rule, Side, States, Test};
use thin_target::{phys_named, Elem, Flags, Kind};

use super::mfma_form;

/// The largest wait-state count among [`rules()`] (16-pass XDL -> VALU/VMEM/DS/SrcAB = 19).
pub const WINDOW: u32 = 19;

/// gfx942's hazard model.
pub struct Model;

impl HazardModel for Model {
    fn rules(&self) -> &[Rule] {
        rules()
    }
}

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(build_rules)
}

/// Number of wait-state rules in gfx942's hazard table.
pub fn rule_count() -> u32 {
    rules().len() as u32
}

/// CDNA3 matrix execution pipeline family and hardware pass count (`2, 4, 8, 16`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatClass {
    /// CDNA3 XDL pipeline (`f16`, `bf16`, `xf32`, `fp8`/`bf8`, `i8`, and all `v_smfmac_*`).
    Xdl(u32),
    /// Legacy SGEMM MAI pipeline (`f32`-input `v_mfma_f32_*_f32`).
    Smfma(u32),
    /// Double-precision DGEMM pipeline (`v_mfma_f64_*_f64`, `4` or `8` passes).
    Dgemm(u32),
}

impl MatClass {
    fn of(i: &Instr) -> Option<(MatClass, bool)> {
        let mf = mfma_form(&i.form().op)?;
        let p = mf.passes;
        let cls = if mf.d == Elem::F64 {
            MatClass::Dgemm(p)
        } else if mf.a == Elem::F32 {
            MatClass::Smfma(p)
        } else {
            MatClass::Xdl(p)
        };
        Some((cls, mf.sparse))
    }

    fn passes(self) -> u32 {
        match self {
            MatClass::Xdl(p) | MatClass::Smfma(p) | MatClass::Dgemm(p) => p,
        }
    }
}

fn no_late(_: &Snapshot, _: &Instr) -> u32 {
    0
}

fn is_dot(i: &Instr) -> bool {
    i.form().op.starts_with("v_dot")
}

fn valu_to_mat(_: &Snapshot, prev: &Instr, _: &Instr) -> u32 {
    if is_dot(prev) {
        0
    } else {
        2
    }
}

/// Matrix -> VALU RAW/WAW and Matrix -> Matrix non-SrcC (`SrcA`/`SrcB`/`Idx`) RAW.
fn mat_to_valu_or_ab(_: &Snapshot, prev: &Instr, _: &Instr) -> u32 {
    match MatClass::of(prev).map(|x| x.0) {
        Some(MatClass::Xdl(p)) => p + 3,
        Some(MatClass::Smfma(p)) => p + 2,
        Some(MatClass::Dgemm(4)) => 6,
        Some(MatClass::Dgemm(_)) => 11,
        None => 0,
    }
}

/// Matrix -> VMEM / DS RAW (`DGEMM` takes 9 / 18 states; `XDL` / `SMFMA` match VALU).
fn mat_to_mem(_: &Snapshot, prev: &Instr, _: &Instr) -> u32 {
    match MatClass::of(prev).map(|x| x.0) {
        Some(MatClass::Xdl(p)) => p + 3,
        Some(MatClass::Smfma(p)) => p + 2,
        Some(MatClass::Dgemm(4)) => 9,
        Some(MatClass::Dgemm(_)) => 18,
        None => 0,
    }
}

fn reg_extent(s: &Snapshot, r: &Reg) -> (char, u32, u32) {
    (r.file, s.phys(r), r.n)
}

fn extents_overlap(a: (char, u32, u32), b: (char, u32, u32)) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

fn vcc_num() -> usize {
    phys_named("vcc").expect("vcc").number
}

fn exec_num() -> usize {
    phys_named("exec").expect("exec").number
}

fn valu_writes_scalar(i: &Instr) -> bool {
    i.form().kind == Kind::Valu
        && (i.sgpr_valu_def() || i.writes_exec() || i.writes().contains(&vcc_num()))
}

fn valu_sgpr_overlap(s: &Snapshot, prev: &Instr, cur: &Instr) -> bool {
    if !valu_writes_scalar(prev) {
        return false;
    }
    let sgpr_hit = prev.defs().filter(|r| r.file == 's').any(|d| {
        let de = reg_extent(s, d);
        cur.uses()
            .filter(|u| u.file == 's')
            .any(|u| extents_overlap(de, reg_extent(s, u)))
    });
    if sgpr_hit {
        return true;
    }
    let vcc = vcc_num();
    if prev.writes().contains(&vcc) && cur.reads().contains(&vcc) {
        return true;
    }
    // Explicit EXEC / EXEC_LO / EXEC_HI constant read (excluding implicit VALU lane-mask use).
    let exec = exec_num();
    prev.writes_exec()
        && cur.ops().iter().enumerate().any(|(k, op)| {
            (cur.roles()[k] & 1 != 0) && op.whole().is_some_and(|p| p.number == exec)
        })
}

/// CDNA3 Table 11: VALU writing SGPR/VCC/EXEC -> VALU/Matrix reading it:
/// - `v_div_fmas_*` reading `VCC`: 4 wait states (`s_nop 3`)
/// - `v_readlane_b32` / `v_writelane_b32` reading `src1` lane-select SGPR/VCC: 4 wait states (`s_nop 3`)
/// - Other VALU / Matrix SGPR, `VCC`, or explicit `EXEC` constant read: 2 wait states (`s_nop 1`)
fn valu_sgpr_to_valu_or_mat(s: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    if !valu_writes_scalar(prev) {
        return 0;
    }
    let vcc = vcc_num();
    let cur_op = cur.form().op.as_str();
    if cur_op.starts_with("v_div_fmas_") && prev.writes().contains(&vcc) {
        return 4;
    }
    if matches!(cur_op, "v_readlane_b32" | "v_writelane_b32") {
        if let Some(lane_sel) = cur.ops().get(2) {
            let sel_hit = match lane_sel {
                Operand::Reg(u) if u.file == 's' => {
                    let ue = reg_extent(s, u);
                    prev.defs()
                        .filter(|r| r.file == 's')
                        .any(|d| extents_overlap(reg_extent(s, d), ue))
                }
                op => op.whole().is_some_and(|p| {
                    (p.number == vcc && prev.writes().contains(&vcc))
                        || (p.number == exec_num() && prev.writes_exec())
                }),
            };
            if sel_hit {
                return 4;
            }
        }
    }
    if valu_sgpr_overlap(s, prev, cur) {
        2
    } else {
        0
    }
}

/// CDNA3 Table 11: VALU writing SGPR/VCC/EXEC -> VMEM/SMEM reading it: 5 wait states (`s_nop 4`).
fn valu_sgpr_to_mem(s: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    if valu_sgpr_overlap(s, prev, cur) {
        5
    } else {
        0
    }
}

fn hwreg_code(op: &Operand) -> Option<u32> {
    match op {
        Operand::Imm(Imm {
            value: ImmValue::Int(v),
            ..
        }) => Some((*v as u32) & 0xFFFF),
        _ => None,
    }
}

fn setreg_info(i: &Instr) -> Option<(u32, bool)> {
    match i.form().op.as_str() {
        "s_setreg_b32" | "s_setreg_imm32_b32" => {
            let code = hwreg_code(i.ops().first()?)?;
            let id = code & 0x3F;
            let offset = (code >> 6) & 0x1F;
            let size = ((code >> 11) & 0x1F) + 1;
            let touches_vskip = id == 1 && offset <= 28 && 28 < offset + size;
            Some((id, touches_vskip))
        }
        "s_setvskip" => Some((1, false)),
        _ => None,
    }
}

fn setreg_to_salu(_: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    let Some((p_id, _)) = setreg_info(prev) else {
        return 0;
    };
    let cur_op = cur.form().op.as_str();
    if p_id == 3 && cur_op == "s_rfe_b64" {
        return 1;
    }
    let c_id = match cur_op {
        "s_getreg_b32" => cur.ops().get(1).and_then(hwreg_code).map(|c| c & 0x3F),
        "s_setreg_b32" | "s_setreg_imm32_b32" if prev.form().op != "s_setvskip" => {
            cur.ops().first().and_then(hwreg_code).map(|c| c & 0x3F)
        }
        _ => None,
    };
    if c_id == Some(p_id) {
        2
    } else {
        0
    }
}

fn setreg_vskip_to_vec(_: &Snapshot, prev: &Instr, _: &Instr) -> u32 {
    if setreg_info(prev).is_some_and(|(_, vskip)| vskip) {
        2
    } else {
        0
    }
}

/// Partial-overlap or cross-opcode `SrcC` wait states (`getMFMAOverlappedSrcCWaitStates`).
fn overlapped_src_c(prev: MatClass, cur: MatClass) -> u32 {
    match (prev, cur) {
        (MatClass::Smfma(_) | MatClass::Dgemm(_), MatClass::Xdl(_)) => 0,
        (MatClass::Dgemm(4), _) => 4,
        (MatClass::Dgemm(_), _) => 9,
        (MatClass::Smfma(p), _) => p,
        (MatClass::Xdl(p), _) => p + 1,
    }
}

/// Matrix -> Matrix `SrcC` RAW (`getMFMAReadWaitStates` with `IsSrcC = true`).
fn mat_to_src_c(s: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    let (Some((p_cls, _)), Some((c_cls, c_sparse))) = (MatClass::of(prev), MatClass::of(cur))
    else {
        return 0;
    };
    let c_idx = if c_sparse { 0 } else { 3 };
    let full_reg = match (
        prev.ops().first().and_then(Operand::reg),
        cur.ops().get(c_idx).and_then(Operand::reg),
    ) {
        (Some(pd), Some(cc)) => reg_extent(s, pd) == reg_extent(s, cc),
        _ => false,
    };
    if full_reg {
        let base = if matches!((p_cls, c_cls), (MatClass::Dgemm(4), MatClass::Dgemm(4))) {
            4
        } else if p_cls.passes() == 2 {
            2
        } else {
            0
        };
        if prev.form().op == cur.form().op {
            base
        } else {
            base.max(overlapped_src_c(p_cls, c_cls))
        }
    } else {
        overlapped_src_c(p_cls, c_cls)
    }
}

/// Late `SrcC` read -> VALU/VMEM/DS write WAR (`SMFMAReadAsCWindow` on CDNA3: XDL only, `passes - 1`).
fn mat_src_c_late(_: &Snapshot, prev: &Instr) -> u32 {
    match MatClass::of(prev) {
        Some((MatClass::Xdl(p), false)) => p.saturating_sub(1),
        _ => 0,
    }
}

fn mat_src_c_war(s: &Snapshot, prev: &Instr, _: &Instr) -> u32 {
    mat_src_c_late(s, prev)
}

/// `v_mfma_f64_*` -> `v_fma_f64` / `v_fmac_f64` pipeline hazard (2 wait states regardless of regs).
fn dgemm_to_fma64(_: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    if !matches!(MatClass::of(prev), Some((MatClass::Dgemm(_), _))) {
        return 0;
    }
    let op = cur.form().op.as_str();
    if op == "v_fma_f64" || op.starts_with("v_fmac_f64") {
        2
    } else {
        0
    }
}

fn dot_same_non_src_c(_: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    if is_dot(prev) && prev.form().op == cur.form().op {
        3
    } else {
        0
    }
}

fn dot_diff(_: &Snapshot, prev: &Instr, cur: &Instr) -> u32 {
    if is_dot(prev) && prev.form().op != cur.form().op {
        3
    } else {
        0
    }
}

fn side(kinds: &[Kind], all: Flags, any: Flags, tests: &[Test]) -> Side {
    Side {
        kinds: kinds.to_vec(),
        all,
        any,
        tests: tests.to_vec(),
    }
}

fn rule(prev: Side, cur: Side, dep: Dep, states: u32) -> Rule {
    Rule {
        prev,
        cur,
        dep,
        states: States::Const(states),
    }
}

fn dyn_rule(
    prev: Side,
    cur: Side,
    dep: Dep,
    max: u32,
    pair: fn(&Snapshot, &Instr, &Instr) -> u32,
    late: fn(&Snapshot, &Instr) -> u32,
) -> Rule {
    Rule {
        prev,
        cur,
        dep,
        states: States::Dynamic(DynamicStates { max, pair, late }),
    }
}

/// CDNA3 architectural wait-state rules (AMD Instinct CDNA3 ISA Guide,
/// Chapter 4 "Program Organization / Wait States", cross-checked against
/// `GCNHazardRecognizers.cpp` and `llc -mcpu=gfx942`).
fn build_rules() -> Vec<Rule> {
    let salu = || side(&[Kind::Salu], Flags::NONE, Flags::NONE, &[]);
    let valu = || side(&[Kind::Valu], Flags::NONE, Flags::NONE, &[]);
    let valu_or_mat = || side(&[Kind::Valu, Kind::Matrix], Flags::NONE, Flags::NONE, &[]);
    let mat = || side(&[Kind::Matrix], Flags::NONE, Flags::NONE, &[]);
    let vmem_ds = || side(&[Kind::Vmem, Kind::Ds], Flags::NONE, Flags::NONE, &[]);
    let valu_vmem_ds = || {
        side(
            &[Kind::Valu, Kind::Vmem, Kind::Ds],
            Flags::NONE,
            Flags::NONE,
            &[],
        )
    };
    let any_vec = || {
        side(
            &[Kind::Valu, Kind::Matrix, Kind::Vmem, Kind::Ds],
            Flags::NONE,
            Flags::NONE,
            &[],
        )
    };
    vec![
        // 1. VALU writing SGPR/VCC/EXEC -> VALU/Matrix reading it:
        //    4 wait states on v_div_fmas (VCC) or v_readlane/v_writelane (SGPR lane-select),
        //    2 wait states on general SGPR/VCC/explicit-EXEC constant read.
        dyn_rule(
            valu(),
            valu_or_mat(),
            Dep::None,
            4,
            valu_sgpr_to_valu_or_mat,
            no_late,
        ),
        // 1b. VALU writing SGPR/VCC -> VMEM/SMEM reading SGPR (SADDR/SRSRC/SOFFSET): 5 wait states (S_NOP 4).
        dyn_rule(
            valu(),
            side(&[Kind::Vmem, Kind::Smem], Flags::NONE, Flags::NONE, &[]),
            Dep::None,
            5,
            valu_sgpr_to_mem,
            no_late,
        ),
        // 1c. VALU -> READLANE / WRITELANE RAW on VGPR: 1 wait state (S_NOP 0).
        rule(
            valu(),
            side(
                &[Kind::Valu],
                Flags::NONE,
                Flags::READLANE | Flags::WRITELANE,
                &[],
            ),
            Dep::Raw,
            1,
        ),
        // 2a. s_setreg / s_setvskip -> s_getreg / s_setreg on same HWReg (2 states) or s_rfe_b64 on TRAPSTS (1 state).
        dyn_rule(salu(), salu(), Dep::None, 2, setreg_to_salu, no_late),
        // 2b. s_setreg MODE.vskip -> any vector op: 2 wait states (S_NOP 1).
        dyn_rule(
            salu(),
            any_vec(),
            Dep::None,
            2,
            setreg_vskip_to_vec,
            no_late,
        ),
        // 3a. VALU writing EXEC (v_cmpx) -> DPP op: 5 wait states (S_NOP 4).
        rule(
            side(&[Kind::Valu], Flags::CMPX, Flags::NONE, &[]),
            side(&[Kind::Valu], Flags::DPP, Flags::NONE, &[]),
            Dep::None,
            5,
        ),
        // 3b. VALU writing EXEC (v_cmpx) -> v_readlane / v_readfirstlane / v_writelane: 4 wait states (S_NOP 3).
        rule(
            side(&[Kind::Valu], Flags::CMPX, Flags::NONE, &[]),
            side(
                &[Kind::Valu],
                Flags::NONE,
                Flags::READLANE | Flags::WRITELANE,
                &[],
            ),
            Dep::None,
            4,
        ),
        // 3c. VALU writing EXEC (v_cmpx) -> Matrix (MFMA / SMFMAC): 4 wait states (S_NOP 3).
        rule(
            side(&[Kind::Valu], Flags::CMPX, Flags::NONE, &[]),
            mat(),
            Dep::None,
            4,
        ),
        // 4. VALU -> DPP RAW: 2 wait states (S_NOP 1).
        rule(
            valu(),
            side(&[Kind::Valu], Flags::DPP, Flags::NONE, &[]),
            Dep::Raw,
            2,
        ),
        // 5. Transcendental VALU -> non-transcendental VALU RAW: 1 wait state (S_NOP 0).
        rule(
            side(&[Kind::Valu], Flags::TRANS, Flags::NONE, &[]),
            valu(),
            Dep::Raw,
            1,
        ),
        // 6. Partial-dst VALU (SDWA / D16 / op_sel hi-half / FP8 byte-sel) -> VALU RAW or WAW: 1 wait state (S_NOP 0).
        rule(
            side(&[Kind::Valu], Flags::NONE, Flags::NONE, &[Test::PartialDst]),
            valu(),
            Dep::RawOrWaw,
            1,
        ),
        // 7. SALU writing M0 -> DS / VMEM / SALU / VALU reading M0: 1 wait state (S_NOP 0).
        rule(
            side(&[], Flags::NONE, Flags::NONE, &[Test::WritesM0]),
            side(
                &[Kind::Ds, Kind::Vmem, Kind::Salu, Kind::Valu],
                Flags::NONE,
                Flags::NONE,
                &[],
            ),
            Dep::Raw,
            1,
        ),
        // 8. Legacy VALU (non-dot) writing AGPR/VGPR -> Matrix reading it: 2 wait states (S_NOP 1).
        dyn_rule(valu(), mat(), Dep::Raw, 2, valu_to_mat, no_late),
        // 8b. v_dot* -> same v_dot* reading SrcA/SrcB (3 states) or different VALU/VMEM/DS RAW/WAW (3 states).
        dyn_rule(
            valu(),
            valu(),
            Dep::ReadsNotAsSrcC,
            3,
            dot_same_non_src_c,
            no_late,
        ),
        dyn_rule(valu(), valu_vmem_ds(), Dep::RawOrWaw, 3, dot_diff, no_late),
        // 9. Matrix -> VALU RAW or WAW: pass-count-aware (XDL: p+3, SMFMA: p+2, DGEMM: 6/11).
        dyn_rule(
            mat(),
            valu(),
            Dep::RawOrWaw,
            WINDOW,
            mat_to_valu_or_ab,
            no_late,
        ),
        // 9b. Matrix -> VMEM / DS RAW: pass-count-aware (XDL: p+3, SMFMA: p+2, DGEMM: 9/18).
        dyn_rule(mat(), vmem_ds(), Dep::Raw, WINDOW, mat_to_mem, no_late),
        // 9c. DGEMM -> v_fma_f64 / v_fmac_f64 pipeline hazard: 2 wait states.
        dyn_rule(mat(), valu(), Dep::None, 2, dgemm_to_fma64, no_late),
        // 10. Matrix -> Matrix reading result as SrcA/SrcB/Idx (non-SrcC): pass-count-aware.
        dyn_rule(
            mat(),
            mat(),
            Dep::ReadsNotAsSrcC,
            WINDOW,
            mat_to_valu_or_ab,
            no_late,
        ),
        // 11. Matrix -> Matrix reading result as SrcC: 0 on full same-op chain (2 on 2-pass, 4 on 4x4 f64),
        //     pass-count-aware on cross-opcode or partial register overlap.
        dyn_rule(mat(), mat(), Dep::ReadsAsSrcC, 17, mat_to_src_c, no_late),
        // 12. XDL Matrix reading SrcC late -> VALU/VMEM/DS overwriting SrcC: p-1 wait states (0 on SMFMA/DGEMM).
        dyn_rule(
            mat(),
            valu_vmem_ds(),
            Dep::WritesSrcC,
            15,
            mat_src_c_war,
            mat_src_c_late,
        ),
        // 13. Wide store (>2 dwords) without SGPR soffset -> VALU overwriting store data: 2 wait states (S_NOP 1).
        rule(
            side(
                &[Kind::Vmem, Kind::Ds],
                Flags::STORE,
                Flags::NONE,
                &[Test::SgprSoffset(false)],
            ),
            valu(),
            Dep::WritesWideData,
            2,
        ),
    ]
}
