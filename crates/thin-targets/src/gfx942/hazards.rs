//! gfx942's hazard model: CDNA3 architectural wait-state rules.

use std::sync::OnceLock;

use thin_passes::hazards::{Dep, HazardModel, Rule, Side, States, Test};
use thin_target::{Flags, Kind};

/// The largest wait-state count among [` rules()`].
pub const WINDOW: u32 = 18;

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

/// CDNA3 architectural wait-state rules (AMD Instinct CDNA3 ISA Guide,
/// Chapter 4 "Program Organization / Wait States", cross-checked against
/// `llc -mcpu=gfx942 -run-pass=post-RA-hazard-rec`).
fn build_rules() -> Vec<Rule> {
    let valu = || side(&[Kind::Valu], Flags::NONE, Flags::NONE, &[]);
    let valu_or_mat = || side(&[Kind::Valu, Kind::Matrix], Flags::NONE, Flags::NONE, &[]);
    let mat = || side(&[Kind::Matrix], Flags::NONE, Flags::NONE, &[]);
    let vmem_ds = || side(&[Kind::Vmem, Kind::Ds], Flags::NONE, Flags::NONE, &[]);
    vec![
        // 1. VALU writing SGPR/VCC -> VALU/Matrix reading it: 2 wait states (S_NOP 1).
        rule(
            side(
                &[Kind::Valu],
                Flags::NONE,
                Flags::NONE,
                &[Test::SgprValuDef],
            ),
            valu_or_mat(),
            Dep::Raw,
            2,
        ),
        // 1b. VALU writing SGPR -> VMEM/SMEM reading SGPR (SADDR/SRSRC/SOFFSET): 5 wait states (S_NOP 4).
        rule(
            side(
                &[Kind::Valu],
                Flags::NONE,
                Flags::NONE,
                &[Test::SgprValuDef],
            ),
            side(&[Kind::Vmem, Kind::Smem], Flags::NONE, Flags::NONE, &[]),
            Dep::Raw,
            5,
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
        // 2. VALU carry-out (VCC) -> VALU reading VCC: 2 wait states (S_NOP 1).
        rule(
            side(&[Kind::Valu], Flags::NONE, Flags::CMP, &[]),
            valu_or_mat(),
            Dep::Raw,
            2,
        ),
        // 3. VALU writing EXEC (v_cmpx) -> DPP / lane op: 5 wait states (S_NOP 4).
        rule(
            side(&[Kind::Valu], Flags::CMPX, Flags::NONE, &[]),
            side(
                &[Kind::Valu],
                Flags::NONE,
                Flags::DPP | Flags::READLANE | Flags::WRITELANE,
                &[],
            ),
            Dep::None,
            5,
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
        // 6. Partial-dst VALU (SDWA / D16) -> VALU RAW: 1 wait state (S_NOP 0).
        rule(
            side(&[Kind::Valu], Flags::NONE, Flags::NONE, &[Test::PartialDst]),
            valu(),
            Dep::Raw,
            1,
        ),
        // 7. SALU writing M0 -> DS / VMEM / SALU reading M0: 1 wait state (S_NOP 0).
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
        // 8. VALU writing AGPR/VGPR -> Matrix (MFMA / SMFMAC) reading it: 2 wait states (S_NOP 1).
        rule(valu(), mat(), Dep::Raw, 2),
        // 9. Matrix -> VALU / VMEM / DS reading result: up to 18 wait states (16-pass MFMA).
        rule(mat(), valu(), Dep::Raw, WINDOW),
        rule(mat(), vmem_ds(), Dep::Raw, WINDOW),
        // 10. Matrix -> Matrix reading result as SrcA/SrcB (non-SrcC): up to 18 wait states.
        rule(mat(), mat(), Dep::ReadsNotAsSrcC, WINDOW),
        // 11. Matrix -> Matrix reading result as SrcC of a different op: up to 16 wait states.
        rule(mat(), mat(), Dep::RawUnlessAccumulates, 16),
        // 12. Matrix reading SrcC late -> VALU/Matrix overwriting SrcC: up to 14 wait states.
        rule(mat(), valu(), Dep::WritesSrcC, 14),
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
