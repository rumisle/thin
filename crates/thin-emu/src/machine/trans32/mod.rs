//! gfx1201's f32 transcendental unit, bit for bit: v_rcp_f32, v_rsq_f32,
//! v_sqrt_f32, v_log_f32 and v_exp_f32.
//!
//! The unit is faithful, not correctly rounded (RDNA4 ISA guide: "1
//! ULP"): its results differ from the correctly rounded ones on 3–37% of
//! inputs. The algorithm is AMD rocjitsu's (ROCm/rocm-systems,
//! emulation/rocjitsu, MIT; see NOTICE): a 32- or 64-segment fixed-point
//! cubic whose linear and quadratic terms round at their own widths, then
//! one rounding to the significand. log adds two narrow intervals around
//! 1 and the exponent in fixed point. For rcp, rsq and sqrt a sparse table
//! of one-ULP fix-ups completes the mapping; thin measured its own tables
//! on gfx1201 over every mantissa and exponent parity (`fixups.rs`).
//!
//! Proven on gfx1201 over all 2^32 inputs of each op: every word agrees
//! (see `tests/hw/test_emu_ops.py` `test_an_f32_transcendental_is_the_gpus`
//! for the standing check on a dense sweep). Denormal inputs read as zero
//! and denormal results flush to zero, sign kept; a NaN input is quieted,
//! payload kept; an invalid one gives the default NaN 0xFFC00000.

mod fixups;

use fixups::{RCP_FIXUPS, RSQ_FIXUPS, SQRT_FIXUPS};

/// v_rcp_f32's cubics, one per 2^18 mantissas: constant, linear, quadratic
/// and cubic coefficients in quarter units of the result's ULP.
const RCP: [[u32; 4]; 32] = [
    [67108864, 2097138, 65462, 1925],
    [65075262, 1971966, 59693, 1705],
    [63161283, 1857674, 54582, 1516],
    [61356675, 1753039, 50039, 1352],
    [59652323, 1657001, 45986, 1210],
    [58040098, 1568644, 42359, 1086],
    [56512727, 1487171, 39104, 977],
    [55063683, 1411884, 36174, 882],
    [53687091, 1342172, 33529, 798],
    [52377649, 1277499, 31136, 724],
    [51130563, 1217390, 28966, 658],
    [49941480, 1161426, 26992, 599],
    [48806446, 1109234, 25194, 547],
    [47721858, 1060483, 23552, 501],
    [46684427, 1014876, 22050, 459],
    [45691141, 972149, 20673, 421],
    [44739242, 932065, 19408, 388],
    [43826196, 894410, 18244, 357],
    [42949672, 858991, 17171, 330],
    [42107522, 825636, 16181, 305],
    [41297762, 794186, 15266, 282],
    [40518559, 764500, 14418, 262],
    [39768215, 736447, 13632, 243],
    [39045157, 709911, 12902, 226],
    [38347922, 684783, 12223, 210],
    [37675151, 660966, 11591, 196],
    [37025580, 638371, 11002, 183],
    [36398027, 616915, 10452, 171],
    [35791394, 596522, 9938, 160],
    [35204649, 577124, 9458, 150],
    [34636833, 558658, 9007, 140],
    [34087042, 541063, 8585, 131],
];

/// v_rsq_f32's cubics, one per 2^18 of exponent parity · 2^23 + mantissa,
/// in quarter units of the result's ULP.
const RSQ: [[u32; 4]; 64] = [
    [67108864, 1048572, 24555, 606],
    [66084241, 1001273, 22738, 545],
    [65105161, 957426, 21104, 491],
    [64168347, 916688, 19629, 445],
    [63270843, 878759, 18295, 403],
    [62409975, 843376, 17084, 367],
    [61583317, 810305, 15983, 335],
    [60788660, 779340, 14978, 306],
    [60023992, 750298, 14060, 280],
    [59287474, 723016, 13219, 257],
    [58577418, 697349, 12446, 237],
    [57892279, 673165, 11735, 218],
    [57230631, 650347, 11080, 201],
    [56591163, 628789, 10475, 186],
    [55972661, 608397, 9915, 173],
    [55374006, 589084, 9396, 160],
    [54794158, 570771, 8914, 149],
    [54232151, 553388, 8467, 139],
    [53687091, 536870, 8050, 129],
    [53158141, 521157, 7661, 121],
    [52644524, 506196, 7298, 113],
    [52145512, 491938, 6959, 105],
    [51660427, 478336, 6641, 99],
    [51188632, 465350, 6343, 93],
    [50729532, 452941, 6064, 87],
    [50282567, 441074, 5802, 82],
    [49847212, 429717, 5555, 77],
    [49422973, 418838, 5322, 73],
    [49009384, 408411, 5103, 68],
    [48606008, 398409, 4897, 65],
    [48212431, 388809, 4702, 61],
    [47828262, 379589, 4517, 58],
    [47453132, 741452, 17363, 428],
    [46728614, 708007, 16078, 385],
    [46036300, 677002, 14922, 347],
    [45373873, 648196, 13880, 314],
    [44739242, 621376, 12936, 285],
    [44130517, 596356, 12080, 259],
    [43545981, 572972, 11301, 236],
    [42984073, 551076, 10591, 216],
    [42443372, 530541, 9942, 198],
    [41922574, 511250, 9347, 182],
    [41420490, 493100, 8801, 167],
    [40936023, 475999, 8298, 154],
    [40468167, 459864, 7835, 142],
    [40015995, 444621, 7407, 132],
    [39578648, 430202, 7011, 122],
    [39155335, 416545, 6644, 113],
    [38745320, 403596, 6303, 105],
    [38347922, 391304, 5987, 98],
    [37962506, 379624, 5692, 91],
    [37588482, 368514, 5417, 85],
    [37225300, 357935, 5160, 80],
    [36872445, 347852, 4920, 74],
    [36529438, 338235, 4696, 70],
    [36195829, 329052, 4485, 65],
    [35871196, 320278, 4288, 61],
    [35555144, 311887, 4102, 58],
    [35247302, 303855, 3928, 54],
    [34947319, 296163, 3763, 51],
    [34654868, 288790, 3609, 48],
    [34369638, 281718, 3462, 45],
    [34091337, 274930, 3325, 43],
    [33819688, 268410, 3194, 41],
];

/// v_sqrt_f32's cubics, one per 2^19 of exponent parity · 2^23 + mantissa
/// (constant and cubic in quarter ULP, quadratic in eighth ULP).
const SQRT: [[u32; 4]; 32] = [
    [0, 2097143, 32722, 474],
    [1032684, 2034529, 29882, 409],
    [2035417, 1977208, 27431, 356],
    [3010662, 1924474, 25297, 312],
    [3960563, 1875745, 23425, 275],
    [4886999, 1830541, 21774, 244],
    [5791627, 1788454, 20308, 218],
    [6675918, 1749143, 18999, 195],
    [7541186, 1712315, 17825, 176],
    [8388608, 1677719, 16767, 159],
    [9219243, 1645139, 15810, 145],
    [10034053, 1614387, 14940, 132],
    [10833909, 1585296, 14147, 120],
    [11619604, 1557724, 13422, 110],
    [12391866, 1531542, 12757, 102],
    [13151360, 1506637, 12145, 94],
    [13898700, 2965809, 46276, 670],
    [15359137, 2877259, 42260, 578],
    [16777216, 2796194, 38793, 503],
    [18156420, 2721617, 35775, 441],
    [19499783, 2652705, 33129, 389],
    [20809961, 2588776, 30793, 346],
    [22089298, 2529256, 28720, 308],
    [23339875, 2473662, 26869, 277],
    [24563549, 2421579, 25208, 249],
    [25761984, 2372654, 23712, 225],
    [26936680, 2326579, 22358, 205],
    [28088995, 2283088, 21129, 186],
    [29220162, 2241948, 20008, 171],
    [30331303, 2202954, 18982, 156],
    [31423446, 2165927, 18042, 144],
    [32497533, 2130707, 17176, 133],
];

/// v_log_f32's cubics by the mantissa's top five bits; the 33rd refines the
/// upper half of the second interval.
const LOG: [[u32; 4]; 33] = [
    [15, 12102179, 12095174, 1925858],
    [47667825, 11735468, 11378345, 1799053],
    [93912517, 11390291, 10714741, 1610016],
    [138816594, 11064855, 10111507, 1477533],
    [182455591, 10757500, 9557803, 1359325],
    [224898847, 10466758, 9048378, 1253542],
    [266210154, 10191316, 8578611, 1158417],
    [306448312, 9930003, 8144494, 1072666],
    [345667679, 9681753, 7742500, 995087],
    [383918558, 9445614, 7369567, 924970],
    [421247645, 9220718, 7022931, 861214],
    [457698312, 9006284, 6700175, 803096],
    [493310964, 8801596, 6399165, 750066],
    [528123258, 8606005, 6118007, 701755],
    [562170399, 8418918, 5854955, 657310],
    [595485263, 8239793, 5608532, 616715],
    [628098719, 8068131, 5377341, 579376],
    [660039693, 7903475, 5160150, 544955],
    [691335339, 7745406, 4955875, 513361],
    [722011229, 7593536, 4763477, 484038],
    [752091435, 7447506, 4582059, 456830],
    [781598650, 7306987, 4410807, 431594],
    [810554297, 7171673, 4248984, 408222],
    [838978616, 7041279, 4095937, 386761],
    [866890752, 6915542, 3950988, 366593],
    [894308853, 6794217, 3813578, 347636],
    [921250083, 6677075, 3683220, 330396],
    [947730761, 6563905, 3559471, 313859],
    [973766367, 6454506, 3441820, 298469],
    [999371610, 6348694, 3329908, 284397],
    [1024560488, 6246298, 3223482, 270890],
    [1049346329, 6147160, 3121894, 258238],
    [47667912, 11735350, 11363743, 1719989],
];

/// v_exp_f32's cubics by the top five fraction bits of x (constant,
/// linear, quadratic and cubic coefficients).
const EXP: [[u32; 4]; 32] = [
    [268435456, 5814541, 8060268, 29],
    [274313427, 5941862, 8236765, 29],
    [280320109, 6071972, 8417127, 30],
    [286458319, 6204931, 8601438, 31],
    [292730941, 6340801, 8789785, 31],
    [299140913, 6479647, 8982256, 32],
    [305691246, 6621533, 9178942, 33],
    [312385013, 6766525, 9379934, 33],
    [319225354, 6914693, 9585328, 34],
    [326215479, 7066105, 9795219, 35],
    [333358667, 7220833, 10009707, 36],
    [340658273, 7378948, 10228891, 36],
    [348117717, 7540526, 10452874, 37],
    [355740503, 7705642, 10681762, 38],
    [363530205, 7874374, 10915663, 39],
    [371490480, 8046800, 11154685, 40],
    [379625062, 8223002, 11398940, 41],
    [387937769, 8403062, 11648545, 42],
    [396432501, 8587066, 11903614, 42],
    [405113242, 8775098, 12164270, 43],
    [413984066, 8967247, 12430633, 44],
    [423049137, 9163605, 12702828, 45],
    [432312707, 9364261, 12980984, 46],
    [441779122, 9569312, 13265230, 47],
    [451452826, 9778853, 13555701, 48],
    [461338356, 9992982, 13852532, 49],
    [471440350, 10211799, 14155863, 50],
    [481763549, 10435409, 14465836, 52],
    [492312797, 10663914, 14782597, 53],
    [503093043, 10897424, 15106293, 54],
    [514109347, 11136046, 15437078, 55],
    [525366876, 11379894, 15775106, 56],
];

/// `x >> shift` rounded half to even.
fn round_even(x: u64, shift: u32) -> u64 {
    let whole = x >> shift;
    let rest = x & ((1 << shift) - 1);
    let half = 1u64 << (shift - 1);
    whole + u64::from(rest > half || (rest == half && whole & 1 == 1))
}

/// The cubic of v_rcp_f32 on a 23-bit mantissa: the result's bits for an
/// input in [1, 2), before the measured fix-ups.
fn rcp_cubic(m: u32) -> u32 {
    let [c0, c1, c2, c3] = RCP[(m >> 18) as usize];
    let f = u64::from(m & 0x3_FFFF);
    let p = u64::from(c1) * f;
    let shift = if p >= 1 << 38 { 15 } else { 14 };
    let linear = round_even(p, shift) << (shift - 14);
    let inner = (u64::from(c2) << 18) - u64::from(c3) * f;
    let g = u128::from(f & !1);
    let quad = ((u128::from(inner) * g * g) >> 51) as u64;
    let v = (u64::from(c0) * 16 - linear + quad * 2 + 32) >> 6;
    0x3E80_0000 + v as u32
}

/// The cubic of v_rsq_f32 on index = exponent parity · 2^23 + mantissa:
/// the result's bits for an input in [1, 4).
fn rsq_cubic(index: u32) -> u32 {
    let [c0, c1, c2, c3] = RSQ[(index >> 18) as usize];
    let f = u64::from(index & 0x3_FFFF);
    let p = u64::from(c1) * f;
    let shift = if p >= 1 << 37 { 14 } else { 13 };
    let linear = round_even(p, shift) << (shift - 13);
    let inner = (u64::from(c2) << 18) - u64::from(c3) * f;
    let g = u128::from(f & !1);
    let quad = ((u128::from(inner) * g * g) >> 51) as u64;
    0x3E80_0000 + ((u64::from(c0) * 32 - linear + quad * 4 + 64) >> 7) as u32
}

/// The cubic of v_sqrt_f32 on index = exponent parity · 2^23 + mantissa:
/// the result's bits for an input in [1, 4).
fn sqrt_cubic(index: u32) -> u32 {
    let [c0, c1, c2, c3] = SQRT[(index >> 19) as usize];
    let f = i64::from(index & 0x7_FFFF);
    let p = u64::from(c1) * f as u64;
    let bias = if p >= 1 << 40 { 1u64 << 16 } else { 1 << 15 };
    let linear = ((p + bias) >> 17) as i64;
    let g = f & !3;
    let square = (g * g) >> 14;
    let inner = (i64::from(c2) << 8) + ((256 - i64::from(c3) * f) >> 10);
    let q = inner * square;
    let negated = if q >= 1 << 47 {
        (((1i64 << 23) - q) >> 24) * 2
    } else {
        ((1i64 << 22) - q) >> 23
    };
    let sum = (i64::from(c0) << 22) + (linear << 19) + negated * 4096;
    let (whole, rest) = (sum >> 24, sum & ((1 << 24) - 1));
    let half = 1i64 << 23;
    let m = whole + i64::from(rest > half || (rest == half && whole & 1 == 1));
    0x3F80_0000 + m as u32
}

/// A fix-up table's entry for `key` (entries `key << 1 | up`, sorted): +1
/// or −1 ULP, or none.
fn fixup(table: &[u32], key: u32) -> i32 {
    match table.binary_search_by(|e| (e >> 1).cmp(&key)) {
        Ok(i) => {
            if table[i] & 1 == 1 {
                1
            } else {
                -1
            }
        }
        Err(_) => 0,
    }
}

/// v_rcp_f32. Denormal inputs read as zero (a signed infinity), results
/// below the normal range flush to a signed zero.
pub fn rcp(x: u32) -> u32 {
    let (sign, mag) = (x & 0x8000_0000, x & 0x7FFF_FFFF);
    if mag > 0x7F80_0000 {
        return x | 0x0040_0000;
    }
    if mag < 0x0080_0000 {
        return sign | 0x7F80_0000;
    }
    if mag == 0x7F80_0000 {
        return sign;
    }
    let m = mag & 0x7F_FFFF;
    let n = (rcp_cubic(m) as i32 + fixup(&RCP_FIXUPS, m)) as u32;
    let e = (n >> 23) as i32 + 127 - (mag >> 23) as i32;
    if e <= 0 {
        sign
    } else if e >= 255 {
        sign | 0x7F80_0000
    } else {
        sign | (e as u32) << 23 | n & 0x7F_FFFF
    }
}

/// v_rsq_f32: a negative input gives the default NaN; ±0 and denormals a
/// signed infinity; +∞ gives +0.
pub fn rsq(x: u32) -> u32 {
    let (sign, mag) = (x & 0x8000_0000, x & 0x7FFF_FFFF);
    if mag > 0x7F80_0000 {
        return x | 0x0040_0000;
    }
    if mag < 0x0080_0000 {
        return sign | 0x7F80_0000;
    }
    if sign != 0 {
        return crate::machine::numbers::DNAN32;
    }
    if mag == 0x7F80_0000 {
        return 0;
    }
    let e = (mag >> 23) as i32 - 127;
    let parity = (e & 1) as u32;
    let index = parity << 23 | mag & 0x7F_FFFF;
    let n = (rsq_cubic(index) as i32 + fixup(&RSQ_FIXUPS, index)) as u32;
    let re = (n >> 23) as i32 - (e - parity as i32) / 2;
    (re as u32) << 23 | n & 0x7F_FFFF
}

/// v_sqrt_f32: a negative input gives the default NaN; denormals read as a
/// signed zero; ±0 and +∞ are kept.
pub fn sqrt(x: u32) -> u32 {
    let (sign, mag) = (x & 0x8000_0000, x & 0x7FFF_FFFF);
    if mag > 0x7F80_0000 {
        return x | 0x0040_0000;
    }
    if mag < 0x0080_0000 {
        return sign;
    }
    if sign != 0 {
        return crate::machine::numbers::DNAN32;
    }
    if mag == 0x7F80_0000 {
        return x;
    }
    let e = (mag >> 23) as i32 - 127;
    let parity = (e & 1) as u32;
    let index = parity << 23 | mag & 0x7F_FFFF;
    let n = (sqrt_cubic(index) as i32 + fixup(&SQRT_FIXUPS, index)) as u32;
    let re = (n >> 23) as i32 + (e - parity as i32) / 2;
    (re as u32) << 23 | n & 0x7F_FFFF
}

/// v_exp_f32 (2^x): x reduced to a fixed-point phase with 29 fraction bits
/// (a negative x by its ones' complement), the fraction's top five bits
/// pick a cubic, whose linear and quadratic terms are rounded at their own
/// widths before one rounding to the result's significand.
pub fn exp(x: u32) -> u32 {
    let mag = x & 0x7FFF_FFFF;
    let negative = x >> 31 == 1;
    if mag >= 0x7F80_0000 {
        if mag > 0x7F80_0000 {
            return x | 0x0040_0000;
        }
        return if negative { 0 } else { 0x7F80_0000 };
    }
    if mag < 0x3380_0000 {
        return 0x3F80_0000;
    }
    if !negative && mag >= 0x4300_0000 {
        return 0x7F80_0000;
    }
    if negative && mag > 0x42FC_0000 {
        return 0;
    }
    let e = (mag >> 23) as i32 - 127;
    let m = u64::from(mag & 0x7F_FFFF | 0x80_0000);
    let shift = e + 6;
    let abs = if shift >= 0 { m << shift } else { m >> -shift };
    let phase = if negative {
        -(abs as i64) - 1
    } else {
        abs as i64
    };
    let f = (phase as u64 & 0xFF_FFFF) as u32;
    let [c0, c1, c2, c3] = EXP[((phase as u64 >> 24) & 31) as usize];
    let lp = u64::from(c1) * u64::from(f);
    let linear = if lp >= 1 << 47 {
        round_even(lp, 24)
    } else {
        round_even(lp, 23) >> 1
    };
    let t = u64::from(f >> 6);
    let square = (t * t) >> 12;
    let inner = round_even((u64::from(c2) << 7) + u64::from(c3) * t, 7);
    let p = inner * square;
    let quad = if p >= 1 << 47 {
        round_even(p, 24) * 2
    } else {
        round_even(p, 23)
    };
    let sum = ((u64::from(c0) + linear) << 8) + quad;
    (((phase >> 29) + 126) * 0x80_0000 + round_even(sum, 13) as i64) as u32
}

/// log2 just below 1 (x in [1 − 2^-6, 1)), from the distance k = 1 − x in
/// units of 2^-24: its own two cubics, terms normalised by four-bit steps
/// to keep precision, the magnitude rounded up.
fn log_below_one(x: u32) -> u32 {
    let k = 0x3F80_0000 - x;
    let upper = k >= 0x2_0000;
    let (c1, c2, c3) = if upper {
        (12_102_221u64, 12_097_323u64, 2_088_428u64)
    } else {
        (12_102_203, 12_101_979, 2_035_248)
    };
    let width = 32 - k.leading_zeros();
    let norm = if width < 18 {
        4 * ((18 - width) / 4)
    } else {
        0
    };
    let ls = 14u32.saturating_sub(norm);
    let ss = 12u32.saturating_sub(norm);
    let k = u64::from(k);
    let linear = ((2 * c1 * k) >> ls) << ls;
    let square = ((k * k) >> ss) << ss;
    let inner = round_even((c2 << 22) + c3 * (k - 1), 22);
    let ps = 38 - norm;
    let quad = (inner * square + ((1u64 << ps) - 1)) >> ps;
    let fixed = (linear << 2) + (quad << (16 - norm)) - if upper { 100 << 16 } else { 0 };
    let mut lead = 63 - fixed.leading_zeros();
    let shift = lead - 23;
    let mut m = (fixed + ((1u64 << shift) - 1)) >> shift;
    if m == 1 << 24 {
        m >>= 1;
        lead += 1;
    }
    0x8000_0000 | (lead + 127 - 50) << 23 | (m as u32 & 0x7F_FFFF)
}

/// log2 just above 1 (x in (1, 1 + 2^-5)), from the distance i = x − 1 in
/// units of 2^-23: three cubics, the first normalised by four-bit steps;
/// the truncated significand is advanced by one.
fn log_above_one(x: u32) -> u32 {
    let i = x - 0x3F80_0000;
    let row = if i < 0x1_0000 {
        0
    } else if i < 0x2_0000 {
        1
    } else {
        2
    };
    const C0: [u64; 3] = [0, 192, 2856];
    const C1: [u64; 3] = [12_102_203, 12_102_186, 12_102_072];
    const C2: [u64; 3] = [12_102_094, 12_097_614, 12_084_310];
    const C3: [u64; 3] = [1_999_052, 1_948_887, 1_883_467];
    let width = 32 - i.leading_zeros();
    let norm = if row == 0 { 4 * ((18 - width) / 4) } else { 0 };
    let i = u64::from(i);
    let n = i << norm;
    let linear = ((C1[row] * n) >> 13) << 2;
    let square = (n * i) >> 12;
    let inner = round_even((C2[row] << 22) - C3[row] * (2 * i + 1), 22);
    let p = inner * square;
    let quad = if p >= 1 << 47 {
        round_even(p, 24) * 2
    } else {
        round_even(p, 23)
    };
    let fixed = C0[row] + linear - quad;
    let mut lead = 63 - fixed.leading_zeros();
    let shift = lead - 23;
    let mut m = (fixed >> shift) + 1;
    if m == 1 << 24 {
        m >>= 1;
        lead += 1;
    }
    (lead + 127 - 35 - norm) << 23 | (m as u32 & 0x7F_FFFF)
}

/// log2 elsewhere: the mantissa's top five bits pick a cubic (a 33rd
/// refines the upper half of the second interval), the exponent is added
/// in fixed point and the sum rounded toward −∞.
fn log_ordinary(x: u32) -> u32 {
    let f = u64::from(x & 0x3_FFFF);
    let mut index = ((x >> 18) & 31) as usize;
    if index == 1 && f >= 0x2_0000 {
        index = 32;
    }
    let [c0, c1, c2, c3] = LOG[index].map(u64::from);
    let linear = (c1 * f) >> 16;
    let inner = round_even((c2 << 22) - c3 * (2 * f + 1), 22);
    let square = (f * f) >> 12;
    let p = inner * square;
    let quad = if p >= 1 << 47 {
        round_even(p, 24) * 2
    } else {
        round_even(p, 23)
    };
    let e = i64::from((x >> 23) as i32 - 127);
    let fixed = e * (1i64 << 35) + (((c0 + linear) << 5) - quad) as i64;
    let negative = fixed < 0;
    let mag = fixed.unsigned_abs();
    let mut lead = 63 - mag.leading_zeros();
    let shift = lead - 23;
    let up = if negative { (1u64 << shift) - 1 } else { 0 };
    let mut m = (mag + up) >> shift;
    if m == 1 << 24 {
        m >>= 1;
        lead += 1;
    }
    let s = if negative { 0x8000_0000 } else { 0 };
    s | (lead + 127 - 35) << 23 | (m as u32 & 0x7F_FFFF)
}

/// v_log_f32 (log2 x): a denormal input reads as zero (−∞), a negative
/// one gives the default NaN.
pub fn log(x: u32) -> u32 {
    let mag = x & 0x7FFF_FFFF;
    if mag > 0x7F80_0000 {
        return x | 0x0040_0000;
    }
    if mag < 0x0080_0000 {
        return 0xFF80_0000;
    }
    if x >> 31 == 1 {
        return crate::machine::numbers::DNAN32;
    }
    if mag == 0x7F80_0000 {
        return 0x7F80_0000;
    }
    if x == 0x3F80_0000 {
        return 0;
    }
    if (0x3F7C_0000..0x3F80_0000).contains(&x) {
        return log_below_one(x);
    }
    if x > 0x3F80_0000 && x < 0x3F84_0000 {
        return log_above_one(x);
    }
    log_ordinary(x)
}
