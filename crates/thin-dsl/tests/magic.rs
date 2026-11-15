//! The divisor triple divides exactly.

/// The quotient the lowering computes from (m, s, short).
fn quotient(x: u32, (m, s, short): (u64, u32, bool)) -> u32 {
    let q = ((u64::from(x) * m) >> 32) as u32;
    match short {
        true => q >> s,
        false => (((x - q) >> 1) + q) >> (s - 1),
    }
}

/// A small deterministic generator (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

const EDGES: [u32; 5] = [0, 1, 1 << 31, u32::MAX - 1, u32::MAX];

fn check(d: u32, rng: &mut Rng, n: usize) {
    let t = thin_dsl::magic(d).expect("a divisor");
    let xs = EDGES.iter().copied().chain((0..n).map(|_| rng.next()));
    for x in xs {
        assert_eq!(quotient(x, t), x / d, "{x} / {d} by {t:?}");
    }
}

#[test]
fn small_and_edge_divisors_are_exact() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for d in (2..=64).chain([641, 6700417, 1 << 31, u32::MAX - 1, u32::MAX]) {
        check(d, &mut rng, 20_000);
    }
}

#[test]
fn random_divisors_are_exact() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..1000 {
        let d = rng.next().max(2) >> (rng.next() % 31);
        check(d.max(2), &mut rng, 2_000);
    }
}

#[test]
fn zero_is_refused() {
    assert!(thin_dsl::magic(0).is_err());
}
