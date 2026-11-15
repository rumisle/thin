//! Layout keys, refusals, round trips and coords.

use super::*;
use crate::dtype::dtype;

fn refused(key: &str) -> bool {
    matches!(Layout::parse(key), Err(Error::Layout(_)))
}

#[test]
fn keys_round_trip_and_parse_strictly() {
    for key in [
        "row_major",
        "row_major(ld=6)",
        "row_major(pad_to=4)",
        "col_major",
    ] {
        assert_eq!(Layout::parse(key).unwrap().key(), key);
    }
    let layout = Layout::new("row_major", &[("pad_to", 2), ("ld", 0)]).unwrap();
    assert_eq!(layout.key(), "row_major(pad_to=2)");
    assert_eq!(layout.args(), [("pad_to", 2)]);
    for key in [
        "row_major(ld=0)",
        "row_major()",
        "row_major(ld=6 )",
        "row_major( ld=6)",
        "row_major(ld=+6)",
        "row_major(ld=06)",
        "row_major(ld=x)",
        "row_major(ld=6",
        "row_major(ld)",
        "row_major(width=6)",
        "row_major(ld=6,ld=6)",
        "rowmajor",
        "col_major(ld=1)",
    ] {
        assert!(refused(key), "{key}");
    }
    let pair = Layout::of(&PAIR, &[("b", 1), ("a", 2)]).unwrap();
    assert_eq!(pair.key(), "pair(a=2,b=1)");
    assert_eq!(Layout::parse_in(&[&PAIR], "pair(a=2,b=1)").unwrap(), pair);
    assert!(Layout::parse_in(&[&PAIR], "pair(b=1,a=2)").is_err());
}

/// A definition of two parameters, for argument order.
static PAIR: Definition = Definition {
    name: "pair",
    params: &[("a", 0), ("b", 0)],
    build: |_, _, _| Ok(Arrangement::Recipe(Recipe::new(0, 0, vec![0], vec![0]))),
};

#[test]
fn row_major_refusals() {
    let layout_error = |r: Result<Layout>| matches!(r, Err(Error::Layout(_)));
    assert!(layout_error(Layout::new(
        "row_major",
        &[("ld", 8), ("pad_to", 2)]
    )));
    assert!(layout_error(Layout::new("row_major", &[("ld", -1)])));
    assert!(layout_error(Layout::new("row_major", &[("pad_to", -2)])));
    let d = dtype("e2m1").unwrap();
    let ld4 = Layout::parse("row_major(ld=4)").unwrap();
    assert!(matches!(nbytes(&ld4, d, (1, 2, 5)), Err(Error::Layout(_))));
    let plain = Layout::parse("row_major").unwrap();
    assert!(matches!(
        nbytes(&plain, d, (1, 2, 3)),
        Err(Error::Layout(_))
    ));
    assert_eq!(nbytes(&ld4, d, (1, 2, 4)).unwrap(), 4);
    let col = Layout::parse("col_major").unwrap();
    assert!(matches!(nbytes(&col, d, (1, 2, 4)), Err(Error::Layout(_))));
    assert!(matches!(
        nbytes(&plain, d, (1, 0, 4)),
        Err(Error::Layout(_))
    ));
}

/// Fills `n` storage bytes of a dtype with a pattern whose values fit it.
fn elements(name: &str, n: usize) -> Vec<u8> {
    let d = dtype(name).unwrap();
    let mask = if d.width < 8 {
        (1u16 << d.width) - 1
    } else {
        0xFF
    };
    (0..n * d.storage_bytes())
        .map(|i| ((i * 37 + 11) as u16 & mask) as u8)
        .collect()
}

/// to_bytes then from_bytes gives the elements back, and every offset's
/// coords hold the byte the layout put there.
fn round_trip(layout: &Layout, name: &str, shape: Shape) {
    let d = dtype(name).unwrap();
    let bits = elements(name, shape.0 * shape.1 * shape.2);
    let size = nbytes(layout, d, shape).unwrap();
    let raw = to_bytes(&bits, layout, d, shape).unwrap();
    let mut back = vec![0; bits.len()];
    from_bytes(&raw, layout, d, shape, &mut back).unwrap();
    assert_eq!(back, bits, "{layout:?} {name} {shape:?}");
    let row = logical::row_bytes(d, shape.2).unwrap();
    let mut packed = vec![0; shape.0 * shape.1 * row];
    pack(&bits, d, shape, &mut packed).unwrap();
    let offsets: Vec<u64> = (0..size as u64).collect();
    let mut at = vec![0; 3 * size];
    coords(layout, d, shape, &offsets, &mut at).unwrap();
    for (offset, c) in at.chunks_exact(3).enumerate() {
        if c[0] >= 0 {
            let index = (c[0] as usize * shape.1 + c[1] as usize) * row + c[2] as usize;
            assert_eq!(
                raw[offset], packed[index],
                "{layout:?} {name} {shape:?} @{offset}"
            );
        }
    }
}

#[test]
fn round_trips_over_shapes_and_dtypes() {
    let shapes = [(1, 1, 1), (2, 3, 5), (3, 4, 7), (1, 5, 2), (2, 1, 9)];
    for shape in shapes {
        for name in ["u8", "bf16", "f32", "u64", "e4m3fn"] {
            for key in [
                "row_major",
                "row_major(ld=11)",
                "row_major(pad_to=4)",
                "col_major",
            ] {
                round_trip(&Layout::parse(key).unwrap(), name, shape);
            }
        }
        for name in ["e2m1", "e2m3", "e3m2"] {
            for key in ["row_major(pad_to=4)", "row_major(ld=12)"] {
                round_trip(&Layout::parse(key).unwrap(), name, shape);
            }
        }
    }
}

/// Two copies of each matrix's bytes between fill, after a transpose.
static FRAMED: Definition = Definition {
    name: "framed",
    params: &[],
    build: |dtype, (stack, rows, cols), _| {
        let Some(dtype) = dtype else {
            return Ok(Arrangement::Recipe(Recipe::new(0, 0, vec![0], vec![0])));
        };
        let row_bytes = cols * dtype.storage_bytes();
        Ok(Arrangement::Recipe(Recipe {
            out: Some(vec![stack, 1, row_bytes * (rows + 1)]),
            pad: vec![(0, 0), (0, 0), (1, 2)],
            repeat: vec![(1, 2)],
            fill: 0xAA,
            ..Recipe::new(
                rows + 1,
                row_bytes,
                vec![stack, rows + 1, row_bytes],
                vec![0, 2, 1],
            )
        }))
    },
};

#[test]
fn recipes_pad_repeat_and_check_their_copies() {
    let layout = Layout::of(&FRAMED, &[]).unwrap();
    let d = dtype("u8").unwrap();
    let shape = (1, 2, 2);
    assert_eq!(nbytes(&layout, d, shape).unwrap(), 18);
    let mut raw = to_bytes(&[1, 2, 3, 4], &layout, d, shape).unwrap();
    let copy = [0xAA, 1, 3, 0xAA, 2, 4, 0xAA, 0xAA, 0xAA];
    assert_eq!(raw, [copy, copy].concat());
    let mut at = vec![0; 6];
    coords(&layout, d, shape, &[14, 12], &mut at).unwrap();
    assert_eq!(at, [0, 1, 1, -1, -1, -1]);
    let mut back = [0; 4];
    from_bytes(&raw, &layout, d, shape, &mut back).unwrap();
    assert_eq!(back, [1, 2, 3, 4]);
    raw[0] = 0;
    raw[9] = 0;
    from_bytes(&raw, &layout, d, shape, &mut back).unwrap();
    raw[10] = 9;
    assert!(matches!(
        from_bytes(&raw, &layout, d, shape, &mut back),
        Err(Error::Layout(_))
    ));
    assert!(matches!(
        coords(&layout, d, shape, &[18], &mut at[..3]),
        Err(Error::Layout(_))
    ));
}

/// Each pair of bytes swapped, decoded from its first period only.
static SWAPPED: Definition = Definition {
    name: "swapped",
    params: &[],
    build: |dtype, (stack, rows, cols), _| {
        let row_bytes = dtype.map_or(0, |d| cols * d.storage_bytes());
        Ok(Arrangement::Map(Map {
            rows,
            row_bytes,
            size: stack * rows * row_bytes,
            decode: Box::new(move |offsets, out| {
                for (&o, c) in offsets.iter().zip(out) {
                    assert!(o < 2, "only the first period is decoded");
                    *c = [0, 0, 1 - o as usize];
                }
            }),
            period: Some((2, 2)),
        }))
    },
};

#[test]
fn maps_decode_one_period() {
    let layout = Layout::of(&SWAPPED, &[]).unwrap();
    let d = dtype("bf16").unwrap();
    let shape = (2, 1, 3);
    let bits: Vec<u8> = (0..12).collect();
    let raw = to_bytes(&bits, &layout, d, shape).unwrap();
    assert_eq!(raw, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]);
    let mut back = vec![0; 12];
    from_bytes(&raw, &layout, d, shape, &mut back).unwrap();
    assert_eq!(back, bits);
    let mut at = vec![0; 3];
    coords(&layout, d, shape, &[7], &mut at).unwrap();
    assert_eq!(at, [1, 0, 0]);
    round_trip(&layout, "f16", (3, 2, 5));
}

/// A map one row taller than the buffer: refused, since a Map has no padding.
static TALL: Definition = Definition {
    name: "tall",
    params: &[],
    build: |dtype, (stack, rows, cols), _| {
        let row_bytes = dtype.map_or(0, |d| cols * d.storage_bytes());
        Ok(Arrangement::Map(Map {
            rows: rows + 1,
            row_bytes,
            size: stack * (rows + 1) * row_bytes,
            decode: Box::new(|offsets, out| {
                for (&o, c) in offsets.iter().zip(out) {
                    *c = [0, 0, o as usize];
                }
            }),
            period: None,
        }))
    },
};

#[test]
fn maps_have_no_padding() {
    let layout = Layout::of(&TALL, &[]).unwrap();
    let u8 = dtype("u8").unwrap();
    assert!(matches!(
        nbytes(&layout, u8, (1, 1, 2)),
        Err(Error::Layout(_))
    ));
}
