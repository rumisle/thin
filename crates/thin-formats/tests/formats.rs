//! Golden checks for element encodings, layouts and code-object containers.

use serde_json::Value;
use sha2::{Digest, Sha256};
use thin_formats::{dtype, layout, EncodeOptions, Layout};
use thin_test_support::golden::{self, hex, hex_bytes, msgpack_of, split_id};
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::{Case, Suite};

fn bytes_of(v: &Value, width: usize, out: &mut Vec<u8>) {
    match v {
        Value::Array(items) => items.iter().for_each(|i| bytes_of(i, width, out)),
        Value::Number(n) => {
            out.extend_from_slice(&n.as_u64().expect("an element").to_le_bytes()[..width])
        }
        _ => panic!("not an element array: {v}"),
    }
}

fn storage_width(name: &str) -> usize {
    dtype(name).expect("a dtype").storage_bytes()
}

/// The Python class an error is raised as.
fn class_of(e: thin_formats::Error) -> &'static str {
    match e {
        thin_formats::Error::Type(_) => "TypeError",
        thin_formats::Error::Value(_) => "ValueError",
        thin_formats::Error::Layout(_) => "LayoutError",
        thin_formats::Error::CodeObject(_) => "CodeObjectError",
    }
}

fn err(e: thin_formats::Error) -> String {
    format!("{e:?}")
}

/// `decode.json`: every pattern of the small formats, chosen bf16/f16 ones.
fn decode_case(case: &Case<'_>) -> Result<(), String> {
    let (name, key) = split_id(&case.id);
    let pattern = if case.file[name].is_array() {
        key.parse().unwrap()
    } else {
        hex(key)
    };
    let bits = pattern.to_le_bytes()[..storage_width(name)].to_vec();
    let mut out = [0u32];
    thin_formats::decode(&bits, dtype(name).map_err(err)?, &mut out).map_err(err)?;
    expect_eq(
        u64::from(out[0]),
        hex(case.value.as_str().ok_or("not hex")?),
    )
}

/// `encode.json`: 273 inputs under each option set.
fn encode_case(case: &Case<'_>) -> Result<(), String> {
    let name = case.value["dtype"].as_str().ok_or("no dtype")?;
    let kw = &case.value["kwargs"];
    let options = EncodeOptions {
        saturate: kw["saturate"].as_bool().ok_or("no saturate")?,
        nan: kw["nan"].as_i64(),
        inf: kw["inf"].as_i64(),
    };
    let inputs: Vec<u32> = case.file["inputs_f32_hex"]
        .as_array()
        .ok_or("no inputs")?
        .iter()
        .map(|h| hex(h.as_str().unwrap()) as u32)
        .collect();
    let width = storage_width(name);
    let mut out = vec![0u8; inputs.len() * width];
    thin_formats::encode(&inputs, dtype(name).map_err(err)?, options, &mut out).map_err(err)?;
    let mut want = Vec::new();
    bytes_of(&case.value["bits"], width, &mut want);
    expect_eq(out, want)?;
    if dtype(name)
        .map_err(err)?
        .encoding
        .is_some_and(|e| e.man <= 6)
    {
        table_matches(name, options)?;
    }
    Ok(())
}

/// The encode table of an option set equals the definition on every input
/// that indexes each entry, at the extremes of its low 16 bits: zero for a
/// clear sticky bit, 1, 0x8000 and 0xFFFF for a set one.
fn table_matches(name: &str, options: EncodeOptions) -> Result<(), String> {
    let d = dtype(name).map_err(err)?;
    let mut table = vec![0u8; 1 << thin_formats::TABLE_INDEX_BITS];
    thin_formats::encode_table(d, options, &mut table).map_err(err)?;
    let mut inputs = Vec::with_capacity(table.len() * 3);
    let mut entries = Vec::with_capacity(table.len() * 3);
    for (i, &entry) in table.iter().enumerate() {
        let high = (i as u32 >> 1) << 16;
        let lows: &[u32] = if i & 1 == 0 {
            &[0]
        } else {
            &[1, 0x8000, 0xFFFF]
        };
        for &low in lows {
            inputs.push(high | low);
            entries.push(entry);
        }
    }
    let mut direct = vec![0u8; inputs.len()];
    thin_formats::encode(&inputs, d, options, &mut direct).map_err(err)?;
    expect_eq(direct, entries)
}

/// `mx.json`: codes and scales of f32 rows.
fn mx_case(case: &Case<'_>) -> Result<(), String> {
    let element = dtype(case.value["dtype"].as_str().ok_or("no dtype")?).map_err(err)?;
    if let Some(amax) = case.value["scale_bytes_amax_f32_hex"].as_array() {
        let amax: Vec<u32> = amax
            .iter()
            .map(|h| hex(h.as_str().unwrap()) as u32)
            .collect();
        let mut out = vec![0u8; amax.len()];
        thin_formats::scale_bytes(&amax, element, &mut out).map_err(err)?;
        let mut want = Vec::new();
        bytes_of(&case.value["bytes"], 1, &mut want);
        return expect_eq(out, want);
    }
    if let Some(shape) = case.value["shape"].as_array() {
        let shape: Vec<usize> = shape.iter().map(|d| d.as_u64().unwrap() as usize).collect();
        let n = shape.iter().product::<usize>() * usize::from(!shape.is_empty());
        let scales_len = n / thin_formats::GROUP;
        let (mut codes, mut scales) = (vec![0u8; n], vec![0u8; scales_len]);
        let got = thin_formats::quantize(&vec![0; n], &shape, element, &mut codes, &mut scales);
        return match case.value["refused"].as_str() {
            Some(class) => expect_eq(got.map_err(class_of), Err(class)),
            None => got.map_err(err),
        };
    }
    let rows = case.value["x_f32_hex"].as_array().ok_or("no x")?;
    let k = rows[0].as_array().map_or(0, Vec::len);
    let x: Vec<u32> = rows
        .iter()
        .flat_map(|r| r.as_array().into_iter().flatten())
        .map(|h| hex(h.as_str().unwrap()) as u32)
        .collect();
    let (mut codes, mut scales) = (vec![0u8; x.len()], vec![0u8; x.len() / 32]);
    let shape = [rows.len(), k];
    thin_formats::quantize(&x, &shape, element, &mut codes, &mut scales).map_err(err)?;
    let (mut want_codes, mut want_scales) = (Vec::new(), Vec::new());
    bytes_of(&case.value["codes"], 1, &mut want_codes);
    bytes_of(&case.value["scales"], 1, &mut want_scales);
    expect_eq((codes, scales), (want_codes, want_scales))
}

fn shape_of(v: &Value) -> Result<thin_formats::Shape, String> {
    let d: Vec<usize> = v
        .as_array()
        .ok_or("no shape")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    Ok((d[0], d[1], d[2]))
}

/// `pack.json`: a buffer's logical bytes, and back.
fn pack_case(case: &Case<'_>) -> Result<(), String> {
    let name = case.value["dtype"].as_str().ok_or("no dtype")?;
    let bits = &case.value["bits"];
    let shape = (
        bits.as_array().map_or(0, Vec::len),
        bits[0].as_array().map_or(0, Vec::len),
        bits[0][0].as_array().map_or(0, Vec::len),
    );
    let mut elements = Vec::new();
    bytes_of(bits, storage_width(name), &mut elements);
    let mut want = Vec::new();
    bytes_of(&case.value["logical"], 1, &mut want);
    let mut out = vec![0u8; want.len()];
    let d = dtype(name).map_err(err)?;
    layout::pack(&elements, d, shape, &mut out).map_err(err)?;
    expect_eq(&out, &want)?;
    let mut back = vec![0u8; elements.len()];
    layout::unpack(&out, d, shape, &mut back).map_err(err)?;
    expect_eq(back, elements)
}

/// `layout.json`: bits → bytes → coords for each key.
fn layout_case(case: &Case<'_>) -> Result<(), String> {
    if let Some(class) = case.value["error"].as_str() {
        return layout_refusal(case, class);
    }
    let key = Layout::parse(case.value["key"].as_str().ok_or("no key")?).map_err(err)?;
    let name = case.value["dtype"].as_str().ok_or("no dtype")?;
    let d = dtype(name).map_err(err)?;
    let shape = shape_of(&case.value["shape"])?;
    let nbytes = case.value["nbytes"].as_u64().ok_or("no nbytes")? as usize;
    expect_eq(thin_formats::nbytes(&key, d, shape).map_err(err)?, nbytes)?;
    let mut elements = Vec::new();
    bytes_of(&case.value["bits"], storage_width(name), &mut elements);
    let out = thin_formats::to_bytes(&elements, &key, d, shape).map_err(err)?;
    let mut want = Vec::new();
    bytes_of(&case.value["bytes"], 1, &mut want);
    expect_eq(&out, &want)?;
    let mut back = vec![0u8; elements.len()];
    thin_formats::from_bytes(&out, &key, d, shape, &mut back).map_err(err)?;
    expect_eq(&back, &elements)?;
    let offsets: Vec<u64> = (0..nbytes as u64).collect();
    let mut got = vec![0i64; 3 * nbytes];
    thin_formats::coords(&key, d, shape, &offsets, &mut got).map_err(err)?;
    for axis in 0..3 {
        let want: Vec<i64> = case.value["coords"][axis]
            .as_array()
            .ok_or("no coords")?
            .iter()
            .map(|c| c.as_i64().unwrap())
            .collect();
        expect_eq(
            got.iter()
                .skip(axis)
                .step_by(3)
                .copied()
                .collect::<Vec<_>>(),
            want,
        )?;
    }
    Ok(())
}

/// A layout.json case whose bits are refused: by class, and a LayoutError
/// only when `layout_error` says so.
fn layout_refusal(case: &Case<'_>, class: &str) -> Result<(), String> {
    let key = Layout::parse(case.value["key"].as_str().ok_or("no key")?).map_err(err)?;
    let name = case.value["dtype"].as_str().ok_or("no dtype")?;
    let shape = shape_of(&case.value["shape"])?;
    let mut elements = Vec::new();
    bytes_of(&case.value["bits"], storage_width(name), &mut elements);
    let got = thin_formats::to_bytes(&elements, &key, dtype(name).map_err(err)?, shape);
    let layout_error = case.value["layout_error"].as_bool().unwrap_or(false);
    let want = if layout_error { "LayoutError" } else { class };
    expect_eq(got.map(drop).map_err(class_of), Err(want))
}

/// `msgpack.json`: each value's smallest encoding, and its read-back.
fn msgpack_case(case: &Case<'_>) -> Result<(), String> {
    if let Some(class) = case.value["refused"].as_str() {
        // A JSON float has no thin_formats::Value, so the Rust API cannot
        // be handed one: the type refuses it. The binding refuses a Python
        // float the same way, as a CodeObjectError (tests/test_formats.py).
        return match golden::msgpack_value(&case.value["value"]) {
            Some(value) => expect_eq(
                thin_formats::msgpack_write(&value).map_err(class_of),
                Err(class),
            ),
            None => expect_eq(class, "CodeObjectError"),
        };
    }
    let value = msgpack_of(&case.value["value"]);
    let want = hex_bytes(case.value["hex"].as_str().ok_or("no hex")?);
    expect_eq(
        thin_formats::msgpack_write(&value).map_err(err)?,
        want.clone(),
    )?;
    expect_eq(
        thin_formats::msgpack_read(&want).map_err(err)?,
        (value, want.len()),
    )
}

/// `codeobject.json` with `codeobject_k.co`: write, then read and code.
fn codeobject_file(case: &Case<'_>) -> Result<(), String> {
    let doc = case.value;
    let args = &doc["write_args"];
    let symbol = args["symbol"].as_str().ok_or("no symbol")?;
    let image = thin_formats::write(
        symbol,
        &hex_bytes(args["text_hex"].as_str().ok_or("no text")?),
        &hex_bytes(args["descriptor_hex"].as_str().ok_or("no descriptor")?),
        &msgpack_of(&args["metadata"]),
        args["e_flags"].as_u64().ok_or("no e_flags")? as u32,
        &thin_targets::MACHINES,
    )
    .map_err(err)?;
    let want = golden::bytes(
        "formats",
        doc["image_file"].as_str().ok_or("no image file")?,
    );
    expect_eq(image.len() as u64, doc["image_len"].as_u64().unwrap_or(0))?;
    expect_eq(
        format!("{:x}", Sha256::digest(&image)),
        doc["image_sha256"].as_str().unwrap_or("").to_owned(),
    )?;
    expect_eq(&image, &want)?;
    let object = thin_formats::read(&image).map_err(err)?;
    let sections: Vec<Value> = (object.sections.iter())
        .map(|s| serde_json::json!([s.name, s.kind, s.flags, s.addr, s.offset, s.size, s.link]))
        .collect();
    expect_eq(&Value::Array(sections), &doc["sections"])?;
    expect_eq(
        object.descriptor.to_vec(),
        hex_bytes(
            doc["descriptor_after_write_hex"]
                .as_str()
                .ok_or("no descriptor")?,
        ),
    )?;
    expect_eq(object.symbol.as_str(), symbol)?;
    expect_eq(
        object.target(&thin_targets::MACHINES).map_err(err)?,
        doc["target"].as_str().unwrap_or(""),
    )?;
    expect_eq(u64::from(object.lds()), doc["lds"].as_u64().unwrap_or(0))?;
    expect_eq(
        u64::from(object.kernarg()),
        doc["kernarg"].as_u64().unwrap_or(0),
    )?;
    write_e_flags(&doc["write_e_flags"])?;
    note_padding(&doc["note_padding"])?;
    let code = thin_formats::code(&image).map_err(err)?;
    expect_eq(
        code.metadata.as_str(),
        doc["code_metadata"].as_str().ok_or("no metadata")?,
    )?;
    for (key, digest) in thin_formats::Code::DIGESTS.iter().zip(code.digests()) {
        expect_eq(
            digest.as_str(),
            doc["code_digests"][key].as_str().unwrap_or(""),
        )?;
    }
    Ok(())
}

/// `codeobject.json` `note_padding`: descsz is the unpadded msgpack
/// length, the `.note` section the padded total.
fn note_padding(doc: &Value) -> Result<(), String> {
    let args = &doc["write_args"];
    let image = thin_formats::write(
        args["symbol"].as_str().ok_or("no symbol")?,
        &hex_bytes(args["text_hex"].as_str().ok_or("no text")?),
        &hex_bytes(args["descriptor_hex"].as_str().ok_or("no descriptor")?),
        &msgpack_of(&args["metadata"]),
        args["e_flags"].as_u64().ok_or("no e_flags")? as u32,
        &thin_targets::MACHINES,
    )
    .map_err(err)?;
    expect_eq(image.len() as u64, doc["image_len"].as_u64().unwrap_or(0))?;
    expect_eq(
        format!("{:x}", Sha256::digest(&image)),
        doc["image_sha256"].as_str().unwrap_or("").to_owned(),
    )?;
    let object = thin_formats::read(&image).map_err(err)?;
    let note = (object.sections.iter())
        .find(|s| s.name == ".note")
        .ok_or("no .note section")?;
    let at = note.offset as usize;
    let bytes = &image[at..at + note.size as usize];
    expect_eq(note.size, doc["note_size"].as_u64().unwrap_or(0))?;
    expect_eq(
        bytes.to_vec(),
        hex_bytes(doc["note_hex"].as_str().unwrap_or("")),
    )?;
    let word = |i: usize| u64::from(u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()));
    expect_eq(word(0), doc["namesz"].as_u64().unwrap_or(0))?;
    expect_eq(word(4), doc["descsz"].as_u64().unwrap_or(0))?;
    expect_eq(word(8), doc["type"].as_u64().unwrap_or(0))?;
    expect_eq(word(4), doc["msgpack_len"].as_u64().unwrap_or(0))?;
    expect_eq(note.size, 12 + 8 + word(4).div_ceil(4) * 4)
}

/// `codeobject.json` `write_e_flags`: any e_flags with a known machine
/// number is written and read back as given; any other is refused.
fn write_e_flags(doc: &Value) -> Result<(), String> {
    let text = hex_bytes(doc["text_hex"].as_str().ok_or("no text")?);
    let descriptor = hex_bytes(doc["descriptor_hex"].as_str().ok_or("no descriptor")?);
    let metadata = msgpack_of(&doc["metadata"]);
    for case in doc["cases"].as_array().ok_or("no cases")? {
        let e_flags = case["e_flags"].as_u64().ok_or("no e_flags")? as u32;
        let image = thin_formats::write(
            "k",
            &text,
            &descriptor,
            &metadata,
            e_flags,
            &thin_targets::MACHINES,
        );
        if let Some(class) = case["refused"].as_str() {
            expect_eq(image.map(drop).map_err(class_of), Err(class))?;
            continue;
        }
        let object = thin_formats::read(&image.map_err(err)?).map_err(err)?;
        expect_eq(
            u64::from(object.e_flags),
            case["read_e_flags"].as_u64().unwrap_or(0),
        )?;
        let target = object.target(&thin_targets::MACHINES).map_err(err)?;
        expect_eq(target, case["target"].as_str().unwrap_or(""))?;
    }
    Ok(())
}

/// `recipe.json`: Recipes applied to logical bytes [2][1][3] of u8.
fn recipe_case(case: &Case<'_>) -> Result<(), String> {
    let v = case.value;
    let sizes = |key: &str| -> Vec<usize> {
        v[key]
            .as_array()
            .into_iter()
            .flatten()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect()
    };
    let pairs = |key: &str| -> Vec<(usize, usize)> {
        v[key]
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| {
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect()
    };
    let recipe = layout::Recipe {
        pad: pairs("pad"),
        repeat: pairs("repeat"),
        fill: v["fill"].as_u64().ok_or("no fill")? as u8,
        ..layout::Recipe::new(
            v["rows"].as_u64().ok_or("no rows")? as usize,
            v["row_bytes"].as_u64().ok_or("no row_bytes")? as usize,
            sizes("dims"),
            sizes("perm"),
        )
    };
    let u8 = dtype("u8").map_err(err)?;
    let arranged =
        layout::Arranged::new(layout::Arrangement::Recipe(recipe), u8, (2, 1, 3), "recipe")
            .map_err(err)?;
    let nbytes = v["nbytes"].as_u64().ok_or("no nbytes")? as usize;
    expect_eq(arranged.nbytes(), nbytes)?;
    expect_eq(arranged.storage_shape(), Some(sizes("padded").as_slice()))?;
    let mut logical = Vec::new();
    bytes_of(&v["logical"], 1, &mut logical);
    let mut raw = vec![0u8; nbytes];
    arranged.to_bytes(&logical, &mut raw).map_err(err)?;
    let mut want = Vec::new();
    bytes_of(&v["bytes"], 1, &mut want);
    expect_eq(&raw, &want)?;
    let offsets: Vec<u64> = (0..nbytes as u64).collect();
    let mut got = vec![0i64; 3 * nbytes];
    arranged.coords(&offsets, &mut got).map_err(err)?;
    for axis in 0..3 {
        let want: Vec<i64> = (v["coords"][axis].as_array().ok_or("no coords")?)
            .iter()
            .map(|c| c.as_i64().unwrap())
            .collect();
        let got: Vec<i64> = got.iter().skip(axis).step_by(3).copied().collect();
        expect_eq(got, want)?;
    }
    let mut back = vec![0u8; logical.len()];
    arranged.from_bytes(&raw, &mut back).map_err(err)?;
    let mut inverted = Vec::new();
    bytes_of(&v["inverted"], 1, &mut inverted);
    expect_eq(&back, &inverted)?;
    if let Some(class) = v["copies_differ"].as_str() {
        *raw.last_mut().ok_or("no bytes")? ^= 1;
        expect_eq(
            arranged.from_bytes(&raw, &mut back).map_err(class_of),
            Err(class),
        )?;
    }
    Ok(())
}

fn main() {
    let mut suite = Suite::new("formats");
    suite.file("decode.json", decode_case);
    suite.section("encode.json", "cases", encode_case);
    suite.file("mx.json", mx_case);
    suite.file("pack.json", pack_case);
    suite.file("layout.json", layout_case);
    suite.section("recipe.json", "cases", recipe_case);
    suite.file("msgpack.json", msgpack_case);
    suite.whole("codeobject.json", codeobject_file);
    suite.finish();
}
