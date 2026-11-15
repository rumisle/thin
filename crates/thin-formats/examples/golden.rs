//! Evaluate generator requests through the native format implementation.
//! Reads a JSON request list from stdin and writes one result list to stdout.

use std::io::Read;

use serde_json::{json, Value};
use thin_formats::{layout, Error};

fn class(error: Error) -> &'static str {
    match error {
        Error::Type(_) => "TypeError",
        Error::Value(_) => "ValueError",
        Error::Layout(_) => "LayoutError",
        Error::CodeObject(_) => "CodeObjectError",
    }
}

fn sizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

fn pairs(v: &Value) -> Vec<(usize, usize)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v[0].as_u64().unwrap() as usize,
                v[1].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

fn nested(bytes: &[u8], rows: usize, columns: usize) -> Value {
    json!(bytes
        .chunks(rows * columns)
        .map(|matrix| matrix
            .chunks(columns)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

fn value(v: &Value) -> Result<thin_formats::Value, Error> {
    use thin_formats::Value as V;
    Ok(match v {
        Value::Null => V::Nil,
        Value::Bool(b) => V::Bool(*b),
        Value::String(s) => V::Str(s.clone()),
        Value::Number(n) => V::Int(
            n.as_str()
                .parse()
                .map_err(|_| Error::CodeObject("non-integer metadata".into()))?,
        ),
        Value::Array(a) => V::Array(a.iter().map(value).collect::<Result<_, _>>()?),
        Value::Object(m) => {
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            V::Map(
                keys.into_iter()
                    .map(|k| Ok((k.clone(), value(&m[k])?)))
                    .collect::<Result<_, Error>>()?,
            )
        }
    })
}

fn run(v: &Value) -> Result<Value, Error> {
    if v["op"] == "msgpack" {
        let bytes = thin_formats::msgpack_write(&value(&v["value"])?)?;
        return Ok(json!({"hex":bytes.iter().map(|b|format!("{b:02x}")).collect::<String>()}));
    }
    let shape = sizes(&v["shape"]);
    let shape = (shape[0], shape[1], shape[2]);
    let logical: Vec<u8> = v["bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    if v["op"] == "pack" {
        let dtype = thin_formats::dtype(v["dtype"].as_str().unwrap())?;
        let row = (shape.2 * dtype.width as usize).div_ceil(8);
        let mut out = vec![0; shape.0 * shape.1 * row];
        layout::pack(&logical, dtype, shape, &mut out)?;
        return Ok(json!({"logical":nested(&out,shape.1,row)}));
    }
    let recipe = layout::Recipe {
        pad: pairs(&v["pad"]),
        repeat: pairs(&v["repeat"]),
        fill: v["fill"].as_u64().unwrap() as u8,
        ..layout::Recipe::new(shape.1, shape.2, sizes(&v["dims"]), sizes(&v["perm"]))
    };
    let arranged = layout::Arranged::new(
        layout::Arrangement::Recipe(recipe),
        thin_formats::dtype("u8")?,
        shape,
        "recipe",
    )?;
    let n = arranged.nbytes();
    let mut out = vec![0; n];
    arranged.to_bytes(&logical, &mut out)?;
    let mut coords = vec![0i64; 3 * n];
    arranged.coords(&(0..n as u64).collect::<Vec<_>>(), &mut coords)?;
    let mut back = vec![0; logical.len()];
    arranged.from_bytes(&out, &mut back)?;
    let mut result = json!({"padded":arranged.storage_shape(),"nbytes":n,"bytes":out,"coords":(0..3).map(|axis|coords.iter().skip(axis).step_by(3).copied().collect::<Vec<_>>()).collect::<Vec<_>>(),"inverted":nested(&back,shape.1,shape.2)});
    if !v["repeat"].as_array().unwrap().is_empty() {
        *out.last_mut().unwrap() ^= 1;
        result["copies_differ"] =
            Value::from(class(arranged.from_bytes(&out, &mut back).unwrap_err()));
    }
    Ok(result)
}

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let requests: Vec<Value> = serde_json::from_str(&text).unwrap();
    let results: Vec<_> = requests
        .iter()
        .map(|v| run(v).unwrap_or_else(|e| json!({"refused":class(e)})))
        .collect();
    println!("{}", serde_json::to_string(&results).unwrap());
}
