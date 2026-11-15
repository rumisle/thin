//! Golden checks for provenance records, byte rules and matrix/store checking.

use serde_json::Value;
use thin_emu::gemm::{DCols, GemmHooks, Kind, Layout, Problem, RecordOperand, Shape};
use thin_emu::tags::{
    self, rules, Checker, Form, MatrixForm, MatrixOp, MatrixOperand, OperandFormat, ProblemFacts,
    StoreCall, Tables,
};
use thin_test_support::golden_tests::expect_eq;
use thin_test_support::tagjson::{lane, record, show, tag};
use thin_test_support::{Case, Suite};

fn i(v: &Value) -> i64 {
    v.as_i64().unwrap_or(0)
}

/// `encoding.json`: the format and the numbering of a fresh table.
fn encoding(case: &Case<'_>) -> Result<(), String> {
    let doc = case.value;
    let mut t = Tables::default();
    let c = &doc["constants"];
    let consts = [
        ("ACC_MARK", tags::ACC_MARK),
        ("ACC_BASE", tags::ACC_BASE),
        ("ENC_SPAN", tags::ENC_SPAN),
        ("SUM_BASE", tags::SUM_BASE),
        ("TAG_SHIFT", i64::from(tags::TAG_SHIFT)),
    ];
    for (name, value) in consts {
        expect_eq((name, value), (name, i(&c[name])))?;
    }
    for m in doc["memory"].as_array().into_iter().flatten() {
        let got = tags::mem_tag(i(&m["buffer"]) as usize, i(&m["offset"]) as u64);
        expect_eq(got, i(&m["tag"]))?;
        expect_eq(
            (tags::tag_buf(got), tags::tag_off(got) as i64),
            (i(&m["buffer"]), i(&m["offset"])),
        )?;
    }
    for r in doc["records"].as_array().into_iter().flatten() {
        let (got, form) = if let Some(p) = r.get("poison") {
            let g = t.poison(p.as_str().unwrap_or(""));
            (g, Form::Table((-g - 2) as usize))
        } else {
            t.record_form(&record(r))
        };
        expect_eq(got, i(&r["tag"]))?;
        match form {
            Form::Encoded(kid) => expect_eq(Some(kid as i64), r["kid"].as_i64())?,
            Form::Table(rid) => expect_eq(Some(rid as i64), r["rid"].as_i64())?,
        }
        if let Some(rec) = r["decoded"].get("rec") {
            expect_eq(t.record_of(got), record(rec))?;
        }
    }
    for s in doc["sums"].as_array().into_iter().flatten() {
        let (x, y) = (lane(&mut t, &s["x"]), lane(&mut t, &s["y"]));
        let got = rules::sum_lane(&mut t, &x, &y);
        let raw: Vec<i64> = s["raw"].as_array().unwrap().iter().map(i).collect();
        expect_eq(got.to_vec(), raw)?;
        expect_eq(got, lane(&mut t, &s["out"]))?;
    }
    for s in doc["sum_sources"].as_array().into_iter().flatten() {
        let g = tag(&mut t, &s["tag"]);
        let want: Vec<i64> = s["sources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| tag(&mut t, v))
            .collect();
        expect_eq(t.sources(g), want)?;
    }
    Ok(())
}

fn problems(t: &mut Tables, doc: &Value) {
    for p in doc["problems"].as_array().into_iter().flatten() {
        let facts = ProblemFacts {
            k: i(&p["K"]),
            n: p["N"].as_i64(),
            source: p["record_operand"]["source"].as_i64(),
        };
        let pid = t.register(facts);
        assert_eq!(pid, i(&p["pid"]));
    }
}

/// `union.json`: combine and the float rule.
fn union(case: &Case<'_>) -> Result<(), String> {
    let doc = case.value;
    let mut t = Tables::default();
    problems(&mut t, doc);
    for c in doc["union"].as_array().into_iter().flatten() {
        let (x, y) = (tag(&mut t, &c["x"]), tag(&mut t, &c["y"]));
        let got = tags::combine::combine(&mut t, x, y, c["stat"].as_bool().unwrap_or(false));
        let want = tag(&mut t, &c["out"]);
        if got != want {
            return Err(format!(
                "{}: {} != {}",
                c["case"],
                show(&t, got),
                show(&t, want)
            ));
        }
    }
    for c in doc["float"].as_array().into_iter().flatten() {
        let (x, y) = (lane(&mut t, &c["x"]), lane(&mut t, &c["y"]));
        let mut ctx = rules::Ctx {
            tables: &mut t,
            sum_mode: c["sum_mode"].as_bool().unwrap_or(false),
        };
        let stat = c["stat"].as_bool().unwrap_or(false);
        let got = rules::float(
            &mut ctx,
            &[x, x],
            &[y, y],
            stat,
            c["add"].as_bool().unwrap_or(false),
        );
        match (got, c.get("error")) {
            (Err(e), Some(w)) => expect_eq(Some(e.message.as_str()), w.as_str())?,
            (Ok(lanes), None) => {
                let want = lane(&mut t, &c["out"]);
                if lanes[1] != want {
                    return Err(format!("{}: {:?} != {:?}", c["case"], lanes[1], want));
                }
            }
            (got, want) => return Err(format!("{}: {got:?}, want {want:?}", c["case"])),
        }
    }
    Ok(())
}

/// `bytes.json`: the bit rules and the dword provenance.
fn bytes(case: &Case<'_>) -> Result<(), String> {
    let doc = case.value;
    let mut t = Tables::default();
    for (key, rule) in [
        (
            "and",
            rules::and_lane as fn(u32, &[i64; 4], u32, &[i64; 4]) -> [i64; 4],
        ),
        ("or", rules::or_lane),
    ] {
        for c in doc[key].as_array().into_iter().flatten() {
            let (xt, yt) = (lane(&mut t, &c["xt"]), lane(&mut t, &c["yt"]));
            let got = rule(i(&c["x"]) as u32, &xt, i(&c["y"]) as u32, &yt);
            expect_eq(got, lane(&mut t, &c["out"]))?;
        }
    }
    for c in doc["shift"].as_array().into_iter().flatten() {
        let l = lane(&mut t, &c["t"]);
        let got = rules::shift_lane(
            &l,
            i(&c["amount"]) as u32,
            c["left"].as_bool().unwrap_or(false),
        );
        expect_eq(got, lane(&mut t, &c["out"]))?;
    }
    for c in doc["lane_source"].as_array().into_iter().flatten() {
        let l = lane(&mut t, &c["t"]);
        expect_eq(tags::src(&l), tag(&mut t, &c["out"]))?;
    }
    let per = &doc["shift_per_lane"];
    let l = lane(&mut t, &per["t"]);
    for (lane_id, want) in per["lanes"].as_object().into_iter().flatten() {
        let n: u32 = lane_id.parse().unwrap_or(0);
        let amount = if n == 5 { 3 } else { 8 * (n % 4) };
        expect_eq(rules::shift_lane(&l, amount, true), lane(&mut t, want))?;
    }
    Ok(())
}

/// A layout from the vectors' notation; a `table` of `off div L` / `off mod L`
/// is `rowmajor(kind, L, 1)`.
fn layout(v: &Value) -> Option<Layout> {
    let (name, p) = v.as_object()?.iter().next()?;
    let get = |k: &str| p[k].as_i64().unwrap_or(0);
    let kind = Kind::named(p["kind"].as_str().unwrap_or("D"));
    let shape = match name.as_str() {
        "rowmajor" | "rowmajor_d" => {
            let s = (
                get("ld_bytes"),
                p["elem_bytes"].as_i64().unwrap_or(1),
                get("rows_per_batch"),
                get("batch_bytes"),
            );
            if name == "rowmajor" {
                Shape::RowMajor {
                    ld: s.0,
                    elem: s.1,
                    rows_per_batch: s.2,
                    batch_bytes: s.3,
                }
            } else {
                Shape::RowMajorD {
                    ld: s.0,
                    elem: s.1,
                    rows_per_batch: s.2,
                    batch_bytes: s.3,
                }
            }
        }
        "table" => {
            let ld = p["row"].as_str()?.rsplit(' ').next()?.parse().ok()?;
            Shape::RowMajor {
                ld,
                elem: 1,
                rows_per_batch: 0,
                batch_bytes: 0,
            }
        }
        _ => return None,
    };
    let kind = if name == "rowmajor_d" { Kind::D } else { kind };
    Some(Layout::new(kind, shape))
}

/// `layouts.json`: each constructor's decode.
fn layouts(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let l = layout(&c["layout"]).ok_or("no layout")?;
    expect_eq(l.kind.name(), c["kind"].as_str().unwrap_or(""))?;
    expect_eq(l.elem_bytes(), i(&c["elem_bytes"]))?;
    let offs: Vec<u64> = c["offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| i(v) as u64)
        .collect();
    for (field, want) in c["decoded"].as_object().into_iter().flatten() {
        let got: Vec<i64> = offs
            .iter()
            .map(|o| {
                let d = l.decode(*o);
                match field.as_str() {
                    "row" | "m" => d.row,
                    "k" | "n" => d.k,
                    _ => d.sub,
                }
            })
            .collect();
        let want: Vec<i64> = want.as_array().unwrap().iter().map(i).collect();
        expect_eq((field.as_str(), got), (field.as_str(), want))?;
    }
    Ok(())
}

/// A table over 0..512 from a vector's formula text.
fn rows_formula(f: &str) -> Vec<i64> {
    (0..512)
        .map(|m| match f {
            "m + 16" => m + 16,
            "-1 for m = 0, else m" => {
                if m == 0 {
                    -1
                } else {
                    m
                }
            }
            _ => m,
        })
        .collect()
}

/// `stores.json`: the store and atomic checks.
fn stores(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let mut t = Tables::default();
    let mut p = Problem::new(vec![("D".into(), layout(&c["layout"]))], i(&c["K"]));
    p.fail_fast = false;
    if let Some(f) = c["d_rows"].as_str() {
        p.d_rows = Some(rows_formula(f));
    }
    if c["d_cols"].as_str() == Some("(2n, 2n + 1)") {
        let a: Vec<i64> = (0..512).map(|n| 2 * n).collect();
        let b: Vec<i64> = (0..512).map(|n| 2 * n + 1).collect();
        p.d_cols = Some(DCols::Tuple(vec![DCols::Row(a), DCols::Row(b)]));
    }
    if let Some(f) = c["d_kblocks"].as_str() {
        let set: std::collections::BTreeSet<i64> = f
            .trim_matches(['{', '}'])
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        p.d_kblocks = Some([(0, set)].into());
    }
    if let Some(sc) = c["scatter"].as_object() {
        p.scatter = Some(
            sc.iter()
                .map(|(k, v)| {
                    (
                        k.parse().unwrap_or(0),
                        v.as_array().unwrap().iter().map(i).collect(),
                    )
                })
                .collect(),
        );
    }
    let mut h = GemmHooks::new(&mut t, vec![p], vec!["D".into()]);
    let mut offs = Vec::new();
    let mut tags_ = Vec::new();
    for b in c["bytes"].as_array().into_iter().flatten() {
        offs.push(i(&b[0]) as u64);
        tags_.push(tag(&mut t, &b[1]));
    }
    let data = vec![vec![0u8; offs.len()]];
    let call = StoreCall {
        buffer: 0,
        name: "D",
        offsets: &[offs],
        data: &data,
        tags: &[tags_],
        wave: 0,
    };
    let got = if c["check"] == "atomic" {
        h.atomic(&mut t, &call).map(|_| ())
    } else {
        h.store(&mut t, &call)
    };
    expect_eq(
        got.err().map(|e| e.message),
        c["raised"].as_str().map(str::to_owned),
    )?;
    let p = &h.probs[0];
    let want: Vec<String> = c["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_str().unwrap_or("").to_owned())
        .collect();
    expect_eq(&p.errors, &want)?;
    let got: Vec<Vec<i64>> = p
        .stores()
        .iter()
        .map(|((m, n, a), k)| {
            let mut v = vec![*m, *n];
            v.extend(a.iter());
            v.push(*k as i64);
            v
        })
        .collect();
    let want: Vec<Vec<i64>> = c["stores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(i).collect())
        .collect();
    expect_eq(got, want)
}

fn form(f: &Value) -> (MatrixForm, OperandFormat) {
    let (w, r, k) = (i(&f["W"]) as usize, i(&f["R"]) as usize, i(&f["K"]));
    let hwk: Vec<Vec<i64>> = f["hwk"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q.as_array().unwrap().iter().map(i).collect())
        .collect();
    let sixteen = f["elem"] == "16-bit";
    let fmt = OperandFormat {
        name: f["elem"].as_str().unwrap_or("").into(),
        bytes: if sixteen { 16 } else { 32 },
        elem_bytes: if sixteen { 2 } else { 1 },
        hwk,
    };
    // d_map[e][l] = [src0 row slot, src1 row slot].
    let d_map = f["d_map"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            e.as_array()
                .unwrap()
                .iter()
                .map(|p| (i(&p[0]) as usize, i(&p[1]) as usize))
                .collect()
        })
        .collect();
    (
        MatrixForm {
            w,
            r,
            k_hw: k,
            d_map,
        },
        fmt,
    )
}

fn operand_tags(t: &mut Tables, s: &Value, w: usize, nbytes: usize) -> Vec<Vec<i64>> {
    let mut out = vec![vec![0; nbytes]; w];
    if let Some(b) = s["buffer"].as_i64() {
        for (l, offs) in s["offsets"].as_array().into_iter().flatten().enumerate() {
            for (p, o) in offs.as_array().into_iter().flatten().enumerate() {
                if let (Some(o), true) = (o.as_i64(), l < w && p < nbytes) {
                    out[l][p] = tags::mem_tag(b as usize, o as u64);
                }
            }
        }
    }
    for e in s["other"].as_array().into_iter().flatten() {
        let (l, p) = (i(&e[0]) as usize, i(&e[1]) as usize);
        out[l][p] = tag(t, &e[2]);
    }
    out
}

/// `matrix.json.gz`: the matrix-op check on abstract forms.
fn matrix(case: &Case<'_>) -> Result<(), String> {
    let c = case.value;
    let mut t = Tables::default();
    let (form, fmt) = form(&c["form"]);
    let p = &c["problem"];
    let names: Vec<String> = p["layouts"].as_object().unwrap().keys().cloned().collect();
    let mut prob = Problem::new(
        p["layouts"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, l)| (n.clone(), layout(l)))
            .collect(),
        i(&p["K"]),
    );
    prob.fail_fast = false;
    if let Some(ro) = p.get("record_operand") {
        let src = &ro["source"];
        let pid = t.register(ProblemFacts {
            k: i(&src["K"]),
            n: src["N"].as_i64(),
            source: None,
        });
        prob.record_operand = Some(RecordOperand {
            side: i64::from(ro["side"] == "B"),
            source: pid,
            map: None,
        });
    }
    let mut h = GemmHooks::new(&mut t, vec![prob], names.clone());
    let dd = form.d_dwords();
    let mut acc = tags::AccState::new(dd, form.w);
    let mut marked = vec![false; dd];
    let dst = format!("a[0:{}]", dd - 1);
    for op in c["ops"].as_array().into_iter().flatten() {
        let s0 = operand_tags(&mut t, &op["src0"], form.w, fmt.bytes);
        let s1 = operand_tags(&mut t, &op["src1"], form.w, fmt.bytes);
        let mop = MatrixOp {
            form: &form,
            text: "<ins>",
            dst: &dst,
            entry: 0,
            marked: marked.clone(),
            c_zero: op["c_fresh"].as_bool().unwrap_or(false),
            src: [
                MatrixOperand {
                    tags: &s0,
                    format: &fmt,
                },
                MatrixOperand {
                    tags: &s1,
                    format: &fmt,
                },
            ],
        };
        h.matrix(&mut t, &mop, &mut acc).map_err(|e| e.message)?;
        marked = vec![true; dd];
    }
    let mut errors: Vec<String> = Vec::new();
    for e in &h.probs[0].errors {
        if !errors.contains(e) {
            errors.push(e.clone());
        }
    }
    let want: Vec<String> = c["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_str().unwrap_or("").to_owned())
        .collect();
    expect_eq(errors, want)?;
    for (e, want) in c["acc"].as_array().into_iter().flatten().enumerate() {
        let entry = &acc.entries[e];
        let any = entry
            .m
            .iter()
            .zip(&entry.n)
            .any(|(m, n)| *m >= 0 && *n >= 0);
        if want.is_null() {
            expect_eq(any, false)?;
            continue;
        }
        let ints = |v: &Value| v.as_array().unwrap().iter().map(i).collect::<Vec<i64>>();
        expect_eq(
            entry.k.iter().copied().collect::<Vec<_>>(),
            ints(&want["k"]),
        )?;
        expect_eq(entry.pid, i(&want["pid"]))?;
        expect_eq(&entry.m, &ints(&want["m"]))?;
        expect_eq(&entry.n, &ints(&want["n"]))?;
    }
    Ok(())
}

fn main() {
    let mut suite = Suite::new("emu-tags");
    suite.whole("encoding.json", encoding);
    suite.whole("union.json", union);
    suite.whole("bytes.json", bytes);
    suite.section("layouts.json", "cases", layouts);
    suite.section("matrix.json.gz", "cases", matrix);
    suite.section("stores.json", "cases", stores);
    suite.finish();
}
