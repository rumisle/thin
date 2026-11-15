//! Encodes gfx1201 lines from stdin, one per line: the bytes in hex, or
//! `! <refusal>`; with `--decode`, also the decoded line. A check against
//! `llvm-mc -triple=amdgcn -mcpu=gfx1201 -show-encoding`.

use std::io::BufRead;

use thin_target::Encoded;
use thin_targets::{decode, get};

fn main() {
    let decode_too = std::env::args().any(|a| a == "--decode");
    let hooks = get("gfx1201").expect("gfx1201").hooks();
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("utf-8");
        match hooks.encode(line.trim()) {
            Ok(Encoded::Bytes(b)) => {
                let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
                let back = if decode_too {
                    match decode("gfx1201", &b) {
                        Ok(is) => format!("  {}", is[0]),
                        Err(e) => format!("  ! {e}"),
                    }
                } else {
                    String::new()
                };
                println!("{hex}{back}");
            }
            Ok(Encoded::Branch { raw, label }) => println!("branch {raw:08x} {label}"),
            Err(e) => println!("! {e}"),
        }
    }
}
