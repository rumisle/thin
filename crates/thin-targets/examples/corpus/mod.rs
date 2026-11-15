//! Shared input preparation and ELF corpus walker used by the examples.
//!
//! The symbol/range walker shared by the examples. Pass the `.co` files to
//! measure, not a directory containing duplicate `.elf` extractions or
//! unrelated helper libraries. Compressed offload bundles
//! are unbundled for the selected target, one image at a time.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Loads an ELF or unbundles an offload image for `target`. Bundles require
/// clang-offload-bundler on PATH; stdout avoids a persistent corpus copy.
pub fn image(path: &Path, target: &str) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !bytes.starts_with(b"CCOB") && !bytes.starts_with(b"__CLANG_OFFLOAD_BUNDLE__") {
        return Ok(bytes);
    }
    let output = Command::new("clang-offload-bundler")
        .args(["--unbundle", "--type=o"])
        .arg(format!("--targets=hipv4-amdgcn-amd-amdhsa--{target}"))
        .arg(format!("--input={}", path.display()))
        .arg("--output=/dev/stdout")
        .output()
        .map_err(|e| format!("{}: unbundle: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{}: unbundle: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}

fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}

/// A kernel: its name and its code's byte range in the image.
pub struct Kernel {
    /// The function symbol, or the executable section fallback.
    pub name: String,
    /// Inclusive file offset.
    pub start: usize,
    /// Exclusive file offset, with unsized-symbol padding removed.
    pub end: usize,
}

fn cstr(b: &[u8], at: usize) -> String {
    let end = b[at..]
        .iter()
        .position(|&c| c == 0)
        .map_or(b.len(), |n| at + n);
    String::from_utf8_lossy(&b[at..end]).into_owned()
}

/// The kernels of an ELF64 image.
pub fn kernels(img: &[u8]) -> Result<Vec<Kernel>, String> {
    if img.len() < 64 || &img[..4] != b"\x7fELF" || img[4] != 2 {
        return Err("not an ELF64".into());
    }
    let (shoff, shentsize, shnum) = (u64_at(img, 0x28), u16_at(img, 0x3A), u16_at(img, 0x3C));
    let sh = |i: usize| shoff + i * shentsize;
    let section = |i: usize| {
        let h = sh(i);
        (
            u32_at(img, h + 4),
            u64_at(img, h + 8),
            u64_at(img, h + 0x10),
            u64_at(img, h + 0x18),
            u64_at(img, h + 0x20),
            u32_at(img, h + 0x28),
        )
    };
    let mut out = Vec::new();
    let mut exec = Vec::new();
    for i in 0..shnum {
        let (kind, flags, addr, offset, size, link) = section(i);
        if flags & 4 != 0 && kind == 1 {
            exec.push((i, addr, offset, size));
        }
        if kind != 2 {
            continue;
        }
        let strtab = section(link).3;
        for s in (offset..offset + size).step_by(24) {
            let (info, shndx) = (img[s + 4], u16_at(img, s + 6));
            let (value, len) = (u64_at(img, s + 8), u64_at(img, s + 16));
            let Some(&(_, saddr, soff, _)) = exec.iter().find(|e| e.0 == shndx) else {
                continue;
            };
            if info & 0xF == 2 {
                let start = soff + (value - saddr);
                out.push(Kernel {
                    name: cstr(img, strtab + u32_at(img, s)),
                    start,
                    // A sized symbol ends there; an unsized one (some
                    // toolchains, e.g. Tensile, emit them) at the next.
                    end: if len > 0 { start + len } else { 0 },
                });
            }
        }
        let end_of = |start: usize| {
            let (_, _, off, size) = *exec
                .iter()
                .find(|e| e.2 <= start && start < e.2 + e.3)
                .unwrap();
            off + size
        };
        out.sort_by_key(|k| k.start);
        out.dedup_by_key(|k| k.start);
        for i in 0..out.len() {
            if out[i].end == 0 {
                let section_end = end_of(out[i].start);
                let next = out
                    .get(i + 1)
                    .map_or(section_end, |k| k.start.min(section_end));
                // Drop the alignment padding: zero words and s_code_end.
                let mut end = next;
                while end >= out[i].start + 4 && matches!(u32_at(img, end - 4), 0 | 0xBF9F_0000) {
                    end -= 4;
                }
                out[i].end = end;
            }
        }
    }
    if out.is_empty() {
        for (_, _, offset, size) in exec {
            out.push(Kernel {
                name: "(text)".into(),
                start: offset,
                end: offset + size,
            });
        }
    }
    Ok(out)
}

/// Expands input files and directories in deterministic path order.
pub fn files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(p)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            out.extend(files(&entries));
        } else {
            out.push(p.clone());
        }
    }
    out
}
