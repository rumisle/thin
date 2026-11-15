//! From a `.s` to a code object.

use thin_formats::Value;
use thin_target::{assemble, read_s, SFile, Target};

/// A refusal of the driver: an `AsmError` message.
pub type AsmResult<T> = Result<T, String>;

fn asm(e: thin_target::Error) -> String {
    match e {
        thin_target::Error::Asm(m)
        | thin_target::Error::Value(m)
        | thin_target::Error::Definition(m) => m,
        thin_target::Error::Key(m) => m,
    }
}

/// Reads a `.s` once.
pub fn source(text: &str) -> AsmResult<SFile> {
    read_s(text, false).map_err(asm)
}

/// The kernel symbol: `.amdhsa_kernel`'s argument.
pub fn symbol(file: &SFile) -> AsmResult<String> {
    let sym = file
        .once
        .get(".amdhsa_kernel")
        .map(|(a, _)| a.as_str())
        .unwrap_or("");
    if is_symbol(sym) {
        return Ok(sym.to_owned());
    }
    Err(format!(".amdhsa_kernel '{sym}': no kernel symbol"))
}

/// Whether a name is a symbol the assembler takes: `[A-Za-z_.][A-Za-z0-9_.$]*`
/// (an identifier of LLVM's assembler; `$` does not start one).
pub fn is_symbol(name: &str) -> bool {
    let mut chars = name.chars();
    let first = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || "_.".contains(c));
    first && chars.all(|c| c.is_ascii_alphanumeric() || "_.$".contains(c))
}

/// The target `.amdgcn_target` names.
pub fn target(file: &SFile) -> AsmResult<&'static Target> {
    let (arg, line) = file
        .once
        .get(".amdgcn_target")
        .map_or(("(none)", 0), |(a, l)| (a.as_str(), *l));
    let name = arg
        .strip_prefix("\"amdgcn-amd-amdhsa--")
        .and_then(|r| r.strip_suffix('"'))
        .filter(|n| {
            !n.is_empty()
                && n.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        });
    name.and_then(|n| thin_targets::get(n).ok()).ok_or_else(|| {
        format!("line {line}: .amdgcn_target {arg}: not a target this encoder knows")
    })
}

/// The code items' bytes.
pub fn encode(file: &SFile) -> AsmResult<Vec<u8>> {
    let t = target(file)?;
    assemble(&file.code, t.hooks()).map_err(asm)
}

/// The kernel descriptor.
pub fn descriptor(file: &SFile) -> AsmResult<[u8; 64]> {
    if !file.once.contains_key(".amdhsa_kernel") {
        return Err("no .amdhsa_kernel".into());
    }
    let t = target(file)?;
    thin_target::descriptor(file, t).map_err(asm)
}

/// The metadata document.
pub fn metadata(file: &SFile) -> AsmResult<Value> {
    if !file.once.contains_key(".amdgpu_metadata") {
        return Err("no .amdgpu_metadata block".into());
    }
    crate::meta::read(&file.meta)
}

/// The code object of a `.s` text.
pub fn code_object(text: &str) -> AsmResult<Vec<u8>> {
    let file = source(text)?;
    let sym = symbol(&file)?;
    let t = target(&file)?;
    let code = encode(&file)?;
    let desc = descriptor(&file)?;
    let meta = metadata(&file)?;
    thin_formats::write(
        &sym,
        &code,
        &desc,
        &meta,
        t.hooks().e_flags(),
        &thin_targets::MACHINES,
    )
    .map_err(|e| e.to_string())
}
