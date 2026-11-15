"""The co-residency probe kernel."""

from __future__ import annotations


def gen_arrive(wgsize: int = 64):
    """Each workgroup adds 1 to counter[0], then every wave reads it `spin` times;
    out[wgid] is the largest count its first wave saw. Every workgroup sees n
    only if all n were resident at once: no wave finishes before its spin ends."""
    from thin import KernelDef
    from thin.dsl import Args, O, ptr, u32
    from thin.dsl import max as dmax

    def body(b, k):
        V = b.v  # noqa: N806 - the DSL's spelling
        counter, out, spin = k.load()
        zero = V.u32(0, name="zero")
        with b.lanes((k.tid & 31) == 0):  # every wave stays until its spin ends
            with b.lanes(k.tid == 0):
                one, old = V.u32(1, name="one"), V.u32(name="old")
                # Request the atomic's old value; `patched` adds the returning
                # policy and system scope explicitly to every access.
                b.emit(O.op("global_atomic_add_u32", old, zero, one, counter))
                b.wait(old, "the old count")
            seen = V.u32(0, name="seen")
            for _ in b.loop(spin, name="k"):
                c = V.u32(name="c")
                b.emit(O.global_load(c, zero, counter))
                b.wait(c, "the count")
                seen[...] = dmax(seen, c)
            with b.lanes(k.tid == 0):
                at = V.u32(k.wgid * 4, name="at")
                b.emit(O.global_store(at, seen, out))

    args = Args(counter=ptr, out=ptr, spin=u32.within(1, 1 << 24))
    return KernelDef(f"arrive_b{wgsize}", args, body, wgsize=wgsize, max_grid=(65536,))


def patched(k_def):
    """A Kernel of the KernelDef with its atomic returning and every global access at
    SCOPE_SYS, patched into the .s before it is assembled."""
    import re

    from thin import codegen
    from thin.runtime.kernel import Kernel

    asm = k_def.compile().text
    asm = re.sub(r"^(\tglobal_atomic\S+ [^/\n]*?)(\s*(//.*)?)$", r"\1 th:TH_ATOMIC_RETURN\2", asm,
                 flags=re.M)  # fmt: skip
    asm = re.sub(r"^(\tglobal_\S+ [^/\n]*?)(\s*(//.*)?)$", r"\1 scope:SCOPE_SYS\2", asm, flags=re.M)
    _, manifest = codegen.split_manifest(asm)
    co = codegen.code_object(asm)
    return Kernel(manifest.symbol, asm, manifest, co, {}, f"hw:{manifest.symbol}+rtn+sys", k_def)
