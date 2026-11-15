"""The ISA table generator.

It rebuilds a target's ISA tables and operand table from AMD's pinned,
machine-readable ISA XML bundle, and writes them as canonical JSON into
crates/thin-targets/data/<target>/, where they are checked in. It is a build
tool: never run while compiling, and the bundle is never part of the package.

    python -m tools.isagen --bundle AMD_GPU_MR_ISA_XML_2025_09_05.zip --target gfx1201
"""
