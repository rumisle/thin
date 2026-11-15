# Generated ISA tables

Each `<target>/` directory holds the two tables `tools/isagen` generates
from AMD's machine-readable ISA bundle, as canonical JSON
(keys sorted, separators `,` and `:`, ASCII):

- `isa_tables.json`: the encodings and opcodes the encoder reads;
- `operand_table.json`: the operand rows the forms are built from.

They are checked in; nothing generates them at build time, and the bundle
is never part of the repository or a package. Their sha256 must equal
`crates/thin-targets/tests/data/target/tables_manifest.json`, which a test checks.

## Regenerating

The bundle is pinned: `AMD_GPU_MR_ISA_XML_2025_09_05.zip` from AMD's
GPUOpen downloads, <https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2025_09_05.zip>,
sha256 `6c656832cffa8198fe8db72007697d592fdeadc14194b9b4fbdcc8ec7f5c5223`.
The generator refuses any other file.

    curl -LO https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2025_09_05.zip
    python -m tools.isagen --bundle AMD_GPU_MR_ISA_XML_2025_09_05.zip --target gfx1201
    THIN_ISA_BUNDLE=$PWD/AMD_GPU_MR_ISA_XML_2025_09_05.zip pytest tests/test_isagen.py

The generator uses the standard library only and takes about a quarter of
a second per target.

## Upstream licence

The pinned XML's `Document` header declares `License` = `MIT`, `Sensitivity`
= `AMD Public Use.`, and copyright (c) 2025 Advanced Micro Devices, Inc., or
its affiliates. NOTICE reproduces the copyright and MIT permission terms for
these generated tables and their golden copies. The XML bundle itself is
not distributed with thin.
