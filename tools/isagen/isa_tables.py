"""The ISA tables: encodings and opcodes, for the encoder."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict

from tools.isagen.targets import TargetRules

STRIP = [
    ("_VOP_SDWA_SDST_ENC", "_SDWA_SDST"),
    ("_INST_LITERAL", "_LIT"),
    ("_VOP_DPP16", "_DPP16"),
    ("_VOP_DPP8", "_DPP8"),
    ("_VOP_SDWA", "_SDWA"),
    ("_VOP_DPP", "_DPP16"),
    ("_MFMA", "_MFMA"),
]
"""The source spellings of the variant suffixes, in replacement order."""

RENAME = {
    "VOP3_SDST_ENC": "VOP3SD",
    "VOP3_SDST_ENC_LIT": "VOP3SD_LIT",
    "VOP3_SDST_ENC_DPP16": "VOP3SD_DPP16",
    "VOP3_SDST_ENC_DPP8": "VOP3SD_DPP8",
    "VOPDXY": "VOPD",
    "VOPDXY_LIT": "VOPD_LIT",
    "VDS": "DS",
}

SUFFIXES = ["_LIT", "_DPP16", "_DPP8", "_SDWA_SDST", "_SDWA", "_MFMA"]
"""The variant suffixes, in order."""

SUFFIX_ONLY = ["_LIT", "_DPP16", "_DPP16", "_DPP8", "_SDWA", "_SDWA_SDST", "_MFMA"]
"""The order in which an opcode's encodings are tested for a shared suffix."""

FIELD_NAMES = [
    ("opsel_hi_2", "opsel_hi2"),
    ("op_sel_hi_2", "opsel_hi2"),
    ("op_sel", "opsel"),
    ("bound_ctrl", "bc"),
    ("tgt", "target"),
    ("row_en", "row"),
    ("unorm", "unrm"),
    ("clamp", "clmp"),
    ("wait_exp", "waitexp"),
    ("simm32", "literal"),
    ("dpp_ctrl", "dpp"),
]

FIXES = {
    "rdna4": {
        "SOP1": {80: "S_GET_BARRIER_STATE", 81: "S_BARRIER_INIT", 82: "S_BARRIER_JOIN"},
        "SOPP": {9: "S_WAITCNT", 21: "S_BARRIER_LEAVE"},
    }
}
"""Each architecture's opcodes the XML lacks."""

VGPR_FIELDS = {
    "vdst",
    "vdstx",
    "vsrc0",
    "vsrc1",
    "vsrc2",
    "vsrc3",
    "vsrcx1",
    "vsrcy1",
    "vaddr",
    "vdata",
    "data",
    "data0",
    "data1",
    "addr",
    "vsrc",
}


def strip(encoding: str) -> str:
    """An encoding name without ``ENC_``, ``_NSA1`` and the variant spellings."""
    name = encoding.removeprefix("ENC_").replace("_NSA1", "")
    for old, new in STRIP:
        name = name.replace(old, new)
    return name


def rename(name: str) -> str:
    """The class name a stripped encoding is stored under."""
    return RENAME.get(name, name)


def base(name: str) -> str:
    """A class name with every variant suffix removed."""
    for suffix in SUFFIXES:
        name = name.replace(suffix, "")
    return name


def field_name(name: str) -> str:
    """An XML field name, lowercased and renamed."""
    name = name.lower()
    for old, new in FIELD_NAMES:
        name = name.replace(old, new)
    return name


def flat_map(encoding: str, instruction: str) -> str:
    """The class a flat-family encoding gives an instruction."""
    if encoding in ("FLAT_GLBL", "FLAT_GLOBAL"):
        return "GLOBAL"
    if encoding == "FLAT_SCRATCH":
        return "SCRATCH"
    if encoding in ("FLAT", "VFLAT", "VGLOBAL", "VSCRATCH"):
        prefix = "V" if encoding.startswith("V") else ""
        if instruction.startswith("GLOBAL_"):
            return prefix + "GLOBAL"
        if instruction.startswith("SCRATCH_"):
            return prefix + "SCRATCH"
        return prefix + "FLAT"
    return encoding


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    return "" if found is None or found.text is None else found.text.strip()


def _encodings(isa: ET.Element) -> dict[str, tuple[list, int | None]]:
    """Each kept encoding's raw fields (name, hi, lo) and identifier bits."""
    encodings = {}
    for encoding in isa.iterfind("Encodings/Encoding"):
        name = _text(encoding, "EncodingName")
        kept = name.startswith("ENC_") or name in ("VOP3_SDST_ENC", "VOPDXY")
        kept = kept or any(source in name for source, _ in STRIP)
        if not kept or "NSA" in name:
            continue
        fields = []
        for field in encoding.iterfind("MicrocodeFormat/BitMap/Field"):
            bit_range = field.find("BitLayout/Range")
            if bit_range is None:
                continue
            offset, count = int(_text(bit_range, "BitOffset")), int(_text(bit_range, "BitCount"))
            fields.append((field_name(_text(field, "FieldName")), offset + count - 1, offset))
        identifier = None
        first = encoding.find("EncodingIdentifiers/EncodingIdentifier")
        selector = next((f for f in fields if f[0] == "encoding"), None)
        if first is not None and selector is not None:
            bits = first.text.strip()
            hi, lo = selector[1] % 32, selector[2] % 32
            identifier = int("".join(bits[len(bits) - 1 - b] for b in range(hi, lo - 1, -1)), 2)
        encodings[rename(strip(name))] = (fields, identifier)
    return encodings


def default_encoding(encoding: ET.Element) -> bool:
    """Whether an instruction encoding is the one both tables are made from.

    The XML lists each encoding once per literal condition (``has_lit_0``,
    ``has_lit_1_has_lit_2``,...): the same fields followed by literal dwords.
    Only the ``default`` condition describes the instruction itself; an
    instruction that always takes a literal has its literal form as default.
    """
    return _text(encoding, "EncodingCondition") == "default"


class _Instructions:
    """The opcodes, encodings and operands of the instructions."""

    def __init__(self, isa: ET.Element, rules: TargetRules):
        self.enum: dict[str, dict[int, str]] = defaultdict(dict)
        self.seen: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.types: dict[tuple[str, str], dict] = {}
        self.short: set[str] = set()
        drop = set(rules.drop)
        for instruction in isa.iterfind("Instructions/Instruction"):
            raw = _text(instruction, "InstructionName")
            low = raw.lower()
            name = rules.renames.get(low, raw).upper()
            for encoding in instruction.iterfind("InstructionEncodings/InstructionEncoding"):
                en = _text(encoding, "EncodingName")
                if (low, en) in drop or not default_encoding(encoding):
                    continue
                if en in ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC"):
                    self.short.add(name)
                self._add(name, encoding)
        for fmt, opcodes in FIXES.get(rules.arch, {}).items():
            self.enum[fmt].update(opcodes)
        self.suffix_only: dict[tuple[str, int], str] = {}
        for key, names in self.seen.items():
            shared = next((s for s in SUFFIX_ONLY if all(s in n for n in names)), None)
            if shared is not None:
                self.suffix_only[key] = shared

    def _add(self, name: str, encoding: ET.Element) -> None:
        stripped = flat_map(strip(_text(encoding, "EncodingName")), name)
        fmt = rename(stripped)
        bfmt = base(fmt)
        opcode = int(_text(encoding, "Opcode"))
        if fmt not in self.seen[(bfmt, opcode)]:
            self.seen[(bfmt, opcode)].append(fmt)
        if "ADDTID" in name:
            if stripped == "GLOBAL":
                self.enum["FLAT"][opcode] = name
            elif stripped == "VGLOBAL":
                self.enum["VFLAT"][opcode] = name
        self.enum[bfmt][opcode] = name
        operands = encoding.find("Operands")
        info = {}
        for operand in encoding.iterfind("Operands/Operand"):
            field = _text(operand, "FieldName")
            if field and not ("_LOAD_LDS_" in name and field.lower() == "vdst"):
                info[field.lower()] = (
                    _text(operand, "DataFormatName"),
                    int(_text(operand, "OperandSize") or 0),
                    _text(operand, "OperandType"),
                )
        empty_operands = operands is not None and operands.find("Operand") is None
        if (info or empty_operands) and not (fmt != bfmt and (name, bfmt) in self.types):
            self.types[(name, bfmt)] = info

    def has_suffix_only(self, suffix: str) -> bool:
        return suffix in self.suffix_only.values()

    def vdst_type(self, name: str, fmt: str) -> str:
        return self.types.get((name, fmt), {}).get("vdst", ("", 0, ""))[2]


def _kind(field: tuple, fmt: str, identifier: int | None, rules: TargetRules) -> dict:
    """A field's kind and value in class fmt."""
    name, hi, lo = field
    bits, b = hi - lo + 1, base(fmt)

    def kinded(kind: str, value: int = 0, top: int = hi) -> dict:
        return {"name": name, "hi": top, "lo": lo, "kind": kind, "value": value}

    if name == "encoding" and identifier is not None:
        return kinded("fixed", identifier)
    if name == "op" and fmt not in ("DPP", "SDWA"):
        return kinded("op")
    if name in VGPR_FIELDS and bits == 8:
        return kinded("vgpr")
    table = [
        (name == "sbase" and bits == 6, "sbase", 0),
        (name in ("srsrc", "ssamp") and bits == 5, "srsrc", 0),
        (name in ("sdst", "sdata") and bits == 7, "sgpr", 0),
        (name == "saddr" and bits == 7 and b == "FLAT", "sgpr", 0),
        (name == "saddr" and bits == 7, "sgpr", rules.saddr_default),
        (name == "soffset" and bits == 7, "sgpr", rules.soffset_default),
        (name.startswith("ssrc") and bits == 8, "ssrc", 0),
        (name == "soffset" and bits == 8 and b == "MUBUF", "ssrc", 128),
        (name in ("saddr", "soffset") and bits == 8, "ssrc", 124),
        (name.startswith("src") and bits == 9, "src", 0),
    ]
    for applies, kind, value in table:
        if applies:
            return kinded(kind, value)
    if name == "offset" and b in ("GLOBAL", "SCRATCH"):
        return kinded("signed", 0, top=12)
    if name == "offset" and b == "SMEM":
        return kinded("signed")
    if b == "VOP3P" and name == "opsel_hi":
        return kinded("bits", 3)
    if b == "VOP3P" and name == "opsel_hi2":
        return kinded("bits", 1)
    return kinded("bits")


def _class(name: str, fields: list[dict]) -> dict:
    fields = sorted(fields, key=lambda f: (-f["hi"], f["name"]))
    return {"class": name, "bytes": (max(f["hi"] for f in fields) + 8) // 8, "fields": fields}


def _classes(encodings: dict, instructions: _Instructions, rules: TargetRules) -> dict[str, dict]:
    """Every encoding class with its kinded fields."""
    classes: dict[str, dict] = {}
    kinded: dict[str, list[dict]] = {}
    bases = sorted(n for n in encodings if base(n) == n)
    for name in bases:
        fields, identifier = encodings[name]
        if name in ("FLAT", "VFLAT") and any(f[0] == "seg" for f in fields):
            for cls, seg in (("FLAT", 0), ("GLOBAL", 2), ("SCRATCH", 1)):
                kinded[cls] = [
                    {"name": "seg", "hi": f[1], "lo": f[2], "kind": "fixed", "value": seg}
                    if f[0] == "seg"
                    else _kind(f, cls, identifier, rules)
                    for f in fields
                ]
                classes[cls] = _class(cls, kinded[cls])
        elif name not in ("FLAT_GLOBAL", "FLAT_SCRATCH", "FLAT_GLBL", "DPP", "SDWA"):
            kinded[name] = [_kind(f, name, identifier, rules) for f in fields]
            classes[name] = _class(name, kinded[name])
    for name in sorted(n for n in encodings if base(n) != n):
        if base(name) not in bases or base(name) not in kinded:
            continue
        base_fields = kinded[base(name)]
        base_names = {f["name"] for f in base_fields}
        extra = [f for f in encodings[name][0] if f[0] not in base_names]
        suffix = next((s for s in SUFFIXES if name.endswith(s)), None)
        if extra or (suffix is not None and instructions.has_suffix_only(suffix)):
            kept = [
                f
                for f in base_fields
                if not any(e[2] <= f["hi"] and f["lo"] <= e[1] for e in extra)
            ]
            classes[name] = _class(name, kept + [_kind(f, name, None, rules) for f in extra])
    for name, hi, lo in (("VOP1", 24, 17), ("VOP3", 7, 0)):
        if name not in kinded:
            continue
        if not any(
            instructions.vdst_type(n, name).startswith("OPR_SREG")
            for n in instructions.enum[name].values()
        ):
            continue
        sdst = [
            {"name": "vdst", "hi": hi, "lo": lo, "kind": "ssrc", "value": 0}
            if f["name"] == "vdst"
            else f
            for f in kinded[name]
        ]
        classes[name + "_SDST"] = _class(name + "_SDST", sdst)
        literal = next(
            (f for f in encodings.get(name + "_LIT", ([], None))[0] if f[0] == "literal"), None
        )
        if literal is not None:
            extra = {
                "name": "literal",
                "hi": literal[1],
                "lo": literal[2],
                "kind": "bits",
                "value": 0,
            }
            classes[name + "_SDST_LIT"] = _class(name + "_SDST_LIT", [*sdst, extra])
    return classes


def _ops(encodings: dict, instructions: _Instructions) -> dict[str, dict]:
    """Every mnemonic's class, opcode and operands."""
    ops = {}
    for fmt in sorted(instructions.enum):
        if not ((fmt in encodings and base(fmt) == fmt) or fmt in ("GLOBAL", "SCRATCH")):
            continue
        suffix = {"VOP1": "_E32", "VOP2": "_E32", "VOPC": "_E32", "VOP3": "_E64"}.get(fmt, "")
        for opcode, name in sorted(instructions.enum[fmt].items()):
            if "ADDTID" in name and fmt == "FLAT":
                continue
            msuf = "" if fmt == "VOP3" and name not in instructions.short else suffix
            vdst = instructions.vdst_type(name, fmt)
            if fmt == "VOP1" and vdst.startswith("OPR_SREG"):
                cls = "VOP1_SDST"
            elif fmt == "VOP3" and (vdst.startswith("OPR_SREG") or opcode < 256):
                cls = "VOP3_SDST"
            elif (fmt, opcode) in instructions.suffix_only:
                cls = fmt + instructions.suffix_only[(fmt, opcode)]
            else:
                cls = fmt
            operands = instructions.types.get((name, fmt), {})
            ops[(name + msuf).lower()] = {
                "mnemonic": (name + msuf).lower(),
                "class": cls,
                "opcode": opcode,
                "member": name + msuf,
                "operands": [
                    {"field": field, "format": fmt_name, "bits": bits, "type": kind}
                    for field, (fmt_name, bits, kind) in sorted(operands.items())
                ],
            }
    return ops


def generate(isa: ET.Element, rules: TargetRules) -> dict:
    """``{encodings, ops}`` sorted by class name and by mnemonic."""
    encodings = _encodings(isa)
    instructions = _Instructions(isa, rules)
    classes = _classes(encodings, instructions, rules)
    ops = _ops(encodings, instructions)
    return {
        "encodings": [classes[name] for name in sorted(classes)],
        "ops": [ops[name] for name in sorted(ops)],
    }
