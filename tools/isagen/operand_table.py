"""The operand table: roles and slots, for the forms."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from tools.isagen.targets import TargetRules

IMPLICIT = {
    "OPR_SDST_EXEC": "exec",
    "OPR_SSRC_SPECIAL_SCC": "scc",
    "OPR_SDST_M0": "m0",
    "OPR_VCC": "vcc",
}
"""The named register each implicit operand type stands for."""

SCALAR_TYPES = ("OPR_SREG", "OPR_SDST", "OPR_SSRC", "OPR_SRC", "OPR_SMEM_OFFSET", "OPR_VCC")
"""Operand-type prefixes that take scalar registers, and so have SGPR widths."""

SHORT = ("ENC_VOP1", "ENC_VOP2", "ENC_VOPC")


class GenerationError(ValueError):
    """Two rows of one key disagree on their roles."""


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    return "" if found is None or found.text is None else found.text.strip()


class _Table:
    """The rows by (mnemonic, count), each kept as first put."""

    def __init__(self, rules: TargetRules):
        self.rules = rules
        self.rows: dict[tuple[str, int], dict] = {}
        self.registers = {**IMPLICIT, **rules.unwritten}
        self.tied = re.compile(rules.tied)

    def register(self, name: str) -> str:
        if self.rules.mask_dwords == 1 and name in ("exec", "vcc"):
            return name + "_lo"
        return name

    def put(self, mnemonic: str, row: dict) -> None:
        key = (mnemonic, len(row["slots"]))
        if key not in self.rows:
            self.rows[key] = {"mnemonic": mnemonic, "count": key[1], **row}
        elif self.rows[key]["roles"] != row["roles"]:
            raise GenerationError(f"{key}: roles {row['roles']} after {self.rows[key]['roles']}")

    def add(self, name: str, encoding: ET.Element, short: bool) -> None:
        """The rows of one encoding of an instruction."""
        rules, en = self.rules, _text(encoding, "EncodingName")
        mnemonic = name
        if "_DPP" in en:
            mnemonic += "_dpp"
        elif "_SDWA" in en:
            mnemonic += "_sdwa"
        elif en in ("ENC_VOP3", "VOP3_SDST_ENC") and short:
            mnemonic += "_e64"
        roles, widths, slots, defs, uses = [], [], [], set(), set()
        for operand in encoding.iterfind("Operands/Operand"):
            kind = _text(operand, "OperandType")
            if _text(operand, "FieldName").lower().startswith("scale_src"):
                kind = "OPR_SRC_VGPR"
            reads, writes = operand.get("Input") == "true", operand.get("Output") == "true"
            if operand.get("IsImplicit") == "true" or kind in rules.unwritten:
                register = self.registers.get(kind)
                if register is not None:
                    if writes:
                        defs.add(self.register(register))
                    if reads:
                        uses.add(self.register(register))
                continue
            roles.append("rw" if reads and writes else "w" if writes else "r" if reads else "-")
            if _text(operand, "DataFormatName") == "FMT_NUM_M64":
                size = 32 * rules.mask_dwords
            else:
                size = int(_text(operand, "OperandSize") or 32)
            widths.append([max(1, size // 32)] if kind.startswith(SCALAR_TYPES) else [])
            slots.append([kind, size])
        if en.startswith(rules.vector) and mnemonic not in rules.no_exec and slots:
            uses.add(self.register("exec"))
        if en.startswith(rules.flat) and "_lds_" not in mnemonic:
            uses.discard("m0")
        elif en.startswith(rules.flat) and "_load_lds_" in mnemonic and roles and roles[0] == "w":
            roles, widths, slots = roles[1:], widths[1:], slots[1:]
            uses.add("m0")
        if roles and (self.tied.match(mnemonic) or mnemonic.endswith("_dpp")):
            roles[0] = "rw"
        if roles and (
            mnemonic == "v_swap_b32"
            or (mnemonic.startswith("v_permlane") and mnemonic.endswith("_swap_b32"))
        ):
            roles[0:2] = ["rw", "rw"]
        for key in [mnemonic, *([rules.alias[mnemonic]] if mnemonic in rules.alias else [])]:
            self._put_all(key, en, roles, widths, slots, sorted(defs), sorted(uses))

    def _put_all(self, m, en, roles, widths, slots, defs, uses) -> None:
        rules = self.rules

        def row(roles, widths, slots, encoding=en, uses=uses, defs=defs):
            return {
                "roles": roles,
                "defs": defs,
                "uses": uses,
                "sgpr_widths": widths,
                "encoding": encoding,
                "slots": slots,
            }

        atomic = "atomic" in m
        if atomic and en.startswith(rules.flat):
            self.put(m, row(["w", *roles[1:]], widths, slots))
            self.put(m, row(roles[1:], widths[1:], slots[1:]))
        elif atomic and en.startswith(rules.rtn_encodings):
            self.put(m, row(["r", *roles[1:]], widths, slots))
            self.put(m + "+rtn", row(["rw", *roles[1:]], widths, slots))
        else:
            self.put(m, row(roles, widths, slots))
        dma = en.startswith("ENC_MUBUF") and m in rules.lds_dma
        if dma and len(slots) == 4:
            lds = row(
                ["r", "r", "r"],
                widths[1:],
                slots[1:],
                encoding=f"{en} lds",
                uses=sorted({*uses, "m0"}),
                defs=[],
            )
            self.put(m, lds)


def generate(isa: ET.Element, rules: TargetRules) -> list:
    """The rows keyed (mnemonic, count), sorted by that key."""
    table = _Table(rules)
    drop = set(rules.drop)
    for instruction in isa.iterfind("Instructions/Instruction"):
        raw = _text(instruction, "InstructionName").lower()
        name = rules.renames.get(raw, raw)
        encodings = [
            e
            for e in instruction.iterfind("InstructionEncodings/InstructionEncoding")
            if (raw, _text(e, "EncodingName")) not in drop
        ]
        names = [_text(e, "EncodingName") for e in encodings]
        short = any(n in SHORT for n in names)
        for encoding, en in zip(encodings, names, strict=True):
            cond = _text(encoding, "EncodingCondition")
            if en.startswith(rules.skip) or cond not in ("default", "has_dpp", "has_sdwa"):
                continue
            table.add(name, encoding, short)
    for mnemonic, count in rules.hand:
        table.rows[(mnemonic, count)] = {
            "mnemonic": mnemonic,
            "count": count,
            "roles": [],
            "defs": [],
            "uses": [],
            "sgpr_widths": [],
            "encoding": "ENC_SOPP",
            "slots": [],
        }
    for (mnemonic, count), row in table.rows.items():
        wide = table.rows.get((mnemonic + "_e64", count))
        if row["encoding"] in SHORT and wide is not None:
            row["sgpr_widths"] = [
                sorted({*a, *b})
                for a, b in zip(row["sgpr_widths"], wide["sgpr_widths"], strict=True)
            ]
    return [table.rows[key] for key in sorted(table.rows)]
