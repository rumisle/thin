"""Records: one self-contained JSON line per arm.

Each carries the versions it was measured under; reading groups by them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

__all__ = ["SCHEMA", "current", "device", "lines", "read", "save", "validators", "version"]

#: The schema string of a record line.
# Record-layout identifier, independent of the package release version.
SCHEMA = "metron/5"
_LEVEL = "rounds of one race"


def _verdict(v) -> dict | None:
    return None if v is None else {"ratio": v.ratio, "ci": list(v.ci), "is": v.verdict}


def lines(r) -> list[dict]:
    """One dict per arm, in arm order, with exactly keys."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdicts = r.verdicts
    out = []
    for name, a in r.arms.items():
        out.append(
            {
                "schema": SCHEMA,
                "time": stamp,
                "point": r.point,
                "arm": name,
                "fingerprint": a.fingerprint,
                "reference": r.reference,
                "state": r.state,
                "us": a.us,
                "ci": None if a.ci is None else list(a.ci),
                "times": list(a.times),
                "ranks_us": None if a.ranks_us is None else list(a.ranks_us),
                "rounds": r.rounds,
                "level": _LEVEL,
                "checked": a.checked,
                "wrong": a.wrong,
                "verdict": _verdict(verdicts.get(name)),
                "status": r.status,
                "conditions": r.conditions,
                "flags": list(r.flags),
                "device": r.device,
                "validators": r.validators,
                "phases_s": r.phases_s,
            }
        )
    return out


def save(r, path) -> None:
    """Appends each line as sorted-key JSON, one per line."""
    with open(path, "a") as f:
        for line in lines(r):
            f.write(json.dumps(line, sort_keys=True) + "\n")


def _text(path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def device() -> dict:  # pragma: no cover - hardware
    """The current device.

    Name, arch, compute units, its PCI address, driver, vbios and firmware
    versions, and whether the runner holds it.
    """
    import torch

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    pci = getattr(p, "pci_bus_id", None)
    domain = getattr(p, "pci_domain_id", 0)
    slot = getattr(p, "pci_device_id", 0)
    address = None if pci is None else f"{domain:04x}:{pci:02x}:{slot:02x}.0"
    sysfs = Path(f"/sys/bus/pci/devices/{address}") if address else None
    firmware = None
    if sysfs is not None and (sysfs / "fw_version").is_dir():
        firmware = {f.name: _text(f) for f in sorted((sysfs / "fw_version").iterdir())}
    return {
        "name": p.name,
        "arch": p.gcnArchName.split(":", 1)[0],
        "cus": p.multi_processor_count,
        "pci": address,
        "amdgpu": _text("/sys/module/amdgpu/version"),
        "vbios": None if sysfs is None else _text(sysfs / "vbios_version"),
        "firmware": firmware,
        "exclusive": os.environ.get("NM_GPU_EXCLUSIVE"),
    }


def _torch_versions() -> tuple:
    try:
        import torch
    except ImportError:
        return None, None
    return torch.__version__, getattr(torch.version, "hip", None)


def validators(dev: dict, launches: str | None = None) -> dict:
    """The versions a number is comparable under.

    launches: "graph" or "plain", the race's path; by default this process's
    graph_replays_ok().
    """
    torch, hip = _torch_versions()
    if launches is None:
        from metron._memory import graph_replays_ok

        launches = "graph" if graph_replays_ok() else "plain"
    return {
        "metron": version(),
        "torch": torch,
        "hip": hip,
        "amdgpu": dev.get("amdgpu"),
        "vbios": dev.get("vbios"),
        "arch": dev.get("arch"),
        "launches": launches,
    }


def current() -> dict:  # pragma: no cover - hardware
    """validators(device()) of this process."""
    return validators(device())


_VERSION: list = []


def _measuring(root: Path) -> list[Path]:
    skip = {"thin.py", "maker.py"}
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return sorted(
        (p for p in files if p.name not in skip and (p.suffix in (".py", ".co", ".json"))),
        key=lambda p: p.relative_to(root).parts,
    )


def version() -> str:
    """16 hex digits of sha256 over the harness's measuring files.

    Its modules and kept kernels, not the adapter nor the maker.
    """
    if not _VERSION:
        root = Path(__file__).parent
        h = hashlib.sha256()
        for p in _measuring(root):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
        _VERSION.append(h.hexdigest()[:16])
    return _VERSION[0]


def read(path) -> list:
    """[(validators, [record,...])] of a records file.

    This schema's lines grouped by their validators (sorted-key JSON), the
    groups by first appearance. A line that is blank, no JSON object, or a
    record without validators is skipped.
    """
    groups: dict = {}
    for line in Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA or "validators" not in rec:
            continue
        key = json.dumps(rec["validators"], sort_keys=True)
        groups.setdefault(key, (rec["validators"], []))[1].append(rec)
    return list(groups.values())
