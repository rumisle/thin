"""The pinned ISA XML bundle."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

FILE = "AMD_GPU_MR_ISA_XML_2025_09_05.zip"
URL = f"https://gpuopen.com/download/{FILE}"
SHA256 = "6c656832cffa8198fe8db72007697d592fdeadc14194b9b4fbdcc8ec7f5c5223"


class BundleError(ValueError):
    """The bundle is not the pinned one, or lacks a target's member."""


def check(path: Path) -> None:
    """Refuses any bundle but the pinned one: newer ones changed WMMA sizes."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise BundleError(f"{path}: sha256 {digest}, not the pinned {FILE} ({SHA256}); get {URL}")


def isa_root(path: Path, member: str) -> ET.Element:
    """The ``ISA`` element of a target's XML member, after checking the bundle."""
    check(path)
    with zipfile.ZipFile(path) as bundle:
        names = [n for n in bundle.namelist() if n.endswith(member)]
        if len(names) != 1:
            raise BundleError(f"{path}: no single member {member} (found {names})")
        root = ET.fromstring(bundle.read(names[0]))
    return root if root.tag == "ISA" else root.find(".//ISA")
