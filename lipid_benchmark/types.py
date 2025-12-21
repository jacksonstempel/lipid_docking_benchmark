from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairEntry:
    pdbid: str
    ref_path: Path
    boltz_path: Path
    vina_path: Path

