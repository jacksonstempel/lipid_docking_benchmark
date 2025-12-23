"""
File I/O helpers for the benchmark.

Plain-language overview

- The benchmark inputs are defined by a “pairs CSV” (a simple table of file paths).
- This module loads that CSV, checks the required columns exist, and turns each row into a
  `PairEntry` object with absolute file paths.
- It also writes output CSV files in a consistent way (including creating folders).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class PairEntry:
    pdbid: str
    ref_path: Path
    boltz_path: Path
    vina_path: Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Return the repository root directory.

    How it works:
    - Start from `start` (or the current working directory if omitted).
    - Walk up parent directories until we find `config.yaml`.

    Why `config.yaml`?
    - It is a simple, reliable “marker file” that only exists at the repo root in this project.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise RuntimeError(f"Could not find config.yaml in {here} or any parent directory.")


def default_pairs_path(project_root: Path) -> Path:
    """
    Return the default pairs CSV path for this repository.

    Where it comes from:
    - If `config.yaml` exists and defines `paths.pairs`, we use that.
    - Otherwise we fall back to `structures/benchmark_entries.csv`.

    This is intentionally the only repository “layout” setting: the pairs CSV itself is the
    source of truth for where the input files live.
    """
    cfg_path = project_root / "config.yaml"
    if cfg_path.exists():
        import yaml  # type: ignore

        data = yaml.safe_load(cfg_path.read_text()) or {}
        paths = data.get("paths", {}) or {}
        pairs = paths.get("pairs")
        if pairs:
            return (project_root / str(pairs)).resolve()
    return (project_root / "structures" / "benchmark_entries.csv").resolve()


def _resolve_existing(project_root: Path, path_str: str, *, label: str, pdbid: str) -> Path:
    """
    Resolve a path from the CSV into an absolute path and confirm it exists.

    - Relative paths are interpreted relative to the repository root.
    - Missing files raise a clear `FileNotFoundError` that includes the PDB ID and column label.
    """
    path = Path(path_str).expanduser()
    path = (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{pdbid}: {label} does not exist: {path}")
    return path


def read_pairs_csv(project_root: Path, pairs_path: Path) -> List[PairEntry]:
    """
    Read a pairs CSV and return validated, absolute file paths.

    Expected columns (header row):
    - `pdbid`: identifier for the target (used for naming outputs)
    - `ref`: experimental reference structure path (CIF/PDB)
    - `boltz_pred`: Boltz prediction path (typically CIF)
    - `vina_pred`: Vina docking results path (typically PDBQT)

    Each row becomes a `PairEntry(pdbid, ref_path, boltz_path, vina_path)`.
    """
    required = {"pdbid", "ref", "boltz_pred", "vina_pred"}
    entries: List[PairEntry] = []
    with pairs_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Pairs CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            pdbid = (row.get("pdbid") or "").strip().upper()
            if not pdbid:
                continue
            ref = _resolve_existing(project_root, (row.get("ref") or "").strip(), label="ref", pdbid=pdbid)
            boltz = _resolve_existing(
                project_root, (row.get("boltz_pred") or "").strip(), label="boltz_pred", pdbid=pdbid
            )
            vina = _resolve_existing(project_root, (row.get("vina_pred") or "").strip(), label="vina_pred", pdbid=pdbid)
            entries.append(PairEntry(pdbid=pdbid, ref_path=ref, boltz_path=boltz, vina_path=vina))
    if not entries:
        raise ValueError(f"No entries found in pairs CSV: {pairs_path}")
    return entries


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """
    Write a CSV file (creating parent folders if needed).

    `rows` is a list of dictionaries. `fieldnames` defines the column order.
    Any missing key in a row is written as an empty string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
