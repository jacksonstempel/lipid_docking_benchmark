#!/usr/bin/env python3
"""
Create a new pairs CSV using an alternate Boltz prediction directory.

The benchmark reads a "pairs CSV" (structures/benchmark_entries.csv by default)
with columns: pdbid, ref, boltz_pred, vina_pred.

This script rewrites only the `boltz_pred` column to point at a new directory
(e.g., structures/boltz_isaac/) while keeping `ref` and `vina_pred` identical.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


REQUIRED_COLUMNS = ("pdbid", "ref", "boltz_pred", "vina_pred")


def _find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise RuntimeError(f"Could not find config.yaml in {here} or any parent directory.")


def _read_pairs(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or any(c not in reader.fieldnames for c in REQUIRED_COLUMNS):
            raise ValueError(f"Pairs CSV must contain columns: {', '.join(REQUIRED_COLUMNS)}")
        rows = []
        for row in reader:
            pdbid = (row.get("pdbid") or "").strip().upper()
            if not pdbid:
                continue
            rows.append({k: (row.get(k) or "").strip() for k in REQUIRED_COLUMNS})
        if not rows:
            raise ValueError(f"No rows found in pairs CSV: {path}")
        return rows


def _write_pairs(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-pairs",
        type=Path,
        default=Path("structures/benchmark_entries.csv"),
        help="Input pairs CSV (default: structures/benchmark_entries.csv).",
    )
    parser.add_argument(
        "--out-pairs",
        type=Path,
        default=Path("structures/benchmark_entries_boltz_isaac.csv"),
        help="Output pairs CSV (default: structures/benchmark_entries_boltz_isaac.csv).",
    )
    parser.add_argument(
        "--boltz-dir",
        type=Path,
        default=Path("structures/boltz_isaac"),
        help="Directory containing Boltz CIFs named <PDBID>_model_0.cif (default: structures/boltz_isaac).",
    )
    parser.add_argument(
        "--suffix",
        default="_model_0.cif",
        help="Filename suffix appended to PDBID inside --boltz-dir (default: _model_0.cif).",
    )
    args = parser.parse_args(argv)

    project_root = _find_project_root()
    in_pairs = (project_root / args.in_pairs).resolve() if not args.in_pairs.is_absolute() else args.in_pairs.resolve()
    out_pairs = (project_root / args.out_pairs).resolve() if not args.out_pairs.is_absolute() else args.out_pairs.resolve()
    boltz_dir = (project_root / args.boltz_dir).resolve() if not args.boltz_dir.is_absolute() else args.boltz_dir.resolve()

    rows = _read_pairs(in_pairs)

    boltz_dir_rel = boltz_dir.relative_to(project_root).as_posix().rstrip("/")
    missing: List[str] = []
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        pdbid = row["pdbid"].strip().upper()
        boltz_rel = f"{boltz_dir_rel}/{pdbid}{args.suffix}"
        out_row = dict(row)
        out_row["pdbid"] = pdbid
        out_row["boltz_pred"] = boltz_rel
        out_rows.append(out_row)

        if not (project_root / boltz_rel).exists():
            missing.append(boltz_rel)

    if missing:
        raise SystemExit(
            "Missing Boltz predictions (first 10 shown):\n"
            + "\n".join(missing[:10])
        )

    _write_pairs(out_pairs, out_rows)
    print(f"Wrote: {out_pairs.relative_to(project_root)} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
