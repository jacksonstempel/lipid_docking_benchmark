#!/usr/bin/env python3
"""Batch runner for the simple ligand pose evaluator.

Reads a CSV of ref/pred pairs and writes a tidy results CSV. Each row is
evaluated independently; failures are captured per-row rather than aborting
the entire run.

Defaults: with no arguments, it will try to use the Boltz pair list
(`paths.boltz_pairs` from `config.yaml` if available, else `scripts/boltz_pairs.csv`)
and write results to a timestamped file in `analysis/`, e.g. `boltz_batch_results_YYYYMMDD_HHMMSS.csv`.
Use `--kind vina` to pick the Vina defaults. You can always override with `--pairs`/`--out`.
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path as _Path

    _PROJECT_ROOT = _Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.ligand_pose_core import AtomPairingError, LigandSelectionError, measure_ligand_pose

LOGGER = logging.getLogger("measure_ligand_pose_batch")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch measure ligand pose RMSD over ref/pred pairs.")
    p.add_argument("--pairs", help="Input CSV with columns: pdbid,ref,pred (default from config or scripts/<kind>_pairs.csv)")
    p.add_argument(
        "--out",
        help="Output CSV path for results (default is analysis/<kind>_batch_results_YYYYMMDD_HHMMSS.csv)",
    )
    p.add_argument("--kind", choices=["boltz", "vina"], default="boltz", help="Default pair list/output selection.")
    p.add_argument(
        "--max-poses",
        type=int,
        default=1,
        help="Evaluate up to this many poses/models in the prediction and report the best RMSD (default: 1).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def _read_pairs(csv_path: Path) -> List[Dict[str, str]]:
    required = {"pdbid", "ref", "pred"}
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV must have columns: {', '.join(sorted(required))}")
        for row in reader:
            rows.append({k: (row.get(k) or "").strip() for k in required})
    return rows


def _ensure_path(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _default_paths(kind: str) -> tuple[Path, Path]:
    """Return default (pairs_path, out_template_path) for the given kind."""
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "config.yaml"
    pairs = None
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_path.read_text()) or {}
            cfg_paths = data.get("paths", {}) or {}
            key = f"{kind}_pairs"
            if key in cfg_paths:
                pairs = (project_root / cfg_paths[key]).resolve()
        except Exception:
            pairs = None
    if pairs is None:
        pairs = (project_root / "scripts" / f"{kind}_pairs.csv").resolve()
    out_template = (project_root / "analysis" / f"{kind}_batch_results.csv").resolve()
    return pairs, out_template


def _timestamped_out(template: Path) -> Path:
    """Append a timestamp before the extension of the given template path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = template.stem
    suffix = template.suffix
    return template.with_name(f"{stem}_{ts}{suffix}")


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.pairs:
        pairs_path = Path(args.pairs).expanduser().resolve()
    else:
        pairs_path, _ = _default_paths(args.kind)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        _, out_template = _default_paths(args.kind)
        out_path = _timestamped_out(out_template)

    try:
        pairs = _read_pairs(pairs_path)
    except Exception as exc:
        LOGGER.error("Failed to read pairs file: %s", exc)
        return 2

    results: List[Dict[str, str | float | int]] = []
    success = 0
    failure = 0

    for row in pairs:
        pdbid = row["pdbid"]
        ref_str = row["ref"]
        pred_str = row["pred"]
        LOGGER.info("Evaluating %s", pdbid)
        try:
            ref_path = _ensure_path(ref_str, "Reference")
            pred_path = _ensure_path(pred_str, "Prediction")
            res = measure_ligand_pose(ref_path, pred_path, max_poses=max(1, args.max_poses))
            results.append(
                {
                    "pdbid": pdbid,
                    "ref": str(ref_path),
                    "pred": str(pred_path),
                    "ligand_id": res.get("ligand_id", ""),
                    "pairing_method": res.get("pairing_method", ""),
                    "ligand_heavy_atoms": res.get("ligand_heavy_atoms", ""),
                    "ligand_rmsd": res.get("ligand_rmsd", ""),
                    "protein_pairs": res.get("protein_pairs", ""),
                    "protein_rmsd": res.get("protein_rmsd", ""),
                    "status": "ok",
                    "error": "",
                }
            )
            success += 1
        except (LigandSelectionError, AtomPairingError, FileNotFoundError, RuntimeError) as exc:
            LOGGER.warning("Failed for %s: %s", pdbid, exc)
            results.append(
                {
                    "pdbid": pdbid,
                    "ref": ref_str,
                    "pred": pred_str,
                    "ligand_id": "",
                    "pairing_method": "",
                    "ligand_heavy_atoms": "",
                    "ligand_rmsd": "",
                    "protein_pairs": "",
                    "protein_rmsd": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            failure += 1
        except Exception as exc:  # pragma: no cover - safeguard
            LOGGER.exception("Unexpected failure for %s: %s", pdbid, exc)
            results.append(
                {
                    "pdbid": pdbid,
                    "ref": ref_str,
                    "pred": pred_str,
                    "ligand_id": "",
                    "pairing_method": "",
                    "ligand_heavy_atoms": "",
                    "ligand_rmsd": "",
                    "protein_pairs": "",
                    "protein_rmsd": "",
                    "status": "error",
                    "error": f"Unexpected: {exc}",
                }
            )
            failure += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdbid",
        "ref",
        "pred",
        "ligand_id",
        "pairing_method",
        "ligand_heavy_atoms",
        "ligand_rmsd",
        "protein_pairs",
        "protein_rmsd",
        "status",
        "error",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    LOGGER.info("Completed batch: success=%d, failure=%d. Results at %s", success, failure, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
