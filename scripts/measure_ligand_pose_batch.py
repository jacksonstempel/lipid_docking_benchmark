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
import contextlib
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from scripts.lib.ligand_pose_core import AtomPairingError, LigandSelectionError, measure_ligand_pose, measure_ligand_pose_all

LOGGER = logging.getLogger("measure_ligand_pose_batch")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch measure ligand pose RMSD over ref/pred pairs.")
    p.add_argument(
        "--pairs",
        help="Input CSV with columns: pdbid,ref,<pred-column>. Defaults to config paths.pairs or scripts/pairs.csv",
    )
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
    p.add_argument("--quiet", action="store_true", default=True, help="Suppress per-entry logs and show compact progress.")
    p.add_argument("--no-quiet", action="store_false", dest="quiet", help="Disable quiet mode.")
    p.add_argument(
        "--pred-column",
        help="Column name to use as prediction path (default: boltz_pred for boltz, vina_pred for vina; falls back to pred).",
    )
    return p


@contextlib.contextmanager
def _suppress_stderr_fd():
    """Silence C-level stderr (e.g., RDKit) within the block."""
    try:
        fd = sys.stderr.fileno()
    except Exception:
        # Fallback to Python-level redirection only.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(fd)
    os.dup2(devnull, fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def _read_pairs(csv_path: Path, pred_column: str) -> List[Dict[str, str]]:
    required = {"pdbid", "ref", pred_column}
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Input CSV must have columns: {', '.join(sorted(required))}")
        for row in reader:
            rows.append(
                {
                    "pdbid": (row.get("pdbid") or "").strip(),
                    "ref": (row.get("ref") or "").strip(),
                    "pred": (row.get(pred_column) or "").strip(),
                }
            )
    return rows


def _ensure_path(path_str: str, label: str, project_root: Path) -> Path:
    """Resolve path (relative or absolute) and verify it exists."""
    path = Path(path_str).expanduser()
    # If path is relative, resolve from project root
    if not path.is_absolute():
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _default_pairs_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_path.read_text()) or {}
            cfg_paths = data.get("paths", {}) or {}
            if "pairs" in cfg_paths:
                return (project_root / cfg_paths["pairs"]).resolve()
        except Exception:
            pass
    return (project_root / "scripts" / "pairs.csv").resolve()


def _default_out_template(kind: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / "analysis" / f"{kind}_batch_results.csv").resolve()


def _timestamped_out(template: Path) -> Path:
    """Append a timestamp before the extension of the given template path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = template.stem
    suffix = template.suffix
    return template.with_name(f"{stem}_{ts}{suffix}")


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    pairs_path = Path(args.pairs).expanduser().resolve() if args.pairs else _default_pairs_path()
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = _timestamped_out(_default_out_template(args.kind))

    try:
        pred_col = args.pred_column or ("boltz_pred" if args.kind == "boltz" else "vina_pred")
        pairs = _read_pairs(pairs_path, pred_col)
    except Exception as exc:
        LOGGER.error("Failed to read pairs file: %s", exc)
        return 2

    results: List[Dict[str, str | float | int]] = []
    success = 0
    failure = 0

    total = len(pairs)
    for idx, row in enumerate(pairs, start=1):
        pdbid = row["pdbid"]
        ref_str = row["ref"]
        pred_str = row["pred"]
        if not args.quiet:
            LOGGER.info("Evaluating %s", pdbid)
        try:
            ref_path = _ensure_path(ref_str, "Reference", project_root)
            pred_path = _ensure_path(pred_str, "Prediction", project_root)
            if args.quiet:
                with _suppress_stderr_fd():
                    entries = measure_ligand_pose_all(ref_path, pred_path, max_poses=max(1, args.max_poses))
            else:
                entries = measure_ligand_pose_all(ref_path, pred_path, max_poses=max(1, args.max_poses))
            for res in entries:
                results.append(
                    {
                        "pdbid": pdbid,
                        "ref": str(ref_path),
                        "pred": str(pred_path),
                        "pose_index": res.get("pose_index", ""),
                        "ligand_id": res.get("ligand_id", ""),
                        "pairing_method": res.get("pairing_method", ""),
                        "ligand_heavy_atoms": res.get("ligand_heavy_atoms", ""),
                        "ligand_rmsd": res.get("ligand_rmsd", ""),
                        "protein_pairs": res.get("protein_pairs", ""),
                        "protein_rmsd": res.get("protein_rmsd", ""),
                        "status": res.get("status", "ok"),
                        "error": res.get("error", ""),
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
                    "pose_index": "",
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

        if args.quiet:
            print(f"\rPose RMSD ({args.kind}): {idx}/{total}", end="", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdbid",
        "ref",
        "pred",
        "pose_index",
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

    if args.quiet:
        print()  # newline after progress
    LOGGER.info("Completed batch: success=%d, failure=%d. Results at %s", success, failure, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
