#!/usr/bin/env python3
"""Pose benchmarking CLI for protein–ligand predictions.

The script aligns predicted structures to reference complexes, evaluates ligand
placement with locked RMSD metrics, and records results in the shared aggregate
outputs:

* `analysis/raw_data/all_results.csv` – detailed rows for every protein, pose, and
  ligand comparison appended across runs.
* `analysis/aggregates/<label>/full_run_summary_<timestamp>.csv` – per-protein
  summaries with additional aggregate statistics for the run.

Inputs can originate from any docking or diffusion workflow (Boltz, Vina, MOE,
etc.) so long as each prediction is supplied as a CIF/PDB (optionally ligand-only
with a matching protein template).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

# Allow running as `python scripts/pose_benchmark.py` without installing the package.
if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.config import load_config
from scripts.lib.paths import (
    PathResolver,
    find_prediction_cif,
    find_reference_cif,
    normalize_pdbid,
)
from scripts.lib.pose_pipeline import run_pose_benchmark
from scripts.lib.results_io import (
    append_all_results,
    build_and_write_summary,
    current_timestamp,
    infer_source_label,
    resolve_summary_directory,
    build_random_aggregate_from_details,
)

LOGGER = logging.getLogger("pose_benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align predicted protein–ligand poses to reference structures and report locked RMSD metrics.",
    )
    parser.add_argument("pdbid", help="PDB ID (e.g., 1HMS)")
    parser.add_argument("--config", default=None, help="Path to project config (default: config.yaml)")
    parser.add_argument("--refs", default=None, help="Override references root directory")
    parser.add_argument("--preds", default=None, help="Override predictions root directory")
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Override analysis root directory (default from config)",
    )
    parser.add_argument(
        "--pred-file",
        "--pred",
        dest="pred_file",
        default=None,
        help="Explicit path to predicted CIF/PDB (skips auto-discovery)",
    )
    parser.add_argument(
        "--pose-count",
        type=int,
        default=1,
        help="Number of models/poses to evaluate from the prediction (default: 1)",
    )
    parser.add_argument(
        "--ref-file",
        "--ref",
        dest="ref_file",
        default=None,
        help="Explicit path to reference CIF/PDB (skips auto-discovery)",
    )
    parser.add_argument(
        "--include-h",
        action="store_true",
        help="Include hydrogens when pairing ligands (default: heavy atoms only)",
    )
    parser.add_argument(
        "--include-small",
        action="store_true",
        help="Include ligands with fewer than the default heavy-atom threshold.",
    )
    parser.add_argument(
        "--no-pocket",
        action="store_true",
        help="Disable pocket-local alignment (report global frame only).",
    )
    parser.add_argument(
        "--pocket-radius",
        type=float,
        default=5.0,
        help="Å radius for pocket Cα alignment (default: 5.0 Å).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Capture per-pair protein diagnostics in data.csv (slower).",
    )
    parser.add_argument(
        "--source-label",
        default=None,
        help="Label to record in aggregate outputs (default: inferred from prediction path)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def resolve_prediction_path(pdbid: str, resolver: PathResolver, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    path = find_prediction_cif(pdbid, resolver.preds_root)
    if path is None:
        raise FileNotFoundError(f"Unable to locate prediction CIF for {pdbid} under {resolver.preds_root}")
    return path


def resolve_reference_path(pdbid: str, resolver: PathResolver, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    path = find_reference_cif(pdbid, resolver.refs_root)
    if path is None:
        raise FileNotFoundError(f"Unable to locate reference CIF for {pdbid} under {resolver.refs_root}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    resolver = PathResolver(
        config,
        refs=args.refs,
        preds=args.preds,
        analysis_dir=args.analysis_dir,
    )

    pdbid = normalize_pdbid(args.pdbid)

    try:
        pred_path = resolve_prediction_path(pdbid, resolver, args.pred_file)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    if not pred_path.exists():
        LOGGER.error("Prediction file does not exist: %s", pred_path)
        return 2

    try:
        ref_path = resolve_reference_path(pdbid, resolver, args.ref_file)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    if not ref_path.exists():
        LOGGER.error("Reference file does not exist: %s", ref_path)
        return 2

    try:
        result = run_pose_benchmark(
            pdbid=pdbid,
            resolver=resolver,
            project_root=config.base_dir,
            ref_path=ref_path,
            pred_path=pred_path,
            pose_count=max(1, args.pose_count),
            include_h=args.include_h,
            include_small=args.include_small,
            enable_pocket=not args.no_pocket,
            pocket_radius=args.pocket_radius,
            capture_full=args.full,
        )
    except Exception as exc:
        LOGGER.exception("Benchmark failed for %s: %s", pdbid, exc)
        return 2

    run_timestamp = current_timestamp()
    source_label = args.source_label or infer_source_label([pred_path, resolver.preds_root])

    raw_data_path = resolver.analysis_root / "raw_data" / "all_results.csv"
    append_all_results(result.get("details", []), raw_data_path, source_label, run_timestamp)

    details = result.get("details", [])
    summaries = [(pdbid, result.get("summary", {}))]
    summary_dir = resolve_summary_directory(resolver.aggregates_root, source_label)
    summary_path = summary_dir / f"full_run_summary_{run_timestamp}.csv"
    extra_rows = build_random_aggregate_from_details(details, source_label, run_timestamp)
    build_and_write_summary(summaries, source_label, summary_path, run_timestamp, extra_rows=extra_rows)
    LOGGER.info("Wrote summary to %s", summary_path)

    best_pose = result.get("summary", {}).get("best_pose")
    if isinstance(best_pose, dict):
        LOGGER.info(
            "Best pose: locked_global=%.3f Å (pairs=%s) for %s ↔ %s",
            float(best_pose.get("rmsd_locked_global", float("nan"))),
            best_pose.get("n", "?"),
            best_pose.get("pred"),
            best_pose.get("ref"),
        )
    else:
        LOGGER.info("Benchmark completed for %s", pdbid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
