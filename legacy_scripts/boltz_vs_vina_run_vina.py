#!/usr/bin/env python3
"""Batch benchmarking runner for AutoDock Vina predictions.

Reads predicted poses from `model_outputs/vina/<PDBID>/latest/<PDBID>_vina_pose.pdbqt`,
aligns them to reference complexes, and writes unified outputs under the chosen
analysis directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_SCRIPT_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.config import load_config
from scripts.lib.paths import (
    PathResolver,
    find_reference_cif,
    find_vina_pose,
    normalize_pdbid,
)
from scripts.lib.pose_pipeline import run_pose_benchmark
from scripts.lib.results_io import (
    append_all_results,
    build_and_write_summary,
    build_random_aggregate_from_details,
    current_timestamp,
    infer_source_label,
    resolve_summary_directory,
)


LOGGER = logging.getLogger("vina_benchmark_runner")


@dataclass
class Target:
    pdbid: str
    ref: Path
    pred: Path


def _read_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.is_file():
        return None
    ids: List[str] = []
    for line in path.read_text().splitlines():
        pid = normalize_pdbid(line)
        if pid:
            ids.append(pid)
    return ids or None


def discover_targets(
    resolver: PathResolver,
    preds_root: Path,
    ids: Optional[List[str]],
    verbose: bool,
) -> List[Target]:
    if ids:
        todo = [normalize_pdbid(pid) for pid in ids if normalize_pdbid(pid)]
    else:
        todo = sorted(p.name for p in preds_root.iterdir() if p.is_dir())

    targets: List[Target] = []
    for pid in todo:
        ref = find_reference_cif(pid, resolver.refs_root)
        if ref is None:
            if verbose:
                LOGGER.warning("[WARN] Missing ref for %s", pid)
            continue
        pred = find_vina_pose(pid, preds_root)
        if pred is None:
            if verbose:
                LOGGER.warning("[WARN] Missing Vina pose for %s", pid)
            continue
        targets.append(Target(pid, ref, pred))
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch runner for pose benchmarking using Vina predictions",
    )
    parser.add_argument("--config", default=None, help="Path to config file (default: config.yaml)")
    parser.add_argument("--refs", default=None, help="Override references root directory")
    parser.add_argument("--preds", default=None, help="Override Vina predictions root directory")
    parser.add_argument("--analysis-dir", default=None, help="Override analysis root directory")
    parser.add_argument("--ids", type=Path, help="Optional text file of PDB IDs (one per line)")
    parser.add_argument("--pose-count", type=int, default=1, help="Number of poses to evaluate per prediction")
    parser.add_argument("--include-h", action="store_true", help="Include hydrogens when pairing ligands")
    parser.add_argument("--include-small", action="store_true", help="Include small ligands below the heavy-atom threshold")
    parser.add_argument("--no-pocket", action="store_true", help="Disable pocket-local alignment")
    parser.add_argument("--pocket-radius", type=float, default=5.0, help="Å radius for pocket Cα alignment")
    parser.add_argument("--full", action="store_true", help="Capture per-pair protein diagnostics")
    parser.add_argument("--source-label", default=None, help="Label for aggregate outputs (default: inferred from prediction path)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    preds_root = Path(args.preds).expanduser().resolve() if args.preds else config.paths.model_outputs / "vina"
    resolver = PathResolver(
        config,
        refs=args.refs,
        preds=preds_root,
        analysis_dir=args.analysis_dir,
    )

    run_timestamp = current_timestamp()
    source_candidates: List[Path | str] = []
    source_candidates.append(preds_root)
    source_label = args.source_label or infer_source_label(source_candidates) or "vina"

    ids_list = _read_ids(args.ids)
    if ids_list and args.verbose:
        LOGGER.info("Loaded %d IDs from %s", len(ids_list), args.ids)

    targets = discover_targets(resolver, preds_root, ids_list, args.verbose)
    if not targets:
        LOGGER.error("No targets found. Check --refs/--preds and --ids.")
        return 2

    all_details: List[dict] = []
    protein_summaries: List[tuple[str, dict]] = []
    failures: List[str] = []

    for target in targets:
        LOGGER.info("Running benchmark for %s", target.pdbid)
        try:
            result = run_pose_benchmark(
                pdbid=target.pdbid,
                resolver=resolver,
                project_root=config.base_dir,
                ref_path=target.ref,
                pred_path=target.pred,
                pose_count=max(1, args.pose_count),
                include_h=args.include_h,
                include_small=args.include_small,
                enable_pocket=not args.no_pocket,
                pocket_radius=args.pocket_radius,
                capture_full=args.full,
            )
        except Exception as exc:  # pragma: no cover - top-level guard
            LOGGER.exception("Benchmark failed for %s: %s", target.pdbid, exc)
            failures.append(target.pdbid)
            continue

        all_details.extend(result.get("details", []))
        protein_summaries.append((target.pdbid, result.get("summary", {})))

    if not protein_summaries:
        LOGGER.error("No successful benchmarks were recorded.")
        return 2

    raw_data_path = resolver.analysis_root / "raw_data" / "all_results.csv"
    append_all_results(all_details, raw_data_path, source_label, run_timestamp)

    summary_dir = resolve_summary_directory(resolver.aggregates_root, source_label)
    summary_path = summary_dir / f"full_run_summary_{run_timestamp}.csv"
    extra_rows = build_random_aggregate_from_details(all_details, source_label, run_timestamp)
    build_and_write_summary(protein_summaries, source_label, summary_path, run_timestamp, extra_rows=extra_rows)

    LOGGER.info("Wrote aggregate results: %s", raw_data_path)
    LOGGER.info("Wrote summary: %s", summary_path)

    if failures:
        LOGGER.warning("Failed targets: %s", ",".join(failures))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
