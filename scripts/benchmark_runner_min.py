#!/usr/bin/env python3
"""Batch pose benchmarking for Boltz and Vina predictions.

The runner loads repository defaults from ``config.yaml`` and, by default,
evaluates every protein that has predictions from both Boltz and Vina plus a
reference structure. Results are written to a single timestamped CSV under the
analysis directory and contain paired rows (Boltz then Vina) for each PDB ID.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
import tempfile
import gemmi
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.config import load_config
from scripts.lib.paths import (
    PathResolver,
    find_prediction_cif,
    find_reference_cif,
    find_vina_pose,
    normalize_pdbid,
)
from scripts.lib.pose_pipeline import run_pose_benchmark
from scripts.lib.results_io import current_timestamp
from scripts.lib.structures import is_protein_res, load_structure


LOGGER = logging.getLogger("benchmark_runner")

OUTPUT_COLUMNS = [
    "pdb_id",
    "method",
    "protein_rmsd",
    "ligand",
    "rmsd_global",
    "rmsd_pocket",
    "n_residues",
    "policy",
]


@dataclass
class Target:
    """Container for a benchmark-ready protein."""

    pdbid: str
    ref: Path
    boltz_pred: Path
    vina_pred: Path
    residue_count: int


def _read_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.is_file():
        return None
    ids: List[str] = []
    for line in path.read_text().splitlines():
        pid = normalize_pdbid(line)
        if pid:
            ids.append(pid)
    return ids or None


def _ids_from_boltz(root: Path) -> set[str]:
    ids: set[str] = set()
    if not root.exists():
        return ids
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_output"):
            ids.add(normalize_pdbid(name.split("_", 1)[0]))
    return ids


def _ids_from_vina(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {normalize_pdbid(entry.name) for entry in root.iterdir() if entry.is_dir()}


def _count_protein_residues(ref_path: Path) -> int:
    structure = load_structure(ref_path)
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_protein_res(residue):
                    count += 1
    return count


def _discover_targets(
    *,
    refs_root: Path,
    boltz_root: Path,
    vina_root: Path,
    ids: Optional[Sequence[str]],
    verbose: bool,
) -> List[Target]:
    if ids:
        candidates = [normalize_pdbid(pid) for pid in ids if normalize_pdbid(pid)]
    else:
        candidates = sorted(_ids_from_boltz(boltz_root) & _ids_from_vina(vina_root))

    targets: List[Target] = []
    for pid in candidates:
        ref = find_reference_cif(pid, refs_root)
        if ref is None:
            if verbose:
                LOGGER.warning("[WARN] Missing reference for %s", pid)
            continue

        boltz_pred = find_prediction_cif(pid, boltz_root)
        if boltz_pred is None:
            if verbose:
                LOGGER.warning("[WARN] Missing Boltz prediction for %s", pid)
            continue

        vina_pred = find_vina_pose(pid, vina_root)
        if vina_pred is None:
            if verbose:
                LOGGER.warning("[WARN] Missing Vina pose for %s", pid)
            continue

        residue_count = _count_protein_residues(ref)
        targets.append(Target(pid, ref.resolve(), boltz_pred.resolve(), vina_pred.resolve(), residue_count))

    return targets


def _build_row(
    *,
    pid: str,
    method: str,
    residue_count: int,
    summary: Dict[str, object],
) -> Dict[str, object]:
    protein_rmsd = summary.get("protein_rmsd_ca_pruned")
    best_pose = summary.get("best_pose") or {}

    def _best_value(key: str) -> float | None:
        value = best_pose.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    ligand_data = best_pose.get("ref") or best_pose.get("pred") or {}
    ligand_label = ligand_data.get("name") or ""
    policy = best_pose.get("policy") or ""

    rmsd_global = _best_value("rmsd_locked_global")
    rmsd_pocket = _best_value("rmsd_locked_pocket")

    row = {
        "pdb_id": pid,
        "method": method,
        "protein_rmsd": 0.0 if method == "vina" else (float(protein_rmsd) if protein_rmsd is not None else ""),
        "ligand": ligand_label,
        "rmsd_global": rmsd_global if rmsd_global is not None else "",
        "rmsd_pocket": rmsd_pocket if rmsd_pocket is not None else "",
        "n_residues": residue_count,
        "policy": policy,
    }
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch runner for pose benchmarking using Boltz and Vina predictions",
    )
    parser.add_argument("--config", default=None, help="Path to config file (default: config.yaml)")
    parser.add_argument("--refs", default=None, help="Override references root directory")
    parser.add_argument("--boltz-preds", default=None, help="Override Boltz predictions root directory")
    parser.add_argument("--vina-preds", default=None, help="Override Vina predictions root directory")
    parser.add_argument("--analysis-dir", default=None, help="Override analysis output directory")
    parser.add_argument("--vina-topk", type=int, default=1, help="Number of Vina poses to evaluate per target across candidate files")
    parser.add_argument("--vina-all-runs", action="store_true", help="Consider poses from all vina_run_* directories (not only latest)")
    parser.add_argument("--ids", type=Path, help="Optional text file of PDB IDs (one per line)")
    parser.add_argument("--pose-count", type=int, default=1, help="Number of poses to evaluate per prediction")
    parser.add_argument("--include-h", action="store_true", help="Include hydrogens when pairing ligands")
    parser.add_argument("--include-small", action="store_true", help="Include ligands below the heavy-atom threshold")
    parser.add_argument("--no-pocket", action="store_true", help="Disable pocket-local alignment")
    parser.add_argument("--pocket-radius", type=float, default=5.0, help="Å radius for pocket Cα alignment")
    parser.add_argument("--full", action="store_true", help="Capture per-pair protein diagnostics (slower)")
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
    refs_root = Path(args.refs).expanduser().resolve() if args.refs else config.paths.refs
    boltz_root = Path(args.boltz_preds).expanduser().resolve() if args.boltz_preds else config.paths.boltz_preds
    vina_root = Path(args.vina_preds).expanduser().resolve() if args.vina_preds else config.paths.vina_preds
    analysis_root = Path(args.analysis_dir).expanduser().resolve() if args.analysis_dir else config.paths.analysis_root
    analysis_root.mkdir(parents=True, exist_ok=True)

    ids_list = _read_ids(args.ids)
    if ids_list and args.verbose:
        LOGGER.info("Loaded %d IDs from %s", len(ids_list), args.ids)

    targets = _discover_targets(
        refs_root=refs_root,
        boltz_root=boltz_root,
        vina_root=vina_root,
        ids=ids_list,
        verbose=args.verbose,
    )
    if not targets:
        LOGGER.error("No targets found with reference + Boltz + Vina predictions.")
        return 2

    boltz_resolver = PathResolver(
        config,
        refs=refs_root,
        preds=boltz_root,
        analysis_dir=analysis_root,
    )
    vina_resolver = PathResolver(
        config,
        refs=refs_root,
        preds=vina_root,
        analysis_dir=analysis_root,
    )

    rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    for target in targets:
        LOGGER.info("Benchmarking %s", target.pdbid)
        for method, pred_path, resolver in (
            ("boltz", target.boltz_pred, boltz_resolver),
            ("vina", target.vina_pred, vina_resolver),
        ):
            try:
                # For Vina, optionally evaluate top-K poses from multiple candidate files (oracle selection)
                if method == "vina" and (args.vina_topk > 1 or args.vina_all_runs):
                    vina_dir = vina_root / target.pdbid
                    candidates: List[Path] = []
                    latest = vina_dir / "latest" / f"{target.pdbid}_vina_pose.pdbqt"
                    if latest.is_file():
                        candidates.append(latest)
                    if args.vina_all_runs:
                        for p in sorted(vina_dir.glob("vina_run_*/*_vina_pose.pdbqt")):
                            if p.is_file():
                                candidates.append(p)
                    # Deduplicate while preserving order
                    seen = set()
                    uniq: List[Path] = []
                    for c in candidates:
                        key = str(c.resolve())
                        if key not in seen:
                            uniq.append(c)
                            seen.add(key)
                    if not uniq:
                        uniq = [pred_path]
                    # Take top-K by recency
                    uniq.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    uniq = uniq[: max(1, args.vina_topk)]
                    # Evaluate each candidate file with top-K poses from that file and keep the best (oracle)
                    best_result = None
                    best_rmsd = float("inf")
                    for cand in uniq:
                        cand_result = run_pose_benchmark(
                            pdbid=target.pdbid,
                            resolver=resolver,
                            project_root=config.base_dir,
                            ref_path=target.ref,
                            pred_path=cand,
                            pose_count=max(1, args.vina_topk),
                            include_h=args.include_h,
                            include_small=args.include_small,
                            enable_pocket=not args.no_pocket,
                            pocket_radius=args.pocket_radius,
                            capture_full=args.full,
                        )
                        bp = (cand_result.get("summary", {}) or {}).get("best_pose", {})
                        val = bp.get("rmsd_locked_global")
                        try:
                            valf = float(val)
                        except (TypeError, ValueError):
                            valf = float("inf")
                        if valf < best_rmsd:
                            best_rmsd = valf
                            best_result = cand_result
                    result = best_result if best_result is not None else cand_result
                else:
                    result = run_pose_benchmark(
                        pdbid=target.pdbid,
                        resolver=resolver,
                        project_root=config.base_dir,
                        ref_path=target.ref,
                        pred_path=pred_path,
                        pose_count=max(1, args.pose_count),
                        include_h=args.include_h,
                        include_small=args.include_small,
                        enable_pocket=not args.no_pocket,
                        pocket_radius=args.pocket_radius,
                        capture_full=args.full,
                    )
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.exception("Benchmark failed for %s (%s): %s", target.pdbid, method, exc)
                failures.append((target.pdbid, method))
                continue

            summary = result.get("summary", {})
            rows.append(
                _build_row(
                    pid=target.pdbid,
                    method=method,
                    residue_count=target.residue_count,
                    summary=summary,
                )
            )

    if not rows:
        LOGGER.error("All benchmarks failed.")
        return 2

    timestamp = current_timestamp()
    output_path = analysis_root / f"benchmark_{timestamp}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    LOGGER.info("Wrote benchmark results: %s", output_path)

    if failures:
        failed = ", ".join(f"{pid}:{method}" for pid, method in failures)
        LOGGER.warning("Failures encountered for: %s", failed)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
