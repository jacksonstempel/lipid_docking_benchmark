#!/usr/bin/env python3
"""
Minimal, deterministic batch runner for pose_benchmark.py.

Loads repository defaults from config.yaml, runs pose_benchmark.py for each
target, and writes a condensed aggregate CSV under analysis/aggregates/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from scripts.lib.aggregation import collect_analysis_csvs, write_condensed_csv
from scripts.lib.config import load_config
from scripts.lib.paths import (
    PathResolver,
    find_prediction_cif,
    find_reference_cif,
    list_candidate_ids,
    normalize_pdbid,
)


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
    ids: Optional[List[str]],
    verbose: bool,
) -> List[Target]:
    if ids:
        todo = [normalize_pdbid(pid) for pid in ids if normalize_pdbid(pid)]
    else:
        todo = list_candidate_ids(resolver.refs_root)

    targets: List[Target] = []
    for pid in todo:
        ref = find_reference_cif(pid, resolver.refs_root)
        if ref is None:
            if verbose:
                print(f"[WARN] Missing ref for {pid}")
            continue
        pred = find_prediction_cif(pid, resolver.preds_root)
        if pred is None:
            if verbose:
                print(f"[WARN] Missing pred for {pid}")
            continue
        targets.append(Target(pid, ref, pred))
    return targets


def _common_pose_args(config_path: Path, resolver: PathResolver) -> List[str]:
    return [
        "--config",
        str(config_path),
        "--refs",
        str(resolver.refs_root),
        "--preds",
        str(resolver.preds_root),
        "--analysis-dir",
        str(resolver.analysis_root),
    ]


def run_pose(
    pose_script: Path,
    target: Target,
    *,
    common_args: List[str],
    full: bool,
    verbose: bool,
) -> int:
    cmd = [
        sys.executable,
        str(pose_script),
        target.pdbid,
        "--ref-file",
        str(target.ref),
        "--pred-file",
        str(target.pred),
    ]
    cmd.extend(common_args)
    if full:
        cmd.append("--full")
    if verbose:
        cmd.append("-v")
    print("[RUN]", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(f"[ERROR] pose_benchmark.py not found: {pose_script}")
        return 127


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal deterministic batch wrapper for pose_benchmark.py"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to project config (default: config.yaml)",
    )
    parser.add_argument(
        "--refs",
        default=None,
        help="Override references root directory",
    )
    parser.add_argument(
        "--preds",
        default=None,
        help="Override predictions root directory",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Override analysis root directory",
    )
    parser.add_argument(
        "--aggregates-dir",
        default=None,
        help="Override aggregate output directory",
    )
    parser.add_argument(
        "--pose",
        default=None,
        help="Path to pose_benchmark.py (default from config)",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        help="Optional text file of PDB IDs (one per line)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Pass --full through to pose_benchmark.py",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    resolver = PathResolver(
        config,
        refs=args.refs,
        preds=args.preds,
        analysis_dir=args.analysis_dir,
        aggregates_dir=args.aggregates_dir,
    )

    pose_script = (
        Path(args.pose).expanduser().resolve()
        if args.pose
        else config.scripts.pose_benchmark
    )
    if not pose_script.is_file():
        print(f"[ERROR] pose_benchmark.py not found at {pose_script}")
        return 2

    ids_list = _read_ids(args.ids)
    if ids_list and args.verbose:
        print(f"[INFO] Loaded {len(ids_list)} IDs from {args.ids}")

    targets = discover_targets(resolver, ids_list, args.verbose)
    if not targets:
        print("[ERROR] No targets found. Check --refs/--preds and --ids.")
        return 2

    common_args = _common_pose_args(config.source, resolver)

    failed: List[str] = []
    for target in targets:
        rc = run_pose(
            pose_script,
            target,
            common_args=common_args,
            full=args.full,
            verbose=args.verbose,
        )
        if rc != 0:
            print(f"[WARN] pose_benchmark failed for {target.pdbid} (rc={rc})")
            failed.append(target.pdbid)

    completed_ids = [t.pdbid for t in targets if t.pdbid not in failed]

    per_target_csvs: List[Path] = []
    missing_csvs: List[tuple[str, Path]] = []
    if completed_ids:
        per_target_csvs, missing_csvs = collect_analysis_csvs(resolver, completed_ids)
        for pid, path in missing_csvs:
            print(f"[WARN] Missing analysis CSV for {pid}: {path}")

    ts = datetime.now()
    fname = f"aggregate_{ts.month}.{ts.day}_{ts.hour}.{ts.minute:02d}.csv"
    aggregate_path = resolver.aggregates_path(fname)

    if per_target_csvs:
        rows_written, errors = write_condensed_csv(per_target_csvs, aggregate_path)
        print(f"[OK] Wrote {rows_written} rows → {aggregate_path}")
        for csv_path, exc in errors:
            print(f"[WARN] Failed reading {csv_path}: {exc}")
    else:
        print("[WARN] No per-target analysis CSVs found; aggregate not written.")

    if failed:
        print("[INFO] Failed targets:", ",".join(failed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
