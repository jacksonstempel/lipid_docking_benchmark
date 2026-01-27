#!/usr/bin/env python3
"""
Analyze GNINA vs Vina on this benchmark with an emphasis on three hypotheses:

H1 (Sampling): GNINA improves sampling (best-of-20 oracle performance).
H2 (Ranking): CNN rescoring improves ranking (top-1 vs best-of-20 gap).
H3 (Headgroup/contact fidelity): GNINA improves headgroup placement and contact overlap.

Inputs expected:
- Baseline Vina results: output/benchmark/benchmark_allposes.csv (from the main benchmark run).
- GNINA outputs: flat/<PDBID>.pdbqt directories (multi-pose PDBQT per target).

This script will:
- Compute GNINA per-pose ligand/headgroup RMSDs using the repo evaluation (RDKit mapping),
  recording any per-pose failures.
- Run the benchmark pipeline on GNINA top-1 poses (vina-max-poses=1) to compute contact metrics
  (head_env / typed overlap) in the same way Vina was evaluated.
- Write per-target and summary CSVs under --out-dir.

Example (subset5):
  python scripts/analyze_gnina_experiment.py \
      --pairs structures/benchmark_entries.csv \
      --baseline-allposes output/benchmark/benchmark_allposes.csv \
      --gnina-cnn-flat output/gnina/runs/gnina_subset5_cnn_rescore/flat \
      --gnina-nocnn-flat output/gnina/runs/gnina_subset5_no_cnn/flat \
      --out-dir output/gnina/analysis/gnina_subset5_analysis

Example (CNN-only; e.g., if the no-CNN run is still in progress):
  python scripts/analyze_gnina_experiment.py \
      --pairs structures/benchmark_entries.csv \
      --baseline-allposes output/benchmark/benchmark_allposes.csv \
      --gnina-cnn-flat output/gnina/runs/gnina_cnn_complete_only/flat \
      --out-dir output/gnina/analysis/gnina_cnn_complete_analysis
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from lipid_benchmark.rmsd import measure_ligand_pose_all


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pct(values: np.ndarray, pred) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(pred(values)) * 100.0)


def _summ(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "pct_lt2": float("nan"),
            "pct_lt5": float("nan"),
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
        "pct_lt2": _pct(values, lambda v: v < 2.0),
        "pct_lt5": _pct(values, lambda v: v < 5.0),
    }


def _pdbids_from_pairs(pairs_csv: Path) -> List[str]:
    rows = _read_csv_dicts(pairs_csv)
    return [r["pdbid"] for r in rows if r.get("pdbid")]


def _pdbids_from_flat(flat_dir: Path) -> List[str]:
    return sorted(p.stem for p in flat_dir.glob("*.pdbqt"))


def _write_pairs_with_alt_vina(
    *,
    pairs_csv: Path,
    pdbids: List[str],
    vina_flat_dir: Path,
    out_pairs: Path,
) -> None:
    rows = _read_csv_dicts(pairs_csv)
    keep = set(pdbids)
    rows = [r for r in rows if r.get("pdbid") in keep]
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdbid", "ref", "boltz_pred", "vina_pred"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "vina_pred": str(vina_flat_dir / f"{r['pdbid']}.pdbqt")})


def _run_benchmark_top1_contacts(*, pairs_csv: Path, out_dir: Path) -> Path:
    """
    Run the benchmark with vina-max-poses=1 to compute contact metrics for the top-1 pose only.
    Returns the resulting benchmark_allposes.csv path under out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    allposes_path = out_dir / "benchmark_allposes.csv"
    if allposes_path.exists():
        return allposes_path
    # Important: when evaluating alternative "vina_pred" files (e.g., GNINA outputs), we must
    # avoid reusing cached normalized complexes / contacts from earlier runs, since the cache
    # keys are per-pdbid and pose_index and do not incorporate the underlying pose geometry.
    cache_dir = out_dir / ".cache_lipid_benchmark"
    cmd = [
        "python",
        "scripts/benchmark.py",
        "--pairs",
        str(pairs_csv),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(cache_dir),
        "--vina-max-poses",
        "1",
        "--workers",
        "1",
        "--no-cache-normalized",
        "--no-cache-contacts",
        "--allow-errors",
        "--quiet",
    ]
    subprocess.run(cmd, check=True)
    return allposes_path


def _baseline_vina_from_allposes(
    *,
    baseline_allposes_csv: Path,
    pdbids: List[str],
    max_poses: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Return dicts keyed by pdbid:
      vina_top1_ligand_rmsd, vina_bestK_ligand_rmsd, vina_top1_headgroup_rmsd, vina_bestK_headgroup_rmsd,
      and the raw top-1 contact metric fields from the pose_index=1 vina_pose row.
    """
    rows = _read_csv_dicts(baseline_allposes_csv)
    by: Dict[str, List[Dict[str, str]]] = {p: [] for p in pdbids}
    for r in rows:
        if r.get("method") != "vina_pose":
            continue
        pid = r.get("pdbid") or ""
        if pid in by:
            by[pid].append(r)

    top1_lig: Dict[str, float] = {}
    bestk_lig: Dict[str, float] = {}
    top1_head: Dict[str, float] = {}
    bestk_head: Dict[str, float] = {}
    top1_contacts: Dict[str, Dict[str, str]] = {}
    for pid in pdbids:
        poses = by.get(pid, [])
        poses.sort(key=lambda r: int(float(r.get("pose_index") or 0)))
        lig_vals = np.array([_safe_float(r.get("ligand_rmsd")) for r in poses[:max_poses]], float)
        head_vals = np.array([_safe_float(r.get("headgroup_rmsd")) for r in poses[:max_poses]], float)
        top1_lig[pid] = float(lig_vals[0]) if lig_vals.size else float("nan")
        top1_head[pid] = float(head_vals[0]) if head_vals.size else float("nan")
        bestk_lig[pid] = float(np.nanmin(lig_vals)) if lig_vals.size else float("nan")
        bestk_head[pid] = float(np.nanmin(head_vals)) if head_vals.size else float("nan")
        top1_contacts[pid] = poses[0] if poses else {}
    return top1_lig, bestk_lig, top1_head, bestk_head, top1_contacts


def _gnina_rmsd_eval(
    *,
    pdbids: List[str],
    flat_dir: Path,
    max_poses: int,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    List[Dict[str, object]],
    Dict[str, int],
]:
    """
    Evaluate GNINA poses with the repo RMSD code.

    Returns:
      top1 ligand RMSD, best-of-K ligand RMSD, top1 headgroup RMSD, best-of-K headgroup RMSD,
      per-pose failures, ok pose counts per target.
    """
    top1_lig: Dict[str, float] = {}
    best_lig: Dict[str, float] = {}
    top1_head: Dict[str, float] = {}
    best_head: Dict[str, float] = {}
    failures: List[Dict[str, object]] = []
    ok_counts: Dict[str, int] = {}

    for pid in pdbids:
        pred_path = flat_dir / f"{pid}.pdbqt"
        rows = measure_ligand_pose_all(
            f"structures/experimental/{pid}.cif",
            pred_path,
            max_poses=max_poses,
            align_protein=False,
        )
        ok = [r for r in rows if r.get("status") == "ok"]
        err = [r for r in rows if r.get("status") != "ok"]
        ok_counts[pid] = len(ok)

        r1 = next((r for r in rows if int(r.get("pose_index") or 0) == 1), None)
        top1_lig[pid] = _safe_float(r1.get("ligand_rmsd")) if r1 and r1.get("status") == "ok" else float("nan")
        top1_head[pid] = _safe_float(r1.get("headgroup_rmsd")) if r1 and r1.get("status") == "ok" else float("nan")

        best_lig[pid] = min((_safe_float(r.get("ligand_rmsd")) for r in ok), default=float("nan"))
        best_head[pid] = min((_safe_float(r.get("headgroup_rmsd")) for r in ok), default=float("nan"))

        for e in err:
            failures.append(
                {
                    "pdbid": pid,
                    "pose_index": int(e.get("pose_index") or 0),
                    "error": str(e.get("error") or ""),
                }
            )

    return top1_lig, best_lig, top1_head, best_head, failures, ok_counts


def _extract_top1_contact_metrics(*, allposes_csv: Path, pdbids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extract top-1 contact metrics from a benchmark_allposes.csv where vina-max-poses=1 was used.
    """
    rows = _read_csv_dicts(allposes_csv)
    keep = set(pdbids)
    by: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if r.get("method") != "vina_pose":
            continue
        pid = r.get("pdbid") or ""
        if pid in keep:
            by[pid] = r
    return by


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="structures/benchmark_entries.csv", help="Pairs CSV (default: structures/benchmark_entries.csv).")
    p.add_argument(
        "--baseline-allposes",
        default="output/benchmark/benchmark_allposes.csv",
        help="Baseline benchmark_allposes.csv containing Vina poses (default: output/benchmark/benchmark_allposes.csv).",
    )
    p.add_argument("--gnina-cnn-flat", required=True, help="GNINA CNN run flat/ directory (contains <PDBID>.pdbqt).")
    p.add_argument(
        "--gnina-nocnn-flat",
        default="",
        help="Optional GNINA no-CNN run flat/ directory (contains <PDBID>.pdbqt). If omitted, only CNN analysis is run.",
    )
    p.add_argument(
        "--out-dir",
        default="output/gnina/analysis/gnina_analysis",
        help="Output directory (default: output/gnina/analysis/gnina_analysis).",
    )
    p.add_argument("--max-poses", type=int, default=20, help="Max poses to consider for best-of-K (default: 20).")
    p.add_argument(
        "--pdbids",
        default="",
        help="Optional comma-separated PDBIDs to restrict analysis; otherwise uses intersection of pairs and GNINA outputs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pairs_csv = Path(args.pairs)
    baseline_allposes = Path(args.baseline_allposes)
    gnina_cnn_flat = Path(args.gnina_cnn_flat)
    gnina_nocnn_flat = Path(args.gnina_nocnn_flat) if args.gnina_nocnn_flat.strip() else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_poses = int(args.max_poses)
    if max_poses < 1:
        raise SystemExit("--max-poses must be >= 1")

    if args.pdbids.strip():
        pdbids = [p.strip() for p in args.pdbids.split(",") if p.strip()]
    else:
        pairs_pdbids = set(_pdbids_from_pairs(pairs_csv))
        cnn_pdbids = set(_pdbids_from_flat(gnina_cnn_flat))
        if gnina_nocnn_flat is not None:
            nocnn_pdbids = set(_pdbids_from_flat(gnina_nocnn_flat))
            pdbids = sorted(pairs_pdbids & cnn_pdbids & nocnn_pdbids)
        else:
            pdbids = sorted(pairs_pdbids & cnn_pdbids)
    if not pdbids:
        raise SystemExit("No PDBIDs to analyze (check --pairs and GNINA flat directories).")

    # Baseline Vina sampling/ranking from existing benchmark output.
    vina_top1_lig, vina_best_lig, vina_top1_head, vina_best_head, vina_top1_contacts = _baseline_vina_from_allposes(
        baseline_allposes_csv=baseline_allposes,
        pdbids=pdbids,
        max_poses=max_poses,
    )

    # GNINA RMSD eval (sampling + ranking).
    cnn_top1_lig, cnn_best_lig, cnn_top1_head, cnn_best_head, cnn_failures, cnn_ok_counts = _gnina_rmsd_eval(
        pdbids=pdbids,
        flat_dir=gnina_cnn_flat,
        max_poses=max_poses,
    )
    if gnina_nocnn_flat is not None:
        nocnn_top1_lig, nocnn_best_lig, nocnn_top1_head, nocnn_best_head, nocnn_failures, nocnn_ok_counts = _gnina_rmsd_eval(
            pdbids=pdbids,
            flat_dir=gnina_nocnn_flat,
            max_poses=max_poses,
        )
    else:
        nocnn_top1_lig = {}
        nocnn_best_lig = {}
        nocnn_top1_head = {}
        nocnn_best_head = {}
        nocnn_failures = []
        nocnn_ok_counts = {}

    # Contact/interaction metrics for GNINA top-1 poses (pipeline-based).
    pairs_cnn = out_dir / "pairs_gnina_cnn.csv"
    _write_pairs_with_alt_vina(pairs_csv=pairs_csv, pdbids=pdbids, vina_flat_dir=gnina_cnn_flat, out_pairs=pairs_cnn)

    bench_cnn_dir = out_dir / "benchmark_top1_gnina_cnn"
    cnn_allposes_path = _run_benchmark_top1_contacts(pairs_csv=pairs_cnn, out_dir=bench_cnn_dir)

    cnn_contacts = _extract_top1_contact_metrics(allposes_csv=cnn_allposes_path, pdbids=pdbids)
    if gnina_nocnn_flat is not None:
        pairs_nocnn = out_dir / "pairs_gnina_nocnn.csv"
        _write_pairs_with_alt_vina(pairs_csv=pairs_csv, pdbids=pdbids, vina_flat_dir=gnina_nocnn_flat, out_pairs=pairs_nocnn)
        bench_nocnn_dir = out_dir / "benchmark_top1_gnina_nocnn"
        nocnn_allposes_path = _run_benchmark_top1_contacts(pairs_csv=pairs_nocnn, out_dir=bench_nocnn_dir)
        nocnn_contacts = _extract_top1_contact_metrics(allposes_csv=nocnn_allposes_path, pdbids=pdbids)
    else:
        nocnn_contacts = {}

    contact_fields = ["head_env_jaccard", "headgroup_typed_jaccard", "head_env_f1", "headgroup_typed_f1"]

    per_target_rows: List[Dict[str, object]] = []
    for pid in pdbids:
        row: Dict[str, object] = {
            "pdbid": pid,
            # Baseline Vina
            "vina_top1_ligand_rmsd": vina_top1_lig.get(pid, float("nan")),
            "vina_bestK_ligand_rmsd": vina_best_lig.get(pid, float("nan")),
            "vina_gap_ligand_rmsd": vina_top1_lig.get(pid, float("nan")) - vina_best_lig.get(pid, float("nan")),
            "vina_top1_headgroup_rmsd": vina_top1_head.get(pid, float("nan")),
            "vina_bestK_headgroup_rmsd": vina_best_head.get(pid, float("nan")),
            # GNINA CNN
            "gnina_cnn_top1_ligand_rmsd": cnn_top1_lig.get(pid, float("nan")),
            "gnina_cnn_bestK_ligand_rmsd": cnn_best_lig.get(pid, float("nan")),
            "gnina_cnn_gap_ligand_rmsd": cnn_top1_lig.get(pid, float("nan")) - cnn_best_lig.get(pid, float("nan")),
            "gnina_cnn_top1_headgroup_rmsd": cnn_top1_head.get(pid, float("nan")),
            "gnina_cnn_bestK_headgroup_rmsd": cnn_best_head.get(pid, float("nan")),
            "gnina_cnn_ok_poses": cnn_ok_counts.get(pid, 0),
        }
        if gnina_nocnn_flat is not None:
            row.update(
                {
                    "gnina_nocnn_top1_ligand_rmsd": nocnn_top1_lig.get(pid, float("nan")),
                    "gnina_nocnn_bestK_ligand_rmsd": nocnn_best_lig.get(pid, float("nan")),
                    "gnina_nocnn_gap_ligand_rmsd": nocnn_top1_lig.get(pid, float("nan")) - nocnn_best_lig.get(pid, float("nan")),
                    "gnina_nocnn_top1_headgroup_rmsd": nocnn_top1_head.get(pid, float("nan")),
                    "gnina_nocnn_bestK_headgroup_rmsd": nocnn_best_head.get(pid, float("nan")),
                    "gnina_nocnn_ok_poses": nocnn_ok_counts.get(pid, 0),
                }
            )
        for f in contact_fields:
            row[f"vina_{f}"] = vina_top1_contacts.get(pid, {}).get(f, "")
            row[f"gnina_cnn_{f}"] = cnn_contacts.get(pid, {}).get(f, "")
            if gnina_nocnn_flat is not None:
                row[f"gnina_nocnn_{f}"] = nocnn_contacts.get(pid, {}).get(f, "")
        per_target_rows.append(row)

    per_target_path = out_dir / "per_target.csv"
    with per_target_path.open("w", newline="") as f:
        fieldnames = sorted({k for r in per_target_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_target_rows)

    failures_path = out_dir / "pose_failures.csv"
    all_failures: List[Dict[str, object]] = []
    for r in cnn_failures:
        all_failures.append({**r, "method": "gnina_cnn"})
    if gnina_nocnn_flat is not None:
        for r in nocnn_failures:
            all_failures.append({**r, "method": "gnina_nocnn"})
    with failures_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "pdbid", "pose_index", "error"])
        w.writeheader()
        w.writerows(all_failures)

    def _arr(col: str) -> np.ndarray:
        return np.array([_safe_float(r.get(col)) for r in per_target_rows], float)

    summary_rows: List[Dict[str, object]] = []

    # H1: sampling capability (best-of-K ligand RMSD)
    metrics = ["vina_bestK_ligand_rmsd", "gnina_cnn_bestK_ligand_rmsd"]
    if gnina_nocnn_flat is not None:
        metrics.append("gnina_nocnn_bestK_ligand_rmsd")
    for metric in metrics:
        summary_rows.append({"metric": metric, **_summ(_arr(metric))})

    # H2: ranking quality (gap + top-1 ligand RMSD)
    metrics = [
        "vina_gap_ligand_rmsd",
        "gnina_cnn_gap_ligand_rmsd",
        "vina_top1_ligand_rmsd",
        "gnina_cnn_top1_ligand_rmsd",
    ]
    if gnina_nocnn_flat is not None:
        metrics.extend(["gnina_nocnn_gap_ligand_rmsd", "gnina_nocnn_top1_ligand_rmsd"])
    for metric in metrics:
        summary_rows.append({"metric": metric, **_summ(_arr(metric))})

    # H3: headgroup accuracy (top-1 headgroup RMSD)
    metrics = ["vina_top1_headgroup_rmsd", "gnina_cnn_top1_headgroup_rmsd"]
    if gnina_nocnn_flat is not None:
        metrics.append("gnina_nocnn_top1_headgroup_rmsd")
    for metric in metrics:
        summary_rows.append({"metric": metric, **_summ(_arr(metric))})

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "mean", "median", "p25", "p75", "min", "max", "pct_lt2", "pct_lt5"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    def _fmt(s: Dict[str, object]) -> str:
        return (
            f"n={int(s['n'])} mean={float(s['mean']):.3f} median={float(s['median']):.3f} "
            f"<2Å={float(s['pct_lt2']):.1f}% <5Å={float(s['pct_lt5']):.1f}%"
        )

    print(f"Analyzed {len(pdbids)} targets (best-of-K uses K={max_poses}).")
    print(f"Wrote: {per_target_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {failures_path}")

    print("\n[H1] Sampling capability (best-of-K ligand RMSD)")
    keys = ["vina_bestK_ligand_rmsd", "gnina_cnn_bestK_ligand_rmsd"]
    if gnina_nocnn_flat is not None:
        keys.append("gnina_nocnn_bestK_ligand_rmsd")
    for key in keys:
        row = next(r for r in summary_rows if r["metric"] == key)
        print(f"  {key}: {_fmt(row)}")

    print("\n[H2] Ranking quality (gap = top1 - bestK; smaller is better)")
    keys = ["vina_gap_ligand_rmsd", "gnina_cnn_gap_ligand_rmsd"]
    if gnina_nocnn_flat is not None:
        keys.append("gnina_nocnn_gap_ligand_rmsd")
    for key in keys:
        row = next(r for r in summary_rows if r["metric"] == key)
        print(f"  {key}: n={int(row['n'])} mean={float(row['mean']):.3f} median={float(row['median']):.3f}")

    print("\n[H3] Headgroup accuracy (top-1 headgroup RMSD)")
    keys = ["vina_top1_headgroup_rmsd", "gnina_cnn_top1_headgroup_rmsd"]
    if gnina_nocnn_flat is not None:
        keys.append("gnina_nocnn_top1_headgroup_rmsd")
    for key in keys:
        row = next(r for r in summary_rows if r["metric"] == key)
        print(f"  {key}: {_fmt(row)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
