#!/usr/bin/env python3
"""
Analyze a unified benchmark SQLite database and emit manuscript-ready outputs.

This script:
  - builds per-target metrics used in GNINA/Vina/Boltz figures
  - writes summary tables (numeric + formatted)
  - computes torsion-bin statistics with Wilson confidence intervals
  - regenerates adversarial mutagenesis figures (WT vs Gly vs Phe)
  - regenerates manuscript figures from the database
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from lipid_benchmark.analysis_db import (
    build_per_target,
    summarize_methods,
    torsion_bin_table,
)


def _load_allposes(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM allposes", con)
    finally:
        con.close()


def _load_torsions(db_path: Path) -> pd.Series:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT pdbid, torsdof FROM targets", con)
    finally:
        con.close()
    return df.set_index("pdbid")["torsdof"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the benchmark SQLite database.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("output/benchmark/benchmark_full.sqlite"),
        help="SQLite database path (default: output/benchmark/benchmark_full.sqlite).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/analysis/db_pipeline"),
        help="Output directory for analysis CSVs.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("manuscript/figures"),
        help="Figure output directory (default: manuscript/figures).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("output/benchmark/benchmark_summary.csv"),
        help="Baseline benchmark_summary.csv (used for WT frames; default: output/benchmark/benchmark_summary.csv).",
    )
    parser.add_argument(
        "--adversarial-root",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A"),
        help="Root directory for adversarial mutagenesis outputs (default: output/adversarial/bs_mutagenesis_cutoff5A).",
    )
    parser.add_argument(
        "--adversarial-protein-rmsd-cutoffs",
        default="2.0,1.5",
        help="Comma-separated protein RMSD cutoffs to render for adversarial plots (default: '2.0,1.5').",
    )
    parser.add_argument("--k", type=int, default=20, help="Top-K for best-of-K analyses (default: 20).")
    args = parser.parse_args()

    allposes = _load_allposes(args.db)
    per_target = build_per_target(allposes, k=int(args.k))

    summary = summarize_methods(per_target)
    torsions = _load_torsions(args.db)
    torsion_table = torsion_bin_table(per_target, torsions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_target.to_csv(args.out_dir / "per_target.csv", index=False)
    summary.numeric.to_csv(args.out_dir / "summary_table_numeric.csv", index=False)
    summary.formatted.to_csv(args.out_dir / "summary_table_formatted.csv", index=False)
    torsion_table.numeric.to_csv(args.out_dir / "torsion_table_numeric.csv", index=False)
    torsion_table.formatted.to_csv(args.out_dir / "torsion_table_formatted.csv", index=False)

    # Generate manuscript figures directly from the per-target table.
    import importlib.util
    import sys

    plot_path = Path(__file__).resolve().parents[0] / "plot_results.py"
    spec = importlib.util.spec_from_file_location("plot_results", plot_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot_results from {plot_path}")
    plots = importlib.util.module_from_spec(spec)
    sys.modules["plot_results"] = plots
    spec.loader.exec_module(plots)

    gnina_frames = plots.GninaFrames(per_target=per_target)
    plots.plot_top1_rmsd_methods(gnina_frames, out_dir=args.fig_dir, preview_png=False)
    plots.plot_per_target_comparison_gnina(gnina_frames, out_dir=args.fig_dir, preview_png=False)
    plots.plot_sampling_vs_ranking_gnina(gnina_frames, out_dir=args.fig_dir, preview_png=False)
    plots.plot_contact_overlap_methods(gnina_frames, out_dir=args.fig_dir, preview_png=False)

    # Adversarial mutagenesis: use WT frames from the baseline summary CSV and mutant summaries
    # from the adversarial output directory (Boltz-only rows are used for mutants).
    try:
        frames = plots._load_frames(Path(args.summary_csv))  # private helper; keeps WT Boltz/Vina aligned
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Unable to load baseline summary CSV for adversarial plots: {args.summary_csv}") from e

    gly_summary = Path(args.adversarial_root) / "benchmark_gly" / "benchmark_summary.csv"
    phe_summary = Path(args.adversarial_root) / "benchmark_phe" / "benchmark_summary.csv"

    cutoffs: list[float] = []
    for raw in str(args.adversarial_protein_rmsd_cutoffs).split(","):
        raw = raw.strip()
        if not raw:
            continue
        cutoffs.append(float(raw))
    if not cutoffs:
        cutoffs = [2.0]

    for cutoff in cutoffs:
        stem = f"fig_adversarial_mutagenesis_prot{cutoff:g}A"
        plots.plot_adversarial_mutagenesis(
            frames,
            gly_summary_csv=gly_summary,
            phe_summary_csv=phe_summary,
            out_dir=args.fig_dir,
            protein_rmsd_cutoff_a=float(cutoff),
            stem=stem,
            preview_png=False,
        )


if __name__ == "__main__":
    main()
