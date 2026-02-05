#!/usr/bin/env python3
"""
Run the canonical publication workflow end-to-end.

Stages:
1) benchmark
2) database build
3) analysis + figures
4) manuscript number verification
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run(cmd: Iterable[str], *, cwd: Path, dry_run: bool) -> None:
    cmd_list = [str(c) for c in cmd]
    pretty = " ".join(cmd_list)
    print(f"$ {pretty}")
    if dry_run:
        return
    subprocess.run(cmd_list, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run benchmark -> db -> analysis -> verify.")
    parser.add_argument("--python", default=sys.executable, help="Python executable (default: current interpreter).")

    parser.add_argument("--pairs", default="structures/benchmark_entries.csv", help="Benchmark pair CSV path.")
    parser.add_argument("--benchmark-out-dir", default="output/benchmark", help="Benchmark output directory.")

    parser.add_argument("--db-path", default="output/benchmark/benchmark_full.sqlite", help="Unified SQLite output path.")
    parser.add_argument("--analysis-out-dir", default="output/analysis/db_pipeline", help="Analysis CSV output directory.")
    parser.add_argument("--fig-dir", default="manuscript/figures", help="Figure output directory.")

    parser.add_argument("--summary-csv", default="output/benchmark/benchmark_summary.csv", help="Baseline summary CSV used by analysis script.")
    parser.add_argument(
        "--adversarial-root",
        default="output/adversarial/bs_mutagenesis_cutoff5A",
        help="Adversarial output root used by analysis script.",
    )
    parser.add_argument(
        "--adversarial-protein-rmsd-cutoffs",
        default="2.0,1.5",
        help="Comma-separated cutoffs forwarded to analysis script.",
    )
    parser.add_argument("--k", type=int, default=20, help="Top-K used in analysis stage.")

    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark stage.")
    parser.add_argument("--skip-db", action="store_true", help="Skip database stage.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip manuscript verification stage.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    py = str(args.python)

    if not args.skip_benchmark:
        _run(
            [
                py,
                "scripts/benchmark.py",
                "--pairs",
                str(args.pairs),
                "--out-dir",
                str(args.benchmark_out_dir),
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_db:
        _run(
            [
                py,
                "scripts/build_benchmark_db.py",
                "--out",
                str(args.db_path),
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_analysis:
        _run(
            [
                py,
                "scripts/analyze_benchmark_db.py",
                "--db",
                str(args.db_path),
                "--out-dir",
                str(args.analysis_out_dir),
                "--fig-dir",
                str(args.fig_dir),
                "--summary-csv",
                str(args.summary_csv),
                "--adversarial-root",
                str(args.adversarial_root),
                "--adversarial-protein-rmsd-cutoffs",
                str(args.adversarial_protein_rmsd_cutoffs),
                "--k",
                str(int(args.k)),
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_verify:
        _run(
            [py, "scripts/verify_manuscript_numbers.py"],
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
