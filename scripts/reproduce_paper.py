#!/usr/bin/env python3
"""
Run the canonical manuscript-reproduction workflow end-to-end.

Stages:
1) database build from the manuscript archive
2) analysis + figures
3) manuscript number verification
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from lipid_benchmark.public_archive import canonical_repro_archive, require_paths


def _run(cmd: Iterable[str], *, cwd: Path) -> None:
    cmd_list = [str(c) for c in cmd]
    pretty = " ".join(cmd_list)
    print(f"$ {pretty}")
    subprocess.run(cmd_list, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    archive = canonical_repro_archive(repo_root)
    parser = argparse.ArgumentParser(description="Rebuild the manuscript analysis bundle from the manuscript archive.")
    parser.add_argument("--python", default=sys.executable, help="Python executable (default: current interpreter).")
    parser.add_argument("--db-path", default="output/benchmark/benchmark_full.sqlite", help="Unified SQLite output path.")
    parser.add_argument("--analysis-out-dir", default="output/analysis/db_pipeline", help="Analysis CSV output directory.")
    parser.add_argument("--fig-dir", default="manuscript/figures", help="Figure output directory.")
    parser.add_argument("--summary-csv", default=str(archive.baseline_summary), help="Baseline summary CSV used by analysis.")
    parser.add_argument("--adversarial-root", default=str(archive.adversarial_root), help="Tracked adversarial summary root used by analysis.")
    parser.add_argument(
        "--adversarial-protein-rmsd-cutoffs",
        default="2.0,1.5",
        help="Comma-separated cutoffs forwarded to analysis script.",
    )
    parser.add_argument("--k", type=int, default=20, help="Top-K used in analysis stage.")

    args = parser.parse_args(argv)
    py = str(args.python)
    require_paths(
        [
            archive.baseline_allposes,
            archive.baseline_summary,
            archive.gnina_cnn_allposes,
            archive.gnina_nocnn_allposes,
            archive.adversarial_gly_allposes,
            archive.adversarial_gly_summary,
            archive.adversarial_phe_allposes,
            archive.adversarial_phe_summary,
            archive.vina_exh256_allposes,
            archive.vina_exh256_summary,
            archive.boltz_high_sampling_allposes,
            archive.boltz_high_sampling_summary,
        ]
    )

    _run(
        [
            py,
            "scripts/build_benchmark_db.py",
            "--out",
            str(args.db_path),
        ],
        cwd=repo_root,
    )

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
    )

    _run(
        [py, "scripts/verify_manuscript_numbers.py"],
        cwd=repo_root,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
