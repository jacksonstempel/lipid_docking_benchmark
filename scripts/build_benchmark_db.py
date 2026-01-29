#!/usr/bin/env python3
"""
Build a single SQLite database containing all per-pose benchmark metrics.

The output database is intended to be the *one* file needed to reproduce
all manuscript tables and figures. It merges:
  - Boltz + Vina benchmark_allposes.csv
  - GNINA CNN benchmark_allposes.csv
  - GNINA no-CNN benchmark_allposes.csv
  - Adversarial mutagenesis (Boltz-only) benchmark_allposes.csv for Gly/Phe arms
and normalizes method labels to:
  boltz, vina_pose, gnina_cnn_pose, gnina_nocnn_pose, boltz_bs5A_gly, boltz_bs5A_phe.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_allposes(path: Path, *, method_map: dict[str, str] | None = None, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if method_map:
        df = df[df["method"].isin(method_map.keys())].copy()
        df["method"] = df["method"].map(method_map)
    df["source"] = source
    return df


def _read_torsdof(pdbqt_path: Path) -> int | None:
    try:
        for line in pdbqt_path.read_text().splitlines():
            if line.startswith("TORSDOF"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(float(parts[1]))
    except OSError:
        return None
    return None


def build_db(
    *,
    out_path: Path,
    baseline_allposes: Path,
    gnina_cnn_allposes: Path,
    gnina_nocnn_allposes: Path,
    adversarial_gly_allposes: Path | None,
    adversarial_phe_allposes: Path | None,
    adversarial_gly_summary: Path | None,
    adversarial_phe_summary: Path | None,
    vina_dir: Path,
) -> None:
    base = _load_allposes(baseline_allposes, source="baseline", method_map=None)
    gnina_cnn = _load_allposes(
        gnina_cnn_allposes,
        source="gnina_cnn",
        method_map={"vina_pose": "gnina_cnn_pose"},
    )
    gnina_nocnn = _load_allposes(
        gnina_nocnn_allposes,
        source="gnina_nocnn",
        method_map={"vina_pose": "gnina_nocnn_pose"},
    )

    # Drop Boltz rows from GNINA exports to avoid duplicate Boltz entries.
    gnina_cnn = gnina_cnn[gnina_cnn["method"] != "boltz"].copy()
    gnina_nocnn = gnina_nocnn[gnina_nocnn["method"] != "boltz"].copy()

    frames = [base, gnina_cnn, gnina_nocnn]

    # Adversarial mutagenesis: keep only the Boltz prediction rows and normalize method labels.
    if adversarial_gly_allposes is not None and adversarial_gly_allposes.exists():
        gly = _load_allposes(
            adversarial_gly_allposes,
            source="adversarial_bs5A_gly_v1",
            method_map={"boltz": "boltz_bs5A_gly"},
        )
        frames.append(gly)
    if adversarial_phe_allposes is not None and adversarial_phe_allposes.exists():
        phe = _load_allposes(
            adversarial_phe_allposes,
            source="adversarial_bs5A_phe_v1",
            method_map={"boltz": "boltz_bs5A_phe"},
        )
        frames.append(phe)

    allposes = pd.concat(frames, ignore_index=True)

    pdbids = sorted(allposes["pdbid"].dropna().unique().tolist())
    torsions = []
    for pdbid in pdbids:
        torsdof = _read_torsdof(vina_dir / f"{pdbid}.pdbqt")
        torsions.append({"pdbid": pdbid, "torsdof": torsdof})
    targets = pd.DataFrame(torsions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(out_path)
    try:
        allposes.to_sql("allposes", con, index=False)
        targets.to_sql("targets", con, index=False)

        con.execute("CREATE INDEX idx_allposes_pdbid ON allposes(pdbid)")
        con.execute("CREATE INDEX idx_allposes_method ON allposes(method)")
        con.execute("CREATE INDEX idx_allposes_method_pose ON allposes(method, pose_index)")

        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_allposes": str(baseline_allposes),
            "gnina_cnn_allposes": str(gnina_cnn_allposes),
            "gnina_nocnn_allposes": str(gnina_nocnn_allposes),
            "adversarial_gly_allposes": str(adversarial_gly_allposes) if adversarial_gly_allposes else "",
            "adversarial_phe_allposes": str(adversarial_phe_allposes) if adversarial_phe_allposes else "",
            "adversarial_gly_summary": str(adversarial_gly_summary) if adversarial_gly_summary else "",
            "adversarial_phe_summary": str(adversarial_phe_summary) if adversarial_phe_summary else "",
            "vina_dir": str(vina_dir),
            "schema_version": "1",
        }
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))

        # Minimal adversarial summary table (top-1 only) used for reporting.
        # This is intentionally a small, schema-stable slice (avoid duplicating the full CSV).
        def _load_adv_summary(path: Path, *, variant: str, source: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            # Clarity: the adversarial experiment is about Boltz. Vina is unchanged and does
            # not have a meaningful "mutant" counterpart, so we only store Boltz rows here.
            df = df[df["method"] == "boltz"].copy()
            keep_cols = [
                "pdbid",
                "method",
                "pose_index",
                "ligand_rmsd",
                "headgroup_rmsd",
                "protein_rmsd",
                "head_env_f1",
                "headgroup_typed_f1",
            ]
            df = df[keep_cols].copy()
            df.insert(0, "source", source)
            df.insert(0, "variant", variant)
            return df

        adv_frames = []
        if adversarial_gly_summary is not None and Path(adversarial_gly_summary).exists():
            adv_frames.append(_load_adv_summary(Path(adversarial_gly_summary), variant="bs5A_gly", source="adversarial_bs5A_gly_v1"))
        if adversarial_phe_summary is not None and Path(adversarial_phe_summary).exists():
            adv_frames.append(_load_adv_summary(Path(adversarial_phe_summary), variant="bs5A_phe", source="adversarial_bs5A_phe_v1"))
        if adv_frames:
            adv = pd.concat(adv_frames, ignore_index=True)
            adv.to_sql("adversarial_summary", con, index=False)

        con.commit()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified benchmark SQLite database.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/benchmark/benchmark_full.sqlite"),
        help="Output SQLite file (default: output/benchmark/benchmark_full.sqlite).",
    )
    parser.add_argument(
        "--baseline-allposes",
        type=Path,
        default=Path("output/benchmark/benchmark_allposes.csv"),
        help="Baseline Boltz+Vina benchmark_allposes.csv.",
    )
    parser.add_argument(
        "--gnina-cnn-allposes",
        type=Path,
        default=Path(
            "output/gnina/analysis/gnina_full_analysis_v7/benchmark_allposes_gnina_cnn_full/benchmark_allposes.csv"
        ),
        help="GNINA CNN benchmark_allposes.csv.",
    )
    parser.add_argument(
        "--gnina-nocnn-allposes",
        type=Path,
        default=Path(
            "output/gnina/analysis/gnina_full_analysis_v7/benchmark_allposes_gnina_nocnn_full/benchmark_allposes.csv"
        ),
        help="GNINA no-CNN benchmark_allposes.csv.",
    )
    parser.add_argument(
        "--adversarial-gly-allposes",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_allposes.csv"),
        help="Adversarial Gly benchmark_allposes.csv (Boltz+Vina; Boltz rows are ingested).",
    )
    parser.add_argument(
        "--adversarial-phe-allposes",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/benchmark_phe/benchmark_allposes.csv"),
        help="Adversarial Phe benchmark_allposes.csv (Boltz+Vina; Boltz rows are ingested).",
    )
    parser.add_argument(
        "--adversarial-gly-summary",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_summary.csv"),
        help="Adversarial Gly benchmark_summary.csv (top-1 only).",
    )
    parser.add_argument(
        "--adversarial-phe-summary",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/benchmark_phe/benchmark_summary.csv"),
        help="Adversarial Phe benchmark_summary.csv (top-1 only).",
    )
    parser.add_argument(
        "--vina-dir",
        type=Path,
        default=Path("structures/vina"),
        help="Directory containing Vina PDBQT files (for TORSDOF).",
    )
    args = parser.parse_args()
    build_db(
        out_path=args.out,
        baseline_allposes=args.baseline_allposes,
        gnina_cnn_allposes=args.gnina_cnn_allposes,
        gnina_nocnn_allposes=args.gnina_nocnn_allposes,
        adversarial_gly_allposes=args.adversarial_gly_allposes if args.adversarial_gly_allposes.exists() else None,
        adversarial_phe_allposes=args.adversarial_phe_allposes if args.adversarial_phe_allposes.exists() else None,
        adversarial_gly_summary=args.adversarial_gly_summary if args.adversarial_gly_summary.exists() else None,
        adversarial_phe_summary=args.adversarial_phe_summary if args.adversarial_phe_summary.exists() else None,
        vina_dir=args.vina_dir,
    )


if __name__ == "__main__":
    main()
