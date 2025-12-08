#!/usr/bin/env python3
"""Run RMSD + contact metrics for Boltz and Vina in one go, with minimal clutter."""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:  # pragma: no cover
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from contact_tools import run_batch_contacts
from scripts import compute_contact_metrics
from scripts.lib.constants import VINA_MAX_POSES
from scripts.measure_ligand_pose_batch import _default_pairs_path


def _run_cmd(cmd: list[str]) -> None:
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    analysis_dir = project_root / "analysis"
    final_dir = analysis_dir / "final"
    tmp_dir = analysis_dir / "tmp"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = _default_pairs_path()

    # RMSD batch: Boltz
    boltz_out = tmp_dir / "boltz_batch_results.csv"
    print("[INFO] Running RMSD batch (Boltz)...")
    _run_cmd(
        [
            "python",
            str(project_root / "scripts" / "measure_ligand_pose_batch.py"),
            "--pairs",
            str(pairs_path),
            "--pred-column",
            "boltz_pred",
            "--out",
            str(boltz_out),
            "--kind",
            "boltz",
            "--max-poses",
            "1",
            "--quiet",
        ]
    )

    # RMSD batch: Vina (VINA_MAX_POSES poses)
    vina_out = tmp_dir / "vina_batch_results.csv"
    print(f"[INFO] Running RMSD batch (Vina, {VINA_MAX_POSES} poses)...")
    _run_cmd(
        [
            "python",
            str(project_root / "scripts" / "measure_ligand_pose_batch.py"),
            "--pairs",
            str(pairs_path),
            "--pred-column",
            "vina_pred",
            "--out",
            str(vina_out),
            "--kind",
            "vina",
            "--max-poses",
            str(VINA_MAX_POSES),
            "--quiet",
        ]
    )

    # Contact extraction
    print("[INFO] Extracting contacts (refs/Boltz/Vina)...")
    rc = run_batch_contacts.main(quiet=True)
    if rc != 0:
        raise RuntimeError("Contact extraction failed.")

    # Contact metrics (per-pose + summary)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_pose_path = final_dir / f"full_benchmark_allposes_{ts}.csv"
    summary_path = final_dir / f"full_benchmark_summary_{ts}.csv"
    print("[INFO] Computing contact metrics...")
    rc = compute_contact_metrics.main(
        [
            "--per-pose-out",
            str(per_pose_path),
            "--summary-out",
            str(summary_path),
            "--boltz-rmsd",
            str(boltz_out),
            "--vina-rmsd",
            str(vina_out),
            "--quiet",
        ]
    )
    if rc != 0:
        raise RuntimeError("Contact metrics computation failed.")

    # Clean intermediates
    if (analysis_dir / "pandamap_contacts").exists():
        shutil.rmtree(analysis_dir / "pandamap_contacts", ignore_errors=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[INFO] Done.")
    print(f"[INFO] Outputs: per-pose {per_pose_path}, summary {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
