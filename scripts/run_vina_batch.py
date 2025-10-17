#!/usr/bin/env python3
"""Fire-and-forget helper to run Vina across every prepared target."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Toggle to force rerunning even when outputs already exist.
FORCE_RERUN = False


def iter_prepared_ids(prep_root: Path) -> Iterable[str]:
    for entry in sorted(prep_root.iterdir()):
        if entry.is_dir():
            yield entry.name.upper()


def has_existing_run(vina_root: Path, pdbid: str) -> bool:
    target_dir = vina_root / pdbid
    return any(target_dir.glob("vina_run_*"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    prep_root = repo_root / "docking" / "prep"
    vina_root = repo_root / "model_outputs" / "vina"
    run_vina_script = repo_root / "scripts" / "run_vina.py"

    if not prep_root.exists():
        print(f"[error] Prep directory missing: {prep_root}", file=sys.stderr)
        return 1
    if not run_vina_script.exists():
        print(f"[error] Cannot find run_vina.py at {run_vina_script}", file=sys.stderr)
        return 1

    targets = list(iter_prepared_ids(prep_root))
    if not targets:
        print("[warn] No prepared targets found; run prep_vina_from_refs.py first.")
        return 0

    vina_bin = sys.executable
    python_cmd = [vina_bin, str(run_vina_script)]

    for pdbid in targets:
        if not FORCE_RERUN and has_existing_run(vina_root, pdbid):
            print(f"[skip] {pdbid} already has Vina outputs; set FORCE_RERUN=True to redo.")
            continue

        print(f"\n=== {pdbid} ===", flush=True)
        cmd = python_cmd + [pdbid]
        try:
            subprocess.run(
                cmd,
                cwd=repo_root,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] {pdbid} failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode

    print("\n[done] Vina batch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
