#!/usr/bin/env python3
"""Measure ligand pose quality between a reference and a prediction.

This script is deliberately minimal for biologist-friendly use:
  - You provide two structure files (reference and prediction).
  - It aligns proteins (Cα), auto-detects the single significant ligand,
    matches ligand atoms (chemistry-aware), and reports heavy-atom RMSD.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/measure_ligand_pose.py` without installation.
if __package__ in {None, ""}:  # pragma: no cover
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.ligand_pose_core import (
    AtomPairingError,
    LigandSelectionError,
    measure_ligand_pose,
)

LOGGER = logging.getLogger("measure_ligand_pose")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align proteins, auto-detect the ligand, and report heavy-atom ligand RMSD.",
    )
    parser.add_argument("--ref", required=True, help="Path to reference structure (mmCIF or PDB).")
    parser.add_argument("--pred", required=True, help="Path to predicted structure (PDB/PDBQT/CIF).")
    parser.add_argument(
        "--max-poses",
        type=int,
        default=1,
        help="Evaluate up to this many poses/models in the prediction and report the best RMSD (default: 1).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def _check_file(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        ref_path = _check_file(args.ref, "Reference file")
        pred_path = _check_file(args.pred, "Prediction file")
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        result = measure_ligand_pose(ref_path, pred_path, max_poses=max(1, args.max_poses))
    except (LigandSelectionError, AtomPairingError, RuntimeError) as exc:
        LOGGER.error("Measurement failed: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - safeguard
        LOGGER.exception("Unexpected failure: %s", exc)
        return 2

    print(f"Reference:  {result['ref_path']}")
    print(f"Prediction: {result['pred_path']}")
    print(f"Ligand:     {result['ligand_id']}")
    print()
    print("Protein alignment:")
    print(f"  Cα pairs used: {result['protein_pairs']}")
    print(f"  RMSD:          {result['protein_rmsd']:.3f} Å")
    print()
    print("Ligand:")
    print(f"  Pairing method: {result['pairing_method']}")
    print(f"  Heavy atoms:    {result['ligand_heavy_atoms']}")
    print(f"  RMSD:           {result['ligand_rmsd']:.3f} Å")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
