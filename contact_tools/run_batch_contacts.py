#!/usr/bin/env python3
"""Batch-run contact extraction for refs, Boltz, and Vina structures.

Usage:
  python contact_tools/run_batch_contacts.py

Behavior:
  - Scans the standard directories for available structures:
      benchmark_references/<pdbid>.cif
      model_outputs/boltz/<pdbid>_model_0.cif
      model_outputs/vina/<pdbid>.pdbqt
  - Builds the largest common PDB ID set present in all three locations.
  - For each PDB ID, extracts contacts for ref, Boltz, and Vina; if any step
    fails for that PDB ID, it is skipped to keep all three outputs aligned.
  - Emits three aggregated CSVs under analysis/pandamap_contacts/:
      ref_contacts.csv, boltz_contacts.csv, vina_contacts.csv
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Tuple

from contact_tools.measure_contacts import extract_contacts
from scripts.lib.constants import VINA_MAX_POSES
from scripts.lib.structures import load_structure

LOGGER = logging.getLogger("run_batch_contacts")

REF_POSES = 1
BOLTZ_POSES = 1


def _collect_ids(ref_dir: Path, boltz_dir: Path, vina_dir: Path) -> Tuple[set[str], set[str], set[str], list[str]]:
    ref_ids = {p.stem for p in ref_dir.glob("*.cif")}
    boltz_ids = {p.name.split("_")[0] for p in boltz_dir.glob("*.cif")}
    vina_ids = {p.stem for p in vina_dir.glob("*.pdbqt")}
    common = sorted(ref_ids & boltz_ids & vina_ids)
    return ref_ids, boltz_ids, vina_ids, common


def main(quiet: bool | None = None) -> int:
    quiet = True if quiet is None else bool(quiet)
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="[%(levelname)s] %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    ref_dir = project_root / "benchmark_references"
    boltz_dir = project_root / "model_outputs" / "boltz"
    vina_dir = project_root / "model_outputs" / "vina"
    out_dir = project_root / "analysis" / "pandamap_contacts"

    # Clean old files to avoid clutter
    if out_dir.exists():
        for f in out_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_ids, boltz_ids, vina_ids, pdbids = _collect_ids(ref_dir, boltz_dir, vina_dir)
    LOGGER.info(
        "Found IDs — ref: %d, boltz: %d, vina: %d; common triplets: %d",
        len(ref_ids),
        len(boltz_ids),
        len(vina_ids),
        len(pdbids),
    )

    agg = {"ref": [], "boltz": [], "vina": []}
    success = 0
    skipped = 0
    total = len(pdbids)
    for idx, pdbid in enumerate(pdbids, start=1):
        ref_path = ref_dir / f"{pdbid}.cif"
        boltz_path = boltz_dir / f"{pdbid}_model_0.cif"
        vina_path = vina_dir / f"{pdbid}.pdbqt"

        try:
            ref_structure = load_structure(ref_path)
            ref_contacts = extract_contacts(
                ref_path, ref_structure=ref_structure, max_models=REF_POSES
            )
            boltz_contacts = extract_contacts(
                boltz_path, ref_structure=ref_structure, max_models=BOLTZ_POSES
            )
            vina_contacts = extract_contacts(
                vina_path, ref_structure=ref_structure, max_models=VINA_MAX_POSES
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s due to error: %s", pdbid, exc)
            skipped += 1
            continue

        # If any set is empty, treat this as a failure for that PDB.
        if not ref_contacts or not boltz_contacts or not vina_contacts:
            LOGGER.warning(
                "Skipping %s because one or more contact sets are empty (ref=%d, boltz=%d, vina=%d)",
                pdbid,
                len(ref_contacts),
                len(boltz_contacts),
                len(vina_contacts),
            )
            skipped += 1
            continue

        for c in ref_contacts:
            agg["ref"].append({"pdbid": pdbid, **c})
        for c in boltz_contacts:
            agg["boltz"].append({"pdbid": pdbid, **c})
        for c in vina_contacts:
            agg["vina"].append({"pdbid": pdbid, **c})
        success += 1
        if quiet:
            print(f"\rPose Contacts: {idx}/{total}", end="", flush=True)

    # Write aggregated CSVs
    for kind, rows in agg.items():
        out_file = out_dir / f"{kind}_contacts.csv"
        fieldnames = ["pdbid", "pose_index", "ligand_atom", "residue", "contact_type", "distance"]
        with out_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.debug("Wrote %d rows to %s", len(rows), out_file)

    if quiet:
        print()
    if skipped:
        LOGGER.error("Contact extraction finished with issues. Successful PDBs: %d, skipped: %d", success, skipped)
        return 1

    LOGGER.info("Contact extraction completed successfully for %d PDBs.", success)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract contacts for refs/Boltz/Vina.")
    parser.add_argument("--quiet", action="store_true", default=True, help="Suppress detailed logs and show compact progress.")
    parser.add_argument("--no-quiet", action="store_false", dest="quiet", help="Disable quiet mode.")
    args = parser.parse_args()
    raise SystemExit(main(quiet=args.quiet))
