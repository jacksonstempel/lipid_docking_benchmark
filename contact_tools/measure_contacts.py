#!/usr/bin/env python3
"""Extract ligand–protein contacts for a structure (optionally multi-pose).

Usage:
  python contact_tools/measure_contacts.py --in path/to/structure.cif [--out contacts.csv]

Behavior:
  - Auto-selects a single ligand (same heuristics as the simplified pose scripts).
  - Uses PandaMap for contact detection; if PandaMap is not available or fails,
    the extraction fails with a clear error.
  - Can iterate over multiple poses/models in the input (e.g., multi-model PDBQT);
    each pose is handled independently and annotated with its pose index.
  - Writes a CSV of contacts with columns: pose_index, ligand_atom, residue, contact_type, distance.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple

import numpy as np
import contextlib
import os

if __package__ in {None, ""}:  # pragma: no cover
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.ligand_pose_core import _filter_ligands, _guess_pdbid_from_path, _select_single_ligand
from scripts.lib.ligands import apply_template_names, collect_ligands, load_ligand_template_names
from scripts.lib.structures import ensure_protein_backbone, is_protein_res, load_structure, split_models

LOGGER = logging.getLogger("measure_contacts")


def _extract_with_pandamap(struct_path: Path, ligand) -> List[Dict[str, object]]:
    """Call PandaMap HybridProtLigMapper to get typed interactions."""
    # Avoid local contact_tools shadowing the real PandaMap package
    from importlib import import_module

    project_root = Path(__file__).resolve().parent.parent
    orig_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in ("", str(project_root))]
    try:
        pm = import_module("pandamap")
        Mapper = getattr(pm, "HybridProtLigMapper", None)
        if Mapper is None:
            raise ImportError("HybridProtLigMapper not found in PandaMap.")
        mapper = Mapper(str(struct_path), ligand_resname=ligand.res_name)
        # Suppress verbose stdout/stderr from PandaMap itself.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
            devnull
        ):
            mapper.detect_interactions()
        contacts: List[Dict[str, object]] = []
        for ctype, entries in mapper.interactions.items():
            for entry in entries:
                lig_atom = entry.get("ligand_atom")
                prot_res = entry.get("protein_residue")
                dist = entry.get("distance") or entry.get("dist")
                # Residue identifier from BioPython residue object
                residue_id = ""
                try:
                    chain_id = prot_res.get_parent().id if prot_res else ""
                    resname = prot_res.resname if prot_res else ""
                    resnum = prot_res.id[1] if prot_res else ""
                    residue_id = f"{chain_id}:{resname}:{resnum}"
                except (AttributeError, TypeError):
                    residue_id = ""
                # Atom name
                if hasattr(lig_atom, "get_id"):
                    lig_name = lig_atom.get_id()
                else:
                    lig_name = str(lig_atom)
                contacts.append(
                    {
                        "ligand_atom": lig_name,
                        "residue": residue_id,
                        "contact_type": ctype,
                        "distance": float(dist) if dist is not None else "",
                    }
                )
        return contacts
    finally:
        sys.path = orig_path


def _contacts_for_pose(structure, ref_structure, template_names, pose_index: int) -> List[Dict[str, object]]:
    """Run ligand selection and contact detection for one pose."""
    if ref_structure is not None:
        structure = ensure_protein_backbone(structure, ref_structure)

    ligands = collect_ligands(structure, include_h=False)
    if template_names:
        apply_template_names(ligands, template_names)
    candidates = _filter_ligands(ligands)
    if not candidates:
        raise RuntimeError("No obvious ligand found after filtering.")
    ligand = _select_single_ligand(structure, include_h=False)

    tmp_pdb = None
    tmp_pdb = Path(NamedTemporaryFile(suffix=".pdb", delete=False).name)
    try:
        structure.write_pdb(str(tmp_pdb))
        pose_contacts = _extract_with_pandamap(tmp_pdb, ligand)
    finally:
        if tmp_pdb and tmp_pdb.exists():
            try:
                tmp_pdb.unlink()
            except OSError:
                pass

    for c in pose_contacts:
        c["pose_index"] = pose_index
    return pose_contacts


def _convert_pdbqt_to_pdb(struct_path: Path) -> Tuple[Path, Path | None]:
    """Convert PDBQT to multi-model PDB if needed; return (prepared_path, temp_to_cleanup).

    For Vina outputs, this preserves MODEL/ENDMDL separation so that each pose
    remains a distinct model. Non-protein residues are marked as HETATM to help
    PandaMap recognize the ligand.
    """
    if struct_path.suffix.lower() != ".pdbqt":
        return struct_path, None

    # Load with gemmi so we keep all models, then re-write as a clean PDB.
    structure = load_structure(struct_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_protein_res(residue):
                    residue.het_flag = "H"

    tmp = NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    structure.write_pdb(str(tmp_path))
    return tmp_path, tmp_path


def extract_contacts(
    struct_path: Path,
    out_path: Path | None = None,
    *,
    ref_structure=None,
    max_models: int = 1,
) -> List[Dict[str, object]]:
    """Extract contacts for up to `max_models` poses; returns a flat list."""
    prepared_path, tmp_cleanup = _convert_pdbqt_to_pdb(struct_path)
    structure = load_structure(prepared_path)

    pdbid_guess = _guess_pdbid_from_path(struct_path)
    template_names = None
    if pdbid_guess:
        project_root = Path(__file__).resolve().parent.parent
        template_names = load_ligand_template_names(project_root, pdbid_guess, include_h=False)

    poses = split_models(structure, max_models)
    if not poses:
        raise RuntimeError("Structure contains no models/poses.")

    all_contacts: List[Dict[str, object]] = []
    for pose_index, pose_struct in enumerate(poses, start=1):
        all_contacts.extend(
            _contacts_for_pose(pose_struct, ref_structure, template_names, pose_index)
        )

    if tmp_cleanup and tmp_cleanup.exists():
        try:
            tmp_cleanup.unlink()
        except OSError:
            pass

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["pose_index", "ligand_atom", "residue", "contact_type", "distance"]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_contacts)
        LOGGER.info("Wrote %d contacts to %s", len(all_contacts), out_path)
    return all_contacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract ligand–protein contacts for a single structure.")
    p.add_argument("--in", dest="inp", required=True, help="Input structure (CIF/PDB/PDBQT).")
    p.add_argument("--out", dest="out", help="Output CSV path (default: alongside input as <name>_contacts.csv).")
    p.add_argument("--max-poses", type=int, default=1, help="Max poses/models to consider (default: 1).")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    struct_path = Path(args.inp).expanduser().resolve()
    if not struct_path.exists():
        LOGGER.error("Input does not exist: %s", struct_path)
        return 2
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = struct_path.with_name(f"{struct_path.stem}_contacts.csv")

    try:
        extract_contacts(struct_path, out_path, max_models=max(1, args.max_poses))
    except (FileNotFoundError, OSError, RuntimeError, ValueError, ImportError) as exc:
        LOGGER.error("Failed to extract contacts: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
