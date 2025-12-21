from __future__ import annotations

from pathlib import Path
from typing import Sequence, Set

import numpy as np
from scipy.spatial import cKDTree

from .normalization import NORMALIZED_LIGAND_RESNAME
from .structures import is_protein_res, load_structure

DEFAULT_HEAD_ENV_CUTOFF_A = 5.0


def headgroup_environment_residues(
    struct_path: Path,
    *,
    headgroup_atom_names: Sequence[str],
    cutoff_a: float = DEFAULT_HEAD_ENV_CUTOFF_A,
    ligand_resname: str = NORMALIZED_LIGAND_RESNAME,
) -> Set[str]:
    """Return protein residue IDs within `cutoff_a` Å of any headgroup atom.

    Residue IDs are returned as "<chain>:<resname>:<resnum>" (chain is 1-char).
    """
    wanted = [str(name).strip() for name in headgroup_atom_names if str(name).strip()]
    if not wanted:
        return set()

    structure = load_structure(struct_path)
    model = structure[0]

    ligand_res = None
    for chain in model:
        for residue in chain:
            if residue.het_flag == "H" and residue.name == ligand_resname:
                ligand_res = residue
                break
        if ligand_res is not None:
            break
    if ligand_res is None:
        raise RuntimeError(f"{struct_path}: missing normalized ligand residue {ligand_resname}")

    lig_atoms = {atom.name.strip(): atom for atom in ligand_res}
    head_xyz: list[list[float]] = []
    missing: list[str] = []
    for name in wanted:
        atom = lig_atoms.get(name)
        if atom is None:
            missing.append(name)
            continue
        if atom.element.name.upper() == "H":
            continue
        head_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if missing:
        raise RuntimeError(f"{struct_path}: missing {len(missing)} headgroup atoms (e.g., {missing[:5]})")
    if not head_xyz:
        return set()

    prot_xyz: list[list[float]] = []
    prot_res_ids: list[str] = []
    for chain in model:
        chain_id = (chain.name or "").strip()[:1]
        for residue in chain:
            if not is_protein_res(residue):
                continue
            res_id = f"{chain_id}:{residue.name}:{residue.seqid.num}"
            for atom in residue:
                if atom.element.name.upper() == "H":
                    continue
                prot_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                prot_res_ids.append(res_id)

    if not prot_xyz:
        return set()

    tree = cKDTree(np.array(head_xyz, dtype=float))
    dists, _ = tree.query(np.array(prot_xyz, dtype=float), k=1)
    residues = {rid for rid, d in zip(prot_res_ids, dists) if float(d) <= float(cutoff_a)}
    return residues
