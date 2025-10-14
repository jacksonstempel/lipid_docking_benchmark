from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import MIN_LIGAND_HEAVY_ATOMS
from .structures import is_protein_res, is_water_res, load_structure


@dataclass
class SimpleAtom:
    name: str
    element: str
    xyz: np.ndarray


@dataclass
class SimpleResidue:
    chain_id: str
    res_name: str
    res_id: str
    atoms: List[SimpleAtom]

    def heavy_atom_count(self) -> int:
        return sum(1 for atom in self.atoms if atom.element != "H")

    def to_dict(self) -> Dict[str, str]:
        return {"chain": self.chain_id, "name": self.res_name, "id": self.res_id}


def load_ligand_template_names(project_root: Path, pdbid: str, *, include_h: bool = False) -> List[str] | None:
    """Return atom names from the docking prep template (if present)."""
    template_path = project_root / "docking" / "prep" / pdbid / "ligand.pdb"
    if not template_path.is_file():
        return None
    try:
        structure = load_structure(template_path)
    except Exception:
        return None

    names: List[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_protein_res(residue) or is_water_res(residue):
                    continue
                for atom in residue:
                    element = atom.element.name.upper()
                    if element == "H" and not include_h:
                        continue
                    names.append(atom.name.strip())
    return names or None


def apply_template_names(residues: Sequence[SimpleResidue], template_names: Sequence[str] | None) -> None:
    if not template_names:
        return
    for residue in residues:
        if len(residue.atoms) != len(template_names):
            continue
        for atom, name in zip(residue.atoms, template_names):
            atom.name = name


def collect_ligands(structure: "gemmi.Structure", *, include_h: bool = False) -> List[SimpleResidue]:
    """Collect non-protein, non-water residues with altLoc deduplication."""
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_protein_res(residue) or is_water_res(residue):
                    continue
                chain_id = chain.name if chain.name else "L"
                key = (chain_id, residue.name, str(residue.seqid))
                if key not in groups:
                    groups[key] = {"atoms": [], "altloc": {}}
                    order.append(key)
                bucket = groups[key]
                for atom in residue:
                    element = atom.element.name.upper()
                    if element == "H" and not include_h:
                        continue
                    xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                    simple_atom = SimpleAtom(name=atom.name.strip(), element=element, xyz=xyz)
                    raw_altloc = getattr(atom, "altloc", "").strip() if hasattr(atom, "altloc") else ""
                    altloc = raw_altloc if raw_altloc and raw_altloc != "\x00" else ""
                    if altloc:
                        key_alt = (atom.name.strip().upper(), altloc)
                        occ = getattr(atom, "occ", 1.0)
                        prev = bucket["altloc"].get(key_alt)
                        if prev is None or occ > prev[1]:
                            bucket["altloc"][key_alt] = (simple_atom, occ)
                    else:
                        bucket["atoms"].append(simple_atom)

    ligands: List[SimpleResidue] = []
    for chain_id, res_name, res_id in order:
        entry = groups[(chain_id, res_name, res_id)]
        atoms: List[SimpleAtom] = list(entry["atoms"])
        atoms.extend(atom for atom, _ in entry["altloc"].values())
        if atoms:
            ligands.append(SimpleResidue(chain_id, res_name, res_id, atoms))
    return ligands


def filter_large_ligands(residues: Sequence[SimpleResidue]) -> List[SimpleResidue]:
    return [residue for residue in residues if residue.heavy_atom_count() >= MIN_LIGAND_HEAVY_ATOMS]


def pairs_by_name(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int, int]]:
    lookup_pred = {atom.name.upper(): index for index, atom in enumerate(pred.atoms) if atom.element != "H"}
    lookup_ref = {atom.name.upper(): index for index, atom in enumerate(ref.atoms) if atom.element != "H"}
    common = sorted(set(lookup_pred) & set(lookup_ref))
    return [(lookup_pred[name], lookup_ref[name]) for name in common]


def locked_rmsd(
    pred_atoms: Sequence[SimpleAtom],
    ref_atoms: Sequence[SimpleAtom],
    pairs: Sequence[Tuple[int, int]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, int]:
    if len(pairs) < 3:
        return float("inf"), 0

    P = np.array([pred_atoms[i].xyz for i, _ in pairs], dtype=float)
    Q = np.array([ref_atoms[j].xyz for _, j in pairs], dtype=float)
    P = P @ rotation + translation
    rmsd = float(np.sqrt(((P - Q) ** 2).sum() / P.shape[0]))
    return rmsd, P.shape[0]
