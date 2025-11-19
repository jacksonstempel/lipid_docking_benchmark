from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

from .alignment import ChainPair, kabsch
from .ligands import SimpleResidue
from .structures import is_protein_res

if TYPE_CHECKING:
    import gemmi


def residue_key(chain: "gemmi.Chain", residue: "gemmi.Residue") -> Tuple[str, str]:
    return chain.name, str(residue.seqid)


def _ca_position(residue: "gemmi.Residue") -> np.ndarray | None:
    for atom in residue:
        if atom.name.strip().upper() == "CA":
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
    return None


def find_pocket_keys(reference: "gemmi.Structure", ligand: SimpleResidue, radius: float) -> Set[Tuple[str, str]]:
    ligand_positions = np.array([atom.xyz for atom in ligand.atoms], dtype=float)
    keys: Set[Tuple[str, str]] = set()
    for model in reference:
        for chain in model:
            for residue in chain:
                if not is_protein_res(residue):
                    continue
                ca_atom = _ca_position(residue)
                if ca_atom is None:
                    continue
                for atom in residue:
                    if atom.element.name.upper() == "H":
                        continue
                    atom_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                    if np.min(np.linalg.norm(ligand_positions - atom_pos, axis=1)) <= radius:
                        keys.add(residue_key(chain, residue))
                        break
    return keys


def _reference_to_prediction(chain_pairs: Sequence[ChainPair]) -> Dict[Tuple[str, str], Tuple[str, str, "gemmi.Residue", "gemmi.Residue"]]:
    mapping: Dict[Tuple[str, str], Tuple[str, str, "gemmi.Residue", "gemmi.Residue"]] = {}
    for pair in chain_pairs:
        for pred_index, ref_index in pair.res_pairs:
            ref_res = pair.ref.res_objs[ref_index]
            pred_res = pair.pred.res_objs[pred_index]
            mapping[(pair.ref.chain.name, str(ref_res.seqid))] = (
                pair.pred.chain.name,
                str(pred_res.seqid),
                pred_res,
                ref_res,
            )
    return mapping


def local_pocket_fit(
    reference: "gemmi.Structure",
    chain_pairs: Sequence[ChainPair],
    ref_ligand: SimpleResidue,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    keys = find_pocket_keys(reference, ref_ligand, radius)
    ref_to_pred = _reference_to_prediction(chain_pairs)
    positions_pred: List[np.ndarray] = []
    positions_ref: List[np.ndarray] = []
    for key in keys:
        if key not in ref_to_pred:
            continue
        _, _, pred_res, ref_res = ref_to_pred[key]
        pred_ca = _ca_position(pred_res)
        ref_ca = _ca_position(ref_res)
        if pred_ca is None or ref_ca is None:
            continue
        positions_pred.append(pred_ca)
        positions_ref.append(ref_ca)
    if len(positions_pred) < 3:
        return np.eye(3), np.zeros(3), 0
    pred_matrix = np.stack(positions_pred, axis=0)
    ref_matrix = np.stack(positions_ref, axis=0)
    rotation, translation = kabsch(pred_matrix, ref_matrix)
    return rotation, translation, len(positions_pred)
