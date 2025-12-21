from __future__ import annotations

from pathlib import Path
from typing import List

import gemmi
import numpy as np

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    # Common modified residues seen in PDB files (often marked HETATM but part of the polymer)
    "MSE": "M",  # selenomethionine
    "CSO": "C",  # cysteine sulfinic acid
    "OCS": "C",  # cysteine sulfonic acid
}

WATER_RES_NAMES = {"HOH", "H2O", "WAT"}


def is_protein_res(residue: gemmi.Residue) -> bool:
    """Return True when the residue belongs to the standard protein alphabet."""
    return residue.name.upper() in AA3_TO_1 and residue.name.upper() not in WATER_RES_NAMES


def is_water_res(residue: gemmi.Residue) -> bool:
    """Return True for water molecules (HOH/H2O/WAT)."""
    return residue.name.upper() in WATER_RES_NAMES


def load_structure(path: Path) -> gemmi.Structure:
    """Load a structure allowing CIF/PDB/PDBQT inputs."""
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent", ".pdbqt"}:
        structure = gemmi.read_pdb(str(path))
    else:
        structure = gemmi.read_structure(str(path))
    structure.remove_empty_chains()
    return structure


def apply_rt_to_structure(structure: gemmi.Structure, rotation: np.ndarray, translation: np.ndarray) -> None:
    """Apply a rotation/translation in place."""
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    vector = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    rotated = vector @ rotation + translation
                    atom.pos = gemmi.Position(float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _clone_residue(src: gemmi.Residue) -> gemmi.Residue:
    out = gemmi.Residue()
    out.name = src.name
    out.seqid = src.seqid
    out.het_flag = src.het_flag
    for atom in src:
        out.add_atom(atom.clone())
    return out


def _clone_model_subset(
    model: gemmi.Model, *, keep_proteins: bool, keep_non_proteins: bool, fallback_chain: str = "L"
) -> gemmi.Model:
    new_model = gemmi.Model(model.num)
    for chain in model:
        new_chain = gemmi.Chain(chain.name or fallback_chain)
        for residue in chain:
            as_protein = is_protein_res(residue)
            if (as_protein and keep_proteins) or (not as_protein and keep_non_proteins):
                new_chain.add_residue(_clone_residue(residue))
        if len(new_chain):
            new_model.add_chain(new_chain)
    return new_model


def structure_has_protein(structure: gemmi.Structure) -> bool:
    return any(is_protein_res(residue) for model in structure for chain in model for residue in chain)


def ensure_protein_backbone(prediction: gemmi.Structure, reference: gemmi.Structure) -> gemmi.Structure:
    """Guarantee a protein backbone is available by grafting from the reference when needed."""
    if structure_has_protein(prediction):
        return prediction

    combined = gemmi.Structure()
    combined.cell = reference.cell
    combined.spacegroup_hm = reference.spacegroup_hm

    proteins = _clone_model_subset(reference[0], keep_proteins=True, keep_non_proteins=False)
    ligands = _clone_model_subset(prediction[0], keep_proteins=False, keep_non_proteins=True)

    if len(proteins):
        combined.add_model(proteins)
    if len(ligands):
        if len(combined):
            for chain in ligands:
                combined[0].add_chain(chain)
        else:
            combined.add_model(ligands)

    combined.remove_empty_chains()
    return combined


def split_models(structure: gemmi.Structure, count: int) -> List[gemmi.Structure]:
    """Return up to `count` models, each as an isolated structure."""
    total = len(structure)
    if total == 0:
        return []

    limit = max(1, min(count, total))
    out: List[gemmi.Structure] = []
    for index in range(limit):
        model = structure[index]
        new_structure = gemmi.Structure()
        new_structure.cell = structure.cell
        new_structure.spacegroup_hm = structure.spacegroup_hm
        new_model = gemmi.Model(index + 1)
        for chain in model:
            new_chain = gemmi.Chain(chain.name)
            for residue in chain:
                new_chain.add_residue(_clone_residue(residue))
            if len(new_chain):
                new_model.add_chain(new_chain)
        new_structure.add_model(new_model)
        new_structure.remove_empty_chains()
        out.append(new_structure)
    return out


def write_pdb_structure(structure: gemmi.Structure, destination: Path) -> None:
    """Write a multi-model PDB file."""
    lines: List[str] = []
    serial = 1
    for model_index, model in enumerate(structure, start=1):
        lines.append(f"MODEL     {model_index}")
        for chain in model:
            for residue in chain:
                resname = residue.name
                chain_name = chain.name[:1] if chain.name else "A"
                seqid = residue.seqid.num if hasattr(residue.seqid, "num") else int(str(residue.seqid).strip() or 0)
                icode = residue.seqid.icode if hasattr(residue.seqid, "icode") else " "
                record = "ATOM  " if is_protein_res(residue) else "HETATM"
                for atom in residue:
                    atom_name = atom.name.strip()
                    padded = atom_name.rjust(4) if len(atom.element.name.strip()) == 1 and len(atom_name) < 4 else atom_name.ljust(4)
                    x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                    element = atom.element.name.strip().rjust(2)
                    lines.append(
                        f"{record}{serial:5d} {padded} {resname:>3s} {chain_name:1s}{seqid:4d}{icode:1s}   "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
                    )
                    serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    destination.write_text("\n".join(lines) + "\n")
