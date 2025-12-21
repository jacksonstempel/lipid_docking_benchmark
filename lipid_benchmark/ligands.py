from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .constants import MIN_LIGAND_HEAVY_ATOMS

from rdkit import Chem
try:  # pragma: no cover
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover
    pass
from rdkit.Chem import rdFMCS, rdchem
from rdkit.Geometry import Point3D
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


def load_ligand_template_names(
    project_root: Path,
    pdbid: str,
    *,
    include_h: bool = False,
    prefer_pdbqt: bool | None = None,
) -> List[str] | None:
    """Return atom names from the docking prep template (if present).

    Prefer the PDBQT used for docking if available (to preserve the atom order
    that Vina will output), falling back to the PDB copy otherwise.
    """
    prep_dir = project_root / "docking" / "prep" / pdbid
    pdbqt_path = prep_dir / "ligand.pdbqt"
    pdb_path = prep_dir / "ligand.pdb"
    if prefer_pdbqt is True:
        template_path = pdbqt_path if pdbqt_path.is_file() else pdb_path
    elif prefer_pdbqt is False:
        template_path = pdb_path if pdb_path.is_file() else pdbqt_path
    else:
        # Default preference: PDB (unique atom names), then PDBQT
        template_path = pdb_path if pdb_path.is_file() else pdbqt_path
    if not template_path.is_file():
        return None
    try:
        structure = load_structure(template_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
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


def find_ligand_by_id(structure: "gemmi.Structure", ligand_id: str) -> SimpleResidue:
    """Find a ligand by the "<chain>:<resname>:<resid>" identifier."""
    ligands = collect_ligands(structure, include_h=False)
    for lig in ligands:
        if f"{lig.chain_id}:{lig.res_name}:{lig.res_id}" == ligand_id:
            return lig
    raise RuntimeError(f"Ligand not found for ligand_id={ligand_id}")


def filter_large_ligands(residues: Sequence[SimpleResidue]) -> List[SimpleResidue]:
    return [residue for residue in residues if residue.heavy_atom_count() >= MIN_LIGAND_HEAVY_ATOMS]


def pairs_by_name(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int, int]]:
    """Match atoms by label with an element-consistency check.

    Only pairs heavy atoms (excludes H) and requires that the elements for the
    matched labels are identical between prediction and reference.
    """
    lookup_pred = {
        atom.name.upper(): (index, atom.element.upper())
        for index, atom in enumerate(pred.atoms)
        if atom.element != "H"
    }
    lookup_ref = {
        atom.name.upper(): (index, atom.element.upper())
        for index, atom in enumerate(ref.atoms)
        if atom.element != "H"
    }
    pairs: List[Tuple[int, int]] = []
    for name in sorted(set(lookup_pred) & set(lookup_ref)):
        ip, ep = lookup_pred[name]
        ir, er = lookup_ref[name]
        if ep == er:
            pairs.append((ip, ir))
    return pairs


# Simplistic covalent radii (Å) for bond inference
_COV_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}

# Common element symbols seen in this benchmark. Treat unknown/blank symbols as carbon
# to avoid noisy RDKit errors on malformed inputs while still producing a usable
# molecule for MCS-based pairing.
_ELEMENT_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}


def _element_radius(sym: str) -> float:
    return _COV_RADII.get(sym.upper(), 0.77)


def _build_rdkit_mol_from_residue(residue: SimpleResidue) -> tuple["Chem.Mol" | None, List[int]]:
    """Construct an RDKit molecule from a residue via proximity-based bonding.

    Returns (mol, rd_to_res_index)."""
    atoms = residue.atoms
    if not atoms:
        return None, []

    # Build atoms
    rw = rdchem.RWMol()
    rd_to_res: List[int] = []
    for idx, a in enumerate(atoms):
        if a.element == "H":
            continue
        elem = (a.element or "").strip().upper()
        z = _ELEMENT_TO_Z.get(elem, 6)
        atom = rdchem.Atom(int(z))
        rd_idx = rw.AddAtom(atom)
        rd_to_res.append(idx)

    # Coordinates
    # Infer bonds from distances with a tolerance
    coords = np.array([atoms[i].xyz for i in rd_to_res], float)
    n = len(rd_to_res)
    for i in range(n):
        ei = atoms[rd_to_res[i]].element
        ri = _element_radius(ei)
        for j in range(i + 1, n):
            ej = atoms[rd_to_res[j]].element
            rj = _element_radius(ej)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            # Allow a generous 0.5 Å cushion (typical single bonds ~1.5 Å) to tolerate
            # crystallographic noise without fusing nonbonded atoms (>3 Å apart).
            if d <= (ri + rj + 0.5):
                try:
                    rw.AddBond(i, j, rdchem.BondType.SINGLE)
                except (RuntimeError, ValueError, TypeError):
                    pass

    mol = rw.GetMol()
    conf = rdchem.Conformer(mol.GetNumAtoms())
    for rd_idx, res_idx in enumerate(rd_to_res):
        xyz = atoms[res_idx].xyz
        conf.SetAtomPosition(rd_idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    # Avoid verbose RDKit warnings: catch errors quietly and still return a usable molecule.
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except (RuntimeError, ValueError):
        pass
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except (RuntimeError, ValueError):
        pass
    return mol, rd_to_res


def pairs_by_rdkit(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int, int]]:
    """Map atoms using RDKit MCS (element-level) and return index pairs.

    Falls back to an empty list if mapping fails.
    """
    mol_p, map_p = _build_rdkit_mol_from_residue(pred)
    mol_r, map_r = _build_rdkit_mol_from_residue(ref)
    if mol_p is None or mol_r is None or mol_p.GetNumAtoms() == 0 or mol_r.GetNumAtoms() == 0:
        return []

    try:
        mcs = rdFMCS.FindMCS(
            [mol_p, mol_r],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=5,
        )
        if not mcs or not mcs.smartsString:
            return []
        query = Chem.MolFromSmarts(mcs.smartsString)
        if query is None:
            return []
        match_p = mol_p.GetSubstructMatches(query)
        match_r = mol_r.GetSubstructMatches(query)
        if not match_p or not match_r:
            return []
        # Take first match (maximum size by MCS definition)
        a_p = list(match_p[0])
        a_r = list(match_r[0])
        if len(a_p) != len(a_r) or len(a_p) < 3:
            return []
        # Translate RDKit indices back to residue atom indices
        pairs: List[Tuple[int, int]] = []
        for ip, ir in zip(a_p, a_r):
            pairs.append((map_p[ip], map_r[ir]))
        return pairs
    except (RuntimeError, ValueError):
        return []

def headgroup_indices_functional(residue: SimpleResidue) -> List[int]:
    """Return ligand atom indices (in residue.atoms) considered part of the headgroup.

    Heuristic (functional-group oriented):
      1) If any phosphorus atoms exist: take atoms within 2 bonds of P.
      2) Else if any nitrogen with degree >= 3: take atoms within 2 bonds of N.
      3) Else if any carbon with >=2 oxygen neighbors: take that carbon + its O neighbors.
      4) Else fall back to hetero atoms (O/N/P/S).
    """
    try:
        mol, rd_to_res = _build_rdkit_mol_from_residue(residue)
    except (RuntimeError, ValueError):
        return []
    if mol is None or mol.GetNumAtoms() == 0:
        return []

    def _expand(seed_idxs: List[int], depth: int) -> set[int]:
        seen = set(seed_idxs)
        frontier = set(seed_idxs)
        for _ in range(depth):
            next_frontier = set()
            for idx in frontier:
                atom = mol.GetAtomWithIdx(idx)
                for neigh in atom.GetNeighbors():
                    nid = neigh.GetIdx()
                    if nid not in seen:
                        seen.add(nid)
                        next_frontier.add(nid)
            frontier = next_frontier
        return seen

    p_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    if p_atoms:
        head = _expand(p_atoms, 2)
        return sorted({rd_to_res[i] for i in head if i < len(rd_to_res)})

    n_atoms = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and a.GetDegree() >= 3
    ]
    if n_atoms:
        head = _expand(n_atoms, 2)
        return sorted({rd_to_res[i] for i in head if i < len(rd_to_res)})

    c_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        o_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if len(o_neighbors) >= 2:
            c_atoms.append(atom.GetIdx())
    if c_atoms:
        head = set()
        for idx in c_atoms:
            head.add(idx)
            atom = mol.GetAtomWithIdx(idx)
            for neigh in atom.GetNeighbors():
                if neigh.GetSymbol() == "O":
                    head.add(neigh.GetIdx())
        return sorted({rd_to_res[i] for i in head if i < len(rd_to_res)})

    hetero = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in {"O", "N", "P", "S"}
    ]
    return sorted({rd_to_res[i] for i in hetero if i < len(rd_to_res)})


def locked_rmsd(
    pred_atoms: Sequence[SimpleAtom],
    ref_atoms: Sequence[SimpleAtom],
    pairs: Sequence[Tuple[int, int]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, int]:
    # RMSD can be computed for any non-empty set of paired atoms. A ≥3 constraint is
    # only needed when you want to *fit* a rotation/translation from the pairs.
    if len(pairs) < 1:
        return float("inf"), 0

    P = np.array([pred_atoms[i].xyz for i, _ in pairs], dtype=float)
    Q = np.array([ref_atoms[j].xyz for _, j in pairs], dtype=float)
    P = P @ rotation + translation
    rmsd = float(np.sqrt(((P - Q) ** 2).sum() / P.shape[0]))
    return rmsd, P.shape[0]
