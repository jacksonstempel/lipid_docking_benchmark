from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gemmi

from .rmsd import LigandSelectionError, _filter_ligands, _select_single_ligand
from .ligands import SimpleResidue, collect_ligands, find_ligand_by_id, headgroup_indices_functional, pairs_by_rdkit
from .structures import is_protein_res, load_structure, split_models, write_pdb_structure


NORMALIZED_LIGAND_RESNAME = "LIG"
HEADGROUP_SELECTION_VERSION = "functional_refmap_v1"


@dataclass(frozen=True)
class NormalizedFiles:
    pdbid: str
    ref_pdb: Path
    boltz_pdb: Path
    vina_pdbs: List[Path]
    audit_json: Path
    ref_headgroup_atoms: List[str]
    boltz_headgroup_atoms: List[str]
    vina_headgroup_atoms: List[List[str]]


def _assign_atom_names(ligand: SimpleResidue) -> List[str]:
    counts: Counter[str] = Counter()
    names: List[str] = []
    for atom in ligand.atoms:
        elem = (atom.element or "C").strip().upper()
        counts[elem] += 1
        names.append(f"{elem}{counts[elem]}")
    return names


def _map_ref_indices_to_pred(
    pdbid: str,
    *,
    ref_ligand: SimpleResidue,
    pred_ligand: SimpleResidue,
    ref_indices: List[int],
    label: str,
) -> List[int]:
    pairs = pairs_by_rdkit(pred_ligand, ref_ligand)
    if not pairs:
        raise RuntimeError(f"{pdbid}: RDKit atom mapping failed for {label} ligand.")
    ref_to_pred = {ref_idx: pred_idx for pred_idx, ref_idx in pairs}
    missing = [ref_idx for ref_idx in ref_indices if ref_idx not in ref_to_pred]
    if missing:
        raise RuntimeError(f"{pdbid}: missing {len(missing)} mapped headgroup atoms for {label} ligand.")
    return [ref_to_pred[ref_idx] for ref_idx in ref_indices]


def _clone_residue(residue: gemmi.Residue) -> gemmi.Residue:
    clone = gemmi.Residue()
    clone.name = residue.name
    clone.seqid = residue.seqid
    clone.het_flag = residue.het_flag
    for atom in residue:
        clone.add_atom(atom.clone())
    return clone


def _build_ligand_residue(
    ligand: SimpleResidue,
    *,
    resname: str,
    atom_names: List[str] | None = None,
) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = resname
    residue.seqid = gemmi.SeqId("1")
    residue.het_flag = "H"
    if atom_names is None:
        atom_names = _assign_atom_names(ligand)
    for atom, atom_name in zip(ligand.atoms, atom_names):
        elem = (atom.element or "C").strip().upper()
        gatom = gemmi.Atom()
        gatom.name = atom_name
        try:
            gatom.element = gemmi.Element(elem)
        except Exception:
            gatom.element = gemmi.Element("C")
        gatom.pos = gemmi.Position(float(atom.xyz[0]), float(atom.xyz[1]), float(atom.xyz[2]))
        residue.add_atom(gatom)
    return residue


def _pick_ligand_chain_id(model: gemmi.Model) -> str:
    used = {chain.name[:1] for chain in model if chain.name}
    for candidate in ("L", "Z", "X", "Y"):
        if candidate not in used:
            return candidate
    return "L"


def _build_complex(
    protein_structure: gemmi.Structure,
    ligand: SimpleResidue,
    *,
    ligand_resname: str = NORMALIZED_LIGAND_RESNAME,
    ligand_atom_names: List[str] | None = None,
) -> gemmi.Structure:
    new = gemmi.Structure()
    new.cell = protein_structure.cell
    new.spacegroup_hm = protein_structure.spacegroup_hm
    model = gemmi.Model(1)

    # Add protein residues only.
    for chain in protein_structure[0]:
        new_chain = gemmi.Chain(chain.name or "A")
        for residue in chain:
            if is_protein_res(residue):
                new_chain.add_residue(_clone_residue(residue))
        if len(new_chain):
            model.add_chain(new_chain)

    # Add the selected ligand as its own chain.
    lig_clone = _build_ligand_residue(ligand, resname=ligand_resname, atom_names=ligand_atom_names)
    lig_chain_id = _pick_ligand_chain_id(model)
    lig_chain = gemmi.Chain(lig_chain_id)
    lig_chain.add_residue(lig_clone)
    model.add_chain(lig_chain)

    new.add_model(model)
    new.remove_empty_chains()
    return new


def _select_ligand_or_raise(structure: gemmi.Structure, *, pdbid: str) -> SimpleResidue:
    ligands = collect_ligands(structure, include_h=False)
    filtered = _filter_ligands(ligands)
    if not filtered:
        raise LigandSelectionError(f"{pdbid}: no ligand candidates after filtering.")
    if len(filtered) == 1:
        return filtered[0]
    # Still ambiguous: fall back to the existing heuristic (largest ligand).
    return _select_single_ligand(structure, include_h=False)


def normalize_entry(
    pdbid: str,
    ref_path: Path,
    boltz_path: Path,
    vina_path: Path,
    *,
    out_dir: Path,
    vina_max_poses: int = 3,
    use_cache: bool = True,
) -> NormalizedFiles:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_dir = out_dir / pdbid
    entry_dir.mkdir(parents=True, exist_ok=True)

    ref_structure = load_structure(ref_path)
    boltz_structure = load_structure(boltz_path)
    vina_structure = load_structure(vina_path)

    ref_ligand = _select_ligand_or_raise(ref_structure, pdbid=pdbid)
    boltz_ligand = _select_ligand_or_raise(boltz_structure, pdbid=pdbid)

    vina_models = split_models(vina_structure, vina_max_poses)
    if not vina_models:
        raise LigandSelectionError(f"{pdbid}: Vina file has no poses/models.")

    vina_ligand_ids: List[str] = []
    for model in vina_models:
        lig = _select_ligand_or_raise(model, pdbid=pdbid)
        vina_ligand_ids.append(f"{lig.chain_id}:{lig.res_name}:{lig.res_id}")

    return normalize_entry_from_selected(
        pdbid,
        ref_structure,
        boltz_structure,
        vina_models,
        ref_ligand=ref_ligand,
        boltz_ligand_id=f"{boltz_ligand.chain_id}:{boltz_ligand.res_name}:{boltz_ligand.res_id}",
        vina_ligand_ids=vina_ligand_ids,
        out_dir=out_dir,
        use_cache=use_cache,
    )


def normalize_entry_from_selected(
    pdbid: str,
    ref_structure: gemmi.Structure,
    boltz_structure: gemmi.Structure,
    vina_models: List[gemmi.Structure],
    *,
    ref_ligand: SimpleResidue,
    boltz_ligand_id: str,
    vina_ligand_ids: List[str],
    out_dir: Path,
    use_cache: bool = True,
) -> NormalizedFiles:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry_dir = out_dir / pdbid
    entry_dir.mkdir(parents=True, exist_ok=True)

    if len(vina_models) != len(vina_ligand_ids):
        raise RuntimeError(
            f"{pdbid}: vina pose count mismatch (models={len(vina_models)}, ligand_ids={len(vina_ligand_ids)})"
        )

    ref_pdb = entry_dir / "ref.pdb"
    boltz_pdb = entry_dir / "boltz.pdb"
    vina_pdbs = [entry_dir / f"vina_pose_{i}.pdb" for i in range(1, len(vina_ligand_ids) + 1)]
    audit_json = entry_dir / "audit.json"

    ref_atom_names = _assign_atom_names(ref_ligand)
    ref_head_indices = headgroup_indices_functional(ref_ligand)
    ref_head_atoms = [ref_atom_names[i] for i in ref_head_indices]

    boltz_ligand = find_ligand_by_id(boltz_structure, boltz_ligand_id)
    boltz_atom_names = _assign_atom_names(boltz_ligand)
    boltz_head_indices = _map_ref_indices_to_pred(
        pdbid,
        ref_ligand=ref_ligand,
        pred_ligand=boltz_ligand,
        ref_indices=ref_head_indices,
        label="boltz",
    )
    boltz_head_atoms = [boltz_atom_names[i] for i in boltz_head_indices]

    vina_ligands: List[SimpleResidue] = []
    vina_headgroup_atoms: List[List[str]] = []
    vina_atom_names: List[List[str]] = []
    for pose_struct, ligand_id in zip(vina_models, vina_ligand_ids):
        vina_ligand = find_ligand_by_id(pose_struct, ligand_id)
        names = _assign_atom_names(vina_ligand)
        head_indices = _map_ref_indices_to_pred(
            pdbid,
            ref_ligand=ref_ligand,
            pred_ligand=vina_ligand,
            ref_indices=ref_head_indices,
            label=f"vina pose {ligand_id}",
        )
        head_atoms = [names[i] for i in head_indices]
        vina_ligands.append(vina_ligand)
        vina_atom_names.append(names)
        vina_headgroup_atoms.append(head_atoms)

    expected = {
        "pdbid": pdbid,
        "normalized_ligand_resname": NORMALIZED_LIGAND_RESNAME,
        "headgroup_selection": HEADGROUP_SELECTION_VERSION,
        "ref_ligand_id": f"{ref_ligand.chain_id}:{ref_ligand.res_name}:{ref_ligand.res_id}",
        "boltz_ligand_id": boltz_ligand_id,
        "vina_ligand_ids": vina_ligand_ids,
        "ref_headgroup_atoms": ref_head_atoms,
        "boltz_headgroup_atoms": boltz_head_atoms,
        "vina_headgroup_atoms": vina_headgroup_atoms,
    }

    have_files = ref_pdb.exists() and boltz_pdb.exists() and all(p.exists() for p in vina_pdbs)
    if use_cache and have_files and audit_json.exists():
        try:
            audit = json.loads(audit_json.read_text())
        except (OSError, json.JSONDecodeError):
            audit = {}

        if audit == expected:
            return NormalizedFiles(
                pdbid=pdbid,
                ref_pdb=ref_pdb,
                boltz_pdb=boltz_pdb,
                vina_pdbs=vina_pdbs,
                audit_json=audit_json,
                ref_headgroup_atoms=ref_head_atoms,
                boltz_headgroup_atoms=boltz_head_atoms,
                vina_headgroup_atoms=vina_headgroup_atoms,
            )

        identity_keys = ("pdbid", "normalized_ligand_resname", "ref_ligand_id", "boltz_ligand_id", "vina_ligand_ids")
        if all(audit.get(k) == expected.get(k) for k in identity_keys):
            audit_json.write_text(json.dumps(expected, indent=2))
            return NormalizedFiles(
                pdbid=pdbid,
                ref_pdb=ref_pdb,
                boltz_pdb=boltz_pdb,
                vina_pdbs=vina_pdbs,
                audit_json=audit_json,
                ref_headgroup_atoms=ref_head_atoms,
                boltz_headgroup_atoms=boltz_head_atoms,
                vina_headgroup_atoms=vina_headgroup_atoms,
            )

    ref_complex = _build_complex(ref_structure, ref_ligand, ligand_atom_names=ref_atom_names)
    write_pdb_structure(ref_complex, ref_pdb)

    boltz_complex = _build_complex(boltz_structure, boltz_ligand, ligand_atom_names=boltz_atom_names)
    write_pdb_structure(boltz_complex, boltz_pdb)

    for pose_index, (vina_ligand, names) in enumerate(zip(vina_ligands, vina_atom_names), start=1):
        pose_complex = _build_complex(ref_structure, vina_ligand, ligand_atom_names=names)
        pose_pdb = vina_pdbs[pose_index - 1]
        write_pdb_structure(pose_complex, pose_pdb)

    audit_json.write_text(json.dumps(expected, indent=2))

    return NormalizedFiles(
        pdbid=pdbid,
        ref_pdb=ref_pdb,
        boltz_pdb=boltz_pdb,
        vina_pdbs=vina_pdbs,
        audit_json=audit_json,
        ref_headgroup_atoms=ref_head_atoms,
        boltz_headgroup_atoms=boltz_head_atoms,
        vina_headgroup_atoms=vina_headgroup_atoms,
    )
