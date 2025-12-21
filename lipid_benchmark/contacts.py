from __future__ import annotations

import contextlib
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
from scipy.spatial import cKDTree

from .normalization import NORMALIZED_LIGAND_RESNAME
from .rmsd import HEADGROUP_ELEMS
from .structures import is_protein_res, load_structure

HEADGROUP_INTERACTION_TYPES = {"hydrogen_bonds", "ionic", "salt_bridge", "attractive_charge"}
_CONTACT_FIELDS = ["ligand_atom", "ligand_element", "residue", "contact_type", "distance"]

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


def extract_contacts(struct_path: Path, *, ligand_resname: str) -> List[Dict[str, object]]:
    from pandamap import HybridProtLigMapper

    mapper = HybridProtLigMapper(str(struct_path), ligand_resname=ligand_resname)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        mapper.detect_interactions()

    contacts: List[Dict[str, object]] = []
    for ctype, entries in mapper.interactions.items():
        ctype_str = str(ctype)
        for entry in entries:
            lig_atom = entry.get("ligand_atom")
            prot_res = entry.get("protein_residue")
            dist = entry.get("distance") or entry.get("dist")
            chain_id = ""
            resname = ""
            resnum = ""
            try:
                chain_id = prot_res.get_parent().id if prot_res else ""
                resname = prot_res.resname if prot_res else ""
                resnum = prot_res.id[1] if prot_res else ""
            except (AttributeError, TypeError):
                pass

            lig_name = ""
            lig_elem = ""
            if hasattr(lig_atom, "get_id"):
                lig_name = lig_atom.get_id()
            elif lig_atom is not None:
                lig_name = str(lig_atom)

            if hasattr(lig_atom, "element"):
                lig_elem = str(lig_atom.element or "").strip()
            if not lig_elem and hasattr(lig_atom, "get_element"):
                lig_elem = str(lig_atom.get_element() or "").strip()
            if not lig_elem and lig_name:
                lig_elem = "".join(ch for ch in lig_name if ch.isalpha())[:2]
            lig_elem = lig_elem.upper()

            contacts.append(
                {
                    "ligand_atom": lig_name,
                    "ligand_element": lig_elem,
                    "residue": f"{chain_id}:{resname}:{resnum}",
                    "contact_type": ctype_str,
                    "distance": float(dist) if dist is not None else "",
                }
            )
    return contacts


def filter_headgroup_contacts(
    contacts: Sequence[Dict[str, object]],
    *,
    allowed_atoms: Set[str] | None = None,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for contact in contacts:
        ctype = str(contact.get("contact_type") or "")
        if ctype not in HEADGROUP_INTERACTION_TYPES:
            continue
        if allowed_atoms is not None:
            lig_name = str(contact.get("ligand_atom") or "").strip()
            if lig_name and lig_name in allowed_atoms:
                out.append(contact)
            continue
        elem = str(contact.get("ligand_element") or "").upper()
        if elem in HEADGROUP_ELEMS:
            out.append(contact)
    return out


def contacts_to_typed_set(contacts: Sequence[Dict[str, object]], *, residue_key: str = "residue") -> Set[str]:
    out: Set[str] = set()
    for c in contacts:
        resid = str(c.get(residue_key) or "")
        ctype = str(c.get("contact_type") or "")
        if resid and ctype:
            out.add(f"{resid}|{ctype}")
    return out


def interaction_type_counts(contacts: Sequence[Dict[str, object]]) -> str:
    counter: Counter[str] = Counter()
    for c in contacts:
        ctype = str(c.get("contact_type") or "")
        if ctype:
            counter[ctype] += 1
    if not counter:
        return "none=0"
    parts = [f"{k}={counter[k]}" for k in sorted(counter)]
    return ";".join(parts)


def load_contacts_csv(path: Path) -> List[Dict[str, object]]:
    import csv

    contacts: List[Dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "distance" in row and row["distance"] not in ("", None):
                try:
                    row["distance"] = float(row["distance"])
                except (ValueError, TypeError):
                    pass
            contacts.append(row)
    return contacts


def write_contacts_csv(path: Path, contacts: Sequence[Dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CONTACT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(contacts)


def cached_contacts(
    struct_path: Path,
    cache_path: Path,
    *,
    ligand_resname: str,
    use_cache: bool,
) -> List[Dict[str, object]]:
    if use_cache and cache_path.exists():
        try:
            if cache_path.stat().st_mtime >= struct_path.stat().st_mtime:
                return load_contacts_csv(cache_path)
        except OSError:
            pass
    contacts = extract_contacts(struct_path, ligand_resname=ligand_resname)
    write_contacts_csv(cache_path, contacts)
    return contacts
