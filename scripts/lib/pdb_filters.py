from __future__ import annotations

import re
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import gemmi
import yaml

from .pdb_download import (
    fetch_assembly_json,
    fetch_chemcomp_json,
    fetch_entry_json,
    fetch_nonpolymer_entity_json,
    fetch_polymer_entity_json,
)


@dataclass(frozen=True)
class FilterConfig:
    res_max: float = 3.0
    tier: str = "cluster"  # "laptop" or "cluster"
    size_laptop: int = 350
    size_cluster: int = 900
    exclude_gpcr: bool = True
    whitelist: Set[str] = frozenset()
    blocklist: Set[str] = frozenset()

    @property
    def size_max(self) -> int:
        return self.size_laptop if self.tier == "laptop" else self.size_cluster


class ChemCompCache:
    def __init__(self) -> None:
        self.type_by_comp: Dict[str, Optional[str]] = {}
        self.is_lipid_like_by_comp: Dict[str, Optional[bool]] = {}
        self._lock = threading.Lock()

    def get_type(self, comp_id: str) -> Optional[str]:
        comp_id = comp_id.upper()
        with self._lock:
            if comp_id in self.type_by_comp:
                return self.type_by_comp[comp_id]
        try:
            data = fetch_chemcomp_json(comp_id)
            typ = None
            rcsb = data.get("rcsb_chem_comp") or {}
            typ = rcsb.get("type") or (data.get("chem_comp") or {}).get("type")
        except Exception:
            typ = None
        with self._lock:
            self.type_by_comp[comp_id] = typ
        return typ

    def is_lipid_like(self, comp_id: str) -> Optional[bool]:
        comp_id = comp_id.upper()
        with self._lock:
            if comp_id in self.is_lipid_like_by_comp:
                return self.is_lipid_like_by_comp[comp_id]
        try:
            data = fetch_chemcomp_json(comp_id)
        except Exception:
            with self._lock:
                self.is_lipid_like_by_comp[comp_id] = None
            return None
        chem = (data.get("chem_comp") or {})
        name = (chem.get("name") or "").lower()
        desc = (data.get("rcsb_chem_comp_descriptor") or {})
        smiles = (desc.get("smilesstereo") or desc.get("smiles") or "")
        info = (data.get("rcsb_chem_comp_info") or {})
        heavy = info.get("atom_count_heavy") or 0

        name_hits = any(
            kw in name
            for kw in [
                "fatty acid",
                "cholesterol",
                "phosphatidyl",
                "phosphatidic",
                "diacylglycerol",
                "triacylglycerol",
                "ceramide",
                "sphingolipid",
                "cardiolipin",
                "lysophosphatidyl",
                "phosphocholine",
                "phosphoethanolamine",
            ]
        )
        # crude long-chain carbon pattern in SMILES
        long_chain = False
        if smiles:
            long_chain = bool(re.search(r"C{8,}", smiles))

        is_lipid_like = bool(name_hits or (heavy and heavy >= 16 and long_chain))
        with self._lock:
            self.is_lipid_like_by_comp[comp_id] = is_lipid_like
        return is_lipid_like


def load_id_list(path: Optional[Path]) -> Set[str]:
    if not path:
        return set()
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text()) or {}
    items = data.get("allowed") or data.get("excluded") or []
    return {str(x).strip().upper() for x in items}


def is_gpcr_entry(entry_json: dict) -> bool:
    # Heuristic: keywords or title mentions GPCR/7TM OR entities have PFAM names typical for GPCRs.
    title = (entry_json.get("struct") or {}).get("title", "").upper()
    if any(k in title for k in ("GPCR", "G PROTEIN-COUPLED RECEPTOR", "SEVEN TRANSMEMBRANE", "7TM")):
        return True
    kws = entry_json.get("struct_keywords") or {}
    text = (kws.get("text") or "").upper()
    if any(k in text for k in ("GPCR", "G PROTEIN-COUPLED RECEPTOR", "SEVEN TRANSMEMBRANE", "7TM")):
        return True
    # Polymer entity PFAM names check
    ent_ids = (entry_json.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
    for eid in ent_ids:
        try:
            p = fetch_polymer_entity_json(entry_json["rcsb_id"], eid)
        except Exception:
            continue
        cont = p.get("rcsb_polymer_entity_container_identifiers") or {}
        pfam_names = [str(x).upper() for x in (cont.get("pfam_names") or [])]
        if any("7TM" in n or "GPCR" in n for n in pfam_names):
            return True
    return False


def is_protein_only(entry_json: dict) -> bool:
    # Ensure all polymer entities are protein (exclude DNA/RNA)
    ent_ids = (entry_json.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
    for eid in ent_ids:
        try:
            p = fetch_polymer_entity_json(entry_json["rcsb_id"], eid)
        except Exception:
            return False
        poly = p.get("entity_poly") or {}
        typ = (poly.get("type") or "").upper()
        if "DNA" in typ or "RNA" in typ or "NUCLEIC" in typ:
            return False
    return True


def load_assembly_mmcif(pdbid: str, assembly_id: str, cif_path: Path) -> gemmi.Structure:
    from .pdb_download import download_mmcif_assembly

    # Reuse existing CIF if present to avoid redundant downloads on reruns
    if not cif_path.exists():
        download_mmcif_assembly(pdbid, assembly_id, cif_path)
    doc = gemmi.cif.read_file(str(cif_path))
    st = gemmi.make_structure_from_block(doc.sole_block())
    return st


def count_polymer_chains(st: gemmi.Structure) -> int:
    c = 0
    for model in st:
        for chain in model:
            if chain.get_polymer():
                c += 1
        break  # only first model
    return c


def chain_polymer_length(st: gemmi.Structure) -> int:
    # Returns residue count for the single polymer chain (assumes monomer)
    for model in st:
        for chain in model:
            pol = chain.get_polymer()
            if pol:
                return len(pol)
        break
    return 0


def collect_nonpolymer_comp_ids(st: gemmi.Structure) -> List[str]:
    comps: list[str] = []
    for model in st:
        for chain in model:
            pol = chain.get_polymer()
            pol_keys = set()
            if pol:
                for r in pol:
                    sid = r.seqid
                    pol_keys.add((chain.name, sid.num, sid.icode))
            for res in chain:
                sid = res.seqid
                key = (chain.name, sid.num, sid.icode)
                if key not in pol_keys:
                    comps.append(res.name.upper())
        break
    return comps


@dataclass
class EntryDecision:
    accept: bool
    reason: str
    details: dict


def evaluate_entry(
    pdbid: str,
    out_cif: Path,
    cfg: FilterConfig,
    chem_cache: Optional[ChemCompCache] = None,
) -> EntryDecision:
    chem_cache = chem_cache or ChemCompCache()
    # Fetch entry and preferred assembly
    entry = fetch_entry_json(pdbid)
    entry_id = entry.get("rcsb_id", pdbid)

    method = ", ".join(sorted(set([m.get("method") for m in (entry.get("exptl") or []) if m.get("method")])))
    res = None
    try:
        res_list = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        res = min(res_list) if res_list else None
    except Exception:
        res = None

    if cfg.exclude_gpcr and is_gpcr_entry(entry):
        return EntryDecision(False, "excluded_gpcr", {"method": method, "resolution": res})

    if not is_protein_only(entry):
        return EntryDecision(False, "not_protein_only", {"method": method, "resolution": res})

    if res is not None and res > cfg.res_max:
        return EntryDecision(False, "resolution_too_high", {"method": method, "resolution": res})

    # We will decide lipid ligand presence after reading the assembly via residue composition.

    preferred = (entry.get("rcsb_entry_info") or {}).get("preferred_assembly_id") or "1"
    # Prefilter: assembly must be monomeric per metadata
    try:
        asm_json = fetch_assembly_json(entry_id, str(preferred))
        pdbx_asm = asm_json.get("pdbx_struct_assembly") or {}
        oli_cnt = pdbx_asm.get("oligomeric_count")
        if isinstance(oli_cnt, str):
            try:
                oli_cnt = int(oli_cnt)
            except Exception:
                oli_cnt = None
        if oli_cnt != 1:
            return EntryDecision(False, "not_monomer_meta", {"oligomeric_count": oli_cnt, "method": method, "resolution": res})
    except Exception:
        # if assembly metadata missing, fall back to structure check
        pass

    # If any nonpolymer entity is on the detergent blocklist, reject early (cheap)
    try:
        np_ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("nonpolymer_entity_ids") or []
        for np_id in np_ids:
            np = fetch_nonpolymer_entity_json(entry_id, np_id)
            chem = (np.get("chem_comp") or {})
            comp_id = (chem.get("id") or "").upper()
            if comp_id in cfg.blocklist:
                return EntryDecision(False, "detergent_blocklisted", {"blocklisted": comp_id, "method": method, "resolution": res})
    except Exception:
        pass

    # Load assembly mmCIF
    tmp_cif = out_cif.with_suffix(".tmp.cif")
    st = load_assembly_mmcif(entry_id, str(preferred), tmp_cif)

    # Monomer check
    n_poly_chains = count_polymer_chains(st)
    if n_poly_chains != 1:
        try:
            tmp_cif.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return EntryDecision(False, "not_monomer", {"polymer_chains": n_poly_chains, "method": method, "resolution": res})

    # Size check
    length = chain_polymer_length(st)
    if length == 0 or length > cfg.size_max:
        try:
            tmp_cif.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return EntryDecision(False, "size_out_of_bounds", {"length": length, "method": method, "resolution": res})

    # Nonpolymer components (from assembled structure)
    comps = collect_nonpolymer_comp_ids(st)
    comp_set = {c for c in comps}
    # Classify each comp_id
    lipid_instances = 0
    lipid_comp_ids: Set[str] = set()
    nonlipid_nonwhitelist: Set[str] = set()
    blocklist_hit: Optional[str] = None
    for comp in comps:
        comp_u = comp.upper()
        if comp_u in cfg.blocklist:
            blocklist_hit = comp_u
            break
        lipid_like = chem_cache.is_lipid_like(comp_u)
        if lipid_like:
            lipid_instances += 1
            lipid_comp_ids.add(comp_u)
        else:
            if comp_u not in cfg.whitelist:
                nonlipid_nonwhitelist.add(comp_u)

    if blocklist_hit is not None:
        # Cleanup tmp
        try:
            tmp_cif.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return EntryDecision(False, "detergent_blocklisted", {
            "blocklisted": blocklist_hit,
            "method": method,
            "resolution": res,
        })

    # Must be exactly one lipid instance in the assembled structure
    if lipid_instances != 1:
        try:
            tmp_cif.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return EntryDecision(False, "lipid_instance_count!=1", {
            "lipid_instances": lipid_instances,
            "lipid_comps": sorted(lipid_comp_ids),
            "other_nonpolymers": sorted(comp_set - lipid_comp_ids),
            "method": method,
            "resolution": res,
        })

    # Non-lipid residues must be either empty or on whitelist
    if nonlipid_nonwhitelist:
        try:
            tmp_cif.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return EntryDecision(False, "nonlipid_not_whitelisted", {
            "nonwhitelisted": sorted(nonlipid_nonwhitelist),
            "method": method,
            "resolution": res,
        })

    # If we made it here, accept. Keep the downloaded assembly CIF.
    # Determine a rough category label for review
    category = categorize_entry(entry)
    details = {
        "method": method,
        "resolution": res,
        "assembly_id": preferred,
        "polymer_length": length,
        "lipid_comp_id": next(iter(lipid_comp_ids)) if lipid_comp_ids else None,
        "nonlipid_whitelisted": sorted(comp_set - lipid_comp_ids),
        "category": category,
        "title": (entry.get("struct") or {}).get("title"),
    }
    # Move tmp to final path
    try:
        if out_cif.exists():
            out_cif.unlink()
        tmp_cif.replace(out_cif)
    except Exception:
        pass
    return EntryDecision(True, "accepted", details)


def categorize_entry(entry_json: dict) -> str:
    # Lightweight heuristic; best effort
    title = (entry_json.get("struct") or {}).get("title", "").lower()
    kw = ((entry_json.get("struct_keywords") or {}).get("text") or "").lower()
    # Enzyme signal
    if re.search(r"\bEC\s*\d", title) or "enzyme" in kw:
        return "enzyme"
    # Nuclear receptor / steroid-like keywords
    if any(k in title for k in ["nuclear receptor", "estrogen receptor", "androgen receptor", "ppar", "rxr", "l xr", "gr ", "glucocorticoid receptor", "ar "]):
        return "intracellular_receptor"
    # Transporter hints
    if "transport" in kw or "carrier" in kw or "binding protein" in title and "fatty acid" in title:
        return "lipid_transport"
    # Membrane receptor hints
    if "receptor" in title and "membrane" in kw:
        return "membrane_receptor_extracellular"
    if "membrane" in kw or "membrane" in title:
        return "membrane_pocket_binder"
    return "uncertain"
