from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

from .alignment import extract_chain_sequences, pair_chains
from .constants import AA3_TO_1


def _pdb_chain_id(chain_name: str) -> str:
    chain_name = (chain_name or "").strip()
    return chain_name[:1] if chain_name else ""


def build_residue_id_map(pred_structure, ref_structure) -> Dict[str, str]:
    pred_chains = extract_chain_sequences(pred_structure)
    ref_chains = extract_chain_sequences(ref_structure)
    chain_pairs = pair_chains(pred_chains, ref_chains)
    if not chain_pairs:
        raise RuntimeError("Unable to pair protein chains between prediction and reference.")

    mapping: Dict[str, str] = {}
    for pair in chain_pairs:
        pred_chain = _pdb_chain_id(pair.pred.chain.name)
        ref_chain = _pdb_chain_id(pair.ref.chain.name)
        for pred_idx, ref_idx in pair.res_pairs:
            pred_res = pair.pred.res_objs[pred_idx]
            ref_res = pair.ref.res_objs[ref_idx]
            pred_id = f"{pred_chain}:{pred_res.name}:{pred_res.seqid.num}"
            ref_id = f"{ref_chain}:{ref_res.name}:{ref_res.seqid.num}"
            mapping[pred_id] = ref_id
    if not mapping:
        raise RuntimeError("Residue mapping is empty after chain pairing.")
    return mapping


@dataclass(frozen=True)
class ResidueMapQc:
    chain_pairs: int
    aligned_pairs: int
    identity: float
    pred_residues: int
    ref_residues: int
    coverage_pred: float
    coverage_ref: float


def build_residue_id_map_with_qc(pred_structure, ref_structure) -> Tuple[Dict[str, str], ResidueMapQc]:
    pred_chains = extract_chain_sequences(pred_structure)
    ref_chains = extract_chain_sequences(ref_structure)
    chain_pairs = pair_chains(pred_chains, ref_chains)
    if not chain_pairs:
        raise RuntimeError("Unable to pair protein chains between prediction and reference.")

    pred_total = sum(len(c.res_objs) for c in pred_chains)
    ref_total = sum(len(c.res_objs) for c in ref_chains)

    mapping: Dict[str, str] = {}
    matches = 0
    aligned = 0
    for pair in chain_pairs:
        pred_chain = _pdb_chain_id(pair.pred.chain.name)
        ref_chain = _pdb_chain_id(pair.ref.chain.name)
        for pred_idx, ref_idx in pair.res_pairs:
            pred_res = pair.pred.res_objs[pred_idx]
            ref_res = pair.ref.res_objs[ref_idx]
            pred_aa = AA3_TO_1.get(pred_res.name.upper(), "X")
            ref_aa = AA3_TO_1.get(ref_res.name.upper(), "X")
            if pred_aa == ref_aa:
                matches += 1
            aligned += 1
            pred_id = f"{pred_chain}:{pred_res.name}:{pred_res.seqid.num}"
            ref_id = f"{ref_chain}:{ref_res.name}:{ref_res.seqid.num}"
            mapping[pred_id] = ref_id
    if not mapping:
        raise RuntimeError("Residue mapping is empty after chain pairing.")

    identity = (matches / aligned) if aligned else 0.0
    cov_pred = (aligned / pred_total) if pred_total else 0.0
    cov_ref = (aligned / ref_total) if ref_total else 0.0
    qc = ResidueMapQc(
        chain_pairs=len(chain_pairs),
        aligned_pairs=aligned,
        identity=identity,
        pred_residues=pred_total,
        ref_residues=ref_total,
        coverage_pred=cov_pred,
        coverage_ref=cov_ref,
    )
    return mapping, qc


def _parse_residue_id(res_id: str) -> Tuple[str, str, int] | None:
    parts = (res_id or "").split(":")
    if len(parts) != 3:
        return None
    chain, name, num = parts
    try:
        return chain, name, int(num)
    except ValueError:
        return None


def remap_residue_ids(residue_ids: Iterable[str], mapping: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for rid in residue_ids:
        parsed = _parse_residue_id(rid)
        if not parsed:
            out.add(rid)
            continue
        chain, name, num = parsed
        key = f"{chain}:{name}:{num}"
        out.add(mapping.get(key, key))
    return out


def remap_typed_ids(typed_ids: Iterable[str], mapping: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for item in typed_ids:
        if "|" not in item:
            out.add(item)
            continue
        residue, ctype = item.split("|", 1)
        mapped = mapping.get(residue, residue)
        out.add(f"{mapped}|{ctype}")
    return out
