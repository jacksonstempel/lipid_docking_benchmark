#!/usr/bin/env python3
"""
Generate adversarial Boltz input YAMLs by mutating headgroup-contact residues.

This is a data-prep step for the "binding-site mutagenesis" experiment:
  - Identify residues whose *side-chain heavy atoms* are within a cutoff of any
    lipid *headgroup heavy atom* in the experimental reference structure.
  - Mutate those residues to Gly (G) or Phe (F) in the Boltz input sequence.
  - Keep the ligand definition unchanged.

Outputs are written under output/ (derived artifacts).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import gemmi
import numpy as np
from Bio.Align import PairwiseAligner
from scipy.spatial import cKDTree


# Backbone heavy atoms to exclude when selecting "side-chain heavy atoms".
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pairs_pdbids(pairs_csv: Path) -> List[str]:
    pdbids: List[str] = []
    with pairs_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdbid = (row.get("pdbid") or "").strip().upper()
            if pdbid:
                pdbids.append(pdbid)
    if not pdbids:
        raise RuntimeError(f"No pdbids found in pairs CSV: {pairs_csv}")
    return pdbids


def _read_structure_pdb(path: Path) -> gemmi.Structure:
    if not path.exists():
        raise FileNotFoundError(path)
    return gemmi.read_pdb(str(path))


def _is_protein_res(res: gemmi.Residue) -> bool:
    # Keep this local to avoid importing the package in early bootstraps.
    from lipid_benchmark.structures import AA3_TO_1, WATER_RES_NAMES

    name = res.name.upper()
    return name in AA3_TO_1 and name not in WATER_RES_NAMES


def _aa3_to_1(resname: str) -> str:
    from lipid_benchmark.structures import AA3_TO_1

    return AA3_TO_1.get(resname.upper(), "X")


def _protein_sequence_and_residue_ids(structure: gemmi.Structure) -> Tuple[str, List[str]]:
    """Return (sequence, residue_ids) for the first protein chain in model 0."""
    model = structure[0]
    for chain in model:
        residues = [r for r in chain if _is_protein_res(r)]
        if not residues:
            continue
        seq = "".join(_aa3_to_1(r.name) for r in residues)
        chain_id = (chain.name or "").strip()[:1] or "A"
        res_ids = [f"{chain_id}:{r.name}:{r.seqid.num}" for r in residues]
        return seq, res_ids
    raise RuntimeError("No protein chain found in structure.")


def _headgroup_coords(structure: gemmi.Structure, headgroup_atom_names: Sequence[str]) -> np.ndarray:
    """Return (N,3) array of headgroup heavy-atom coordinates in normalized LIG residue."""
    wanted = {str(n).strip().upper() for n in headgroup_atom_names if str(n).strip()}
    if not wanted:
        raise RuntimeError("Empty headgroup atom-name list.")

    model = structure[0]
    coords: List[List[float]] = []
    for chain in model:
        for residue in chain:
            if residue.het_flag != "H":
                continue
            if residue.name.strip().upper() != "LIG":
                continue
            for atom in residue:
                if atom.element.name.upper() == "H":
                    continue
                if atom.name.strip().upper() in wanted:
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not coords:
        raise RuntimeError(f"Could not locate any headgroup atoms in LIG for names={sorted(wanted)[:5]}...")
    return np.array(coords, dtype=float)


def _binding_site_residue_indices(
    structure: gemmi.Structure,
    *,
    headgroup_atom_names: Sequence[str],
    cutoff_a: float,
) -> List[int]:
    """Return indices (0-based in the protein sequence) of residues contacting the headgroup."""
    head_xyz = _headgroup_coords(structure, headgroup_atom_names)
    head_tree = cKDTree(head_xyz)

    model = structure[0]
    # Collect side-chain heavy atoms and map them to residue indices.
    prot_atom_xyz: List[List[float]] = []
    prot_atom_res_idx: List[int] = []

    # Assume single-chain proteins in this benchmark; take the first protein chain.
    protein_chain = None
    for chain in model:
        if any(_is_protein_res(r) for r in chain):
            protein_chain = chain
            break
    if protein_chain is None:
        raise RuntimeError("No protein chain found for binding-site detection.")

    residues = [r for r in protein_chain if _is_protein_res(r)]
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.element.name.upper() == "H":
                continue
            if atom.name.strip().upper() in _BACKBONE_ATOMS:
                continue
            prot_atom_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
            prot_atom_res_idx.append(i)

    if not prot_atom_xyz:
        return []

    dists, _ = head_tree.query(np.array(prot_atom_xyz, dtype=float), k=1)
    contacting = {ri for ri, d in zip(prot_atom_res_idx, dists) if float(d) <= float(cutoff_a)}
    return sorted(contacting)


def _align_index_map(ref_seq: str, query_seq: str) -> Dict[int, int]:
    """Map indices from ref_seq -> query_seq using a global alignment."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    aln = aligner.align(ref_seq, query_seq)[0]
    aligned_ref, aligned_query = aln.aligned

    mapping: Dict[int, int] = {}
    for (rs, re), (qs, qe) in zip(aligned_ref, aligned_query):
        for i, j in zip(range(rs, re), range(qs, qe)):
            mapping[i] = j
    return mapping


def _mutate_sequence(seq: str, indices: Iterable[int], new_aa: str) -> Tuple[str, int]:
    """Return (mutated_seq, n_actual_changes)."""
    s = list(seq)
    changed = 0
    for idx in set(int(i) for i in indices):
        if idx < 0 or idx >= len(s):
            continue
        if s[idx] != new_aa:
            changed += 1
        s[idx] = new_aa
    return "".join(s), changed


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        default="",
        help="Pairs CSV (default: config.yaml paths.pairs, else structures/benchmark_entries.csv).",
    )
    p.add_argument("--cutoff-a", type=float, default=5.0, help="Distance cutoff in angstroms (default: 5.0).")
    p.add_argument(
        "--normalized-dir",
        default=".cache/lipid_benchmark/normalized",
        help="Normalized cache dir containing <PDBID>/ref.pdb + audit.json (default: .cache/lipid_benchmark/normalized).",
    )
    p.add_argument(
        "--inputs-dir",
        default="prediction_inputs/boltz_inputs",
        help="Directory containing baseline Boltz input YAMLs (default: prediction_inputs/boltz_inputs).",
    )
    p.add_argument(
        "--out-root",
        default="output/adversarial/bs_mutagenesis_cutoff5A",
        help="Output root directory (default: output/adversarial/bs_mutagenesis_cutoff5A).",
    )
    args = p.parse_args(argv)

    root = _project_root()

    # Resolve pairs CSV via repo config when not provided.
    if str(args.pairs).strip():
        pairs_csv = Path(str(args.pairs)).expanduser()
        pairs_csv = (root / pairs_csv).resolve() if not pairs_csv.is_absolute() else pairs_csv.resolve()
    else:
        from lipid_benchmark.io import default_pairs_path

        pairs_csv = default_pairs_path(root)

    pdbids = _load_pairs_pdbids(pairs_csv)

    normalized_dir = (root / Path(str(args.normalized_dir))).resolve()
    inputs_dir = (root / Path(str(args.inputs_dir))).resolve()
    out_root = (root / Path(str(args.out_root))).resolve()
    out_gly = out_root / "boltz_inputs_gly"
    out_phe = out_root / "boltz_inputs_phe"
    out_root.mkdir(parents=True, exist_ok=True)
    out_gly.mkdir(parents=True, exist_ok=True)
    out_phe.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "mutation_summary.csv"
    fieldnames = [
        "pdbid",
        "cutoff_a",
        "targeted_residues",
        "targeted_count",
        "mapped_count",
        "unmapped_count",
        "actual_changes_gly",
        "actual_changes_phe",
    ]

    import yaml  # local dependency already used elsewhere in the repo

    processed = 0
    with summary_path.open("w", newline="") as fsum:
        # Use real newlines (not a literal backslash-n) so tools like pandas read this normally.
        writer = csv.DictWriter(fsum, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for pdbid in pdbids:
            base_yaml = inputs_dir / f"{pdbid}.yaml"
            if not base_yaml.exists():
                raise FileNotFoundError(f"Missing Boltz input YAML for {pdbid}: {base_yaml}")

            # Reference structure + headgroup names from the normalized cache.
            entry_dir = normalized_dir / pdbid
            ref_pdb = entry_dir / "ref.pdb"
            audit_json = entry_dir / "audit.json"
            if not ref_pdb.exists() or not audit_json.exists():
                raise FileNotFoundError(f"Missing normalized cache for {pdbid}: {ref_pdb} / {audit_json}")
            audit = json.loads(audit_json.read_text())
            head_atoms = audit.get("ref_headgroup_atoms") or []

            ref_struct = _read_structure_pdb(ref_pdb)
            ref_seq, ref_res_ids = _protein_sequence_and_residue_ids(ref_struct)

            contact_ref_indices = _binding_site_residue_indices(ref_struct, headgroup_atom_names=head_atoms, cutoff_a=float(args.cutoff_a))
            targeted_res_ids = [ref_res_ids[i] for i in contact_ref_indices if 0 <= i < len(ref_res_ids)]

            # Load baseline Boltz YAML and enforce single-chain protein id.
            base = yaml.safe_load(base_yaml.read_text())
            if not isinstance(base, dict) or "sequences" not in base:
                raise RuntimeError(f"{pdbid}: unexpected YAML structure in {base_yaml}")
            seqs = base.get("sequences") or []
            if not isinstance(seqs, list) or len(seqs) < 2:
                raise RuntimeError(f"{pdbid}: expected sequences=[protein, ligand] in {base_yaml}")
            prot = (seqs[0] or {}).get("protein")
            if not isinstance(prot, dict):
                raise RuntimeError(f"{pdbid}: expected protein entry in {base_yaml}")
            prot_id = prot.get("id")
            if isinstance(prot_id, list):
                raise RuntimeError(f"{pdbid}: multi-chain protein id not allowed (id={prot_id})")
            prot_seq = str(prot.get("sequence") or "")
            if not prot_seq:
                raise RuntimeError(f"{pdbid}: missing protein.sequence in {base_yaml}")

            # Map reference residue indices -> Boltz sequence indices.
            ref_to_yaml = _align_index_map(ref_seq, prot_seq)
            yaml_indices = [ref_to_yaml[i] for i in contact_ref_indices if i in ref_to_yaml]
            unmapped = [i for i in contact_ref_indices if i not in ref_to_yaml]

            gly_seq, gly_changed = _mutate_sequence(prot_seq, yaml_indices, "G")
            phe_seq, phe_changed = _mutate_sequence(prot_seq, yaml_indices, "F")

            gly_yaml = json.loads(json.dumps(base))  # cheap deep copy for simple YAML dicts
            phe_yaml = json.loads(json.dumps(base))
            gly_yaml["sequences"][0]["protein"]["sequence"] = gly_seq
            phe_yaml["sequences"][0]["protein"]["sequence"] = phe_seq

            # Write mutated YAMLs.
            (out_gly / f"{pdbid}.yaml").write_text(
                yaml.safe_dump(gly_yaml, sort_keys=False),
                encoding="utf-8",
            )
            (out_phe / f"{pdbid}.yaml").write_text(
                yaml.safe_dump(phe_yaml, sort_keys=False),
                encoding="utf-8",
            )

            writer.writerow(
                {
                    "pdbid": pdbid,
                    "cutoff_a": float(args.cutoff_a),
                    "targeted_residues": ";".join(targeted_res_ids),
                    "targeted_count": len(targeted_res_ids),
                    "mapped_count": len(set(yaml_indices)),
                    "unmapped_count": len(unmapped),
                    "actual_changes_gly": gly_changed,
                    "actual_changes_phe": phe_changed,
                }
            )
            processed += 1

    print(f"Wrote {processed} mutant inputs to: {out_root}")
    print(f"- Gly: {out_gly}")
    print(f"- Phe: {out_phe}")
    print(f"- Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
