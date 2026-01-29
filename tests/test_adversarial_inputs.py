import csv
import json
import unittest
from pathlib import Path

import gemmi
import numpy as np
import yaml
from Bio.Align import PairwiseAligner
from scipy.spatial import cKDTree


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _aa3_to_1(resname: str) -> str:
    from lipid_benchmark.structures import AA3_TO_1

    return AA3_TO_1.get(resname.upper(), "X")


def _is_protein_res(res: gemmi.Residue) -> bool:
    from lipid_benchmark.structures import AA3_TO_1, WATER_RES_NAMES

    name = res.name.upper()
    return name in AA3_TO_1 and name not in WATER_RES_NAMES


def _protein_chain(structure: gemmi.Structure) -> gemmi.Chain:
    model = structure[0]
    for chain in model:
        if any(_is_protein_res(r) for r in chain):
            return chain
    raise RuntimeError("No protein chain found")


def _protein_sequence_and_residue_ids(structure: gemmi.Structure) -> tuple[str, list[str]]:
    chain = _protein_chain(structure)
    residues = [r for r in chain if _is_protein_res(r)]
    seq = "".join(_aa3_to_1(r.name) for r in residues)
    chain_id = (chain.name or "").strip()[:1] or "A"
    res_ids = [f"{chain_id}:{r.name}:{r.seqid.num}" for r in residues]
    return seq, res_ids


def _headgroup_coords(structure: gemmi.Structure, head_atom_names: list[str]) -> np.ndarray:
    wanted = {str(n).strip().upper() for n in head_atom_names if str(n).strip()}
    if not wanted:
        raise RuntimeError("Empty headgroup atoms list")

    coords: list[list[float]] = []
    model = structure[0]
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
        raise RuntimeError("Could not locate any headgroup atoms in normalized LIG residue")
    return np.array(coords, dtype=float)


def _binding_site_residue_indices(
    structure: gemmi.Structure,
    *,
    head_atom_names: list[str],
    cutoff_a: float,
) -> list[int]:
    head_xyz = _headgroup_coords(structure, head_atom_names)
    head_tree = cKDTree(head_xyz)

    chain = _protein_chain(structure)
    residues = [r for r in chain if _is_protein_res(r)]

    prot_atom_xyz: list[list[float]] = []
    prot_atom_res_idx: list[int] = []
    for i, residue in enumerate(residues):
        for atom in residue:
            if atom.element.name.upper() == "H":
                continue
            if atom.name.strip().upper() in BACKBONE_ATOMS:
                continue
            prot_atom_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
            prot_atom_res_idx.append(i)

    if not prot_atom_xyz:
        return []

    dists, _ = head_tree.query(np.array(prot_atom_xyz, dtype=float), k=1)
    contacting = {ri for ri, d in zip(prot_atom_res_idx, dists) if float(d) <= float(cutoff_a)}
    return sorted(contacting)


def _align_index_map(ref_seq: str, query_seq: str) -> dict[int, int]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    aln = aligner.align(ref_seq, query_seq)[0]
    aligned_ref, aligned_query = aln.aligned

    mapping: dict[int, int] = {}
    for (rs, re), (qs, qe) in zip(aligned_ref, aligned_query):
        for i, j in zip(range(rs, re), range(qs, qe)):
            mapping[i] = j
    return mapping


class TestAdversarialMutantInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = _project_root()
        cls.out_root = cls.root / "output" / "adversarial" / "bs_mutagenesis_cutoff5A"
        cls.gly_dir = cls.out_root / "boltz_inputs_gly"
        cls.phe_dir = cls.out_root / "boltz_inputs_phe"
        cls.summary_csv = cls.out_root / "mutation_summary.csv"

        if not cls.gly_dir.is_dir() or not cls.phe_dir.is_dir() or not cls.summary_csv.is_file():
            raise unittest.SkipTest(
                "Adversarial mutant inputs not found. Run:\n"
                "  python scripts/adversarial_make_mutant_inputs.py --cutoff-a 5.0 "
                "--out-root output/adversarial/bs_mutagenesis_cutoff5A"
            )

        from lipid_benchmark.io import default_pairs_path

        cls.pairs_csv = default_pairs_path(cls.root)
        cls.normalized_dir = cls.root / ".cache" / "lipid_benchmark" / "normalized"
        cls.inputs_dir = cls.root / "prediction_inputs" / "boltz_inputs"

        # Load pdbids
        cls.pdbids: list[str] = []
        with cls.pairs_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdbid = (row.get("pdbid") or "").strip().upper()
                if pdbid:
                    cls.pdbids.append(pdbid)
        if not cls.pdbids:
            raise RuntimeError(f"No pdbids found in pairs CSV: {cls.pairs_csv}")

        # Load mutation summary table
        cls.summary_by_pdbid: dict[str, dict[str, str]] = {}
        with cls.summary_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdbid = (row.get("pdbid") or "").strip().upper()
                if pdbid:
                    cls.summary_by_pdbid[pdbid] = row

    def test_all_targets_present_and_single_chain(self):
        gly_stems = {p.stem.upper() for p in self.gly_dir.glob("*.yaml")}
        phe_stems = {p.stem.upper() for p in self.phe_dir.glob("*.yaml")}
        expected = set(self.pdbids)

        self.assertEqual(gly_stems, expected)
        self.assertEqual(phe_stems, expected)
        self.assertEqual(set(self.summary_by_pdbid), expected)

        # Baseline + mutant YAML sanity
        for pdbid in self.pdbids:
            base_path = self.inputs_dir / f"{pdbid}.yaml"
            gly_path = self.gly_dir / f"{pdbid}.yaml"
            phe_path = self.phe_dir / f"{pdbid}.yaml"

            self.assertTrue(base_path.is_file(), base_path)
            self.assertTrue(gly_path.is_file(), gly_path)
            self.assertTrue(phe_path.is_file(), phe_path)

            base = yaml.safe_load(base_path.read_text())
            gly = yaml.safe_load(gly_path.read_text())
            phe = yaml.safe_load(phe_path.read_text())

            for obj in (base, gly, phe):
                self.assertIsInstance(obj, dict)
                seqs = obj.get("sequences")
                self.assertIsInstance(seqs, list)
                self.assertEqual(len(seqs), 2)
                self.assertIn("protein", seqs[0])
                self.assertIn("ligand", seqs[1])
                prot = seqs[0]["protein"]
                self.assertIsInstance(prot.get("id"), str, f"{pdbid}: protein.id must be a string (single chain)")
                self.assertIsInstance(prot.get("sequence"), str)
                self.assertGreater(len(prot["sequence"]), 0)

                lig = seqs[1]["ligand"]
                self.assertIn("ccd", lig, f"{pdbid}: ligand must be CCD-based")

            # Mutant YAMLs must match baseline except for the protein sequence.
            base_copy = json.loads(json.dumps(base))
            gly_copy = json.loads(json.dumps(gly))
            phe_copy = json.loads(json.dumps(phe))

            base_seq = base_copy["sequences"][0]["protein"]["sequence"]
            gly_copy["sequences"][0]["protein"]["sequence"] = base_seq
            phe_copy["sequences"][0]["protein"]["sequence"] = base_seq

            self.assertEqual(gly_copy, base_copy, f"{pdbid}: gly YAML differs from baseline beyond protein.sequence")
            self.assertEqual(phe_copy, base_copy, f"{pdbid}: phe YAML differs from baseline beyond protein.sequence")

    def test_mutations_match_binding_site_definition(self):
        cutoff_a = 5.0

        for pdbid in self.pdbids:
            base_path = self.inputs_dir / f"{pdbid}.yaml"
            gly_path = self.gly_dir / f"{pdbid}.yaml"
            phe_path = self.phe_dir / f"{pdbid}.yaml"

            base = yaml.safe_load(base_path.read_text())
            gly = yaml.safe_load(gly_path.read_text())
            phe = yaml.safe_load(phe_path.read_text())

            base_seq = base["sequences"][0]["protein"]["sequence"]
            gly_seq = gly["sequences"][0]["protein"]["sequence"]
            phe_seq = phe["sequences"][0]["protein"]["sequence"]

            self.assertEqual(len(gly_seq), len(base_seq), f"{pdbid}: gly sequence length changed")
            self.assertEqual(len(phe_seq), len(base_seq), f"{pdbid}: phe sequence length changed")

            entry_dir = self.normalized_dir / pdbid
            ref_pdb = entry_dir / "ref.pdb"
            audit_json = entry_dir / "audit.json"
            self.assertTrue(ref_pdb.is_file(), ref_pdb)
            self.assertTrue(audit_json.is_file(), audit_json)

            audit = json.loads(audit_json.read_text())
            head_atoms = audit.get("ref_headgroup_atoms") or []
            self.assertTrue(head_atoms, f"{pdbid}: missing ref_headgroup_atoms in audit.json")

            ref_struct = gemmi.read_pdb(str(ref_pdb))
            ref_seq, ref_res_ids = _protein_sequence_and_residue_ids(ref_struct)

            contact_ref_indices = _binding_site_residue_indices(ref_struct, head_atom_names=head_atoms, cutoff_a=cutoff_a)
            self.assertGreater(len(contact_ref_indices), 0, f"{pdbid}: no binding-site residues found at {cutoff_a} A")

            # The summary CSV should reflect the same targeted residues.
            summary = self.summary_by_pdbid[pdbid]
            targeted_count = int(summary["targeted_count"])
            self.assertEqual(targeted_count, len(contact_ref_indices), f"{pdbid}: targeted_count mismatch")
            targeted_res_ids = [ref_res_ids[i] for i in contact_ref_indices]
            have_res_ids = set((summary["targeted_residues"] or "").split(";")) if summary.get("targeted_residues") else set()
            self.assertEqual(set(targeted_res_ids), have_res_ids, f"{pdbid}: targeted residue list mismatch")
            self.assertEqual(int(summary["unmapped_count"]), 0, f"{pdbid}: unexpected unmapped residues")

            # Map binding-site residue indices from ref -> baseline Boltz sequence indices.
            ref_to_yaml = _align_index_map(ref_seq, base_seq)
            yaml_indices = {ref_to_yaml[i] for i in contact_ref_indices}
            self.assertEqual(len(yaml_indices), int(summary["mapped_count"]), f"{pdbid}: mapped_count mismatch")

            # Mutations must be applied exactly at those indices, and nowhere else.
            for i in range(len(base_seq)):
                if i in yaml_indices:
                    self.assertEqual(gly_seq[i], "G", f"{pdbid}: gly not applied at index {i}")
                    self.assertEqual(phe_seq[i], "F", f"{pdbid}: phe not applied at index {i}")
                else:
                    self.assertEqual(gly_seq[i], base_seq[i], f"{pdbid}: gly changed non-target index {i}")
                    self.assertEqual(phe_seq[i], base_seq[i], f"{pdbid}: phe changed non-target index {i}")

            # Summary "actual changes" should match how many residues truly changed identity.
            exp_gly_changes = sum(1 for i in yaml_indices if base_seq[i] != "G")
            exp_phe_changes = sum(1 for i in yaml_indices if base_seq[i] != "F")
            self.assertEqual(int(summary["actual_changes_gly"]), exp_gly_changes, f"{pdbid}: actual_changes_gly mismatch")
            self.assertEqual(int(summary["actual_changes_phe"]), exp_phe_changes, f"{pdbid}: actual_changes_phe mismatch")


if __name__ == "__main__":
    unittest.main()

