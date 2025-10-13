from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .paths import PathResolver

FIELD_MAP: Dict[str, List[str]] = {
    "protein_rmsd": [
        "protein_rmsd",
        "protein_rmsd_ca_pruned",
        "protein_rmsd_ca_all_under_pruned",
    ],
    "protein_rmsd_ca_allfit": [
        "protein_rmsd_ca_allfit",
        "protein_ca_rmsd_allfit",
    ],
    "ligand": [
        "ref_resname",
        "pred_resname",
        "ligand",
        "ligand_name",
        "resname",
    ],
    "policy": [
        "policy",
        "pairing_policy",
    ],
    "rmsd_locked_global": [
        "rmsd_locked_global",
        "ligand_rmsd_locked_global",
        "ligand_rmsd_global",
    ],
    "rmsd_locked_pocket": [
        "rmsd_locked_pocket",
        "ligand_rmsd_locked_pocket",
        "ligand_rmsd_pocket",
    ],
    "n_residues": [
        "protein_pairs_pruned",
        "protein_pairs_all",
        "n_residues",
        "residues_total",
    ],
    "n_ligand_atoms": [
        "n",
        "n_ligand_atoms",
        "ligand_atom_count",
        "n_atoms_ligand",
    ],
    "n_pocket_residues": [
        "pocket_pairs",
        "n_pocket_residues",
        "pocket_residue_count",
        "pocket_residues",
    ],
}


def _first_nonempty(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def select_best_ligand_row(rows: List[Dict[str, str]]) -> Dict[str, str] | None:
    ligands = [r for r in rows if r.get("type", "").lower() == "ligand"]
    best_row: Dict[str, str] | None = None
    best_metric = float("inf")
    for row in ligands:
        candidate = _first_nonempty(row, FIELD_MAP["rmsd_locked_global"])
        try:
            score = float(candidate)
        except Exception:
            score = float("inf")
        if score < best_metric:
            best_metric = score
            best_row = row
    return best_row


def collect_analysis_csvs(
    resolver: PathResolver,
    ids: Iterable[str],
) -> Tuple[List[Path], List[Tuple[str, Path]]]:
    found: List[Path] = []
    missing: List[Tuple[str, Path]] = []
    for pdbid in ids:
        csv_path = resolver.analysis_paths_for(pdbid).analysis_csv
        if csv_path.is_file():
            found.append(csv_path)
        else:
            missing.append((pdbid, csv_path))
    return found, missing


def write_condensed_csv(
    per_target_csvs: Iterable[Path], destination: Path
) -> Tuple[int, List[Tuple[Path, Exception]]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "pdbid",
        "protein_rmsd",
        "protein_rmsd_ca_allfit",
        "ligand",
        "policy",
        "rmsd_locked_global",
        "rmsd_locked_pocket",
        "n_residues",
        "n_ligand_atoms",
        "n_pocket_residues",
    ]
    rows_written = 0
    errors: List[Tuple[Path, Exception]] = []
    with destination.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for csv_path in per_target_csvs:
            try:
                with csv_path.open("r", newline="") as fin:
                    reader = csv.DictReader(fin)
                    grouped: Dict[str, List[Dict[str, str]]] = {}
                    for row in reader:
                        pid = (
                            row.get("pdbid")
                            or row.get("PDBID")
                            or row.get("id")
                            or ""
                        ).strip()
                        if not pid:
                            continue
                        grouped.setdefault(pid, []).append(row)
                for pid, rows in sorted(grouped.items()):
                    protein = next(
                        (r for r in rows if r.get("type", "").lower() == "protein"),
                        None,
                    )
                    ligand = select_best_ligand_row(rows)
                    out_row: Dict[str, str] = {"pdbid": pid}
                    if protein:
                        out_row["protein_rmsd"] = _first_nonempty(
                            protein, FIELD_MAP["protein_rmsd"]
                        )
                        out_row["protein_rmsd_ca_allfit"] = _first_nonempty(
                            protein, FIELD_MAP["protein_rmsd_ca_allfit"]
                        )
                        out_row["n_residues"] = _first_nonempty(
                            protein, FIELD_MAP["n_residues"]
                        )
                    else:
                        out_row["protein_rmsd"] = ""
                        out_row["protein_rmsd_ca_allfit"] = ""
                        out_row["n_residues"] = ""
                    if ligand:
                        out_row["ligand"] = _first_nonempty(
                            ligand, FIELD_MAP["ligand"]
                        )
                        out_row["policy"] = _first_nonempty(
                            ligand, FIELD_MAP["policy"]
                        )
                        out_row["rmsd_locked_global"] = _first_nonempty(
                            ligand, FIELD_MAP["rmsd_locked_global"]
                        )
                        out_row["rmsd_locked_pocket"] = _first_nonempty(
                            ligand, FIELD_MAP["rmsd_locked_pocket"]
                        )
                        out_row["n_ligand_atoms"] = _first_nonempty(
                            ligand, FIELD_MAP["n_ligand_atoms"]
                        )
                        out_row["n_pocket_residues"] = _first_nonempty(
                            ligand, FIELD_MAP["n_pocket_residues"]
                        )
                    else:
                        out_row["ligand"] = ""
                        out_row["policy"] = ""
                        out_row["rmsd_locked_global"] = ""
                        out_row["rmsd_locked_pocket"] = ""
                        out_row["n_ligand_atoms"] = ""
                        out_row["n_pocket_residues"] = ""
                    writer.writerow(out_row)
                    rows_written += 1
            except Exception as exc:
                errors.append((csv_path, exc))
                continue
    return rows_written, errors
