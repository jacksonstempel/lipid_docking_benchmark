from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import List, Sequence


ALL_RESULTS_COLUMNS = [
    "run_timestamp",
    "source_label",
    "record_type",
    "pdbid",
    "pose_index",
    "protein_metric_index",
    "protein_metric",
    "protein_value",
    "protein_pairs_pruned",
    "protein_rmsd_ca_pruned",
    "protein_pairs_all",
    "protein_rmsd_ca_all_under_pruned",
    "protein_rmsd_ca_allfit",
    "pred_chain",
    "pred_resname",
    "pred_resid",
    "ref_chain",
    "ref_resname",
    "ref_resid",
    "policy",
    "atom_pairs",
    "rmsd_locked_global",
    "rmsd_locked_pocket",
    "pocket_pairs",
]

SUMMARY_COLUMNS = [
    "run_timestamp",
    "source_label",
    "record_type",
    "metric_name",
    "metric_value",
    "pdbid",
    "best_pose_index",
    "best_pred_chain",
    "best_pred_resname",
    "best_pred_resid",
    "best_ref_chain",
    "best_ref_resname",
    "best_ref_resid",
    "best_atom_pairs",
    "best_locked_global_rmsd",
    "best_locked_pocket_rmsd",
    "protein_rmsd_ca_pruned",
    "protein_pairs_pruned",
    "evaluated_pose_count",
    "ligand_matched_count",
]


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_all_results(detail_rows: Sequence[dict], raw_data_path: Path, source_label: str, run_timestamp: str) -> None:
    if not detail_rows:
        return
    _ensure_parent(raw_data_path)
    rows = []
    for row in detail_rows:
        output = {column: "" for column in ALL_RESULTS_COLUMNS}
        output["run_timestamp"] = run_timestamp
        output["source_label"] = source_label
        for key in (
            "record_type",
            "pdbid",
            "pose_index",
            "protein_metric_index",
            "protein_metric",
            "protein_value",
            "protein_pairs_pruned",
            "protein_rmsd_ca_pruned",
            "protein_pairs_all",
            "protein_rmsd_ca_all_under_pruned",
            "protein_rmsd_ca_allfit",
            "pred_chain",
            "pred_resname",
            "pred_resid",
            "ref_chain",
            "ref_resname",
            "ref_resid",
            "policy",
            "atom_pairs",
            "rmsd_locked_global",
            "rmsd_locked_pocket",
            "pocket_pairs",
        ):
            if key in row:
                output[key] = _format_value(row.get(key))
        rows.append(output)

    write_header = not raw_data_path.exists()
    with raw_data_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _build_protein_summary_row(pdbid: str, summary: dict, source_label: str, run_timestamp: str) -> dict:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "run_timestamp": run_timestamp,
            "source_label": source_label,
            "record_type": "protein_summary",
            "pdbid": pdbid,
            "protein_rmsd_ca_pruned": _format_value(summary.get("protein_rmsd_ca_pruned")),
            "protein_pairs_pruned": _format_value(summary.get("protein_pairs_pruned")),
            "evaluated_pose_count": _format_value(summary.get("evaluated_pose_count")),
            "ligand_matched_count": _format_value(summary.get("ligand_matched_count")),
        }
    )
    best_pose = summary.get("best_pose")
    if isinstance(best_pose, dict):
        row["best_pose_index"] = _format_value(best_pose.get("pose_index"))
        row["best_pred_chain"] = best_pose.get("pred", {}).get("chain", "")
        row["best_pred_resname"] = best_pose.get("pred", {}).get("name", "")
        row["best_pred_resid"] = best_pose.get("pred", {}).get("id", "")
        row["best_ref_chain"] = best_pose.get("ref", {}).get("chain", "")
        row["best_ref_resname"] = best_pose.get("ref", {}).get("name", "")
        row["best_ref_resid"] = best_pose.get("ref", {}).get("id", "")
        row["best_atom_pairs"] = _format_value(best_pose.get("n"))
        row["best_locked_global_rmsd"] = _format_value(best_pose.get("rmsd_locked_global"))
        row["best_locked_pocket_rmsd"] = _format_value(best_pose.get("rmsd_locked_pocket"))
        row["best_locked_pocket_rmsd"] = _format_value(best_pose.get("rmsd_locked_pocket"))
        row["best_atom_pairs"] = _format_value(best_pose.get("n"))
    if "best_locked_pocket_rmsd" not in row:
        row["best_locked_pocket_rmsd"] = _format_value(summary.get("best_pose", {}).get("rmsd_locked_pocket"))
    if "best_atom_pairs" not in row:
        row["best_atom_pairs"] = _format_value(summary.get("best_pose", {}).get("n"))
    return row


def _aggregate_metric_rows(protein_rows: Sequence[dict], source_label: str, run_timestamp: str) -> List[dict]:
    locked_globals = [
        float(row["best_locked_global_rmsd"])
        for row in protein_rows
        if row["best_locked_global_rmsd"]
    ]
    protein_rmsds = [
        float(row["protein_rmsd_ca_pruned"])
        for row in protein_rows
        if row["protein_rmsd_ca_pruned"]
    ]
    ligand_matches = [
        float(row["ligand_matched_count"])
        for row in protein_rows
        if row["ligand_matched_count"]
    ]

    metrics = []
    if locked_globals:
        metrics.append(("mean_best_locked_global_rmsd", mean(locked_globals)))
        metrics.append(("median_best_locked_global_rmsd", median(locked_globals)))
    if protein_rmsds:
        metrics.append(("mean_protein_rmsd_ca_pruned", mean(protein_rmsds)))
        metrics.append(("median_protein_rmsd_ca_pruned", median(protein_rmsds)))
    metrics.append(("protein_count", len(protein_rows)))
    metrics.append(("successful_protein_count", sum(1 for row in protein_rows if row["best_locked_global_rmsd"])))
    if ligand_matches:
        metrics.append(("total_ligand_matches", int(sum(ligand_matches))))

    aggregate_rows: List[dict] = []
    for name, value in metrics:
        agg = {column: "" for column in SUMMARY_COLUMNS}
        agg.update(
            {
                "run_timestamp": run_timestamp,
                "source_label": source_label,
                "record_type": "aggregate_metric",
                "metric_name": name,
                "metric_value": _format_value(value),
            }
        )
        aggregate_rows.append(agg)
    return aggregate_rows


def write_summary_file(summary_rows: Sequence[dict], summary_path: Path) -> None:
    _ensure_parent(summary_path)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)


def resolve_summary_directory(base: Path, label: str) -> Path:
    if base.name.lower() == label.lower():
        return base
    return base / label


def build_and_write_summary(
    per_protein_summaries: Sequence[tuple[str, dict]],
    source_label: str,
    summary_path: Path,
    run_timestamp: str,
) -> None:
    protein_rows = [
        _build_protein_summary_row(pdbid, summary, source_label, run_timestamp)
        for pdbid, summary in per_protein_summaries
    ]
    aggregate_rows = _aggregate_metric_rows(protein_rows, source_label, run_timestamp)
    write_summary_file([*protein_rows, *aggregate_rows], summary_path)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_source_label(candidates: Sequence[Path | str], default: str = "unspecified") -> str:
    for candidate in candidates:
        text = str(candidate).lower()
        if "boltz" in text:
            return "boltz"
        if "vina" in text:
            return "vina"
        if "moe" in text:
            return "moe"
    return default
