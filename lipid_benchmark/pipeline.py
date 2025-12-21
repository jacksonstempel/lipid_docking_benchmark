"""
Benchmark “pipeline” orchestration (the main scientific workflow).

Plain-language overview

For each target (one row in the pairs CSV), we compute two kinds of evaluation:

1) Geometry (RMSD)
   - Align the protein backbone (for Boltz) and compute ligand RMSD.
   - Evaluate multiple Vina poses and keep “best” poses by RMSD.

2) Interactions (contacts)
   - Convert structures to a clean, consistent format (“normalized complexes”).
   - Use PandaMap to detect protein–ligand interactions and score overlap with the reference.
   - Focus scoring on lipid headgroup interactions (more biologically meaningful than tails).

Outputs are returned as rows suitable for CSV writing:
- `allposes`: one row per evaluated pose (Boltz + each Vina pose)
- `summary`: one row per target for Boltz and best Vina poses
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

from .contacts import (
    cached_contacts,
    contacts_to_typed_set,
    filter_headgroup_contacts,
    interaction_type_counts,
)
from .contacts import DEFAULT_HEAD_ENV_CUTOFF_A, headgroup_environment_residues
from .normalization import NORMALIZED_LIGAND_RESNAME, normalize_entry_from_selected
from .residue_mapping import ResidueMapQc, build_residue_id_map_with_qc, remap_residue_ids, remap_typed_ids
from .rmsd import _select_single_ligand, measure_ligand_pose_all
from .structures import load_structure, split_models
from .io import PairEntry

LOGGER = logging.getLogger("lipid_benchmark")

NA = "NA"

BENCHMARK_FIELDNAMES = [
    "pdbid",
    "method",
    "pose_index",
    "ref_ligand_id",
    "pred_ligand_id",
    "pairing_method",
    "ligand_heavy_atoms",
    "ligand_rmsd",
    "headgroup_atoms",
    "headgroup_rmsd",
    "protein_pairs",
    "protein_rmsd",
    "headgroup_contacts_ref",
    "headgroup_contacts_pred",
    "headgroup_types_ref",
    "headgroup_types_pred",
    "head_env_precision",
    "head_env_recall",
    "head_env_f1",
    "head_env_jaccard",
    "head_env_shared",
    "head_env_ref_size",
    "head_env_pred_size",
    "headgroup_typed_precision",
    "headgroup_typed_recall",
    "headgroup_typed_f1",
    "headgroup_typed_jaccard",
    "headgroup_typed_shared",
    "headgroup_typed_ref_size",
    "headgroup_typed_pred_size",
]


def _set_metrics_na_if_ref_empty(ref: Set[str], pred: Set[str], prefix: str) -> Dict[str, float | int | str]:
    if not ref:
        return {
            f"{prefix}_precision": NA,
            f"{prefix}_recall": NA,
            f"{prefix}_f1": NA,
            f"{prefix}_jaccard": NA,
            f"{prefix}_shared": 0,
            f"{prefix}_ref_size": 0,
            f"{prefix}_pred_size": len(pred),
        }

    shared = len(ref & pred)
    tp = shared
    fp = len(pred - ref)
    fn = len(ref - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    denom = len(ref | pred)
    jaccard = tp / denom if denom else 0.0
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_jaccard": jaccard,
        f"{prefix}_shared": shared,
        f"{prefix}_ref_size": len(ref),
        f"{prefix}_pred_size": len(pred),
    }


def _contact_cache_dir(normalized_dir: Path, pdbid: str) -> Path:
    """
    Return the cache folder used to store contact detection outputs for one target.

    Contacts are cached because PandaMap contact detection can be expensive, and cached
    results make re-runs much faster (and easier to reproduce).
    """
    return normalized_dir.parent / "contacts" / pdbid


def _safe_int(value) -> int | str:
    """
    Convert a value to an integer if possible; otherwise return an empty string.

    This exists so we can write “missing/unknown” values cleanly into CSV files.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def _safe_float(value) -> float | str:
    """
    Convert a value to a float if possible; otherwise return an empty string.

    This exists so we can write “missing/unknown” values cleanly into CSV files.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _finite_float_or_na(value) -> float | str:
    """
    Parse a numeric value and return `NA` when it is missing or not finite.

    This keeps CSV outputs consistent when upstream tools report "nan"/"inf".
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return NA
    return parsed if math.isfinite(parsed) else NA


def _require_all_ok(pdbid: str, rows: Sequence[Dict[str, object]], *, label: str) -> None:
    """
    Validate that a batch stage succeeded for all requested poses.

    Many stages return rows like `{"status": "ok"}` or `{"status": "error", "error": "..."}`.
    If any pose failed, we raise an error early with a readable message.
    """
    bad = [r for r in rows if r.get("status") != "ok"]
    if not bad:
        return
    msg = bad[0].get("error") or "unknown error"
    raise RuntimeError(f"{pdbid}: {label} failed for {len(bad)}/{len(rows)} poses (first error: {msg})")


def _run_entry(
    entry: PairEntry,
    *,
    vina_max_poses: int,
    normalized_dir: Path,
    cache_normalized: bool,
    cache_contacts: bool,
    stage_cb: Callable[[str, str], None] | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Run the full benchmark for one target (one PDB ID).

    This is the “workhorse” called by `run_benchmark()`. It returns:
    - `entry_all`: per-pose rows (Boltz pose + each Vina pose)
    - `entry_summary`: per-target summary rows (Boltz + best Vina poses)

    Optional callbacks (`stage_cb`) allow a UI (like the TUI) to display progress such as
    “RMSD: 1ABC boltz” or “Contacts: 1ABC vina pose 3”.
    """
    ref_structure = load_structure(entry.ref_path)
    ref_ligand = _select_single_ligand(ref_structure, include_h=False)
    ref_ligand_id = f"{ref_ligand.chain_id}:{ref_ligand.res_name}:{ref_ligand.res_id}"

    boltz_structure = load_structure(entry.boltz_path)
    boltz_models = split_models(boltz_structure, 1)
    if not boltz_models:
        raise RuntimeError(f"{entry.pdbid}: no models/poses found in Boltz prediction")
    boltz_pose = boltz_models[0]

    if stage_cb is not None:
        stage_cb("RMSD", f"{entry.pdbid} boltz")
    boltz_rmsd_rows = measure_ligand_pose_all(entry.ref_path, entry.boltz_path, max_poses=1, align_protein=True)
    _require_all_ok(entry.pdbid, boltz_rmsd_rows, label="boltz RMSD")
    boltz_rmsd = boltz_rmsd_rows[0]
    boltz_ligand_id = str(boltz_rmsd.get("pred_ligand_id") or "")
    if not boltz_ligand_id:
        raise RuntimeError(f"{entry.pdbid}: missing boltz pred_ligand_id from RMSD stage")

    if stage_cb is not None:
        stage_cb("RMSD", f"{entry.pdbid} vina")
    vina_rmsd_rows = measure_ligand_pose_all(
        entry.ref_path,
        entry.vina_path,
        max_poses=max(1, vina_max_poses),
        align_protein=False,
    )
    _require_all_ok(entry.pdbid, vina_rmsd_rows, label="vina RMSD")

    vina_structure = load_structure(entry.vina_path)
    vina_models = split_models(vina_structure, max(1, vina_max_poses))
    if len(vina_models) != len(vina_rmsd_rows):
        raise RuntimeError(
            f"{entry.pdbid}: pose count mismatch between RMSD and model split (rmsd={len(vina_rmsd_rows)}, poses={len(vina_models)})"
        )

    vina_ligand_ids: List[str] = []
    for rmsd_row in vina_rmsd_rows:
        pred_ligand_id = str(rmsd_row.get("pred_ligand_id") or "")
        if not pred_ligand_id:
            raise RuntimeError(f"{entry.pdbid}: missing vina pred_ligand_id for pose {rmsd_row.get('pose_index')}")
        vina_ligand_ids.append(pred_ligand_id)

    if stage_cb is not None:
        stage_cb("Normalize", entry.pdbid)
    normalized = normalize_entry_from_selected(
        entry.pdbid,
        ref_structure,
        boltz_pose,
        vina_models,
        ref_ligand=ref_ligand,
        boltz_ligand_id=boltz_ligand_id,
        vina_ligand_ids=vina_ligand_ids,
        out_dir=normalized_dir,
        use_cache=cache_normalized,
    )
    expected_head_atoms = int(float(boltz_rmsd.get("headgroup_atoms") or 0))
    if len(normalized.ref_headgroup_atoms) != expected_head_atoms:
        raise RuntimeError(
            f"{entry.pdbid}: headgroup atom count mismatch between RMSD ({expected_head_atoms}) and normalization ({len(normalized.ref_headgroup_atoms)})"
        )
    if len(normalized.boltz_headgroup_atoms) != len(normalized.ref_headgroup_atoms):
        raise RuntimeError(
            f"{entry.pdbid}: boltz headgroup atom count mismatch (ref={len(normalized.ref_headgroup_atoms)}, boltz={len(normalized.boltz_headgroup_atoms)})"
        )
    for idx, (rmsd_row, atoms) in enumerate(zip(vina_rmsd_rows, normalized.vina_headgroup_atoms), start=1):
        rmsd_head = int(float(rmsd_row.get("headgroup_atoms") or 0))
        if rmsd_head != expected_head_atoms:
            raise RuntimeError(
                f"{entry.pdbid}: vina pose {idx} headgroup atom count mismatch vs boltz RMSD (vina={rmsd_head}, boltz={expected_head_atoms})"
            )
        if len(atoms) != len(normalized.ref_headgroup_atoms):
            raise RuntimeError(
                f"{entry.pdbid}: vina pose {idx} headgroup atom count mismatch (ref={len(normalized.ref_headgroup_atoms)}, vina={len(atoms)})"
            )

    head_env_cutoff_a = DEFAULT_HEAD_ENV_CUTOFF_A
    ref_head_env = headgroup_environment_residues(
        normalized.ref_pdb,
        headgroup_atom_names=normalized.ref_headgroup_atoms,
        cutoff_a=head_env_cutoff_a,
    )

    contacts_dir = _contact_cache_dir(normalized_dir, entry.pdbid)
    contacts_dir.mkdir(parents=True, exist_ok=True)

    if stage_cb is not None:
        stage_cb("PandaMap", f"{entry.pdbid} ref")
    ref_contacts_all = cached_contacts(
        normalized.ref_pdb,
        contacts_dir / "ref_contacts.csv",
        ligand_resname=NORMALIZED_LIGAND_RESNAME,
        use_cache=cache_contacts,
    )
    ref_head_contacts = filter_headgroup_contacts(ref_contacts_all, allowed_atoms=set(normalized.ref_headgroup_atoms))
    ref_head_typed = contacts_to_typed_set(ref_head_contacts)
    ref_head_types = interaction_type_counts(ref_head_contacts)
    ref_head_contact_count = len(ref_head_contacts)

    if stage_cb is not None:
        stage_cb("PandaMap", f"{entry.pdbid} boltz")
    boltz_map, boltz_map_qc = build_residue_id_map_with_qc(boltz_pose, ref_structure)
    boltz_head_env_pred = headgroup_environment_residues(
        normalized.boltz_pdb,
        headgroup_atom_names=normalized.boltz_headgroup_atoms,
        cutoff_a=head_env_cutoff_a,
    )
    boltz_head_env = remap_residue_ids(boltz_head_env_pred, boltz_map)
    boltz_contacts_all = cached_contacts(
        normalized.boltz_pdb,
        contacts_dir / "boltz_contacts.csv",
        ligand_resname=NORMALIZED_LIGAND_RESNAME,
        use_cache=cache_contacts,
    )
    boltz_head_contacts = filter_headgroup_contacts(
        boltz_contacts_all,
        allowed_atoms=set(normalized.boltz_headgroup_atoms),
    )
    boltz_head_typed = remap_typed_ids(contacts_to_typed_set(boltz_head_contacts), boltz_map)
    boltz_head_types = interaction_type_counts(boltz_head_contacts)
    boltz_head_contact_count = len(boltz_head_contacts)

    _write_boltz_mapping_qc(
        contacts_dir / "boltz_residue_mapping_qc.json",
        entry.pdbid,
        boltz_map_qc,
        boltz_map,
        boltz_head_contacts,
        head_env_cutoff_a=head_env_cutoff_a,
        use_cache=cache_contacts,
    )

    allposes: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []

    boltz_row = {
        "pdbid": entry.pdbid,
        "method": "boltz",
        "pose_index": int(boltz_rmsd["pose_index"]),
        "ref_ligand_id": ref_ligand_id,
        "pred_ligand_id": boltz_ligand_id,
        "pairing_method": boltz_rmsd.get("pairing_method", ""),
        "ligand_heavy_atoms": _safe_int(boltz_rmsd.get("ligand_heavy_atoms")),
        "ligand_rmsd": _safe_float(boltz_rmsd.get("ligand_rmsd")),
        "headgroup_atoms": int(float(boltz_rmsd.get("headgroup_atoms") or 0)),
        "headgroup_rmsd": _finite_float_or_na(boltz_rmsd.get("headgroup_rmsd")),
        "protein_pairs": _safe_int(boltz_rmsd.get("protein_pairs")),
        "protein_rmsd": _safe_float(boltz_rmsd.get("protein_rmsd")),
        "headgroup_contacts_ref": ref_head_contact_count,
        "headgroup_contacts_pred": boltz_head_contact_count,
        "headgroup_types_ref": ref_head_types,
        "headgroup_types_pred": boltz_head_types,
        **_set_metrics_na_if_ref_empty(ref_head_env, boltz_head_env, "head_env"),
        **_set_metrics_na_if_ref_empty(ref_head_typed, boltz_head_typed, "headgroup_typed"),
    }
    allposes.append(boltz_row)
    summary.append(boltz_row)

    vina_rows: List[Dict[str, object]] = []
    if stage_cb is not None:
        stage_cb("PandaMap", f"{entry.pdbid} vina")
    for rmsd_row in vina_rmsd_rows:
        pose_index = int(rmsd_row["pose_index"])
        pose_idx = pose_index - 1
        if pose_idx < 0 or pose_idx >= len(normalized.vina_pdbs):
            raise RuntimeError(f"{entry.pdbid}: normalized Vina pose missing for pose_index={pose_index}")

        pose_path = normalized.vina_pdbs[pose_idx]
        pose_head_env = headgroup_environment_residues(
            pose_path,
            headgroup_atom_names=normalized.vina_headgroup_atoms[pose_idx],
            cutoff_a=head_env_cutoff_a,
        )
        pose_cache = contacts_dir / f"vina_pose_{pose_index}_contacts.csv"
        pose_contacts_all = cached_contacts(
            pose_path,
            pose_cache,
            ligand_resname=NORMALIZED_LIGAND_RESNAME,
            use_cache=cache_contacts,
        )
        allowed_atoms = set(normalized.vina_headgroup_atoms[pose_idx])
        pose_head_contacts = filter_headgroup_contacts(pose_contacts_all, allowed_atoms=allowed_atoms)
        pose_head_typed = contacts_to_typed_set(pose_head_contacts)
        pose_head_types = interaction_type_counts(pose_head_contacts)
        pose_head_contact_count = len(pose_head_contacts)

        row = {
            "pdbid": entry.pdbid,
            "method": "vina_pose",
            "pose_index": pose_index,
            "ref_ligand_id": ref_ligand_id,
            "pred_ligand_id": str(rmsd_row.get("pred_ligand_id") or ""),
            "pairing_method": rmsd_row.get("pairing_method", ""),
            "ligand_heavy_atoms": _safe_int(rmsd_row.get("ligand_heavy_atoms")),
            "ligand_rmsd": _safe_float(rmsd_row.get("ligand_rmsd")),
            "headgroup_atoms": int(float(rmsd_row.get("headgroup_atoms") or 0)),
            "headgroup_rmsd": _finite_float_or_na(rmsd_row.get("headgroup_rmsd")),
            "protein_pairs": NA,
            "protein_rmsd": NA,
            "headgroup_contacts_ref": ref_head_contact_count,
            "headgroup_contacts_pred": pose_head_contact_count,
            "headgroup_types_ref": ref_head_types,
            "headgroup_types_pred": pose_head_types,
            **_set_metrics_na_if_ref_empty(ref_head_env, pose_head_env, "head_env"),
            **_set_metrics_na_if_ref_empty(ref_head_typed, pose_head_typed, "headgroup_typed"),
        }
        vina_rows.append(row)
        allposes.append(row)

    def _as_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    best_vina = min(vina_rows, key=lambda r: _as_float(r.get("ligand_rmsd")))
    summary.append({**best_vina, "method": "vina_best"})

    best_headgroup = min(vina_rows, key=lambda r: _as_float(r.get("headgroup_rmsd")))
    summary.append({**best_headgroup, "method": "vina_best_headgroup"})

    return allposes, summary


def _write_boltz_mapping_qc(
    path: Path,
    pdbid: str,
    qc: ResidueMapQc,
    mapping: Dict[str, str],
    head_contacts: Sequence[Dict[str, object]],
    *,
    head_env_cutoff_a: float,
    use_cache: bool,
) -> None:
    if use_cache and path.exists():
        return
    residues = {str(c.get("residue") or "") for c in head_contacts if str(c.get("residue") or "")}
    unmapped = sorted(r for r in residues if r not in mapping)
    data = {
        "pdbid": pdbid,
        "chain_pairs": qc.chain_pairs,
        "aligned_pairs": qc.aligned_pairs,
        "identity": qc.identity,
        "pred_residues": qc.pred_residues,
        "ref_residues": qc.ref_residues,
        "coverage_pred": qc.coverage_pred,
        "coverage_ref": qc.coverage_ref,
        "head_env_cutoff_a": float(head_env_cutoff_a),
        "unmapped_headgroup_contact_residues": len(unmapped),
        "unmapped_headgroup_contact_residue_examples": unmapped[:10],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_benchmark(
    entries: Sequence[PairEntry],
    *,
    vina_max_poses: int,
    normalized_dir: Path,
    quiet: bool,
    workers: int = 1,
    cache_normalized: bool = True,
    cache_contacts: bool = True,
    progress_cb: Callable[[int, int, str], None] | None = None,
    stage_cb: Callable[[str, str], None] | None = None,
    entry_cb: Callable[[List[Dict[str, object]], List[Dict[str, object]], int, int], None] | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Run the benchmark for a set of targets.

    Inputs
    - `entries`: targets to evaluate (each includes paths to ref/Boltz/Vina files)
    - `vina_max_poses`: maximum number of Vina poses to evaluate per target
    - `normalized_dir`: where normalized complexes are written (and optionally cached)
    - `workers`: number of parallel processes (1 = run in the current process)

    Outputs
    - `allposes`: rows for `benchmark_allposes.csv` (one row per evaluated pose)
    - `summary`: rows for `benchmark_summary.csv` (best-per-target summaries)

    Notes
    - When `workers > 1`, targets are processed in parallel, which is faster but may make
      logs appear out of order.
    - `progress_cb`, `stage_cb`, and `entry_cb` are hooks used by the TUI to render progress
      and live summary statistics while the benchmark is running.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    allposes: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []

    total = len(entries)
    if workers == 1:
        for idx, entry in enumerate(entries, start=1):
            if progress_cb is None:
                if quiet:
                    print(f"\rBenchmark: {idx}/{total}", end="", flush=True)
                else:
                    LOGGER.info("[%d/%d] %s", idx, total, entry.pdbid)
            entry_all, entry_summary = _run_entry(
                entry,
                vina_max_poses=vina_max_poses,
                normalized_dir=normalized_dir,
                cache_normalized=cache_normalized,
                cache_contacts=cache_contacts,
                stage_cb=stage_cb,
            )
            allposes.extend(entry_all)
            summary.extend(entry_summary)
            if progress_cb is not None:
                progress_cb(idx, total, entry.pdbid)
            if entry_cb is not None:
                entry_cb(entry_all, entry_summary, idx, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_entry,
                    entry,
                    vina_max_poses=vina_max_poses,
                    normalized_dir=normalized_dir,
                    cache_normalized=cache_normalized,
                    cache_contacts=cache_contacts,
                ): entry
                for entry in entries
            }
            completed = 0
            for future in as_completed(futures):
                entry = futures[future]
                entry_all, entry_summary = future.result()
                allposes.extend(entry_all)
                summary.extend(entry_summary)
                completed += 1
                if progress_cb is None:
                    if quiet:
                        print(f"\rBenchmark: {completed}/{total}", end="", flush=True)
                    else:
                        LOGGER.info("Completed %s (%d/%d)", entry.pdbid, completed, total)
                else:
                    progress_cb(completed, total, entry.pdbid)
                if entry_cb is not None:
                    entry_cb(entry_all, entry_summary, completed, total)

    allposes.sort(key=lambda r: (r.get("pdbid", ""), r.get("method", ""), int(r.get("pose_index", 0) or 0)))
    summary.sort(key=lambda r: (r.get("pdbid", ""), r.get("method", "")))

    if quiet:
        print()
    return allposes, summary
