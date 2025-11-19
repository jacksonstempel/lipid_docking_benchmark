from __future__ import annotations

import logging
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .alignment import FitResult, chimera_pruned_fit, extract_chain_sequences, pair_chains
from .ligands import (
    SimpleResidue,
    apply_template_names,
    collect_ligands,
    filter_large_ligands,
    load_ligand_template_names,
    locked_rmsd,
    pairs_by_rdkit,
    pairs_by_name,
    randomized_copy,
)
from .paths import PathResolver
from .pockets import local_pocket_fit
from .structures import (
    apply_rt_to_structure,
    ensure_protein_backbone,
    load_structure,
    split_models,
    protein_bounding_box,
)

LOGGER = logging.getLogger(__name__)
INF = 1e9


class LigandMatch(Dict[str, object]):
    """Helper dict subtype for summary serialization."""


def _select_reference_ligand(ref_ligands: List[SimpleResidue], ref_record: Dict[str, object]) -> SimpleResidue | None:
    chain = str(ref_record.get("chain", ""))
    name = str(ref_record.get("name", ""))
    resid = str(ref_record.get("id", ""))
    for ligand in ref_ligands:
        if ligand.chain_id == chain and ligand.res_name == name and ligand.res_id == resid:
            return ligand
    return ref_ligands[0] if ref_ligands else None


def _protein_alignment_pairs(chain_pairs) -> Tuple[np.ndarray, np.ndarray]:
    coords_pred: List[np.ndarray] = []
    coords_ref: List[np.ndarray] = []
    for pair in chain_pairs:
        for pred_idx, ref_idx in pair.res_pairs:
            ca_pred = pair.pred.ca_xyz[pred_idx]
            ca_ref = pair.ref.ca_xyz[ref_idx]
            if ca_pred is None or ca_ref is None:
                continue
            coords_pred.append(ca_pred)
            coords_ref.append(ca_ref)
    if len(coords_pred) < 3:
        raise RuntimeError("Not enough Cα pairs to align proteins")
    return np.stack(coords_pred, axis=0), np.stack(coords_ref, axis=0)


def _template_names(project_root: Path, pdbid: str, pred_path: Path, include_h: bool) -> List[str] | None:
    try:
        # For Vina PDBQT outputs, prefer PDB template names (unique) rather than
        # PDBQT names (often generic like 'C', 'O').
        prefer_pdbqt = False if pred_path.suffix.lower() == ".pdbqt" else None
        return load_ligand_template_names(
            project_root,
            pdbid,
            include_h=include_h,
            prefer_pdbqt=prefer_pdbqt,
        )
    except Exception:
        return None


def run_pose_benchmark(
    *,
    pdbid: str,
    resolver: PathResolver,
    project_root: Path,
    ref_path: Path,
    pred_path: Path,
    pose_count: int,
    include_h: bool,
    include_small: bool,
    enable_pocket: bool,
    pocket_radius: float,
    capture_full: bool,
) -> Dict[str, object]:
    ref_structure = load_structure(ref_path)
    pred_structure = load_structure(pred_path)
    pose_structures = split_models(pred_structure, pose_count)
    if not pose_structures:
        raise RuntimeError("Prediction file contains no models")

    ref_ligands_all = collect_ligands(ref_structure, include_h=include_h)
    ref_ligands = ref_ligands_all if include_small else filter_large_ligands(ref_ligands_all)

    template_names = _template_names(project_root, pdbid, pred_path, include_h)

    detail_rows: List[Dict[str, Any]] = []
    best_metrics: List[LigandMatch] = []
    protein_metrics: FitResult | None = None
    best_pose_entry: LigandMatch | None = None
    best_chain_pairs = None
    ligand_evaluated_total = 0
    ligand_matched_total = 0
    random_rows: List[Dict[str, Any]] = []

    for pose_index, pose_structure in enumerate(pose_structures, start=1):
        LOGGER.info("Evaluating pose %d/%d", pose_index, len(pose_structures))

        pose_with_backbone = ensure_protein_backbone(pose_structure, ref_structure)
        pred_chains = extract_chain_sequences(pose_with_backbone)
        ref_chains = extract_chain_sequences(ref_structure)
        chain_pairs = pair_chains(pred_chains, ref_chains)

        P, Q = _protein_alignment_pairs(chain_pairs)
        fit = chimera_pruned_fit(P, Q, cutoff=2.0, keep_mask=capture_full and protein_metrics is None)
        LOGGER.info(
            "Pose %d: Protein Cα RMSD (pruned=%d) = %.3f Å",
            pose_index,
            fit.n_pruned,
            fit.rmsd_pruned,
        )
        if protein_metrics is None:
            protein_metrics = fit
            detail_rows.append(
                {
                    "record_type": "protein_fit",
                    "pdbid": pdbid,
                    "pose_index": 1,
                    "protein_pairs_pruned": fit.n_pruned,
                    "protein_rmsd_ca_pruned": fit.rmsd_pruned,
                    "protein_pairs_all": fit.n_all,
                    "protein_rmsd_ca_all_under_pruned": fit.rmsd_all_under_pruned,
                    "protein_rmsd_ca_allfit": fit.rmsd_allfit,
                }
            )
            if capture_full and fit.kept_mask is not None:
                distances = np.sqrt(((P @ fit.R + fit.t - Q) ** 2).sum(axis=1))
                for idx, (kept_flag, dist) in enumerate(zip(list(fit.kept_mask), distances.tolist())):
                    detail_rows.append(
                        {
                            "record_type": "protein_pair",
                            "pdbid": pdbid,
                            "pose_index": 1,
                            "protein_metric_index": idx,
                            "protein_metric": "kept",
                            "protein_value": int(kept_flag),
                        }
                    )
                    detail_rows.append(
                        {
                            "record_type": "protein_pair",
                            "pdbid": pdbid,
                            "pose_index": 1,
                            "protein_metric_index": idx,
                            "protein_metric": "dist_A",
                            "protein_value": dist,
                        }
                    )

        apply_rt_to_structure(pose_with_backbone, fit.R, fit.t)

        pred_ligands_all = collect_ligands(pose_with_backbone, include_h=include_h)
        apply_template_names(pred_ligands_all, template_names)
        pred_ligands = pred_ligands_all if include_small else filter_large_ligands(pred_ligands_all)

        LOGGER.info(
            "Pose %d: Reference ligands %d; predicted candidates %d",
            pose_index,
            len(ref_ligands),
            len(pred_ligands),
        )
        if not ref_ligands or not pred_ligands:
            continue

        ligand_evaluated_total += len(ref_ligands)

        cost = np.full((len(pred_ligands), len(ref_ligands)), INF, dtype=float)
        cached_pairs: List[List[List[Tuple[int, int]]]] = [
            [[] for _ in ref_ligands] for _ in pred_ligands
        ]
        pair_policies: List[List[str]] = [["" for _ in ref_ligands] for _ in pred_ligands]
        for i, pred_ligand in enumerate(pred_ligands):
            for j, ref_ligand in enumerate(ref_ligands):
                # Prefer chemistry-aware mapping via RDKit; fallback to by-name
                rd_pairs = pairs_by_rdkit(pred_ligand, ref_ligand)
                if len(rd_pairs) >= 3:
                    atom_pairs = rd_pairs
                    policy = "chem"
                else:
                    atom_pairs = pairs_by_name(pred_ligand, ref_ligand)
                    policy = "by-name"
                cached_pairs[i][j] = atom_pairs
                pair_policies[i][j] = policy
                if len(atom_pairs) >= 3:
                    lg_cost, _ = locked_rmsd(pred_ligand.atoms, ref_ligand.atoms, atom_pairs, np.eye(3), np.zeros(3))
                    if math.isfinite(lg_cost):
                        cost[i, j] = lg_cost

        for j, ref_ligand in enumerate(ref_ligands):
            if not np.isfinite(cost[:, j]).any():
                raise RuntimeError(
                    f"By-name pairing failed (pose {pose_index}) for reference ligand {ref_ligand.res_name}"
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(row_ind, col_ind) if np.isfinite(cost[i, j])]

        for i, j in matches:
            pred_ligand = pred_ligands[i]
            ref_ligand = ref_ligands[j]
            atom_pairs = cached_pairs[i][j]
            policy = pair_policies[i][j] if pair_policies else "by-name"
            rmsd_global, pair_count = locked_rmsd(
                pred_ligand.atoms, ref_ligand.atoms, atom_pairs, np.eye(3), np.zeros(3)
            )
            pocket_pairs = 0
            rmsd_pocket = rmsd_global
            if enable_pocket and not math.isinf(rmsd_global):
                R_local, t_local, pocket_pairs = local_pocket_fit(
                    ref_structure,
                    chain_pairs,
                    ref_ligand,
                    pocket_radius,
                )
                if pocket_pairs >= 3:
                    rmsd_pocket, _ = locked_rmsd(
                        pred_ligand.atoms, ref_ligand.atoms, atom_pairs, R_local, t_local
                    )

            detail_rows.append(
                {
                    "record_type": "ligand_match",
                    "pdbid": pdbid,
                    "pose_index": pose_index,
                    "protein_pairs_pruned": fit.n_pruned,
                    "protein_rmsd_ca_pruned": fit.rmsd_pruned,
                    "protein_pairs_all": fit.n_all,
                    "protein_rmsd_ca_all_under_pruned": fit.rmsd_all_under_pruned,
                    "protein_rmsd_ca_allfit": fit.rmsd_allfit,
                    "pred_chain": pred_ligand.chain_id,
                    "pred_resname": pred_ligand.res_name,
                    "pred_resid": pred_ligand.res_id,
                    "ref_chain": ref_ligand.chain_id,
                    "ref_resname": ref_ligand.res_name,
                    "ref_resid": ref_ligand.res_id,
                    "policy": policy,
                    "atom_pairs": pair_count,
                    "rmsd_locked_global": rmsd_global,
                    "rmsd_locked_pocket": rmsd_pocket,
                    "pocket_pairs": pocket_pairs,
                }
            )

            result_entry: LigandMatch = LigandMatch(
                pose_index=pose_index,
                pred=pred_ligand.to_dict(),
                ref=ref_ligand.to_dict(),
                policy=policy,
                n=pair_count,
                rmsd_locked_global=rmsd_global,
                rmsd_locked_pocket=rmsd_pocket,
                pocket_pairs=pocket_pairs,
            )
            best_metrics.append(result_entry)
            ligand_matched_total += 1

            if (best_pose_entry is None) or (
                rmsd_global < float(best_pose_entry["rmsd_locked_global"])
            ):
                best_pose_entry = result_entry
                best_chain_pairs = chain_pairs

            LOGGER.info(
                "Pose %d: Ligand %s:%s (%s) ↔ %s:%s (%s) — %s locked_global=%.3f Å, locked_pocket=%.3f Å (n=%d, pocket_pairs=%d)",
                pose_index,
                pred_ligand.chain_id,
                pred_ligand.res_name,
                pred_ligand.res_id,
                ref_ligand.chain_id,
                ref_ligand.res_name,
                ref_ligand.res_id,
                policy,
                rmsd_global,
                rmsd_pocket,
                pair_count,
                pocket_pairs,
            )

        # Random ligand placement control: evaluate a randomized copy of each reference ligand
        # using the same protein alignment and pocket fit context.
        seed_bytes = hashlib.sha256(pdbid.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        box_min, box_max = protein_bounding_box(pose_with_backbone)
        for ref_ligand in ref_ligands:
            rand_pred = randomized_copy(ref_ligand, rng=rng, box_min=box_min, box_max=box_max)
            atom_pairs = pairs_by_name(rand_pred, ref_ligand)
            rmsd_global, pair_count = locked_rmsd(rand_pred.atoms, ref_ligand.atoms, atom_pairs, np.eye(3), np.zeros(3))
            rmsd_pocket = rmsd_global
            pocket_pairs = 0
            if enable_pocket and not math.isinf(rmsd_global):
                R_local, t_local, pocket_pairs = local_pocket_fit(
                    ref_structure,
                    chain_pairs,
                    ref_ligand,
                    pocket_radius,
                )
                if pocket_pairs >= 3:
                    rmsd_pocket, _ = locked_rmsd(rand_pred.atoms, ref_ligand.atoms, atom_pairs, R_local, t_local)
            random_rows.append(
                {
                    "record_type": "ligand_random",
                    "pdbid": pdbid,
                    "pose_index": pose_index,
                    "protein_pairs_pruned": fit.n_pruned,
                    "protein_rmsd_ca_pruned": fit.rmsd_pruned,
                    "protein_pairs_all": fit.n_all,
                    "protein_rmsd_ca_all_under_pruned": fit.rmsd_all_under_pruned,
                    "protein_rmsd_ca_allfit": fit.rmsd_allfit,
                    "pred_chain": rand_pred.chain_id,
                    "pred_resname": rand_pred.res_name,
                    "pred_resid": rand_pred.res_id,
                    "ref_chain": ref_ligand.chain_id,
                    "ref_resname": ref_ligand.res_name,
                    "ref_resid": ref_ligand.res_id,
                    "policy": "random",
                    "atom_pairs": pair_count,
                    "rmsd_locked_global": rmsd_global,
                    "rmsd_locked_pocket": rmsd_pocket,
                    "pocket_pairs": pocket_pairs,
                }
            )

    if protein_metrics is None:
        raise RuntimeError("Protein alignment metrics were not computed.")

    summary: Dict[str, Any] = {
        "protein_rmsd_ca_pruned": protein_metrics.rmsd_pruned,
        "protein_pairs_pruned": protein_metrics.n_pruned,
        "protein_rmsd_ca_all_under_pruned": protein_metrics.rmsd_all_under_pruned,
        "protein_pairs_all": protein_metrics.n_all,
        "protein_rmsd_ca_allfit": protein_metrics.rmsd_allfit,
        "ligand_best_metrics": list(best_metrics),
        "evaluated_pose_count": len(pose_structures),
        "ligand_evaluated_count": ligand_evaluated_total,
        "ligand_matched_count": ligand_matched_total,
    }
    if best_pose_entry is not None:
        summary["best_pose"] = dict(best_pose_entry)

    return {
        "details": [*detail_rows, *random_rows],
        "summary": summary,
    }
