from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .alignment import FitResult, chimera_pruned_fit, extract_chain_sequences, pair_chains
from .ligands import (
    SimpleResidue,
    collect_ligands,
    headgroup_indices_functional,
    locked_rmsd,
    pairs_by_rdkit,
)
from .structures import apply_rt_to_structure, ensure_protein_backbone, load_structure, split_models

LOGGER = logging.getLogger(__name__)

# Common solvents/ions/noise that should not be picked as the "main ligand".
# Require RDKit mapping to cover ≥90% of ligand heavy atoms to avoid aligning only a
# small substructure (e.g., just a headgroup) and falsely counting two different
# ligands as matched.
MIN_LIGAND_COVERAGE = 0.9

# Small ligands inflate reporting noise, so the default pipeline drops any with fewer
# heavy atoms than this constant unless the CLI requests otherwise. Lipid ligands in
# this benchmark are far larger; <10 heavy atoms is typical for buffer/ion fragments,
# so this floor avoids accidentally picking solvent as the “ligand”.
MIN_LIGAND_HEAVY_ATOMS = 10
_IGNORED_RES_NAMES = {
    # Solvents / buffers
    "HOH",
    "H2O",
    "WAT",
    "DOD",
    "GOL",
    "EDO",
    "PGO",
    "MPD",
    "PEG",
    "PEO",
    "BME",
    "ACT",
    "CIT",
    "TAR",
    "MES",
    "HEPES",
    "EPE",
    # Ions
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "MN",
    "ZN",
    "FE",
    "CO",
    "CU",
    "NI",
    "SO4",
    "PO4",
    "NO3",
    "BR",
    "I",
}

HEADGROUP_ELEMS = {"O", "N", "P", "S"}


class LigandSelectionError(RuntimeError):
    """Raised when the ligand cannot be unambiguously identified."""


class AtomPairingError(RuntimeError):
    """Raised when ligand atom pairing fails."""


def _protein_alignment_pairs(chain_pairs) -> Tuple[np.ndarray, np.ndarray]:
    """Collect corresponding Cα coordinates from matched chain pairs."""
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


def _filter_ligands(ligands: List[SimpleResidue]) -> List[SimpleResidue]:
    """Return ligands that look like the main small molecule (heavy, not solvent/ion)."""
    selected: List[SimpleResidue] = []
    for lig in ligands:
        resname = lig.res_name.upper()
        if resname in _IGNORED_RES_NAMES:
            continue
        if lig.heavy_atom_count() < MIN_LIGAND_HEAVY_ATOMS:
            continue
        selected.append(lig)
    return selected


def _select_single_ligand(structure, *, include_h: bool = False) -> SimpleResidue:
    """Auto-select one significant ligand; pick best if multiple candidates."""
    ligands = collect_ligands(structure, include_h=include_h)
    filtered = _filter_ligands(ligands)
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 0:
        raise LigandSelectionError(
            "No obvious ligand found after filtering out protein, water, solvents, ions, and tiny fragments."
        )
    # Tie-break: choose the largest by heavy atoms, then deterministic by identifiers.
    filtered_sorted = sorted(
        filtered,
        key=lambda lig: (
            -lig.heavy_atom_count(),
            lig.res_name,
            lig.chain_id,
            str(lig.res_id),
        ),
    )
    chosen = filtered_sorted[0]
    LOGGER.info(
        "Multiple candidate ligands found (%d); chose %s:%s:%s by size/name order.",
        len(filtered),
        chosen.chain_id,
        chosen.res_name,
        chosen.res_id,
    )
    return chosen


def _pair_ligand_atoms(pred: SimpleResidue, ref: SimpleResidue) -> Tuple[List[Tuple[int, int]], str]:
    """Return atom index pairs using chemistry-aware mapping only.

    Requires the RDKit mapping to cover at least MIN_LIGAND_COVERAGE of the
    reference ligand heavy atoms.
    """
    rd_pairs = pairs_by_rdkit(pred, ref)
    ref_heavy = ref.heavy_atom_count()
    matched = len(rd_pairs)
    coverage = matched / ref_heavy if ref_heavy > 0 else 0.0
    if matched >= 3 and coverage >= MIN_LIGAND_COVERAGE:
        return rd_pairs, "RDKit"
    raise AtomPairingError(
        f"RDKit atom pairing failed coverage check: matched {matched}/{ref_heavy} atoms "
        f"(coverage={coverage:.2f}, required>={MIN_LIGAND_COVERAGE:.2f})."
    )


def _best_pred_ligand_by_rmsd(
    ref_ligand: SimpleResidue, candidates: List[SimpleResidue]
) -> Tuple[SimpleResidue, List[Tuple[int, int]], str, float, int]:
    """Choose predicted ligand with lowest chem-only RMSD to the reference ligand.

    This is intentionally name-agnostic: docking outputs (especially Vina) often
    rename ligands (e.g., UNL), so we rely on RDKit chemistry + a coverage gate.
    """
    def _centroid_hvy(lig: SimpleResidue) -> np.ndarray:
        coords = np.array([a.xyz for a in lig.atoms if a.element != "H"], float)
        if coords.size == 0:
            return np.zeros(3, float)
        return coords.mean(axis=0)

    ref_centroid = _centroid_hvy(ref_ligand)
    best = None
    best_pairs: List[Tuple[int, int]] = []
    best_policy = ""
    best_rmsd = float("inf")
    best_count = 0
    pool_sorted = sorted(candidates, key=lambda lig: float(np.linalg.norm(_centroid_hvy(lig) - ref_centroid)))
    for lig in pool_sorted:
        try:
            pairs, policy = _pair_ligand_atoms(lig, ref_ligand)
            rmsd, pair_count = locked_rmsd(
                lig.atoms,
                ref_ligand.atoms,
                pairs,
                np.eye(3),
                np.zeros(3),
            )
            if not np.isfinite(rmsd):
                continue
            if rmsd < best_rmsd:
                best = lig
                best_pairs = pairs
                best_policy = policy
                best_rmsd = rmsd
                best_count = pair_count
        except Exception:
            continue
    if best is None:
        raise AtomPairingError("Could not confidently match ligand atoms between reference and prediction.")
    return best, best_pairs, best_policy, best_rmsd, best_count


def _headgroup_pair_subset(ref_ligand: SimpleResidue, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Filter atom pairs to a headgroup-only subset.

    Definition (functional-group oriented): use the reference ligand's detected
    headgroup atoms (phosphate, charged N, or polar carbonyls) and keep only
    pairs that map onto those atoms.
    """
    head_indices = set(headgroup_indices_functional(ref_ligand))
    if not head_indices:
        return []
    return [(pred_idx, ref_idx) for pred_idx, ref_idx in pairs if ref_idx in head_indices]


def measure_ligand_pose_all(
    ref_path: Path | str,
    pred_path: Path | str,
    *,
    max_poses: int = 1,
    align_protein: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate ligand pose quality for each pose/model independently."""
    ref_path = Path(ref_path).expanduser().resolve()
    pred_path = Path(pred_path).expanduser().resolve()

    LOGGER.info("Loading reference: %s", ref_path)
    ref_structure = load_structure(ref_path)
    LOGGER.info("Loading prediction: %s", pred_path)
    pred_structure = load_structure(pred_path)

    pose_structures = split_models(pred_structure, max_poses)
    if not pose_structures:
        raise RuntimeError("Prediction file contains no models.")

    LOGGER.info("Ensuring prediction has protein backbone for alignment context.")

    # Prep that is identical for all poses.
    ref_chains = extract_chain_sequences(ref_structure)
    ref_ligand = _select_single_ligand(ref_structure, include_h=False)

    entries: List[Dict[str, object]] = []
    for pose_index, pose_structure in enumerate(pose_structures, start=1):
        try:
            if align_protein:
                pose_with_backbone = ensure_protein_backbone(pose_structure, ref_structure)

                pred_chains = extract_chain_sequences(pose_with_backbone)
                chain_pairs = pair_chains(pred_chains, ref_chains)
                if not chain_pairs:
                    raise RuntimeError("Unable to match protein chains between reference and prediction.")

                P, Q = _protein_alignment_pairs(chain_pairs)
                fit: FitResult = chimera_pruned_fit(P, Q, cutoff=2.0)
                LOGGER.info(
                    "Pose %d: Protein alignment pruned Cα RMSD = %.3f Å (pairs=%d)",
                    pose_index,
                    fit.rmsd_pruned,
                    fit.n_pruned,
                )

                apply_rt_to_structure(pose_with_backbone, fit.R, fit.t)
            else:
                pose_with_backbone = pose_structure

            LOGGER.info("Pose %d: Auto-selecting ligand in prediction.", pose_index)
            pred_ligands = collect_ligands(pose_with_backbone, include_h=False)

            pred_ligand_candidates = _filter_ligands(pred_ligands)
            if len(pred_ligand_candidates) == 0:
                raise LigandSelectionError("Prediction: no obvious ligand found after filtering.")
            pred_ligand, pairs, policy, rmsd, pair_count = _best_pred_ligand_by_rmsd(
                ref_ligand, pred_ligand_candidates
            )
            if len(pred_ligand_candidates) > 1:
                LOGGER.info(
                    "Pose %d: multiple candidate ligands found (%d); chose %s:%s:%s by lowest chem RMSD.",
                    pose_index,
                    len(pred_ligand_candidates),
                    pred_ligand.chain_id,
                    pred_ligand.res_name,
                    pred_ligand.res_id,
                )

            pred_ligand_id = f"{pred_ligand.chain_id}:{pred_ligand.res_name}:{pred_ligand.res_id}"
            head_pairs = _headgroup_pair_subset(ref_ligand, pairs)
            head_rmsd = float("nan")
            head_count = 0
            if head_pairs:
                head_rmsd, head_count = locked_rmsd(
                    pred_ligand.atoms,
                    ref_ligand.atoms,
                    head_pairs,
                    np.eye(3),
                    np.zeros(3),
                )

            entry = {
                "pose_index": pose_index,
                "pred_ligand_id": pred_ligand_id,
                "pairing_method": policy,
                "ligand_heavy_atoms": pair_count,
                "ligand_rmsd": rmsd,
                "headgroup_atoms": head_count,
                "headgroup_rmsd": head_rmsd,
                "protein_pairs": fit.n_all if align_protein else "",
                "protein_rmsd": fit.rmsd_allfit if align_protein else "",
                "status": "ok",
                "error": "",
            }
            entries.append(entry)
        except Exception as exc:
            # Log at DEBUG level (only visible with --verbose)
            LOGGER.debug("Pose %d failed: %s", pose_index, exc)
            # Record failure in output for transparency
            error_msg = str(exc)
            # Truncate very long errors to keep CSV readable
            if len(error_msg) > 150:
                error_msg = error_msg[:147] + "..."
            entries.append({
                "pose_index": pose_index,
                "pred_ligand_id": "",
                "pairing_method": "",
                "ligand_heavy_atoms": "",
                "ligand_rmsd": "",
                "headgroup_atoms": "",
                "headgroup_rmsd": "",
                "protein_pairs": "",
                "protein_rmsd": "",
                "status": "error",
                "error": error_msg,
            })

    if not entries:
        raise RuntimeError("All poses failed.")
    return entries
