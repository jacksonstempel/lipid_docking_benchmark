from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .alignment import FitResult, chimera_pruned_fit, extract_chain_sequences, pair_chains
from .constants import MIN_LIGAND_HEAVY_ATOMS
from .ligands import (
    SimpleResidue,
    apply_template_names,
    collect_ligands,
    load_ligand_template_names,
    locked_rmsd,
    pairs_by_name,
    pairs_by_rdkit,
)
from .structures import apply_rt_to_structure, ensure_protein_backbone, load_structure, split_models

LOGGER = logging.getLogger(__name__)

# Common solvents/ions/noise that should not be picked as the "main ligand".
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


def _guess_pdbid_from_path(path: Path) -> str | None:
    """Best-effort PDB ID guess from filename (used for template names)."""
    stem = path.stem.upper()
    cleaned = "".join(ch for ch in stem if ch.isalnum())
    if 3 <= len(cleaned) <= 6:
        return cleaned
    return None


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
    """Return atom index pairs using chemistry-aware mapping only."""
    rd_pairs = pairs_by_rdkit(pred, ref)
    if len(rd_pairs) >= 3:
        return rd_pairs, "chem"
    raise AtomPairingError("RDKit atom pairing failed to find a confident mapping (no fallback to by-name).")


def _pick_pred_ligand(ref_ligand: SimpleResidue, candidates: List[SimpleResidue]) -> SimpleResidue:
    """Choose the predicted ligand that best corresponds to the reference ligand."""
    same_name = [lig for lig in candidates if lig.res_name.upper() == ref_ligand.res_name.upper()]
    if same_name:
        exact = [
            lig
            for lig in same_name
            if lig.res_id == ref_ligand.res_id and lig.chain_id == ref_ligand.chain_id
        ]
        pool = exact if exact else same_name
    else:
        pool = candidates
    pool_sorted = sorted(
        pool,
        key=lambda lig: (
            -lig.heavy_atom_count(),
            lig.res_name,
            lig.chain_id,
            str(lig.res_id),
        ),
    )
    return pool_sorted[0]


def _best_pred_ligand_by_rmsd(ref_ligand: SimpleResidue, candidates: List[SimpleResidue]) -> Tuple[SimpleResidue, List[Tuple[int, int]], str, float, int]:
    """Choose predicted ligand by closest same-resname candidate with lowest chem-only RMSD."""
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
    pool = [lig for lig in candidates if lig.res_name.upper() == ref_ligand.res_name.upper()]
    preferred_only = True
    if not pool:
        # Fall back to any candidate when names do not match (common for Vina UNL ligands).
        pool = list(candidates)
        preferred_only = False
    pool_sorted = sorted(pool, key=lambda lig: float(np.linalg.norm(_centroid_hvy(lig) - ref_centroid)))
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
    if not preferred_only:
        LOGGER.info(
            "Prediction ligand name mismatch; chose %s:%s:%s by lowest chem RMSD.",
            best.chain_id,
            best.res_name,
            best.res_id,
        )
    return best, best_pairs, best_policy, best_rmsd, best_count


def measure_ligand_pose(ref_path: Path | str, pred_path: Path | str, *, max_poses: int = 1) -> Dict[str, object]:
    """Evaluate ligand pose quality between a reference and prediction structure.

    Steps:
    1. Load ref/pred structures.
    2. Ensure prediction has protein backbone.
    3. For up to `max_poses` models: align proteins (Cα, pruned Kabsch).
    4. Auto-select a single significant ligand in each.
    5. Pair ligand heavy atoms (RDKit chem-only).
    6. Compute heavy-atom ligand RMSD in the aligned frame; return the best pose.
    """
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

    template_names: List[str] | None = None
    pdbid_guess = _guess_pdbid_from_path(ref_path)
    if pdbid_guess:
        project_root = Path(__file__).resolve().parent.parent.parent
        template_names = load_ligand_template_names(
            project_root,
            pdbid_guess,
            include_h=False,
        )

    best: Dict[str, object] | None = None
    errors: List[str] = []

    for pose_index, pose_structure in enumerate(pose_structures, start=1):
        try:
            pose_with_backbone = ensure_protein_backbone(pose_structure, ref_structure)

            pred_chains = extract_chain_sequences(pose_with_backbone)
            ref_chains = extract_chain_sequences(ref_structure)
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

            LOGGER.info("Pose %d: Auto-selecting ligand in reference.", pose_index)
            ref_ligand = _select_single_ligand(ref_structure, include_h=False)
            LOGGER.info("Pose %d: Auto-selecting ligand in prediction.", pose_index)
            pred_ligands = collect_ligands(pose_with_backbone, include_h=False)
            apply_template_names(pred_ligands, template_names)

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

            ligand_id = f"{ref_ligand.chain_id}:{ref_ligand.res_name}:{ref_ligand.res_id}"
            entry = {
                "ref_path": str(ref_path),
                "pred_path": str(pred_path),
                "pose_index": pose_index,
                "ligand_id": ligand_id,
                "pairing_method": policy,
                "ligand_heavy_atoms": pair_count,
                "ligand_rmsd": rmsd,
                "protein_pairs": fit.n_pruned,
                "protein_rmsd": fit.rmsd_pruned,
                "protein_pairs_all": fit.n_all,
                "protein_rmsd_all_under_pruned": fit.rmsd_all_under_pruned,
                "protein_rmsd_allfit": fit.rmsd_allfit,
            }
            if best is None or rmsd < float(best["ligand_rmsd"]):
                best = entry
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pose {pose_index}: {exc}")
            continue

    if best is None:
        raise RuntimeError("All poses failed. Errors: " + "; ".join(errors))
    return best
