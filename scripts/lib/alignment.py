from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from Bio.Align import PairwiseAligner
from scipy.optimize import linear_sum_assignment

from .constants import AA3_TO_1
from .structures import is_protein_res

if TYPE_CHECKING:
    import gemmi


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the optimal rotation/translation that aligns P to Q."""
    centered_P = P - P.mean(axis=0, keepdims=True)
    centered_Q = Q - Q.mean(axis=0, keepdims=True)
    covariance = centered_P.T @ centered_Q
    left, _, right_t = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(left @ right_t))
    correction = np.diag([1.0, 1.0, determinant])
    rotation = left @ correction @ right_t
    translation = Q.mean(axis=0) - P.mean(axis=0) @ rotation
    return rotation, translation


@dataclass
class ChainSeq:
    chain: "gemmi.Chain"
    seq: str
    ca_xyz: List[np.ndarray | None]
    res_objs: List["gemmi.Residue"]


@dataclass
class ChainPair:
    pred: ChainSeq
    ref: ChainSeq
    res_pairs: List[Tuple[int, int]]


def _get_ca(residue: "gemmi.Residue"):
    for atom in residue:
        if atom.name.strip().upper() == "CA":
            return atom
    return None


def extract_chain_sequences(structure: "gemmi.Structure") -> List[ChainSeq]:
    sequences: List[ChainSeq] = []
    model = structure[0]
    for chain in model:
        residues = [residue for residue in chain if is_protein_res(residue)]
        if not residues:
            continue
        seq = "".join(AA3_TO_1.get(residue.name.upper(), "X") for residue in residues)
        ca_atoms: List[np.ndarray | None] = []
        for residue in residues:
            atom = _get_ca(residue)
            if atom is None:
                ca_atoms.append(None)
            else:
                ca_atoms.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float))
        sequences.append(ChainSeq(chain=chain, seq=seq, ca_xyz=ca_atoms, res_objs=residues))
    return sequences


def identity_and_pairs(pred: ChainSeq, ref: ChainSeq) -> Tuple[float, List[Tuple[int, int]]]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(pred.seq, ref.seq)[0]
    aligned_pred, aligned_ref = alignment.aligned[0], alignment.aligned[1]

    residue_pairs: List[Tuple[int, int]] = []
    matches = 0
    aligned_len = 0
    for (pred_start, pred_stop), (ref_start, ref_stop) in zip(aligned_pred, aligned_ref):
        for ip, ir in zip(range(pred_start, pred_stop), range(ref_start, ref_stop)):
            res_pred = pred.res_objs[ip]
            res_ref = ref.res_objs[ir]
            if AA3_TO_1.get(res_pred.name.upper(), "X") == AA3_TO_1.get(res_ref.name.upper(), "X"):
                matches += 1
            aligned_len += 1
            residue_pairs.append((ip, ir))
    identity = (matches / aligned_len) if aligned_len else 0.0
    return identity, residue_pairs


def pair_chains(pred_chains: List[ChainSeq], ref_chains: List[ChainSeq]) -> List[ChainPair]:
    if not pred_chains or not ref_chains:
        return []
    cost_matrix = np.full((len(pred_chains), len(ref_chains)), 1.0, dtype=float)
    pair_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i, pred in enumerate(pred_chains):
        for j, ref in enumerate(ref_chains):
            identity, pairs = identity_and_pairs(pred, ref)
            cost_matrix[i, j] = 1.0 - identity
            pair_cache[(i, j)] = pairs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    out: List[ChainPair] = []
    for i, j in zip(row_ind, col_ind):
        pairs = pair_cache[(i, j)]
        if pairs:
            out.append(ChainPair(pred_chains[i], ref_chains[j], pairs))
    return out


@dataclass
class FitResult:
    R: np.ndarray
    t: np.ndarray
    rmsd_pruned: float
    n_pruned: int
    rmsd_all_under_pruned: float
    n_all: int
    rmsd_allfit: float
    kept_mask: np.ndarray | None = None


def chimera_pruned_fit(P: np.ndarray, Q: np.ndarray, cutoff: float = 2.0, *, keep_mask: bool = False) -> FitResult:
    """ChimeraX-style iterative pruning alignment."""
    n = P.shape[0]
    if n < 3:
        raise RuntimeError("Not enough pairs for fitting")

    R_all, t_all = kabsch(P, Q)
    rmsd_allfit = float(np.sqrt(((P @ R_all + t_all - Q) ** 2).sum() / n))

    keep = np.ones(n, dtype=bool)
    rotation = np.eye(3)
    translation = np.zeros(3)
    for _ in range(200):
        indices = np.where(keep)[0]
        if indices.size < 3:
            break

        Pk, Qk = P[indices], Q[indices]
        rotation, translation = kabsch(Pk, Qk)
        distances = np.sqrt(((P[indices] @ rotation + translation - Q[indices]) ** 2).sum(axis=1))
        above = distances > cutoff
        n_above = int(np.count_nonzero(above))
        if n_above == 0:
            break

        drop10 = max(1, int(np.ceil(0.10 * indices.size)))
        drop50 = max(1, int(np.ceil(0.50 * n_above)))

        if drop50 <= drop10:
            candidates = np.where(above)[0]
            worst = np.argsort(distances[candidates])[-drop50:]
            to_drop = candidates[worst]
        else:
            worst = np.argsort(distances)[-drop10:]
            to_drop = worst
        keep[indices[to_drop]] = False

    remaining = np.where(keep)[0]
    Pk, Qk = P[remaining], Q[remaining]
    if Pk.shape[0] >= 3:
        rotation, translation = kabsch(Pk, Qk)
    rmsd_pruned = float(np.sqrt(((Pk @ rotation + translation - Qk) ** 2).sum() / max(Pk.shape[0], 1)))
    rmsd_all_under = float(np.sqrt(((P @ rotation + translation - Q) ** 2).sum() / n))

    return FitResult(
        R=rotation,
        t=translation,
        rmsd_pruned=rmsd_pruned,
        n_pruned=remaining.size,
        rmsd_all_under_pruned=rmsd_all_under,
        n_all=n,
        rmsd_allfit=rmsd_allfit,
        kept_mask=keep if keep_mask else None,
    )
