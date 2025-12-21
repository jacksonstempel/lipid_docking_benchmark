from __future__ import annotations

from typing import Dict, Set

NA = "NA"


def set_metrics(ref: Set[str], pred: Set[str], prefix: str) -> Dict[str, float | int]:
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


def set_metrics_na_if_ref_empty(ref: Set[str], pred: Set[str], prefix: str) -> Dict[str, float | int | str]:
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
    return set_metrics(ref, pred, prefix)
