"""
Canonical data paths for the repository.

This module centralizes the locations of the manuscript archive, the curated
model-input packages, and the raw-prediction bundles so that scripts and tests
can resolve them through a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReproArchive:
    root: Path
    baseline_allposes: Path
    baseline_summary: Path
    gnina_cnn_allposes: Path
    gnina_nocnn_allposes: Path
    adversarial_root: Path
    adversarial_gly_allposes: Path
    adversarial_gly_summary: Path
    adversarial_phe_allposes: Path
    adversarial_phe_summary: Path
    mutation_summary: Path
    vina_exh256_root: Path
    vina_exh256_allposes: Path
    vina_exh256_summary: Path
    boltz_high_sampling_root: Path
    boltz_high_sampling_allposes: Path
    boltz_high_sampling_summary: Path


@dataclass(frozen=True)
class PredictionInputs:
    root: Path
    boltz_inputs_root: Path
    vina_inputs_root: Path
    vina_box_root: Path
    vina_prep_root: Path


@dataclass(frozen=True)
class RawPredictions:
    root: Path
    gnina_root: Path
    gnina_cnn_root: Path
    gnina_cnn_pairs: Path
    gnina_nocnn_root: Path
    gnina_nocnn_pairs: Path
    adversarial_root: Path
    adversarial_bs5A_root: Path
    adversarial_gly_root: Path
    adversarial_gly_pairs: Path
    adversarial_phe_root: Path
    adversarial_phe_pairs: Path
    adversarial_mutation_summary: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_repro_archive(root: Path | None = None) -> ReproArchive:
    repo = (root or repo_root()).resolve()
    archive_root = repo / "data" / "reproducibility"
    adversarial_root = archive_root / "adversarial"
    vina_exh256_root = archive_root / "robustness" / "vina_exhaustiveness_256"
    boltz_high_sampling_root = archive_root / "robustness" / "boltz_high_sampling"
    return ReproArchive(
        root=archive_root,
        baseline_allposes=archive_root / "baseline" / "benchmark_allposes.csv",
        baseline_summary=archive_root / "baseline" / "benchmark_summary.csv",
        gnina_cnn_allposes=archive_root / "gnina" / "benchmark_allposes_gnina_cnn.csv",
        gnina_nocnn_allposes=archive_root / "gnina" / "benchmark_allposes_gnina_nocnn.csv",
        adversarial_root=adversarial_root,
        adversarial_gly_allposes=adversarial_root / "benchmark_gly" / "benchmark_allposes.csv",
        adversarial_gly_summary=adversarial_root / "benchmark_gly" / "benchmark_summary.csv",
        adversarial_phe_allposes=adversarial_root / "benchmark_phe" / "benchmark_allposes.csv",
        adversarial_phe_summary=adversarial_root / "benchmark_phe" / "benchmark_summary.csv",
        mutation_summary=adversarial_root / "mutation_summary.csv",
        vina_exh256_root=vina_exh256_root,
        vina_exh256_allposes=vina_exh256_root / "benchmark_allposes.csv",
        vina_exh256_summary=vina_exh256_root / "benchmark_summary.csv",
        boltz_high_sampling_root=boltz_high_sampling_root,
        boltz_high_sampling_allposes=boltz_high_sampling_root / "benchmark_allposes.csv",
        boltz_high_sampling_summary=boltz_high_sampling_root / "benchmark_summary.csv",
    )


def canonical_prediction_inputs(root: Path | None = None) -> PredictionInputs:
    repo = (root or repo_root()).resolve()
    inputs_root = repo / "prediction_inputs"
    vina_root = inputs_root / "vina_inputs"
    return PredictionInputs(
        root=inputs_root,
        boltz_inputs_root=inputs_root / "boltz_inputs",
        vina_inputs_root=vina_root,
        vina_box_root=vina_root / "box",
        vina_prep_root=vina_root / "prep",
    )


def canonical_raw_predictions(root: Path | None = None) -> RawPredictions:
    repo = (root or repo_root()).resolve()
    raw_root = repo / "data" / "raw_predictions"
    gnina_root = raw_root / "gnina"
    adversarial_root = raw_root / "adversarial"
    adversarial_bs5a_root = adversarial_root / "bs_mutagenesis_cutoff5A"
    return RawPredictions(
        root=raw_root,
        gnina_root=gnina_root,
        gnina_cnn_root=gnina_root / "cnn_rescore_exh8_cpu24",
        gnina_cnn_pairs=gnina_root / "cnn_rescore_exh8_cpu24" / "pairs.csv",
        gnina_nocnn_root=gnina_root / "no_cnn_exh8_cpu24",
        gnina_nocnn_pairs=gnina_root / "no_cnn_exh8_cpu24" / "pairs.csv",
        adversarial_root=adversarial_root,
        adversarial_bs5A_root=adversarial_bs5a_root,
        adversarial_gly_root=adversarial_bs5a_root / "gly",
        adversarial_gly_pairs=adversarial_bs5a_root / "gly" / "pairs.csv",
        adversarial_phe_root=adversarial_bs5a_root / "phe",
        adversarial_phe_pairs=adversarial_bs5a_root / "phe" / "pairs.csv",
        adversarial_mutation_summary=adversarial_bs5a_root / "mutation_summary.csv",
    )


def require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_str}")
