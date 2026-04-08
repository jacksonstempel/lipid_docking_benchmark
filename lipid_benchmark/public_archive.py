"""
Canonical tracked manuscript archive paths.

The public repository ships a small, explicit archive of benchmark-result CSVs
that is sufficient to rebuild the manuscript analysis bundle. Scripts that
operate on the publication companion should use these paths directly rather than
probing local `output/` directories.
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


def require_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required tracked manuscript-archive files:\n{missing_str}")
