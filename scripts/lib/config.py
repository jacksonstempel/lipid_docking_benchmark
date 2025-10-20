from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - helps with friendly error messaging
    raise RuntimeError(
        "PyYAML is required to load the project configuration. "
        "Install it with `pip install pyyaml`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve_path(base: Path, raw: Optional[str | os.PathLike[str]], default: Path | str) -> Path:
    candidate: Path
    if raw is None:
        candidate = Path(default)
    else:
        candidate = Path(raw)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


@dataclass(frozen=True)
class PathSettings:
    refs: Path
    preds: Path
    boltz_preds: Path
    vina_preds: Path
    analysis_root: Path
    proteins_dir: Path
    aggregates_dir: Path
    benchmark_inputs: Path
    model_outputs: Path


@dataclass(frozen=True)
class ScriptSettings:
    pose_benchmark: Path


@dataclass(frozen=True)
class ReportSettings:
    missing_benchmark_runs: Path
    unrun_proteins: Path


@dataclass(frozen=True)
class Config:
    """Container for repository configuration with resolved absolute paths."""

    source: Path
    base_dir: Path
    raw: Dict[str, Any]
    paths: PathSettings
    scripts: ScriptSettings
    reports: ReportSettings


def load_config(path: Optional[str | os.PathLike[str]] = None) -> Config:
    """Load configuration from YAML and resolve repository paths."""
    cfg_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
    else:
        data = {}
    base_dir = cfg_path.parent.resolve()

    paths_cfg: Dict[str, Any] = data.get("paths", {})

    refs = _resolve_path(base_dir, paths_cfg.get("refs"), "raw_structures/benchmark_references")
    preds = _resolve_path(base_dir, paths_cfg.get("preds", paths_cfg.get("model_outputs")), "model_outputs")
    boltz_preds = _resolve_path(base_dir, paths_cfg.get("boltz_preds"), preds / "boltz")
    vina_preds = _resolve_path(base_dir, paths_cfg.get("vina_preds"), preds / "vina")
    analysis_root = _resolve_path(base_dir, paths_cfg.get("analysis_root"), "analysis")
    proteins_dir = _resolve_path(
        base_dir,
        paths_cfg.get("analysis_proteins"),
        analysis_root / "proteins",
    )
    aggregates_dir = _resolve_path(
        base_dir,
        paths_cfg.get("analysis_aggregates"),
        analysis_root / "aggregates",
    )
    benchmark_inputs = _resolve_path(
        base_dir,
        paths_cfg.get("benchmark_inputs"),
        "model_inputs/benchmark_inputs",
    )
    model_outputs = _resolve_path(
        base_dir,
        paths_cfg.get("model_outputs"),
        preds,
    )

    scripts_cfg: Dict[str, Any] = data.get("scripts", {})
    pose_benchmark_script = _resolve_path(
        base_dir,
        scripts_cfg.get("pose_benchmark"),
        "scripts/pose_benchmark.py",
    )

    reports_cfg: Dict[str, Any] = data.get("reports", {})
    missing_runs_path = _resolve_path(
        base_dir,
        reports_cfg.get("missing_benchmark_runs"),
        aggregates_dir / "missing_benchmark_runs.csv",
    )
    unrun_proteins_path = _resolve_path(
        base_dir,
        reports_cfg.get("unrun_proteins"),
        benchmark_inputs.parent / "unrun_proteins.txt",
    )

    paths = PathSettings(
        refs=refs,
        preds=preds,
        boltz_preds=boltz_preds,
        vina_preds=vina_preds,
        analysis_root=analysis_root,
        proteins_dir=proteins_dir,
        aggregates_dir=aggregates_dir,
        benchmark_inputs=benchmark_inputs,
        model_outputs=model_outputs,
    )
    scripts = ScriptSettings(pose_benchmark=pose_benchmark_script)
    reports = ReportSettings(
        missing_benchmark_runs=missing_runs_path,
        unrun_proteins=unrun_proteins_path,
    )

    return Config(
        source=cfg_path,
        base_dir=base_dir,
        raw=data,
        paths=paths,
        scripts=scripts,
        reports=reports,
    )
