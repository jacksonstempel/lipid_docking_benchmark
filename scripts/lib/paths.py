from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .config import Config


def normalize_pdbid(pdbid: str) -> str:
    """Return an uppercase, stripped PDB identifier."""
    return pdbid.strip().upper()


def _coerce_path(value: Optional[str | os.PathLike[str] | Path], default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    return path.resolve()


def _rebase_if_within(path: Path, old_base: Path, new_base: Path) -> Path:
    try:
        rel = path.relative_to(old_base)
    except ValueError:
        return path
    return (new_base / rel).resolve()


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    structures_dir: Path
    analysis_csv: Path
    data_csv: Path
    summary_json: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.structures_dir.mkdir(parents=True, exist_ok=True)


class PathResolver:
    """Resolve repository paths, allowing CLI overrides on top of config defaults."""

    def __init__(
        self,
        config: Config,
        *,
        refs: Optional[str | os.PathLike[str] | Path] = None,
        preds: Optional[str | os.PathLike[str] | Path] = None,
        analysis_dir: Optional[str | os.PathLike[str] | Path] = None,
        proteins_dir: Optional[str | os.PathLike[str] | Path] = None,
        aggregates_dir: Optional[str | os.PathLike[str] | Path] = None,
    ) -> None:
        self._config = config
        self._refs = _coerce_path(refs, config.paths.refs)
        self._preds = _coerce_path(preds, config.paths.preds)

        analysis_root = _coerce_path(analysis_dir, config.paths.analysis_root)
        if proteins_dir is not None:
            proteins_root = _coerce_path(proteins_dir, analysis_root / "proteins")
        else:
            proteins_root = _rebase_if_within(
                config.paths.proteins_dir,
                config.paths.analysis_root,
                analysis_root,
            )
        if aggregates_dir is not None:
            aggregates_root = _coerce_path(aggregates_dir, analysis_root / "aggregates")
        else:
            aggregates_root = _rebase_if_within(
                config.paths.aggregates_dir,
                config.paths.analysis_root,
                analysis_root,
            )

        self._analysis_root = analysis_root
        self._proteins_root = proteins_root
        self._aggregates_root = aggregates_root

    @property
    def config(self) -> Config:
        return self._config

    @property
    def refs_root(self) -> Path:
        return self._refs

    @property
    def preds_root(self) -> Path:
        return self._preds

    @property
    def analysis_root(self) -> Path:
        return self._analysis_root

    @property
    def proteins_root(self) -> Path:
        return self._proteins_root

    @property
    def aggregates_root(self) -> Path:
        return self._aggregates_root

    def analysis_paths_for(self, pdbid: str) -> AnalysisPaths:
        pid = normalize_pdbid(pdbid)
        root = self._proteins_root / pid
        structures_dir = root / "structures"
        analysis_csv = root / "analysis.csv"
        data_csv = root / "data.csv"
        summary_json = root / "summary.json"
        return AnalysisPaths(
            root=root,
            structures_dir=structures_dir,
            analysis_csv=analysis_csv,
            data_csv=data_csv,
            summary_json=summary_json,
        )

    def aggregates_path(self, filename: str) -> Path:
        return self._aggregates_root / filename


def find_reference_cif(pdbid: str, refs_root: Path) -> Optional[Path]:
    """Find a reference CIF under refs_root. Prefer exact stem match; shallower path wins."""
    pid = normalize_pdbid(pdbid)
    wanted = pid.lower()
    candidates: list[Path] = []

    sub = refs_root / pid
    if sub.is_dir():
        candidates.extend(sorted(p for p in sub.glob("*.cif") if p.is_file()))

    explicit = refs_root / f"{pid}.cif"
    if explicit.is_file():
        candidates.append(explicit)

    try:
        for path in refs_root.rglob("*.cif"):
            if path.stem.lower() == wanted:
                candidates.append(path)
    except Exception:
        pass

    if not candidates:
        return None

    unique: dict[str, Path] = {}
    for c in candidates:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        unique.setdefault(key, c)

    def score(path: Path) -> tuple[int, str]:
        return (len(path.parts), str(path))

    return sorted(unique.values(), key=score)[0]


def _has_predictions_segment(path: Path, pdbid: str) -> bool:
    parts = [p.lower() for p in path.parts]
    try:
        idx = parts.index("predictions")
    except ValueError:
        return False
    return idx + 1 < len(parts) and parts[idx + 1] == pdbid.lower()


def find_prediction_cif(pdbid: str, preds_root: Path) -> Optional[Path]:
    """Find the canonical prediction CIF for a target."""
    pid = normalize_pdbid(pdbid)
    canonical = preds_root / f"{pid}_output/boltz_results_{pid}/predictions/{pid}/{pid}_model_0.cif"
    if canonical.is_file():
        return canonical

    base = preds_root / f"{pid}_output"
    if not base.is_dir():
        return None

    matches = [p for p in base.rglob(f"{pid}_model_0.cif") if p.is_file()]
    if not matches:
        return None

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    def score(path: Path) -> tuple[int, int, float, str]:
        return (
            0 if _has_predictions_segment(path, pid) else 1,
            len(path.parts),
            -mtime(path),
            str(path),
        )

    return sorted(matches, key=score)[0]


def find_vina_pose(pdbid: str, preds_root: Path) -> Optional[Path]:
    """Locate the AutoDock Vina pose (PDBQT) for the given target."""
    pid = normalize_pdbid(pdbid)
    base = preds_root / pid
    if not base.exists():
        return None

    latest = base / "latest" / f"{pid}_vina_pose.pdbqt"
    if latest.is_file():
        return latest

    candidates = [p for p in base.glob(f"vina_run_*/*_vina_pose.pdbqt") if p.is_file()]
    if not candidates:
        return None

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    return max(candidates, key=mtime)


def list_candidate_ids(refs_root: Path) -> list[str]:
    """Discover PDB IDs from the references root."""
    seen: set[str] = set()
    for path in refs_root.rglob("*.cif"):
        stem = path.stem
        if stem and len(stem) >= 4:
            seen.add(normalize_pdbid(stem))
    return sorted(seen)
