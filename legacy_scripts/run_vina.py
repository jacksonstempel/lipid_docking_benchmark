#!/usr/bin/env python3
"""
Prepare and run an AutoDock Vina docking job for a given PDB ID.

The script expects that `prep_vina_from_refs.py` has already created the
receptor/ligand PDBQT files and box definition under `docking/prep/<PDBID>`
and `docking/vina/box/<PDBID>.txt`.

Usage:
    python scripts/run_vina.py 1FDQ

Outputs will be written to `model_outputs/vina/<PDBID>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(root: Path, pdbid: str) -> tuple[dict, Path]:
    manifest_path = root / "docking" / "prep" / pdbid / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prep manifest not found for {pdbid!r}. "
            f"Expected at {manifest_path} – run prep_vina_from_refs.py first."
        )
    with manifest_path.open() as fh:
        data = json.load(fh)
    return data, manifest_path


def resolve_vina_binary(candidate: str | None) -> str:
    """
    Determine the path to the AutoDock Vina executable.
    Candidate precedence:
      1. --vina-bin flag (if provided)
      2. VINA_BIN environment variable
      3. 'vina' found on PATH
    """
    search = candidate or os.environ.get("VINA_BIN") or "vina"
    path = shutil.which(search)
    if path:
        return path
    # Allow direct path when shutil.which cannot resolve (e.g. relative ./vina)
    maybe_path = Path(search)
    if maybe_path.exists() and os.access(maybe_path, os.X_OK):
        return str(maybe_path.resolve())
    raise FileNotFoundError(
        f"Could not locate AutoDock Vina executable. "
        f"Tried '{search}'. Set --vina-bin or VINA_BIN."
    )


def prepare_config_lines(
    receptor: Path,
    ligand: Path,
    center: list[float],
    size: list[float],
    out_path: Path,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    seed: int | None,
    cpu: int | None,
) -> list[str]:
    lines = [
        f"receptor = {receptor}",
        f"ligand = {ligand}",
        f"center_x = {center[0]:.3f}",
        f"center_y = {center[1]:.3f}",
        f"center_z = {center[2]:.3f}",
        f"size_x = {size[0]:.3f}",
        f"size_y = {size[1]:.3f}",
        f"size_z = {size[2]:.3f}",
        f"out = {out_path}",
        f"exhaustiveness = {exhaustiveness}",
        f"num_modes = {num_modes}",
        f"energy_range = {energy_range}",
    ]
    if seed is not None:
        lines.append(f"seed = {seed}")
    if cpu is not None:
        lines.append(f"cpu = {cpu}")
    return lines


def run_vina(cmd: list[str], cwd: Path, log_path: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    with log_path.open("w") as log_fh:
        log_fh.write("# AutoDock Vina run\n")
        log_fh.write(f"# Command: {' '.join(cmd)}\n")
        log_fh.write(f"# Working dir: {cwd}\n\n")
        log_fh.flush()
        try:
            subprocess.run(
                cmd,
                cwd=str(cwd),
                check=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"AutoDock Vina failed with exit code {exc.returncode}") from exc


def write_metadata(
    dst: Path,
    *,
    pdbid: str,
    timestamp: str,
    vina_exec: str,
    manifest_path: Path,
    config_path: Path,
    pose_path: Path,
    log_path: Path,
    center: list[float],
    size: list[float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    seed: int | None,
    cpu: int | None,
) -> None:
    metadata = {
        "pdbid": pdbid,
        "timestamp_utc": timestamp,
        "vina_executable": vina_exec,
        "prep_manifest": str(manifest_path),
        "config_file": str(config_path),
        "output_pose": str(pose_path),
        "output_log": str(log_path),
        "center": center,
        "size": size,
        "parameters": {
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "energy_range": energy_range,
            "seed": seed,
            "cpu": cpu,
        },
    }
    with dst.open("w") as fh:
        json.dump(metadata, fh, indent=2)


def update_latest_symlink(output_dir: Path, run_dir: Path) -> None:
    latest_link = output_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except OSError as err:
        print(f"[warn] Could not update latest symlink: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoDock Vina for a prepared PDB ID.")
    parser.add_argument("pdbid", help="PDB identifier (case-insensitive)")
    parser.add_argument(
        "--vina-bin",
        dest="vina_bin",
        help="Path or name of the AutoDock Vina executable (defaults to $VINA_BIN or 'vina').",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=16,
        help="Vina exhaustiveness parameter (default: 16).",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=20,
        help="Number of poses to keep (default: 20).",
    )
    parser.add_argument(
        "--energy-range",
        type=float,
        default=3.0,
        help="Maximum energy difference between the best and worst pose (default: 3.0 kcal/mol).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic runs.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Number of CPU threads to use (let Vina auto-detect by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdbid = args.pdbid.upper()
    root = repo_root()

    manifest, manifest_path = load_manifest(root, pdbid)
    vina_exec = resolve_vina_binary(args.vina_bin)

    receptor_path = (root / manifest["receptor_pdbqt"]).resolve()
    ligand_path = (root / manifest["ligand_pdbqt"]).resolve()
    if not receptor_path.exists():
        raise FileNotFoundError(f"Receptor PDBQT missing: {receptor_path}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand PDBQT missing: {ligand_path}")

    try:
        center = [float(c) for c in manifest["center"]]
        size = [float(s) for s in manifest["size"]]
    except Exception as err:
        raise ValueError(f"Invalid center/size values in manifest: {err}") from err

    output_dir = root / "model_outputs" / "vina" / pdbid
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"vina_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    pose_path = run_dir / f"{pdbid}_vina_pose.pdbqt"
    log_path = run_dir / f"{pdbid}_vina.log"
    config_path = run_dir / "vina_config.txt"
    manifest_copy = run_dir / "prep_manifest.json"
    metadata_path = run_dir / "run_metadata.json"

    config_lines = prepare_config_lines(
        receptor=receptor_path,
        ligand=ligand_path,
        center=center,
        size=size,
        out_path=pose_path,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        seed=args.seed,
        cpu=args.cpu,
    )
    config_path.write_text("\n".join(config_lines) + "\n")
    shutil.copy2(manifest_path, manifest_copy)

    run_vina([vina_exec, "--config", str(config_path)], cwd=run_dir, log_path=log_path)

    if not pose_path.exists():
        raise RuntimeError("Vina finished without producing an output pose file.")

    write_metadata(
        metadata_path,
        pdbid=pdbid,
        timestamp=timestamp,
        vina_exec=vina_exec,
        manifest_path=manifest_path,
        config_path=config_path,
        pose_path=pose_path,
        log_path=log_path,
        center=center,
        size=size,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        seed=args.seed,
        cpu=args.cpu,
    )

    update_latest_symlink(output_dir, run_dir)
    print(f"[done] Results written to {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - show friendly error
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
