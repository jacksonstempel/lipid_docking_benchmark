#!/usr/bin/env bash
# Run AutoDock Vina docking for one (or many) curated targets as a Slurm array task.
#
# This script expects the "vina reproducibility inputs" directory layout:
#   $INPUT_DIR/
#     box/<PDBID>.txt                    (one line: cx cy cz sx sy sz)
#     prep/<PDBID>/receptor.pdbqt
#     prep/<PDBID>/ligand.pdbqt
#
# It writes one combined multi-pose PDBQT per target to:
#   $OUT_ROOT/flat/<PDBID>.pdbqt
# and a per-target log to:
#   $OUT_ROOT/logs/<PDBID>.log

#SBATCH -J vina
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o vina_%A_%a.out
#SBATCH -e vina_%A_%a.err

set -euo pipefail

INPUT_DIR="${INPUT_DIR:?INPUT_DIR not set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT not set}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:?EXHAUSTIVENESS not set}"
NUM_MODES="${NUM_MODES:?NUM_MODES not set}"
SEED="${SEED:-}"
CPU="${CPU:?CPU not set}"
VINA_BIN="${VINA_BIN:-}"
VINA_MODULE="${VINA_MODULE:-}"
SHARDS="${SHARDS:-}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/flat" "$OUT_ROOT/logs"

if [[ -n "$VINA_MODULE" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load "$VINA_MODULE"
  else
    echo "VINA_MODULE was set, but 'module' is not available in this shell." >&2
    exit 2
  fi
fi

if [[ -n "$VINA_BIN" ]]; then
  vina_cmd="$VINA_BIN"
else
  vina_cmd="vina"
fi

if ! command -v "$vina_cmd" >/dev/null 2>&1; then
  echo "Could not find Vina executable: $vina_cmd" >&2
  echo "Set VINA_BIN=/path/to/vina or VINA_MODULE=<module name>." >&2
  exit 2
fi

echo "INPUT_DIR=$INPUT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "EXHAUSTIVENESS=$EXHAUSTIVENESS"
echo "NUM_MODES=$NUM_MODES"
echo "CPU=$CPU"
if [[ -n "$SEED" ]]; then
  echo "SEED=$SEED"
fi
echo "VINA_BIN=$vina_cmd"

if [[ ! -d "$INPUT_DIR/box" || ! -d "$INPUT_DIR/prep" ]]; then
  echo "Expected $INPUT_DIR to contain 'box/' and 'prep/' directories." >&2
  exit 2
fi

shopt -s nullglob
box_glob=("$INPUT_DIR"/box/*.txt)
shopt -u nullglob
if [[ "${#box_glob[@]}" -eq 0 ]]; then
  echo "No box/*.txt files found under: $INPUT_DIR" >&2
  exit 2
fi
mapfile -t pdbids < <(for box_file in "${box_glob[@]}"; do basename "$box_file" .txt; done | sort)

task_id="${SLURM_ARRAY_TASK_ID:-1}"
if [[ -n "$SHARDS" ]]; then
  if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [[ "$SHARDS" -lt 1 ]]; then
    echo "Invalid SHARDS value: $SHARDS" >&2
    exit 2
  fi
  echo "SHARDS=$SHARDS"
fi

run_one() {
  local pdbid="$1"
  local prep_dir="$INPUT_DIR/prep/$pdbid"
  local receptor="$prep_dir/receptor.pdbqt"
  local ligand="$prep_dir/ligand.pdbqt"
  local box_file="$INPUT_DIR/box/$pdbid.txt"
  local out_pdbqt="$OUT_ROOT/flat/$pdbid.pdbqt"
  local log_file="$OUT_ROOT/logs/$pdbid.log"

  if [[ ! -f "$receptor" ]]; then
    echo "Missing receptor: $receptor" >&2
    return 3
  fi
  if [[ ! -f "$ligand" ]]; then
    echo "Missing ligand: $ligand" >&2
    return 3
  fi
  if [[ ! -f "$box_file" ]]; then
    echo "Missing box file: $box_file" >&2
    return 3
  fi

  read -r cx cy cz sx sy sz < "$box_file"
  if [[ -z "${sz:-}" ]]; then
    echo "Invalid box file (expected 6 floats): $box_file" >&2
    return 3
  fi

  cmd=(
    "$vina_cmd"
    --receptor "$receptor"
    --ligand "$ligand"
    --center_x "$cx" --center_y "$cy" --center_z "$cz"
    --size_x "$sx" --size_y "$sy" --size_z "$sz"
    --exhaustiveness "$EXHAUSTIVENESS"
    --num_modes "$NUM_MODES"
    --cpu "$CPU"
    --verbosity 2
    --out "$out_pdbqt"
  )
  if [[ -n "$SEED" ]]; then
    cmd+=(--seed "$SEED")
  fi

  echo "[vina] $pdbid -> $out_pdbqt"
  # Some Vina builds (including the one on conda-forge/bioconda) do not support
  # a `--log` flag, so write a per-target log by redirecting stdout/stderr.
  if ! "${cmd[@]}" >"$log_file" 2>&1; then
    echo "Vina failed for $pdbid. See: $log_file" >&2
    return 4
  fi
  if [[ ! -s "$out_pdbqt" ]]; then
    echo "Vina did not produce an output file for $pdbid: $out_pdbqt" >&2
    echo "First 40 log lines ($log_file):" >&2
    sed -n '1,40p' "$log_file" >&2
    return 5
  fi
}

fail=0
if [[ -z "$SHARDS" ]]; then
  pdbid="${pdbids[$((task_id-1))]:-}"
  if [[ -z "${pdbid:-}" ]]; then
    echo "No PDBID for SLURM_ARRAY_TASK_ID=$task_id (have ${#pdbids[@]} total)" >&2
    exit 2
  fi
  run_one "$pdbid" || fail=1
else
  count=0
  for ((i=task_id-1; i<${#pdbids[@]}; i+=SHARDS)); do
    run_one "${pdbids[$i]}" || fail=1
    count=$((count+1))
  done
  echo "Shard complete. Ran $count targets."
fi

if [[ "$fail" -ne 0 ]]; then
  echo "One or more Vina runs failed." >&2
  exit 3
fi

echo "Done. Flat outputs in: $OUT_ROOT/flat"
