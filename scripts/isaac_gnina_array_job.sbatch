#!/usr/bin/env bash
# Run GNINA docking for one (or many) curated targets as a Slurm array task.
#
# This script is designed to reuse the repo's existing "vina reproducibility inputs":
#   $INPUT_DIR/
#     box/<PDBID>.txt                    (one line: cx cy cz sx sy sz)
#     prep/<PDBID>/receptor_no_ligand.pdb
#     prep/<PDBID>/ligand.pdbqt
#
# Outputs:
#   $OUT_ROOT/flat/<PDBID>.pdbqt         (multi-pose GNINA output)
#   $OUT_ROOT/logs/<PDBID>.log           (per-target GNINA log)
#
# Notes:
# - We run GNINA inside an Apptainer image (SIF), with GPU support when USE_GPU=1.
# - The ligand is converted to SDF inside the container using OpenBabel to ensure
#   GNINA sees explicit bonds/rotors consistently.

#SBATCH -J gnina
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o gnina_%A_%a.out
#SBATCH -e gnina_%A_%a.err

set -euo pipefail

# ISAAC policy: when requesting multiple CPUs, set OpenMP thread count explicitly.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Avoid accidental oversubscription from BLAS/numexpr libraries.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

INPUT_DIR="${INPUT_DIR:?INPUT_DIR not set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT not set}"
GNINA_SIF="${GNINA_SIF:?GNINA_SIF not set}"
CUDA_MODULE="${CUDA_MODULE:?CUDA_MODULE not set}"
USE_GPU="${USE_GPU:?USE_GPU not set}"
DEVICE="${DEVICE:?DEVICE not set}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:?EXHAUSTIVENESS not set}"
NUM_MODES="${NUM_MODES:?NUM_MODES not set}"
SEED="${SEED:-}"
CPU="${CPU:?CPU not set}"
CNN_SCORING="${CNN_SCORING:?CNN_SCORING not set}"
SCORING="${SCORING:?SCORING not set}"
SHARDS="${SHARDS:-}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/flat" "$OUT_ROOT/logs"

if [[ ! -f "$GNINA_SIF" ]]; then
  echo "GNINA_SIF not found: $GNINA_SIF" >&2
  echo "Create it first (ISAAC docs recommend pulling rather than building), e.g.:" >&2
  echo "  apptainer pull $GNINA_SIF docker://gnina/gnina:latest" >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR/box" || ! -d "$INPUT_DIR/prep" ]]; then
  echo "Expected $INPUT_DIR to contain 'box/' and 'prep/' directories." >&2
  exit 2
fi

if [[ "${USE_GPU}" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
  else
    echo "USE_GPU=1 but 'module' is not available in this shell." >&2
    exit 2
  fi
fi

echo "INPUT_DIR=$INPUT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "GNINA_SIF=$GNINA_SIF"
echo "USE_GPU=$USE_GPU"
echo "DEVICE=$DEVICE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CPU=$CPU"
echo "EXHAUSTIVENESS=$EXHAUSTIVENESS"
echo "NUM_MODES=$NUM_MODES"
echo "CNN_SCORING=$CNN_SCORING"
echo "SCORING=$SCORING"
if [[ -n "$SEED" ]]; then
  echo "SEED=$SEED"
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

apptainer_cmd=(apptainer exec)
if [[ "${USE_GPU}" == "1" ]]; then
  apptainer_cmd+=(--nv)
fi
apptainer_cmd+=(
  --bind "$INPUT_DIR":"$INPUT_DIR"
  --bind "$OUT_ROOT":"$OUT_ROOT"
  "$GNINA_SIF"
)

run_one() {
  local pdbid="$1"
  local prep_dir="$INPUT_DIR/prep/$pdbid"
  local receptor="$prep_dir/receptor_no_ligand.pdb"
  local ligand_pdbqt="$prep_dir/ligand.pdbqt"
  local box_file="$INPUT_DIR/box/$pdbid.txt"
  local out_pdbqt="$OUT_ROOT/flat/$pdbid.pdbqt"
  local log_file="$OUT_ROOT/logs/$pdbid.log"
  local tmp_dir
  tmp_dir="$(mktemp -d "${OUT_ROOT}/_tmp_${pdbid}_XXXX")"

  if [[ ! -f "$receptor" ]]; then
    echo "Missing receptor: $receptor" >&2
    return 3
  fi
  if [[ ! -f "$ligand_pdbqt" ]]; then
    echo "Missing ligand: $ligand_pdbqt" >&2
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

  local lig_sdf="$tmp_dir/ligand.sdf"
  if ! "${apptainer_cmd[@]}" obabel -ipdbqt "$ligand_pdbqt" -osdf -O "$lig_sdf" >"$tmp_dir/obabel.log" 2>&1; then
    echo "OpenBabel conversion failed for $pdbid. See: $tmp_dir/obabel.log" >&2
    return 4
  fi

  local gnina_args=(
    gnina
    -r "$receptor"
    -l "$lig_sdf"
    --center_x "$cx" --center_y "$cy" --center_z "$cz"
    --size_x "$sx" --size_y "$sy" --size_z "$sz"
    --exhaustiveness "$EXHAUSTIVENESS"
    --num_modes "$NUM_MODES"
    --cpu "$CPU"
    --cnn_scoring "$CNN_SCORING"
    --scoring "$SCORING"
    --addH 0
    --log "$log_file"
    --out "$out_pdbqt"
  )
  if [[ -n "$SEED" ]]; then
    gnina_args+=(--seed "$SEED")
  fi
  if [[ "${USE_GPU}" == "1" ]]; then
    gnina_args+=(--device "$DEVICE")
  else
    gnina_args+=(--no_gpu)
  fi

  echo "[gnina] $pdbid -> $out_pdbqt"
  if ! "${apptainer_cmd[@]}" "${gnina_args[@]}" >>"$log_file" 2>&1; then
    echo "GNINA failed for $pdbid. See: $log_file" >&2
    return 5
  fi

  if [[ ! -s "$out_pdbqt" ]]; then
    echo "GNINA did not produce an output file for $pdbid: $out_pdbqt" >&2
    echo "First 60 log lines ($log_file):" >&2
    sed -n '1,60p' "$log_file" >&2
    return 6
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
    count=$((count + 1))
  done
  echo "Shard complete. Ran $count targets."
fi

if [[ "$fail" -ne 0 ]]; then
  echo "One or more GNINA runs failed." >&2
  exit 3
fi

echo "Done. Flat outputs in: $OUT_ROOT/flat"
