#!/usr/bin/env bash
# Run one Boltz prediction (one YAML) as a Slurm array task.

#SBATCH -J boltz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o boltz_%A_%a.out
#SBATCH -e boltz_%A_%a.err

set -euo pipefail

INPUT_DIR="${INPUT_DIR:?INPUT_DIR not set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT not set}"
ENV_PATH="${ENV_PATH:?ENV_PATH not set}"
BOLTZ_CACHE_ROOT="${BOLTZ_CACHE:?BOLTZ_CACHE not set}"
SAMPLING_STEPS="${SAMPLING_STEPS:?SAMPLING_STEPS not set}"
RECYCLING_STEPS="${RECYCLING_STEPS:?RECYCLING_STEPS not set}"
DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:?DIFFUSION_SAMPLES not set}"
MAX_PARALLEL_SAMPLES="${MAX_PARALLEL_SAMPLES:-}"
PREPROCESSING_THREADS="${PREPROCESSING_THREADS:?PREPROCESSING_THREADS not set}"
NUM_WORKERS="${NUM_WORKERS:?NUM_WORKERS not set}"
SHARDS="${SHARDS:-}"

module load anaconda3/2021.05
source "${ANACONDA3_SH:?ANACONDA3_SH not set by anaconda module}"
conda activate "$ENV_PATH"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/flat"

echo "OUT_ROOT=$OUT_ROOT"
echo "SAMPLING_STEPS=$SAMPLING_STEPS"
echo "RECYCLING_STEPS=$RECYCLING_STEPS"
echo "DIFFUSION_SAMPLES=$DIFFUSION_SAMPLES"
echo "PREPROCESSING_THREADS=$PREPROCESSING_THREADS"
echo "NUM_WORKERS=$NUM_WORKERS"
if [[ -n "$MAX_PARALLEL_SAMPLES" ]]; then
  echo "MAX_PARALLEL_SAMPLES=$MAX_PARALLEL_SAMPLES"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"

shopt -s nullglob
yaml_glob=("$INPUT_DIR"/*.yaml)
shopt -u nullglob
if [[ "${#yaml_glob[@]}" -eq 0 ]]; then
  echo "No *.yaml files found in: $INPUT_DIR" >&2
  exit 2
fi
mapfile -t yamls < <(printf '%s\n' "${yaml_glob[@]}" | sort)

task_id="${SLURM_ARRAY_TASK_ID:-1}"
# First-principles/simple mode: use one shared cache directory for this run.
export BOLTZ_CACHE="$BOLTZ_CACHE_ROOT"
mkdir -p "$BOLTZ_CACHE"
echo "BOLTZ_CACHE=$BOLTZ_CACHE"
if [[ -n "$SHARDS" ]]; then
  if ! [[ "$SHARDS" =~ ^[0-9]+$ ]] || [[ "$SHARDS" -lt 1 ]]; then
    echo "Invalid SHARDS value: $SHARDS" >&2
    exit 2
  fi
  echo "SHARDS=$SHARDS"
fi

batch_dir="$OUT_ROOT/_batch_inputs/${SLURM_ARRAY_JOB_ID:-nojob}_${task_id}"
rm -rf "$batch_dir"
mkdir -p "$batch_dir"

assigned=0
if [[ -z "$SHARDS" ]]; then
  yaml_path="${yamls[$((task_id-1))]:-}"
  if [[ -z "${yaml_path:-}" ]]; then
    echo "No YAML found for SLURM_ARRAY_TASK_ID=${task_id} in $INPUT_DIR" >&2
    exit 2
  fi
  # Keep this dead-simple: copy inputs (no symlinks) into a per-task staging dir.
  cp -f "$yaml_path" "$batch_dir/$(basename "$yaml_path")"
  assigned=1
else
  # Round-robin assignment so N shards cover all inputs with similar load.
  for ((i=task_id-1; i<${#yamls[@]}; i+=SHARDS)); do
    yaml_path="${yamls[$i]}"
    cp -f "$yaml_path" "$batch_dir/$(basename "$yaml_path")"
    assigned=$((assigned+1))
  done
fi
echo "Batch YAMLs: $assigned"

run_root="$OUT_ROOT/_runs/${SLURM_ARRAY_JOB_ID:-nojob}_${task_id}"
rm -rf "$run_root"
mkdir -p "$run_root"

fail=0
done_count=0
total_count="$assigned"

for y in "$batch_dir"/*.yaml; do
  pdbid="$(basename "$y" .yaml)"
  echo "== [$((done_count+1))/$total_count] $pdbid =="

  one_in="$run_root/${pdbid}/in"
  one_out="$run_root/${pdbid}/out"
  rm -rf "$one_in" "$one_out"
  mkdir -p "$one_in" "$one_out"
  cp -f "$y" "$one_in/${pdbid}.yaml"

  cmd=(boltz predict "$one_in" \
    --use_msa_server \
    --no_kernels \
    --sampling_steps "$SAMPLING_STEPS" \
    --recycling_steps "$RECYCLING_STEPS" \
    --diffusion_samples "$DIFFUSION_SAMPLES" \
    --preprocessing-threads "$PREPROCESSING_THREADS" \
    --num_workers "$NUM_WORKERS" \
    --output_format mmcif \
    --out_dir "$one_out" \
    --override)
  if [[ -n "$MAX_PARALLEL_SAMPLES" ]]; then
    cmd+=(--max_parallel_samples "$MAX_PARALLEL_SAMPLES")
  fi

  if ! "${cmd[@]}"; then
    echo "Boltz failed for $pdbid" >&2
    fail=1
    done_count=$((done_count+1))
    continue
  fi

  pred_dir="$one_out/boltz_results_${pdbid}/predictions/$pdbid"
  if [[ ! -d "$pred_dir" ]]; then
    echo "Could not locate prediction folder for $pdbid under: $one_out" >&2
    fail=1
    done_count=$((done_count+1))
    continue
  fi

  # Copy "flat" artifacts immediately as each target completes.
  shopt -s nullglob
  for f in "$pred_dir/${pdbid}_model_"*.cif; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$OUT_ROOT/flat/$(basename "$f")"
  done
  for f in "$pred_dir"/confidence_"${pdbid}"_model_*.json; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$OUT_ROOT/flat/$(basename "$f")"
  done

  done_count=$((done_count+1))
done

echo "Batch complete (fail=$fail). Flat outputs in: $OUT_ROOT/flat"
exit "$fail"
