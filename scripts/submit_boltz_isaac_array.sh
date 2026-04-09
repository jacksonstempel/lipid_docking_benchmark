#!/usr/bin/env bash
#
# Submit a Slurm array to run Boltz over all YAML inputs in a directory.
#
# Example (ISAAC login node):
#   bash scripts/submit_boltz_isaac_array.sh \
#     -A acf-utk0011 -p campus-gpu --qos campus-gpu \
#     --input-dir prediction_inputs/boltz_inputs \
#     --out-root /path/to/boltz_isaac_run \
#     --env-path /path/to/envs/boltz \
#     --cache-dir /path/to/boltz_cache \
#     --max-parallel 4
#

set -euo pipefail

account=""
partition=""
qos=""
input_dir="prediction_inputs/boltz_inputs"
out_root=""
env_path=""
cache_dir=""
time_limit="04:00:00"
max_parallel_jobs="4"
gpus="1"
gres=""
cpus_per_task="8"
sampling_steps="200"
recycling_steps="3"
diffusion_samples="1"
max_parallel_samples=""
preprocessing_threads=""
num_workers="2"
shards=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -A|--account)
      account="$2"
      shift 2
      ;;
    -p|--partition)
      partition="$2"
      shift 2
      ;;
    --qos)
      qos="$2"
      shift 2
      ;;
    --input-dir)
      input_dir="$2"
      shift 2
      ;;
    --out-root)
      out_root="$2"
      shift 2
      ;;
    --env-path)
      env_path="$2"
      shift 2
      ;;
    --cache-dir)
      cache_dir="$2"
      shift 2
      ;;
    -t|--time)
      time_limit="$2"
      shift 2
      ;;
    --max-parallel)
      max_parallel_jobs="$2"
      shift 2
      ;;
    --max-parallel-all)
      max_parallel_jobs="all"
      shift 1
      ;;
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --gres)
      gres="$2"
      shift 2
      ;;
    --shards)
      shards="$2"
      shift 2
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift 2
      ;;
    --sampling-steps)
      sampling_steps="$2"
      shift 2
      ;;
    --recycling-steps)
      recycling_steps="$2"
      shift 2
      ;;
    --diffusion-samples)
      diffusion_samples="$2"
      shift 2
      ;;
    --max-parallel-samples)
      max_parallel_samples="$2"
      shift 2
      ;;
    --preprocessing-threads)
      preprocessing_threads="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 -A ACCOUNT -p PARTITION [--qos QOS] [OPTIONS]

Required:
  -A, --account ACCOUNT       Slurm account/project to charge
  -p, --partition PARTITION   Slurm partition (e.g. campus-gpu)

Options:
  --qos QOS                   Slurm QoS (if required by your partition)
  --input-dir DIR             Directory with *.yaml (default: prediction_inputs/boltz_inputs)
  --out-root DIR              Output root for raw Boltz outputs (required)
  --env-path PATH             Conda env path on ISAAC (required)
  --cache-dir DIR             Boltz cache dir on ISAAC (required)
  -t, --time HH:MM:SS         Walltime per task (default: $time_limit)
  --max-parallel N            Max concurrent array tasks (default: $max_parallel_jobs; use 'all' for no cap)
  --max-parallel-all          Alias for --max-parallel all
  --gpus N                    GPUs per task (default: $gpus). Prefer --gres for specific models.
  --gres SPEC                 Slurm GRES string (e.g. gpu:v100s:2). Overrides --gpus.
  --shards N                  Submit only N array tasks; each task runs ~1/N of YAMLs (round-robin). Useful when QoS limits submitted jobs.
  --cpus-per-task N           CPU cores per task (default: $cpus_per_task)
  --sampling-steps N          Boltz sampling steps (default: $sampling_steps)
  --recycling-steps N         Boltz recycling steps (default: $recycling_steps)
  --diffusion-samples N       Boltz diffusion samples (default: $diffusion_samples)
  --max-parallel-samples N    Boltz max_parallel_samples (default: unset)
  --preprocessing-threads N   Boltz preprocessing threads (default: --cpus-per-task)
  --num-workers N             Boltz dataloader workers (default: $num_workers)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$account" ]]; then
  echo "Missing required --account / -A" >&2
  exit 2
fi
if [[ -z "$partition" ]]; then
  echo "Missing required --partition / -p" >&2
  exit 2
fi
if [[ -z "$out_root" ]]; then
  echo "Missing required --out-root" >&2
  exit 2
fi
if [[ -z "$env_path" ]]; then
  echo "Missing required --env-path" >&2
  exit 2
fi
if [[ -z "$cache_dir" ]]; then
  echo "Missing required --cache-dir" >&2
  exit 2
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Input dir not found: $input_dir" >&2
  exit 2
fi
if [[ ! -d "$env_path" ]]; then
  echo "Conda env not found at: $env_path" >&2
  exit 2
fi

shopt -s nullglob
input_yaml_files=("$input_dir"/*.yaml)
shopt -u nullglob
num_inputs="${#input_yaml_files[@]}"
if [[ "$num_inputs" -eq 0 ]]; then
  echo "No *.yaml files found in: $input_dir" >&2
  exit 2
fi

mkdir -p "$out_root"

if [[ -n "$shards" ]]; then
  if ! [[ "$shards" =~ ^[0-9]+$ ]] || [[ "$shards" -lt 1 ]]; then
    echo "--shards must be a positive integer, got: $shards" >&2
    exit 2
  fi
  array_spec="1-${shards}"
else
  if [[ "$max_parallel_jobs" == "all" || "$max_parallel_jobs" == "0" ]]; then
    array_spec="1-${num_inputs}"
  else
    array_spec="1-${num_inputs}%${max_parallel_jobs}"
  fi
fi
sbatch_args=(
  -A "$account"
  -p "$partition"
  --array="$array_spec"
  --time="$time_limit"
  --cpus-per-task="$cpus_per_task"
)
if [[ -n "$gres" ]]; then
  sbatch_args+=(--gres="$gres")
else
  sbatch_args+=(--gpus="$gpus")
fi
if [[ -n "$qos" ]]; then
  sbatch_args+=(--qos="$qos")
fi

echo "Submitting Boltz array:"
echo "  inputs:   $input_dir ($num_inputs YAMLs)"
echo "  out_root: $out_root"
echo "  cache:    $cache_dir"
echo "  env:      $env_path"
echo "  array:    $array_spec"
echo "  cpus:     $cpus_per_task"
if [[ -n "$gres" ]]; then
  echo "  gpus:     (gres) $gres"
else
  echo "  gpus:     $gpus"
fi
echo "  boltz:    sampling_steps=$sampling_steps recycling_steps=$recycling_steps diffusion_samples=$diffusion_samples"
if [[ -n "$max_parallel_samples" ]]; then
  echo "  boltz:    max_parallel_samples=$max_parallel_samples"
fi
if [[ -z "$preprocessing_threads" ]]; then
  preprocessing_threads="$cpus_per_task"
fi
echo "  boltz:    preprocessing_threads=$preprocessing_threads num_workers=$num_workers"
if [[ -n "$shards" ]]; then
  echo "  shards:   $shards (each task runs ~${num_inputs}/${shards} YAMLs)"
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
job_script="$script_dir/isaac_boltz_array_job.sbatch"
if [[ ! -f "$job_script" ]]; then
  echo "Job script not found next to submit script: $job_script" >&2
  exit 2
fi

INPUT_DIR="$input_dir" OUT_ROOT="$out_root" ENV_PATH="$env_path" BOLTZ_CACHE="$cache_dir" \
SAMPLING_STEPS="$sampling_steps" RECYCLING_STEPS="$recycling_steps" DIFFUSION_SAMPLES="$diffusion_samples" \
MAX_PARALLEL_SAMPLES="$max_parallel_samples" \
PREPROCESSING_THREADS="$preprocessing_threads" NUM_WORKERS="$num_workers" \
SHARDS="$shards" \
  sbatch "${sbatch_args[@]}" "$job_script"
