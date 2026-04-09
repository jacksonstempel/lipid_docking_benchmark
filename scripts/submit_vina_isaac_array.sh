#!/usr/bin/env bash
#
# Submit a Slurm array to run AutoDock Vina over curated targets.
#
# Inputs:
#   --input-dir should point at a directory containing:
#     box/<PDBID>.txt and prep/<PDBID>/{receptor.pdbqt,ligand.pdbqt}
#
# Output:
#   --out-root/flat/<PDBID>.pdbqt (multi-pose file per target)
#
# Example (ISAAC login node):
#   bash scripts/submit_vina_isaac_array.sh \
#     -A acf-utk0011 -p campus --qos campus \
#     --input-dir prediction_inputs/vina_inputs \
#     --out-root /path/to/vina_isaac_exh256 \
#     --shards 10 \
#     --cpus-per-task 16 \
#     --exhaustiveness 256 --num-modes 20 --seed 0
#
# Notes:
# - Use --shards to reduce the number of submitted array tasks (each task runs ~1/N targets).
# - Use --cpus-per-task to allocate cores; Vina uses --cpu to decide thread count.

set -euo pipefail

account=""
partition=""
qos=""
input_dir="prediction_inputs/vina_inputs"
out_root=""
time_limit="04:00:00"
max_parallel_jobs="all"
cpus_per_task="16"
exhaustiveness="256"
num_modes="20"
seed="0"
vina_bin=""
vina_module=""
shards=""
mem=""

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
    -t|--time)
      time_limit="$2"
      shift 2
      ;;
    --max-parallel)
      max_parallel_jobs="$2"
      shift 2
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift 2
      ;;
    --mem)
      mem="$2"
      shift 2
      ;;
    --exhaustiveness)
      exhaustiveness="$2"
      shift 2
      ;;
    --num-modes)
      num_modes="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --vina-bin)
      vina_bin="$2"
      shift 2
      ;;
    --vina-module)
      vina_module="$2"
      shift 2
      ;;
    --shards)
      shards="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 -A ACCOUNT -p PARTITION [--qos QOS] [OPTIONS]

Required:
  -A, --account ACCOUNT       Slurm account/project to charge
  -p, --partition PARTITION   Slurm partition (e.g. campus)

Options:
  --qos QOS                   Slurm QoS (if required by your partition)
  --input-dir DIR             Directory with box/ and prep/ (default: prediction_inputs/vina_inputs)
  --out-root DIR              Output root for raw Vina outputs (required)
  -t, --time HH:MM:SS         Walltime per array task (default: $time_limit)
  --max-parallel N|all        Max concurrent array tasks (default: $max_parallel_jobs)
  --shards N                  Submit only N array tasks; each task runs ~1/N targets (round-robin). Useful if you hit submit/job limits.
  --cpus-per-task N           CPU cores per task (default: $cpus_per_task)
  --mem MEM                   Memory per task (e.g. 8G). If unset, cluster default applies.

Vina settings:
  --exhaustiveness N          Vina exhaustiveness (default: $exhaustiveness)
  --num-modes N               Vina num_modes (default: $num_modes)
  --seed N                    Vina seed for reproducibility (default: $seed; set empty to omit)
  --vina-bin PATH             Use a specific vina executable (default: 'vina' on PATH)
  --vina-module NAME          Module to load before running Vina; only use when you know the exact module name
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

if [[ ! -d "$input_dir/box" || ! -d "$input_dir/prep" ]]; then
  echo "Expected input dir to contain box/ and prep/: $input_dir" >&2
  exit 2
fi

shopt -s nullglob
box_files=("$input_dir/box"/*.txt)
shopt -u nullglob
num_targets="${#box_files[@]}"
if [[ "$num_targets" -eq 0 ]]; then
  echo "No box/*.txt files found in: $input_dir" >&2
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
    array_spec="1-${num_targets}"
  else
    array_spec="1-${num_targets}%${max_parallel_jobs}"
  fi
fi

sbatch_args=(
  -A "$account"
  -p "$partition"
  --array="$array_spec"
  --time="$time_limit"
  --cpus-per-task="$cpus_per_task"
)
if [[ -n "$mem" ]]; then
  sbatch_args+=(--mem="$mem")
fi
if [[ -n "$qos" ]]; then
  sbatch_args+=(--qos="$qos")
fi

echo "Submitting Vina array:"
echo "  targets:  $input_dir ($num_targets box files)"
echo "  out_root: $out_root"
echo "  array:    $array_spec"
echo "  cpus:     $cpus_per_task (vina --cpu will match)"
echo "  vina:     exhaustiveness=$exhaustiveness num_modes=$num_modes seed=${seed:-<unset>}"
if [[ -n "$shards" ]]; then
  echo "  shards:   $shards (each task runs ~${num_targets}/${shards} targets)"
fi
if [[ -n "$vina_bin" ]]; then
  echo "  vina_bin: $vina_bin"
fi
if [[ -n "$vina_module" ]]; then
  echo "  vina_module: $vina_module"
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
job_script="$script_dir/isaac_vina_array_job.sbatch"
if [[ ! -f "$job_script" ]]; then
  echo "Job script not found next to submit script: $job_script" >&2
  exit 2
fi

INPUT_DIR="$input_dir" OUT_ROOT="$out_root" \
EXHAUSTIVENESS="$exhaustiveness" NUM_MODES="$num_modes" \
SEED="$seed" CPU="$cpus_per_task" \
VINA_BIN="$vina_bin" VINA_MODULE="$vina_module" \
SHARDS="$shards" \
  sbatch "${sbatch_args[@]}" "$job_script"
