#!/usr/bin/env bash
#
# Submit a Slurm array to run GNINA over curated targets on ISAAC.
#
# Inputs:
#   --input-dir should point at a directory containing:
#     box/<PDBID>.txt and prep/<PDBID>/{receptor_no_ligand.pdb,ligand.pdbqt}
#
# Output:
#   --out-root/flat/<PDBID>.pdbqt (multi-pose GNINA output per target)
#
# Before first use, pull a GNINA Apptainer image on ISAAC:
#   apptainer pull $HOME/gnina.sif docker://gnina/gnina:latest
#
# Example (ISAAC login node; GPU partition):
#   bash scripts/submit_gnina_isaac_array.sh \
#     -A acf-utk0011 -p campus-gpu --qos campus-gpu \
#     --input-dir prediction_inputs/vina_inputs \
#     --out-root /path/to/gnina_isaac_run \
#     --gnina-sif /path/to/gnina.sif \
#     --cuda-module cuda/12.2.0-binary \
#     --cpus-per-task 8 --gpus 1 \
#     --exhaustiveness 8 --num-modes 20 --seed 0
#
# Notes:
# - Use --shards to reduce submitted array tasks (each task runs ~1/N targets).
# - GNINA defaults to CNN reranking (cnn_scoring=rescore). You can change with --cnn-scoring.

set -euo pipefail

account=""
partition=""
qos=""
input_dir="prediction_inputs/vina_inputs"
out_root=""
time_limit="06:00:00"
max_parallel_jobs="all"
cpus_per_task="8"
gpus="1"
gres=""
mem=""
shards=""
dependency=""

gnina_sif=""
cuda_module="cuda/12.2.0-binary"
use_gpu="1"
device="0"
exhaustiveness="8"
num_modes="20"
seed="0"
cnn_scoring="rescore"
scoring="vina"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -A|--account) account="$2"; shift 2 ;;
    -p|--partition) partition="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --input-dir) input_dir="$2"; shift 2 ;;
    --out-root) out_root="$2"; shift 2 ;;
    -t|--time) time_limit="$2"; shift 2 ;;
    --max-parallel) max_parallel_jobs="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --gres) gres="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --shards) shards="$2"; shift 2 ;;
    --dependency) dependency="$2"; shift 2 ;;
    --gnina-sif) gnina_sif="$2"; shift 2 ;;
    --cuda-module) cuda_module="$2"; shift 2 ;;
    --no-gpu) use_gpu="0"; shift 1 ;;
    --device) device="$2"; shift 2 ;;
    --exhaustiveness) exhaustiveness="$2"; shift 2 ;;
    --num-modes) num_modes="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --cnn-scoring) cnn_scoring="$2"; shift 2 ;;
    --scoring) scoring="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 -A ACCOUNT -p PARTITION [--qos QOS] [OPTIONS]

Required:
  -A, --account ACCOUNT       Slurm account/project to charge
  -p, --partition PARTITION   Slurm partition (e.g. campus-gpu)

Inputs/outputs:
  --input-dir DIR             Directory with box/ and prep/ (default: prediction_inputs/vina_inputs)
  --out-root DIR              Output root for raw GNINA outputs (required)

Resources:
  -t, --time HH:MM:SS         Walltime per array task (default: $time_limit)
  --max-parallel N|all        Max concurrent array tasks (default: $max_parallel_jobs)
  --shards N                  Submit only N array tasks; each task runs ~1/N targets (round-robin).
  --cpus-per-task N           CPU cores per task (default: $cpus_per_task)
  --gpus N                    GPUs per task (default: $gpus; ignored if --no-gpu)
  --gres SPEC                 Slurm GRES string (e.g. gpu:v100s:1). Overrides --gpus.
  --mem MEM                   Memory per task (e.g. 16G).
  --dependency SPEC           Slurm dependency (e.g. afterok:123456).

GNINA/Apptainer:
  --gnina-sif PATH            GNINA Apptainer image (required)
  --cuda-module NAME          CUDA module to load (default: $cuda_module)
  --no-gpu                    Disable GPU (passes --no_gpu to GNINA)
  --device N                  GPU device index for GNINA (default: $device)

GNINA settings:
  --exhaustiveness N          GNINA exhaustiveness (default: $exhaustiveness)
  --num-modes N               GNINA num_modes (default: $num_modes)
  --seed N                    Seed (default: $seed; set empty to omit)
  --cnn-scoring MODE          cnn_scoring (default: $cnn_scoring; e.g. none,rescore,refinement,all)
  --scoring NAME              empirical scoring function (default: $scoring; e.g. vina,vinardo)
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
if [[ -z "$gnina_sif" ]]; then
  echo "Missing required --gnina-sif" >&2
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
if [[ -n "$dependency" ]]; then
  sbatch_args+=(--dependency="$dependency")
fi
if [[ "$use_gpu" == "1" ]]; then
  if [[ -n "$gres" ]]; then
    sbatch_args+=(--gres="$gres")
  else
    sbatch_args+=(--gpus="$gpus")
  fi
fi

echo "Submitting GNINA array:"
echo "  targets:  $input_dir ($num_targets box files)"
echo "  out_root: $out_root"
echo "  array:    $array_spec"
echo "  cpus:     $cpus_per_task"
echo "  use_gpu:  $use_gpu"
if [[ "$use_gpu" == "1" ]]; then
  if [[ -n "$gres" ]]; then
    echo "  gpus:     (gres) $gres"
  else
    echo "  gpus:     $gpus"
  fi
  echo "  cuda:     $cuda_module"
  echo "  device:   $device"
fi
echo "  gnina_sif: $gnina_sif"
echo "  gnina:     exhaustiveness=$exhaustiveness num_modes=$num_modes seed=${seed:-<unset>} cnn_scoring=$cnn_scoring scoring=$scoring"
if [[ -n "$shards" ]]; then
  echo "  shards:   $shards (each task runs ~${num_targets}/${shards} targets)"
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
job_script="$script_dir/isaac_gnina_array_job.sbatch"
if [[ ! -f "$job_script" ]]; then
  echo "Job script not found next to submit script: $job_script" >&2
  exit 2
fi

INPUT_DIR="$input_dir" OUT_ROOT="$out_root" \
GNINA_SIF="$gnina_sif" CUDA_MODULE="$cuda_module" USE_GPU="$use_gpu" DEVICE="$device" \
EXHAUSTIVENESS="$exhaustiveness" NUM_MODES="$num_modes" SEED="$seed" CPU="$cpus_per_task" \
CNN_SCORING="$cnn_scoring" SCORING="$scoring" SHARDS="$shards" \
  sbatch "${sbatch_args[@]}" "$job_script"
