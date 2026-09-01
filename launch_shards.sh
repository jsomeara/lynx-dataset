#!/usr/bin/env bash
set -euo pipefail

# launch_all_shards.sh
#
# Submit one SLURM array covering every shard in shards.jsonl with NO array concurrency throttle.
#
# Defaults are tuned for UW Hyak Klone checkpoint jobs and this dataset:
#   account:         stf
#   partition:       ckpt
#   CPUs/job:        1
#   memory/job:      3G
#   time/job:        02:00:00
#   array concurrency: unlimited (scheduler decides how many can run)
#
# Each array task looks up exactly one shard from shards.jsonl and calls
# process_shard.py. process_shard.py is resumable, so if Hyak preempts and
# requeues the task, the same task resumes its .partial.h5.
#
# Usage:
#
#   bash launch_all_shards.sh \
#     --account stf \
#     --partition ckpt
#
# Optional:
#
#   bash launch_all_shards.sh \
#     --account stf \
#     --partition ckpt \
#     --mem 3G \
#     --cpus 1 \
#     --time 02:00:00 \
# #     --fasta data/hg38.fa
#
# The generated SLURM script is left in .lynx_slurm/process_all_shards.slurm
# for inspection/reuse.

ACCOUNT="stf"
PARTITION="ckpt"
MEM="3G"
CPUS="1"
TIME_LIMIT="02:00:00"
SHARDS="shards.jsonl"
TRACKS="tracks.jsonl"
RAW_DIR="raw_data"
FASTA="data/hg38.fa"
BLACKLIST="data/hg38-blacklist.v2.bed.gz"
UNMAPPABLE="data/unmap_macro.bed"
OUTPUT_ROOT="LYNX_DATASET"
PROCESS_SCRIPT="process_shard.py"
TRACK_BATCH_SIZE="16"

LOG_DIR="slurm_logs"
SLURM_DIR=".lynx_slurm"

usage() {
  cat <<EOF
Usage: $0 [options]

Required only if defaults are wrong:
  --account ACCOUNT          Slurm account       [default: $ACCOUNT]
  --partition PARTITION      Slurm partition     [default: $PARTITION]

Resource options:
  --mem MEM                  Memory per task     [default: $MEM]
  --cpus N                   CPUs per task       [default: $CPUS]
  --time HH:MM:SS            Max Slurm runtime   [default: $TIME_LIMIT]

Dataset options:
  --shards PATH              [default: $SHARDS]
  --tracks PATH              [default: $TRACKS]
  --raw-dir PATH             [default: $RAW_DIR]
  --fasta PATH               [default: $FASTA]
  --blacklist PATH           [default: $BLACKLIST]
  --unmappable PATH          [default: $UNMAPPABLE]
  --output-root PATH         [default: $OUTPUT_ROOT]
  --process-script PATH      [default: $PROCESS_SCRIPT]
  --track-batch-size N       [default: $TRACK_BATCH_SIZE]

Other:
  --log-dir PATH             [default: $LOG_DIR]
  --dry-run                  Generate script but do not call sbatch
  -h, --help                 Show this help
EOF
}

DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --tracks) TRACKS="$2"; shift 2 ;;
    --raw-dir) RAW_DIR="$2"; shift 2 ;;
    --fasta) FASTA="$2"; shift 2 ;;
    --blacklist) BLACKLIST="$2"; shift 2 ;;
    --unmappable) UNMAPPABLE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --process-script) PROCESS_SCRIPT="$2"; shift 2 ;;
    --track-batch-size) TRACK_BATCH_SIZE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for f in "$SHARDS" "$TRACKS" "$FASTA" "$BLACKLIST" "$UNMAPPABLE" "$PROCESS_SCRIPT"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: required path does not exist: $f" >&2
    exit 2
  fi
done

if [[ ! -d "$RAW_DIR" ]]; then
  echo "ERROR: raw-data directory does not exist: $RAW_DIR" >&2
  exit 2
fi

N_SHARDS="$(grep -cve '^[[:space:]]*$' "$SHARDS")"

if [[ "$N_SHARDS" -lt 1 ]]; then
  echo "ERROR: no shards found in $SHARDS" >&2
  exit 2
fi

ARRAY_MAX=$((N_SHARDS - 1))

mkdir -p "$LOG_DIR" "$SLURM_DIR" "$OUTPUT_ROOT"

# Convert key paths to absolute paths now. Requeued jobs can then safely restart
# regardless of what directory Slurm happens to restore.
abspath() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

SHARDS_ABS="$(abspath "$SHARDS")"
TRACKS_ABS="$(abspath "$TRACKS")"
RAW_DIR_ABS="$(abspath "$RAW_DIR")"
FASTA_ABS="$(abspath "$FASTA")"
BLACKLIST_ABS="$(abspath "$BLACKLIST")"
UNMAPPABLE_ABS="$(abspath "$UNMAPPABLE")"
OUTPUT_ROOT_ABS="$(abspath "$OUTPUT_ROOT")"
PROCESS_SCRIPT_ABS="$(abspath "$PROCESS_SCRIPT")"
LOG_DIR_ABS="$(abspath "$LOG_DIR")"
WORKDIR_ABS="$(pwd -P)"

SLURM_SCRIPT="$SLURM_DIR/process_all_shards.slurm"

cat > "$SLURM_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lynx-shard
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=0-${ARRAY_MAX}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=$LOG_DIR_ABS/%x_%A_%a.out
#SBATCH --error=$LOG_DIR_ABS/%x_%A_%a.err
#SBATCH --chdir=$WORKDIR_ABS
#SBATCH --export=ALL

set -uo pipefail

SHARDS="$SHARDS_ABS"
TRACKS="$TRACKS_ABS"
RAW_DIR="$RAW_DIR_ABS"
FASTA="$FASTA_ABS"
BLACKLIST="$BLACKLIST_ABS"
UNMAPPABLE="$UNMAPPABLE_ABS"
OUTPUT_ROOT="$OUTPUT_ROOT_ABS"
PROCESS_SCRIPT="$PROCESS_SCRIPT_ABS"
TRACK_BATCH_SIZE="$TRACK_BATCH_SIZE"

echo "============================================================"
echo "LYNX shard task"
echo "============================================================"
echo "Job ID:             \${SLURM_JOB_ID:-}"
echo "Array job ID:       \${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID:      \${SLURM_ARRAY_TASK_ID:-}"
echo "Node:               \${SLURMD_NODENAME:-}"
echo "Account:            $ACCOUNT"
echo "Partition:          $PARTITION"
echo "CPUs:               $CPUS"
echo "Memory request:     $MEM"
echo "Started:            \$(date --iso-8601=seconds)"
echo

# Array task 0 reads JSONL line 1, task 1 line 2, etc.
LINE_NO=\$((SLURM_ARRAY_TASK_ID + 1))

# Parse the selected JSONL row using Python rather than jq, so there is no jq
# dependency on the compute node.
read -r SPLIT SHARD_ID < <(
  python - "\$SHARDS" "\$LINE_NO" <<'PY'
import json
import sys

path = sys.argv[1]
wanted = int(sys.argv[2])

with open(path) as f:
    for i, line in enumerate(f, 1):
        if i != wanted:
            continue
        rec = json.loads(line)
        print(rec["split"], rec["shard_id"])
        break
    else:
        raise SystemExit(f"Could not find JSONL line {wanted}")
PY
)

echo "Shard:              \$SHARD_ID"
echo "Split:              \$SPLIT"
echo

# If a previous execution already finalized this shard, process_shard.py exits
# successfully without recomputing it.
#
# Exit 99 is the explicit "checkpointed because a stop signal arrived" code
# used by process_shard.py. We explicitly ask Slurm to requeue in that case.
# Hyak's ckpt partitions also automatically requeue checkpoint jobs when they
# are preempted by resource owners or hit checkpoint churn.
set +e

uv run python "\$PROCESS_SCRIPT" \
  --split "\$SPLIT" \
  --shard-id "\$SHARD_ID" \
  --shards "\$SHARDS" \
  --tracks "\$TRACKS" \
  --raw-dir "\$RAW_DIR" \
  --fasta "\$FASTA" \
  --blacklist "\$BLACKLIST" \
  --unmappable "\$UNMAPPABLE" \
  --output-root "\$OUTPUT_ROOT" \
  --track-batch-size "\$TRACK_BATCH_SIZE"

RC=\$?

set -e

if [[ "\$RC" -eq 99 ]]; then
  echo
  echo "process_shard checkpointed cleanly after a Slurm signal."
  echo "Requesting requeue of Slurm job \${SLURM_JOB_ID}."
  scontrol requeue "\${SLURM_JOB_ID}"
  exit 0
fi

if [[ "\$RC" -ne 0 ]]; then
  echo
  echo "ERROR: process_shard exited with code \$RC."
  echo "This is treated as a genuine processing error, not an automatic retry."
  exit "\$RC"
fi

echo
echo "Finished: \$(date --iso-8601=seconds)"
EOF

chmod +x "$SLURM_SCRIPT"

echo "Generated: $SLURM_SCRIPT"
echo
echo "Configuration:"
echo "  account:          $ACCOUNT"
echo "  partition:        $PARTITION"
echo "  shards:           $N_SHARDS"
echo "  array:            0-${ARRAY_MAX} (no concurrency limit)"
echo "  CPUs/job:         $CPUS"
echo "  memory/job:       $MEM"
echo "  time/job:         $TIME_LIMIT"
echo "  output root:      $OUTPUT_ROOT_ABS"
echo "  logs:             $LOG_DIR_ABS"
echo
echo "Checkpoint behavior:"
echo "  --requeue enabled"
echo "  USR1 requested 120 s before Slurm time-limit termination"
echo "  process_shard resumes from its .partial.h5"
echo "  exit code 99 explicitly self-requeues the array task"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: not submitting."
  echo "Submit manually with:"
  echo "  sbatch $SLURM_SCRIPT"
  exit 0
fi

echo "Submitting..."
sbatch "$SLURM_SCRIPT"
