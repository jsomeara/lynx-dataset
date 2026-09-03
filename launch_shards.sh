#!/usr/bin/env bash
set -euo pipefail

# launch_shards.sh
#
# Submit SLURM jobs for Lynx shards.
#
# DEFAULT behavior:
#   Submit only shards that are NOT already finalized successfully.
#
# This makes the script safe to run again after a large run:
#   - complete final .h5 exists -> skip
#   - .partial.h5 exists       -> submit and resume it
#   - neither exists           -> submit from scratch
#   - suspicious final .h5     -> submit; process_shard.py will fail loudly
#                                rather than overwrite it
#
# A compact repair manifest is generated containing only unfinished shards.
# The SLURM array indexes that repair manifest, so a second launch does not
# waste scheduler jobs on shards that are already complete.
#
# Usage:
#
#   bash launch_shards.sh \
#     --account stf \
#     --partition compute
#
# Dry run:
#
#   bash launch_shards.sh \
#     --account stf \
#     --partition compute \
#     --dry-run
#
# Force submission of ALL shards, including already-complete ones:
#
#   bash launch_shards.sh \
#     --account stf \
#     --partition compute \
#     --all
#
# The generated files are placed in:
#
#   .lynx_slurm/process_shards.slurm
#   .lynx_slurm/repair_shards.jsonl

ACCOUNT="stf"
PARTITION="ckpt"
MEM="3G"
CPUS="1"
TIME_LIMIT="04:00:00"

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

DRY_RUN=0
SUBMIT_ALL=0

usage() {
  cat <<EOF
Usage: $0 [options]

Scheduler:
  --account ACCOUNT          Slurm account       [default: $ACCOUNT]
  --partition PARTITION      Slurm partition     [default: $PARTITION]

Resources:
  --mem MEM                  Memory per task     [default: $MEM]
  --cpus N                   CPUs per task       [default: $CPUS]
  --time HH:MM:SS            Max runtime         [default: $TIME_LIMIT]

Dataset:
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
  --all                      Submit every shard, even if already complete
  --dry-run                  Build manifests/scripts but do not call sbatch
  -h, --help                 Show this help
EOF
}

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
    --all) SUBMIT_ALL=1; shift ;;
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

mkdir -p "$LOG_DIR" "$SLURM_DIR" "$OUTPUT_ROOT"

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

REPAIR_MANIFEST="$SLURM_DIR/repair_shards.jsonl"
SLURM_SCRIPT="$SLURM_DIR/process_shards.slurm"

# ---------------------------------------------------------------------------
# Build a manifest containing only unfinished shards.
#
# A shard is considered complete only when:
#   OUTPUT_ROOT/<split>/<shard_id>.h5
# exists, opens as HDF5, has matching shard_id, and root attr complete == 1.
#
# Anything else is repair work:
#   - partial checkpoint
#   - missing shard
#   - malformed/incomplete final file
# ---------------------------------------------------------------------------

python - \
  "$SHARDS_ABS" \
  "$OUTPUT_ROOT_ABS" \
  "$REPAIR_MANIFEST" \
  "$SUBMIT_ALL" <<'PY'
import json
import sys
from pathlib import Path

import h5py

shards_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
repair_path = Path(sys.argv[3])
submit_all = bool(int(sys.argv[4]))

total = 0
complete = 0
partial = 0
missing = 0
bad_final = 0
selected = []

with shards_path.open() as f:
    for original_index, line in enumerate(f):
        if not line.strip():
            continue

        rec = json.loads(line)
        total += 1

        split = rec["split"]
        shard_id = rec["shard_id"]

        final_path = output_root / split / f"{shard_id}.h5"
        partial_path = output_root / split / f".{shard_id}.partial.h5"

        is_complete = False
        final_problem = None

        if final_path.exists():
            try:
                with h5py.File(final_path, "r") as h5:
                    is_complete = (
                        int(h5.attrs.get("complete", 0)) == 1
                        and str(h5.attrs.get("shard_id", "")) == shard_id
                    )

                if not is_complete:
                    final_problem = "final_exists_but_not_marked_complete"

            except Exception as e:
                final_problem = f"final_unreadable:{type(e).__name__}"

        if is_complete:
            complete += 1
            status = "complete"

        elif partial_path.exists():
            partial += 1
            status = "partial"

        elif final_path.exists():
            bad_final += 1
            status = final_problem or "bad_final"

        else:
            missing += 1
            status = "missing"

        if submit_all or not is_complete:
            selected.append({
                "original_index": original_index,
                "split": split,
                "shard_id": shard_id,
                "status": status,
            })

repair_path.parent.mkdir(parents=True, exist_ok=True)

with repair_path.open("w") as out:
    for rec in selected:
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")

print("=" * 72)
print("SHARD STATUS")
print("=" * 72)
print(f"total shards:          {total}")
print(f"complete:              {complete}")
print(f"partial checkpoints:   {partial}")
print(f"missing outputs:       {missing}")
print(f"suspicious finals:     {bad_final}")
print(f"selected for submit:   {len(selected)}")
print(f"repair manifest:       {repair_path}")
PY

N_SELECTED="$(grep -cve '^[[:space:]]*$' "$REPAIR_MANIFEST" || true)"

if [[ "$N_SELECTED" -eq 0 ]]; then
  echo
  echo "All shards are already complete. Nothing to submit."
  exit 0
fi

ARRAY_MAX=$((N_SELECTED - 1))

REPAIR_MANIFEST_ABS="$(abspath "$REPAIR_MANIFEST")"

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
#SBATCH --signal=B:USR1@300
#SBATCH --output=$LOG_DIR_ABS/%x_%A_%a.out
#SBATCH --error=$LOG_DIR_ABS/%x_%A_%a.err
#SBATCH --chdir=$WORKDIR_ABS
#SBATCH --export=ALL

set -uo pipefail

REPAIR_MANIFEST="$REPAIR_MANIFEST_ABS"
SHARDS="$SHARDS_ABS"
TRACKS="$TRACKS_ABS"
RAW_DIR="$RAW_DIR_ABS"
FASTA="$FASTA_ABS"
BLACKLIST="$BLACKLIST_ABS"
UNMAPPABLE="$UNMAPPABLE_ABS"
OUTPUT_ROOT="$OUTPUT_ROOT_ABS"
PROCESS_SCRIPT="$PROCESS_SCRIPT_ABS"
TRACK_BATCH_SIZE="$TRACK_BATCH_SIZE"

LINE_NO=\$((SLURM_ARRAY_TASK_ID + 1))

read -r SPLIT SHARD_ID STATUS ORIGINAL_INDEX < <(
  python - "\$REPAIR_MANIFEST" "\$LINE_NO" <<'PY'
import json
import sys

path = sys.argv[1]
wanted = int(sys.argv[2])

with open(path) as f:
    for i, line in enumerate(f, 1):
        if i != wanted:
            continue

        rec = json.loads(line)
        print(
            rec["split"],
            rec["shard_id"],
            rec["status"],
            rec["original_index"],
        )
        break
    else:
        raise SystemExit(f"Could not find repair-manifest line {wanted}")
PY
)

echo "============================================================"
echo "LYNX shard task"
echo "============================================================"
echo "Job ID:             \${SLURM_JOB_ID:-}"
echo "Array job ID:       \${SLURM_ARRAY_JOB_ID:-}"
echo "Repair array ID:    \${SLURM_ARRAY_TASK_ID:-}"
echo "Original shard idx: \$ORIGINAL_INDEX"
echo "Shard:              \$SHARD_ID"
echo "Split:              \$SPLIT"
echo "Prior status:       \$STATUS"
echo "Node:               \${SLURMD_NODENAME:-}"
echo "Account:            $ACCOUNT"
echo "Partition:          $PARTITION"
echo "CPUs:               $CPUS"
echo "Memory request:     $MEM"
echo "Started:            \$(date --iso-8601=seconds)"
echo

# IMPORTANT:
# --signal=B:USR1@300 sends the early-warning signal to THIS batch shell.
# process_shard.py runs as a child process, so the shell must explicitly
# forward USR1/TERM/INT to Python.  The previous launcher did not do that:
# the shell itself was interrupted before it could inspect Python's exit code.
#
# Use the project's venv Python directly so CHILD_PID is the actual Python
# process (rather than a uv wrapper process).
PYTHON_BIN="$WORKDIR_ABS/.venv/bin/python"

if [[ ! -x "\$PYTHON_BIN" ]]; then
  echo "ERROR: expected virtualenv Python not found: \$PYTHON_BIN" >&2
  exit 2
fi

CHILD_PID=""

forward_signal() {
  local sig="\$1"

  echo
  echo "[launcher] received SIG\$sig at \$(date --iso-8601=seconds)."

  if [[ -n "\${CHILD_PID:-}" ]] && kill -0 "\$CHILD_PID" 2>/dev/null; then
    echo "[launcher] forwarding SIG\$sig to process_shard.py PID \$CHILD_PID."
    kill "-\$sig" "\$CHILD_PID" 2>/dev/null || true
  else
    echo "[launcher] process_shard.py is no longer running."
  fi
}

trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

set +e

"\$PYTHON_BIN" "\$PROCESS_SCRIPT" \
  --split "\$SPLIT" \
  --shard-id "\$SHARD_ID" \
  --shards "\$SHARDS" \
  --tracks "\$TRACKS" \
  --raw-dir "\$RAW_DIR" \
  --fasta "\$FASTA" \
  --blacklist "\$BLACKLIST" \
  --unmappable "\$UNMAPPABLE" \
  --output-root "\$OUTPUT_ROOT" \
  --track-batch-size "\$TRACK_BATCH_SIZE" &

CHILD_PID=\$!

# A trapped signal interrupts bash's wait builtin.  Do not mistake that for
# process_shard.py exiting. Keep waiting until the Python child really exits.
while true; do
  wait "\$CHILD_PID"
  RC=\$?

  if kill -0 "\$CHILD_PID" 2>/dev/null; then
    echo "[launcher] wait was interrupted by a signal; Python is still alive."
    continue
  fi

  break
done

set -e

if [[ "\$RC" -eq 99 ]]; then
  echo
  echo "[launcher] process_shard.py checkpointed cleanly and exited 99."

  if [[ -n "\${SLURM_ARRAY_JOB_ID:-}" && -n "\${SLURM_ARRAY_TASK_ID:-}" ]]; then
    REQUEUE_TARGET="\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
  else
    REQUEUE_TARGET="\${SLURM_JOB_ID}"
  fi

  echo "[launcher] explicitly requeueing \$REQUEUE_TARGET."

  if ! scontrol requeue "\$REQUEUE_TARGET"; then
    echo "ERROR: scontrol requeue failed for \$REQUEUE_TARGET." >&2
    exit 1
  fi

  # This execution is done. Slurm will start the same array task again.
  exit 0
fi

if [[ "\$RC" -ne 0 ]]; then
  echo
  echo "ERROR: process_shard exited with code \$RC."
  echo "This is treated as a genuine processing error."
  exit "\$RC"
fi

echo
echo "Finished: \$(date --iso-8601=seconds)"
EOF

chmod +x "$SLURM_SCRIPT"

echo
echo "Generated: $SLURM_SCRIPT"
echo
echo "Configuration:"
echo "  account:          $ACCOUNT"
echo "  partition:        $PARTITION"
echo "  repair shards:    $N_SELECTED"
echo "  array:            0-${ARRAY_MAX} (no concurrency limit)"
echo "  CPUs/job:         $CPUS"
echo "  memory/job:       $MEM"
echo "  time/job:         $TIME_LIMIT"
echo "  output root:      $OUTPUT_ROOT_ABS"
echo "  logs:             $LOG_DIR_ABS"
echo
echo "Rerun behavior:"
echo "  complete .h5 files are skipped before submission"
echo "  .partial.h5 files are submitted and resumed"
echo "  missing outputs are submitted from scratch"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: not submitting."
  echo "Submit manually with:"
  echo "  sbatch $SLURM_SCRIPT"
  exit 0
fi

echo "Submitting..."
sbatch "$SLURM_SCRIPT"
