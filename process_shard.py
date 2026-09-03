#!/usr/bin/env python3
"""
process_shard.py

Process ONE shard from shards.jsonl into a lightly gzip-compressed HDF5 file:

    LYNX_DATASET/train/train-00000.h5
    LYNX_DATASET/val/val-00000.h5
    LYNX_DATASET/test/test-00000.h5

Designed for SLURM:
  - Uses a persistent .partial.h5 checkpoint.
  - Marks completed track batches inside the partial HDF5.
  - Re-running the same command resumes from completed batches.
  - Handles SIGTERM/SIGUSR1/SIGINT by finishing the current batch, closing
    the HDF5 cleanly, then exiting with code 99.
  - Final output is created only by atomic rename after all batches finish.
  - Existing partial checkpoints are write-probed in a child process.
    Structurally corrupted partials are quarantined and rebuilt automatically.

Preprocessing
-------------
For every track:
  1. Read 1-bp BigWig signal.
  2. Sparse BigWig NaN -> 0.
  3. If a chromosome is entirely absent from a sparse BigWig -> all-zero signal.
  4. If a chromosome exists but is too short for a requested interval -> ERROR.
  5. Normalize the track to target total 100,000,000 using BigWig header sumData:
         factor = 100_000_000 / sumData
  6. If sum_stat == "sum_sqrt", apply x^0.75.
     sum / mean tracks remain linear.
  7. No log1p.
  8. No Borzoi 32-bp soft/hard clipping.
  9. No Borzoi manifest scale factor.
 10. Blacklist/unmappable positions -> target 0 and valid_mask=0.
 11. Before float16 storage, cap values to float16 max (65504).
     The script records per-track and total cap counts in the HDF5.

HDF5 datasets
-------------
  sequence       uint8   [windows, bp]
                 A=0 C=1 G=2 T=3 other/N=4

  targets        float16 [windows, bp, tracks]

  valid_mask     uint8   [windows, bp]

  chrom          string  [windows]
  start          int64   [windows]
  end            int64   [windows]

  track_id       string  [tracks]
  assay          string  [tracks]
  provider       string  [tracks]
  sum_stat       string  [tracks]
  norm_factor    float64 [tracks]
  capped_values  uint64  [tracks]

  batch_complete uint8   [num_batches]
                 internal resume/checkpoint state

Compression
-----------
  gzip level 1 + shuffle

Example
-------
First train shard:

  uv run python process_shard.py \
    --split train \
    --shard-number 1 \
    --fasta data/hg38.fa

If preempted, run the exact same command again. It resumes from the existing
.partial.h5 checkpoint.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm


DEFAULT_TARGET_TOTAL = 100_000_000.0
DEFAULT_OUTPUT_ROOT = Path("LYNX_DATASET")
F16_MAX = float(np.finfo(np.float16).max)

STOP_REQUESTED = False
STOP_SIGNAL = None


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _request_stop(signum, frame):
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum

    name = signal.Signals(signum).name
    print(
        f"\n[checkpoint] received {name}; "
        "will stop cleanly after the current track batch.",
        flush=True,
    )


def install_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_stop)

    # SLURM setups commonly use SIGUSR1 as an early-warning signal.
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def safe_filename(track_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", track_id).strip("_")


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def read_tracks(path: Path) -> list[dict]:
    tracks = []

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                tracks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path}:{line_no}: invalid JSON: {e}"
                ) from e

    if not tracks:
        raise RuntimeError(f"No tracks found in {path}")

    return tracks


def read_shard(
    shards_path: Path,
    split: str,
    shard_number: int | None,
    shard_id: str | None,
) -> dict:
    if shard_id is None:
        if shard_number is None:
            raise ValueError("Provide --shard-number or --shard-id")

        if shard_number < 1:
            raise ValueError("--shard-number is 1-based and must be >= 1")

        shard_id = f"{split}-{shard_number - 1:05d}"

    with shards_path.open() as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("shard_id") == shard_id:
                if record.get("split") != split:
                    raise RuntimeError(
                        f"{shard_id} belongs to split={record.get('split')}, "
                        f"not requested split={split}"
                    )
                return record

    raise RuntimeError(f"Shard {shard_id!r} not found in {shards_path}")


# ---------------------------------------------------------------------------
# Genomic masks
# ---------------------------------------------------------------------------

def read_bed(path: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}

    with open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue

            fields = line.rstrip().split()

            if len(fields) < 3:
                raise RuntimeError(f"{path}:{line_no}: expected BED3+")

            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            if end > start:
                intervals.setdefault(chrom, []).append((start, end))

    merged = {}

    for chrom, xs in intervals.items():
        xs.sort()
        acc = []

        for start, end in xs:
            if not acc or start > acc[-1][1]:
                acc.append([start, end])
            else:
                acc[-1][1] = max(acc[-1][1], end)

        merged[chrom] = [(s, e) for s, e in acc]

    return merged


def make_bad_mask(
    chrom: str,
    start: int,
    end: int,
    blacklist,
    unmappable,
) -> np.ndarray:
    bad = np.zeros(end - start, dtype=bool)

    for source in (blacklist, unmappable):
        for s, e in source.get(chrom, []):
            if e <= start:
                continue
            if s >= end:
                break

            left = max(start, s) - start
            right = min(end, e) - start

            if right > left:
                bad[left:right] = True

    return bad


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

def encode_dna(sequence: str) -> np.ndarray:
    raw = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)

    table = np.full(256, 4, dtype=np.uint8)
    table[ord("A")] = 0
    table[ord("C")] = 1
    table[ord("G")] = 2
    table[ord("T")] = 3

    return table[raw]


# ---------------------------------------------------------------------------
# Track preprocessing
# ---------------------------------------------------------------------------

def track_norm_factor(bw, target_total: float) -> float:
    """
    Fast per-track library-depth normalization using BigWig header sumData.
    """
    header = bw.header()
    total = float(header.get("sumData", 0.0) or 0.0)

    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"Invalid BigWig sumData={total}")

    return target_total / total


def transformed_values(
    bw,
    chrom: str,
    start: int,
    end: int,
    sum_stat: str,
    norm_factor: float,
    bad_mask: np.ndarray,
    track_id: str,
) -> np.ndarray:
    """
    Return one processed 1-bp window as float32.

    Critical bounds behavior:
      - chromosome entirely absent -> zeros
      - chromosome present but requested end > BigWig chromosome size -> ERROR
    """
    chroms = bw.chroms()

    if chrom not in chroms:
        # Verified sparse/subset BigWigs can omit chromosomes entirely.
        values = np.zeros(end - start, dtype=np.float32)

    else:
        bw_chrom_size = int(chroms[chrom])

        if start < 0 or end <= start or end > bw_chrom_size:
            raise RuntimeError(
                f"BigWig interval mismatch for track={track_id}: "
                f"requested {chrom}:{start}-{end}, "
                f"BigWig chromosome length={bw_chrom_size}. "
                "Chromosome exists, so this is not treated as sparse zero."
            )

        values = np.asarray(
            bw.values(chrom, start, end, numpy=True),
            dtype=np.float32,
        )

        if values.size != end - start:
            raise RuntimeError(
                f"BigWig returned {values.size} values for "
                f"{track_id} {chrom}:{start}-{end}; expected {end-start}"
            )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    values *= np.float32(norm_factor)

    if sum_stat == "sum_sqrt":
        values = np.power(
            np.maximum(values, 0.0),
            np.float32(0.75),
        ).astype(np.float32, copy=False)

    elif sum_stat in {"sum", "mean"}:
        pass

    else:
        raise RuntimeError(
            f"Unsupported sum_stat={sum_stat!r} for track={track_id}"
        )

    values[bad_mask] = 0.0

    return values


# ---------------------------------------------------------------------------
# HDF5 initialization / resume
# ---------------------------------------------------------------------------

def expected_track_ids(tracks: list[dict]) -> list[str]:
    return [str(t["id"]) for t in tracks]


def decode_h5_strings(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def initialize_partial_h5(
    partial_path: Path,
    shard_id: str,
    split: str,
    windows: list[dict],
    tracks: list[dict],
    sequence: np.ndarray,
    valid_mask: np.ndarray,
    target_total: float,
    track_batch_size: int,
):
    n_windows, seq_len = sequence.shape
    n_tracks = len(tracks)
    n_batches = math.ceil(n_tracks / track_batch_size)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(partial_path, "w") as h5:
        h5.attrs["format_version"] = 2
        h5.attrs["shard_id"] = shard_id
        h5.attrs["split"] = split
        h5.attrs["target_total"] = float(target_total)
        h5.attrs["track_batch_size"] = int(track_batch_size)
        h5.attrs["nan_policy"] = "NaN->0"
        h5.attrs["absent_chrom_policy"] = "entirely absent chromosome->0"
        h5.attrs["short_chrom_policy"] = "error"
        h5.attrs["mask_policy"] = "blacklist+unmappable target=0 valid_mask=0"
        h5.attrs["power_transform"] = "x^0.75 only when sum_stat=sum_sqrt"
        h5.attrs["clip_policy"] = "float16 storage cap only"
        h5.attrs["float16_cap"] = F16_MAX
        h5.attrs["borzoi_scale_policy"] = "not applied"
        h5.attrs["complete"] = 0

        h5.create_dataset(
            "sequence",
            data=sequence,
            dtype=np.uint8,
            chunks=(1, min(seq_len, 65536)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        h5.create_dataset(
            "valid_mask",
            data=valid_mask,
            dtype=np.uint8,
            chunks=(1, min(seq_len, 65536)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        h5.create_dataset(
            "chrom",
            data=np.asarray([w["chrom"] for w in windows], dtype=object),
            dtype=str_dtype,
        )

        h5.create_dataset(
            "start",
            data=np.asarray([w["start"] for w in windows], dtype=np.int64),
        )

        h5.create_dataset(
            "end",
            data=np.asarray([w["end"] for w in windows], dtype=np.int64),
        )

        h5.create_dataset(
            "track_id",
            data=np.asarray(expected_track_ids(tracks), dtype=object),
            dtype=str_dtype,
        )

        h5.create_dataset(
            "assay",
            data=np.asarray(
                [str(t.get("assay", "")) for t in tracks],
                dtype=object,
            ),
            dtype=str_dtype,
        )

        h5.create_dataset(
            "provider",
            data=np.asarray(
                [str(t.get("provider", "")) for t in tracks],
                dtype=object,
            ),
            dtype=str_dtype,
        )

        h5.create_dataset(
            "sum_stat",
            data=np.asarray(
                [
                    str((t.get("preprocessing") or {}).get("sum_stat", ""))
                    for t in tracks
                ],
                dtype=object,
            ),
            dtype=str_dtype,
        )

        h5.create_dataset(
            "norm_factor",
            shape=(n_tracks,),
            dtype=np.float64,
            fillvalue=np.nan,
        )

        h5.create_dataset(
            "capped_values",
            shape=(n_tracks,),
            dtype=np.uint64,
            fillvalue=0,
        )

        h5.create_dataset(
            "batch_complete",
            shape=(n_batches,),
            dtype=np.uint8,
            fillvalue=0,
        )

        h5.create_dataset(
            "targets",
            shape=(n_windows, seq_len, n_tracks),
            dtype=np.float16,
            chunks=(
                1,
                min(seq_len, 4096),
                min(track_batch_size, n_tracks),
            ),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fillvalue=np.float16(0.0),
        )

        h5.flush()


def validate_partial_h5(
    partial_path: Path,
    shard_id: str,
    split: str,
    windows: list[dict],
    tracks: list[dict],
    track_batch_size: int,
):
    """
    Fail loudly if an existing checkpoint does not match this invocation.
    """
    try:
        with h5py.File(partial_path, "r") as h5:
            if str(h5.attrs.get("shard_id")) != shard_id:
                raise RuntimeError("checkpoint shard_id mismatch")

            if str(h5.attrs.get("split")) != split:
                raise RuntimeError("checkpoint split mismatch")

            if int(h5.attrs.get("track_batch_size")) != track_batch_size:
                raise RuntimeError(
                    "checkpoint track_batch_size differs from current "
                    "--track-batch-size; resume with the same value"
                )

            if h5["targets"].shape[0] != len(windows):
                raise RuntimeError("checkpoint window count mismatch")

            if h5["targets"].shape[2] != len(tracks):
                raise RuntimeError("checkpoint track count mismatch")

            ids = decode_h5_strings(h5["track_id"][:])

            if ids != expected_track_ids(tracks):
                raise RuntimeError("checkpoint track ordering mismatch")

    except OSError as e:
        raise RuntimeError(
            f"Existing checkpoint {partial_path} cannot be opened. "
            "It may have been corrupted by a hard kill. "
            "Move/delete it explicitly before retrying; it will NOT be "
            f"overwritten automatically. Original error: {e}"
        ) from e


def probe_partial_h5_writeability(partial_path: Path) -> tuple[bool, str]:
    """Probe an existing checkpoint's next HDF5 write in a child process."""
    probe_code = """
import sys
import h5py
import numpy as np

path = sys.argv[1]
with h5py.File(path, 'r+') as h5:
    bc = np.asarray(h5['batch_complete'][:], dtype=np.uint8)
    incomplete = np.flatnonzero(bc == 0)
    batch_size = int(h5.attrs['track_batch_size'])
    n_tracks = h5['targets'].shape[2]
    if incomplete.size:
        batch_index = int(incomplete[0])
    else:
        batch_index = max(0, len(bc) - 1)
    t0 = batch_index * batch_size
    t1 = min(n_tracks, t0 + batch_size)
    if t0 >= n_tracks:
        raise RuntimeError('checkpoint probe computed invalid track range')
    bp1 = min(4096, h5['targets'].shape[1])
    block = h5['targets'][0, :bp1, t0:t1]
    h5['targets'][0, :bp1, t0:t1] = block
    h5.flush()
"""
    try:
        result = subprocess.run(
            [sys.executable, '-c', probe_code, str(partial_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        return False, f'checkpoint write probe timed out after {e.timeout}s'

    if result.returncode == 0:
        return True, ''

    detail = (result.stderr or result.stdout or '').strip()
    if len(detail) > 4000:
        detail = detail[-4000:]
    return False, (
        f'checkpoint write probe failed with exit code {result.returncode}'
        + (f'\n{detail}' if detail else '')
    )


def quarantine_corrupt_partial(partial_path: Path) -> Path:
    """Move a bad checkpoint aside so a fresh one can be created safely."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = partial_path.name
    if base.endswith('.partial.h5'):
        base = base[:-len('.partial.h5')]
    candidate = partial_path.with_name(f'{base}.corrupt-{stamp}.h5')
    suffix = 1
    while candidate.exists():
        candidate = partial_path.with_name(f'{base}.corrupt-{stamp}-{suffix}.h5')
        suffix += 1
    partial_path.replace(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    install_signal_handlers()

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
    )

    ap.add_argument(
        "--shard-number",
        type=int,
        default=None,
        help="1-based: 1 -> train-00000",
    )

    ap.add_argument(
        "--shard-id",
        default=None,
        help="Exact id, e.g. train-00000; overrides --shard-number",
    )

    ap.add_argument("--shards", type=Path, default=Path("shards.jsonl"))
    ap.add_argument("--tracks", type=Path, default=Path("tracks.jsonl"))
    ap.add_argument("--raw-dir", type=Path, default=Path("raw_data"))

    ap.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="hg38 FASTA, e.g. data/hg38.fa",
    )

    ap.add_argument(
        "--blacklist",
        type=Path,
        default=Path("data/hg38-blacklist.v2.bed.gz"),
    )

    ap.add_argument(
        "--unmappable",
        type=Path,
        default=Path("data/unmap_macro.bed"),
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    ap.add_argument(
        "--target-total",
        type=float,
        default=DEFAULT_TARGET_TOTAL,
    )

    ap.add_argument(
        "--track-batch-size",
        type=int,
        default=16,
        help=(
            "Checkpoint granularity. A completed batch is never recomputed "
            "after preemption. Keep the same value when resuming."
        ),
    )

    args = ap.parse_args()

    if args.track_batch_size < 1:
        raise ValueError("--track-batch-size must be >= 1")

    shard = read_shard(
        args.shards,
        args.split,
        args.shard_number,
        args.shard_id,
    )

    shard_id = shard["shard_id"]
    windows = shard["windows"]

    if not windows:
        raise RuntimeError(f"{shard_id} has no windows")

    lengths = {int(w["end"]) - int(w["start"]) for w in windows}

    if len(lengths) != 1:
        raise RuntimeError(f"Mixed window lengths: {sorted(lengths)}")

    seq_len = lengths.pop()

    tracks = read_tracks(args.tracks)
    n_windows = len(windows)
    n_tracks = len(tracks)
    n_batches = math.ceil(n_tracks / args.track_batch_size)

    track_paths = []

    for track in tracks:
        path = args.raw_dir / f"{safe_filename(str(track['id']))}.bw"

        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing BigWig for track {track['id']}: {path}"
            )

        track_paths.append(path)

    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{shard_id}.h5"
    partial_path = output_dir / f".{shard_id}.partial.h5"

    print("=" * 76)
    print("PROCESS SHARD")
    print("=" * 76)
    print(f"shard:             {shard_id}")
    print(f"split:             {args.split}")
    print(f"windows:           {n_windows}")
    print(f"window length:     {seq_len:,}")
    print(f"tracks:            {n_tracks}")
    print(f"batches:           {n_batches}")
    print(f"track batch size:  {args.track_batch_size}")
    print(f"float16 cap:       {F16_MAX:g}")
    print(f"output:            {output_path}")
    print(f"checkpoint:        {partial_path}")
    print()

    # If final output exists, do not silently overwrite it.
    if output_path.exists():
        try:
            with h5py.File(output_path, "r") as h5:
                complete = int(h5.attrs.get("complete", 0))
                same_shard = str(h5.attrs.get("shard_id")) == shard_id

            if complete == 1 and same_shard:
                print("Final shard already exists and is marked complete.")
                return
        except OSError:
            pass

        raise RuntimeError(
            f"Final output already exists but is not a verified complete shard: "
            f"{output_path}. Refusing to overwrite."
        )

    blacklist = read_bed(args.blacklist)
    unmappable = read_bed(args.unmappable)

    # -------------------------------------------------------
    # Validate/probe an existing checkpoint before resuming.
    # -------------------------------------------------------

    if partial_path.exists():
        print("Existing checkpoint found; validating for resume...")
        validate_partial_h5(
            partial_path,
            shard_id,
            args.split,
            windows,
            tracks,
            args.track_batch_size,
        )
        print("Checkpoint metadata is compatible; probing HDF5 writeability...")
        probe_ok, probe_detail = probe_partial_h5_writeability(partial_path)

        if not probe_ok:
            print()
            print("[checkpoint] existing partial is NOT safely writable.")
            print("[checkpoint] likely damaged by an earlier hard kill or filesystem write failure.")
            if probe_detail:
                print("[checkpoint] probe diagnostic:")
                for line in probe_detail.splitlines():
                    print(f"    {line}")
            quarantined = quarantine_corrupt_partial(partial_path)
            print(f"[checkpoint] quarantined corrupt file as: {quarantined}")
            print("[checkpoint] rebuilding this shard from scratch.")
            print()
        else:
            print("Checkpoint is compatible and writeable.")

    # -------------------------------------------------------
    # Create checkpoint if needed.
    # -------------------------------------------------------

    if not partial_path.exists():
        print("Creating new checkpoint: reading DNA + masks...")

        fasta = Fasta(
            str(args.fasta),
            as_raw=True,
            sequence_always_upper=True,
        )

        sequence = np.empty((n_windows, seq_len), dtype=np.uint8)
        valid_mask = np.empty((n_windows, seq_len), dtype=np.uint8)

        for wi, w in enumerate(tqdm(windows, unit="window")):
            chrom = w["chrom"]
            start = int(w["start"])
            end = int(w["end"])

            seq = str(fasta[chrom][start:end])

            if len(seq) != seq_len:
                raise RuntimeError(
                    f"{chrom}:{start}-{end}: FASTA returned {len(seq)} bp; "
                    f"expected {seq_len}"
                )

            sequence[wi] = encode_dna(seq)

            bad = make_bad_mask(
                chrom,
                start,
                end,
                blacklist,
                unmappable,
            )

            valid_mask[wi] = (~bad).astype(np.uint8)

        fasta.close()

        initialize_partial_h5(
            partial_path=partial_path,
            shard_id=shard_id,
            split=args.split,
            windows=windows,
            tracks=tracks,
            sequence=sequence,
            valid_mask=valid_mask,
            target_total=args.target_total,
            track_batch_size=args.track_batch_size,
        )

        del sequence
        del valid_mask

        print("Checkpoint created.")

    # Existing healthy checkpoints were already validated/probed above.

    # Load the mask once into RAM.
    with h5py.File(partial_path, "r") as h5:
        valid_mask = h5["valid_mask"][:]
        batch_complete = h5["batch_complete"][:]

    already_done = int(batch_complete.sum())

    print(
        f"Resume status: {already_done}/{n_batches} batches already complete."
    )

    cumulative_caps = 0
    run_start = time.time()

    # -------------------------------------------------------
    # Process checkpointed batches.
    # -------------------------------------------------------

    for batch_index, t0 in enumerate(
        range(0, n_tracks, args.track_batch_size)
    ):
        t1 = min(n_tracks, t0 + args.track_batch_size)

        if batch_complete[batch_index]:
            print(
                f"batch {batch_index + 1}/{n_batches}: "
                f"tracks {t0}-{t1 - 1} [checkpointed; skip]"
            )
            continue

        print(
            f"batch {batch_index + 1}/{n_batches}: "
            f"tracks {t0}-{t1 - 1}"
        )

        batch_n = t1 - t0

        batch = np.empty(
            (n_windows, seq_len, batch_n),
            dtype=np.float32,
        )

        batch_factors = np.empty(batch_n, dtype=np.float64)
        batch_caps = np.zeros(batch_n, dtype=np.uint64)

        for local_ti in tqdm(
            range(batch_n),
            unit="track",
            leave=False,
        ):
            track = tracks[t0 + local_ti]
            path = track_paths[t0 + local_ti]
            track_id = str(track["id"])

            bw = pyBigWig.open(str(path))

            if bw is None:
                raise RuntimeError(f"Could not open {path}")

            try:
                factor = track_norm_factor(
                    bw,
                    args.target_total,
                )

                batch_factors[local_ti] = factor

                sum_stat = (
                    track.get("preprocessing") or {}
                ).get("sum_stat")

                for wi, w in enumerate(windows):
                    chrom = w["chrom"]
                    start = int(w["start"])
                    end = int(w["end"])

                    bad = valid_mask[wi] == 0

                    batch[wi, :, local_ti] = transformed_values(
                        bw=bw,
                        chrom=chrom,
                        start=start,
                        end=end,
                        sum_stat=sum_stat,
                        norm_factor=factor,
                        bad_mask=bad,
                        track_id=track_id,
                    )

            finally:
                bw.close()

            # Cap count is per track for transparency.
            cap_count = int(
                np.count_nonzero(
                    batch[:, :, local_ti] > F16_MAX
                )
            )

            batch_caps[local_ti] = cap_count

        batch_min = float(np.min(batch))
        batch_max = float(np.max(batch))
        batch_cap_count = int(batch_caps.sum())

        print(
            f"  batch min/max: {batch_min:.6g} / {batch_max:.6g} | "
            f"values capped for float16: {batch_cap_count:,}"
        )

        # Explicit storage cap. This is intentionally NOT biological clipping.
        np.minimum(
            batch,
            np.float32(F16_MAX),
            out=batch,
        )

        # Guard against anything else becoming non-finite.
        if not np.isfinite(batch).all():
            raise RuntimeError(
                f"Non-finite values remain in batch {batch_index + 1} "
                "after preprocessing/capping."
            )

        batch_f16 = batch.astype(np.float16)

        # ---------------------------------------------------
        # Checkpoint commit.
        #
        # Open HDF5 only for the commit, flush, and close before continuing.
        # batch_complete is written LAST, so a resumed job never considers a
        # partially committed batch complete.
        # ---------------------------------------------------

        with h5py.File(partial_path, "r+") as h5:
            h5["targets"][:, :, t0:t1] = batch_f16
            h5["norm_factor"][t0:t1] = batch_factors
            h5["capped_values"][t0:t1] = batch_caps

            h5.flush()

            # Commit marker LAST.
            h5["batch_complete"][batch_index] = np.uint8(1)
            h5.attrs["last_completed_batch"] = int(batch_index)
            h5.attrs["total_capped_values"] = int(
                h5["capped_values"][:].sum()
            )
            h5.flush()

        batch_complete[batch_index] = 1
        cumulative_caps += batch_cap_count

        del batch
        del batch_f16

        if STOP_REQUESTED:
            sig_name = (
                signal.Signals(STOP_SIGNAL).name
                if STOP_SIGNAL is not None
                else "stop request"
            )

            print(
                f"\n[checkpoint] stopping after completed batch "
                f"{batch_index + 1}/{n_batches} due to {sig_name}.",
                flush=True,
            )
            print(
                "Re-run the same command to resume.",
                flush=True,
            )
            sys.exit(99)

    # -------------------------------------------------------
    # Final verification + atomic finalize.
    # -------------------------------------------------------

    with h5py.File(partial_path, "r+") as h5:
        complete = h5["batch_complete"][:]

        if not np.all(complete == 1):
            missing = np.flatnonzero(complete == 0).tolist()
            raise RuntimeError(
                f"Internal error: not all batches are complete: {missing}"
            )

        total_caps = int(h5["capped_values"][:].sum())
        h5.attrs["total_capped_values"] = total_caps
        h5.attrs["complete"] = 1
        h5.attrs["completed_unix_time"] = time.time()
        h5.flush()

    # Atomic rename within the same directory/filesystem.
    partial_path.replace(output_path)

    elapsed = time.time() - run_start
    size_gib = output_path.stat().st_size / (1024 ** 3)

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)
    print(f"output:              {output_path}")
    print(f"size:                {size_gib:.2f} GiB")
    print(f"total capped values: {total_caps:,}")
    print(f"elapsed this run:    {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
