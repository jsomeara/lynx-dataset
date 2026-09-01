#!/usr/bin/env python3
"""
Count fixed-length hg38 windows that:
  1) contain only A/C/G/T (no N or other ambiguous/missing bases), and
  2) have < max_bad_fraction overlap with the UNION of:
       - an unmappability BED
       - the ENCODE hg38 blacklist BED

By default, windows are non-overlapping 262,144-bp tiles within each
contiguous A/C/G/T run. Use --stride to change the tiling stride.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import re
from collections import defaultdict
from pathlib import Path

from pyfaidx import Fasta


CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_bed(path: str, allowed_chroms: set[str]) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue

            fields = line.rstrip().split()
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_no}: expected at least 3 BED columns")

            chrom = fields[0]
            if chrom not in allowed_chroms:
                continue

            start = int(fields[1])
            end = int(fields[2])

            if start < 0 or end <= start:
                raise ValueError(f"{path}:{line_no}: invalid interval {chrom}:{start}-{end}")

            intervals[chrom].append((start, end))

    return intervals


def merge_union(
    first: dict[str, list[tuple[int, int]]],
    second: dict[str, list[tuple[int, int]]],
    chroms: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """Merge both BED sets into one non-overlapping union per chromosome."""
    merged: dict[str, list[tuple[int, int]]] = {}

    for chrom in chroms:
        xs = sorted(first.get(chrom, []) + second.get(chrom, []))
        out: list[list[int]] = []

        for start, end in xs:
            if not out or start > out[-1][1]:
                out.append([start, end])
            else:
                out[-1][1] = max(out[-1][1], end)

        merged[chrom] = [(s, e) for s, e in out]

    return merged


class IntervalOverlap:
    """Fast overlap queries against sorted, merged BED intervals."""

    def __init__(self, intervals: list[tuple[int, int]]):
        self.intervals = intervals
        self.starts = [s for s, _ in intervals]

    def overlap_bp(self, start: int, end: int) -> int:
        if not self.intervals:
            return 0

        # First interval that could overlap the window.
        i = bisect.bisect_right(self.starts, start)
        if i:
            i -= 1

        total = 0
        n = len(self.intervals)

        while i < n:
            s, e = self.intervals[i]

            if s >= end:
                break
            if e > start:
                total += max(0, min(e, end) - max(s, start))

            i += 1

        return total


def find_windows(
    fasta_path: str,
    bad_union: dict[str, list[tuple[int, int]]],
    window_size: int,
    stride: int,
    max_bad_fraction: float,
    chroms: list[str],
) -> list[tuple[str, int, int, int, float]]:
    genome = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    regions: list[tuple[str, int, int, int, float]] = []
    max_bad_bp = max_bad_fraction * window_size

    try:
        for chrom in chroms:
            if chrom not in genome:
                continue

            seq = str(genome[chrom][:])
            bad = IntervalOverlap(bad_union.get(chrom, []))

            # Only contiguous A/C/G/T runs are eligible.
            # Any N or other IUPAC ambiguity breaks the run.
            for match in re.finditer(r"[ACGT]+", seq):
                run_start, run_end = match.span()

                if run_end - run_start < window_size:
                    continue

                start = run_start
                last_start = run_end - window_size

                while start <= last_start:
                    end = start + window_size
                    bad_bp = bad.overlap_bp(start, end)

                    # Strictly LESS than 50% by default.
                    if bad_bp < max_bad_bp:
                        regions.append(
                            (chrom, start, end, bad_bp, bad_bp / window_size)
                        )

                    start += stride
    finally:
        genome.close()

    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="hg38 FASTA, e.g. data/hg38.fa")
    parser.add_argument("--unmappable", required=True, help="Unmappability BED")
    parser.add_argument("--blacklist", required=True, help="ENCODE hg38 blacklist BED(.gz)")
    parser.add_argument("--window-size", type=int, default=262_144)
    parser.add_argument(
        "--stride",
        type=int,
        default=262_144,
        help="Window stride. Default = non-overlapping windows.",
    )
    parser.add_argument(
        "--max-bad-fraction",
        type=float,
        default=0.50,
        help="Require union(unmappable, blacklist) fraction to be STRICTLY below this.",
    )
    parser.add_argument(
        "--output",
        default="genomic_regions.bed",
        help="Output BED file. Default: genomic_regions.bed",
    )
    args = parser.parse_args()

    if args.window_size <= 0 or args.stride <= 0:
        raise ValueError("window-size and stride must be positive")
    if not (0.0 <= args.max_bad_fraction <= 1.0):
        raise ValueError("max-bad-fraction must be between 0 and 1")

    allowed = set(CANONICAL_CHROMS)

    unmappable = read_bed(args.unmappable, allowed)
    blacklist = read_bed(args.blacklist, allowed)
    bad_union = merge_union(unmappable, blacklist, CANONICAL_CHROMS)

    regions = find_windows(
        fasta_path=args.fasta,
        bad_union=bad_union,
        window_size=args.window_size,
        stride=args.stride,
        max_bad_fraction=args.max_bad_fraction,
        chroms=CANONICAL_CHROMS,
    )

    with open(args.output, "w") as out:
        out.write("#chrom\tstart\tend\tbad_bp\tbad_fraction\n")
        for chrom, start, end, bad_bp, bad_fraction in regions:
            out.write(
                f"{chrom}\t{start}\t{end}\t{bad_bp}\t{bad_fraction:.6f}\n"
            )

    print(f"{len(regions)} regions written to {args.output}")


if __name__ == "__main__":
    main()
