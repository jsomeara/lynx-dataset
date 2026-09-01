#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

SEED = 71798669716578
TARGET_SHARDS = 1000
TARGET_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ("train", "val", "test")


def read_regions(path: Path):
    regions = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                raise RuntimeError(f"{path}:{line_no}: expected BED3+")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if end <= start:
                raise RuntimeError(f"{path}:{line_no}: invalid interval")
            regions.append({"chrom": chrom, "start": start, "end": end})
    if not regions:
        raise RuntimeError(f"No regions found in {path}")
    return regions


def counts_for_assignment(assignment, chrom_counts):
    counts = {s: 0 for s in SPLITS}
    for chrom, split in assignment.items():
        counts[split] += chrom_counts[chrom]
    return counts


def score(counts, total):
    out = 0.0
    for split in SPLITS:
        target = TARGET_FRACS[split] * total
        err = (counts[split] - target) / target
        out += err * err
    return out


def initial_assignment(chrom_counts, total):
    targets = {s: TARGET_FRACS[s] * total for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    assignment = {}

    for chrom in sorted(chrom_counts, key=lambda c: (-chrom_counts[c], c)):
        # Assign to split with largest relative deficit.
        split = max(
            SPLITS,
            key=lambda s: ((targets[s] - counts[s]) / targets[s], -counts[s]),
        )
        assignment[chrom] = split
        counts[split] += chrom_counts[chrom]

    return assignment


def improve_assignment(assignment, chrom_counts, total):
    assignment = dict(assignment)

    while True:
        current_counts = counts_for_assignment(assignment, chrom_counts)
        best_score = score(current_counts, total)
        best_action = None
        chroms = sorted(assignment)

        # Single chromosome moves.
        for chrom in chroms:
            old = assignment[chrom]
            if sum(assignment[c] == old for c in chroms) <= 1:
                continue

            for new in SPLITS:
                if new == old:
                    continue

                candidate = dict(current_counts)
                n = chrom_counts[chrom]
                candidate[old] -= n
                candidate[new] += n
                s = score(candidate, total)

                if s < best_score - 1e-15:
                    best_score = s
                    best_action = ("move", chrom, new)

        # Pairwise swaps.
        for i, a in enumerate(chroms):
            sa = assignment[a]
            na = chrom_counts[a]

            for b in chroms[i + 1:]:
                sb = assignment[b]
                if sa == sb:
                    continue

                nb = chrom_counts[b]
                candidate = dict(current_counts)
                candidate[sa] += nb - na
                candidate[sb] += na - nb
                s = score(candidate, total)

                if s < best_score - 1e-15:
                    best_score = s
                    best_action = ("swap", a, b)

        if best_action is None:
            return assignment

        if best_action[0] == "move":
            _, chrom, new = best_action
            assignment[chrom] = new
        else:
            _, a, b = best_action
            assignment[a], assignment[b] = assignment[b], assignment[a]


def allocate_shards(split_counts, target_shards):
    total = sum(split_counts.values())
    target_shards = min(target_shards, total)

    raw = {
        s: target_shards * split_counts[s] / total
        for s in SPLITS
    }

    allocated = {
        s: max(1, int(math.floor(raw[s]))) if split_counts[s] else 0
        for s in SPLITS
    }

    for s in SPLITS:
        allocated[s] = min(allocated[s], split_counts[s])

    current = sum(allocated.values())

    while current < target_shards:
        candidates = [s for s in SPLITS if allocated[s] < split_counts[s]]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda s: (
                raw[s] - math.floor(raw[s]),
                split_counts[s] / max(allocated[s], 1),
            ),
        )
        allocated[chosen] += 1
        current += 1

    while current > target_shards:
        candidates = [s for s in SPLITS if allocated[s] > 1]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda s: (
                raw[s] - math.floor(raw[s]),
                split_counts[s] / allocated[s],
            ),
        )
        allocated[chosen] -= 1
        current -= 1

    return allocated


def partition_evenly(items, n_shards):
    base = len(items) // n_shards
    remainder = len(items) % n_shards

    shards = []
    pos = 0
    for i in range(n_shards):
        size = base + (1 if i < remainder else 0)
        shards.append(items[pos:pos + size])
        pos += size

    assert pos == len(items)
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=Path, default=Path("genomic_regions.bed"))
    ap.add_argument("--output", type=Path, default=Path("shards.jsonl"))
    ap.add_argument("--target-shards", type=int, default=TARGET_SHARDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    regions = read_regions(args.regions)
    total = len(regions)
    chrom_counts = Counter(r["chrom"] for r in regions)

    assignment = initial_assignment(chrom_counts, total)
    assignment = improve_assignment(assignment, chrom_counts, total)
    split_counts = counts_for_assignment(assignment, chrom_counts)

    split_windows = {s: [] for s in SPLITS}
    for region in regions:
        split_windows[assignment[region["chrom"]]].append(region)

    # Deterministically shuffle within each split.
    offsets = {"train": 0x13579BDF, "val": 0x2468ACE0, "test": 0x10203040}
    for split in SPLITS:
        rng = random.Random(args.seed ^ offsets[split])
        rng.shuffle(split_windows[split])

    shard_counts = allocate_shards(split_counts, args.target_shards)

    total_shards = 0
    with args.output.open("w") as out:
        for split in SPLITS:
            shards = partition_evenly(split_windows[split], shard_counts[split])

            for shard_index, windows in enumerate(shards):
                record = {
                    "shard_id": f"{split}-{shard_index:05d}",
                    "split": split,
                    "num_windows": len(windows),
                    "windows": windows,
                }
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                total_shards += 1

    print("=" * 72)
    print("GENOMIC SPLIT")
    print("=" * 72)
    print(f"seed:          {args.seed}")
    print(f"total windows: {total:,}")
    print()

    for split in SPLITS:
        count = split_counts[split]
        pct = 100.0 * count / total
        chroms = sorted(c for c, s in assignment.items() if s == split)

        print(
            f"{split:5s}: {count:6,d} windows "
            f"({pct:6.2f}%; target {100*TARGET_FRACS[split]:.1f}%)"
        )
        print(f"       chromosomes: {', '.join(chroms)}")

    print()
    print("=" * 72)
    print("SHARDS")
    print("=" * 72)

    for split in SPLITS:
        n_windows = split_counts[split]
        n_shards = shard_counts[split]
        print(
            f"{split:5s}: {n_shards:4,d} shards | "
            f"{n_windows:6,d} windows | "
            f"{n_windows // n_shards}-{math.ceil(n_windows / n_shards)} windows/shard"
        )

    print()
    print(f"total shards: {total_shards:,}")
    print(f"wrote:        {args.output}")


if __name__ == "__main__":
    main()
