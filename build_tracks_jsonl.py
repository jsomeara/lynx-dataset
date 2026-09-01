#!/usr/bin/env python3
"""
Build tracks.jsonl for a stripped-down Borzoi human target set.

KEEP:
  - RNA-seq
  - CAGE
  - DNase-seq
  - ATAC-seq

EXCLUDE:
  - ChIP-seq
  - histone-mark tracks

Each JSONL line is one target track.

Notes:
- Borzoi's private /home/drk/... paths are NOT used as download paths.
- ENCODE rows include stable file accessions and API/download endpoints.
- FANTOM5, recount3/GTEx, and CATlas rows include stable public IDs and source info.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests


BORZOI_TARGETS_URL = (
    "https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt"
)


def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def clean(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def strip_strand_suffix(identifier: str) -> tuple[str, str]:
    if identifier.endswith("+"):
        return identifier[:-1], "+"
    if identifier.endswith("-"):
        return identifier[:-1], "-"
    return identifier, "."


def classify_borzoi(row: pd.Series) -> tuple[str, str]:
    p = str(row.get("file", "")).lower()
    d = str(row.get("description", "")).lower()

    if "/rna/encode/" in p:
        return "RNA-seq", "ENCODE"
    if "/dnase/encode/" in p:
        return "DNase-seq", "ENCODE"
    if "/rna/recount3/" in p:
        return "RNA-seq", "recount3/GTEx"
    if "/cage/fantom/" in p:
        return "CAGE", "FANTOM5"
    if "/atac/catlas/" in p:
        return "ATAC-seq", "CATlas"

    if "chip" in p or "chip" in d or "histone" in d:
        return "exclude", "exclude"

    return "exclude", "exclude"


def encode_download_info(accession: str) -> dict:
    return {
        "provider": "ENCODE",
        "file_accession": accession,
        "api_url": f"https://www.encodeproject.org/files/{accession}/?format=json",
        "download_base_url": f"https://www.encodeproject.org/files/{accession}/@@download/",
    }


def source_info(provider: str, stable_id: str) -> dict:
    if provider == "ENCODE":
        return encode_download_info(stable_id)

    if provider == "FANTOM5":
        return {
            "provider": "FANTOM5",
            "sample_id": stable_id,
            "portal": "https://fantom.gsc.riken.jp/5/",
        }

    if provider == "recount3/GTEx":
        return {
            "provider": "recount3",
            "sample_id": stable_id,
            "project": "GTEx",
            "portal": "https://rna.recount.bio/",
        }

    if provider == "CATlas":
        return {
            "provider": "CATlas",
            "track_id": stable_id,
            "portal": "http://catlas.org/humanenhancer/",
        }

    raise ValueError(provider)


def build_records() -> list[dict]:
    df = pd.read_csv(io.StringIO(fetch_text(BORZOI_TARGETS_URL)), sep="\t")

    id_by_index = {
        idx: str(row.get("identifier", idx))
        for idx, row in df.iterrows()
    }

    records = []

    for idx, row in df.iterrows():
        assay, provider = classify_borzoi(row)
        if assay == "exclude":
            continue

        borzoi_identifier = str(row.get("identifier", idx))
        stable_id, strand = strip_strand_suffix(borzoi_identifier)

        pair_identifier = None
        pair_idx = clean(row.get("strand_pair"))

        try:
            pair_idx_int = int(pair_idx)
            if pair_idx_int != idx and pair_idx_int in id_by_index:
                pair_identifier = id_by_index[pair_idx_int]
        except (TypeError, ValueError):
            pass

        records.append({
            "id": borzoi_identifier,
            "dataset": "borzoi",
            "assay": assay,
            "provider": provider,
            "strand": strand,
            "strand_pair": pair_identifier,
            "description": clean(row.get("description")),
            "source": source_info(provider, stable_id),
            "preprocessing": {
                "scale": clean(row.get("scale")),
                "sum_stat": clean(row.get("sum_stat")),
                "clip_soft": clean(row.get("clip_soft")),
                "clip": clean(row.get("clip")),
            },
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tracks.jsonl"))
    args = parser.parse_args()

    print("Loading Borzoi public manifest...")
    records = build_records()

    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    assay_counts = {}
    provider_counts = {}

    for r in records:
        assay_counts[r["assay"]] = assay_counts.get(r["assay"], 0) + 1
        provider_counts[r["provider"]] = provider_counts.get(r["provider"], 0) + 1

    print(f"Total tracks: {len(records)}")

    print("By assay:")
    for assay, n in sorted(assay_counts.items()):
        print(f"  {assay}: {n}")

    print("By provider:")
    for provider, n in sorted(provider_counts.items()):
        print(f"  {provider}: {n}")

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
