#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import streamlit as st

TRACKS_JSONL = Path("tracks.jsonl")
RAW_DIR = Path("raw_data")
UNMAPPABLE_BED = Path("data/unmap_macro.bed")
BLACKLIST_BED = Path("data/hg38-blacklist.v2.bed.gz")
TARGET_TOTAL = 100_000_000.0


def safe_filename(track_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", track_id).strip("_")


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


@st.cache_data
def load_tracks(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@st.cache_data
def load_bed(path: str):
    p = Path(path)
    out = {}
    with open_text(p) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split()
            if len(fields) < 3:
                continue
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if end > start:
                out.setdefault(chrom, []).append((start, end))

    merged = {}
    for chrom, xs in out.items():
        xs.sort()
        acc = []
        for s, e in xs:
            if not acc or s > acc[-1][1]:
                acc.append([s, e])
            else:
                acc[-1][1] = max(acc[-1][1], e)
        merged[chrom] = [(s, e) for s, e in acc]
    return merged


def make_bad_mask(chrom, start, end, blacklist, unmappable):
    mask = np.zeros(end - start, dtype=bool)
    for source in (blacklist, unmappable):
        for s, e in source.get(chrom, []):
            if e <= start:
                continue
            if s >= end:
                break
            left = max(start, s) - start
            right = min(end, e) - start
            if right > left:
                mask[left:right] = True
    return mask


@st.cache_data(show_spinner=True)
def compute_track_total(bw_path: str, blacklist_path: str, unmappable_path: str):
    blacklist = load_bed(blacklist_path)
    unmappable = load_bed(unmappable_path)

    bw = pyBigWig.open(bw_path)
    chroms = bw.chroms()
    total = 0.0

    chunk_size = 2_000_000
    for chrom, chrom_size in chroms.items():
        for start in range(0, chrom_size, chunk_size):
            end = min(chrom_size, start + chunk_size)
            vals = np.asarray(
                bw.values(chrom, start, end, numpy=True),
                dtype=np.float64,
            )
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            bad = make_bad_mask(
                chrom, start, end, blacklist, unmappable
            )

            valid = ~bad
            if np.any(valid):
                total += float(vals[valid].sum())

    bw.close()
    return total


def process_signal(raw_values, bad_mask, sum_stat, norm_factor):
    x = np.nan_to_num(
        raw_values.astype(np.float64, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x *= norm_factor

    if sum_stat == "sum_sqrt":
        x = np.power(np.maximum(x, 0.0), 0.75)

    x[bad_mask] = np.nan
    return x


def downsample_for_plot(x, y, max_points=5000):
    if len(x) <= max_points:
        return x, y

    stride = math.ceil(len(x) / max_points)
    n = (len(x) // stride) * stride

    x2 = x[:n].reshape(-1, stride).mean(axis=1)
    yy = y[:n].reshape(-1, stride)

    with np.errstate(invalid="ignore"):
        y2 = np.nanmean(yy, axis=1)

    return x2, y2


def main():
    st.set_page_config(page_title="Lynx Track Browser", layout="wide")
    st.title("Lynx 1-bp Track Browser")
    st.caption("Browse one human track and compare raw vs proposed 1-bp preprocessing.")

    tracks = load_tracks(str(TRACKS_JSONL))
    blacklist = load_bed(str(BLACKLIST_BED))
    unmappable = load_bed(str(UNMAPPABLE_BED))

    available = []
    for t in tracks:
        path = RAW_DIR / f"{safe_filename(str(t['id']))}.bw"
        if path.exists() and path.stat().st_size > 0:
            available.append((t, path))

    if not available:
        st.error("No downloaded BigWigs matched tracks.jsonl.")
        st.stop()

    labels = [
        f"{t['id']} | {t.get('assay')} | {t.get('provider')} | "
        f"{(t.get('preprocessing') or {}).get('sum_stat')}"
        for t, _ in available
    ]

    selected_label = st.sidebar.selectbox("Track", labels)
    idx = labels.index(selected_label)
    track, bw_path = available[idx]

    prep = track.get("preprocessing") or {}
    sum_stat = prep.get("sum_stat")

    bw = pyBigWig.open(str(bw_path))
    chroms = bw.chroms()
    chrom_names = list(chroms.keys())

    default_chrom = "chr1" if "chr1" in chroms else chrom_names[0]

    if "chrom" not in st.session_state:
        st.session_state.chrom = default_chrom
    if "start" not in st.session_state:
        st.session_state.start = 1_000_000
    if "window" not in st.session_state:
        st.session_state.window = 10_000

    st.sidebar.markdown("### Region")

    chrom = st.sidebar.selectbox(
        "Chromosome",
        chrom_names,
        index=chrom_names.index(st.session_state.chrom)
        if st.session_state.chrom in chrom_names
        else 0,
    )
    st.session_state.chrom = chrom

    window = st.sidebar.number_input(
        "Window size (bp)",
        min_value=100,
        max_value=2_000_000,
        value=int(st.session_state.window),
        step=1000,
    )
    st.session_state.window = int(window)

    start = st.sidebar.number_input(
        "Start",
        min_value=0,
        max_value=max(0, chroms[chrom] - 1),
        value=min(int(st.session_state.start), max(0, chroms[chrom] - 1)),
        step=max(1, int(window)),
    )
    st.session_state.start = int(start)

    c1, c2 = st.sidebar.columns(2)

    if c1.button("← Left"):
        st.session_state.start = max(0, int(start) - int(window))
        st.rerun()

    if c2.button("Right →"):
        st.session_state.start = min(
            max(0, chroms[chrom] - int(window)),
            int(start) + int(window),
        )
        st.rerun()

    end = min(chroms[chrom], int(start) + int(window))

    st.sidebar.markdown("### Display")
    show_mask = st.sidebar.checkbox("Show blacklist/unmappable mask", value=True)
    show_raw = st.sidebar.checkbox("Show raw", value=True)
    show_processed = st.sidebar.checkbox("Show processed", value=True)

    st.sidebar.markdown("### Proposed preprocessing")
    st.sidebar.write("NaN → 0")
    st.sidebar.write("Mask blacklist + unmappable")
    st.sidebar.write("Normalize valid genome-wide total → 100M")
    if sum_stat == "sum_sqrt":
        st.sidebar.write("Apply x^0.75")
    else:
        st.sidebar.write("No power transform")
    st.sidebar.write("No log1p")
    st.sidebar.write("No inherited 32-bp clipping")
    st.sidebar.write("No inherited Borzoi scale factor")

    st.markdown(
        f"**Track:** `{track['id']}`  \n"
        f"**Assay:** {track.get('assay')}  \n"
        f"**Provider:** {track.get('provider')}  \n"
        f"**sum_stat:** `{sum_stat}`  \n"
        f"**Region:** `{chrom}:{start:,}-{end:,}`"
    )

    raw = np.asarray(
        bw.values(chrom, int(start), int(end), numpy=True),
        dtype=np.float64,
    )
    bw.close()

    sparse_nan_count = int(np.isnan(raw).sum())

    bad = make_bad_mask(
        chrom,
        int(start),
        int(end),
        blacklist,
        unmappable,
    )

    with st.spinner("Computing / loading genome-wide normalization factor..."):
        total_signal = compute_track_total(
            str(bw_path),
            str(BLACKLIST_BED),
            str(UNMAPPABLE_BED),
        )

    if total_signal <= 0:
        st.error("Track has zero valid genome-wide signal; cannot normalize.")
        st.stop()

    norm_factor = TARGET_TOTAL / total_signal

    processed = process_signal(
        raw,
        bad,
        sum_stat,
        norm_factor,
    )

    raw_zero_filled = np.nan_to_num(
        raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    raw_masked = raw_zero_filled.astype(float)
    raw_masked[bad] = np.nan

    coords = np.arange(int(start), int(end), dtype=np.float64)

    plot_x, raw_plot = downsample_for_plot(coords, raw_masked)
    _, processed_plot = downsample_for_plot(coords, processed)

    stats_cols = st.columns(5)
    stats_cols[0].metric("Genome-wide valid total", f"{total_signal:,.3g}")
    stats_cols[1].metric("Normalization factor", f"{norm_factor:.6g}")
    stats_cols[2].metric("NaNs in region", f"{sparse_nan_count:,}")
    stats_cols[3].metric("Masked bases", f"{int(bad.sum()):,}")
    stats_cols[4].metric("Region width", f"{end-start:,} bp")

    fig, ax = plt.subplots(figsize=(14, 4))

    if show_raw:
        ax.plot(plot_x, raw_plot, label="Raw (NaN→0)")

    if show_processed:
        ax.plot(plot_x, processed_plot, label="Processed")

    if show_mask:
        in_span = False
        span_start = None

        for i, is_bad in enumerate(bad):
            if is_bad and not in_span:
                in_span = True
                span_start = start + i
            elif not is_bad and in_span:
                ax.axvspan(span_start, start + i, alpha=0.15)
                in_span = False

        if in_span:
            ax.axvspan(span_start, end, alpha=0.15)

    ax.set_xlabel("Genomic position")
    ax.set_ylabel("Signal")
    ax.legend()
    ax.set_title(f"{track['id']} — raw vs processed")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show numeric values around this region"):
        max_rows = min(2000, end - start)
        stride = max(1, (end - start) // max_rows)

        table = pd.DataFrame(
            {
                "position": np.arange(start, end, stride),
                "raw": raw_zero_filled[::stride],
                "processed": processed[::stride],
                "masked": bad[::stride],
            }
        )
        st.dataframe(table, use_container_width=True)

    with st.expander("Track metadata"):
        st.json(track)


if __name__ == "__main__":
    main()
