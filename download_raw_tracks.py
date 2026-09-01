#!/usr/bin/env python3
"""
Final downloader for the stripped-down Borzoi dataset in tracks.jsonl.

Downloads ALL tracks into one uniform raw format:

    raw_data/<track_id>.bw

Supported providers
-------------------
- ENCODE          : RNA-seq + DNase-seq
- recount3 / GTEx : RNA-seq
- FANTOM5         : CAGE
- CATlas          : ATAC-seq

Download behavior
-----------------
- Processes ONE FILE AT A TIME.
- Uses up to --connections HTTP byte-range connections PER FILE (default 30),
  axel-style.
- Progress bar is updated continuously from bytes received across all range
  workers, so it moves smoothly.
- If a server does not support byte-range requests, automatically falls back
  to a normal single-connection download.
- Existing non-empty .bw files are skipped by default.
- Every newly downloaded file is validated with pyBigWig.
- Temporary .part / .parts files are cleaned before retrying.
- Failures print immediately and are also written to:
      raw_data/download_failures.tsv
- The script continues after individual failures so one bad source does not
  abort the entire dataset download.

Source-resolution fixes included
--------------------------------
ENCODE
  - Resolves each ENCFF accession through the ENCODE API and downloads the
    actual BigWig href.

recount3 / GTEx
  - Resolves public sample-level base_sums BigWigs using recount3 mirrors.

FANTOM5
  - First uses UCSC's hg38 FANTOM5 per-sample fwd/rev BigWig mirror.
  - Falls back to the official FANTOM5 hg38 CTSS BigWig directories for
    libraries not mirrored by UCSC.

CATlas
  - Uses live directory indexes rather than guessed filenames.
  - Tolerates punctuation differences in CATlas filenames.
  - Prefers the WUSTL mirror, then falls back to catlas.org.

No training preprocessing is applied here:
  - no binning
  - no x^0.75 transform
  - no clipping/scaling
  - no blacklist correction

This script only materializes the raw genome-wide BigWig tracks.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import re
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import pyBigWig
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Public sources
# ---------------------------------------------------------------------------

BORZOI_TARGETS_URL = (
    "https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt"
)

ENCODE_BASE = "https://www.encodeproject.org"

RECOUNT_MIRRORS = [
    "https://duffel.rail.bio/recount3",
    "https://idies.jhu.edu/recount3/data",
]

UCSC_FANTOM_DIR = "https://hgdownload.soe.ucsc.edu/gbdb/hg38/fantom5/"

FANTOM_ROOT = "https://fantom.gsc.riken.jp/5/datahub/hg38/ctss/"
FANTOM_SUBDIRS = [
    "human.cell_line.LQhCAGE/",
    "human.cell_line.hCAGE/",
    "human.fractionation.hCAGE/",
    "human.primary_cell.LQhCAGE/",
    "human.primary_cell.hCAGE/",
    "human.timecourse.LQhCAGE/",
    "human.timecourse.hCAGE/",
    "human.tissue.hCAGE/",
]

CATLAS_BASES = [
    # Prefer the faster mirror.
    "https://decoder-genetics.wustl.edu/catlasv1/humanenhancer/data/bw/",
    "https://catlas.org/humanenhancer/data/bw/",
]

USER_AGENT = "lynx-dataset-final-downloader/1.0"


# ---------------------------------------------------------------------------
# Logging + HTTP sessions
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()
_thread_local = threading.local()


def log(msg: str) -> None:
    with _print_lock:
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)

    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=0,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        _thread_local.session = s

    return s


def get_with_retry(
    url: str,
    *,
    stream: bool = False,
    timeout=(15, 120),
    headers: dict | None = None,
    tries: int = 4,
):
    last_error = None

    for attempt in range(1, tries + 1):
        try:
            r = get_session().get(
                url,
                stream=stream,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            )
            r.raise_for_status()
            return r

        except Exception as e:
            last_error = e

            if attempt < tries:
                log(
                    f"GET retry {attempt}/{tries}: "
                    f"{url}: {e}"
                )
                time.sleep(min(10, 2 ** attempt))

    raise RuntimeError(
        f"GET failed after {tries} attempts: {url}: {last_error}"
    )


def url_exists(url: str) -> bool:
    """
    Lightweight existence check.

    A 0-byte range request is more reliable than HEAD for several genomics hosts.
    """
    try:
        r = get_session().get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(10, 30),
            allow_redirects=True,
        )

        ok = r.status_code in {200, 206}
        r.close()
        return ok

    except Exception:
        return False


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def safe_filename(track_id: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.+-]+",
        "_",
        track_id,
    ).strip("_")


def file_present(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def valid_bigwig(path: Path) -> bool:
    if not file_present(path):
        return False

    try:
        bw = pyBigWig.open(str(path))

        if bw is None:
            return False

        ok = bool(bw.chroms())
        bw.close()

        return ok

    except Exception:
        return False


def cleanup_partial(dest: Path) -> None:
    Path(str(dest) + ".part").unlink(missing_ok=True)
    shutil.rmtree(
        Path(str(dest) + ".parts"),
        ignore_errors=True,
    )


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

    return tracks


# ---------------------------------------------------------------------------
# HTML directory indexes
# ---------------------------------------------------------------------------

def fetch_bigwig_index(url: str) -> dict[str, str]:
    log(f"Indexing {url}")

    r = get_with_retry(
        url,
        timeout=(15, 180),
    )

    soup = BeautifulSoup(
        r.text,
        "html.parser",
    )

    files = {
        a["href"]: urljoin(
            url,
            a["href"],
        )
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".bw")
    }

    log(
        f"  found {len(files)} BigWigs"
    )

    return files


def canonical_name(name: str) -> str:
    """
    Ignore punctuation/case differences for safe CATlas matching.

    Example:
      Fetal_ParietalChief.bw
      Fetal_Parietal+Chief.bw

    both normalize to:
      fetalparietalchief
    """
    name = re.sub(
        r"\.bw$",
        "",
        name,
        flags=re.I,
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        name.lower(),
    )


# ---------------------------------------------------------------------------
# ENCODE
# ---------------------------------------------------------------------------

def resolve_encode(track: dict) -> list[str]:
    api_url = track.get("source", {}).get("api_url")

    if not api_url:
        accession = (
            track.get("source", {}).get("file_accession")
            or str(track["id"]).rstrip("+-")
        )

        api_url = (
            f"{ENCODE_BASE}/files/{accession}/?format=json"
        )

    meta = get_with_retry(
        api_url,
        timeout=(15, 120),
    ).json()

    if str(
        meta.get("file_format", "")
    ).lower() != "bigwig":
        raise RuntimeError(
            f"ENCODE {track['id']} resolved to "
            f"file_format={meta.get('file_format')!r}, not bigWig"
        )

    href = meta.get("href")

    if not href:
        raise RuntimeError(
            f"ENCODE API returned no href for {track['id']}"
        )

    return [
        urljoin(
            ENCODE_BASE,
            href,
        )
    ]


# ---------------------------------------------------------------------------
# recount3 / GTEx
# ---------------------------------------------------------------------------

def recount_study_candidates(track: dict) -> list[str]:
    """
    Borzoi descriptions generally encode the GTEx tissue after the first colon.
    Try a few path-normalized spellings because recount3 directory names use
    uppercase/underscore conventions.
    """
    desc = str(
        track.get("description")
        or ""
    )

    tissue = (
        desc.split(":", 1)[1].strip()
        if ":" in desc
        else desc.strip()
    )

    candidates = [
        tissue.upper(),
        tissue.upper().replace(" ", "_"),
        tissue.upper().replace("-", "_"),
        tissue.upper().replace(" ", "_").replace("-", "_"),
    ]

    return list(
        dict.fromkeys(candidates)
    )


def resolve_recount3(track: dict) -> list[str]:
    sample = str(
        track.get("source", {}).get("sample_id")
        or track["id"]
    )

    sample_no_version = re.sub(
        r"\.\d+$",
        "",
        sample,
    )

    sample_tail = (
        sample_no_version[-2:]
    )

    candidates = []

    for study in recount_study_candidates(
        track
    ):
        study_tail = study[-2:]

        filename = (
            f"gtex.base_sums."
            f"{study}_{sample}.ALL.bw"
        )

        for mirror in RECOUNT_MIRRORS:
            url = (
                f"{mirror}/human/data_sources/"
                f"gtex/base_sums/"
                f"{study_tail}/{study}/"
                f"{sample_tail}/{filename}"
            )

            candidates.append(url)

    # Keep only URLs that actually exist, preserving mirror order.
    resolved = [
        url
        for url in candidates
        if url_exists(url)
    ]

    resolved = list(
        dict.fromkeys(resolved)
    )

    if not resolved:
        raise RuntimeError(
            f"Could not resolve recount3 BigWig "
            f"for {track['id']}"
        )

    return resolved


# ---------------------------------------------------------------------------
# FANTOM5
# ---------------------------------------------------------------------------

def build_ucsc_fantom_map() -> dict[tuple[str, str], str]:
    """
    UCSC mirrors most Borzoi-era FANTOM5 hg38 tracks as per-sample
    .ctss.fwd.bw / .ctss.rev.bw files.
    """
    files = fetch_bigwig_index(
        UCSC_FANTOM_DIR
    )

    mapping = {}

    for filename, url in files.items():
        m = re.search(
            r"(CNhs\d+)",
            filename,
            flags=re.I,
        )

        if not m:
            continue

        low = filename.lower()

        if ".ctss.fwd.bw" in low:
            strand = "+"

        elif ".ctss.rev.bw" in low:
            strand = "-"

        else:
            continue

        mapping[
            (
                m.group(1),
                strand,
            )
        ] = url

    log(
        f"UCSC FANTOM5 mappings: "
        f"{len(mapping)} strand tracks"
    )

    return mapping


def build_official_fantom_map() -> dict[tuple[str, str], str]:
    """
    Official FANTOM5 hg38 CTSS datahub.

    This catches the handful of Borzoi FANTOM libraries absent from UCSC.
    """
    mapping = {}

    for i, subdir in enumerate(
        FANTOM_SUBDIRS,
        1,
    ):
        log(
            f"FANTOM5 official index "
            f"{i}/{len(FANTOM_SUBDIRS)}: "
            f"{subdir}"
        )

        files = fetch_bigwig_index(
            urljoin(
                FANTOM_ROOT,
                subdir,
            )
        )

        for filename, url in files.items():
            m = re.search(
                r"(CNhs\d+)",
                filename,
            )

            if not m:
                continue

            low = filename.lower()

            if ".ctss.fwd.bw" in low:
                strand = "+"

            elif ".ctss.rev.bw" in low:
                strand = "-"

            else:
                continue

            mapping[
                (
                    m.group(1),
                    strand,
                )
            ] = url

    log(
        f"Official FANTOM5 mappings: "
        f"{len(mapping)} strand tracks"
    )

    return mapping


def resolve_fantom(
    track: dict,
    ucsc_map,
    official_map,
) -> list[str]:
    lib = str(
        track.get("source", {}).get("sample_id")
        or track["id"]
    ).rstrip("+-")

    strand = track.get(
        "strand"
    )

    key = (
        lib,
        strand,
    )

    urls = []

    if key in ucsc_map:
        urls.append(
            ucsc_map[key]
        )

    if key in official_map:
        urls.append(
            official_map[key]
        )

    urls = list(
        dict.fromkeys(urls)
    )

    if not urls:
        raise RuntimeError(
            f"No hg38 FANTOM5 BigWig found "
            f"for {lib} strand {strand}"
        )

    return urls


# ---------------------------------------------------------------------------
# CATlas
# ---------------------------------------------------------------------------

def load_catlas_expected_names() -> dict[str, str]:
    """
    Recover Borzoi's intended CATlas filename basename from the public target
    manifest. We use only the basename, never Borzoi's private /home/drk path.
    """
    r = get_with_retry(
        BORZOI_TARGETS_URL,
        timeout=(15, 120),
    )

    lines = r.text.splitlines()

    if not lines:
        raise RuntimeError(
            "Borzoi target manifest is empty"
        )

    header = lines[0].split("\t")

    expected = {}

    for line in lines[1:]:
        fields = line.split("\t")

        if len(fields) != len(header):
            continue

        row = dict(
            zip(
                header,
                fields,
            )
        )

        old_path = row.get(
            "file",
            "",
        )

        if "/atac/catlas/" not in old_path:
            continue

        expected[
            row["identifier"]
        ] = Path(
            old_path
        ).stem

    log(
        f"Recovered {len(expected)} "
        f"CATlas expected basenames"
    )

    return expected


def build_catlas_indexes():
    """
    Build live filename indexes for both CATlas hosts.

    Hosts remain in CATLAS_BASES priority order:
      1) WUSTL mirror
      2) catlas.org
    """
    indexes = []

    for base in CATLAS_BASES:
        try:
            exact = fetch_bigwig_index(
                base
            )

            grouped = {}

            for filename, url in exact.items():
                grouped.setdefault(
                    canonical_name(filename),
                    [],
                ).append(url)

            canonical = {
                key: urls[0]
                for key, urls
                in grouped.items()
                if len(urls) == 1
            }

            indexes.append(
                {
                    "base": base,
                    "exact": exact,
                    "canonical": canonical,
                }
            )

        except Exception as e:
            log(
                f"CATlas host unavailable: "
                f"{base}: {e}"
            )

    if not indexes:
        raise RuntimeError(
            "No CATlas host could be indexed"
        )

    return indexes


def resolve_catlas(
    track: dict,
    expected_names,
    indexes,
) -> list[str]:
    track_id = str(
        track["id"]
    )

    stem = expected_names.get(
        track_id
    )

    if not stem:
        raise RuntimeError(
            f"No CATlas basename found for "
            f"{track_id}"
        )

    exact_filename = (
        stem + ".bw"
    )

    canon = canonical_name(
        stem
    )

    urls = []

    for index in indexes:
        url = None

        if exact_filename in index["exact"]:
            url = index["exact"][
                exact_filename
            ]

        elif canon in index["canonical"]:
            url = index["canonical"][
                canon
            ]

        if url:
            urls.append(url)

    urls = list(
        dict.fromkeys(urls)
    )

    if not urls:
        raise RuntimeError(
            f"No CATlas mirror match "
            f"for expected basename {stem!r}"
        )

    return urls


# ---------------------------------------------------------------------------
# Source dispatcher
# ---------------------------------------------------------------------------

def resolve_track_urls(
    track: dict,
    *,
    ucsc_fantom_map,
    official_fantom_map,
    catlas_expected,
    catlas_indexes,
) -> list[str]:
    provider = track.get(
        "provider"
    )

    if provider == "ENCODE":
        return resolve_encode(
            track
        )

    if provider == "recount3/GTEx":
        return resolve_recount3(
            track
        )

    if provider == "FANTOM5":
        return resolve_fantom(
            track,
            ucsc_fantom_map,
            official_fantom_map,
        )

    if provider == "CATlas":
        return resolve_catlas(
            track,
            catlas_expected,
            catlas_indexes,
        )

    raise RuntimeError(
        f"Unsupported provider "
        f"{provider!r}"
    )


# ---------------------------------------------------------------------------
# Axel-style one-file-at-a-time downloader
# ---------------------------------------------------------------------------

def probe_range_support(
    url: str,
) -> tuple[int | None, bool]:
    """
    Probe with bytes=0-0.

    Returns:
      (total_size_bytes, supports_ranges)
    """
    log(
        f"Probing range support: {url}"
    )

    try:
        r = get_session().get(
            url,
            headers={
                "Range": "bytes=0-0"
            },
            stream=True,
            timeout=(15, 60),
            allow_redirects=True,
        )

        if r.status_code == 206:
            content_range = (
                r.headers.get(
                    "Content-Range",
                    "",
                )
            )

            match = re.search(
                r"/(\d+)$",
                content_range,
            )

            if match:
                size = int(
                    match.group(1)
                )

                r.close()

                log(
                    f"  ranges supported; "
                    f"size={size / 1e6:.1f} MB"
                )

                return size, True

        size_header = (
            r.headers.get(
                "Content-Length"
            )
        )

        size = (
            int(size_header)
            if size_header
            and size_header.isdigit()
            else None
        )

        status = r.status_code
        r.close()

        log(
            f"  range support not confirmed "
            f"(HTTP {status}); "
            f"using one connection"
        )

        return size, False

    except Exception as e:
        log(
            f"  range probe failed: {e}; "
            f"using one connection"
        )

        return None, False


def download_range(
    url: str,
    start: int,
    end: int,
    part_path: Path,
    connection_num: int,
    progress_callback,
) -> int:
    """
    Download exactly one inclusive byte range.
    """
    expected = (
        end - start + 1
    )

    headers = {
        "Range": (
            f"bytes={start}-{end}"
        )
    }

    last_error = None

    for attempt in range(1, 5):
        try:
            r = get_session().get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 300),
                allow_redirects=True,
            )

            if r.status_code != 206:
                raise RuntimeError(
                    f"HTTP {r.status_code}; "
                    f"expected 206"
                )

            written = 0

            with part_path.open(
                "wb"
            ) as f:
                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    f.write(chunk)

                    n = len(chunk)
                    written += n
                    progress_callback(n)

            r.close()

            if written != expected:
                raise RuntimeError(
                    f"range size mismatch: "
                    f"{written} != {expected}"
                )

            return written

        except Exception as e:
            last_error = e

            part_path.unlink(
                missing_ok=True
            )

            log(
                f"  connection {connection_num} "
                f"retry {attempt}/4 "
                f"bytes {start}-{end}: {e}"
            )

            if attempt < 4:
                time.sleep(
                    min(
                        10,
                        2 ** attempt,
                    )
                )

    raise RuntimeError(
        f"connection {connection_num} "
        f"failed: {last_error}"
    )


def normal_download(
    url: str,
    dest: Path,
    track_id: str,
) -> None:
    cleanup_partial(dest)

    tmp = Path(
        str(dest) + ".part"
    )

    log(
        f"[{track_id}] "
        f"single-connection download"
    )

    t0 = time.time()

    r = get_with_retry(
        url,
        stream=True,
        timeout=(15, 300),
    )

    total_header = (
        r.headers.get(
            "Content-Length"
        )
    )

    total = (
        int(total_header)
        if total_header
        and total_header.isdigit()
        else None
    )

    written = 0

    with tmp.open("wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=track_id,
        mininterval=0.1,
        smoothing=0.1,
        leave=True,
    ) as bar:

        for chunk in r.iter_content(
            chunk_size=1024 * 1024
        ):
            if not chunk:
                continue

            f.write(chunk)

            n = len(chunk)
            written += n
            bar.update(n)

    r.close()

    tmp.replace(dest)

    elapsed = (
        time.time() - t0
    )

    log(
        f"[{track_id}] "
        f"finished "
        f"{written / 1e6:.1f} MB "
        f"in {elapsed:.1f}s"
    )

    if not valid_bigwig(dest):
        dest.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "downloaded file is not "
            "a valid BigWig"
        )


def multipart_download(
    url: str,
    dest: Path,
    connections: int,
    track_id: str,
) -> None:
    """
    Download exactly ONE file with N concurrent range connections.

    No other track is processed until this file finishes.
    """
    cleanup_partial(dest)

    size, supports_ranges = (
        probe_range_support(url)
    )

    if (
        not supports_ranges
        or size is None
        or connections <= 1
    ):
        normal_download(
            url,
            dest,
            track_id,
        )
        return

    connections = max(
        1,
        min(
            connections,
            size,
        ),
    )

    chunk_size = math.ceil(
        size / connections
    )

    part_dir = Path(
        str(dest) + ".parts"
    )

    part_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranges = []

    for i in range(
        connections
    ):
        start = (
            i * chunk_size
        )

        if start >= size:
            break

        end = min(
            size - 1,
            start + chunk_size - 1,
        )

        part_path = (
            part_dir
            / f"part_{i:03d}"
        )

        ranges.append(
            (
                i,
                start,
                end,
                part_path,
            )
        )

    log(
        f"[{track_id}] "
        f"{size / 1e6:.1f} MB "
        f"using {len(ranges)} connections"
    )

    progress_lock = (
        threading.Lock()
    )

    total_downloaded = 0
    t0 = time.time()

    with tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=track_id,
        mininterval=0.1,
        smoothing=0.1,
        leave=True,
    ) as bar:

        def progress_callback(
            n_bytes: int
        ):
            with progress_lock:
                bar.update(
                    n_bytes
                )

        with cf.ThreadPoolExecutor(
            max_workers=len(ranges)
        ) as pool:

            futures = {
                pool.submit(
                    download_range,
                    url,
                    start,
                    end,
                    part_path,
                    i + 1,
                    progress_callback,
                ): (
                    i,
                    start,
                    end,
                    part_path,
                )
                for (
                    i,
                    start,
                    end,
                    part_path,
                ) in ranges
            }

            for future in cf.as_completed(
                futures
            ):
                total_downloaded += (
                    future.result()
                )

    if total_downloaded != size:
        cleanup_partial(dest)

        raise RuntimeError(
            f"multipart total mismatch: "
            f"{total_downloaded} != {size}"
        )

    tmp = Path(
        str(dest) + ".part"
    )

    log(
        f"[{track_id}] "
        f"joining {len(ranges)} parts"
    )

    with tmp.open(
        "wb"
    ) as out:

        for (
            _,
            _,
            _,
            part_path,
        ) in ranges:

            with part_path.open(
                "rb"
            ) as src:

                shutil.copyfileobj(
                    src,
                    out,
                    length=8 * 1024 * 1024,
                )

    joined_size = (
        tmp.stat().st_size
    )

    if joined_size != size:
        cleanup_partial(dest)

        raise RuntimeError(
            f"joined size mismatch: "
            f"{joined_size} != {size}"
        )

    tmp.replace(dest)

    shutil.rmtree(
        part_dir,
        ignore_errors=True,
    )

    elapsed = (
        time.time() - t0
    )

    speed = (
        size / 1e6
        / max(
            elapsed,
            0.001,
        )
    )

    log(
        f"[{track_id}] "
        f"finished "
        f"{size / 1e6:.1f} MB "
        f"in {elapsed:.1f}s "
        f"({speed:.1f} MB/s)"
    )

    if not valid_bigwig(dest):
        dest.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "assembled file is not "
            "a valid BigWig"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path(
            "tracks.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "raw_data"
        ),
    )

    parser.add_argument(
        "--connections",
        type=int,
        default=30,
        help=(
            "HTTP connections PER FILE. "
            "Files are processed one at a time. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--deep-scan",
        action="store_true",
        help=(
            "Validate every existing .bw "
            "with pyBigWig before skipping it. "
            "Slower on network filesystems."
        ),
    )

    args = parser.parse_args()

    if args.connections < 1:
        raise ValueError(
            "--connections must be >= 1"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------

    log(
        f"Reading tracks from "
        f"{args.tracks}"
    )

    tracks = read_tracks(
        args.tracks
    )

    log(
        f"Loaded {len(tracks)} tracks"
    )

    provider_counts = {}

    for track in tracks:
        provider = track.get(
            "provider",
            "UNKNOWN",
        )

        provider_counts[
            provider
        ] = (
            provider_counts.get(
                provider,
                0,
            )
            + 1
        )

    for provider, count in sorted(
        provider_counts.items()
    ):
        log(
            f"  {provider}: {count}"
        )

    # -------------------------------------------------------
    # Find tracks that need downloading
    # -------------------------------------------------------

    checker = (
        valid_bigwig
        if args.deep_scan
        else file_present
    )

    missing = []

    log(
        "Scanning output directory "
        f"{args.output_dir}"
    )

    for i, track in enumerate(
        tracks,
        1,
    ):
        dest = (
            args.output_dir
            / (
                safe_filename(
                    str(
                        track["id"]
                    )
                )
                + ".bw"
            )
        )

        if not checker(dest):
            missing.append(
                track
            )

        if (
            i % 500 == 0
            or i == len(tracks)
        ):
            log(
                f"  scanned "
                f"{i}/{len(tracks)}; "
                f"need download: "
                f"{len(missing)}"
            )

    log(
        f"Tracks to download: "
        f"{len(missing)}"
    )

    if not missing:
        log(
            "Everything is already present."
        )
        return

    missing_providers = {
        track.get("provider")
        for track in missing
    }

    # -------------------------------------------------------
    # Build source indexes only when needed
    # -------------------------------------------------------

    ucsc_fantom_map = {}
    official_fantom_map = {}
    catlas_expected = {}
    catlas_indexes = []

    if "FANTOM5" in missing_providers:
        log(
            "Preparing UCSC FANTOM5 mirror..."
        )

        ucsc_fantom_map = (
            build_ucsc_fantom_map()
        )

        log(
            "Preparing official FANTOM5 fallback..."
        )

        official_fantom_map = (
            build_official_fantom_map()
        )

    if "CATlas" in missing_providers:
        log(
            "Preparing CATlas mirrors..."
        )

        catlas_expected = (
            load_catlas_expected_names()
        )

        catlas_indexes = (
            build_catlas_indexes()
        )

    # -------------------------------------------------------
    # Download sequentially: one track at a time
    # -------------------------------------------------------

    failures = []
    downloaded = 0

    for i, track in enumerate(
        missing,
        1,
    ):
        track_id = str(
            track["id"]
        )

        provider = track.get(
            "provider"
        )

        dest = (
            args.output_dir
            / (
                safe_filename(
                    track_id
                )
                + ".bw"
            )
        )

        cleanup_partial(
            dest
        )

        log(
            f"FILE {i}/{len(missing)}: "
            f"{track_id} "
            f"({provider})"
        )

        try:
            urls = resolve_track_urls(
                track,
                ucsc_fantom_map=ucsc_fantom_map,
                official_fantom_map=official_fantom_map,
                catlas_expected=catlas_expected,
                catlas_indexes=catlas_indexes,
            )

            success = False
            last_error = None

            for source_num, url in enumerate(
                urls,
                1,
            ):
                try:
                    log(
                        f"[{track_id}] "
                        f"source "
                        f"{source_num}/{len(urls)}: "
                        f"{url}"
                    )

                    multipart_download(
                        url=url,
                        dest=dest,
                        connections=args.connections,
                        track_id=track_id,
                    )

                    success = True
                    break

                except Exception as e:
                    last_error = e

                    cleanup_partial(
                        dest
                    )

                    dest.unlink(
                        missing_ok=True
                    )

                    log(
                        f"[{track_id}] "
                        f"source failed: {e}"
                    )

            if not success:
                raise RuntimeError(
                    f"all candidate sources failed: "
                    f"{last_error}"
                )

            downloaded += 1

            log(
                f"[OK] {track_id}"
            )

        except Exception as e:
            error = (
                str(e)
                .replace("\t", " ")
                .replace("\n", " ")
            )

            log(
                f"[FAIL] "
                f"{track_id}: "
                f"{error}"
            )

            failures.append(
                (
                    track_id,
                    provider,
                    error,
                )
            )

    # -------------------------------------------------------
    # Failure report
    # -------------------------------------------------------

    failure_path = (
        args.output_dir
        / "download_failures.tsv"
    )

    if failures:
        with failure_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
            )

            writer.writerow(
                [
                    "track_id",
                    "provider",
                    "error",
                ]
            )

            writer.writerows(
                failures
            )

        log(
            f"Downloaded successfully "
            f"this run: {downloaded}"
        )

        log(
            f"Failures: {len(failures)}"
        )

        log(
            f"Failure report: "
            f"{failure_path}"
        )

        raise SystemExit(2)

    failure_path.unlink(
        missing_ok=True
    )

    log(
        f"All {downloaded} requested "
        f"tracks downloaded successfully."
    )


if __name__ == "__main__":
    main()
