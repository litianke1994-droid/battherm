"""Download the drive-cycle CSVs of the LG 18650HG2 *original* dataset from Mendeley Data.

Dataset: Kollmeyer, Vidal, Naguib, Skells (2020), "LG 18650HG2 Li-ion Battery Data and Example
Deep Neural Network xEV SOC Estimator Script", Mendeley Data, V3, doi:10.17632/cp3473x7xv.3,
licence CC BY 4.0. Files are verified against the SHA-256 published by the API.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/cp3473x7xv"
VERSION = 3
ORIGINAL_ROOT = "LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020"
DRIVE_FILE_RE = re.compile(r"_(UDDS|HWFET|LA92|US06|NN|Mixed\d+)\.csv$", re.IGNORECASE)


HEADERS = {"User-Agent": "battherm-downloader/0.1"}  # the API rejects urllib's default agent


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)


def _get_json(url: str):
    with _open(url) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch(url: str, dest: Path) -> None:
    with _open(url) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hg2(out_dir: Path, log=print) -> list[Path]:
    meta = _get_json(API)  # latest version; checked below
    if meta.get("version") != VERSION:
        raise RuntimeError(f"expected dataset version {VERSION}, API returned {meta.get('version')}")
    folders = {f["id"]: (f["name"], f.get("parent_id")) for f in _get_json(f"{API}/folders/{VERSION}")}

    def folder_path(fid: str) -> list[str]:
        parts = []
        while fid in folders:
            name, fid = folders[fid]
            parts.append(name)
        return parts[::-1]

    wanted = []
    for f in meta["files"]:
        parts = folder_path(f.get("folder_id", ""))
        if parts[:1] == [ORIGINAL_ROOT] and len(parts) == 2 and DRIVE_FILE_RE.search(f["filename"]):
            wanted.append((parts[1], f))
    log(f"{len(wanted)} drive-cycle files, {sum(f['content_details']['size'] for _, f in wanted) / 1e6:.0f} MB")

    paths = []
    for k, (ambient_dir, f) in enumerate(sorted(wanted, key=lambda x: (x[0], x[1]["filename"])), 1):
        cd = f["content_details"]
        dest = Path(out_dir) / ambient_dir / f["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and _sha256(dest) == cd["sha256_hash"]:
            paths.append(dest)
            continue
        tmp = dest.with_suffix(".part")
        _fetch(cd["download_url"], tmp)
        if _sha256(tmp) != cd["sha256_hash"]:
            tmp.unlink()
            raise IOError(f"checksum mismatch for {f['filename']}")
        tmp.replace(dest)
        paths.append(dest)
        log(f"[{k}/{len(wanted)}] {ambient_dir}/{f['filename']} ({cd['size'] / 1e6:.1f} MB)")
    return paths
