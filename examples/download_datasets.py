"""Download all paper experiment datasets into the project ``data/`` folder.

Usage (from the project root or ``examples/``)::

    python examples/download_datasets.py

Requires network access once. Afterward, loaders in ``data_loaders.py`` read
only from local files under ``data/``.
"""

from __future__ import annotations

import io
import json
import re
import socket
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
socket.setdefaulttimeout(120)

# Paper datasets (plus optional YearPredictionMSD stress set).
DATASETS = ("ct_slices", "superconductivity", "tecator", "topo_2_1")
OPTIONAL = ("year_msd",)


def _get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; local-research/1.0)"})
    with urlopen(req, timeout=120) as r:
        return r.read()


def _save_npz(name: str, X: np.ndarray, y: np.ndarray) -> Path:
    out = DATA / f"{name}.npz"
    np.savez_compressed(out, X=X.astype(np.float64), y=y.astype(np.float64))
    print(f"  saved {out.name}: n={X.shape[0]} p={X.shape[1]}", flush=True)
    return out


def _finite_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if hasattr(X, "toarray"):
        X = X.toarray()
    col_ok = np.isfinite(X).all(axis=0)
    X = X[:, col_ok]
    row_ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[row_ok], y[row_ok]


def download_ct_slices() -> Path:
    out = DATA / "ct_slices.npz"
    if out.exists():
        print(f"[skip] {out.name} already exists", flush=True)
        return out
    url = (
        "https://archive.ics.uci.edu/static/public/206/"
        "relative+location+of+ct+slices+on+axial+axis.zip"
    )
    print("Downloading CT slices (UCI 206)...", flush=True)
    raw = _get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            data = np.genfromtxt(f, delimiter=",", skip_header=1)
    X = data[:, 1:-1].astype(float)
    y = data[:, -1].astype(float)
    X, y = _finite_xy(X, y)
    return _save_npz("ct_slices", X, y)


def download_superconductivity() -> Path:
    out = DATA / "superconductivity.npz"
    if out.exists():
        print(f"[skip] {out.name} already exists", flush=True)
        return out
    url = (
        "https://archive.ics.uci.edu/static/public/464/"
        "superconductivty+data.zip"
    )
    print("Downloading superconductivity (UCI 464)...", flush=True)
    raw = _get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith("train.csv"))
        with zf.open(csv_name) as f:
            data = np.genfromtxt(f, delimiter=",", skip_header=1)
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(float)
    X, y = _finite_xy(X, y)
    return _save_npz("superconductivity", X, y)


def download_year_msd() -> Path:
    out = DATA / "year_msd.npz"
    if out.exists():
        print(f"[skip] {out.name} already exists", flush=True)
        return out
    url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
    print("Downloading YearPredictionMSD (UCI 203; large)...", flush=True)
    raw = _get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        txt = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(txt) as f:
            data = np.genfromtxt(f, delimiter=",")
    y = data[:, 0].astype(float)
    X = data[:, 1:].astype(float)
    X, y = _finite_xy(X, y)
    return _save_npz("year_msd", X, y)


def _openml_file_url(data_id: int) -> str:
    meta = json.loads(_get(f"https://www.openml.org/api/v1/json/data/{data_id}").decode("utf-8"))
    url = meta["data_set_description"]["url"]
    return str(url)


def _parse_arff_numeric(raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Minimal dense numeric ARFF reader (attributes + data)."""
    text = raw.decode("utf-8", errors="replace")
    # Strip % comments except inside quoted strings is rare in these files.
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        lines.append(s)

    attrs: list[str] = []
    data_start = None
    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith("@attribute"):
            # @attribute name type
            m = re.match(r"@attribute\s+('([^']+)'|\"([^\"]+)\"|([^\s]+))\s+", line, re.I)
            if m:
                name = m.group(2) or m.group(3) or m.group(4)
                attrs.append(name)
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("ARFF @data section not found")

    rows = []
    for line in lines[data_start:]:
        if line.lower().startswith("@"):
            continue
        # Simple CSV split; quoted fields uncommon for these numeric sets.
        parts = [p.strip().strip("'\"") for p in line.split(",")]
        if len(parts) != len(attrs):
            # skip malformed / sparse
            continue
        try:
            rows.append([float("nan" if p in {"?", ""} else p) for p in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No numeric ARFF rows parsed")
    return np.asarray(rows, dtype=float), attrs


def _download_openml_to_npz(name: str, data_id: int, target_col: str | int | None = None) -> Path:
    out = DATA / f"{name}.npz"
    if out.exists():
        print(f"[skip] {out.name} already exists", flush=True)
        return out
    print(f"Downloading {name} (OpenML id={data_id})...", flush=True)
    file_url = _openml_file_url(data_id)
    print(f"  url: {file_url}", flush=True)
    raw = _get(file_url)
    # Persist raw ARFF for inspection / offline reuse.
    arff_path = DATA / f"{name}.arff"
    arff_path.write_bytes(raw)
    print(f"  wrote {arff_path.name} ({len(raw)} bytes)", flush=True)

    data, attrs = _parse_arff_numeric(raw)
    if isinstance(target_col, int):
        y_idx = target_col
    elif isinstance(target_col, str):
        # case-insensitive match
        low = {a.lower(): i for i, a in enumerate(attrs)}
        if target_col.lower() not in low:
            raise KeyError(f"target '{target_col}' not in attributes: {attrs}")
        y_idx = low[target_col.lower()]
    else:
        y_idx = len(attrs) - 1

    y = data[:, y_idx]
    X = np.delete(data, y_idx, axis=1)
    X, y = _finite_xy(X, y)
    return _save_npz(name, X, y)


def download_tecator() -> Path:
    # OpenML 505: spectra + fat (paper uses fat content).
    # Prefer fat as last of moisture/fat/protein if present.
    out = DATA / "tecator.npz"
    if out.exists():
        print(f"[skip] {out.name} already exists", flush=True)
        return out
    print("Downloading tecator (OpenML 505)...", flush=True)
    file_url = _openml_file_url(505)
    print(f"  url: {file_url}", flush=True)
    raw = _get(file_url)
    (DATA / "tecator.arff").write_bytes(raw)
    data, attrs = _parse_arff_numeric(raw)
    low = [a.lower() for a in attrs]
    if "fat" in low:
        y_idx = low.index("fat")
    else:
        y_idx = len(attrs) - 1
    y = data[:, y_idx]
    X = np.delete(data, y_idx, axis=1)
    X, y = _finite_xy(X, y)
    return _save_npz("tecator", X, y)


def download_topo_2_1() -> Path:
    # Resolve id via name search if needed; known active id often 554.
    meta_list = json.loads(
        _get(
            "https://www.openml.org/api/v1/json/data/list/data_name/topo_2_1/"
            "limit/10/status/active"
        ).decode("utf-8")
    )
    datasets = meta_list.get("data", {}).get("dataset", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
    if not datasets:
        raise RuntimeError("OpenML search returned no topo_2_1 dataset")
    data_id = int(datasets[0]["did"])
    print(f"Resolved topo_2_1 -> OpenML id={data_id}", flush=True)
    return _download_openml_to_npz("topo_2_1", data_id)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    include_optional = "--optional" in argv
    names = list(DATASETS)
    if include_optional:
        names.extend(OPTIONAL)

    print(f"Data directory: {DATA}", flush=True)
    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            if name == "ct_slices":
                download_ct_slices()
            elif name == "superconductivity":
                download_superconductivity()
            elif name == "tecator":
                download_tecator()
            elif name == "topo_2_1":
                download_topo_2_1()
            elif name == "year_msd":
                download_year_msd()
            else:
                raise ValueError(name)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {name}: {failures[-1][1]}", flush=True)

    # Write a small manifest for local experiments.
    manifest = {
        "data_dir": str(DATA),
        "npz": sorted(p.name for p in DATA.glob("*.npz")),
        "failures": failures,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nManifest:", json.dumps(manifest, indent=2), flush=True)
    if failures:
        print("\nSome downloads failed. See messages above for manual steps.", flush=True)
        return 1
    print("\nAll requested datasets are available under data/.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
