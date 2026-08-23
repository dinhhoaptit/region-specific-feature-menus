"""Local dataset loaders for paper experiments.

All public datasets are expected under project ``data/*.npz``, produced by::

    python examples/download_datasets.py

No network access is required at experiment time. Synthetic controls are generated
in-memory (see ``make_hard_synthetic``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def data_dir() -> Path:
    return DATA


def _require_npz(name: str) -> Path:
    path = DATA / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing local dataset file: {path}\n"
            f"Run: python examples/download_datasets.py\n"
            f"If that fails for this dataset, download it manually into {DATA}."
        )
    return path


def _load_npz(name: str, max_rows: int | None = None, seed: int = 0):
    path = _require_npz(name)
    z = np.load(path)
    X = np.asarray(z["X"], dtype=float)
    y = np.asarray(z["y"], dtype=float).ravel()
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_ct_slices(max_rows: int | None = None, seed: int = 0):
    """UCI Relative location of CT slices: n~53500, p=384 (local ``ct_slices.npz``)."""
    return _load_npz("ct_slices", max_rows=max_rows, seed=seed)


def load_superconductivity(max_rows: int | None = None, seed: int = 0):
    """UCI superconductivity: n=21263, p=81 (local ``superconductivity.npz``)."""
    return _load_npz("superconductivity", max_rows=max_rows, seed=seed)


# Backward-compatible alias used by older scripts.
load_superconductivity_uci = load_superconductivity


def load_tecator(max_rows: int | None = None, seed: int = 0):
    """OpenML tecator spectra → fat: n=240, p≈124 (local ``tecator.npz``)."""
    return _load_npz("tecator", max_rows=max_rows, seed=seed)


def load_topo_2_1(max_rows: int | None = None, seed: int = 0):
    """OpenML topo_2_1 weak-signal control (local ``topo_2_1.npz``)."""
    return _load_npz("topo_2_1", max_rows=max_rows, seed=seed)


def load_year_msd(max_rows: int | None = 50000, seed: int = 0):
    """Optional UCI YearPredictionMSD stress set (local ``year_msd.npz``)."""
    return _load_npz("year_msd", max_rows=max_rows, seed=seed)


def make_hard_synthetic(
    *,
    n_per_region: int = 1200,
    p: int = 1000,
    seed: int = 0,
    n_latents: int = 25,
    k_inf: int = 15,
):
    """Hard correlated synthetic with three planted region-specific supports."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])
    Xs, ys = [], []
    for c in centers:
        n = n_per_region
        latents = rng.normal(size=(n, n_latents))
        loadings = rng.normal(scale=0.7, size=(n_latents, p))
        noise = rng.normal(scale=np.sqrt(0.51), size=(n, p))
        X = latents @ loadings + noise
        X[:, 0] += c[0]
        X[:, 1] += c[1]
        support = np.sort(rng.choice(np.arange(2, p), size=k_inf, replace=False))
        coef = rng.normal(scale=2.0, size=k_inf)
        y = X[:, support] @ coef + rng.normal(scale=1.0, size=n)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def list_local_datasets() -> list[str]:
    if not DATA.exists():
        return []
    return sorted(p.stem for p in DATA.glob("*.npz"))


if __name__ == "__main__":
    print(f"data dir: {DATA}")
    print("local npz:", list_local_datasets())
    for name, loader in [
        ("ct_slices", lambda: load_ct_slices(max_rows=2000)),
        ("superconductivity", lambda: load_superconductivity(max_rows=2000)),
        ("tecator", load_tecator),
        ("topo_2_1", lambda: load_topo_2_1(max_rows=2000)),
    ]:
        try:
            X, y = loader()
            print(f"{name}: n={X.shape[0]} p={X.shape[1]}", flush=True)
        except FileNotFoundError as exc:
            print(f"{name}: MISSING ({exc})", flush=True)
    Xs, ys = make_hard_synthetic(n_per_region=100, p=50, seed=0)
    print(f"synthetic: n={Xs.shape[0]} p={Xs.shape[1]}", flush=True)
