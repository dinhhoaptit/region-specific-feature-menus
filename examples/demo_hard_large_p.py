"""Harder large-p benchmark for forward_stepwise region menus.

Uses local ``data/*.npz`` (see ``examples/download_datasets.py``) plus
in-memory hard synthetics with p in {500, 1000, 2000}.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import (
    load_superconductivity,
    load_tecator,
    load_topo_2_1,
)
from src import compare_progressive_policies, fit_region_menus, format_curves_table

warnings.filterwarnings("ignore")


def prepare_xy(X, y, seed: int = 0, test_size: float = 0.3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, X_test, y_train, y_test


def make_hard_synthetic(
    *,
    n_per_region: int = 400,
    p: int = 1000,
    n_informative_per_region: int = 12,
    noise: float = 1.0,
    corr: float = 0.7,
    seed: int = 0,
):
    """Region-heterogeneous high-p regression with correlated noise features."""
    rng = np.random.default_rng(seed)
    n_regions = 3
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 7.0]])

    # Shared latent factors to induce correlation among noise features.
    n_latents = 20
    Xs, ys = [], []
    for rid in range(n_regions):
        n = n_per_region
        latents = rng.normal(size=(n, n_latents))
        loadings = rng.normal(scale=corr, size=(n_latents, p))
        noise_mat = rng.normal(scale=np.sqrt(max(1.0 - corr**2, 1e-6)), size=(n, p))
        X = latents @ loadings + noise_mat
        X[:, 0] += centers[rid, 0]
        X[:, 1] += centers[rid, 1]

        support = rng.choice(np.arange(2, p), size=n_informative_per_region, replace=False)
        support = np.sort(support)
        coef = rng.normal(scale=2.0, size=n_informative_per_region)
        y = X[:, support] @ coef + rng.normal(scale=noise, size=n)
        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.concatenate(ys)


def run_dataset(
    name: str,
    X,
    y,
    *,
    n_regions: int = 3,
    seed: int = 0,
    max_features: int = 30,
) -> None:
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    print("\n" + "=" * 78)
    print(
        f"Dataset: {name}  "
        f"(n_train={len(y_train)}, n_test={len(y_test)}, p={X_train.shape[1]})"
    )
    print("=" * 78)

    min_size = max(25, min(80, len(y_train) // (n_regions * 6)))
    base = {
        "selector": "forward_stepwise",
        "assignment": "soft",
        "soft_temperature": 1.0,
        "min_region_size": min_size,
        "max_features": max_features,
    }
    policies = {
        "region_residual": {
            **base,
            "n_regions": n_regions,
            "partition": "residual",
        },
        "region_kmeans": {
            **base,
            "n_regions": n_regions,
            "partition": "kmeans",
        },
        "global_one_region": {
            **base,
            "n_regions": 1,
            "partition": "kmeans",
            "assignment": "hard",
        },
    }

    probe = fit_region_menus(
        X_train, y_train, random_state=seed, **policies["region_residual"]
    )
    max_b = max(max(len(m.selected) for m in probe.menus), 8)
    max_b = min(max_b, max_features)
    budgets = list(range(0, max_b + 1))

    curves = compare_progressive_policies(
        X_train,
        y_train,
        X_test,
        y_test,
        region_policies=policies,
        budgets=budgets,
        include_baselines=True,
        random_state=seed,
    )
    print("\nRMSE vs feature budget")
    print(format_curves_table(curves))

    print("\nMenus [forward_stepwise / residual/soft]:")
    for row in probe.summary():
        preview = row["menu"][:10]
        more = "" if len(row["menu"]) <= 10 else f" ...(+{len(row['menu']) - 10})"
        print(
            f"  region {row['region']}: n={row['n_samples']}, "
            f"k={row['n_selected']}, menu={preview}{more}"
        )

    for b in sorted({min(5, budgets[-1]), budgets[len(budgets) // 2], budgets[-1]}):
        ranked = []
        for c in curves:
            idx = np.where(c.budgets == b)[0]
            if len(idx):
                ranked.append((c.rmse[idx[0]], c.name))
        ranked.sort()
        print(
            f"\nAt budget={b}, best: "
            + ", ".join(f"{n}={r:.4f}" for r, n in ranked[:4])
        )


def main() -> None:
    print("Hard large-p benchmark (local data/ + in-memory synthetics).")

    # 1) Hard synthetic ladder.
    for p in (500, 1000, 2000):
        X, y = make_hard_synthetic(p=p, n_per_region=350, n_informative_per_region=12)
        run_dataset(
            f"synthetic_hard_p{p}",
            X,
            y,
            n_regions=3,
            max_features=25,
        )

    # 2) Real / public sets from local data/
    real_specs = [
        ("tecator_p124", load_tecator, None, 3, 25),
        ("topo_2_1_p266", load_topo_2_1, 8000, 3, 30),
        ("superconduct_p81", load_superconductivity, 8000, 4, 30),
    ]
    for tag, loader, max_rows, n_regions, max_features in real_specs:
        print(f"\nLoading {tag} ...", flush=True)
        X, y = loader(max_rows=max_rows)
        print(f"  got n={X.shape[0]}, p={X.shape[1]}", flush=True)
        run_dataset(tag, X, y, n_regions=n_regions, max_features=max_features)


if __name__ == "__main__":
    main()
