"""Hard large-p and large-n benchmark (forward_stepwise region menus).

All public datasets are read from local ``data/*.npz`` files.
Run ``python examples/download_datasets.py`` once before experiments.
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
    load_ct_slices,
    load_superconductivity,
    load_tecator,
    load_topo_2_1,
    make_hard_synthetic,
)
from src import compare_progressive_policies, fit_region_menus, format_curves_table

warnings.filterwarnings("ignore")


def prepare_xy(X, y, seed: int = 0, test_size: float = 0.25):
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


def run_dataset(name: str, X, y, *, n_regions: int = 4, max_features: int = 30, seed: int = 0):
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    # Cap test size for speed on huge sets while keeping train large.
    if len(y_test) > 5000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_test), size=5000, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    print("\n" + "=" * 78)
    print(
        f"Dataset: {name}  "
        f"(n_train={len(y_train)}, n_test={len(y_test)}, p={X_train.shape[1]})"
    )
    print("=" * 78)

    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    base = {
        "selector": "forward_stepwise",
        "assignment": "soft",
        "soft_temperature": 1.0,
        "min_region_size": min_size,
        "max_features": max_features,
    }
    policies = {
        "region_residual": {**base, "n_regions": n_regions, "partition": "residual"},
        "region_kmeans": {**base, "n_regions": n_regions, "partition": "kmeans"},
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
    max_b = min(max(max(len(m.selected) for m in probe.menus), 8), max_features)
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
        preview = row["menu"][:8]
        more = "" if len(row["menu"]) <= 8 else f" ...(+{len(row['menu']) - 8})"
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
    print("Hard large-p / large-n benchmark (local data/ only).")

    # Large-p + large-n real data (CT slices).
    for max_rows in (8000, 20000, None):
        tag = "full" if max_rows is None else str(max_rows)
        print(f"\nLoading local CT slices ({tag})...", flush=True)
        X, y = load_ct_slices(max_rows=max_rows, seed=0)
        run_dataset(
            f"uci_ct_slices_p384_n{len(y)}",
            X,
            y,
            n_regions=4,
            max_features=25,
        )

    print("\nLoading local superconductivity...", flush=True)
    X, y = load_superconductivity(max_rows=None)
    run_dataset(
        f"superconduct_p{X.shape[1]}_n{len(y)}",
        X,
        y,
        n_regions=4,
        max_features=25,
    )

    print("\nLoading local topo_2_1...", flush=True)
    X, y = load_topo_2_1(max_rows=None)
    run_dataset(
        f"topo_2_1_p{X.shape[1]}_n{len(y)}",
        X,
        y,
        n_regions=3,
        max_features=25,
    )

    print("\nLoading local tecator...", flush=True)
    X, y = load_tecator()
    run_dataset(
        f"tecator_p{X.shape[1]}_n{len(y)}",
        X,
        y,
        n_regions=3,
        max_features=20,
    )

    # Hard synthetic: generated locally (no download).
    for p, n_per in ((1000, 1500), (2000, 1200)):
        X, y = make_hard_synthetic(n_per_region=n_per, p=p, seed=0)
        run_dataset(
            f"synthetic_hard_p{p}_n{len(y)}",
            X,
            y,
            n_regions=3,
            max_features=25,
        )


if __name__ == "__main__":
    main()
