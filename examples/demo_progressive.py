"""Benchmark progressive revelation: soft/y-aware region menus vs baselines.

Uses synthetic heterogeneous data plus built-in sklearn regression datasets.
No external data files are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import compare_progressive_policies, fit_region_menus, format_curves_table


def make_synthetic(n_per_region: int = 300, p: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    true_supports = {
        0: [2, 3, 4],
        1: [5, 6],
        2: [7, 8, 9, 10],
    }
    true_coefs = {
        0: np.array([2.0, -1.5, 1.0]),
        1: np.array([2.5, 2.0]),
        2: np.array([-2.0, 1.5, 1.0, -1.0]),
    }
    Xs, ys = [], []
    for rid, center in enumerate(centers):
        X = rng.normal(0.0, 1.0, size=(n_per_region, p))
        X[:, 0] += center[0]
        X[:, 1] += center[1]
        support = true_supports[rid]
        y = X[:, support] @ true_coefs[rid] + rng.normal(0.0, 0.5, size=n_per_region)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def prepare_xy(X, y, seed: int = 0, test_size: float = 0.3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def run_dataset(name: str, X, y, *, n_regions: int = 3, seed: int = 0) -> None:
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    print("\n" + "=" * 72)
    print(f"Dataset: {name}  (n_train={len(y_train)}, n_test={len(y_test)}, p={X_train.shape[1]})")
    print("=" * 72)

    policies = {
        "kmeans/hard": {
            "n_regions": n_regions,
            "partition": "kmeans",
            "assignment": "hard",
            "min_region_size": 30,
        },
        "kmeans/soft": {
            "n_regions": n_regions,
            "partition": "kmeans",
            "assignment": "soft",
            "soft_temperature": 1.0,
            "min_region_size": 30,
        },
        "residual/soft": {
            "n_regions": n_regions,
            "partition": "residual",
            "assignment": "soft",
            "soft_temperature": 1.0,
            "y_weight": 1.0,
            "min_region_size": 30,
        },
        "xy/soft": {
            "n_regions": n_regions,
            "partition": "xy",
            "assignment": "soft",
            "soft_temperature": 1.0,
            "y_weight": 1.0,
            "min_region_size": 30,
        },
    }

    # Align budgets across policies: 0..max selected among region models.
    probe = fit_region_menus(
        X_train, y_train, n_regions=n_regions, random_state=seed, min_region_size=30
    )
    max_b = max(len(m.selected) for m in probe.menus)
    max_b = max(max_b, 5)
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

    print("\nRegion menus (residual/soft):")
    model = fit_region_menus(
        X_train,
        y_train,
        n_regions=n_regions,
        partition="residual",
        assignment="soft",
        random_state=seed,
        min_region_size=30,
    )
    for row in model.summary():
        print(
            f"  region {row['region']}: n={row['n_samples']}, "
            f"menu={row['menu']}"
        )

    # Highlight best policy at mid budget and full budget.
    mid = budgets[len(budgets) // 2]
    for b in (mid, budgets[-1]):
        ranked = []
        for c in curves:
            idx = np.where(c.budgets == b)[0]
            if len(idx):
                ranked.append((c.rmse[idx[0]], c.name))
        ranked.sort()
        print(f"\nAt budget={b}, best RMSE: " + ", ".join(f"{n}={r:.4f}" for r, n in ranked[:3]))


def main() -> None:
    print("No external datasets required — using synthetic + sklearn builtins.")

    X_s, y_s = make_synthetic()
    run_dataset("synthetic_heterogeneous", X_s, y_s, n_regions=3, seed=0)

    diabetes = load_diabetes()
    run_dataset("sklearn_diabetes", diabetes.data, diabetes.target, n_regions=3, seed=0)

    housing = fetch_california_housing()
    # Subsample for speed while keeping signal.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(housing.target), size=4000, replace=False)
    run_dataset(
        "sklearn_california_housing_4k",
        housing.data[idx],
        housing.target[idx],
        n_regions=4,
        seed=0,
    )


if __name__ == "__main__":
    main()
