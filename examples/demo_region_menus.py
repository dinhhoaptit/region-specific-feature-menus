"""Demo: region-specific incremental feature menus with VIF selection.

Synthetic data has three regions with different true predictors. We cluster,
run VIF inside each region, and print the resulting feature menus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fit_region_menus


def make_synthetic(n_per_region: int = 250, p: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    feature_names = [f"x{j}" for j in range(p)]

    # Three well-separated regions in the first two coordinates.
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    # Region-specific true supports (remaining features are noise).
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

    Xs, ys, region_ids = [], [], []
    for rid, center in enumerate(centers):
        X = rng.normal(0.0, 1.0, size=(n_per_region, p))
        X[:, 0] += center[0]
        X[:, 1] += center[1]
        support = true_supports[rid]
        y = X[:, support] @ true_coefs[rid] + rng.normal(0.0, 0.5, size=n_per_region)
        Xs.append(X)
        ys.append(y)
        region_ids.append(np.full(n_per_region, rid))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    region_ids = np.concatenate(region_ids)
    return X, y, feature_names, true_supports, region_ids


def main() -> None:
    X, y, names, true_supports, true_regions = make_synthetic()
    menus = fit_region_menus(
        X,
        y,
        n_regions=3,
        feature_names=names,
        partition="residual",
        assignment="soft",
        random_state=0,
        min_region_size=30,
    )

    print("Region-specific VIF feature menus")
    print("=" * 48)
    for row in menus.summary():
        rid = row["region"]
        print(
            f"Region {rid}: n={row['n_samples']}, "
            f"selected={row['n_selected']}, menu={row['menu']}"
        )
        print(f"  true support (by construction): {[names[j] for j in true_supports[rid]]}")

    # Sanity: majority of training points mapped back to their planted region
    # after unsupervised clustering may permute labels — compare supports instead.
    print("\nQuery examples (assign region + return menu)")
    print("-" * 48)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(X), size=5, replace=False):
        rid, menu = menus.menu_for_query(X[i], named=True)
        print(f"row {i:4d} -> region {rid}, menu={menu}, y={y[i]:.2f}, yhat={menus.predict(X[i])[0]:.2f}")

    yhat = menus.predict(X)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    print(f"\nIn-sample RMSE with region models: {rmse:.4f}")


if __name__ == "__main__":
    main()
