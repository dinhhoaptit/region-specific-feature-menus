"""Benchmark forward_stepwise region menus on larger-p public datasets.

Datasets are fetched automatically via sklearn / OpenML — no local files needed.
If OpenML is unreachable, the script still runs synthetic high-p + diabetes.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import compare_progressive_policies, fit_region_menus, format_curves_table


def _to_numeric_xy(bunch_or_xy):
    if isinstance(bunch_or_xy, tuple):
        X, y = bunch_or_xy
    else:
        X, y = bunch_or_xy.data, bunch_or_xy.target
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    # Drop non-finite columns/rows.
    col_ok = np.isfinite(X).all(axis=0)
    X = X[:, col_ok]
    row_ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[row_ok], y[row_ok]


def load_openml_regression(name: str, *, version: int | str = "active", max_rows: int | None = 6000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml(name=name, version=version, as_frame=False, parser="liac-arff")
    X, y = _to_numeric_xy(data)
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def prepare_xy(X, y, seed: int = 0, test_size: float = 0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Replace any leftover NaNs from constant columns.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, X_test, y_train, y_test


def run_dataset(
    name: str,
    X,
    y,
    *,
    n_regions: int = 3,
    seed: int = 0,
    max_features: int | None = 30,
) -> None:
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    print("\n" + "=" * 78)
    print(
        f"Dataset: {name}  "
        f"(n_train={len(y_train)}, n_test={len(y_test)}, p={X_train.shape[1]})"
    )
    print("=" * 78)

    base = {
        "n_regions": n_regions,
        "selector": "forward_stepwise",
        "partition": "residual",
        "assignment": "soft",
        "soft_temperature": 1.0,
        "min_region_size": max(40, len(y_train) // (n_regions * 8)),
        "max_features": max_features,
    }
    policies = {
        "region_fwd": dict(base),
        "region_fwd_kmeans": {
            **base,
            "partition": "kmeans",
        },
        "global_as_one_region": {
            **base,
            "n_regions": 1,
            "partition": "kmeans",
            "assignment": "hard",
        },
    }

    probe = fit_region_menus(X_train, y_train, random_state=seed, **policies["region_fwd"])
    max_b = max(len(m.selected) for m in probe.menus)
    max_b = max(max_b, 8)
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

    print("\nRegion menus [forward_stepwise / residual/soft]:")
    for row in probe.summary():
        menu_preview = row["menu"][:12]
        more = "" if len(row["menu"]) <= 12 else f" ...(+{len(row['menu']) - 12})"
        print(
            f"  region {row['region']}: n={row['n_samples']}, "
            f"k={row['n_selected']}, menu={menu_preview}{more}"
        )

    mid = budgets[len(budgets) // 2]
    for b in (min(5, budgets[-1]), mid, budgets[-1]):
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


def load_candidates() -> list[tuple[str, np.ndarray, np.ndarray, dict]]:
    """Return (name, X, y, run_kwargs) list; skip datasets that fail to download."""
    out: list[tuple[str, np.ndarray, np.ndarray, dict]] = []

    # Always-available controls.
    X, y = make_regression(
        n_samples=2000,
        n_features=200,
        n_informative=15,
        noise=8.0,
        random_state=0,
    )
    out.append(("synthetic_p200_informative15", X, y, {"n_regions": 3, "max_features": 25}))

    diabetes = load_diabetes()
    out.append(("sklearn_diabetes", diabetes.data, diabetes.target, {"n_regions": 3, "max_features": 10}))

    openml_specs = [
        # name, version/max_rows/regions/max_features
        ("space_ga", "active", 5000, 3, 20),
        ("cpu_act", "active", 5000, 3, 25),
        ("topo_2_1", "active", 5000, 3, 30),
        ("superconduct", "active", 6000, 4, 30),
        ("Yolanda", 1, 5000, 3, 30),
    ]
    for name, version, max_rows, n_regions, max_features in openml_specs:
        try:
            print(f"Fetching OpenML dataset: {name} ...")
            X, y = load_openml_regression(name, version=version, max_rows=max_rows)
            if X.shape[1] < 10:
                print(f"  skip {name}: only p={X.shape[1]}")
                continue
            print(f"  got n={X.shape[0]}, p={X.shape[1]}")
            out.append(
                (
                    f"openml_{name}_p{X.shape[1]}",
                    X,
                    y,
                    {"n_regions": n_regions, "max_features": max_features},
                )
            )
        except Exception as exc:  # network / schema issues
            print(f"  could not load {name}: {type(exc).__name__}: {exc}")
    return out


def main() -> None:
    print(
        "Large-p benchmark with selector=forward_stepwise.\n"
        "No local dataset files required (OpenML download if available)."
    )
    datasets = load_candidates()
    for name, X, y, kwargs in datasets:
        try:
            run_dataset(name, X, y, seed=0, **kwargs)
        except Exception as exc:
            print(f"\nFailed on {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
