"""Compare region menus with SAOLA-style and mRMR global acquisition menus.

Writes submission_kais/modern_baselines.json used by Table tab:modern.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import load_ct_slices, load_superconductivity
from examples.generate_paper_figures import prepare_xy
from src.progressive import evaluate_global_menu_progressive, evaluate_region_menus_progressive
from src.region_menus import fit_region_menus
from src.selectors import select_forward_stepwise, select_mrmr, select_saola

OUTDIR = ROOT / "submission_kais"
OUTDIR.mkdir(parents=True, exist_ok=True)


def rmse_at(curve, b: int) -> float | None:
    for bb, v in zip(curve.budgets, curve.rmse):
        if int(bb) == int(b):
            return float(v)
    return None


def run_split(name: str, X, y, *, n_regions: int, max_features: int, seed: int = 0) -> dict:
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    print(f"[fit] {name} region menus (seed={seed})", flush=True)
    region = fit_region_menus(
        X_train,
        y_train,
        n_regions=n_regions,
        partition="residual",
        assignment="soft",
        soft_temperature=1.0,
        min_region_size=min_size,
        max_features=max_features,
        selector="forward_stepwise",
        random_state=seed,
    )
    print(f"[fit] {name} global stepwise / SAOLA / mRMR", flush=True)
    g_step = select_forward_stepwise(X_train, y_train, max_features=max_features)
    g_saola = select_saola(X_train, y_train, max_features=max_features)
    g_mrmr = select_mrmr(X_train, y_train, max_features=max_features, random_state=seed)

    probe = max(len(m.selected) for m in region.menus)
    max_b = min(max(probe, 8), max_features)
    budgets = list(range(0, max_b + 1))

    curves = {
        "region_menu": evaluate_region_menus_progressive(
            region, X_test, y_test, budgets=budgets, name="region_menu"
        ),
        "global_stepwise": evaluate_global_menu_progressive(
            X_train, y_train, X_test, y_test, menu=g_step.selected, budgets=budgets, name="global_stepwise"
        ),
        "global_saola": evaluate_global_menu_progressive(
            X_train, y_train, X_test, y_test, menu=g_saola.selected, budgets=budgets, name="global_saola"
        ),
        "global_mrmr": evaluate_global_menu_progressive(
            X_train, y_train, X_test, y_test, menu=g_mrmr.selected, budgets=budgets, name="global_mrmr"
        ),
    }
    report = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "p": int(X_train.shape[1]),
        "menu_lengths": {
            "region_mean": float(np.mean([len(m.selected) for m in region.menus])),
            "global_stepwise": int(len(g_step.selected)),
            "global_saola": int(len(g_saola.selected)),
            "global_mrmr": int(len(g_mrmr.selected)),
        },
        "rmse": {
            key: {str(int(b)): rmse_at(curve, int(b)) for b in (1, 3, 5, 8, 10, 15, 20, 25) if rmse_at(curve, int(b)) is not None}
            for key, curve in curves.items()
        },
        "curves": {k: v.as_dict() for k, v in curves.items()},
    }
    return report


def main() -> None:
    out: dict = {"seed": 0, "datasets": {}}
    jobs = [
        ("ct_slices", lambda: load_ct_slices(max_rows=8000, seed=0), 4, 25),
        ("superconductivity", lambda: load_superconductivity(max_rows=None, seed=0), 4, 25),
    ]
    for name, loader, r, mmax in jobs:
        print(f"=== {name} ===", flush=True)
        X, y = loader()
        out["datasets"][name] = run_split(name, X, y, n_regions=r, max_features=mmax, seed=0)

    path = OUTDIR / "modern_baselines.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    for name, block in out["datasets"].items():
        print(f"\n{name}")
        for method, vals in block["rmse"].items():
            row = "  ".join(f"b{b}={vals[b]:.3f}" for b in sorted(vals, key=int) if b in ("5", "10", "15"))
            print(f"  {method:18s} {row}")


if __name__ == "__main__":
    main()
