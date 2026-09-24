"""Job (1): local residual-greedy as an instance-adaptive anytime competitor.

This is *not* global forward stepwise. Global stepwise already exists as a
baseline: it yields one menu for every query. Local residual-greedy (LRG)
chooses the next unused feature on a k-NN neighbourhood in the *already
observed* coordinates, then predicts with nested OLS on that query-specific set.

Run:
    python examples/conditional_greedy.py
    python examples/conditional_greedy.py --ct --max-rows 8000

Requires ``data/*.npz`` only for the optional public-data flags.
Writes ``submission_paa/conditional_greedy.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import make_hard_synthetic
from examples.generate_paper_figures import prepare_xy
from src.conditional_greedy import evaluate_conditional_greedy_progressive
from src.progressive import evaluate_global_menu_progressive, evaluate_region_menus_progressive
from src.region_menus import fit_region_menus
from src.selectors import select_forward_stepwise

OUT = ROOT / "submission_paa" / "conditional_greedy.json"


def _at(curve, b: int) -> float | None:
    for bb, v in zip(curve.budgets, curve.rmse):
        if int(bb) == int(b):
            return float(v)
    return None


def run_split(name: str, X, y, *, n_regions: int, max_features: int, seed: int = 0):
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    min_size = max(40, min(200, len(y_train) // max(n_regions * 5, 1)))
    print(f"[fit] {name} region menus n_train={len(y_train)} p={X_train.shape[1]}", flush=True)
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
    glob = select_forward_stepwise(X_train, y_train, max_features=max_features)
    budgets = list(range(0, max_features + 1))
    print(f"[eval] {name} region / global / LRG", flush=True)
    rc = evaluate_region_menus_progressive(region, X_test, y_test, budgets=budgets)
    gc = evaluate_global_menu_progressive(
        X_train, y_train, X_test, y_test, menu=glob.selected, budgets=budgets, name="global_stepwise"
    )
    lrg = evaluate_conditional_greedy_progressive(
        X_train,
        y_train,
        X_test,
        y_test,
        max_features=max_features,
        k_neighbors=80,
        n_index=min(6000, len(y_train)),
        budgets=budgets,
        random_state=seed,
    )
    row = {
        "dataset": name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "p": int(X_train.shape[1]),
        "n_unique_sets": lrg.n_unique_sets,
        "lrg_mean_path_jaccard": lrg.mean_path_jaccard,
        "rmse": {},
    }
    for b in (1, 5, 13, 25):
        if b > max_features:
            continue
        row["rmse"][str(b)] = {
            "region": _at(rc, b),
            "global_stepwise": _at(gc, b),
            "local_residual_greedy": _at(lrg.curve, b),
        }
    print(json.dumps(row["rmse"], indent=2))
    print(
        f"  LRG unique acquired-sets by budget: {lrg.n_unique_sets[:6]}... "
        f"path Jaccard={lrg.mean_path_jaccard:.3f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", action="store_true", help="Also run UCI CT slices")
    parser.add_argument("--sc", action="store_true", help="Also run superconductivity")
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--max-features", type=int, default=25)
    args = parser.parse_args()

    results = []
    print("[synthetic] small heterogeneous control", flush=True)
    Xs, ys = make_hard_synthetic(n_per_region=250, p=80, seed=0, n_latents=8, k_inf=8)
    results.append(
        run_split("synthetic_p80", Xs, ys, n_regions=3, max_features=min(15, args.max_features))
    )

    if args.ct:
        from examples.data_loaders import load_ct_slices

        X, y = load_ct_slices(max_rows=args.max_rows, seed=0)
        results.append(run_split("ct_slices", X, y, n_regions=4, max_features=args.max_features))
    if args.sc:
        from examples.data_loaders import load_superconductivity

        X, y = load_superconductivity(max_rows=None, seed=0)
        results.append(run_split("superconductivity", X, y, n_regions=4, max_features=args.max_features))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"jobs": "conditional_greedy", "results": results}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
