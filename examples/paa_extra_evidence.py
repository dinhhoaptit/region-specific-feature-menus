"""Extra PAA evidence: regional menus as knowledge + localized Lasso neighbor.

Writes JSON under submission_paa/ and prints a LaTeX-ready table.
Uses the same splits/preprocessing as generate_paper_figures.py.
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
from src import compare_progressive_policies, fit_region_menus
from src.progressive import evaluate_region_menus_progressive


OUTDIR = ROOT / "submission_paa"
OUTDIR.mkdir(parents=True, exist_ok=True)


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def menu_report(model, k: int = 8) -> dict:
    menus = [list(m.selected) for m in model.menus]
    sizes = [int(m.n_samples) for m in model.menus]
    prefixes = [m[:k] for m in menus]
    pairwise = []
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            pairwise.append(
                {
                    "i": i,
                    "j": j,
                    "jaccard_top": jaccard(prefixes[i], prefixes[j]),
                    "jaccard_full": jaccard(menus[i], menus[j]),
                }
            )
    return {
        "n_regions": int(model.n_regions),
        "sizes": sizes,
        "menus": menus,
        "top": prefixes,
        "pairwise": pairwise,
        "mean_top_jaccard": float(np.mean([p["jaccard_top"] for p in pairwise]))
        if pairwise
        else 1.0,
    }


def rmse_at(curve, b: int) -> float | None:
    for bb, v in zip(curve.budgets, curve.rmse):
        if int(bb) == int(b):
            return float(v)
    return None


def run_dataset(name: str, X, y, *, n_regions: int, max_features: int, seed: int = 0):
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    common = dict(
        n_regions=n_regions,
        partition="residual",
        assignment="soft",
        soft_temperature=1.0,
        min_region_size=min_size,
        max_features=max_features,
        random_state=seed,
    )
    print(f"[fit] {name} stepwise menus", flush=True)
    stepwise = fit_region_menus(X_train, y_train, selector="forward_stepwise", **common)
    print(f"[fit] {name} region Lasso-LARS menus", flush=True)
    lasso = fit_region_menus(X_train, y_train, selector="lasso_lars", **common)

    policies = {
        "region_menu": {**{k: v for k, v in common.items() if k != "random_state"}, "selector": "forward_stepwise"},
        "region_lasso": {**{k: v for k, v in common.items() if k != "random_state"}, "selector": "lasso_lars"},
        "global_stepwise": {
            "n_regions": 1,
            "selector": "forward_stepwise",
            "partition": "kmeans",
            "assignment": "hard",
            "min_region_size": 1,
            "max_features": max_features,
        },
    }
    probe_len = max(len(m.selected) for m in stepwise.menus)
    max_b = min(max(probe_len, 8), max_features)
    budgets = list(range(0, max_b + 1))
    print(f"[eval] {name} progressive curves", flush=True)
    curves = compare_progressive_policies(
        X_train,
        y_train,
        X_test,
        y_test,
        region_policies=policies,
        budgets=budgets,
        include_baselines=False,
        random_state=seed,
    )
    # Also evaluate the already-fitted lasso model on the same budgets
    # (compare_progressive_policies refits; keep explicit curve too).
    lasso_curve = evaluate_region_menus_progressive(
        lasso, X_test, y_test, budgets=budgets, name="region_lasso_fitted"
    )
    curve_dict = {}
    for c in list(curves) + [lasso_curve]:
        curve_dict[c.name] = {
            "budgets": [int(b) for b in c.budgets],
            "rmse": [float(v) for v in c.rmse],
        }

    out = {
        "name": name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "p": int(X_train.shape[1]),
        "stepwise_menus": menu_report(stepwise),
        "lasso_menus": menu_report(lasso),
        "curves": curve_dict,
        "rmse_selected": {
            "region_stepwise_b5": rmse_at(next(c for c in curves if c.name == "region_menu"), 5),
            "region_lasso_b5": rmse_at(next(c for c in curves if c.name == "region_lasso"), 5),
            "global_stepwise_b5": rmse_at(next(c for c in curves if c.name == "global_stepwise"), 5),
            "region_stepwise_b25": rmse_at(next(c for c in curves if c.name == "region_menu"), 25),
            "region_lasso_b25": rmse_at(next(c for c in curves if c.name == "region_lasso"), 25),
            "global_stepwise_b25": rmse_at(next(c for c in curves if c.name == "global_stepwise"), 25),
        },
    }
    return out


def main():
    results = {}
    X, y = load_superconductivity()
    results["superconductivity"] = run_dataset(
        "superconductivity", X, y, n_regions=4, max_features=25
    )
    X, y = load_ct_slices(max_rows=8000, seed=0)
    results["ct_8k"] = run_dataset("ct_8k", X, y, n_regions=4, max_features=25)

    path = OUTDIR / "extra_evidence.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}", flush=True)
    for key, block in results.items():
        print("====", key, flush=True)
        sm = block["stepwise_menus"]
        print("sizes", sm["sizes"], "mean top-8 Jaccard", round(sm["mean_top_jaccard"], 3))
        for i, top in enumerate(sm["top"]):
            print(f"  region {i} n={sm['sizes'][i]} top8={top}")
        print("rmse", block["rmse_selected"])


if __name__ == "__main__":
    main()
