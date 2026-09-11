"""Job (3): train-only rule for global vs regional menus.

Fit both policies on an inner training slice, pick by validation RMSE at
budget 5 (also report train BIC and early-menu Jaccard), then refit the
winner on the full training split. Test RMSE is never used to choose.

    python examples/paa_when_to_regionalize.py --all --max-rows 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import (
    load_ct_slices,
    load_superconductivity,
    load_tecator,
    load_topo_2_1,
    make_hard_synthetic,
)
from examples.generate_paper_figures import prepare_xy
from src.progressive import evaluate_region_menus_progressive
from src.region_menus import fit_region_menus
from src.regionalize import (
    mean_region_vs_global_jaccard,
    pick_by_inner_val,
    pick_by_menu_jaccard,
    pick_by_train_bic,
)

OUT = ROOT / "submission_paa" / "when_to_regionalize.json"
BUDGET = 5


def _at(curve, b: int) -> float | None:
    for bb, v in zip(curve.budgets, curve.rmse):
        if int(bb) == int(b):
            return float(v)
    return None


def _min_size(n_train: int, n_regions: int) -> int:
    return max(40, min(200, n_train // max(n_regions * 5, 1)))


def _fit(X, y, *, n_regions: int, max_features: int, seed: int):
    return fit_region_menus(
        X,
        y,
        n_regions=n_regions,
        partition="residual" if n_regions > 1 else "kmeans",
        assignment="soft" if n_regions > 1 else "hard",
        soft_temperature=1.0,
        min_region_size=_min_size(len(y), n_regions) if n_regions > 1 else 1,
        max_features=max_features,
        selector="forward_stepwise",
        random_state=seed,
    )


def run_one(
    name: str,
    X,
    y,
    *,
    n_regions: int,
    max_features: int,
    seed: int,
):
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )
    print(
        f"[fit] {name} seed={seed} n_fit={len(y_fit)} n_val={len(y_val)} p={X_train.shape[1]}",
        flush=True,
    )
    region_fit = _fit(X_fit, y_fit, n_regions=n_regions, max_features=max_features, seed=seed)
    glob_fit = _fit(X_fit, y_fit, n_regions=1, max_features=max_features, seed=seed)

    pick_val, val_r, val_g = pick_by_inner_val(
        region_fit, glob_fit, X_val, y_val, budget=BUDGET
    )
    pick_bic, bic_r, bic_g = pick_by_train_bic(
        region_fit, glob_fit, X_fit, y_fit, budget=BUDGET
    )
    pick_jac, jac = pick_by_menu_jaccard(region_fit, k=5, max_overlap=0.35)
    jac_vs_g = mean_region_vs_global_jaccard(
        region_fit, glob_fit.menus[0].selected, k=5
    )

    region_full = _fit(X_train, y_train, n_regions=n_regions, max_features=max_features, seed=seed)
    glob_full = _fit(X_train, y_train, n_regions=1, max_features=max_features, seed=seed)
    budgets = list(range(0, max_features + 1))
    rc = evaluate_region_menus_progressive(region_full, X_test, y_test, budgets=budgets)
    gc = evaluate_region_menus_progressive(glob_full, X_test, y_test, budgets=budgets)
    report_bs = tuple(b for b in (1, 5, 13, 25) if b <= max_features)
    test_r = {str(b): _at(rc, b) for b in report_bs}
    test_g = {str(b): _at(gc, b) for b in report_bs}
    oracle = "region" if test_r["5"] <= test_g["5"] else "global"

    def chosen_test(pick: str) -> dict:
        return test_r if pick == "region" else test_g

    row = {
        "dataset": name,
        "seed": seed,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "p": int(X_train.shape[1]),
        "n_regions": n_regions,
        "val_rmse_b5": {"region": val_r, "global": val_g},
        "train_bic_b5": {"region": bic_r, "global": bic_g},
        "mean_top5_jaccard": jac,
        "mean_region_vs_global_top5_jaccard": jac_vs_g,
        "pick_inner_val": pick_val,
        "pick_train_bic": pick_bic,
        "pick_jaccard": pick_jac,
        "oracle_test_b5": oracle,
        "test_rmse": {"region": test_r, "global": test_g},
        "test_b5_inner_val": chosen_test(pick_val)["5"],
        "test_b5_train_bic": chosen_test(pick_bic)["5"],
        "test_b5_jaccard": chosen_test(pick_jac)["5"],
        "test_b5_oracle": chosen_test(oracle)["5"],
        "inner_val_matches_oracle": pick_val == oracle,
        "train_bic_matches_oracle": pick_bic == oracle,
        "jaccard_matches_oracle": pick_jac == oracle,
    }
    print(
        f"  val r/g={val_r:.4f}/{val_g:.4f} pick_val={pick_val} "
        f"bic={pick_bic} jac={jac:.3f}->{pick_jac} oracle={oracle} "
        f"test_b5 r/g={test_r['5']:.4f}/{test_g['5']:.4f}",
        flush=True,
    )
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return {
        "n_splits": n,
        "inner_val_accuracy": float(np_mean([r["inner_val_matches_oracle"] for r in rows])),
        "train_bic_accuracy": float(np_mean([r["train_bic_matches_oracle"] for r in rows])),
        "jaccard_accuracy": float(np_mean([r["jaccard_matches_oracle"] for r in rows])),
        "mean_test_b5_inner_val": float(np_mean([r["test_b5_inner_val"] for r in rows])),
        "mean_test_b5_always_region": float(
            np_mean([r["test_rmse"]["region"]["5"] for r in rows])
        ),
        "mean_test_b5_always_global": float(
            np_mean([r["test_rmse"]["global"]["5"] for r in rows])
        ),
        "mean_test_b5_oracle": float(np_mean([r["test_b5_oracle"] for r in rows])),
        "oracle_picks_region": float(
            np_mean([r["oracle_test_b5"] == "region" for r in rows])
        ),
    }


def np_mean(xs):
    import numpy as np

    return float(np.mean(xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ct", action="store_true")
    parser.add_argument("--sc", action="store_true")
    parser.add_argument("--tecator", action="store_true")
    parser.add_argument("--topo", action="store_true")
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.all:
        args.ct = args.sc = args.tecator = args.topo = True

    jobs = [
        (
            "synthetic_p80",
            lambda: make_hard_synthetic(n_per_region=250, p=80, seed=0, n_latents=8, k_inf=8),
            3,
            15,
        )
    ]
    if args.ct:
        jobs.append(
            ("ct_slices", lambda: load_ct_slices(max_rows=args.max_rows, seed=0), 4, 25)
        )
    if args.sc:
        jobs.append(("superconductivity", lambda: load_superconductivity(max_rows=None, seed=0), 4, 25))
    if args.tecator:
        jobs.append(("tecator", load_tecator, 3, 20))
    if args.topo:
        jobs.append(("topo_2_1", lambda: load_topo_2_1(max_rows=None, seed=0), 3, 25))

    by_ds = {}
    all_rows = []
    for name, loader, n_regions, max_features in jobs:
        X, y = loader()
        rows = []
        for seed in args.seeds:
            rows.append(
                run_one(
                    name,
                    X,
                    y,
                    n_regions=n_regions,
                    max_features=max_features,
                    seed=seed,
                )
            )
        by_ds[name] = {"splits": rows, "summary": summarize(rows)}
        all_rows.extend(rows)
        print(f"[summary] {name} {by_ds[name]['summary']}", flush=True)

    payload = {
        "job": "when_to_regionalize",
        "budget": BUDGET,
        "jaccard_max_overlap": 0.35,
        "datasets": by_ds,
        "overall": summarize(all_rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print("OVERALL", payload["overall"])


if __name__ == "__main__":
    main()
