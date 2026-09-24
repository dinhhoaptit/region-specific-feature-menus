"""PAA steps 7-8: measurement budget to target error, and menu stability across splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import load_ct_slices, load_superconductivity
from examples.generate_paper_figures import prepare_xy
from src import fit_region_menus
from src.progressive import evaluate_region_menus_progressive
from src.selectors import select_forward_stepwise
from src.region_menus import _fit_prefix_models
from src.progressive import _metrics

OUTDIR = ROOT / "submission_paa"
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {"region": "#0072B2", "global": "#D55E00"}


def first_budget_at_or_below(budgets, rmse, target: float) -> int | None:
    for b, v in zip(budgets, rmse):
        if float(v) <= target + 1e-12:
            return int(b)
    return None


def nested_rmse_global(X_train, y_train, X_test, y_test, menu, budgets):
    prefixes = _fit_prefix_models(X_train, y_train, list(menu))
    out = []
    for b in budgets:
        bb = int(max(0, min(int(b), len(prefixes) - 1)))
        yhat = prefixes[bb].predict(X_test)
        rmse, _ = _metrics(y_test, yhat)
        out.append(rmse)
    return out


def one_split(X, y, *, n_regions: int, max_features: int, seed: int, topk: int = 8):
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    model = fit_region_menus(
        X_train,
        y_train,
        n_regions=n_regions,
        selector="forward_stepwise",
        partition="residual",
        assignment="soft",
        soft_temperature=1.0,
        min_region_size=min_size,
        max_features=max_features,
        random_state=seed,
    )
    glob = select_forward_stepwise(X_train, y_train, max_features=max_features)
    max_b = min(max_features, max(len(m.selected) for m in model.menus))
    budgets = list(range(0, max_b + 1))
    rc = evaluate_region_menus_progressive(model, X_test, y_test, budgets=budgets)
    g_rmse = nested_rmse_global(
        X_train, y_train, X_test, y_test, glob.selected, budgets
    )
    cover = set()
    for m in model.menus:
        cover.update(m.selected[:topk])
    return {
        "budgets": [int(b) for b in rc.budgets],
        "region_rmse": [float(v) for v in rc.rmse],
        "global_rmse": [float(v) for v in g_rmse],
        "region_cover_topk": sorted(int(j) for j in cover),
        "global_topk": [int(j) for j in glob.selected[:topk]],
        "region_topk_lists": [list(m.selected[:topk]) for m in model.menus],
    }


def pairwise_jaccard(sets: list[set]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            vals.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(vals)) if vals else 1.0


def run_stability(name, loader, *, n_regions, max_features, seeds, topk=8):
    X, y = loader()
    splits = []
    for seed in seeds:
        print(f"[stability] {name} seed={seed}", flush=True)
        splits.append(one_split(X, y, n_regions=n_regions, max_features=max_features, seed=seed, topk=topk))
    region_sets = [set(s["region_cover_topk"]) for s in splits]
    global_sets = [set(s["global_topk"]) for s in splits]
    p = int(np.asarray(X).shape[1])
    freq = np.zeros(p)
    for s in splits:
        for j in s["region_cover_topk"]:
            freq[j] += 1
    freq = freq / len(splits)
    n_recurring = int(np.sum(freq >= 0.5))
    return {
        "name": name,
        "n_splits": len(seeds),
        "topk": topk,
        "mean_jaccard_region_cover": pairwise_jaccard(region_sets),
        "mean_jaccard_global_topk": pairwise_jaccard(global_sets),
        "n_features_in_half_splits": n_recurring,
        "splits": splits,
    }


def budget_summary_from_split(split, targets=("global_b5", "global_b25")):
    budgets = split["budgets"]
    r = split["region_rmse"]
    g = split["global_rmse"]
    g_map = {int(b): v for b, v in zip(budgets, g)}
    out = {}
    if 5 in g_map:
        out["global_rmse_b5"] = g_map[5]
        out["region_b_to_match_global_b5"] = first_budget_at_or_below(budgets, r, g_map[5])
    if 25 in g_map:
        out["global_rmse_b25"] = g_map[25]
        out["region_b_to_match_global_b25"] = first_budget_at_or_below(budgets, r, g_map[25])
    out["region_rmse_b5"] = next((v for b, v in zip(budgets, r) if int(b) == 5), None)
    return out


def plot_budget_to_target(ct, sc):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, block, title in [
        (axes[0], ct, "CT slices ($n$=8k)"),
        (axes[1], sc, "Superconductivity"),
    ]:
        s0 = block["splits"][0]
        b = s0["budgets"]
        ax.plot(b, s0["region_rmse"], color=COLORS["region"], lw=2.3, label="Region menu")
        ax.plot(b, s0["global_rmse"], color=COLORS["global"], lw=2.1, label="Global stepwise")
        g5 = next(v for bb, v in zip(b, s0["global_rmse"]) if int(bb) == 5)
        g25 = next((v for bb, v in zip(b, s0["global_rmse"]) if int(bb) == 25), None)
        ax.axhline(g5, color=COLORS["global"], ls=":", lw=1.0, alpha=0.8)
        if g25 is not None:
            ax.axhline(g25, color=COLORS["global"], ls="--", lw=1.0, alpha=0.7)
        br5 = first_budget_at_or_below(b, s0["region_rmse"], g5)
        if br5 is not None:
            ax.scatter([br5], [s0["region_rmse"][b.index(br5)]], color=COLORS["region"], zorder=5)
            ax.annotate(
                f"region reaches\nglobal $b$=5 at $b$={br5}",
                xy=(br5, s0["region_rmse"][b.index(br5)]),
                xytext=(br5 + 3, s0["region_rmse"][b.index(br5)] + (0.8 if "CT" in title else 1.2)),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
            )
        ax.set_xlabel("Measurements allowed ($b$)")
        ax.set_ylabel("Progressive RMSE")
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Fewer measurements to match a global-menu error target", fontsize=12, y=1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig6_measurement_budget.{ext}", dpi=300, bbox_inches="tight")
        print("wrote", FIGDIR / f"fig6_measurement_budget.{ext}", flush=True)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "figure.dpi": 120})
    sc = run_stability(
        "superconductivity",
        load_superconductivity,
        n_regions=4,
        max_features=25,
        seeds=list(range(10)),
    )
    ct = run_stability(
        "ct_8k",
        lambda: load_ct_slices(max_rows=8000, seed=0),
        n_regions=4,
        max_features=25,
        seeds=list(range(5)),
    )
    # loader uses seed=0 for row subsample then split seed varies — for CT, subsample is fixed.

    summary = {
        "superconductivity": {
            **{k: v for k, v in sc.items() if k != "splits"},
            "seed0": budget_summary_from_split(sc["splits"][0]),
            "mean_region_b_to_match_global_b5": float(
                np.mean(
                    [
                        budget_summary_from_split(s).get("region_b_to_match_global_b5")
                        for s in sc["splits"]
                        if budget_summary_from_split(s).get("region_b_to_match_global_b5") is not None
                    ]
                )
            ),
            "mean_region_b_to_match_global_b25": float(
                np.nanmean(
                    [
                        np.nan
                        if budget_summary_from_split(s).get("region_b_to_match_global_b25") is None
                        else budget_summary_from_split(s)["region_b_to_match_global_b25"]
                        for s in sc["splits"]
                    ]
                )
            ),
        },
        "ct_8k": {
            **{k: v for k, v in ct.items() if k != "splits"},
            "seed0": budget_summary_from_split(ct["splits"][0]),
            "mean_region_b_to_match_global_b5": float(
                np.mean(
                    [
                        budget_summary_from_split(s)["region_b_to_match_global_b5"]
                        for s in ct["splits"]
                        if budget_summary_from_split(s).get("region_b_to_match_global_b5") is not None
                    ]
                )
            ),
            "mean_region_b_to_match_global_b25": float(
                np.nanmean(
                    [
                        np.nan
                        if budget_summary_from_split(s).get("region_b_to_match_global_b25") is None
                        else budget_summary_from_split(s)["region_b_to_match_global_b25"]
                        for s in ct["splits"]
                    ]
                )
            ),
        },
    }
    # keep compact split metrics only
    for key, raw in [("superconductivity", sc), ("ct_8k", ct)]:
        summary[key]["per_split"] = [budget_summary_from_split(s) for s in raw["splits"]]

    path = OUTDIR / "stability_and_budget.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    plot_budget_to_target(ct, sc)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
