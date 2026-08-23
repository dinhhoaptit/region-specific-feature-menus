"""Generate publication figures for the Knowledge-Based Systems manuscript.

Reads local ``data/*.npz`` produced by ``examples/download_datasets.py``.
Outputs PDF+PNG under submission_kbs/figures/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import (
    load_ct_slices,
    load_superconductivity,
    load_tecator,
    make_hard_synthetic,
)
from src import compare_progressive_policies, fit_region_menus

warnings.filterwarnings("ignore")

FIGDIR = ROOT / "submission_kbs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
CACHE = FIGDIR / "curve_cache.json"

# Colorblind-friendly palette
COLORS = {
    "region": "#0072B2",
    "global": "#D55E00",
    "abs_corr": "#009E73",
    "vif": "#CC79A7",
    "random": "#999999",
    "region_alt": "#56B4E9",
}


def prepare_xy(X, y, seed: int = 0, test_size: float = 0.25, max_test: int = 5000):
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
    if len(y_test) > max_test:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_test), size=max_test, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]
    return X_train, X_test, y_train, y_test


def run_curves(
    name: str,
    X,
    y,
    *,
    n_regions: int = 4,
    max_features: int = 25,
    seed: int = 0,
) -> dict:
    X_train, X_test, y_train, y_test = prepare_xy(X, y, seed=seed)
    print(
        f"[run] {name}: n_train={len(y_train)} n_test={len(y_test)} p={X_train.shape[1]}",
        flush=True,
    )
    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    policies = {
        "region_menu": {
            "n_regions": n_regions,
            "selector": "forward_stepwise",
            "partition": "residual",
            "assignment": "soft",
            "soft_temperature": 1.0,
            "min_region_size": min_size,
            "max_features": max_features,
        },
        "global_stepwise": {
            "n_regions": 1,
            "selector": "forward_stepwise",
            "partition": "kmeans",
            "assignment": "hard",
            "min_region_size": 1,
            "max_features": max_features,
        },
    }
    probe = fit_region_menus(X_train, y_train, random_state=seed, **policies["region_menu"])
    max_b = max(len(m.selected) for m in probe.menus)
    max_b = min(max(max_b, 8), max_features)
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
    out = {"name": name, "curves": {}}
    for c in curves:
        out["curves"][c.name] = {
            "budgets": [int(b) for b in c.budgets],
            "rmse": [float(v) for v in c.rmse],
        }
    return out


def style_ax(ax, xlabel="Feature budget $b$", ylabel="Progressive RMSE"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="best")


def pick_curve(curves: dict, *aliases: str):
    for a in aliases:
        if a in curves:
            return curves[a]
    raise KeyError(aliases)


def plot_progressive(ax, curves: dict, title: str):
    mapping = [
        ("region_menu", "Region menu", COLORS["region"], "-", 2.4),
        ("global_stepwise", "Global stepwise", COLORS["global"], "-", 2.2),
        ("global_vif", "Global VIF", COLORS["vif"], "--", 1.6),
        ("abs_corr", "Abs-corr", COLORS["abs_corr"], "--", 1.6),
        ("random", "Random", COLORS["random"], ":", 1.4),
    ]
    for key, label, color, ls, lw in mapping:
        if key not in curves:
            continue
        c = curves[key]
        ax.plot(c["budgets"], c["rmse"], label=label, color=color, ls=ls, lw=lw)
    ax.set_title(title, fontsize=11)
    style_ax(ax)


def savefig(fig, stem: str):
    for ext in ("pdf", "png"):
        path = FIGDIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}", flush=True)
    plt.close(fig)


def fig_method_schematic():
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, 2.2, 1.4, "Training data\n$(X, y)$"),
        (3.0, 1.0, 2.4, 1.4, "Partition into\nregions"),
        (5.9, 1.0, 2.6, 1.4, "Per-region\nforward stepwise\n(BIC menus)"),
        (9.0, 1.0, 2.6, 1.4, "Nested prefix\nOLS models"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="#E8F1F8",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    for x0, x1 in [(2.5, 3.0), (5.4, 5.9), (8.5, 9.0)]:
        ax.annotate(
            "",
            xy=(x1, 1.7),
            xytext=(x0, 1.7),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
        )

    ax.text(
        6.0,
        0.35,
        "Test: soft/hard region assignment $\\rightarrow$ acquire along local menu $\\rightarrow$ predict at budget $b$",
        ha="center",
        va="center",
        fontsize=9,
        color="#222222",
    )
    ax.set_title("Region-specific incremental feature menus", fontsize=12, pad=8)
    savefig(fig, "fig1_method_overview")


def fig_ct_scaling_from_paper():
    """Scaling panel using manuscript table values (budgets 5 and 25)."""
    n = np.array([8000, 20000, 53500])
    region_b5 = np.array([9.74, 9.63, 9.85])
    global_b5 = np.array([13.09, 13.14, 13.43])
    region_b25 = np.array([7.17, 7.26, 7.36])
    global_b25 = np.array([9.86, 10.01, 9.97])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, r, g, b in [
        (axes[0], region_b5, global_b5, 5),
        (axes[1], region_b25, global_b25, 25),
    ]:
        ax.plot(n, r, "o-", color=COLORS["region"], lw=2.2, label="Region menu")
        ax.plot(n, g, "s-", color=COLORS["global"], lw=2.2, label="Global stepwise")
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels(["8k", "20k", "53.5k"])
        ax.set_xlabel("Sample size $n$")
        ax.set_ylabel("Progressive RMSE")
        ax.set_title(f"CT slices, budget $b={b}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(frameon=False)
    fig.suptitle("Gains persist as $n$ grows (Claim C2)", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig3_ct_scaling")


def fig_summary_bars(results: dict):
    """Relative RMSE reduction at budget 5 and mid/high budget."""
    rows = []
    for key, label in [
        ("ct_8k", "CT ($n$=8k)"),
        ("superconductivity", "Supercond."),
        ("synthetic_p1000", "Synthetic $p$=1k"),
    ]:
        if key not in results:
            continue
        curves = results[key]["curves"]
        try:
            reg = pick_curve(curves, "region_menu")
            glob = pick_curve(curves, "global_stepwise")
        except KeyError:
            continue
        # Align on common budgets
        b_set = sorted(set(reg["budgets"]) & set(glob["budgets"]))
        if 5 in b_set:
            i5 = b_set.index(5)
            # map to indices
            ir = reg["budgets"].index(5)
            ig = glob["budgets"].index(5)
            red5 = 100.0 * (1.0 - reg["rmse"][ir] / glob["rmse"][ig])
        else:
            red5 = np.nan
        b_hi = b_set[min(len(b_set) - 1, max(1, len(b_set) * 3 // 4))]
        ir = reg["budgets"].index(b_hi)
        ig = glob["budgets"].index(b_hi)
        red_hi = 100.0 * (1.0 - reg["rmse"][ir] / glob["rmse"][ig])
        rows.append((label, red5, red_hi, b_hi))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(rows))
    w = 0.36
    r5 = [r[1] for r in rows]
    rh = [r[2] for r in rows]
    ax.bar(x - w / 2, r5, w, color=COLORS["region"], label="Budget $b=5$")
    ax.bar(
        x + w / 2,
        rh,
        w,
        color=COLORS["region_alt"],
        label="Higher budget",
    )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("RMSE reduction vs global stepwise (%)")
    ax.set_title("Relative progressive-RMSE reduction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    # Annotate high-budget labels
    for i, r in enumerate(rows):
        ax.text(i + w / 2, rh[i] + (1 if rh[i] >= 0 else -2), f"$b$={r[3]}", ha="center", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig5_relative_reduction")


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )

    fig_method_schematic()
    fig_ct_scaling_from_paper()

    results = {}
    # CT slices (paper primary setting at n=8k)
    X, y = load_ct_slices(max_rows=8000, seed=0)
    results["ct_8k"] = run_curves("ct_8k", X, y, n_regions=4, max_features=25)

    # Superconductivity
    X, y = load_superconductivity()
    results["superconductivity"] = run_curves(
        "superconductivity", X, y, n_regions=4, max_features=25
    )

    # Hard synthetic (mixed control; p=1000 for runtime)
    X, y = make_hard_synthetic(n_per_region=1200, p=1000, seed=0)
    results["synthetic_p1000"] = run_curves(
        "synthetic_p1000", X, y, n_regions=3, max_features=25
    )

    # Tecator from local data/
    X, y = load_tecator()
    results["tecator"] = run_curves("tecator", X, y, n_regions=3, max_features=20)

    CACHE.write_text(json.dumps(results, indent=2))
    print(f"cached curves -> {CACHE}", flush=True)

    # Progressive curve panels
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    plot_progressive(
        axes[0],
        results["ct_8k"]["curves"],
        "CT slices ($n$=8{,}000, $p$=384)",
    )
    plot_progressive(
        axes[1],
        results["superconductivity"]["curves"],
        "Superconductivity ($n\\approx$21k, $p$=81)",
    )
    fig.tight_layout()
    savefig(fig, "fig2_progressive_main")

    # Mixed / supporting
    n_panels = 1 + int("tecator" in results)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 3.8), squeeze=False)
    plot_progressive(
        axes[0, 0],
        results["synthetic_p1000"]["curves"],
        "Hard synthetic ($p$=1000) — mixed control",
    )
    if "tecator" in results:
        plot_progressive(
            axes[0, 1],
            results["tecator"]["curves"],
            "Tecator ($n$=240, $p$=124)",
        )
    fig.tight_layout()
    savefig(fig, "fig4_mixed_and_tecator")

    fig_summary_bars(results)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
