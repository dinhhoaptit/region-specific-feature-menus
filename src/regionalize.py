"""Train-only rules for choosing a global menu versus regional menus."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .region_menus import RegionFeatureMenus, mixture_gaussian_bic

PolicyName = Literal["region", "global"]


def mean_pairwise_prefix_jaccard(model: RegionFeatureMenus, *, k: int = 5) -> float:
    """Mean Jaccard overlap of the first ``k`` menu features across regions."""
    prefixes = [set(m.selected[:k]) for m in model.menus]
    if len(prefixes) < 2:
        return 1.0
    vals = []
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            a, b = prefixes[i], prefixes[j]
            vals.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(vals))


def mean_region_vs_global_jaccard(
    region: RegionFeatureMenus,
    global_menu: list[int],
    *,
    k: int = 5,
) -> float:
    g = set(global_menu[:k])
    vals = []
    for menu in region.menus:
        a = set(menu.selected[:k])
        vals.append(len(a & g) / max(len(a | g), 1))
    return float(np.mean(vals)) if vals else 1.0


def rmse_at_budget(model: RegionFeatureMenus, X: np.ndarray, y: np.ndarray, budget: int) -> float:
    y = np.asarray(y, dtype=float).ravel()
    pred = model.predict(X, budget=int(budget))
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def pick_by_inner_val(
    region: RegionFeatureMenus,
    glob: RegionFeatureMenus,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    budget: int = 5,
) -> tuple[PolicyName, float, float]:
    r = rmse_at_budget(region, X_val, y_val, budget)
    g = rmse_at_budget(glob, X_val, y_val, budget)
    pick: PolicyName = "region" if r <= g else "global"
    return pick, r, g


def pick_by_train_bic(
    region: RegionFeatureMenus,
    glob: RegionFeatureMenus,
    X: np.ndarray,
    y: np.ndarray,
    *,
    budget: int = 5,
) -> tuple[PolicyName, float, float]:
    rb = mixture_gaussian_bic(region, X, y, budget=budget)
    gb = mixture_gaussian_bic(glob, X, y, budget=budget)
    pick: PolicyName = "region" if rb <= gb else "global"
    return pick, rb, gb


def pick_by_menu_jaccard(
    region: RegionFeatureMenus,
    *,
    k: int = 5,
    max_overlap: float = 0.35,
) -> tuple[PolicyName, float]:
    jac = mean_pairwise_prefix_jaccard(region, k=k)
    pick: PolicyName = "region" if jac < max_overlap else "global"
    return pick, jac
