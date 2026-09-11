"""Progressive feature revelation evaluation: RMSE vs feature budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .region_menus import RegionFeatureMenus, _fit_prefix_models, _marginal_corr_order
from .vif import vif_regression


@dataclass
class ProgressiveCurve:
    """RMSE (and friends) as a function of acquired-feature budget."""

    name: str
    budgets: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    mean_features_used: np.ndarray

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "budgets": self.budgets.tolist(),
            "rmse": self.rmse.tolist(),
            "mae": self.mae.tolist(),
            "mean_features_used": self.mean_features_used.tolist(),
        }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def _predict_with_prefix_list(
    X: np.ndarray,
    region_ids: np.ndarray,
    prefix_models_by_region: list[list],
    budget: int,
) -> np.ndarray:
    yhat = np.empty(X.shape[0], dtype=float)
    for rid, prefixes in enumerate(prefix_models_by_region):
        mask = region_ids == rid
        if not np.any(mask):
            continue
        b = int(max(0, min(budget, len(prefixes) - 1)))
        yhat[mask] = prefixes[b].predict(X[mask])
    return yhat


def evaluate_region_menus_progressive(
    model: RegionFeatureMenus,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    budgets: Sequence[int] | None = None,
    name: str | None = None,
) -> ProgressiveCurve:
    """Evaluate a fitted region-menu model under progressive revelation."""
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()
    max_menu = max((len(m.selected) for m in model.menus), default=0)
    if budgets is None:
        budgets = list(range(0, max_menu + 1))
    budgets_arr = np.asarray(list(budgets), dtype=int)

    rmse_list, mae_list, used_list = [], [], []
    for b in budgets_arr:
        yhat = model.predict(X_test, budget=int(b))
        rmse, mae = _metrics(y_test, yhat)
        # Features used: for soft assignment, expected menu prefix length.
        weights = model.region_weights(X_test)
        used = 0.0
        for rid, menu in enumerate(model.menus):
            used += float(weights[:, rid].mean() * min(int(b), len(menu.selected)))
        rmse_list.append(rmse)
        mae_list.append(mae)
        used_list.append(used)

    return ProgressiveCurve(
        name=name or f"region:{model.partition}/{model.assignment}",
        budgets=budgets_arr,
        rmse=np.asarray(rmse_list),
        mae=np.asarray(mae_list),
        mean_features_used=np.asarray(used_list),
    )


def evaluate_global_menu_progressive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    menu: Sequence[int] | None = None,
    budgets: Sequence[int] | None = None,
    name: str = "global",
    random_state: int | None = 0,
) -> ProgressiveCurve:
    """Progressive revelation with a single global ordered feature menu."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()

    if menu is None:
        order = _marginal_corr_order(X_train, y_train)
        vif = vif_regression(
            X_train, y_train, feature_order=order, random_state=random_state
        )
        menu_list = list(vif.selected)
    else:
        menu_list = list(menu)

    prefixes = _fit_prefix_models(X_train, y_train, menu_list)
    if budgets is None:
        budgets = list(range(0, len(menu_list) + 1))
    budgets_arr = np.asarray(list(budgets), dtype=int)

    rmse_list, mae_list, used_list = [], [], []
    for b in budgets_arr:
        bb = int(max(0, min(int(b), len(prefixes) - 1)))
        yhat = prefixes[bb].predict(X_test)
        rmse, mae = _metrics(y_test, yhat)
        rmse_list.append(rmse)
        mae_list.append(mae)
        used_list.append(float(min(int(b), len(menu_list))))

    return ProgressiveCurve(
        name=name,
        budgets=budgets_arr,
        rmse=np.asarray(rmse_list),
        mae=np.asarray(mae_list),
        mean_features_used=np.asarray(used_list),
    )


def evaluate_random_menu_progressive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_features: int | None = None,
    budgets: Sequence[int] | None = None,
    random_state: int | None = 0,
    name: str = "random",
) -> ProgressiveCurve:
    """Progressive revelation with a random feature order (baseline)."""
    p = np.asarray(X_train).shape[1]
    rng = np.random.default_rng(random_state)
    n_take = p if n_features is None else min(int(n_features), p)
    menu = list(rng.permutation(p)[:n_take])
    return evaluate_global_menu_progressive(
        X_train,
        y_train,
        X_test,
        y_test,
        menu=menu,
        budgets=budgets,
        name=name,
    )


def evaluate_abs_corr_menu_progressive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_features: int | None = None,
    budgets: Sequence[int] | None = None,
    name: str = "abs_corr",
) -> ProgressiveCurve:
    """Acquire features by global |corr| order (no VIF selection)."""
    order = _marginal_corr_order(np.asarray(X_train), np.asarray(y_train).ravel())
    if n_features is not None:
        order = order[: int(n_features)]
    return evaluate_global_menu_progressive(
        X_train,
        y_train,
        X_test,
        y_test,
        menu=order,
        budgets=budgets,
        name=name,
    )


PolicyFactory = Callable[[np.ndarray, np.ndarray], RegionFeatureMenus]


def compare_progressive_policies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    region_policies: dict[str, dict] | None = None,
    budgets: Sequence[int] | None = None,
    include_baselines: bool = True,
    random_state: int | None = 0,
) -> list[ProgressiveCurve]:
    """Fit several region-menu variants and baselines; return RMSE curves."""
    from .region_menus import fit_region_menus

    if region_policies is None:
        region_policies = {
            "kmeans/hard": {"partition": "kmeans", "assignment": "hard"},
            "kmeans/soft": {
                "partition": "kmeans",
                "assignment": "soft",
                "soft_temperature": 1.0,
            },
            "residual/soft": {
                "partition": "residual",
                "assignment": "soft",
                "soft_temperature": 1.0,
                "y_weight": 1.0,
            },
            "xy/soft": {
                "partition": "xy",
                "assignment": "soft",
                "soft_temperature": 1.0,
                "y_weight": 1.0,
            },
        }

    curves: list[ProgressiveCurve] = []
    fitted: list[RegionFeatureMenus] = []
    for name, kwargs in region_policies.items():
        model = fit_region_menus(
            X_train,
            y_train,
            random_state=random_state,
            **kwargs,
        )
        fitted.append(model)
        curves.append(
            evaluate_region_menus_progressive(
                model, X_test, y_test, budgets=budgets, name=name
            )
        )

    if include_baselines:
        max_sel = max((len(m.selected) for model in fitted for m in model.menus), default=5)
        base_budgets = budgets
        if base_budgets is None:
            base_budgets = list(range(0, max_sel + 1))
        curves.append(
            evaluate_global_menu_progressive(
                X_train,
                y_train,
                X_test,
                y_test,
                budgets=base_budgets,
                name="global_vif",
                random_state=random_state,
            )
        )
        curves.append(
            evaluate_abs_corr_menu_progressive(
                X_train,
                y_train,
                X_test,
                y_test,
                n_features=max_sel,
                budgets=base_budgets,
                name="abs_corr",
            )
        )
        curves.append(
            evaluate_random_menu_progressive(
                X_train,
                y_train,
                X_test,
                y_test,
                n_features=max_sel,
                budgets=base_budgets,
                random_state=random_state,
                name="random",
            )
        )
        from .selectors import select_mrmr, select_saola

        saola_menu = select_saola(X_train, y_train, max_features=max_sel).selected
        mrmr_menu = select_mrmr(
            X_train, y_train, max_features=max_sel, random_state=random_state
        ).selected
        curves.append(
            evaluate_global_menu_progressive(
                X_train,
                y_train,
                X_test,
                y_test,
                menu=saola_menu,
                budgets=base_budgets,
                name="global_saola",
            )
        )
        curves.append(
            evaluate_global_menu_progressive(
                X_train,
                y_train,
                X_test,
                y_test,
                menu=mrmr_menu,
                budgets=base_budgets,
                name="global_mrmr",
            )
        )
    return curves


def format_curves_table(curves: Sequence[ProgressiveCurve]) -> str:
    """Pretty text table of RMSE by budget for multiple policies."""
    if not curves:
        return "(no curves)"
    # Union of budgets for display; missing values shown as blank.
    all_budgets = sorted({int(b) for c in curves for b in c.budgets})
    header = ["budget"] + [c.name for c in curves]
    rows = [" | ".join(header)]
    rows.append("-+-".join("-" * len(h) for h in header))
    for b in all_budgets:
        cells = [f"{b:>6d}"]
        for curve in curves:
            match = np.where(curve.budgets == b)[0]
            if len(match) == 0:
                cells.append(f"{'':>10s}")
            else:
                cells.append(f"{curve.rmse[match[0]]:10.4f}")
        rows.append(" | ".join(cells))
    return "\n".join(rows)
