"""Region-specific incremental feature menus via clustering + selectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .selectors import SelectionResult, SelectorName, select_features

PartitionMode = Literal["kmeans", "xy", "residual"]
AssignmentMode = Literal["hard", "soft"]


@dataclass
class PrefixModel:
    """OLS model on a prefix of a region's feature menu."""

    features: list[int]
    intercept: float
    coef: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.features:
            return np.full(X.shape[0], self.intercept)
        return self.intercept + X[:, self.features] @ self.coef


@dataclass
class RegionMenu:
    """Feature acquisition menu for one region."""

    region_id: int
    n_samples: int
    selected: list[int]
    """Feature indices in selector addition order (= progressive revelation order)."""

    selection: SelectionResult
    center: np.ndarray
    """Cluster center in the X-space used for assignment."""

    prefix_models: list[PrefixModel] = field(default_factory=list)
    """prefix_models[b] uses the first b menu features (b=0 is intercept-only)."""

    def menu(self, feature_names: Sequence[str] | None = None) -> list[str] | list[int]:
        return self.selection.menu(feature_names)

    def model_for_budget(self, budget: int) -> PrefixModel:
        if not self.prefix_models:
            raise RuntimeError("prefix_models not fitted for this region")
        b = int(max(0, min(budget, len(self.prefix_models) - 1)))
        return self.prefix_models[b]

    @property
    def vif(self) -> SelectionResult:
        """Backward-compatible alias for ``selection``."""
        return self.selection


@dataclass
class RegionFeatureMenus:
    """Collection of per-region feature menus."""

    n_regions: int
    labels: np.ndarray
    """Hard training assignment of each row to a region."""

    menus: list[RegionMenu]
    feature_names: list[str] | None = None
    cluster_centers: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    clusterer_name: str = "kmeans"
    partition: PartitionMode = "kmeans"
    assignment: AssignmentMode = "hard"
    soft_temperature: float = 1.0
    x_scaler: StandardScaler | None = None
    selector: SelectorName = "forward_stepwise"

    def menu_for_region(
        self, region_id: int, *, named: bool = False
    ) -> list[str] | list[int]:
        menu = self.menus[region_id]
        if named and self.feature_names is not None:
            return menu.menu(self.feature_names)
        return list(menu.selected)

    def _scaled_X(self, X: np.ndarray) -> np.ndarray:
        if self.x_scaler is None:
            return X
        return self.x_scaler.transform(X)

    def region_weights(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_regions) assignment weights."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = self._scaled_X(X)
        centers = self.cluster_centers
        d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        if self.assignment == "hard" or self.n_regions == 1:
            w = np.zeros_like(d2)
            w[np.arange(len(X)), np.argmin(d2, axis=1)] = 1.0
            return w

        temp = max(float(self.soft_temperature), 1e-8)
        logits = -d2 / temp
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def assign(self, X: np.ndarray) -> np.ndarray:
        """Hard region ids (argmax of weights)."""
        return np.argmax(self.region_weights(X), axis=1)

    def menu_for_query(
        self,
        x: np.ndarray,
        *,
        named: bool = False,
        menu_mode: Literal["primary", "soft_rank"] = "primary",
    ) -> tuple[int, list[str] | list[int]]:
        """Return (primary_region_id, ordered feature menu) for a query."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        weights = self.region_weights(x)[0]
        primary = int(np.argmax(weights))
        if menu_mode == "primary" or self.assignment == "hard":
            return primary, self.menu_for_region(primary, named=named)

        p = x.shape[1]
        scores = np.full(p, 0.0)
        present = np.zeros(p, dtype=bool)
        for rid, menu in enumerate(self.menus):
            w = float(weights[rid])
            if w <= 0 or not menu.selected:
                continue
            for rank, j in enumerate(menu.selected):
                scores[j] += w * (1.0 / (rank + 1.0))
                present[j] = True
        order = [int(j) for j in np.argsort(-scores) if present[j]]
        if named and self.feature_names is not None:
            return primary, [self.feature_names[j] for j in order]
        return primary, order

    def predict(self, X: np.ndarray, *, budget: int | None = None) -> np.ndarray:
        """Predict with full menus or a feature budget (prefix models)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        weights = self.region_weights(X)
        yhat = np.zeros(X.shape[0], dtype=float)
        for rid, menu in enumerate(self.menus):
            w = weights[:, rid]
            if not np.any(w > 0):
                continue
            if budget is None:
                pred = menu.selection.predict(X)
            else:
                pred = menu.model_for_budget(budget).predict(X)
            yhat += w * pred
        return yhat

    def predict_progressive(self, X: np.ndarray, budgets: Sequence[int]) -> np.ndarray:
        """Return predictions of shape (n_budgets, n_samples)."""
        return np.vstack([self.predict(X, budget=b) for b in budgets])

    def summary(self) -> list[dict]:
        rows = []
        for menu in self.menus:
            names = (
                menu.menu(self.feature_names)
                if self.feature_names is not None
                else menu.selected
            )
            menu_vals = [int(v) if not isinstance(v, str) else v for v in names]
            rows.append(
                {
                    "region": menu.region_id,
                    "n_samples": menu.n_samples,
                    "n_selected": len(menu.selected),
                    "menu": menu_vals,
                    "selector": menu.selection.method,
                }
            )
        return rows


def _marginal_corr_order(X: np.ndarray, y: np.ndarray) -> list[int]:
    """Order features by descending |Pearson correlation| with y."""
    y = y - y.mean()
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-12:
        return list(range(X.shape[1]))
    Xc = X - X.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(Xc, axis=0)
    x_norm = np.where(x_norm < 1e-12, 1.0, x_norm)
    corr = (Xc.T @ y) / (x_norm * y_norm)
    return list(np.argsort(-np.abs(corr)))


def _fit_prefix_models(X: np.ndarray, y: np.ndarray, selected: list[int]) -> list[PrefixModel]:
    """Fit nested OLS models for budgets 0..len(selected)."""
    n = X.shape[0]
    models: list[PrefixModel] = []
    models.append(PrefixModel(features=[], intercept=float(y.mean()), coef=np.empty(0)))
    for b in range(1, len(selected) + 1):
        feats = selected[:b]
        Xb = np.column_stack([np.ones(n), X[:, feats]])
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        models.append(
            PrefixModel(
                features=list(feats),
                intercept=float(beta[0]),
                coef=beta[1:].astype(float),
            )
        )
    return models


def _partition_labels(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_regions: int,
    partition: PartitionMode,
    y_weight: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Return labels, X-space centers (scaled), and the X scaler."""
    n = X.shape[0]
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    if n_regions == 1:
        labels = np.zeros(n, dtype=int)
        centers = Xs.mean(axis=0, keepdims=True)
        return labels, centers, x_scaler

    n_regions = int(min(n_regions, n))
    if partition == "kmeans":
        Z = Xs
    elif partition == "xy":
        ys = StandardScaler().fit_transform(y.reshape(-1, 1)) * float(y_weight)
        Z = np.hstack([Xs, ys])
    elif partition == "residual":
        lr = LinearRegression()
        lr.fit(Xs, y)
        resid = (y - lr.predict(Xs)).reshape(-1, 1)
        rs = StandardScaler().fit_transform(resid) * float(y_weight)
        Z = np.hstack([Xs, rs])
    else:
        raise ValueError(f"Unknown partition mode: {partition}")

    km = KMeans(n_clusters=n_regions, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)

    centers = np.vstack(
        [
            Xs[labels == rid].mean(axis=0)
            if np.any(labels == rid)
            else Xs.mean(axis=0)
            for rid in range(n_regions)
        ]
    )
    return labels, centers, x_scaler


def _merge_small_regions(
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    X: np.ndarray,
    min_region_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassign tiny regions to the nearest sufficiently large region."""
    labels = labels.copy()
    n_regions = centers.shape[0]
    sizes = np.bincount(labels, minlength=n_regions)
    large = [rid for rid in range(n_regions) if sizes[rid] >= min_region_size]
    if not large:
        labels[:] = 0
        center = X.mean(axis=0, keepdims=True)
        return labels, center

    small = [rid for rid in range(n_regions) if sizes[rid] < min_region_size]
    if not small:
        return labels, centers

    large_centers = centers[large]
    for rid in small:
        mask = labels == rid
        if not np.any(mask):
            continue
        d2 = ((X[mask][:, None, :] - large_centers[None, :, :]) ** 2).sum(axis=2)
        nearest_local = np.argmin(d2, axis=1)
        labels[mask] = np.asarray(large, dtype=int)[nearest_local]

    remaining = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(remaining)}
    labels = np.asarray([remap[int(v)] for v in labels], dtype=int)
    new_centers = np.vstack(
        [X[labels == rid].mean(axis=0) for rid in range(len(remaining))]
    )
    return labels, new_centers


def _empty_selection(y_region: np.ndarray) -> SelectionResult:
    intercept = float(y_region.mean()) if len(y_region) else 0.0
    return SelectionResult(
        selected=[],
        intercept=intercept,
        coef=np.empty(0),
        method="empty",
    )


def fit_region_menus(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_regions: int = 3,
    feature_names: Sequence[str] | None = None,
    feature_order: Sequence[int] | None = None,
    order_by: str = "abs_corr",
    selector: SelectorName = "forward_stepwise",
    partition: PartitionMode = "kmeans",
    assignment: AssignmentMode = "hard",
    soft_temperature: float = 1.0,
    y_weight: float = 1.0,
    min_region_size: int = 20,
    random_state: int | None = 0,
    w0: float = 0.50,
    delta_w: float = 0.05,
    subsample_size: int | None = None,
    max_features: int | None = None,
) -> RegionFeatureMenus:
    """Cluster inputs and run feature selection inside each region.

    Parameters
    ----------
    selector:
        ``forward_stepwise`` (default), ``vif``, or ``lasso_lars``.
    partition:
        ``kmeans``, ``xy``, or ``residual``.
    assignment:
        ``hard`` or ``soft``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    if order_by not in {"abs_corr", "index"}:
        raise ValueError("order_by must be 'abs_corr' or 'index'")
    if assignment not in {"hard", "soft"}:
        raise ValueError("assignment must be 'hard' or 'soft'")
    if selector not in {"vif", "lasso_lars", "forward_stepwise"}:
        raise ValueError("selector must be vif, lasso_lars, or forward_stepwise")

    n, p = X.shape
    names = list(feature_names) if feature_names is not None else None
    if names is not None and len(names) != p:
        raise ValueError("feature_names length must equal number of columns in X")

    labels, centers, x_scaler = _partition_labels(
        X,
        y,
        n_regions=n_regions,
        partition=partition,
        y_weight=y_weight,
        random_state=random_state,
    )
    labels, centers = _merge_small_regions(
        labels, centers, X=x_scaler.transform(X), min_region_size=min_region_size
    )
    n_regions = int(centers.shape[0])
    menus = _menus_from_labels(
        X,
        y,
        labels=labels,
        centers=centers,
        selector=selector,
        feature_order=feature_order,
        order_by=order_by,
        min_region_size=min_region_size,
        max_features=max_features,
        w0=w0,
        delta_w=delta_w,
        subsample_size=subsample_size,
        random_state=random_state,
    )

    return RegionFeatureMenus(
        n_regions=n_regions,
        labels=labels,
        menus=menus,
        feature_names=names,
        cluster_centers=centers,
        clusterer_name=f"{partition}_{assignment}_{selector}",
        partition=partition,
        assignment=assignment,
        soft_temperature=soft_temperature,
        x_scaler=x_scaler,
        selector=selector,
    )


def _menus_from_labels(
    X: np.ndarray,
    y: np.ndarray,
    *,
    labels: np.ndarray,
    centers: np.ndarray,
    selector: SelectorName,
    feature_order: Sequence[int] | None,
    order_by: str,
    min_region_size: int,
    max_features: int | None,
    w0: float,
    delta_w: float,
    subsample_size: int | None,
    random_state: int | None,
) -> list[RegionMenu]:
    n_regions = int(centers.shape[0])
    p = X.shape[1]
    menus: list[RegionMenu] = []
    for rid in range(n_regions):
        mask = labels == rid
        n_r = int(mask.sum())
        if n_r < max(3, min(min_region_size, 5)):
            y_r = y[mask] if n_r else np.array([0.0])
            empty = _empty_selection(y_r)
            menus.append(
                RegionMenu(
                    region_id=rid,
                    n_samples=n_r,
                    selected=[],
                    selection=empty,
                    center=centers[rid],
                    prefix_models=_fit_prefix_models(
                        X[mask] if n_r else np.zeros((1, p)),
                        y_r,
                        [],
                    ),
                )
            )
            continue

        if feature_order is not None:
            order = list(feature_order)
        elif order_by == "abs_corr":
            order = _marginal_corr_order(X[mask], y[mask])
        else:
            order = list(range(p))

        sel_kwargs: dict = {"max_features": max_features}
        if selector == "vif":
            sel_kwargs.update(
                {
                    "feature_order": order,
                    "w0": w0,
                    "delta_w": delta_w,
                    "subsample_size": subsample_size,
                    "random_state": None
                    if random_state is None
                    else random_state + rid + 1,
                }
            )
        elif selector == "lasso_lars":
            sel_kwargs["random_state"] = (
                None if random_state is None else random_state + rid + 1
            )

        selection = select_features(X[mask], y[mask], method=selector, **sel_kwargs)
        prefixes = _fit_prefix_models(X[mask], y[mask], list(selection.selected))
        menus.append(
            RegionMenu(
                region_id=rid,
                n_samples=n_r,
                selected=list(selection.selected),
                selection=selection,
                center=centers[rid],
                prefix_models=prefixes,
            )
        )
    return menus


def _menu_jaccard(a: Sequence[int], b: Sequence[int], k: int = 8) -> float:
    sa, sb = set(list(a)[:k]), set(list(b)[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


@dataclass
class AlternatingFitResult:
    """Two-stage fit (iterate 0) plus residual-reassignment refinements."""

    models: list[RegionFeatureMenus]
    history: list[dict]


def fit_region_menus_alternating(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_refine: int = 3,
    reassign_budget: int = 5,
    **kwargs,
) -> AlternatingFitResult:
    """Alternate menu mining with y-aware reassignment, then refresh X-centers.

    Iterate 0 is the current two-stage pipeline (``fit_region_menus``). Each
    later iterate reassigns every training point to the region whose nested
    prefix model at ``reassign_budget`` has the smallest squared error, updates
    region centers as means in scaled X, and rebuilds stepwise menus.
    Test-time assignment still uses only X (the centers).
    """
    base_kwargs = dict(kwargs)
    assignment = base_kwargs.get("assignment", "soft")
    min_region_size = int(base_kwargs.get("min_region_size", 20))
    selector = base_kwargs.get("selector", "forward_stepwise")
    feature_order = base_kwargs.get("feature_order")
    order_by = base_kwargs.get("order_by", "abs_corr")
    max_features = base_kwargs.get("max_features")
    w0 = base_kwargs.get("w0", 0.50)
    delta_w = base_kwargs.get("delta_w", 0.05)
    subsample_size = base_kwargs.get("subsample_size")
    random_state = base_kwargs.get("random_state", 0)
    partition = base_kwargs.get("partition", "residual")
    soft_temperature = base_kwargs.get("soft_temperature", 1.0)
    feature_names = base_kwargs.get("feature_names")

    model0 = fit_region_menus(X, y, **base_kwargs)
    models = [model0]
    history = [
        {
            "iterate": 0,
            "n_regions": int(model0.n_regions),
            "sizes": [int(m.n_samples) for m in model0.menus],
            "frac_relabeled": 0.0,
            "mean_top8_jaccard_vs_prev": 1.0,
        }
    ]

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    labels = model0.labels.copy()
    x_scaler = model0.x_scaler
    Xs = x_scaler.transform(X) if x_scaler is not None else X
    names = list(feature_names) if feature_names is not None else None

    for t in range(1, int(n_refine) + 1):
        prev = models[-1]
        n_r = prev.n_regions
        err = np.empty((X.shape[0], n_r), dtype=float)
        for rid, menu in enumerate(prev.menus):
            b = int(min(max(reassign_budget, 0), max(len(menu.prefix_models) - 1, 0)))
            pred = menu.model_for_budget(b).predict(X)
            err[:, rid] = (y - pred) ** 2
        new_labels = np.argmin(err, axis=1).astype(int)
        frac = float(np.mean(new_labels != labels))
        labels = new_labels
        centers = np.vstack(
            [
                Xs[labels == rid].mean(axis=0)
                if np.any(labels == rid)
                else Xs.mean(axis=0)
                for rid in range(n_r)
            ]
        )
        labels, centers = _merge_small_regions(
            labels, centers, X=Xs, min_region_size=min_region_size
        )
        n_regions = int(centers.shape[0])
        menus = _menus_from_labels(
            X,
            y,
            labels=labels,
            centers=centers,
            selector=selector,
            feature_order=feature_order,
            order_by=order_by,
            min_region_size=min_region_size,
            max_features=max_features,
            w0=w0,
            delta_w=delta_w,
            subsample_size=subsample_size,
            random_state=random_state,
        )
        jac = []
        for rid, menu in enumerate(menus):
            if rid < len(prev.menus):
                jac.append(_menu_jaccard(prev.menus[rid].selected, menu.selected))
        model = RegionFeatureMenus(
            n_regions=n_regions,
            labels=labels,
            menus=menus,
            feature_names=names,
            cluster_centers=centers,
            clusterer_name=f"{partition}_{assignment}_{selector}_alt{t}",
            partition=partition,
            assignment=assignment,
            soft_temperature=soft_temperature,
            x_scaler=x_scaler,
            selector=selector,
        )
        models.append(model)
        history.append(
            {
                "iterate": t,
                "n_regions": n_regions,
                "sizes": [int(m.n_samples) for m in menus],
                "frac_relabeled": frac,
                "mean_top8_jaccard_vs_prev": float(np.mean(jac)) if jac else 1.0,
            }
        )

    return AlternatingFitResult(models=models, history=history)


def mixture_gaussian_bic(
    model: RegionFeatureMenus,
    X: np.ndarray,
    y: np.ndarray,
    *,
    budget: int,
) -> float:
    """Gaussian BIC of the soft mixture at a fixed prefix budget.

    Parameter count is intercept plus prefix length in each region (shared
    features across regions are counted separately, matching separate OLS menus).
    """
    y = np.asarray(y, dtype=float).ravel()
    yhat = model.predict(X, budget=int(budget))
    n = float(max(len(y), 1))
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-18)
    k = 0
    for menu in model.menus:
        b = int(min(max(budget, 0), max(len(menu.prefix_models) - 1, 0)))
        k += 1 + b
    return float(n * np.log(rss / n) + k * np.log(n))


def _fit_kwargs_view(kwargs: dict) -> dict:
    return {
        "assignment": kwargs.get("assignment", "soft"),
        "min_region_size": int(kwargs.get("min_region_size", 20)),
        "selector": kwargs.get("selector", "forward_stepwise"),
        "feature_order": kwargs.get("feature_order"),
        "order_by": kwargs.get("order_by", "abs_corr"),
        "max_features": kwargs.get("max_features"),
        "w0": kwargs.get("w0", 0.50),
        "delta_w": kwargs.get("delta_w", 0.05),
        "subsample_size": kwargs.get("subsample_size"),
        "random_state": kwargs.get("random_state", 0),
        "partition": kwargs.get("partition", "residual"),
        "soft_temperature": kwargs.get("soft_temperature", 1.0),
        "feature_names": kwargs.get("feature_names"),
        "y_weight": float(kwargs.get("y_weight", 1.0)),
    }


def fit_region_menus_mixture_residual(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_refine: int = 3,
    reassign_budget: int = 5,
    **kwargs,
) -> AlternatingFitResult:
    """Recluster on mixture residuals of current prefix models, then rebuild menus.

    Unlike error-based hard reassignment, points are not moved by comparing
    regional y-fits directly. The extra clustering coordinate is the residual of
    the current *soft mixture* prediction at ``reassign_budget``. Region centers
    remain means in scaled X, so test assignment never uses y.
    """
    view = _fit_kwargs_view(kwargs)
    model0 = fit_region_menus(X, y, **kwargs)
    models = [model0]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    history = [
        {
            "iterate": 0,
            "n_regions": int(model0.n_regions),
            "sizes": [int(m.n_samples) for m in model0.menus],
            "frac_relabeled": 0.0,
            "mean_top8_jaccard_vs_prev": 1.0,
            "train_bic": mixture_gaussian_bic(
                model0, X, y, budget=int(reassign_budget)
            ),
        }
    ]
    x_scaler = model0.x_scaler
    Xs = x_scaler.transform(X) if x_scaler is not None else X
    names = list(view["feature_names"]) if view["feature_names"] is not None else None
    labels = model0.labels.copy()
    n_target = int(model0.n_regions)
    rs_seed = 0 if view["random_state"] is None else int(view["random_state"])

    for t in range(1, int(n_refine) + 1):
        prev = models[-1]
        yhat = prev.predict(X, budget=int(reassign_budget))
        resid = (y - yhat).reshape(-1, 1)
        rs = StandardScaler().fit_transform(resid) * view["y_weight"]
        z = np.hstack([Xs, rs])
        n_cl = int(min(n_target, X.shape[0]))
        km = KMeans(n_clusters=n_cl, n_init=10, random_state=rs_seed + 17 * t)
        new_labels = km.fit_predict(z).astype(int)
        frac = float(np.mean(new_labels != labels))
        labels = new_labels
        centers = np.vstack(
            [
                Xs[labels == rid].mean(axis=0)
                if np.any(labels == rid)
                else Xs.mean(axis=0)
                for rid in range(n_cl)
            ]
        )
        labels, centers = _merge_small_regions(
            labels, centers, X=Xs, min_region_size=view["min_region_size"]
        )
        n_regions = int(centers.shape[0])
        menus = _menus_from_labels(
            X,
            y,
            labels=labels,
            centers=centers,
            selector=view["selector"],
            feature_order=view["feature_order"],
            order_by=view["order_by"],
            min_region_size=view["min_region_size"],
            max_features=view["max_features"],
            w0=view["w0"],
            delta_w=view["delta_w"],
            subsample_size=view["subsample_size"],
            random_state=view["random_state"],
        )
        jac = [
            _menu_jaccard(prev.menus[rid].selected, menu.selected)
            for rid, menu in enumerate(menus)
            if rid < len(prev.menus)
        ]
        model = RegionFeatureMenus(
            n_regions=n_regions,
            labels=labels,
            menus=menus,
            feature_names=names,
            cluster_centers=centers,
            clusterer_name=f"{view['partition']}_{view['assignment']}_{view['selector']}_mixres{t}",
            partition=view["partition"],
            assignment=view["assignment"],
            soft_temperature=view["soft_temperature"],
            x_scaler=x_scaler,
            selector=view["selector"],
        )
        models.append(model)
        history.append(
            {
                "iterate": t,
                "n_regions": n_regions,
                "sizes": [int(m.n_samples) for m in menus],
                "frac_relabeled": frac,
                "mean_top8_jaccard_vs_prev": float(np.mean(jac)) if jac else 1.0,
                "train_bic": mixture_gaussian_bic(
                    model, X, y, budget=int(reassign_budget)
                ),
            }
        )

    return AlternatingFitResult(models=models, history=history)
