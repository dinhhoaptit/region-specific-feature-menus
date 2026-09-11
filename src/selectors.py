"""Pluggable feature selectors that return ordered feature menus.

Supported methods
-----------------
- ``vif``: streamwise VIF + alpha-investing (Lin, Foster & Ungar)
- ``lasso_lars``: LARS/Lasso regularization path; order = first-entry order,
  support = nonzero coeffs at CV-chosen lambda
- ``forward_stepwise``: greedy forward selection with BIC stopping
- ``saola``: online pairwise redundancy pruning (SAOLA-style; Yu et al.)
- ``mrmr``: minimum-redundancy maximum-relevance ranking (Peng et al.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoLarsCV, lars_path

from .vif import vif_regression

SelectorName = Literal["vif", "lasso_lars", "forward_stepwise", "saola", "mrmr"]


@dataclass
class SelectionResult:
    """Ordered feature selection outcome used as a progressive-revelation menu."""

    selected: list[int]
    intercept: float
    coef: np.ndarray
    method: str
    extras: dict = field(default_factory=dict)

    def menu(self, feature_names: Sequence[str] | None = None) -> list[str] | list[int]:
        if feature_names is None:
            return list(self.selected)
        return [feature_names[i] for i in self.selected]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.selected:
            return np.full(X.shape[0], self.intercept)
        return self.intercept + X[:, self.selected] @ self.coef


def _final_ols(X: np.ndarray, y: np.ndarray, selected: list[int]) -> tuple[float, np.ndarray]:
    n = X.shape[0]
    if not selected:
        return float(y.mean()), np.empty(0)
    Xb = np.column_stack([np.ones(n), X[:, selected]])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return float(beta[0]), beta[1:].astype(float)


def select_vif(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_order: Sequence[int] | None = None,
    max_features: int | None = None,
    random_state: int | None = None,
    w0: float = 0.50,
    delta_w: float = 0.05,
    subsample_size: int | None = None,
) -> SelectionResult:
    result = vif_regression(
        X,
        y,
        feature_order=feature_order,
        max_features=max_features,
        random_state=random_state,
        w0=w0,
        delta_w=delta_w,
        subsample_size=subsample_size,
    )
    return SelectionResult(
        selected=list(result.selected),
        intercept=result.intercept,
        coef=result.coef,
        method="vif",
        extras={"n_tested": result.n_tested, "pvalues": result.pvalues},
    )


def select_lasso_lars(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int | None = None,
    cv: int = 5,
    random_state: int | None = None,
) -> SelectionResult:
    """Lasso-LARS path: menu order is first-entry order along the path.

    The selected support is the active set at the CV-chosen alpha. Features are
    ordered by the step at which they first become nonzero on the LARS path,
    which yields a natural progressive-revelation menu.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n < 3:
        intercept, coef = _final_ols(X, y, [])
        return SelectionResult(selected=[], intercept=intercept, coef=coef, method="lasso_lars")

    # Entry order: first alpha where each feature becomes nonzero.
    try:
        alphas, _active, coefs = lars_path(X, y, method="lasso", verbose=0)
    except Exception:
        order = list(np.argsort(-np.abs(np.corrcoef(X, y, rowvar=False)[:-1, -1])))
        selected = order[: max_features or min(5, p)]
        intercept, coef = _final_ols(X, y, selected)
        return SelectionResult(
            selected=selected,
            intercept=intercept,
            coef=coef,
            method="lasso_lars",
            extras={"fallback": True},
        )

    entry_order: list[int] = []
    for col in range(coefs.shape[1]):
        for j in np.where(np.abs(coefs[:, col]) > 1e-12)[0]:
            jj = int(j)
            if jj not in entry_order:
                entry_order.append(jj)

    # CV chooses a sparse support.
    n_cv = int(min(cv, max(n // 5, 2)))
    n_cv = max(n_cv, 2)
    model = LassoLarsCV(cv=n_cv, max_n_alphas=500)
    # LassoLarsCV has no random_state in older sklearn; ignore if unsupported.
    try:
        model.fit(X, y)
    except TypeError:
        model.fit(X, y)

    support = [int(j) for j in np.where(np.abs(model.coef_) > 1e-12)[0]]
    # Order support by LARS entry time; append any CV-only leftovers at end.
    selected = [j for j in entry_order if j in set(support)]
    for j in support:
        if j not in selected:
            selected.append(j)

    if max_features is not None:
        selected = selected[: int(max_features)]

    intercept, coef = _final_ols(X, y, selected)
    return SelectionResult(
        selected=selected,
        intercept=intercept,
        coef=coef,
        method="lasso_lars",
        extras={
            "entry_order": entry_order,
            "alpha_": float(getattr(model, "alpha_", np.nan)),
        },
    )


def select_forward_stepwise(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int | None = None,
    min_bic_improve: float = 1e-9,
    screen_size: int | None = None,
) -> SelectionResult:
    """Greedy forward selection with BIC stopping.

    At each step, add the unused feature that most reduces RSS. Stop when BIC
    no longer improves (or ``max_features`` is reached). Addition order is the
    progressive-revelation menu.

    Uses a QR / residualized-correlation update so each step is O(n * p_cand)
    instead of refitting OLS for every candidate.

    For large ``p``, optionally keep only the top ``screen_size`` features by
    |corr| before stepwise (default: min(p, max(100, 20 * max_features))).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n < 3:
        intercept, coef = _final_ols(X, y, [])
        return SelectionResult(
            selected=[], intercept=intercept, coef=coef, method="forward_stepwise"
        )

    # Optional marginal screening for large-p regimes.
    candidate_idx = np.arange(p)
    if screen_size is None and p > 120:
        k_cap = max_features if max_features is not None else min(40, n // 5)
        screen_size = int(min(p, max(100, 20 * max(k_cap, 1))))
    if screen_size is not None and screen_size < p:
        y0 = y - y.mean()
        y_norm = np.linalg.norm(y0) + 1e-12
        Xc = X - X.mean(axis=0, keepdims=True)
        x_norm = np.linalg.norm(Xc, axis=0)
        x_norm = np.where(x_norm < 1e-12, 1.0, x_norm)
        corr = np.abs((Xc.T @ y0) / (x_norm * y_norm))
        candidate_idx = np.argsort(-corr)[: int(screen_size)]

    max_k = len(candidate_idx) if max_features is None else min(int(max_features), len(candidate_idx))
    cand = [int(j) for j in candidate_idx]
    remaining = set(cand)

    # QR / residualized-correlation path: O(n * p_cand) per step.
    q0 = np.ones(n) / np.sqrt(n)
    y_resid = y - q0 * np.dot(q0, y)
    rss = float(np.dot(y_resid, y_resid))
    rss = max(rss, 1e-12)
    best_bic = n * np.log(rss / n) + 1 * np.log(n)
    bic_path = [float(best_bic)]
    selected: list[int] = []

    X_orth = {j: X[:, j] - q0 * np.dot(q0, X[:, j]) for j in cand}

    while remaining and len(selected) < max_k:
        best_j = None
        best_gain = -1.0
        for j in remaining:
            x = X_orth[j]
            denom = float(np.dot(x, x))
            if denom < 1e-12:
                continue
            gain = float(np.dot(x, y_resid) ** 2 / denom)
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j is None or best_gain <= 0:
            break

        new_rss = max(rss - best_gain, 1e-12)
        k_params = len(selected) + 2
        bic = n * np.log(new_rss / n) + k_params * np.log(n)
        if bic > best_bic - min_bic_improve:
            break

        x = X_orth[best_j]
        q_new = x / np.sqrt(float(np.dot(x, x)))
        y_resid = y_resid - q_new * np.dot(q_new, y_resid)
        rss = new_rss
        best_bic = float(bic)
        bic_path.append(best_bic)
        selected.append(int(best_j))
        remaining.remove(best_j)

        for j in list(remaining):
            x = X_orth[j]
            X_orth[j] = x - q_new * np.dot(q_new, x)

    intercept, coef = _final_ols(X, y, selected)
    return SelectionResult(
        selected=selected,
        intercept=intercept,
        coef=coef,
        method="forward_stepwise",
        extras={
            "bic_path": bic_path,
            "n_candidates": int(len(candidate_idx)),
            "screened": bool(screen_size is not None and screen_size < p),
        },
    )


def _abs_corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    corr = (Xc.T @ Xc) / np.outer(norms, norms)
    return np.abs(np.nan_to_num(corr, nan=0.0))


def _abs_corr_with_y(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y0 = y - y.mean()
    y_norm = float(np.linalg.norm(y0)) + 1e-12
    Xc = X - X.mean(axis=0, keepdims=True)
    x_norm = np.linalg.norm(Xc, axis=0)
    x_norm = np.where(x_norm < 1e-12, 1.0, x_norm)
    return np.abs((Xc.T @ y0) / (x_norm * y_norm))


def select_saola(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int | None = None,
    delta: float = 0.0,
    feature_order: Sequence[int] | None = None,
) -> SelectionResult:
    """SAOLA-style online streaming feature selection for regression menus.

    Features arrive sequentially (default: decreasing |corr| with ``y``).
    A candidate is kept only if its relevance exceeds ``delta`` and it is not
    pairwise-redundant with any already selected feature under the SAOLA
    comparison rules of Yu, Wu, Ding & Pei (TKDD 2016 / ICDM 2014), using
    absolute Pearson correlation as the continuous relevance/redundancy score.
    Acceptance order is the progressive-revelation menu; if fewer than
    ``max_features`` survive, remaining candidates are appended by relevance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n < 3 or p == 0:
        intercept, coef = _final_ols(X, y, [])
        return SelectionResult(selected=[], intercept=intercept, coef=coef, method="saola")

    rel = _abs_corr_with_y(X, y)
    if feature_order is None:
        arrival = list(np.argsort(-rel))
    else:
        arrival = [int(j) for j in feature_order]

    # Cache pairwise |corr| lazily for visited features.
    selected: list[int] = []
    pair_cache: dict[tuple[int, int], float] = {}

    def pair_abs(i: int, j: int) -> float:
        a, b = (i, j) if i <= j else (j, i)
        key = (a, b)
        if key not in pair_cache:
            xi = X[:, a] - X[:, a].mean()
            xj = X[:, b] - X[:, b].mean()
            denom = (np.linalg.norm(xi) * np.linalg.norm(xj)) + 1e-12
            pair_cache[key] = float(abs(np.dot(xi, xj) / denom))
        return pair_cache[key]

    for f in arrival:
        rf = float(rel[f])
        if rf <= float(delta):
            continue
        discard_f = False
        to_remove: list[int] = []
        for s in selected:
            rs = float(rel[s])
            rfs = pair_abs(int(f), int(s))
            # SAOLA pairwise rules with |corr| as the association score.
            if rf <= rs and rfs >= rf:
                discard_f = True
                break
            if rf > rs and rfs >= rs:
                to_remove.append(int(s))
        if discard_f:
            continue
        for s in to_remove:
            if s in selected:
                selected.remove(s)
        selected.append(int(f))
        if max_features is not None and len(selected) >= int(max_features):
            break

    if max_features is not None and len(selected) < int(max_features):
        for f in arrival:
            if f not in selected:
                selected.append(int(f))
            if len(selected) >= int(max_features):
                break

    intercept, coef = _final_ols(X, y, selected)
    return SelectionResult(
        selected=selected,
        intercept=intercept,
        coef=coef,
        method="saola",
        extras={"delta": float(delta), "n_arrival": int(len(arrival))},
    )


def select_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int | None = None,
    random_state: int | None = 0,
) -> SelectionResult:
    """Minimum-redundancy maximum-relevance (mRMR) ranking for regression.

    Relevance uses mutual information with the response; redundancy uses mean
    absolute Pearson correlation with already chosen features (Peng, Long &
    Ding, TPAMI 2005). The incremental ranking is the acquisition menu.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n < 3 or p == 0:
        intercept, coef = _final_ols(X, y, [])
        return SelectionResult(selected=[], intercept=intercept, coef=coef, method="mrmr")

    max_k = p if max_features is None else min(int(max_features), p)
    # Mutual information can be expensive at large n; subsample rows.
    if n > 4000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=4000, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    relevance = mutual_info_regression(Xs, ys, random_state=random_state)
    relevance = np.asarray(relevance, dtype=float)
    relevance = np.nan_to_num(relevance, nan=0.0)

    pair_abs = _abs_corr_matrix(X)
    selected: list[int] = []
    remaining = set(range(p))

    # First feature: pure relevance.
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < max_k:
        best_j = None
        best_score = -np.inf
        for j in remaining:
            red = float(np.mean(pair_abs[j, selected])) if selected else 0.0
            score = float(relevance[j] - red)
            if score > best_score:
                best_score = score
                best_j = int(j)
        if best_j is None:
            break
        selected.append(best_j)
        remaining.remove(best_j)

    intercept, coef = _final_ols(X, y, selected)
    return SelectionResult(
        selected=selected,
        intercept=intercept,
        coef=coef,
        method="mrmr",
        extras={"relevance_top": float(np.max(relevance)) if p else 0.0},
    )


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: SelectorName = "forward_stepwise",
    **kwargs,
) -> SelectionResult:
    """Dispatch to a named feature selector."""
    if method == "vif":
        return select_vif(X, y, **kwargs)
    if method == "lasso_lars":
        # Drop VIF-only kwargs if present.
        allowed = {"max_features", "cv", "random_state"}
        return select_lasso_lars(X, y, **{k: v for k, v in kwargs.items() if k in allowed})
    if method == "forward_stepwise":
        allowed = {"max_features", "min_bic_improve", "screen_size"}
        return select_forward_stepwise(
            X, y, **{k: v for k, v in kwargs.items() if k in allowed}
        )
    if method == "saola":
        allowed = {"max_features", "delta", "feature_order"}
        return select_saola(X, y, **{k: v for k, v in kwargs.items() if k in allowed})
    if method == "mrmr":
        allowed = {"max_features", "random_state"}
        return select_mrmr(X, y, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unknown selector method: {method}")
