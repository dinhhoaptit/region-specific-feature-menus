"""Instance-adaptive greedy acquisition under progressive feature revelation.

Global forward stepwise (already a paper baseline) builds one menu for every
query: the next feature is a function of the selected *set* on the full training
sample, so every test point follows the same path.

This module implements *local residual-greedy* (LRG) acquisition. After the
shared first feature, each query's next coordinate is chosen on a *k*-nearest
neighbourhood in the *already observed* coordinates only. Prediction at budget
``b`` uses nested OLS on the full training sample, restricted to that query's
acquired set (the same nested protocol as regional menus).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .progressive import ProgressiveCurve, _metrics
from .region_menus import PrefixModel


def _ols_on_set(X: np.ndarray, y: np.ndarray, feats: list[int]) -> PrefixModel:
    n = X.shape[0]
    if not feats:
        return PrefixModel(features=[], intercept=float(y.mean()), coef=np.empty(0))
    Xb = np.column_stack([np.ones(n), X[:, feats]])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return PrefixModel(
        features=list(feats),
        intercept=float(beta[0]),
        coef=beta[1:].astype(float),
    )


def _rss_gains(X: np.ndarray, y: np.ndarray, selected: list[int]) -> np.ndarray:
    """RSS reduction of adding each column after OLS on ``selected`` (plus intercept)."""
    n, p = X.shape
    if n < 3:
        return np.full(p, -np.inf)
    if selected:
        A = np.column_stack([np.ones(n), X[:, selected]])
    else:
        A = np.ones((n, 1))
    Q, _ = np.linalg.qr(A, mode="reduced")
    y_res = y - Q @ (Q.T @ y)
    X_res = X - Q @ (Q.T @ X)
    den = np.einsum("ij,ij->j", X_res, X_res)
    num = X_res.T @ y_res
    gain = np.full(p, -np.inf)
    ok = den > 1e-12
    gain[ok] = (num[ok] ** 2) / den[ok]
    for j in selected:
        gain[int(j)] = -np.inf
    return gain


def _first_feature(X: np.ndarray, y: np.ndarray) -> int:
    gain = _rss_gains(X, y, [])
    j = int(np.argmax(gain))
    if not np.isfinite(gain[j]) or gain[j] <= 0:
        y0 = y - y.mean()
        Xc = X - X.mean(axis=0, keepdims=True)
        den = np.linalg.norm(Xc, axis=0)
        den = np.where(den < 1e-12, 1.0, den)
        corr = np.abs(Xc.T @ y0 / (den * (np.linalg.norm(y0) + 1e-12)))
        j = int(np.argmax(corr))
    return j


@dataclass
class ConditionalGreedyResult:
    curve: ProgressiveCurve
    n_unique_sets: list[int]
    mean_path_jaccard: float


def evaluate_conditional_greedy_progressive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_features: int = 25,
    k_neighbors: int = 80,
    n_index: int | None = 6000,
    budgets: list[int] | None = None,
    random_state: int | None = 0,
    name: str = "local_residual_greedy",
) -> ConditionalGreedyResult:
    """Anytime RMSE for local residual-greedy acquisition + nested global OLS.

    Parameters
    ----------
    max_features
        Acquisition horizon (menu length cap).
    k_neighbors
        Neighbourhood size in the *observed* subspace (after the first feature).
    n_index
        Subsample of training points used for neighbour search (OLS still uses
        the full training sample). ``None`` uses all training rows.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()
    n_tr, p = X_train.shape
    n_te = X_test.shape[0]
    B = int(min(max_features, p))
    if budgets is None:
        budgets = list(range(0, B + 1))

    rng = np.random.default_rng(random_state)
    if n_index is None or n_index >= n_tr:
        index = np.arange(n_tr)
    else:
        index = rng.choice(n_tr, size=int(n_index), replace=False)
    Xi, yi = X_train[index], y_train[index]
    n_idx = Xi.shape[0]
    k_use = int(min(k_neighbors, n_idx))

    j0 = _first_feature(X_train, y_train)
    selected: list[list[int]] = [[j0] for _ in range(n_te)]

    def _choose_next(S: list[int], query_ids: np.ndarray) -> None:
        """Pick the next feature for queries that currently share path ``S``."""
        S_arr = np.asarray(S, dtype=int)
        X_obs = Xi[:, S_arr]
        x_obs = X_test[query_ids][:, S_arr]
        # Pairwise distances: n_q x n_idx
        d = np.sum((x_obs[:, None, :] - X_obs[None, :, :]) ** 2, axis=2)
        kk = min(k_use, n_idx)
        neigh = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for row, i in enumerate(query_ids):
            loc = neigh[row]
            gain = _rss_gains(Xi[loc], yi[loc], S)
            if not np.any(np.isfinite(gain) & (gain > 0)):
                gain = _rss_gains(X_train, y_train, S)
            j = int(np.argmax(gain))
            if j in S or not np.isfinite(gain[j]) or gain[j] <= 0:
                unused = [u for u in range(p) if u not in S]
                j = unused[0] if unused else S[-1]
            selected[int(i)] = S + [j]

    for _step in range(1, B):
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, S in enumerate(selected):
            groups[tuple(S)].append(i)
        for key, ids in groups.items():
            _choose_next(list(key), np.asarray(ids, dtype=int))

    ols_cache: dict[tuple[int, ...], PrefixModel] = {}

    def predict_budget(b: int) -> np.ndarray:
        yhat = np.empty(n_te, dtype=float)
        if b <= 0:
            yhat.fill(float(y_train.mean()))
            return yhat
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i in range(n_te):
            groups[tuple(selected[i][:b])].append(i)
        for feats, ids in groups.items():
            model = ols_cache.get(feats)
            if model is None:
                model = _ols_on_set(X_train, y_train, list(feats))
                ols_cache[feats] = model
            idx = np.asarray(ids, dtype=int)
            yhat[idx] = model.predict(X_test[idx])
        return yhat

    rmse_list, mae_list, used_list, n_sets = [], [], [], []
    sets_at_b: dict[int, list[frozenset[int]]] = {}
    for b in budgets:
        bb = int(max(0, min(int(b), B)))
        yhat = predict_budget(bb)
        rmse, mae = _metrics(y_test, yhat)
        rmse_list.append(rmse)
        mae_list.append(mae)
        used_list.append(float(bb))
        bags = [frozenset(selected[i][:bb]) for i in range(n_te)] if bb > 0 else [frozenset()]
        n_sets.append(len(set(bags)))
        sets_at_b[bb] = bags

    # Path diversity at the horizon (Jaccard of acquired sets).
    horizon = min(B, max(int(b) for b in budgets))
    bags_h = [frozenset(selected[i][:horizon]) for i in range(n_te)]
    pair_j = []
    rng2 = np.random.default_rng(0)
    n_pair = min(200, n_te * (n_te - 1) // 2)
    for _ in range(n_pair):
        a, c = rng2.integers(0, n_te, size=2)
        if a == c:
            continue
        sa, sc = bags_h[a], bags_h[c]
        pair_j.append(len(sa & sc) / max(len(sa | sc), 1))
    mean_j = float(np.mean(pair_j)) if pair_j else 1.0

    curve = ProgressiveCurve(
        name=name,
        budgets=np.asarray(list(budgets), dtype=int),
        rmse=np.asarray(rmse_list),
        mae=np.asarray(mae_list),
        mean_features_used=np.asarray(used_list),
    )
    return ConditionalGreedyResult(
        curve=curve,
        n_unique_sets=n_sets,
        mean_path_jaccard=mean_j,
    )
