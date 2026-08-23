"""VIF regression: streamwise feature selection with alpha-investing.

Based on Lin, Foster & Ungar, "VIF Regression: A Fast Regression Algorithm
For Large Data" — one-pass candidate testing with a subsample-corrected
t-statistic (variance inflation factor) and an alpha-investing rule for
mFDR control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm


@dataclass
class VIFResult:
    """Outcome of a VIF streamwise selection run."""

    selected: list[int]
    """Feature indices in the order they were added (acquisition menu)."""

    coefficients: np.ndarray
    """OLS coefficients on [intercept] + selected features (length 1+|selected|)."""

    intercept: float
    coef: np.ndarray
    """Coefficients for selected features only (aligned with ``selected``)."""

    wealth_path: list[float] = field(default_factory=list)
    pvalues: dict[int, float] = field(default_factory=dict)
    """Two-sided p-value at the moment each selected feature was tested."""

    n_tested: int = 0
    subsample_size: int = 0

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


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return X


def _center_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    return X - means, means


def _ols_fit(Xc: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return coef (no intercept on centered data), fitted values, RMSE."""
    n, k = Xc.shape
    if k == 0:
        fitted = np.zeros_like(y)
        df = max(n - 1, 1)
        rmse = float(np.sqrt(np.sum(y**2) / df))
        return np.empty(0), fitted, rmse

    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    fitted = Xc @ coef
    resid = y - fitted
    df = max(n - k - 1, 1)
    rmse = float(np.sqrt(np.sum(resid**2) / df))
    return coef, fitted, rmse


def _vif_t_statistic(
    r: np.ndarray,
    x_new: np.ndarray,
    X_selected_sub: np.ndarray | None,
    x_new_sub: np.ndarray,
    sigma_null: float,
) -> float:
    """Corrected t-ratio from the Fast Evaluation Procedure."""
    norm_x = np.linalg.norm(x_new)
    if norm_x < 1e-12 or sigma_null < 1e-12:
        return 0.0

    x_unit = x_new / norm_x
    gamma_hat = float(np.dot(r, x_unit))

    # Estimate rho^2 = 1 - R^2 of x_new on currently selected features (subsample).
    if X_selected_sub is None or X_selected_sub.shape[1] == 0:
        rho2 = 1.0
    else:
        x_sub = x_new_sub - x_new_sub.mean()
        Xs = X_selected_sub - X_selected_sub.mean(axis=0, keepdims=True)
        # Guard near-collinear subsample design.
        try:
            beta_x, *_ = np.linalg.lstsq(Xs, x_sub, rcond=None)
            fitted_x = Xs @ beta_x
            ss_tot = float(np.dot(x_sub, x_sub))
            if ss_tot < 1e-12:
                return 0.0
            r2 = float(np.dot(fitted_x, fitted_x) / ss_tot)
            r2 = min(max(r2, 0.0), 1.0 - 1e-10)
            rho2 = 1.0 - r2
        except np.linalg.LinAlgError:
            return 0.0

    denom = sigma_null * np.sqrt(rho2)
    if denom < 1e-12:
        return 0.0
    return gamma_hat / denom


def vif_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_order: Iterable[int] | None = None,
    w0: float = 0.50,
    delta_w: float = 0.05,
    subsample_size: int | None = None,
    random_state: int | None = None,
    max_features: int | None = None,
    min_wealth: float = 1e-12,
) -> VIFResult:
    """Run VIF streamwise feature selection.

    Parameters
    ----------
    X, y:
        Design matrix and response. Columns of X are candidate features.
    feature_order:
        Order in which to test features (default: 0..p-1). This order becomes
        the progressive-revelation candidate stream.
    w0, delta_w:
        Alpha-investing initial wealth and payout (paper defaults 0.50, 0.05).
    subsample_size:
        Rows used to estimate the VIF correction. Defaults to
        ``min(n, max(50, int(5 * sqrt(n))))``.
    max_features:
        Optional cap on selected features.
    """
    X = _as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    n, p = X.shape
    if n < 3:
        raise ValueError("Need at least 3 observations for VIF regression")

    rng = np.random.default_rng(random_state)
    order = list(range(p) if feature_order is None else feature_order)
    for j in order:
        if j < 0 or j >= p:
            raise ValueError(f"feature index {j} out of range [0, {p})")

    if subsample_size is None:
        subsample_size = int(min(n, max(50, 5 * np.sqrt(n))))
    subsample_size = int(min(max(subsample_size, 10), n))

    # Center features and response for the selection loop.
    Xc, x_means = _center_columns(X)
    y_mean = float(y.mean())
    yc = y - y_mean

    idx_sub = np.sort(rng.choice(n, size=subsample_size, replace=False))
    X_sub = Xc[idx_sub]

    selected: list[int] = []
    r = yc.copy()
    _, _, sigma = _ols_fit(np.empty((n, 0)), yc)
    # Paper initializes sigma with sd(y); after centering this matches.
    sigma = float(np.std(y, ddof=1)) if n > 1 else 1.0

    wealth = float(w0)
    wealth_path = [wealth]
    last_reject_step = 0
    pvalues: dict[int, float] = {}
    step = 0

    for j in order:
        if wealth <= min_wealth:
            break
        if max_features is not None and len(selected) >= max_features:
            break

        step += 1
        alpha_i = wealth / (1.0 + step - last_reject_step)
        alpha_i = float(min(max(alpha_i, 0.0), 1.0 - 1e-12))

        X_sel_sub = X_sub[:, selected] if selected else None
        t_stat = _vif_t_statistic(
            r=r,
            x_new=Xc[:, j],
            X_selected_sub=X_sel_sub,
            x_new_sub=X_sub[:, j],
            sigma_null=sigma,
        )
        # Two-sided p-value (statistically standard form of the paper's test).
        p_value = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))

        if p_value < alpha_i:
            selected.append(j)
            pvalues[j] = p_value
            coef_c, fitted, sigma = _ols_fit(Xc[:, selected], yc)
            r = yc - fitted
            wealth = wealth + delta_w
            last_reject_step = step
        else:
            wealth = wealth - alpha_i / (1.0 - alpha_i)

        wealth_path.append(wealth)

    # Final OLS on original (uncentered) scale with intercept.
    if selected:
        X_final = np.column_stack([np.ones(n), X[:, selected]])
        beta, *_ = np.linalg.lstsq(X_final, y, rcond=None)
        intercept = float(beta[0])
        coef = beta[1:].astype(float)
        coefficients = beta.astype(float)
    else:
        intercept = y_mean
        coef = np.empty(0)
        coefficients = np.array([intercept], dtype=float)

    return VIFResult(
        selected=selected,
        coefficients=coefficients,
        intercept=intercept,
        coef=coef,
        wealth_path=wealth_path,
        pvalues=pvalues,
        n_tested=step,
        subsample_size=subsample_size,
    )
