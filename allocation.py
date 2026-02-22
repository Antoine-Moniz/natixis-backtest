"""
allocation.py — Allocation Equal Risk Contribution (ERC) Long-Only.

Le portefeuille est 100% investi (Σ w = 1, w_i ≥ 0).
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════
#  ERC Long-Only
# ═══════════════════════════════════════════════════════════════════

def _erc_weights_long_only(cov: pd.DataFrame, tickers: list) -> np.ndarray:
    """
    Résout le problème ERC (equal risk contribution) pour un groupe
    de tickers (long seulement, poids positifs normalisés à 1).

    min  Σ_i Σ_j ( w_i * (Σ w)_i - w_j * (Σ w)_j )^2
    """
    n = len(tickers)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Sous-matrice covariance
    cov_sub = cov.loc[tickers, tickers].values

    def objective(w):
        sigma = w @ cov_sub @ w
        if sigma <= 0:
            return 1e12
        marginal = cov_sub @ w
        rc = w * marginal
        rc_target = sigma / n
        return np.sum((rc - rc_target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(1e-6, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(objective, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12})

    if res.success:
        return res.x / res.x.sum()  # normalisation sécurité
    else:
        # fallback : equal weight
        return np.ones(n) / n


def erc_allocation(returns: pd.DataFrame,
                   stock_list: list,
                   window: int = 12) -> pd.Series:
    """
    Allocation ERC Long-Only.
    Calcule la matrice de covariance sur les `window` dernières observations,
    puis optimise pour que chaque titre contribue également au risque.

    Returns : Series(ticker → weight), avec Σ = 1, tous positifs.
    """
    sub = returns[stock_list].dropna(axis=0, how="all")

    # Si pas assez de données → fallback equal weight
    if len(sub) < max(window, 3):
        n = len(stock_list)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=stock_list)

    # Covariance sur les dernières observations
    cov = sub.iloc[-window:].cov()

    w = _erc_weights_long_only(cov, stock_list)
    return pd.Series(w, index=stock_list)


def allocate(stock_list: list, returns: pd.DataFrame, window: int = 12) -> pd.Series:
    """Point d'entrée unique pour l'allocation ERC Long-Only."""
    return erc_allocation(returns, stock_list, window=window)
