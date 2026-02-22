"""
allocation.py — Méthodes d'allocation market-neutral.

Méthodes :
  1. Equal Weight  (EW)
  2. Equal Risk Contribution (ERC)

Contrainte market-neutral : Σ w_long = +1,  Σ w_short = -1
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════
#  1. Equal Weight
# ═══════════════════════════════════════════════════════════════════

def equal_weight(long_list: list, short_list: list) -> pd.Series:
    """
    Allocation Equal Weight market-neutral.
    - Chaque long  : +1 / n_long
    - Chaque short : -1 / n_short

    Returns : Series(ticker → weight)
    """
    weights = {}
    n_long  = len(long_list)
    n_short = len(short_list)

    if n_long > 0:
        w_l = 1.0 / n_long
        for t in long_list:
            weights[t] = w_l

    if n_short > 0:
        w_s = -1.0 / n_short
        for t in short_list:
            weights[t] = w_s

    return pd.Series(weights)


# ═══════════════════════════════════════════════════════════════════
#  2. Equal Risk Contribution (ERC)
# ═══════════════════════════════════════════════════════════════════

def _erc_weights_long_only(cov: pd.DataFrame, tickers: list) -> np.ndarray:
    """
    Résout le problème ERC (equal risk contribution) pour un groupe
    de tickers (long seulement, poids positifs normalisés à 1).

    min  Σ_i Σ_j ( w_i * (Σ w)_i - w_j * (Σ w)_j )^2

    On utilise l'approche classique d'optimisation numérique.
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
        # risk contributions
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


def erc_weights(returns: pd.DataFrame,
                long_list: list,
                short_list: list,
                window: int = 12) -> pd.Series:
    """
    Allocation ERC market-neutral.
    On calcule la matrice de covariance sur les `window` dernières observations,
    puis on optimise séparément pour le côté long et le côté short.

    Returns : Series(ticker → weight), avec Σ long = +1 et Σ short = -1.
    """
    all_tickers = long_list + short_list
    sub = returns[all_tickers].dropna(axis=0, how="all")

    # Si pas assez de données → fallback equal weight
    if len(sub) < max(window, 3):
        return equal_weight(long_list, short_list)

    # Covariance sur les dernières observations
    cov = sub.iloc[-window:].cov()

    weights = {}

    # Long
    if long_list:
        w_l = _erc_weights_long_only(cov, long_list)
        for i, t in enumerate(long_list):
            weights[t] = w_l[i]   # positif, somme = 1

    # Short
    if short_list:
        w_s = _erc_weights_long_only(cov, short_list)
        for i, t in enumerate(short_list):
            weights[t] = -w_s[i]  # négatif, somme = -1

    return pd.Series(weights)


# ═══════════════════════════════════════════════════════════════════
#  3. Dispatch par nom
# ═══════════════════════════════════════════════════════════════════

ALLOC_FUNCS = {
    "equal_weight": lambda long, short, **kw: equal_weight(long, short),
    "erc":          lambda long, short, **kw: erc_weights(kw["returns"], long, short),
}


def allocate(method: str, long_list: list, short_list: list, **kwargs) -> pd.Series:
    """Retourne les poids market-neutral pour la méthode choisie."""
    return ALLOC_FUNCS[method](long_list, short_list, **kwargs)
