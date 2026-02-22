"""
costs.py — Calcul des coûts de transaction.

Convention :
  turnover  = Σ |w_t - w_{t-1}| / 2
  cost      = turnover × transaction_cost_bps / 10_000
"""

import pandas as pd
import numpy as np
from config import TRANSACTION_COST_BPS


def compute_turnover(weights_new: pd.Series, weights_old: pd.Series) -> float:
    """
    Turnover entre deux vecteurs de poids.
    Convention : Σ |w_new - w_old| / 2
    """
    if weights_old.empty:
        return np.abs(weights_new).sum() / 2.0
    if weights_new.empty:
        return np.abs(weights_old).sum() / 2.0

    # Aligner les index (tickers qui apparaissent / disparaissent)
    all_tickers = weights_new.index.union(weights_old.index)
    w_new = pd.Series(0.0, index=all_tickers)
    w_old = pd.Series(0.0, index=all_tickers)

    for t in weights_new.index:
        w_new[t] = weights_new[t]
    for t in weights_old.index:
        w_old[t] = weights_old[t]

    diff = (w_new - w_old).abs().sum() / 2.0
    return diff


def compute_cost(turnover: float, cost_bps: int = TRANSACTION_COST_BPS) -> float:
    """Coût de transaction = turnover × bps / 10 000."""
    return turnover * cost_bps / 10_000
