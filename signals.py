"""
signals.py — Score composite pour sélection Long-Only des top N titres.

Facteurs du score composite :
  1. Momentum 12m-1m  : rendement cumulé sur 12 mois, excluant le dernier mois
  2. Vol Mean Reversion : ratio vol courte / vol longue (inversé)
  3. Vol Spread        : volatilité réalisée 12m inversée (long low-vol)

Le score final est un z-score pondéré de ces 3 facteurs.
"""

import pandas as pd
import numpy as np
from config import N_STOCKS, SIGNAL_WEIGHTS, BUFFER_RANK


# ═══════════════════════════════════════════════════════════════════
#  Signaux individuels
# ═══════════════════════════════════════════════════════════════════

def momentum_12m1m(prices: pd.DataFrame, t: int) -> pd.Series:
    """
    Momentum 12m - 1m :  rendement sur les 12 derniers mois,
    sans le dernier mois (pour éviter le reversal court terme).
    """
    if t < 12:
        return pd.Series(dtype=float)

    p_now   = prices.iloc[t]
    p_1m    = prices.iloc[t - 1]
    p_12m   = prices.iloc[t - 12]

    ret_12m = p_now / p_12m - 1
    ret_1m  = p_now / p_1m - 1
    signal = ret_12m - ret_1m

    return signal.dropna()


def mean_reversion_1m(prices: pd.DataFrame, t: int, lookback: int = 12) -> pd.Series:
    """
    Vol Mean Reversion : ratio vol courte (3m) / vol longue (12m), inversé.
    Ratio bas = vol en compression → score élevé (favorable).
    """
    if t < lookback:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - lookback:t + 1]
    rets = sub.pct_change().iloc[1:]

    vol_long = rets.std()
    vol_long = vol_long.replace(0, np.nan)

    short_window = 3
    if len(rets) >= short_window:
        vol_short = rets.iloc[-short_window:].std()
    else:
        vol_short = rets.std()

    ratio = vol_short / vol_long
    signal = -ratio

    return signal.dropna()


def vol_spread(prices: pd.DataFrame, t: int, window: int = 12) -> pd.Series:
    """
    Vol Spread (low-vol premium) :
    Volatilité réalisée sur les `window` derniers mois, inversée.
    Les moins volatils ont un score élevé.
    """
    if t < window:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - window:t + 1]
    rets = sub.pct_change().iloc[1:]
    vol = rets.std()

    signal = -vol
    return signal.dropna()


# ═══════════════════════════════════════════════════════════════════
#  Dispatch : appeler un signal par nom
# ═══════════════════════════════════════════════════════════════════

SIGNAL_FUNCS = {
    "momentum":        momentum_12m1m,
    "mean_reversion":  mean_reversion_1m,
    "vol_spread":      vol_spread,
}


def compute_signal(signal_name: str, prices: pd.DataFrame, t: int) -> pd.Series:
    """Calcule le signal demandé à la date t."""
    func = SIGNAL_FUNCS[signal_name]
    return func(prices, t)


# ═══════════════════════════════════════════════════════════════════
#  Score composite + sélection top N
# ═══════════════════════════════════════════════════════════════════

def _zscore(s: pd.Series) -> pd.Series:
    """Standardise une Series en z-score."""
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return s * 0.0
    return (s - mu) / sigma


def compute_composite_score(
    prices: pd.DataFrame,
    t: int,
    members: list | None = None,
    weights: dict | None = None,
) -> pd.Series:
    """
    Calcule le score composite pondéré (z-score) à partir de tous les signaux.

    Parameters
    ----------
    prices  : DataFrame (date × ticker)
    t       : index position de la date de rebalancement
    members : tickers de l'univers à cette date
    weights : dict signal_name → poids (défaut = SIGNAL_WEIGHTS)

    Returns
    -------
    score : Series(ticker → score composite), trié décroissant
    """
    if weights is None:
        weights = SIGNAL_WEIGHTS

    all_zscores = {}
    for sig_name, sig_weight in weights.items():
        raw = compute_signal(sig_name, prices, t)
        if raw.empty:
            continue
        if members is not None:
            raw = raw.reindex(members).dropna()
        all_zscores[sig_name] = _zscore(raw) * sig_weight

    if not all_zscores:
        return pd.Series(dtype=float)

    # Aligner tous les z-scores sur les mêmes tickers
    df_z = pd.DataFrame(all_zscores)
    # Garder uniquement les tickers ayant tous les signaux
    df_z = df_z.dropna()

    if df_z.empty:
        return pd.Series(dtype=float)

    composite = df_z.sum(axis=1)
    return composite.sort_values(ascending=False)


def select_top_n(
    prices: pd.DataFrame,
    t: int,
    members: list | None = None,
    n: int = N_STOCKS,
    prev_stocks: list | None = None,
    buffer_rank: int = BUFFER_RANK,
) -> list:
    """
    Sélectionne les top N titres par score composite avec mécanisme de buffer.

    Le buffer réduit le turnover : un titre déjà en portefeuille n'est remplacé
    que s'il tombe en dessous du rang `buffer_rank` (défaut 40).
    Les nouveaux titres doivent être dans le top N strict.

    Returns
    -------
    list de tickers (N titres)
    """
    score = compute_composite_score(prices, t, members=members)
    if score.empty:
        return []

    # Premier mois ou pas de portefeuille précédent : top N strict
    if prev_stocks is None or len(prev_stocks) == 0:
        return score.head(n).index.tolist()

    # Buffer : garder les titres existants qui sont encore dans le top buffer_rank
    top_buffer = score.head(buffer_rank).index.tolist()
    top_strict = score.head(n).index.tolist()

    # 1. Garder les anciens titres qui sont encore dans le buffer
    kept = [s for s in prev_stocks if s in top_buffer]

    # 2. Compléter avec les meilleurs nouveaux titres (top strict) si on n'a pas assez
    if len(kept) < n:
        for s in top_strict:
            if s not in kept:
                kept.append(s)
            if len(kept) >= n:
                break

    # 3. Si on a trop de titres (cas rare), garder les N meilleurs par score
    if len(kept) > n:
        kept_scores = score.reindex(kept).dropna().sort_values(ascending=False)
        kept = kept_scores.head(n).index.tolist()

    return kept
