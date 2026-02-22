"""
signal.py — Calcul des 3 signaux cross-section (basés prix uniquement).

Signaux :
  1. Momentum 12m-1m  : rendement cumulé sur 12 mois, excluant le dernier mois
  2. Mean Reversion 1m : rendement du dernier mois (inversé → short-term reversal)
  3. Vol Spread        : volatilité réalisée 12m (long low-vol / short high-vol)
"""

import pandas as pd
import numpy as np
from config import QUANTILE_LONG, QUANTILE_SHORT


# ═══════════════════════════════════════════════════════════════════
#  Signaux individuels
# ═══════════════════════════════════════════════════════════════════

def momentum_12m1m(prices: pd.DataFrame, t: int) -> pd.Series:
    """
    Momentum 12m - 1m :  rendement sur les 12 derniers mois,
    sans le dernier mois (pour éviter le reversal court terme).

    Parameters
    ----------
    prices : DataFrame (date × ticker)
    t      : index (position) de la date de rebal dans prices.index

    Returns
    -------
    signal : Series(ticker → score), NaN si pas assez d'historique
    """
    if t < 12:
        return pd.Series(dtype=float)

    p_now   = prices.iloc[t]          # prix à t
    p_1m    = prices.iloc[t - 1]      # prix à t-1
    p_12m   = prices.iloc[t - 12]     # prix à t-12

    # rendement 12m
    ret_12m = p_now / p_12m - 1
    # rendement 1m
    ret_1m  = p_now / p_1m - 1
    # momentum = 12m - 1m (skip le dernier mois)
    signal = ret_12m - ret_1m

    return signal.dropna()


def mean_reversion_1m(prices: pd.DataFrame, t: int, lookback: int = 12) -> pd.Series:
    """
    Mean Reversion de Volatilite (Volatility Mean Reversion).

    Constat empirique : sur large-caps US mensuelles (2016-2025), toute
    strategie de reversal de PRIX (acheter les perdants, vendre les gagnants)
    echoue car le momentum domine massivement a cet horizon.

    En revanche, la VOLATILITE est un processus fortement mean-reverting :
    apres un pic de volatilite, la vol tend a se normaliser.
    Ce phenomene est la base du "Volatility Risk Premium".

    Signal en 3 etapes :
      1. Calculer la vol realisee courte (3 derniers mois)
      2. Calculer la vol realisee longue (12 derniers mois)
      3. Ratio = vol_courte / vol_longue
         - Ratio BAS = volatilite en compression (retour au calme)
           → rendements futurs ajustes au risque favorables → LONG
         - Ratio HAUT = volatilite en expansion (turbulence recente)
           → incertitude elevee, rendements futurs defavorables → SHORT

    Difference avec le signal Vol Spread :
      - Vol Spread : classe par le NIVEAU de vol (long les plus stables absolu)
      - Vol Mean Reversion : classe par la DYNAMIQUE de vol
        (long les titres dont la vol se normalise, short ceux dont elle explose)
      Example : un titre historiquement volatil mais dont la vol BAISSE
      recemment serait SHORT en Vol Spread mais LONG en Vol Mean Reversion.

    Parameters
    ----------
    prices   : DataFrame (date x ticker)
    t        : index position de la date de rebal
    lookback : fenetre pour la vol longue (defaut 12 mois)

    Returns
    -------
    signal : Series(ticker -> score)
    """
    if t < lookback:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - lookback:t + 1]
    rets = sub.pct_change().iloc[1:]           # (lookback, n_tickers)

    # ── Vol longue (12 mois) ──
    vol_long = rets.std()
    vol_long = vol_long.replace(0, np.nan)

    # ── Vol courte (3 derniers mois) ──
    short_window = 3
    if len(rets) >= short_window:
        vol_short = rets.iloc[-short_window:].std()
    else:
        vol_short = rets.std()

    # ── Ratio vol courte / vol longue ──
    ratio = vol_short / vol_long

    # Signal : ratio bas = vol en compression → LONG, ratio haut → SHORT
    signal = -ratio

    return signal.dropna()


def vol_spread(prices: pd.DataFrame, t: int, window: int = 12) -> pd.Series:
    """
    Vol Spread (Long low-vol / Short high-vol) :
    Volatilité réalisée sur les `window` derniers mois (écart-type des
    rendements mensuels). Signal inversé → les moins volatils ont un score élevé.

    Parameters
    ----------
    prices : DataFrame (date × ticker)
    t      : index position de la date de rebal
    window : nb de mois pour le calcul de vol

    Returns
    -------
    signal : Series(ticker → score)
    """
    if t < window:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - window:t + 1]
    rets = sub.pct_change().iloc[1:]   # rendements mensuels sur la fenêtre
    vol = rets.std()

    # signal inversé : faible vol = score élevé (on veut long les low-vol)
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
#  Construction des buckets Long / Short
# ═══════════════════════════════════════════════════════════════════

def make_long_short_buckets(
    signal: pd.Series,
    members: list | None = None,
    q_long: float = QUANTILE_LONG,
    q_short: float = QUANTILE_SHORT,
) -> tuple[list, list]:
    """
    À partir d'un signal cross-section, sépare les tickers en :
      - long  : top q_long  (ex: top 20 %)
      - short : bottom q_short (ex: bottom 20 %)

    Parameters
    ----------
    signal  : Series(ticker → score)
    members : optionnel, restreint aux tickers présents dans l'univers
    q_long  : quantile haut (ex: 0.20 = top 20 %)
    q_short : quantile bas

    Returns
    -------
    (long_list, short_list)
    """
    if members is not None:
        signal = signal.reindex(members).dropna()

    if signal.empty:
        return [], []

    # Seuils
    threshold_long  = signal.quantile(1 - q_long)
    threshold_short = signal.quantile(q_short)

    long_list  = signal[signal >= threshold_long].index.tolist()
    short_list = signal[signal <= threshold_short].index.tolist()

    return long_list, short_list
