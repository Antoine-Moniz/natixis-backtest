"""
risk.py — Stop-loss / règles de risk management (Long-Only).

Stop-loss :
  1. Position absolue     : clôture si perte > X % depuis entrée
  2. Trailing             : clôture si perte > X % depuis le pic
  3. Portfolio drawdown   : réduit l'exposition si drawdown portefeuille > seuil
  4. Volatility-based     : réduit l'exposition si vol réalisée dépasse un seuil
"""

import pandas as pd
import numpy as np
from config import (
    STOPLOSS_POSITION, STOPLOSS_TRAILING,
    STOPLOSS_PORTFOLIO, STOPLOSS_VOL_WINDOW, STOPLOSS_VOL_MULT,
)


# ═══════════════════════════════════════════════════════════════════
#  1.  Stop-loss par position (absolu depuis l'entrée)
# ═══════════════════════════════════════════════════════════════════

def stoploss_position(
    weights: pd.Series,
    entry_prices: pd.Series,
    current_prices: pd.Series,
    threshold: float = STOPLOSS_POSITION,
) -> pd.Series:
    """
    Ferme les positions dont la perte depuis l'entrée dépasse `threshold`.

    Parameters
    ----------
    weights        : Series(ticker → weight) courant
    entry_prices   : Series(ticker → prix d'entrée)
    current_prices : Series(ticker → prix courant)
    threshold      : seuil (ex: -0.10 pour -10 %)

    Returns
    -------
    weights ajustés (positions stoppées mises à 0)
    """
    new_w = weights.copy()
    common = weights.index.intersection(entry_prices.index).intersection(current_prices.index)

    for t in common:
        if weights[t] == 0 or entry_prices[t] == 0:
            continue
        pnl_pct = (current_prices[t] / entry_prices[t]) - 1

        # Long : on ferme si le prix a trop baissé
        if weights[t] > 0 and pnl_pct < threshold:
            new_w[t] = 0.0

    return new_w


# ═══════════════════════════════════════════════════════════════════
#  2.  Stop-loss trailing (depuis le pic)
# ═══════════════════════════════════════════════════════════════════

def stoploss_trailing(
    weights: pd.Series,
    peak_prices: pd.Series,
    current_prices: pd.Series,
    threshold: float = STOPLOSS_TRAILING,
) -> pd.Series:
    """
    Ferme les positions dont la perte depuis le plus haut (pour long)
    ou le plus bas (pour short) dépasse `threshold`.

    Parameters
    ----------
    peak_prices : Series(ticker → prix max/min depuis l'entrée)
    """
    new_w = weights.copy()
    common = weights.index.intersection(peak_prices.index).intersection(current_prices.index)

    for t in common:
        if weights[t] == 0 or peak_prices[t] == 0:
            continue

        if weights[t] > 0:
            # Long : drawdown depuis le peak
            dd = (current_prices[t] / peak_prices[t]) - 1
            if dd < threshold:
                new_w[t] = 0.0

    return new_w


# ═══════════════════════════════════════════════════════════════════
#  3.  Stop-loss portefeuille (drawdown max du portefeuille)
# ═══════════════════════════════════════════════════════════════════

def stoploss_portfolio(
    weights: pd.Series,
    portfolio_value: float,
    portfolio_peak: float,
    threshold: float = STOPLOSS_PORTFOLIO,
) -> pd.Series:
    """
    Si le drawdown du portefeuille dépasse le seuil, réduit toutes les
    positions de moitié (deleveraging).

    Returns
    -------
    weights ajustés (multipliés par 0.5 si stop déclenché, sinon inchangés)
    """
    if portfolio_peak <= 0:
        return weights

    dd = (portfolio_value / portfolio_peak) - 1

    if dd < threshold:
        return weights * 0.5  # deleveraging
    return weights.copy()


# ═══════════════════════════════════════════════════════════════════
#  4.  Stop-loss volatility-based
# ═══════════════════════════════════════════════════════════════════

def stoploss_volatility(
    weights: pd.Series,
    portfolio_returns: pd.Series,
    vol_window: int = STOPLOSS_VOL_WINDOW,
    vol_mult: float = STOPLOSS_VOL_MULT,
    target_vol: float | None = None,
) -> pd.Series:
    """
    Réduit l'exposition proportionnellement si la volatilité réalisée
    dépasse `vol_mult × target_vol`.

    Si target_vol n'est pas spécifié, on utilise la vol historique complète
    comme cible.

    Returns
    -------
    weights ajustés (scale down si vol réalisée > seuil)
    """
    if len(portfolio_returns) < vol_window:
        return weights.copy()

    recent_vol = portfolio_returns.iloc[-vol_window:].std()

    if target_vol is None:
        target_vol = portfolio_returns.std()

    if target_vol <= 0 or recent_vol <= 0:
        return weights.copy()

    if recent_vol > vol_mult * target_vol:
        scale = target_vol / recent_vol
        return weights * scale

    return weights.copy()


# ═══════════════════════════════════════════════════════════════════
#  5.  Dispatch
# ═══════════════════════════════════════════════════════════════════

STOPLOSS_NAMES = [
    "position",
    "trailing",
    "portfolio",
    "volatility",
]
