"""
backtest.py — Moteur de backtest Long-Only ERC.

Exécute le backtest avec score composite + allocation ERC,
et renvoie les PnL, equity curve, turnover, coûts et les poids de la dernière période.
"""

import pandas as pd
import numpy as np

from config import START_DATE, END_DATE, N_STOCKS, REBAL_FREQ
from signals import select_top_n
from allocation import allocate
from costs import compute_turnover, compute_cost
from data_loader import get_members_at_date
from risk import (
    stoploss_position,
    stoploss_trailing,
    stoploss_portfolio,
    stoploss_volatility,
)


# ═══════════════════════════════════════════════════════════════════
#  Dispatch stop-loss (long-only : on ne gère que les positions > 0)
# ═══════════════════════════════════════════════════════════════════

def _apply_stoploss(stoploss_type: str,
                    weights: pd.Series,
                    **kwargs) -> pd.Series:
    """Applique la règle de stop-loss demandée."""
    if stoploss_type == "position":
        return stoploss_position(
            weights,
            kwargs["entry_prices"],
            kwargs["current_prices"],
        )
    elif stoploss_type == "trailing":
        return stoploss_trailing(
            weights,
            kwargs["peak_prices"],
            kwargs["current_prices"],
        )
    elif stoploss_type == "portfolio":
        return stoploss_portfolio(
            weights,
            kwargs["portfolio_value"],
            kwargs["portfolio_peak"],
        )
    elif stoploss_type == "volatility":
        return stoploss_volatility(
            weights,
            kwargs["portfolio_returns"],
        )
    else:
        return weights


# ═══════════════════════════════════════════════════════════════════
#  Backtest Long-Only
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    df_long: pd.DataFrame,
    stoploss_type: str = "position",
    verbose: bool = False,
) -> dict:
    """
    Exécute le backtest Long-Only avec score composite + ERC.

    Parameters
    ----------
    prices       : DataFrame (date × ticker) — prix mensuels
    returns      : DataFrame (date × ticker) — rendements mensuels
    df_long      : DataFrame format long (date, ticker, px_last)
    stoploss_type: type de stop-loss ("position", "trailing", "portfolio", "volatility")
    verbose      : affiche le détail à chaque date

    Returns
    -------
    dict avec clés : pnl, cumulative, turnover_log, cost_log, n_stocks,
                     last_weights, last_date
    """
    # ── Dates de rebalancement dans la fenêtre selon REBAL_FREQ ──
    mask = (prices.index >= pd.Timestamp(START_DATE)) & \
           (prices.index <= pd.Timestamp(END_DATE))
    all_dates = prices.index[mask]
    
    # Appliquer la fréquence de rebalancement configurée
    if REBAL_FREQ == "M":  # Mensuel
        rebal_dates = all_dates.tolist()
    elif REBAL_FREQ == "Q":  # Trimestriel (fin de trimestre)
        # Prendre le dernier jour de chaque trimestre
        quarterly_ends = all_dates.to_period('Q').drop_duplicates(keep='last')
        rebal_dates = [all_dates[all_dates.to_period('Q') == qtr][-1] for qtr in quarterly_ends]
    elif REBAL_FREQ == "A":  # Annuel
        yearly_ends = all_dates.to_period('Y').drop_duplicates(keep='last')
        rebal_dates = [all_dates[all_dates.to_period('Y') == yr][-1] for yr in yearly_ends]
    else:
        # Fallback sur mensuel
        rebal_dates = all_dates.tolist()

    # ── Listes de résultats ──
    pnl_list      = []
    cumul_list    = []
    turnover_list = []
    cost_list     = []
    n_stocks_list = []
    dates_used    = []
    weight_history = []  # Historique des poids pour analyse sectorielle

    nav          = 1.0
    nav_peak     = 1.0
    prev_weights = pd.Series(dtype=float)
    prev_stocks  = []  # pour le buffer de turnover
    entry_prices = pd.Series(dtype=float)
    peak_prices  = pd.Series(dtype=float)
    port_rets_so_far = pd.Series(dtype=float)
    last_weights = pd.Series(dtype=float)

    for date in rebal_dates:
        t = prices.index.get_loc(date)

        # ── 1. Univers à la date ──
        members = get_members_at_date(df_long, date)

        # ── 2. Sélection top N par score composite (avec buffer) ──
        stock_list = select_top_n(prices, t, members=members, n=N_STOCKS,
                                  prev_stocks=prev_stocks)
        if not stock_list:
            continue

        # ── 3. Allocation ERC Long-Only ──
        weights = allocate(stock_list, returns=returns.iloc[:t + 1])

        # ── 4. Stop-loss ──
        current_prices_row = prices.iloc[t]

        if stoploss_type == "position" and not entry_prices.empty:
            weights = _apply_stoploss(
                "position", weights,
                entry_prices=entry_prices,
                current_prices=current_prices_row,
            )
        elif stoploss_type == "trailing" and not peak_prices.empty:
            weights = _apply_stoploss(
                "trailing", weights,
                peak_prices=peak_prices,
                current_prices=current_prices_row,
            )
        elif stoploss_type == "portfolio":
            weights = _apply_stoploss(
                "portfolio", weights,
                portfolio_value=nav,
                portfolio_peak=nav_peak,
            )
        elif stoploss_type == "volatility" and len(port_rets_so_far) >= 2:
            weights = _apply_stoploss(
                "volatility", weights,
                portfolio_returns=port_rets_so_far,
            )

        # Renormaliser les poids après stop-loss (on reste 100% investi)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum

        # ── 5. Turnover & coûts ──
        to   = compute_turnover(weights, prev_weights)
        cost = compute_cost(to)

        # ── 6. Rendement du portefeuille sur la période ──
        period_ret = returns.iloc[t]
        common = weights.index.intersection(period_ret.dropna().index)
        port_ret = (weights[common] * period_ret[common]).sum() - cost

        # ── 7. Mise à jour de la NAV ──
        nav *= (1 + port_ret)
        nav_peak = max(nav_peak, nav)

        # ── 8. Mise à jour du suivi des positions ──
        new_entry = current_prices_row.reindex(weights.index).copy()
        for ticker in weights.index:
            if ticker in prev_weights.index and prev_weights.get(ticker, 0) != 0:
                if ticker in entry_prices.index:
                    new_entry[ticker] = entry_prices[ticker]
        entry_prices = new_entry.dropna()

        # Peak prices (pour trailing stop)
        new_peak = current_prices_row.reindex(weights.index).copy()
        if not peak_prices.empty:
            for ticker in weights.index:
                curr = current_prices_row.get(ticker, np.nan)
                if pd.isna(curr):
                    continue
                old_peak = peak_prices.get(ticker, np.nan)
                if pd.isna(old_peak):
                    continue
                new_peak[ticker] = max(old_peak, curr)
        peak_prices = new_peak.dropna()

        # ── 9. Stocker les résultats ──
        pnl_list.append(port_ret)
        cumul_list.append(nav)
        turnover_list.append(to)
        cost_list.append(cost)
        n_stocks_list.append(len(stock_list))
        dates_used.append(date)
        weight_history.append(weights.copy())  # Sauvegarder les poids

        port_rets_so_far = pd.Series(pnl_list, index=dates_used)
        prev_weights = weights.copy()
        prev_stocks = stock_list
        last_weights = weights.copy()

        if verbose:
            print(f"  {date.strftime('%Y-%m')} | "
                  f"N={len(stock_list):3d} | "
                  f"Ret={port_ret:+.2%} | "
                  f"NAV={nav:.4f} | "
                  f"TO={to:.2%}")

    # ── Construction des Series de sortie ──
    pnl          = pd.Series(pnl_list,      index=dates_used, name="pnl")
    cumulative   = pd.Series(cumul_list,     index=dates_used, name="cumulative")
    turnover_log = pd.Series(turnover_list,  index=dates_used, name="turnover")
    cost_log     = pd.Series(cost_list,      index=dates_used, name="cost")
    n_stocks_log = pd.Series(n_stocks_list,  index=dates_used, name="n_stocks")

    return {
        "pnl":            pnl,
        "cumulative":     cumulative,
        "turnover_log":   turnover_log,
        "cost_log":       cost_log,
        "n_stocks":       n_stocks_log,
        "last_weights":   last_weights,
        "last_date":      dates_used[-1] if dates_used else None,
        "weight_history": weight_history,
        "dates_history":  dates_used,
    }
