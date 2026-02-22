"""
backtest.py — Moteur de backtest L/S market-neutral.

Exécute le backtest pour une combinaison (signal, allocation, stop-loss)
et renvoie les PnL, equity curve, turnover et coûts.
"""

import pandas as pd
import numpy as np

from config import START_DATE, END_DATE
from signals import compute_signal, make_long_short_buckets
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
#  Dispatch stop-loss
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
#  Backtest unitaire
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    df_long: pd.DataFrame,
    signal_name: str,
    alloc_method: str,
    stoploss_type: str = "position",
    verbose: bool = False,
) -> dict:
    """
    Exécute un backtest pour UNE combinaison (signal, allocation, stoploss).

    Parameters
    ----------
    prices       : DataFrame (date × ticker) — prix mensuels
    returns      : DataFrame (date × ticker) — rendements mensuels
    df_long      : DataFrame format long (date, ticker, px_last)
    signal_name  : nom du signal ("momentum", "mean_reversion", "vol_spread")
    alloc_method : méthode d'allocation ("equal_weight", "erc")
    stoploss_type: type de stop-loss ("position", "trailing", "portfolio", "volatility")
    verbose      : affiche le détail à chaque date

    Returns
    -------
    dict avec clés : pnl, cumulative, turnover_log, cost_log
    """
    # ── Dates de rebalancement dans la fenêtre ──
    mask = (prices.index >= pd.Timestamp(START_DATE)) & \
           (prices.index <= pd.Timestamp(END_DATE))
    rebal_dates = prices.index[mask].tolist()

    # ── Listes de résultats ──
    pnl_list      = []
    cumul_list    = []
    turnover_list = []
    cost_list     = []
    n_long_list   = []
    n_short_list  = []
    dates_used    = []

    nav          = 1.0
    nav_peak     = 1.0
    prev_weights = pd.Series(dtype=float)
    entry_prices = pd.Series(dtype=float)
    peak_prices  = pd.Series(dtype=float)
    port_rets_so_far = pd.Series(dtype=float)   # pour stop vol

    for date in rebal_dates:
        t = prices.index.get_loc(date)

        # ── 1. Signal ──
        signal = compute_signal(signal_name, prices, t)
        if signal.empty:
            continue

        # ── 2. Univers à la date ──
        members = get_members_at_date(df_long, date)

        # ── 3. Buckets Long / Short ──
        long_list, short_list = make_long_short_buckets(signal, members=members)
        if not long_list or not short_list:
            continue

        # ── 4. Allocation ──
        weights = allocate(
            alloc_method, long_list, short_list,
            returns=returns.iloc[:t + 1],
        )

        # ── 5. Stop-loss ──
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

        # ── 6. Turnover & coûts ──
        to   = compute_turnover(weights, prev_weights)
        cost = compute_cost(to)

        # ── 7. Rendement du portefeuille sur la période ──
        period_ret = returns.iloc[t]
        common = weights.index.intersection(period_ret.dropna().index)
        port_ret = (weights[common] * period_ret[common]).sum() - cost

        # ── 8. Mise à jour de la NAV ──
        nav *= (1 + port_ret)
        nav_peak = max(nav_peak, nav)

        # ── 9. Mise à jour du suivi des positions ──
        # Entry prices : conserver l'ancien prix d'entrée si la position existait déjà
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
                if weights.get(ticker, 0) > 0:
                    new_peak[ticker] = max(old_peak, curr)
                elif weights.get(ticker, 0) < 0:
                    new_peak[ticker] = min(old_peak, curr)
        peak_prices = new_peak.dropna()

        # ── 10. Stocker les résultats ──
        pnl_list.append(port_ret)
        cumul_list.append(nav)
        turnover_list.append(to)
        cost_list.append(cost)
        n_long_list.append(len(long_list))
        n_short_list.append(len(short_list))
        dates_used.append(date)

        port_rets_so_far = pd.Series(pnl_list, index=dates_used)
        prev_weights = weights.copy()

    # ── Construction des Series de sortie ──
    pnl          = pd.Series(pnl_list,      index=dates_used, name="pnl")
    cumulative   = pd.Series(cumul_list,     index=dates_used, name="cumulative")
    turnover_log = pd.Series(turnover_list,  index=dates_used, name="turnover")
    cost_log     = pd.Series(cost_list,      index=dates_used, name="cost")

    n_long_log   = pd.Series(n_long_list,   index=dates_used, name="n_long")
    n_short_log  = pd.Series(n_short_list,  index=dates_used, name="n_short")

    return {
        "pnl":          pnl,
        "cumulative":   cumulative,
        "turnover_log": turnover_log,
        "cost_log":     cost_log,
        "n_long":       n_long_log,
        "n_short":      n_short_log,
    }


# ═══════════════════════════════════════════════════════════════════
#  Lancement de toutes les combinaisons
# ═══════════════════════════════════════════════════════════════════

def run_all_strategies(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    df_long: pd.DataFrame,
    signals: list,
    alloc_methods: list,
    stoploss_types: list | None = None,
    verbose: bool = False,
) -> dict:
    """
    Exécute run_backtest pour chaque combinaison
    signal × allocation × stop-loss.

    Returns
    -------
    dict[ (signal, alloc, stoploss) ] → résultat du backtest
    """
    if stoploss_types is None:
        stoploss_types = ["position"]

    all_results = {}
    total = len(signals) * len(alloc_methods) * len(stoploss_types)
    count = 0

    for sig in signals:
        for alloc in alloc_methods:
            for sl in stoploss_types:
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] {sig} | {alloc} | SL={sl}")

                result = run_backtest(
                    prices, returns, df_long,
                    signal_name=sig,
                    alloc_method=alloc,
                    stoploss_type=sl,
                    verbose=verbose,
                )
                all_results[(sig, alloc, sl)] = result

    return all_results
