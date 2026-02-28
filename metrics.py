"""
metrics.py — Indicateurs de performance du portefeuille.

Indicateurs :
  1. Sharpe Ratio
  2. Sortino Ratio
  3. Volatilité annualisée
  4. Max Drawdown
  5. Return total / CAGR
  6. VaR / CVaR historique
  7. Turnover moyen + coûts cumulés
"""

import pandas as pd
import numpy as np

# ── Taux sans risque reel (serie mensuelle injectee par main.py) ──
_RF_MONTHLY: pd.Series = pd.Series(dtype=float)


def set_rf_series(rf_monthly: pd.Series) -> None:
    """Stocke la serie mensuelle du taux sans risque."""
    global _RF_MONTHLY
    _RF_MONTHLY = rf_monthly


def get_rf_annual_mean() -> float:
    """Moyenne annualisee du taux sans risque sur la periode."""
    if _RF_MONTHLY.empty:
        return 0.02          # fallback 2 %
    return _RF_MONTHLY.mean()   # deja en taux annualise dans data_loader


# ═══════════════════════════════════════════════════════════════════
#  Indicateurs individuels
# ═══════════════════════════════════════════════════════════════════

def annualized_return(pnl: pd.Series, periods_per_year: int = 12) -> float:
    """CAGR annualisé à partir de rendements périodiques."""
    cum = (1 + pnl).prod()
    n_years = len(pnl) / periods_per_year
    if n_years <= 0 or cum <= 0:
        return 0.0
    return cum ** (1 / n_years) - 1


def annualized_vol(pnl: pd.Series, periods_per_year: int = 12) -> float:
    """Volatilité annualisée."""
    return pnl.std() * np.sqrt(periods_per_year)


def sharpe_ratio(pnl: pd.Series, rf: float | None = None,
                 periods_per_year: int = 12) -> float:
    """Sharpe Ratio annualise (utilise le Rf reel du Treasury Bill 3M)."""
    if rf is None:
        rf = get_rf_annual_mean()
    ret = annualized_return(pnl, periods_per_year)
    vol = annualized_vol(pnl, periods_per_year)
    if vol == 0:
        return 0.0
    return (ret - rf) / vol


def sortino_ratio(pnl: pd.Series, rf: float | None = None,
                  periods_per_year: int = 12) -> float:
    """Sortino Ratio (downside deviation, utilise le Rf reel)."""
    if rf is None:
        rf = get_rf_annual_mean()
    ret = annualized_return(pnl, periods_per_year)
    downside = pnl[pnl < 0]
    if downside.empty:
        return np.inf
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    return (ret - rf) / downside_vol


def max_drawdown(cumulative: pd.Series) -> float:
    """Max Drawdown (valeur négative)."""
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return dd.min()


def var_historical(pnl: pd.Series, confidence: float = 0.95) -> float:
    """VaR historique au seuil de confiance donné."""
    return pnl.quantile(1 - confidence)


def cvar_historical(pnl: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) historique."""
    var = var_historical(pnl, confidence)
    return pnl[pnl <= var].mean()


def total_return(cumulative: pd.Series) -> float:
    """Return total en %."""
    if cumulative.empty:
        return 0.0
    return cumulative.iloc[-1] / cumulative.iloc[0] - 1


def hit_ratio(pnl: pd.Series) -> float:
    """Proportion de mois positifs."""
    if pnl.empty:
        return 0.0
    return (pnl > 0).sum() / len(pnl)


# ═══════════════════════════════════════════════════════════════════
#  Tableau récapitulatif
# ═══════════════════════════════════════════════════════════════════

def compute_all_metrics(
    pnl: pd.Series,
    cumulative: pd.Series,
    turnover_log: pd.Series | None = None,
    cost_log: pd.Series | None = None,
) -> pd.Series:
    """
    Calcule tous les indicateurs et retourne un Series.
    """
    rf = get_rf_annual_mean()
    metrics = {
        "CAGR":             f"{annualized_return(pnl):.2%}",
        "Vol annualisee":   f"{annualized_vol(pnl):.2%}",
        "Sharpe":           f"{sharpe_ratio(pnl):.2f}",
        "Sortino":          f"{sortino_ratio(pnl):.2f}",
        "Rf moyen (ann.)":  f"{rf:.2%}",
        "Max Drawdown":     f"{max_drawdown(cumulative):.2%}",
        "Return total":     f"{total_return(cumulative):.2%}",
        "VaR 95%":          f"{var_historical(pnl):.2%}",
        "CVaR 95%":         f"{cvar_historical(pnl):.2%}",
        "Hit Ratio":        f"{hit_ratio(pnl):.2%}",
    }

    if turnover_log is not None and not turnover_log.empty:
        metrics["Turnover moyen"] = f"{turnover_log.mean():.2%}"

    if cost_log is not None and not cost_log.empty:
        metrics["Couts cumules"] = f"{cost_log.sum():.4%}"

    return pd.Series(metrics, name="Metrics")


def summary_table(all_results: dict) -> pd.DataFrame:
    """
    Crée un tableau comparatif de toutes les stratégies.

    Parameters
    ----------
    all_results : dict[(signal, alloc, stoploss)] → backtest results

    Returns
    -------
    DataFrame : lignes = métriques, colonnes = stratégies
    """
    rows = {}
    for key, res in all_results.items():
        label = f"{key[0]} | {key[1]} | SL={key[2]}"
        m = compute_all_metrics(
            res["pnl"], res["cumulative"],
            res.get("turnover_log"), res.get("cost_log"),
        )
        rows[label] = m

    return pd.DataFrame(rows)
