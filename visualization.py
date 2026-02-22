"""
visualization.py — Graphiques et export des résultats.

Graphs :
  1. Equity curve (toutes stratégies + SPX)
  2. Drawdown
  3. Rolling Sharpe (12 mois)
  4. Histogramme des rendements mensuels
  5. Tableau de métriques
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (save only)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from config import OUTPUT_DIR


# Style global
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


# ═══════════════════════════════════════════════════════════════════
#  1.  Equity Curve
# ═══════════════════════════════════════════════════════════════════

def plot_equity_curves(all_results: dict,
                       rf_monthly: pd.Series | None = None,
                       title: str = "Equity Curves -- L/S Market Neutral",
                       save: bool = True):
    """
    Trace les equity curves de toutes les strategies
    + US Treasury Bill 3M (benchmark risk-free) cumule.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, (key, res) in enumerate(all_results.items()):
        label = f"{key[0]} | {key[1]} | SL={key[2]}"
        cum = res["cumulative"]
        ax.plot(cum.index, cum.values, label=label,
                color=COLORS[idx % len(COLORS)], linewidth=1.5)

    # Benchmark = Risk-free cumule (US Treasury Bill 3M)
    if rf_monthly is not None and not rf_monthly.empty:
        rf_cum = (1 + rf_monthly).cumprod()
        # Normaliser a 1 au debut de la periode des strategies
        first_date = list(all_results.values())[0]["cumulative"].index[0]
        rf_cum_filtered = rf_cum[rf_cum.index >= first_date]
        if not rf_cum_filtered.empty:
            rf_cum_filtered = rf_cum_filtered / rf_cum_filtered.iloc[0]
            ax.plot(rf_cum_filtered.index, rf_cum_filtered.values,
                    label="US T-Bill 3M (Rf benchmark)", color="black",
                    linewidth=1.8, linestyle="--", alpha=0.7)

    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur (base 1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "equity_curves.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  2.  Drawdown
# ═══════════════════════════════════════════════════════════════════

def plot_drawdowns(all_results: dict,
                   title: str = "Drawdowns",
                   save: bool = True):
    """Trace les drawdowns de toutes les stratégies."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, (key, res) in enumerate(all_results.items()):
        label = f"{key[0]} | {key[1]} | SL={key[2]}"
        cum = res["cumulative"]
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.3, color=COLORS[idx % len(COLORS)])
        ax.plot(dd.index, dd.values, label=label,
                color=COLORS[idx % len(COLORS)], linewidth=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "drawdowns.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  3.  Rolling Sharpe (12 mois)
# ═══════════════════════════════════════════════════════════════════

def plot_rolling_sharpe(all_results: dict,
                        window: int = 12,
                        title: str = "Rolling Sharpe Ratio (12m)",
                        save: bool = True):
    """Sharpe glissant sur fenêtre de `window` mois."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, (key, res) in enumerate(all_results.items()):
        label = f"{key[0]} | {key[1]} | SL={key[2]}"
        pnl = res["pnl"]
        rolling_mean = pnl.rolling(window).mean() * 12
        rolling_std  = pnl.rolling(window).std() * np.sqrt(12)
        rolling_sr = rolling_mean / rolling_std
        ax.plot(rolling_sr.index, rolling_sr.values, label=label,
                color=COLORS[idx % len(COLORS)], linewidth=1.2)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "rolling_sharpe.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  4.  Histogramme des rendements
# ═══════════════════════════════════════════════════════════════════

def plot_return_histograms(all_results: dict,
                           title: str = "Distribution des rendements mensuels",
                           save: bool = True):
    """Histogrammes superposés des rendements mensuels."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (key, res) in enumerate(all_results.items()):
        label = f"{key[0]} | {key[1]}"
        pnl = res["pnl"].dropna()
        axes[idx].hist(pnl, bins=30, color=COLORS[idx % len(COLORS)],
                       alpha=0.7, edgecolor="white")
        axes[idx].axvline(pnl.mean(), color="red", linestyle="--",
                          linewidth=1, label=f"Moy={pnl.mean():.2%}")
        axes[idx].set_title(label, fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "return_histograms.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  5.  Tableau de métriques (affiché proprement)
# ═══════════════════════════════════════════════════════════════════

def display_metrics_table(summary_df: pd.DataFrame):
    """
    Affiche le tableau de métriques dans la console.
    (L'export Excel est fait dans backtest_results.xlsx via export_all_results)
    """
    print("\n" + "=" * 80)
    print("  TABLEAU COMPARATIF DES STRATÉGIES")
    print("=" * 80)
    print(summary_df.to_string())


# ═══════════════════════════════════════════════════════════════════
#  6.  Export complet
# ═══════════════════════════════════════════════════════════════════

def export_all_results(all_results: dict, summary_df: pd.DataFrame,
                       save: bool = True):
    """Exporte tous les résultats dans un seul fichier Excel multi-feuilles."""
    if not save:
        return

    out_path = OUTPUT_DIR / "backtest_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Métriques
        summary_df.to_excel(writer, sheet_name="Metrics")

        # PnL et cumulatif pour chaque stratégie
        for key, res in all_results.items():
            sheet = f"{key[0]}_{key[1]}"[:31]  # max 31 chars pour Excel
            df = pd.DataFrame({
                "PnL":        res["pnl"],
                "Cumulative": res["cumulative"],
            })
            if not res["turnover_log"].empty:
                df["Turnover"] = res["turnover_log"]
                df["Cost"]     = res["cost_log"]
            df.to_excel(writer, sheet_name=sheet)

    print(f"\n✅ Résultats exportés dans {out_path}")
