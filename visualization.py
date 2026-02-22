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

# ═══════════════════════════════════════════════════════════════════
#  7.  Heatmap rendements mensuels (calendrier)
# ═══════════════════════════════════════════════════════════════════

def plot_monthly_heatmaps(all_results: dict, save: bool = True):
    """Heatmap annee x mois des rendements pour chaque strategie."""
    keys = list(all_results.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    for idx, key in enumerate(keys):
        ax = axes[idx, 0]
        rets = all_results[key]["pnl"].copy()
        rets.index = pd.to_datetime(rets.index)

        df_heat = pd.DataFrame({
            "year": rets.index.year,
            "month": rets.index.month,
            "ret": rets.values
        })
        pivot = df_heat.pivot_table(index="year", columns="month",
                                     values="ret", aggfunc="sum")
        pivot.columns = [month_labels[int(c)-1] for c in pivot.columns]

        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1e-9)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                        vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                            fontsize=7,
                            color="black" if abs(val) < vmax * 0.6 else "white")

        label = f"{key[0]} | {key[1]} | SL={key[2]}"
        ax.set_title(label, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, format="%.0%%", shrink=0.8)

    fig.suptitle("Heatmap des Rendements Mensuels",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "monthly_heatmaps.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  8.  Turnover mensuel moyen
# ═══════════════════════════════════════════════════════════════════

def plot_turnover(all_results: dict, save: bool = True):
    """Bar chart du turnover mensuel moyen par strategie."""
    labels, turnovers, colors_list = [], [], []
    cmap = plt.cm.Set2

    for idx, (key, res) in enumerate(all_results.items()):
        labels.append(f"{key[0]}\n{key[1]}")
        turnovers.append(res["turnover_log"].mean())
        colors_list.append(cmap(idx / max(1, len(all_results) - 1)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), turnovers,
                  color=colors_list, edgecolor="grey", width=0.6)

    for bar, val in zip(bars, turnovers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Turnover mensuel moyen")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Turnover Mensuel Moyen par Strategie",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "turnover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  9.  Matrice de correlation des strategies
# ═══════════════════════════════════════════════════════════════════

def plot_correlation_matrix(all_results: dict, save: bool = True):
    """Heatmap de correlation entre les rendements des strategies."""
    ret_dict = {}
    for key, res in all_results.items():
        ret_dict[f"{key[0]} | {key[1]}"] = res["pnl"]

    df_rets = pd.DataFrame(ret_dict).dropna()
    corr = df_rets.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")

    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matrice de Correlation des Strategies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "correlation_matrix.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  10.  Rendement annuel par strategie
# ═══════════════════════════════════════════════════════════════════

def plot_annual_returns(all_results: dict, save: bool = True):
    """Bar chart groupe : rendement annuel par strategie."""
    annual_dict = {}
    for key, res in all_results.items():
        label = f"{key[0]} | {key[1]}"
        rets = res["pnl"].copy()
        rets.index = pd.to_datetime(rets.index)
        annual = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual_dict[label] = annual

    df_annual = pd.DataFrame(annual_dict)
    years = df_annual.index.tolist()
    n_strats = len(df_annual.columns)
    width = 0.8 / n_strats

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap_t = plt.cm.tab10

    for i, col in enumerate(df_annual.columns):
        positions = [y + (i - n_strats / 2) * width for y in range(len(years))]
        ax.bar(positions, df_annual[col].values,
               width=width, label=col,
               color=cmap_t(i / max(1, n_strats - 1)),
               edgecolor="grey", linewidth=0.5)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Rendement Annuel par Strategie",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Rendement annuel")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "annual_returns.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  11.  Contribution Long vs Short
# ═══════════════════════════════════════════════════════════════════

def plot_long_short_contribution(all_results: dict, save: bool = True):
    """
    Rendement cumule du leg Long vs leg Short pour chaque strategie.
    Necessite que le backtest retourne 'ret_long_series' et 'ret_short_series'.
    """
    has_data = any("ret_long_series" in res for res in all_results.values())
    if not has_data:
        print("  (i) Long/Short contribution : donnees non disponibles")
        return

    keys = list(all_results.keys())
    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // cols, idx % cols]
        res = all_results[key]

        if "ret_long_series" in res and "ret_short_series" in res:
            cum_long = (1 + res["ret_long_series"]).cumprod()
            cum_short = (1 + res["ret_short_series"]).cumprod()
            cum_total = (1 + res["pnl"]).cumprod()
            ax.plot(cum_long.index, cum_long, label="Long leg", color="green", lw=1.2)
            ax.plot(cum_short.index, cum_short, label="Short leg", color="red", lw=1.2)
            ax.plot(cum_total.index, cum_total, label="L/S Net", color="blue",
                    lw=1.5, linestyle="--")
            ax.legend(fontsize=7)

        ax.set_title(f"{key[0]} | {key[1]}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Contribution Long vs Short (cumule)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "long_short_contribution.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  12.  QQ-Plot des rendements
# ═══════════════════════════════════════════════════════════════════

def plot_qq(all_results: dict, save: bool = True):
    """QQ-plot vs distribution normale pour chaque strategie."""
    import scipy.stats as stats

    keys = list(all_results.keys())
    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // cols, idx % cols]
        rets = all_results[key]["pnl"].dropna().values

        (osm, osr), (slope, intercept, r) = stats.probplot(rets, dist="norm")
        ax.scatter(osm, osr, s=15, alpha=0.6, edgecolor="none",
                   color=COLORS[idx % len(COLORS)])
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.5,
                label=f"R2 = {r**2:.3f}")

        label = f"{key[0]} | {key[1]}"
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Quantiles theoriques (Normal)")
        ax.set_ylabel("Quantiles observes")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("QQ-Plot vs Distribution Normale",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "qq_plots.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  14.  Exposure chart (nb actions long/short + exposition nette)
# ═══════════════════════════════════════════════════════════════════

def plot_exposure(all_results: dict, save: bool = True):
    """
    Nombre d'actions long/short par mois + exposition nette.
    Necessite 'n_long' et 'n_short' dans les resultats du backtest.
    """
    has_data = any("n_long" in res for res in all_results.values())
    if not has_data:
        print("  (i) Exposure chart : donnees n_long/n_short non disponibles")
        return

    keys = list(all_results.keys())
    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // cols, idx % cols]
        res = all_results[key]

        n_long = res["n_long"]
        n_short = res["n_short"]
        net = n_long - n_short

        ax.bar(range(len(n_long)), n_long.values, color="green",
               alpha=0.6, label="Long", width=0.8)
        ax.bar(range(len(n_short)), -n_short.values, color="red",
               alpha=0.6, label="Short", width=0.8)
        ax.plot(range(len(net)), net.values, color="blue",
                linewidth=1.5, label="Net")
        ax.axhline(0, color="black", linewidth=0.5)

        # X labels (annees seulement)
        dates = pd.to_datetime(n_long.index)
        year_labels = [d.strftime("%Y") if d.month == 1 else ""
                       for d in dates]
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(year_labels, fontsize=7, rotation=0)

        ax.set_title(f"{key[0]} | {key[1]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("Nb actions")
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Exposure : Nombre d'actions Long / Short par mois",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "exposure.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  15.  Underwater plot (drawdown avec duree annotee)
# ═══════════════════════════════════════════════════════════════════

def plot_underwater(all_results: dict, save: bool = True):
    """
    Underwater plot : drawdown avec annotation de la duree
    de chaque drawdown majeur (temps pour recuperer le pic).
    """
    keys = list(all_results.keys())
    n = len(keys)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // cols, idx % cols]
        cum = all_results[key]["cumulative"]
        cum.index = pd.to_datetime(cum.index)
        peak = cum.cummax()
        dd = (cum - peak) / peak

        ax.fill_between(dd.index, dd.values, 0,
                        color=COLORS[idx % len(COLORS)], alpha=0.4)
        ax.plot(dd.index, dd.values,
                color=COLORS[idx % len(COLORS)], linewidth=1)

        # Annoter les 3 pires drawdowns avec leur duree
        # Trouver les periodes de drawdown
        in_dd = dd < 0
        dd_periods = []
        start = None
        for i, (dt, val) in enumerate(dd.items()):
            if val < 0 and start is None:
                start = dt
            elif val >= 0 and start is not None:
                # Fin du drawdown
                dd_slice = dd.loc[start:dt]
                trough_date = dd_slice.idxmin()
                trough_val = dd_slice.min()
                duration = (dt - start).days
                dd_periods.append((start, dt, trough_date, trough_val, duration))
                start = None
        # Si on est encore en drawdown a la fin
        if start is not None:
            dd_slice = dd.loc[start:]
            trough_date = dd_slice.idxmin()
            trough_val = dd_slice.min()
            duration = (dd.index[-1] - start).days
            dd_periods.append((start, dd.index[-1], trough_date, trough_val, duration))

        # Trier par profondeur et annoter les 3 pires
        dd_periods.sort(key=lambda x: x[3])
        for j, (s, e, trough, depth, dur) in enumerate(dd_periods[:3]):
            months = dur // 30
            ax.annotate(
                f"{depth:.1%}\n{months}m",
                xy=(trough, depth),
                xytext=(0, -20), textcoords="offset points",
                fontsize=7, fontweight="bold", color="darkred",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
            )

        ax.set_title(f"{key[0]} | {key[1]} | SL={key[2]}",
                     fontsize=9, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Underwater Plot (Drawdown + Duree de recuperation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "underwater.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Export complet
# ═══════════════════════════════════════════════════════════════════

def export_all_results(all_results: dict, summary_df: pd.DataFrame,
                       save: bool = True):
    """Exporte tous les resultats dans un seul fichier Excel multi-feuilles."""
    if not save:
        return

    out_path = OUTPUT_DIR / "backtest_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Metriques
        summary_df.to_excel(writer, sheet_name="Metrics")

        # PnL et cumulatif pour chaque strategie
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

    print(f"\n[OK] Resultats exportes dans {out_path}")
