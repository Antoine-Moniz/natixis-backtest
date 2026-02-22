"""
main.py — Point d'entrée principal du backtest L/S market-neutral.

Usage :
    python main.py

Exécute toutes les combinaisons (signal × allocation × stoploss),
calcule les métriques, génère les graphiques et exporte les résultats.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all
from backtest import run_backtest, run_all_strategies
from metrics import summary_table, compute_all_metrics, set_rf_series
from visualization import (
    plot_equity_curves, plot_drawdowns, plot_rolling_sharpe,
    plot_return_histograms, display_metrics_table, export_all_results,
    plot_monthly_heatmaps, plot_turnover, plot_correlation_matrix,
    plot_annual_returns, plot_long_short_contribution,
    plot_qq, plot_exposure, plot_underwater,
)
from config import SIGNALS, ALLOCATION_METHODS, START_DATE, END_DATE


def main():
    print("=" * 60)
    print("  BACKTEST L/S MARKET-NEUTRAL — S&P 500")
    print("=" * 60)

    # ── 1. Chargement des données ──
    print("\n[1/4] Chargement des données...")
    data = load_all()

    prices     = data["prices"]
    returns    = data["returns"]
    df_long    = data["df_long"]
    rf_m       = data["rf_monthly"]

    # Initialiser le taux sans risque dans metrics
    set_rf_series(rf_m)

    print(f"  Prix     : {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(f"  Periode  : {prices.index[0].strftime('%Y-%m')} -> "
          f"{prices.index[-1].strftime('%Y-%m')}")
    print(f"  Rf pts   : {len(rf_m)}")
    if not rf_m.empty:
        print(f"  Rf moyen : {rf_m.mean() * 12:.2%} annualise")

    # ── 2. Lancement des backtests ──
    print("\n[2/4] Lancement des backtests...")

    # On teste : 3 signaux × 2 allocations × 1 stop-loss (position)
    # Pour tester les 4 stop-loss, décommentez la ligne ci-dessous :
    # stoploss_types = ["position", "trailing", "portfolio", "volatility"]
    stoploss_types = ["position"]

    all_results = run_all_strategies(
        prices, returns, df_long,
        signals=SIGNALS,
        alloc_methods=ALLOCATION_METHODS,
        stoploss_types=stoploss_types,
        verbose=True,
    )

    # ── 3. Métriques ──
    print("\n[3/4] Calcul des métriques...")
    summary_df = summary_table(all_results)
    display_metrics_table(summary_df)

    # ── 4. Graphiques + export ──
    print("\n[4/4] Graphiques et export...")

    # Filtrer Rf sur la meme periode
    rf_filtered = rf_m[
        (rf_m.index >= START_DATE) & (rf_m.index <= END_DATE)
    ]

    plot_equity_curves(all_results, rf_monthly=rf_filtered)
    plot_drawdowns(all_results)
    plot_rolling_sharpe(all_results)
    plot_return_histograms(all_results)

    # Nouveaux graphiques
    plot_monthly_heatmaps(all_results)
    plot_turnover(all_results)
    plot_correlation_matrix(all_results)
    plot_annual_returns(all_results)
    plot_long_short_contribution(all_results)

    # Graphiques avances
    spx_m = data["spx_monthly"]
    spx_ret = spx_m.pct_change().dropna()
    plot_qq(all_results)
    plot_exposure(all_results)
    plot_underwater(all_results)

    export_all_results(all_results, summary_df)

    print("\n" + "=" * 60)
    print("  BACKTEST TERMINÉ !")
    print("=" * 60)


if __name__ == "__main__":
    main()
