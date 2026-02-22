"""
main.py — Point d'entrée du backtest Long-Only ERC — S&P 500.

Usage :
    python main.py

Sélectionne les top 25 titres par score composite (momentum + vol),
alloue en ERC, exécute le backtest et affiche les poids actuels.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from data_loader import load_all
from backtest import run_backtest
from metrics import summary_table, compute_all_metrics, set_rf_series
from visualization import (
    plot_equity_curves, plot_drawdowns, plot_rolling_sharpe,
    plot_return_histograms, display_metrics_table, export_all_results,
    plot_monthly_heatmaps, plot_turnover, plot_correlation_matrix,
    plot_annual_returns,
    plot_qq, plot_underwater,
)
from config import START_DATE, END_DATE, N_STOCKS, OUTPUT_DIR


def print_current_portfolio(result: dict):
    """Affiche les 25 titres et leurs poids pour la dernière période."""
    last_w = result["last_weights"]
    last_date = result["last_date"]

    if last_w.empty or last_date is None:
        print("\n  [!] Aucun portefeuille à afficher.")
        return

    # Trier par poids décroissant
    last_w = last_w.sort_values(ascending=False)

    print("\n" + "=" * 60)
    print(f"  PORTEFEUILLE ACTUEL — {last_date.strftime('%Y-%m-%d')}")
    print(f"  Méthode : Score Composite + ERC Long-Only")
    print(f"  Nombre de titres : {len(last_w)}")
    print("=" * 60)
    print(f"  {'#':<4} {'Ticker':<12} {'Poids':>10}")
    print("  " + "-" * 28)

    for i, (ticker, weight) in enumerate(last_w.items(), 1):
        print(f"  {i:<4} {ticker:<12} {weight:>9.2%}")

    print("  " + "-" * 28)
    print(f"  {'TOTAL':<16} {last_w.sum():>9.2%}")
    print()

    # Export CSV
    csv_path = OUTPUT_DIR / "current_portfolio.csv"
    df_export = pd.DataFrame({
        "Ticker": last_w.index,
        "Weight": last_w.values,
        "Weight_pct": (last_w.values * 100).round(2),
    })
    df_export.to_csv(csv_path, index=False)
    print(f"  [OK] Portefeuille exporté dans {csv_path}")


def main():
    print("=" * 60)
    print("  BACKTEST LONG-ONLY ERC — S&P 500")
    print(f"  Top {N_STOCKS} titres par score composite")
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

    # ── 2. Lancement du backtest ──
    print("\n[2/4] Lancement du backtest Long-Only ERC...")

    result = run_backtest(
        prices, returns, df_long,
        stoploss_type="position",
        verbose=True,
    )

    # Mettre dans le format attendu par les fonctions de visualisation
    strategy_key = ("composite", "erc", "position")
    all_results = {strategy_key: result}

    # ── 3. Métriques ──
    print("\n[3/4] Calcul des métriques...")
    summary_df = summary_table(all_results)
    display_metrics_table(summary_df)

    # ── Affichage du portefeuille actuel (25 titres + poids) ──
    print_current_portfolio(result)

    # ── 4. Graphiques + export ──
    print("\n[4/4] Graphiques et export...")

    # Filtrer Rf sur la meme période
    rf_filtered = rf_m[
        (rf_m.index >= START_DATE) & (rf_m.index <= END_DATE)
    ]

    spx_m = data["spx_monthly"]
    plot_equity_curves(all_results, rf_monthly=rf_filtered, spx_monthly=spx_m)
    plot_drawdowns(all_results)
    plot_rolling_sharpe(all_results)
    plot_return_histograms(all_results)

    # Graphiques supplémentaires
    plot_monthly_heatmaps(all_results)
    plot_turnover(all_results)
    plot_annual_returns(all_results)

    # Graphiques avancés
    plot_qq(all_results)
    plot_underwater(all_results)

    export_all_results(all_results, summary_df)

    print("\n" + "=" * 60)
    print("  BACKTEST TERMINÉ !")
    print("=" * 60)


if __name__ == "__main__":
    main()
