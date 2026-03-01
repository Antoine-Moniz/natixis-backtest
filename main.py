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
    plot_current_portfolio_sectors, plot_sector_allocation_over_time,
    plot_current_portfolio_weights, plot_current_portfolio_pie,
)
from config import START_DATE, END_DATE, N_STOCKS, OUTPUT_DIR, ESG_EXCLUSIONS, ESG_EXCLUDED_TICKERS


def print_current_portfolio(result: dict):
    """Affiche les 20 titres hybrides et leurs poids pour la dernière période."""
    last_w = result["last_weights"]
    last_date = result["last_date"]

    if last_w.empty or last_date is None:
        print("\n  [!] Aucun portefeuille à afficher.")
        return

    # Normaliser les poids pour qu'ils totalisent exactement 1.0
    last_w = last_w / last_w.sum()
    
    # Trier par poids décroissant
    last_w = last_w.sort_values(ascending=False)

    print("\n" + "=" * 60)
    print(f"  PORTEFEUILLE ACTUEL — {last_date.strftime('%Y-%m-%d')}")
    print(f"  Méthode : Stratégie Hybride VALUE/SIZE + COMPOSITE (ESG)")
    print(f"  Allocation : 30% VALUE/SIZE (5 actions) + 70% COMPOSITE (15 actions)")
    print(f"  Nombre total de titres : {len(last_w)}")
    print("=" * 60)
    print(f"  {'#':<4} {'Ticker':<12} {'Poids':>10}")
    print("  " + "-" * 28)

    for i, (ticker, weight) in enumerate(last_w.items(), 1):
        print(f"  {i:<4} {ticker:<12} {weight:>11.4%}")

    print("  " + "-" * 30)
    print(f"  {'TOTAL':<16} {last_w.sum():>11.4%}")
    print()

    # Export Excel avec poids normalisés et format numérique
    xlsx_path = OUTPUT_DIR / "current_portfolio.xlsx"
    df_export = pd.DataFrame({
        "Ticker": last_w.index,
        "Weight": last_w.values,  # Valeurs exactes sans arrondi
    })
    # Noter que la somme est exactement 1.0
    assert abs(df_export["Weight"].sum() - 1.0) < 1e-10, f"Poids non normalisés: {df_export['Weight'].sum()}"
    
    # Exporter vers Excel avec formatage
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Portfolio', index=False, float_format='%.8f')
        print(f"  [OK] Portefeuille exporté dans {xlsx_path}")
        print(f"  [✓] Somme des poids vérifiée: {df_export['Weight'].sum():.10f}")
    except ImportError:
        # Fallback vers CSV si openpyxl n'est pas disponible
        csv_path = OUTPUT_DIR / "current_portfolio.csv"
        df_export.to_csv(csv_path, index=False, float_format='%.8f')
        print(f"  [OK] Portefeuille exporté dans {csv_path} (Excel non disponible)")
        print(f"  [✓] Somme des poids vérifiée: {df_export['Weight'].sum():.10f}")


def main():
    print("=" * 60)
    print("  BACKTEST HYBRIDE VALUE/SIZE + COMPOSITE — S&P 500 (filtre ESG)")
    print(f"  Portefeuille Hybride : 30% VALUE/SIZE (5 actions) + 70% COMPOSITE (15 actions)")
    print("=" * 60)

    # ── 1. Chargement des données ──
    print("\n[1/4] Chargement des données...")
    data = load_all()

    prices      = data["prices"]
    returns     = data["returns"]
    df_long     = data["df_long"]
    rf_annual   = data["rf_annual"]
    df_sectors  = data["df_sectors"]
    fundamentals = data["fundamentals"]

    # Initialiser le taux sans risque dans metrics
    set_rf_series(rf_annual)

    # ── Filtre ESG : exclusion des titres controversés ──
    n_before = prices.shape[1]
    excluded_found = [t for t in ESG_EXCLUDED_TICKERS if t in prices.columns]
    cols_keep = [c for c in prices.columns if c not in ESG_EXCLUDED_TICKERS]

    prices  = prices[cols_keep]
    returns = returns[cols_keep]
    df_long = df_long[~df_long["ticker"].isin(ESG_EXCLUDED_TICKERS)].copy()

    print(f"\n  Filtre ESG appliqué :")
    print(f"    Univers avant  : {n_before} titres")
    print(f"    Titres exclus  : {len(excluded_found)}")
    print(f"    Univers après  : {prices.shape[1]} titres")
    for cat, tickers in ESG_EXCLUSIONS.items():
        found = [t.split()[0] for t in tickers if t in excluded_found]
        if found:
            print(f"      {cat} ({len(found)}) : {', '.join(found)}")

    print(f"\n  Prix     : {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(f"  Periode  : {prices.index[0].strftime('%Y-%m')} -> "
          f"{prices.index[-1].strftime('%Y-%m')}")
    print(f"  Rf pts   : {len(rf_annual)}")
    if not rf_annual.empty:
        print(f"  Rf moyen : {rf_annual.mean():.2%} annualise")

    # ── 2. Lancement du backtest ──
    print("\n[2/4] Lancement du backtest Hybride VALUE/SIZE + COMPOSITE (ESG)...")

    result = run_backtest(
        prices, returns, df_long,
        fundamentals=fundamentals,
        use_hybrid_strategy=True,
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
    
    # ── Graphiques du portefeuille ──
    plot_current_portfolio_weights(result)
    plot_current_portfolio_pie(result)

    # ── Analyse sectorielle ──
    print("\n[3.5/4] Analyse sectorielle...")
    plot_current_portfolio_sectors(result, df_sectors)
    plot_sector_allocation_over_time(all_results, df_sectors)

    # ── 4. Graphiques + export ──
    print("\n[4/4] Graphiques et export...")

    # Filtrer SOFR+4% et Rf sur la même période
    rf_filtered = rf_annual[
        (rf_annual.index >= START_DATE) & (rf_annual.index <= END_DATE)
    ]
    
    sofr_plus_4_filtered = data["sofr_plus_4_m"][
        (data["sofr_plus_4_m"].index >= START_DATE) & 
        (data["sofr_plus_4_m"].index <= END_DATE)
    ]

    spx_m = data["spx_monthly"]
    plot_equity_curves(all_results, rf_monthly=rf_filtered, 
                       sofr_plus_4_m=sofr_plus_4_filtered, spx_monthly=spx_m)
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
