"""
config.py — Paramètres globaux du backtest Long-Only ERC — S&P 500
"""
from pathlib import Path

# ──────────────────────────── Chemins ────────────────────────────
DATA_FILE = Path(__file__).parent / "SPX_universe_with_price_tes_date2.xlsx"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────── Dates ──────────────────────────────
START_DATE = "2016-01-01"   # on démarre après 12 mois pour le momentum
END_DATE   = "2025-12-31"

# ──────────────────────────── Univers ────────────────────────────
MIN_HISTORY_MONTHS = 12     # nb mois d'historique requis pour un ticker

# ──────────────────────────── Sélection ──────────────────────────
N_STOCKS = 20               # nombre de titres pour stratégie hybride (5 VALUE/SIZE + 15 COMPOSITE)
BUFFER_RANK = 25            # un titre existant n'est remplacé que s'il sort du top 25
# ──────────────────────────── Signaux (score composite) ──────────
# Poids de chaque facteur dans le score composite (momentum pur)
SIGNAL_WEIGHTS = {
    "momentum":       0.70,   # momentum 12M-1M (dominant absolu)
    "mean_reversion": 0.10,   # vol mean reversion (minimal)
    "vol_spread":     0.20,   # low volatility (minimal)
}

# ──────────────────────────── Allocation ─────────────────────────
ALLOCATION_METHOD = "erc"    # Equal Risk Contribution uniquement

# ──────────────────────────── Coûts ──────────────────────────────
TRANSACTION_COST_BPS = 5    # 5 bps par transaction

# ──────────────────────────── Stop-loss ──────────────────────────
STOPLOSS_POSITION    = -0.10   # -10 % depuis entrée
STOPLOSS_TRAILING    = -0.08   # -8 % trailing
STOPLOSS_PORTFOLIO   = -0.15   # drawdown max portefeuille -15 %
STOPLOSS_VOL_WINDOW  = 60     # fenêtre vol (jours) pour stop vol-based
STOPLOSS_VOL_MULT    = 2.0    # multiplicateur vol

# ──────────────────────────── Rebalancement ──────────────────────
REBAL_FREQ = "M"            # mensuel (plus de réactivité)

# ──────────────────────────── Risk-free ──────────────────────────
RF_ANNUAL = 0.0             # fallback si la feuille Excel n'est pas disponible

# ──────────────────────────── Exclusion ESG ──────────────────────
ESG_EXCLUSIONS = {
    "Tabac": [
        "MO UN Equity",       # Altria
        "PM UN Equity",       # Philip Morris
    ],
    "Armes / Défense": [
        "LMT UN Equity",      # Lockheed Martin
        "RTX UN Equity",      # Raytheon Technologies
        "NOC UN Equity",      # Northrop Grumman
        "GD UN Equity",       # General Dynamics
        "HII UN Equity",      # Huntington Ingalls Industries
        "LHX UN Equity",      # L3Harris Technologies
        "BA UN Equity",       # Boeing
    ],
    "Énergies fossiles": [
        "XOM UN Equity",      # Exxon Mobil
        "CVX UN Equity",      # Chevron
        "COP UN Equity",      # ConocoPhillips
        "EOG UN Equity",      # EOG Resources
        "OXY UN Equity",      # Occidental Petroleum
        "MPC UN Equity",      # Marathon Petroleum
        "VLO UN Equity",      # Valero Energy
        "PSX UN Equity",      # Phillips 66
        "DVN UN Equity",      # Devon Energy
        "HES UN Equity",      # Hess Corporation
        "HAL UN Equity",      # Halliburton
        "SLB UN Equity",      # Schlumberger
        "MRO UN Equity",      # Marathon Oil
        "CTRA UN Equity",     # Coterra Energy
        "KMI UN Equity",      # Kinder Morgan
        "WMB UN Equity",      # Williams Companies
        "OKE UN Equity",      # ONEOK
        "FCX UN Equity",      # Freeport-McMoRan
    ],
    "Jeux d'argent": [
        "WYNN UW Equity",     # Wynn Resorts
        "MGM UN Equity",      # MGM Resorts
    ],
    "Alcool": [
        "BF/B UN Equity",     # Brown-Forman
        "STZ UN Equity",      # Constellation Brands
        "TAP UN Equity",      # Molson Coors
    ],
}

# Set complet des tickers exclus (utilisé pour filtrer l'univers)
ESG_EXCLUDED_TICKERS = set()
for _cat, _tickers in ESG_EXCLUSIONS.items():
    ESG_EXCLUDED_TICKERS.update(_tickers)

