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
N_STOCKS = 25               # nombre de titres à sélectionner
BUFFER_RANK = 40            # un titre existant n'est remplacé que s'il sort du top 40
# ──────────────────────────── Signaux (score composite) ──────────
# Poids de chaque facteur dans le score composite
SIGNAL_WEIGHTS = {
    "momentum":       0.35,   # momentum 12M-1M
    "mean_reversion": 0.25,   # vol mean reversion
    "vol_spread":     0.40,   # low volatility
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
REBAL_FREQ = "M"            # mensuel (fin de mois)

# ──────────────────────────── Risk-free ──────────────────────────
RF_ANNUAL = 0.0             # fallback si la feuille Excel n'est pas disponible
