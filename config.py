"""
config.py — Paramètres globaux du backtest L/S market-neutral
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

# ──────────────────────────── Signaux ────────────────────────────
SIGNALS = ["momentum", "mean_reversion", "vol_spread"]
QUANTILE_LONG  = 0.20       # top 20 % → long
QUANTILE_SHORT = 0.20       # bottom 20 % → short

# ──────────────────────────── Allocation ─────────────────────────
ALLOCATION_METHODS = ["equal_weight", "erc"]

# ──────────────────────────── Coûts ──────────────────────────────
TRANSACTION_COST_BPS = 10   # 10 bps par transaction

# ──────────────────────────── Stop-loss ──────────────────────────
STOPLOSS_POSITION    = -0.10   # -10 % depuis entrée
STOPLOSS_TRAILING    = -0.08   # -8 % trailing
STOPLOSS_PORTFOLIO   = -0.15   # drawdown max portefeuille -15 %
STOPLOSS_VOL_WINDOW  = 60     # fenêtre vol (jours) pour stop vol-based
STOPLOSS_VOL_MULT    = 2.0    # multiplicateur vol

# ──────────────────────────── Rebalancement ──────────────────────
REBAL_FREQ = "M"            # mensuel (fin de mois)

# ──────────────────────────── Risk-free ──────────────────────────
# RF_ANNUAL n'est plus utilise : le taux sans risque est charge
# dynamiquement depuis la feuille TAUX_SANS_RISQUE (US Treasury Bill 3M)
RF_ANNUAL = 0.0             # fallback si la feuille Excel n'est pas disponible
