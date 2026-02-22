"""
data_loader.py — Chargement et nettoyage des données depuis le fichier Excel Bloomberg.

Structure Excel attendue :
  - SPX_Components        : date | ticker | PX_LAST
  - SPX_PX_LAST_Monthly   : date | spx_px_last
  - SPX_PX_LAST_Daily     : date | spx_px_last
  - TAUX_SANS_RISQUE      : date | US Treasury Bill 3M
"""

import pandas as pd
import numpy as np
from config import DATA_FILE, MIN_HISTORY_MONTHS


# ═══════════════════════════════════════════════════════════════════
#  1.  Chargement brut
# ═══════════════════════════════════════════════════════════════════

def load_raw_components(path=DATA_FILE) -> pd.DataFrame:
    """Charge la feuille SPX_Components (format long : date | ticker | PX_LAST)."""
    df = pd.read_excel(path, sheet_name="SPX_Components")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["px_last"] = pd.to_numeric(df["px_last"], errors="coerce")
    df = df.dropna(subset=["px_last"])
    return df


def load_spx_monthly(path=DATA_FILE) -> pd.Series:
    """Charge SPX PX_LAST mensuel → Series indexée par date."""
    df = pd.read_excel(path, sheet_name="SPX_PX_LAST_Monthly")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    return df["spx_px_last"].dropna()


def load_spx_daily(path=DATA_FILE) -> pd.Series:
    """Charge SPX PX_LAST daily -> Series indexee par date."""
    df = pd.read_excel(path, sheet_name="SPX_PX_LAST_Daily")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    return df["spx_px_last"].dropna()


def load_risk_free(path=DATA_FILE) -> pd.Series:
    """
    Charge le taux sans risque (US Treasury Bill 3M) depuis la feuille
    TAUX_SANS_RISQUE -> Series indexee par date.
    Colonne A = Benchmark (dates),  Colonne B = US Treasury Bill 3M (% annuel).
    On le convertit en taux mensuel : rf_monthly = rf_annual / 100 / 12
    """
    df = pd.read_excel(path, sheet_name="TAUX_SANS_RISQUE")
    # 1re colonne = dates (nommee "Benchmark"), 2e = taux
    date_col = df.columns[0]
    rf_col   = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col).sort_index()

    rf = pd.to_numeric(df[rf_col], errors="coerce").dropna()

    # Convertir en taux mensuel (diviser par 100 car en %, puis /12)
    rf_monthly = rf / 100.0 / 12.0
    rf_monthly.name = "rf_monthly"
    return rf_monthly


# ═══════════════════════════════════════════════════════════════════
#  2.  Pivot : format long → matrice (dates × tickers)
# ═══════════════════════════════════════════════════════════════════

def pivot_prices(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit le format long (date | ticker | PX_LAST)
    en matrice prix :  index=date, columns=ticker, values=PX_LAST.
    """
    piv = df_long.pivot_table(
        index="date", columns="ticker", values="px_last", aggfunc="last"
    )
    piv = piv.sort_index()
    return piv


# ═══════════════════════════════════════════════════════════════════
#  3.  Nettoyage / filtrage de l'univers
# ═══════════════════════════════════════════════════════════════════

def filter_universe(prices: pd.DataFrame,
                    min_months: int = MIN_HISTORY_MONTHS,
                    max_nan_pct: float = 0.30) -> pd.DataFrame:
    """
    Filtre les tickers :
      - au moins `min_months` observations non-NaN
      - pas plus de `max_nan_pct` de NaN sur toute la période
    """
    non_nan_count = prices.notna().sum()
    total = len(prices)
    keep = non_nan_count[
        (non_nan_count >= min_months) & (non_nan_count / total >= (1 - max_nan_pct))
    ].index
    return prices[keep]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples mensuels (log-returns optionnel ci-dessous)."""
    return prices.pct_change()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-returns mensuels."""
    return np.log(prices / prices.shift(1))


# ═══════════════════════════════════════════════════════════════════
#  4.  Composition de l'indice à chaque date de rebal
# ═══════════════════════════════════════════════════════════════════

def get_members_at_date(df_long: pd.DataFrame, date) -> list:
    """
    Retourne la liste des tickers qui étaient dans le SPX à une date donnée.
    (utilise la feuille SPX_Components en format long)
    """
    mask = df_long["date"] == pd.Timestamp(date)
    return df_long.loc[mask, "ticker"].dropna().unique().tolist()


# ═══════════════════════════════════════════════════════════════════
#  5.  Interface principale
# ═══════════════════════════════════════════════════════════════════

def load_all(path=DATA_FILE):
    """
    Point d'entrée unique : retourne un dict avec toutes les données
    nettoyées et prêtes pour le backtest.

    Returns
    -------
    data : dict
        "df_long"       : DataFrame format long (date, ticker, px_last)
        "prices"        : DataFrame matrice (date × ticker) PX_LAST — filtré
        "returns"       : DataFrame matrice rendements mensuels
        "spx_monthly"   : Series SPX mensuel
        "spx_daily"     : Series SPX daily
        "rebal_dates"   : list de dates de rebalancement
    """
    # Chargement
    df_long = load_raw_components(path)
    spx_m = load_spx_monthly(path)
    try:
        spx_d = load_spx_daily(path)
    except Exception:
        spx_d = pd.Series(dtype=float)

    # Taux sans risque
    try:
        rf_m = load_risk_free(path)
    except Exception:
        print("  [!] Feuille TAUX_SANS_RISQUE non trouvee -> Rf = 0")
        rf_m = pd.Series(dtype=float)

    # Pivot + filtre
    prices = pivot_prices(df_long)
    prices = filter_universe(prices)

    # Returns
    rets = compute_returns(prices)

    # Dates de rebal = toutes les dates de la matrice
    rebal_dates = prices.index.tolist()

    return {
        "df_long":     df_long,
        "prices":      prices,
        "returns":     rets,
        "spx_monthly": spx_m,
        "spx_daily":   spx_d,
        "rf_monthly":  rf_m,
        "rebal_dates": rebal_dates,
    }


if __name__ == "__main__":
    data = load_all()
    print("Prix shape       :", data["prices"].shape)
    print("Returns shape    :", data["returns"].shape)
    print("Nb rebal dates   :", len(data["rebal_dates"]))
    print("SPX monthly pts  :", len(data["spx_monthly"]))
    print("SPX daily pts    :", len(data["spx_daily"]))
    print("Rf monthly pts   :", len(data["rf_monthly"]))
    print("Exemple tickers  :", list(data["prices"].columns[:5]))
