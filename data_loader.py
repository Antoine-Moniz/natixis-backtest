"""
data_loader.py - Chargement et nettoyage des donnees depuis le fichier Excel Bloomberg.

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
    """Charge SPX PX_LAST mensuel -> Series indexee par date."""
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
    On le garde en taux annuel pour le calcul du Sharpe ratio.
    """
    df = pd.read_excel(path, sheet_name="TAUX_SANS_RISQUE")
    # 1re colonne = dates (nommee "Benchmark"), 2e = taux
    date_col = df.columns[0]
    rf_col   = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col).sort_index()

    rf = pd.to_numeric(df[rf_col], errors="coerce").dropna()

    # Convertir en decimal (diviser par 100 car en %) mais garder en base annuelle
    rf_annual = rf / 100.0
    rf_annual.name = "rf_annual"
    return rf_annual


def load_sectors(path=DATA_FILE) -> pd.DataFrame:
    """
    Charge les secteurs depuis la feuille SPX_Sector.
    Colonne A = ticker, Colonne B = secteur.
    """
    df = pd.read_excel(path, sheet_name="SPX_Sector")
    # Nettoyer les noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Si les colonnes ne sont pas nommees correctement, utiliser les positions
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: 'ticker', df.columns[1]: 'sector'})
    
    # Nettoyer les donnees
    df['ticker'] = df['ticker'].astype(str).str.strip()
    df['sector'] = df['sector'].astype(str).str.strip()
    df = df.dropna(subset=['ticker', 'sector'])
    
    return df[['ticker', 'sector']]


def load_sofr(path=DATA_FILE) -> pd.Series:
    """
    Charge SOFRRATE depuis la feuille SOFRRATE.
    Colonne A = dates, Colonne B = SOFRRATE Index (% annuel).
    Retourne SOFR + 4% en taux mensuel : (sofr_annual + 4%) / 100 / 12
    """
    df = pd.read_excel(path, sheet_name="SOFRRATE")
    # 1re colonne = dates, 2e = SOFRRATE Index
    date_col = df.columns[0]
    sofr_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.set_index(date_col).sort_index()

    sofr = pd.to_numeric(df[sofr_col], errors="coerce").dropna()

    # SOFR + 4% puis convertir en taux mensuel
    sofr_plus_4_monthly = (sofr + 4.0) / 100.0 / 12.0
    sofr_plus_4_monthly.name = "sofr_plus_4_monthly"
    return sofr_plus_4_monthly


# ═══════════════════════════════════════════════════════════════════
#  2.  Pivot : format long -> matrice (dates x tickers)
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
      - pas plus de `max_nan_pct` de NaN sur toute la periode
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
#  4.  Composition de l'indice a chaque date de rebal
# ═══════════════════════════════════════════════════════════════════

def get_members_at_date(df_long: pd.DataFrame, date) -> list:
    """
    Retourne la liste des tickers qui etaient dans le SPX a une date donnee.
    (utilise la feuille SPX_Components en format long)
    """
    mask = df_long["date"] == pd.Timestamp(date)
    return df_long.loc[mask, "ticker"].dropna().unique().tolist()


# ═══════════════════════════════════════════════════════════════════
#  5.  Interface principale
# ═══════════════════════════════════════════════════════════════════

def load_all(path=DATA_FILE):
    """
    Point d entree unique : retourne un dict avec toutes les donnees
    nettoyees et pretes pour le backtest.

    Returns
    -------
    data : dict
        "df_long"        : DataFrame format long (date, ticker, px_last)
        "prices"         : DataFrame matrice (date x ticker) PX_LAST - filtre
        "returns"        : DataFrame matrice rendements mensuels
        "spx_monthly"    : Series SPX mensuel
        "spx_daily"      : Series SPX daily
        "rf_annual"      : Series taux sans risque annuel
        "sofr_plus_4_m"  : Series SOFR + 4% mensuel
        "df_sectors"     : DataFrame mapping ticker -> secteur
        "rebal_dates"    : list de dates de rebalancement
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
        rf_annual = load_risk_free(path)
    except Exception:
        print("  [!] Feuille TAUX_SANS_RISQUE non trouvee -> Rf = 0")
        rf_annual = pd.Series(dtype=float)

    # SOFR + 4%
    try:
        sofr_plus_4_m = load_sofr(path)
    except Exception:
        print("  [!] Feuille SOFRRATE non trouvee -> SOFR+4% = Rf")
        sofr_plus_4_m = rf_annual.copy()  # fallback sur rf

    # Secteurs
    try:
        df_sectors = load_sectors(path)
    except Exception:
        print("  [!] Feuille SPX_Sector non trouvee -> secteurs vides")
        df_sectors = pd.DataFrame(columns=['ticker', 'sector'])

    # Pivot + filtre
    prices = pivot_prices(df_long)
    prices = filter_universe(prices)

    # Returns
    rets = compute_returns(prices)

    # Dates de rebal = toutes les dates de la matrice
    rebal_dates = prices.index.tolist()

    return {
        "df_long":        df_long,
        "prices":         prices,
        "returns":        rets,
        "spx_monthly":    spx_m,
        "spx_daily":      spx_d,
        "rf_annual":      rf_annual,
        "sofr_plus_4_m":  sofr_plus_4_m,
        "df_sectors":     df_sectors,
        "rebal_dates":    rebal_dates,
    }


if __name__ == "__main__":
    data = load_all()
    print("Prix shape       :", data["prices"].shape)
    print("Returns shape    :", data["returns"].shape)
    print("Nb rebal dates   :", len(data["rebal_dates"]))
    print("SPX monthly pts  :", len(data["spx_monthly"]))
    print("SPX daily pts    :", len(data["spx_daily"]))
    print("Rf annual pts    :", len(data["rf_annual"]))
    print("SOFR+4% pts      :", len(data["sofr_plus_4_m"]))
    print("Secteurs shape   :", data["df_sectors"].shape)
    print("Exemple tickers  :", list(data["prices"].columns[:5]))
