"""
bloomberg.py — Wrapper Bloomberg API (xbbg / blpapi).

Fournit les fonctions d'accès aux données Bloomberg :
  - bdh()              : Historical Data (prix, volumes, etc.)
  - bdp()              : Reference Data (P/E, EPS, etc.)
  - get_index_members(): Composition d'un indice à une date donnée
  - get_px_last_asof() : Prix PX_LAST au plus proche d'une date
  - get_spx_monthly()  : SPX PX_LAST mensuel sur une plage de dates

Requiert :
  - blpapi     : pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi
  - xbbg       : pip install xbbg
  - pandas_market_calendars : pip install pandas_market_calendars

Note : Ce module nécessite un terminal Bloomberg actif sur la machine.
       Si Bloomberg n'est pas disponible, utiliser data_loader.py qui charge
       les données depuis le fichier Excel pré-extrait.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("[bloomberg.py] xbbg non installé — mode offline uniquement.")

try:
    import pandas_market_calendars as mcal
    MCAL_AVAILABLE = True
except ImportError:
    MCAL_AVAILABLE = False
    print("[bloomberg.py] pandas_market_calendars non installé.")


# ═══════════════════════════════════════════════════════════════════
#  Utilitaires
# ═══════════════════════════════════════════════════════════════════

def _check_bloomberg():
    """Vérifie que Bloomberg est disponible."""
    if not BLOOMBERG_AVAILABLE:
        raise RuntimeError(
            "Bloomberg (xbbg/blpapi) non disponible sur cette machine. "
            "Utilisez data_loader.py pour charger les données depuis Excel."
        )


def _chunk(lst: list, n: int = 150):
    """Découpe une liste en sous-listes de taille n (limite Bloomberg API)."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ═══════════════════════════════════════════════════════════════════
#  1.  BDH — Historical Data
# ═══════════════════════════════════════════════════════════════════

def bdh(tickers, fields, start_date, end_date, **kwargs) -> dict:
    """
    Wrapper Bloomberg BDH (Historical Data).

    Parameters
    ----------
    tickers    : str ou list[str]  — ex: "AAPL UW Equity" ou ["AAPL UW Equity", ...]
    fields     : str ou list[str]  — ex: "PX_LAST" ou ["PX_LAST", "PX_VOLUME"]
    start_date : str               — "YYYY-MM-DD"
    end_date   : str               — "YYYY-MM-DD"

    Returns
    -------
    dict[field] = DataFrame(index=dates, columns=tickers)
        — une matrice par field, exactement le format "1 page par field"
          demandé par la consigne (lignes=dates, colonnes=tickers).
    """
    _check_bloomberg()

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(fields, str):
        fields = [fields]

    # Appel Bloomberg par chunks pour éviter les limites API
    all_data = []
    for chunk in _chunk(tickers, n=150):
        df = blp.bdh(
            tickers=chunk,
            flds=fields,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return {f: pd.DataFrame() for f in fields}

    combined = pd.concat(all_data, axis=1)

    # Séparer par field → dict[field] = DataFrame(date × ticker)
    result = {}
    for field in fields:
        if isinstance(combined.columns, pd.MultiIndex):
            # Colonnes = (ticker, field) → filtrer sur le field
            mask = combined.columns.get_level_values(1).str.upper() == field.upper()
            sub = combined.loc[:, mask].copy()
            sub.columns = sub.columns.get_level_values(0)  # garder juste les tickers
        else:
            sub = combined.copy()
        result[field] = sub

    return result


# ═══════════════════════════════════════════════════════════════════
#  2.  BDP — Reference Data (données fondamentales)
# ═══════════════════════════════════════════════════════════════════

def bdp(tickers, fields, overrides=None) -> pd.DataFrame:
    """
    Wrapper Bloomberg BDP (Reference Data / données fondamentales).

    Parameters
    ----------
    tickers   : str ou list[str]  — ex: ["AAPL UW Equity", "MSFT UW Equity"]
    fields    : str ou list[str]  — ex: ["PE_RATIO", "BEST_EPS"]
    overrides : dict ou None      — ex: {"BEST_FPERIOD_OVERRIDE": "1FY"}

    Returns
    -------
    DataFrame(index=tickers, columns=fields)
    """
    _check_bloomberg()

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(fields, str):
        fields = [fields]

    all_data = []
    for chunk in _chunk(tickers, n=150):
        if overrides:
            df = blp.bdp(tickers=chunk, flds=fields, **overrides)
        else:
            df = blp.bdp(tickers=chunk, flds=fields)
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame(columns=fields)

    return pd.concat(all_data, axis=0)


# ═══════════════════════════════════════════════════════════════════
#  3.  Composition d'indice (membres à une date)
# ═══════════════════════════════════════════════════════════════════

def get_index_members(index_ticker: str, date: str,
                      add_suffix: str = " Equity") -> list:
    """
    Récupère la composition d'un indice à une date donnée via
    Bloomberg BDS (INDX_MWEIGHT_HIST).

    Parameters
    ----------
    index_ticker : str  — ex: "SPX Index"
    date         : str  — "YYYY-MM-DD"
    add_suffix   : str  — suffixe à ajouter (" Equity")

    Returns
    -------
    list[str] — liste de tickers (ex: ["AAPL UW Equity", ...])
    """
    _check_bloomberg()

    dt = date.replace("-", "")
    raw = blp.bds(index_ticker, "INDX_MWEIGHT_HIST", END_DATE_OVERRIDE=dt)

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return []

    # Trouver la colonne contenant les tickers
    cols = {str(c).strip().lower(): c for c in raw.columns}
    preferred_keys = [
        "member_ticker_and_exchange_code",
        "index_member",
        "member",
    ]
    tcol = next((cols[k] for k in preferred_keys if k in cols), None)
    if not tcol:
        return []

    s = raw[tcol].astype(str).str.strip()
    s = s[s.notna() & (s != "") & (s.str.lower() != "nan")]

    # Filtre : enlever les IDs numériques Bloomberg
    s = s[~s.str.match(r"^\d")]

    # Garder uniquement "TICKER EXCH" (ex: AAPL UW)
    s = s[s.str.match(r"^[A-Z0-9./-]+ [A-Z]{1,4}$")]

    # Ajouter le suffixe " Equity" si absent
    s = s.apply(lambda x: x if x.upper().endswith(" EQUITY") else x + add_suffix)

    # Dédoublonner en conservant l'ordre
    seen = set()
    out = []
    for x in s.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ═══════════════════════════════════════════════════════════════════
#  4.  PX_LAST as-of (prix au plus proche d'une date)
# ═══════════════════════════════════════════════════════════════════

def get_px_last_asof(tickers: list, asof_date: str,
                     lookback_days: int = 10) -> dict:
    """
    Retourne le dernier prix (PX_LAST) disponible pour chaque ticker
    au plus proche de asof_date (avec un lookback de sécurité).

    Parameters
    ----------
    tickers       : list[str]
    asof_date     : str "YYYY-MM-DD"
    lookback_days : int — nb jours de lookback pour trouver le dernier prix

    Returns
    -------
    dict[ticker] = float (PX_LAST) ou NaN si non disponible
    """
    _check_bloomberg()

    if not tickers:
        return {}

    end = pd.to_datetime(asof_date)
    start = end - pd.Timedelta(days=lookback_days)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    out = {}

    for chunk in _chunk(tickers, n=150):
        df = blp.bdh(
            tickers=chunk,
            flds="PX_LAST",
            start_date=start_s,
            end_date=end_s,
        )

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            for t in chunk:
                out[t] = float("nan")
            continue

        # Cas multi-tickers : colonnes MultiIndex (ticker, field)
        if isinstance(df.columns, pd.MultiIndex):
            lvl1 = df.columns.get_level_values(1).astype(str).str.strip().str.upper()
            mask = (lvl1 == "PX_LAST")

            if not mask.any():
                for t in chunk:
                    out[t] = float("nan")
                continue

            px = df.loc[:, mask].copy()
            px.columns = px.columns.get_level_values(0)

            for t in chunk:
                if t in px.columns:
                    s = px[t].dropna()
                    out[t] = float(s.iloc[-1]) if len(s) else float("nan")
                else:
                    out[t] = float("nan")
        else:
            # Cas 1 seul ticker (structure non MultiIndex)
            col0 = df.columns[0]
            s = df[col0].dropna()
            out[chunk[0]] = float(s.iloc[-1]) if len(s) else float("nan")

    return out


# ═══════════════════════════════════════════════════════════════════
#  5.  SPX PX_LAST mensuel
# ═══════════════════════════════════════════════════════════════════

def get_spx_monthly(dates: list) -> pd.DataFrame:
    """
    Récupère le SPX PX_LAST pour une liste de dates (month-ends).

    Parameters
    ----------
    dates : list[str] — dates au format "YYYY-MM-DD"

    Returns
    -------
    DataFrame avec colonnes ["date", "spx_px_last"]
    """
    _check_bloomberg()

    if not dates:
        return pd.DataFrame(columns=["date", "spx_px_last"])

    start = dates[0]
    end = dates[-1]

    df = blp.bdh(
        tickers="SPX Index",
        flds="PX_LAST",
        start_date=start,
        end_date=end,
    )

    s = df.iloc[:, 0].copy()
    s.index = pd.to_datetime(s.index)

    wanted = pd.to_datetime(pd.Series(dates))
    out = []
    for d in wanted:
        sub = s.loc[:d].dropna()
        out.append(float(sub.iloc[-1]) if len(sub) else float("nan"))

    return pd.DataFrame({
        "date": wanted.dt.strftime("%Y-%m-%d"),
        "spx_px_last": out,
    })


# ═══════════════════════════════════════════════════════════════════
#  6.  Calendrier NYSE (month-ends trading)
# ═══════════════════════════════════════════════════════════════════

def get_monthly_nyse_trading_month_ends(start_date: str,
                                         end_date: str) -> list:
    """
    Renvoie les derniers jours de trading NYSE pour chaque mois
    entre start_date et end_date.

    Returns
    -------
    list[str] — dates au format "YYYY-MM-DD"
    """
    if not MCAL_AVAILABLE:
        raise RuntimeError("pandas_market_calendars non installé.")

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index
    monthly = trading_days.to_series().groupby(trading_days.to_period("M")).max()
    return monthly.dt.strftime("%Y-%m-%d").tolist()


# ═══════════════════════════════════════════════════════════════════
#  7.  Pipeline complet : extraction univers + export Excel
# ═══════════════════════════════════════════════════════════════════

def extract_full_universe(
    index_ticker: str = "SPX Index",
    start_date: str = "2015-01-01",
    end_date: str = "2025-12-31",
    output_path: Path | str | None = None,
    fields: list | None = None,
) -> dict:
    """
    Pipeline complet d'extraction Bloomberg :
      1. Calcule les dates de rebalancement (month-end NYSE)
      2. Pour chaque date, récupère la composition de l'indice
      3. Récupère PX_LAST (et autres fields) pour chaque membre
      4. Récupère SPX PX_LAST mensuel
      5. Exporte en Excel (1 sheet par field + 1 sheet SPX)

    Parameters
    ----------
    index_ticker : str  — ticker de l'indice (ex: "SPX Index")
    start_date   : str  — début de la période
    end_date     : str  — fin de la période
    output_path  : Path — chemin du fichier Excel de sortie
    fields       : list — fields Bloomberg à extraire (défaut: ["PX_LAST"])

    Returns
    -------
    dict avec clés :
      "df_components" : DataFrame format long (date, ticker, PX_LAST, ...)
      "df_spx"        : DataFrame (date, spx_px_last)
      "dates"         : list de dates de rebalancement
    """
    _check_bloomberg()

    if fields is None:
        fields = ["PX_LAST"]

    if output_path is None:
        output_path = Path(__file__).parent / "SPX_universe_with_price_tes_date2.xlsx"
    output_path = Path(output_path)

    # ── 1. Dates de rebalancement ──
    print("[1/4] Calcul des dates de rebalancement (NYSE month-ends)...")
    dates = get_monthly_nyse_trading_month_ends(start_date, end_date)
    print(f"  {len(dates)} dates de rebalancement")

    # ── 2. Extraction univers + prix ──
    print("[2/4] Extraction de l'univers et des prix...")
    rows = []
    for i, d in enumerate(dates):
        print(f"  [{i+1}/{len(dates)}] {d}...", end="", flush=True)

        members = get_index_members(index_ticker, d)
        print(f" members={len(members)}", end="")

        px_map = get_px_last_asof(members, d, lookback_days=10)
        print(" ✓")

        for t in members:
            row = {"date": d, "ticker": t}
            row["PX_LAST"] = px_map.get(t, float("nan"))
            rows.append(row)

    df_components = pd.DataFrame(rows)
    df_components["date"] = pd.to_datetime(df_components["date"])

    # ── 3. SPX mensuel ──
    print("[3/4] Extraction SPX mensuel...")
    df_spx = get_spx_monthly(dates)

    # ── 4. Export Excel ──
    print("[4/4] Export Excel...")
    df_comp_export = df_components.copy()
    df_comp_export["date"] = df_comp_export["date"].dt.strftime("%d/%m/%Y")

    df_spx_export = df_spx.copy()
    df_spx_export["date"] = pd.to_datetime(df_spx_export["date"]).dt.strftime("%d/%m/%Y")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df_comp_export.to_excel(writer, sheet_name="SPX_Components", index=False)
        df_spx_export.to_excel(writer, sheet_name="SPX_PX_LAST_Monthly", index=False)

    print(f"\n✅ Fichier exporté : {output_path}")
    print(f"  Components rows : {len(df_components):,}")
    print(f"  SPX monthly pts : {len(df_spx):,}")

    return {
        "df_components": df_components,
        "df_spx": df_spx,
        "dates": dates,
    }


# ═══════════════════════════════════════════════════════════════════
#  Exécution directe : extraction complète
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = extract_full_universe(
        index_ticker="SPX Index",
        start_date="2015-01-01",
        end_date="2025-12-31",
    )
