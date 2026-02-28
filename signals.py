"""
signals.py — Score composite pour sélection Long-Only des top N titres.

Facteurs du score composite :
  1. Momentum 12m-1m  : rendement cumulé sur 12 mois, excluant le dernier mois
  2. Vol Mean Reversion : ratio vol courte / vol longue (inversé)
  3. Vol Spread        : volatilité réalisée 12m inversée (long low-vol)

Le score final est un z-score pondéré de ces 3 facteurs.
"""

import pandas as pd
import numpy as np
from config import N_STOCKS, SIGNAL_WEIGHTS, BUFFER_RANK


# ═══════════════════════════════════════════════════════════════════
#  Signaux individuels
# ═══════════════════════════════════════════════════════════════════

def momentum_12m1m(prices: pd.DataFrame, t: int, lookback: int = 11) -> pd.Series:
    """
    Momentum pondéré par le temps (12M - 1M).

    Décompose le rendement sur 11 mois (mois -2 à -12, skip le dernier mois)
    en rendements mensuels individuels, puis les pondère linéairement :
    les mois récents pèsent plus que les mois anciens.

    Avantages vs momentum classique :
      - Meilleur Sharpe et Sortino (ajustement risque)
      - Volatilité plus faible, drawdown réduit
      - Hit ratio supérieur
    """
    if t < lookback + 1:
        return pd.Series(dtype=float)

    # Rendements mensuels de t-lookback-1 à t-1 (on skip le mois t)
    sub = prices.iloc[t - lookback - 1:t]  # lookback+1 prix → lookback rendements
    rets = sub.pct_change().iloc[1:]        # lookback rendements (ancien → récent)

    n = len(rets)
    # Poids linéaires croissants : 1, 2, 3, ..., n (récent pèse plus)
    raw_weights = np.arange(1, n + 1, dtype=float)
    raw_weights = raw_weights / raw_weights.sum()

    # Rendement pondéré = somme(w_k * r_k)
    signal = (rets.values * raw_weights[:, np.newaxis]).sum(axis=0)
    signal = pd.Series(signal, index=rets.columns)

    return signal.dropna()


def mean_reversion_1m(prices: pd.DataFrame, t: int, lookback: int = 12) -> pd.Series:
    """
    Vol Mean Reversion : ratio vol courte (3m) / vol longue (12m), inversé.
    Ratio bas = vol en compression → score élevé (favorable).
    """
    if t < lookback:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - lookback:t + 1]
    rets = sub.pct_change().iloc[1:]

    vol_long = rets.std()
    vol_long = vol_long.replace(0, np.nan)

    short_window = 3
    if len(rets) >= short_window:
        vol_short = rets.iloc[-short_window:].std()
    else:
        vol_short = rets.std()

    ratio = vol_short / vol_long
    signal = -ratio

    return signal.dropna()


def vol_spread(prices: pd.DataFrame, t: int, window: int = 12) -> pd.Series:
    """
    Vol Spread (low-vol premium) :
    Volatilité réalisée sur les `window` derniers mois, inversée.
    Les moins volatils ont un score élevé.
    """
    if t < window:
        return pd.Series(dtype=float)

    sub = prices.iloc[t - window:t + 1]
    rets = sub.pct_change().iloc[1:]
    vol = rets.std()

    signal = -vol
    return signal.dropna()


# ═══════════════════════════════════════════════════════════════════
#  Dispatch : appeler un signal par nom
# ═══════════════════════════════════════════════════════════════════

SIGNAL_FUNCS = {
    "momentum":        momentum_12m1m,
    "mean_reversion":  mean_reversion_1m,
    "vol_spread":      vol_spread,
}


def compute_signal(signal_name: str, prices: pd.DataFrame, t: int) -> pd.Series:
    """Calcule le signal demandé à la date t."""
    func = SIGNAL_FUNCS[signal_name]
    return func(prices, t)


# ═══════════════════════════════════════════════════════════════════
#  Score composite + sélection top N
# ═══════════════════════════════════════════════════════════════════

def _zscore(s: pd.Series) -> pd.Series:
    """Standardise une Series en z-score."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return s * 0.0
    return (s - mu) / sigma


def compute_composite_score(
    prices: pd.DataFrame,
    t: int,
    members: list | None = None,
    weights: dict | None = None,
) -> pd.Series:
    """
    Calcule le score composite pondéré (z-score) à partir de tous les signaux.

    Parameters
    ----------
    prices  : DataFrame (date × ticker)
    t       : index position de la date de rebalancement
    members : tickers de l'univers à cette date
    weights : dict signal_name → poids (défaut = SIGNAL_WEIGHTS)

    Returns
    -------
    score : Series(ticker → score composite), trié décroissant
    """
    if weights is None:
        weights = SIGNAL_WEIGHTS

    all_zscores = {}
    for sig_name, sig_weight in weights.items():
        raw = compute_signal(sig_name, prices, t)
        if raw.empty:
            continue
        if members is not None:
            raw = raw.reindex(members).dropna()
        all_zscores[sig_name] = _zscore(raw) * sig_weight

    if not all_zscores:
        return pd.Series(dtype=float)

    # Aligner tous les z-scores sur les mêmes tickers
    df_z = pd.DataFrame(all_zscores)
    # Garder uniquement les tickers ayant tous les signaux
    df_z = df_z.dropna()

    if df_z.empty:
        return pd.Series(dtype=float)

    composite = df_z.sum(axis=1)
    return composite.sort_values(ascending=False)


def select_top_n(
    prices: pd.DataFrame,
    t: int,
    members: list | None = None,
    n: int = N_STOCKS,
    prev_stocks: list | None = None,
    buffer_rank: int = BUFFER_RANK,
) -> list:
    """
    Sélectionne les top N titres par score composite avec mécanisme de buffer.

    Le buffer réduit le turnover : un titre déjà en portefeuille n'est remplacé
    que s'il tombe en dessous du rang `buffer_rank` (défaut 40).
    Les nouveaux titres doivent être dans le top N strict.

    Returns
    -------
    list de tickers (N titres)
    """
    score = compute_composite_score(prices, t, members=members)
    if score.empty:
        return []

    # Premier mois ou pas de portefeuille précédent : top N strict
    if prev_stocks is None or len(prev_stocks) == 0:
        return score.head(n).index.tolist()

    # Buffer : garder les titres existants qui sont encore dans le top buffer_rank
    top_buffer = score.head(buffer_rank).index.tolist()
    top_strict = score.head(n).index.tolist()

    # 1. Garder les anciens titres qui sont encore dans le buffer
    kept = [s for s in prev_stocks if s in top_buffer]

    # 2. Compléter avec les meilleurs nouveaux titres (top strict) si on n'a pas assez
    if len(kept) < n:
        for s in top_strict:
            if s not in kept:
                kept.append(s)
            if len(kept) >= n:
                break

    # 3. Si on a trop de titres (cas rare), garder les N meilleurs par score
    if len(kept) > n:
        kept_scores = score.reindex(kept).dropna().sort_values(ascending=False)
        kept = kept_scores.head(n).index.tolist()

    return kept


# ═══════════════════════════════════════════════════════════════════
#  VALUE et SIZE strategies
# ═══════════════════════════════════════════════════════════════════

def compute_value_score(fundamentals: dict, t: int, members: list = None) -> pd.Series:
    """
    Score VALUE basé sur P/B et P/E ratios.
    Plus le ratio est faible (valorisation cheap), meilleur le score.
    """
    pb_df = fundamentals.get('pb_ratio', pd.DataFrame())
    pe_df = fundamentals.get('pe_ratio', pd.DataFrame())
    
    if pb_df.empty and pe_df.empty:
        return pd.Series(dtype=float)
    
    scores = pd.Series(dtype=float)
    
    # P/B ratio : plus faible = meilleur
    if not pb_df.empty and t < len(pb_df):
        pb_values = pb_df.iloc[t].dropna()
        if len(pb_values) > 1:
            pb_zscore = (pb_values - pb_values.mean()) / pb_values.std()
            pb_score = -pb_zscore  # Inverser : faible P/B = score élevé
            scores = scores.add(pb_score, fill_value=0)
    
    # P/E ratio : plus faible = meilleur
    if not pe_df.empty and t < len(pe_df):
        pe_values = pe_df.iloc[t].dropna()
        if len(pe_values) > 1:
            pe_zscore = (pe_values - pe_values.mean()) / pe_values.std()
            pe_score = -pe_zscore  # Inverser : faible P/E = score élevé  
            scores = scores.add(pe_score, fill_value=0)
    
    # Filtrer par membres autorisés
    if members is not None:
        scores = scores.reindex(members).dropna()
    
    return scores.sort_values(ascending=False)


def compute_size_score(fundamentals: dict, t: int, members: list = None) -> pd.Series:
    """
    Score SIZE basé sur Market Cap.
    Plus la capitalisation est faible (small caps), meilleur le score.
    """
    mcap_df = fundamentals.get('market_cap', pd.DataFrame())
    
    if mcap_df.empty or t >= len(mcap_df):
        return pd.Series(dtype=float)
    
    mcap_values = mcap_df.iloc[t].dropna()
    
    if len(mcap_values) <= 1:
        return pd.Series(dtype=float)
    
    # Inverser : petite capitalisation = score élevé
    mcap_zscore = (mcap_values - mcap_values.mean()) / mcap_values.std()
    size_score = -mcap_zscore
    
    # Filtrer par membres autorisés
    if members is not None:
        size_score = size_score.reindex(members).dropna()
    
    return size_score.sort_values(ascending=False)


def compute_value_size_score(fundamentals: dict, t: int, members: list = None) -> pd.Series:
    """
    Score combiné VALUE + SIZE avec pondération égale.
    """
    value_score = compute_value_score(fundamentals, t, members)
    size_score = compute_size_score(fundamentals, t, members)
    
    # Combiner avec normalisation
    combined_score = pd.Series(dtype=float)
    all_tickers = set()
    
    if not value_score.empty:
        all_tickers.update(value_score.index)
    if not size_score.empty:
        all_tickers.update(size_score.index)
    
    if not all_tickers:
        return pd.Series(dtype=float)
    
    # Normaliser et combiner
    for ticker in all_tickers:
        score = 0.0
        count = 0
        
        if ticker in value_score.index:
            score += value_score[ticker]
            count += 1
        if ticker in size_score.index:
            score += size_score[ticker]
            count += 1
        
        if count > 0:
            combined_score[ticker] = score / count
    
    return combined_score.sort_values(ascending=False)


def select_hybrid_portfolio(prices: pd.DataFrame, fundamentals: dict, t: int, 
                           members: list = None, n_composite: int = 15, n_value_size: int = 5) -> list:
    """
    Sélection hybride : 70% composite (15 actions) + 30% value/size (5 actions).
    Si les données fondamentales sont indisponibles, fallback sur stratégie composite pure.
    """
    total_target = n_composite + n_value_size
    
    # Sélection composite (momentum dominant)
    composite_selected = select_top_n(prices, t, members=members, n=n_composite)
    
    # Vérifier si la sélection composite a fonctionné
    if not composite_selected or len(composite_selected) == 0:
        print(f"    [!] Echec selection composite a t={t}")
        return []
    
    # Tentative de sélection value/size
    value_size_score = compute_value_size_score(fundamentals, t, members=members)
    
    # Si pas de données fondamentales, fallback sur composite étendu
    if value_size_score.empty or len(value_size_score) == 0:
        print(f"    [!] Donnees VALUE/SIZE indisponibles -> fallback composite ({total_target} actions)")
        fallback_selected = select_top_n(prices, t, members=members, n=total_target)
        if fallback_selected and len(fallback_selected) > 0:
            print(f"    [OK] Fallback successful: {len(fallback_selected)} actions")
            return fallback_selected
        else:
            print(f"    [!] Fallback failed, using original composite selection")
            return composite_selected
    
    value_size_selected = value_size_score.head(n_value_size * 2).index.tolist()  # Plus de candidats
    
    # Éviter les doublons
    value_size_final = [ticker for ticker in value_size_selected 
                       if ticker not in composite_selected]
    
    # Si pas assez de titres value/size uniques, compléter avec composite étendu
    if len(value_size_final) < n_value_size:
        needed = n_value_size - len(value_size_final)
        composite_score = compute_composite_score(prices, t, members=members)
        extra_composite = [ticker for ticker in composite_score.index 
                          if ticker not in composite_selected and ticker not in value_size_final]
        value_size_final.extend(extra_composite[:needed])
    
    final_selection = composite_selected + value_size_final[:n_value_size]
    print(f"    [OK] Selection hybride: {len(composite_selected)} COMPOSITE + {len(value_size_final[:n_value_size])} VALUE/SIZE = {len(final_selection)} total")
    
    return final_selection
