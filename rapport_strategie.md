# 📊 Stratégie Hybride VALUE/SIZE + COMPOSITE - Challenge Natixis

## 🎯 **Objectif du Projet**

Stratégie **Long-Only Hybride** combinant approches quantitative et fondamentale avec allocation **Equal Risk Contribution (ERC)** pour le challenge Natixis.

### Contraintes du Challenge
- ✅ **Long uniquement** (pas de vente à découvert)  
- ✅ **20 titres sélectionnés** (5 VALUE/SIZE + 15 COMPOSITE)
- ✅ **Univers S&P 500** (395 titres disponibles → 363 post-ESG)
- ✅ **Politique d'exclusion ESG** intégrée (32 exclusions)
- ✅ **Rebalancement mensuel** (fin de mois)

---

## 📈 **Méthodologie Hybride de Sélection**

### Architecture Double : 30% VALUE/SIZE + 70% COMPOSITE
La stratégie hybride combine deux approches complémentaires :

#### 🏛️ **Poche VALUE/SIZE (30% - 5 actions)**
| Facteur | Description |
|---------|-------------|
| **P/B Ratio** | Price-to-Book ratio (faible = attractif) |
| **P/E Ratio** | Price-to-Earnings ratio (faible = sous-valorisé) |
| **Market Cap** | Capitalisation boursière (biais Small Cap) |

#### ⚡ **Poche COMPOSITE (70% - 15 actions)**
| Facteur | Poids | Description |
|---------|-------|-------------|
| **Momentum** | 70% | Momentum 12M-1M pondéré dans le temps |
| **Mean Reversion** | 10% | Ratio volatilité court/long terme |
| **Low Volatility** | 20% | Volatilité réalisée 12 mois |

### Formules des Scores
```python
# Score VALUE/SIZE
value_size_score = (
    zscore(1/pb_ratio) +  # Inversion pour favoriser faibles P/B
    zscore(1/pe_ratio) +  # Inversion pour favoriser faibles P/E
    zscore(1/market_cap)  # Favorise Small Caps
) / 3

# Score COMPOSITE (momentum-focused)
composite_score = (
    0.70 * zscore(momentum_12m1m_weighted) +
    0.10 * zscore(vol_mean_reversion) +
    0.20 * zscore(low_volatility)
)
```

---

## ⚖️ **Allocation Equal Risk Contribution (ERC)**

### Principe
L'allocation ERC vise à **égaliser la contribution au risque** de chaque position plutôt que les montants investis.

### Optimisation
```python
# Fonction objectif : minimiser l'écart des contributions au risque
def erc_objective(weights, cov_matrix):
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights / portfolio_vol
    contrib = weights * marginal_contrib
    return np.sum((contrib - contrib.mean()) ** 2)

# Contraintes : Long-Only, somme = 1
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Somme = 100%
]
bounds = [(0, None) for _ in range(n_stocks)]  # Poids ≥ 0
```

---

## 🛡️ **Gestion des Risques**

### Stop-Loss Position
- **Seuil** : -10% depuis le prix d'entrée
- **Action** : Liquidation immédiate de la position

### Buffer de Sélection  
- **BUFFER_RANK = 25** : Un titre existant n'est remplacé que s'il sort du **top 25**
- **Objectif** : Réduire le turnover excessif (adapté aux 20 actions) 

### Coûts de Transaction
- **5 basis points** par transaction (achat/vente)
- Impact sur la performance finale

---

## 🌱 **Politique d'Exclusion ESG**

### Secteurs Exclus (32 titres)

| Catégorie | Nombre | Exemples |
|-----------|---------|----------|
| **Tabac** | 
| **Armes/Défense** | 
| **Énergies Fossiles** | 
| **Jeux d'Argent** | 
| **Alcool** |


---

## 🔧 **Évolution de la Stratégie**

### 1. Momentum Focus + Concentration (Phase 1)
**Optimisation** : Poids momentum porté à 70% vs 35% initialement  
**Concentration** : Réduction 25 → 10 actions pour maximiser alpha  
**Résultat** : Sharpe 0.17 → 1.46 (+758%)

### 2. Stratégie Hybride VALUE/SIZE (Phase 2)
**Innovation** : Combinaison 30% VALUE/SIZE + 70% COMPOSITE  
**Diversification** : Extension à 20 actions (5+15) pour équilibrer risque  
**Fondamentaux** : Intégration P/B, P/E, Market Cap pour capture de valeur  
**Résultat** : Maintien performance avec diversification accrue

### 3. Améliorations Techniques
```python
# Fallback robuste en cas d'absence données fondamentales
if value_size_score.empty:
    return select_top_n(prices, t, n=total_target)  # 20 actions composite

# Évitement doublons VALUE/SIZE ↔ COMPOSITE
value_size_final = [t for t in value_size_selected 
                   if t not in composite_selected]
```

---

## 📊 **Performance Stratégie Hybride (2016-2025)**

### Métriques Principales

| Métrique | Stratégie Hybride | Benchmark S&P 500 | vs Benchmark |
|----------|-------------------|-------------------|---------------|
| **CAGR** | 24.64% | ~10-12% | +12-14pp |
| **Volatilité** | 11.92% | ~15-18% | -3-6pp |
| **Sharpe Ratio** | 1.90 | ~0.6-0.8 | +1.1-1.3 |
| **Sortino Ratio** | 3.73 | ~0.9-1.2 | +2.5-2.8 |
| **Max Drawdown** | -8.98% | ~-15-20% | +6-11pp |
| **Hit Ratio** | 73.33% | ~60% | +13pp |

### Performance Cumulative
- **Return Total** : 815.07%
- **Années positives** : 8/10
- **VaR 95%** : -4.22%
- **CVaR 95%** : -5.23%
- **Turnover Moyen** : 44.48%
- **Coûts cumulés** : 2.67%

---

## 🔄 **Architecture Technique**

### Structure du Code

```
Backtest/
├── config.py           # Paramètres globaux + exclusions ESG + hybride
├── data_loader.py      # Chargement données Excel + fondamentaux
├── signals.py          # Signaux COMPOSITE + VALUE/SIZE + hybride
├── allocation.py       # Allocation ERC
├── backtest.py         # Moteur backtest avec stratégie hybride
├── risk.py             # Stop-loss & gestion risque  
├── costs.py            # Coûts de transaction
├── metrics.py          # Calcul métriques performance
├── visualization.py    # Graphiques & exports
└── main.py             # Point d'entrée stratégie hybride
```

### Flux d'Exécution Hybride
1. **Chargement** données (prix, returns, fondamentals, risk-free)
2. **Filtrage ESG** (395 → 363 titres)  
3. **Sélection VALUE/SIZE** (5 actions basées P/B, P/E, Market Cap)
4. **Sélection COMPOSITE** (15 actions momentum-focused)
5. **Combinaison hybride** (évitement doublons, fallback)
6. **Allocation ERC** via optimisation scipy (20 actions)
7. **Application stop-loss** position (-10%)
8. **Calcul coûts** transaction (5bps)
9. **Métriques & graphiques**

---

## 🚀 **Améliorations Futures**

### 1. Données Fondamentales
- **Value** : P/E, P/B, Dividend Yield
- **Quality** : ROE, ROA, Debt/Capital  
- **Growth** : Sales Growth, Earnings Growth

### 2. Diversification Avancée
- **Secteurs GICS** : Contraintes sectorielles (max 30% par secteur)
- **Géographie** : Country of Risk, exposition régionale
- **Market Cap** : Tilt Small/Mid/Large cap

### 3. Risk Management
- **VaR dynamique** : Ajustement positions selon VaR
- **Corrélations** : Monitoring corrélations croisées  
- **Liquidité** : Filtres volume minimum

### 4. Facteurs Alternatifs
- **Momentum cross-sectionnel** : Rank-based momentum
- **Low Beta** : Anomalie Beta  
- **Profitabilité** : Gross Margins, ROIC

---

## 📋 **Conclusions**

### Forces de la Stratégie Hybride
✅ **Performance ajustée du risque** : Sharpe 1.90 vs marché ~0.7  
✅ **Drawdown maîtrisé** : Max DD -8.98% vs marché ~-20%  
✅ **Diversification équilibrée** : 20 actions, 6 secteurs représentés  
✅ **Approche multi-factorielle** : VALUE/SIZE + COMPOSITE complémentaires  
✅ **Robustesse technique** : Fallback automatique, évitement doublons  
✅ **Compliance ESG** : 32 exclusions respectées rigoureusement  

### Défis & Limites
⚠️ **Données fondamentales** : Dépendance qualité données P/B, P/E, Market Cap  
⚠️ **Concentration modérée** : 20 titres vs 25 initialement  
⚠️ **Biais technologique** : Sur-représentation IT (37.9%)  
⚠️ **Market timing** : Pas de signal "cash" en Bear Market  
⚠️ **Fallback fréquent** : Stratégie VALUE/SIZE peu utilisée (données manquantes)  

### Recommandations
1. **Amélioration données fondamentales** : Sources alternatives P/B, P/E
2. **Contraintes sectorielles** : Limite max 40% par secteur GICS
3. **Backtests crise** : Validation 2008, 2020 pour stress-test
4. **Monitoring VALUE/SIZE** : Tracking effectivité poche fondamentale
5. **Out-of-sample** validation sur données post-2025

---
  
*Challenge Natixis - M2 Quantitative Finance*