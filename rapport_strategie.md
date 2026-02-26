# 📊 Backtest Long-Only ERC - Challenge Natixis

## 🎯 **Objectif du Projet**

Sratégie **Long-Only** avec allocation **Equal Risk Contribution (ERC)** pour le challenge Natixis.

### Contraintes du Challenge
- ✅ **Long uniquement** (pas de vente à découvert)  
- ✅ **25 titres sélectionnés** avec poids optimaux
- ✅ **Univers S&P 500** (395 titres disponibles)
- ✅ **Politique d'exclusion ESG** intégrée
- ✅ **Rebalancement mensuel** (fin de mois)

---

## 📈 **Méthodologie de Sélection**

### Score Composite Multi-Factoriel
La stratégie sélectionne les 25 meilleurs titres basés sur un **score composite** pondéré :

| Facteur | Poids | Description |
|---------|--------|-------------|
| **Momentum** | 35% | Momentum 12M-1M pondéré dans le temps |
| **Mean Reversion** | 25% | Ratio volatilité court/long terme |
| **Low Volatility** | 40% | Volatilité réalisée 12 mois |

### Formule du Score
```python
score_composite = (
    0.35 * zscore(momentum_12m1m_weighted) +
    0.25 * zscore(vol_mean_reversion) +
    0.40 * zscore(low_volatility)
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
- **BUFFER_RANK = 40** : Un titre existant n'est remplacé que s'il sort du **top 40**
- **Objectif** : Réduire le turnover excessif 

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

## 🔧 **Améliorations Techniques**

### 1. Momentum Pondéré dans le Temps
**Avant** : Momentum classique 12M-1M uniforme  
**Après** : Pondération linéaire (mois récents = poids plus élevé)

```python
# Calcul momentum pondéré
weights = np.linspace(1, lookback, lookback)  # [1, 2, ..., 11]
weights = weights / weights.sum()  # Normalisation

momentum_weighted = (ret_monthly[:-1] * weights).sum()
```

**Amélioration** : Sharpe 3.118 → 3.426 (+9.9%)

### 2. Buffer Anti-Turnover
**Problème** : Turnover élevé (51.6%) à cause de la sélection top 25 stricte  
**Solution** : Garder les positions existantes si elles restent dans le top 40  
**Résultat** : Turnover réduit à ~41%

---

## 📊 **Performance (2016-2025)**

### Métriques Principales

| Métrique | Valeur | Benchmark S&P 500 |
|----------|--------|-------------------|
| **CAGR** | 24.36% | ~10-12% |
| **Volatilité** | 6.94% | ~15-18% |
| **Sharpe Ratio** | 3.22 | ~0.6-0.8 |
| **Sortino Ratio** | 5.40 | ~0.9-1.2 |
| **Max Drawdown** | -3.80% | ~-15-20% |
| **Hit Ratio** | 83.33% | ~60% |

### Performance Cumulative
- **Return Total** : 779.36%
- **Années positives** : 9/10
- **VaR 95%** : -1.42%
- **Turnover Moyen** : 40.79%

---

## 🔄 **Architecture Technique**

### Structure du Code

```
Backtest/
├── config.py           # Paramètres globaux + exclusions ESG
├── data_loader.py      # Chargement données Excel
├── signals.py          # Calcul des signaux (momentum, vol, etc.)
├── allocation.py       # Allocation ERC
├── backtest.py         # Moteur de backtest
├── risk.py             # Stop-loss & gestion risque  
├── costs.py            # Coûts de transaction
├── metrics.py          # Calcul métriques performance
├── visualization.py    # Graphiques & exports
└── main.py             # Point d'entrée
```

### Flux d'Exécution
1. **Chargement** données (prix, returns, risk-free)
2. **Filtrage ESG** (395 → 363 titres)  
3. **Calcul signaux** mensuels (momentum, vol, mean reversion)
4. **Sélection top 25** avec buffer anti-turnover
5. **Allocation ERC** via optimisation scipy
6. **Application stop-loss** position (-10%)
7. **Calcul coûts** transaction (5bps)
8. **Métriques & graphiques**

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

### Forces de la Stratégie
✅ **Excellence risk/return** : Sharpe 3.22 vs marché ~0.7  
✅ **Faible volatilité** : 6.94% vs marché ~16%  
✅ **Drawdown contrôlé** : Max DD -3.80%  
✅ **Consistance** : Hit ratio 83%, 9/10 années positives  
✅ **Compliance ESG** : 32 exclusions respectées  

### Défis & Limites
⚠️ **Concentration** : 25 titres seulement  
⚠️ **Biais défensif** : Sur-poids Utilities/Staples  
⚠️ **Market timing** : Pas de signal "cash" en Bear Market  
⚠️ **Style drift** : Exposition style non contrôlée  

### Recommandations
1. **Monitoring sectoriel** pour éviter concentrations
2. **Backtests crisis** sur 2008, 2020 pour stress-test  
3. **Out-of-sample** validation sur données récentes
4. **Implémentation graduelle** avec capital limité initial

---
  
*Challenge Natixis - M2 Quantitative Finance*