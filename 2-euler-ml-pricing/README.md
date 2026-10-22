# Pricing d'Options Europeennes : Schema d'Euler et Machine Learning

Projet de Calcul Stochastique - 2eme annee CI FID, ENSA-UIZ.
Pricing d'un Call europeen par schema d'Euler, Monte Carlo et Machine Learning, compare a la formule de Black-Scholes.

## Fichiers
- `euler_ml_pricing.py` : programme principal (calculs et figures)
- `app_streamlit.py` : interface web
- `requirements.txt` : dependances

## Installation

```
pip install -r requirements.txt
```

## Utilisation

Programme principal (les figures sont enregistrees dans `figures/`) :

```
python euler_ml_pricing.py
```

Interface Streamlit :

```
python -m streamlit run app_streamlit.py
```
