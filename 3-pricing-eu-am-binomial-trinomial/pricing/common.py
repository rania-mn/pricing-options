"""Outils communs : payoff d'une option vanille."""
import numpy as np


def payoff(S, K, kind):
    """Payoff à l'exercice : (S-K)+ pour un call, (K-S)+ pour un put."""
    S = np.asarray(S, dtype=float)
    if kind == "call":
        return np.maximum(S - K, 0.0)
    if kind == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("kind doit valoir 'call' ou 'put'")
