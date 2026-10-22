"""Modèle continu de Black-Scholes : prix de référence des arbres."""
import numpy as np
from scipy.special import ndtr as norm_cdf  # fonction de répartition de N(0,1)


def d1_d2(S0, K, T, r, sigma):
    """Quantités d1 et d2 de la formule de Black-Scholes."""
    S0 = np.asarray(S0, dtype=float)
    s = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / s
    return d1, d1 - s


def bs_price(S0, K, T, r, sigma, kind="call"):
    """Prix européen : C = S0 N(d1) - K e^{-rT} N(d2) ; le put se déduit de la parité."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    call = S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    if kind == "call":
        return call
    return call - S0 + K * np.exp(-r * T)  # parité call-put


def bs_delta(S0, K, T, r, sigma, kind="call"):
    """Delta de Black-Scholes (quantité d'actif de la couverture continue)."""
    d1, _ = d1_d2(S0, K, T, r, sigma)
    return norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0
