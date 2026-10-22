"""Méthodes d'accélération de la convergence des arbres vers Black-Scholes.

Le prix CRR converge en O(1/N) mais oscille selon la position du strike par rapport
aux nœuds. Les méthodes ci-dessous éliminent l'oscillation et/ou relèvent l'ordre.
"""
import numpy as np

from .binomial import BinParams, binomial_price, binomial_value, crr_params
from .black_scholes import d1_d2
from .trinomial import trinomial_price

LAMBDA_BOYLE = float(np.sqrt(1.5))   # valeur usuelle de lambda pour le trinomial


# 1) Extrapolation de Richardson sur CRR ---------------------------------------
def crr_richardson(S0, K, T, r, sigma, N, kind="call", american=False):
    """2 P(2N) - P(N) : annule le terme en 1/N s'il était régulier (il oscille)."""
    p1 = binomial_price(S0, K, T, r, sigma, N, kind, american)
    p2 = binomial_price(S0, K, T, r, sigma, 2 * N, kind, american)
    return 2 * p2 - p1


# 2) BBS et BBSR (Broadie-Detemple) ---------------------------------------------
def bbs_price(S0, K, T, r, sigma, N, kind="call", american=False):
    """Binomial Black-Scholes : la dernière période est remplacée par la formule de BS
    (lisse la fonction payoff, qui n'est pas dérivable en K)."""
    return binomial_value(S0, K, N, crr_params(r, sigma, T, N), kind, american,
                          bs_last=(r, sigma, T / N))


def bbsr_price(S0, K, T, r, sigma, N, kind="call", american=False):
    """BBS + Richardson : 2 BBS(N) - BBS(N/2) (N pair)."""
    N += N % 2
    return (2 * bbs_price(S0, K, T, r, sigma, N, kind, american)
            - bbs_price(S0, K, T, r, sigma, N // 2, kind, american))


# 3) Leisen-Reimer : arbre centré sur le strike ---------------------------------
def peizer_pratt(z, n):
    """Inversion de Peizer-Pratt : approche la loi binomiale par une loi normale."""
    z = np.asarray(z, dtype=float)
    expo = -((z / (n + 1 / 3 + 0.1 / (n + 1))) ** 2) * (n + 1 / 6)
    return 0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(expo))


def leisen_reimer_params(S0, K, T, r, sigma, N):
    """Paramètres (N impair, u, d, R, q) : q = PP(d2) et q' = PP(d1), donc u = R q'/q."""
    N += 1 - N % 2                                    # N doit être impair
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    q, q_t = float(peizer_pratt(d2, N)), float(peizer_pratt(d1, N))
    R = np.exp(r * T / N)
    u = R * q_t / q                                    # q' = q u / R (lemme du rapport)
    d = (R - q * u) / (1 - q)                          # martingale : q u + (1-q) d = R
    if not d < R < u:
        raise ValueError("Paramètres Leisen-Reimer hors absence d'arbitrage")
    return N, BinParams(float(u), float(d), float(R), q)


def lr_price(S0, K, T, r, sigma, N, kind="call", american=False):
    """Prix Leisen-Reimer (ordre 2, sans oscillation)."""
    N, p = leisen_reimer_params(S0, K, T, r, sigma, N)
    return binomial_value(S0, K, N, p, kind, american)


# 4) Trinomial centré sur le strike (lambda choisi) -----------------------------
def centered_lambda(S0, K, T, sigma, N):
    """lambda >= 1 tel qu'un nœud de l'arbre tombe exactement sur K (S0 u^m = K)."""
    h = np.sqrt(T / N)
    x = abs(np.log(K / S0))
    m = int(x // (sigma * h))                          # nombre de pas jusqu'à K
    return 1.0 if m == 0 else float(x / (m * sigma * h))


def trinomial_centered_price(S0, K, T, r, sigma, N, kind="call", american=False):
    """Trinomial de Boyle avec lambda centré sur le strike (repli sur lambda = 1)."""
    lam = centered_lambda(S0, K, T, sigma, N)
    try:
        return trinomial_price(S0, K, T, r, sigma, N, kind, american, lam)
    except ValueError:
        return trinomial_price(S0, K, T, r, sigma, N, kind, american, 1.0)


# Registre utilisé par les expériences : (libellé, fonction) ---------------------
def _tri(S0, K, T, r, sigma, N, kind="call", american=False):
    return trinomial_price(S0, K, T, r, sigma, N, kind, american, LAMBDA_BOYLE)


METHODS = {
    "crr": ("CRR", binomial_price),
    "richardson": ("CRR + Richardson", crr_richardson),
    "bbs": ("BBS", bbs_price),
    "bbsr": ("BBSR", bbsr_price),
    "lr": ("Leisen-Reimer", lr_price),
    "trinomial": ("Trinomial (λ=√1,5)", _tri),
    "tri_centre": ("Trinomial centré", trinomial_centered_price),
}
