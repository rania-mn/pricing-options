"""Modèle trinomial de Boyle : multiplicateurs (u, 1, d) avec u d = 1.

Probabilités risque-neutres imposées par le moment d'ordre 2 (voir rapport) :
    p0 = 1 - 1/lam^2,  p+ + p- = 1/lam^2,  p+ u + p0 + p- d = R.
lam = 1 redonne exactement le binomial CRR.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, xlogy

from .common import payoff


@dataclass(frozen=True)
class TriParams:
    """Paramètres d'un arbre trinomial : u, d, R et (pu, p0, pd)."""
    u: float
    d: float
    R: float
    pu: float
    p0: float
    pd: float


def boyle_params(r, sigma, T, N, lam=1.0):
    """Paramètres de Boyle : u = e^{lam sigma sqrt(dt)}, d = 1/u, R = e^{r dt}."""
    if lam < 1:
        raise ValueError("lambda doit être >= 1")
    dt = T / N
    u = np.exp(lam * sigma * np.sqrt(dt))
    d, R, s = 1.0 / u, np.exp(r * dt), 1.0 / lam**2
    pu = (R - 1 + s * (1 - d)) / (u - d)
    pd = (s * (u - 1) - (R - 1)) / (u - d)
    p0 = 1.0 - s
    if min(pu, p0, pd) < 0:
        raise ValueError("Probabilité négative : augmenter N ou diminuer lambda")
    return TriParams(float(u), float(d), float(R), float(pu), float(p0), float(pd))


def trinomial_value(S0, K, N, p, kind="call", american=False):
    """Récurrence rétrograde : indice i <-> k = i - n (k = hausses - baisses)."""
    k = np.arange(-N, N + 1)
    S = S0 * p.u**k                                   # prix terminaux
    V = payoff(S, K, kind)
    for _ in range(N):
        V = (p.pu * V[2:] + p.p0 * V[1:-1] + p.pd * V[:-2]) / p.R
        S = S[1:-1]                                   # niveau précédent
        if american:
            V = np.maximum(V, payoff(S, K, kind))
    return float(V[0])


def trinomial_price(S0, K, T, r, sigma, N, kind="call", american=False, lam=1.0):
    """Prix trinomial de Boyle à N périodes."""
    return trinomial_value(S0, K, N, boyle_params(r, sigma, T, N, lam), kind, american)


def trinomial_closed_form(S0, K, T, r, sigma, N, kind="call", lam=1.0):
    """Formule multinomiale : R^-N somme N!/(i!k!(N-i-k)!) pu^i pd^k p0^(N-i-k) f(S0 u^(i-k))."""
    p = boyle_params(r, sigma, T, N, lam)
    i = np.arange(N + 1)[:, None]                     # nombre de hausses
    k = np.arange(N + 1)[None, :]                     # nombre de baisses
    m = np.clip(N - i - k, 0, None)                   # nombre de stagnations
    logw = (gammaln(N + 1) - gammaln(i + 1) - gammaln(k + 1) - gammaln(m + 1)
            + xlogy(i, p.pu) + xlogy(k, p.pd) + xlogy(m, p.p0))
    w = np.where(N - i - k >= 0, np.exp(logw), 0.0)   # états admissibles seulement
    ST = S0 * p.u ** (i - k)
    return float(np.sum(payoff(ST, K, kind) * w) / p.R**N)


def trinomial_tree(S0, K, T, r, sigma, N, kind="call", american=False, lam=1.0):
    """Arbre complet (petit N) : S[n][i], V[n][i], exercice optimal (i <-> k = i - n)."""
    p = boyle_params(r, sigma, T, N, lam)
    S = [S0 * p.u ** np.arange(-n, n + 1) for n in range(N + 1)]
    V, EX = [None] * (N + 1), [None] * (N + 1)
    V[N] = payoff(S[N], K, kind)
    EX[N] = V[N] > 0
    for n in range(N - 1, -1, -1):
        cont = (p.pu * V[n + 1][2:] + p.p0 * V[n + 1][1:-1] + p.pd * V[n + 1][:-2]) / p.R
        ex = payoff(S[n], K, kind)
        V[n] = np.maximum(cont, ex) if american else cont
        EX[n] = (ex > cont + 1e-12) if american else np.zeros(2 * n + 1, dtype=bool)
    return dict(S=S, V=V, exercise=EX, params=p)


# --- Incomplétude : famille de probabilités risque-neutres (lemme du rapport) ---

def risk_neutral_eps_max(u, d, R):
    """Borne supérieure de p0 = eps pour que p+ et p- restent > 0."""
    return min((R - d) / (1 - d), (u - R) / (u - 1))


def probs_from_eps(u, d, R, eps):
    """(p+, p0, p-) risque-neutres pour p0 = eps : p+ u + eps + p- d = R."""
    pu = (R - d - eps * (1 - d)) / (u - d)
    pd = (u - R - eps * (u - 1)) / (u - d)
    return pu, eps, pd
