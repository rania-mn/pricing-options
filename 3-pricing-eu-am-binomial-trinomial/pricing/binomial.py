"""Modèle binomial (Cox-Ross-Rubinstein) : options européennes et américaines.

Notations du rapport : u = mu (hausse), d (baisse), R = 1 + r (par période),
q = (R - d) / (u - d) (probabilité risque-neutre de hausse).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import binom

from .black_scholes import bs_price
from .common import payoff


@dataclass(frozen=True)
class BinParams:
    """Paramètres d'un arbre binomial : u, d, R et q."""
    u: float
    d: float
    R: float
    q: float

    @property
    def q_tilde(self):
        """q' = q u / R, probabilité de hausse sous la mesure associée au numéraire S."""
        return self.q * self.u / self.R


def make_params(u, d, R):
    """Construit (u, d, R, q) en vérifiant l'absence d'arbitrage d < R < u."""
    if not d < R < u:
        raise ValueError("Arbitrage possible : il faut d < R < u")
    return BinParams(float(u), float(d), float(R), float((R - d) / (u - d)))


def crr_params(r, sigma, T, N):
    """Paramétrage CRR : u = e^{sigma sqrt(dt)}, d = 1/u, R = e^{r dt}."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    return make_params(u, 1.0 / u, np.exp(r * dt))


def binomial_value(S0, K, N, p, kind="call", american=False, bs_last=None):
    """Récurrence rétrograde sur l'arbre recombinant (vectorisée).

    V[j] = valeur au nœud ayant j hausses. Pas d'arbre complet en mémoire.
    bs_last = (r, sigma, dt) : remplace la dernière période par la formule de
    Black-Scholes (lissage BBS de Broadie-Detemple).
    """
    j = np.arange(N + 1)
    S = S0 * p.u**j * p.d ** (N - j)          # prix terminaux
    V = payoff(S, K, kind)
    m = N                                      # niveau courant
    if bs_last is not None and N >= 1:
        r, sigma, dt = bs_last
        S = S[:-1] / p.d                       # prix à la date N-1
        V = bs_price(S, K, dt, r, sigma, kind)
        if american:
            V = np.maximum(V, payoff(S, K, kind))
        m = N - 1
    while m > 0:
        V = (p.q * V[1:] + (1 - p.q) * V[:-1]) / p.R   # valeur de continuation
        S = S[:-1] / p.d                               # niveau m-1 (j hausses)
        if american:
            V = np.maximum(V, payoff(S, K, kind))      # exercice anticipé
        m -= 1
    return float(V[0])


def binomial_price(S0, K, T, r, sigma, N, kind="call", american=False):
    """Prix binomial CRR à N périodes."""
    return binomial_value(S0, K, N, crr_params(r, sigma, T, N), kind, american)


def binomial_closed_form(S0, K, T, r, sigma, N, kind="call"):
    """Formule du théorème : R^-N somme_j f(S0 u^j d^(N-j)) C(N,j) q^j (1-q)^(N-j)."""
    p = crr_params(r, sigma, T, N)
    j = np.arange(N + 1)
    logw = (gammaln(N + 1) - gammaln(j + 1) - gammaln(N - j + 1)
            + j * np.log(p.q) + (N - j) * np.log(1 - p.q))   # log-poids (stabilité)
    ST = S0 * p.u**j * p.d ** (N - j)
    return float(np.sum(payoff(ST, K, kind) * np.exp(logw)) / p.R**N)


def binomial_two_probability_form(S0, K, T, r, sigma, N):
    """Forme CRR du call : S0 Q~(S_N > K) - K R^-N Q(S_N > K) (lemme de changement de proba)."""
    p = crr_params(r, sigma, T, N)
    j = np.arange(N + 1)
    itm = S0 * p.u**j * p.d ** (N - j) > K               # états dans la monnaie
    pq = binom.pmf(j, N, p.q)
    pqt = binom.pmf(j, N, p.q_tilde)
    return float(S0 * pqt[itm].sum() - K * p.R**-N * pq[itm].sum())


def binomial_tree(S0, K, T, r, sigma, N, kind="call", american=False, params=None):
    """Arbre complet (petit N) : prix S, valeurs V, exercice optimal, couverture Delta.

    Delta[n][j] = (V[n+1][j+1] - V[n+1][j]) / (S[n][j] (u - d)) : quantité d'actif de
    la stratégie de réplication (question 6 de l'énoncé CRR).
    """
    p = params or crr_params(r, sigma, T, N)
    S, V, EX, DELTA = [None] * (N + 1), [None] * (N + 1), [None] * (N + 1), [None] * N
    for n in range(N + 1):
        j = np.arange(n + 1)
        S[n] = S0 * p.u**j * p.d ** (n - j)
    V[N] = payoff(S[N], K, kind)
    EX[N] = V[N] > 0
    for n in range(N - 1, -1, -1):
        cont = (p.q * V[n + 1][1:] + (1 - p.q) * V[n + 1][:-1]) / p.R
        ex = payoff(S[n], K, kind)
        V[n] = np.maximum(cont, ex) if american else cont
        EX[n] = (ex > cont + 1e-12) if american else np.zeros(n + 1, dtype=bool)
        DELTA[n] = (V[n + 1][1:] - V[n + 1][:-1]) / (S[n] * (p.u - p.d))
    return dict(S=S, V=V, exercise=EX, delta=DELTA, params=p)


def exercise_boundary(S0, K, T, r, sigma, N, kind="put", smooth=True):
    """Frontière d'exercice anticipé S*(t) d'une option américaine (nan si jamais optimal).

    Pour un put : entre le plus grand nœud où l'exercice est optimal et le nœud suivant.
    La grille des prix est décalée d'une demi-maille entre dates paires et impaires :
    smooth=True moyenne deux niveaux consécutifs pour supprimer ce zigzag.
    """
    tree = binomial_tree(S0, K, T, r, sigma, N, kind, american=True)
    p = tree["params"]
    t = np.arange(N + 1) * T / N
    star = np.full(N + 1, np.nan)
    ratio = p.u / p.d                                   # rapport entre deux nœuds voisins
    for n in range(N + 1):
        S, ex = tree["S"][n], tree["exercise"][n]
        if ex.any():
            if kind == "put":
                s_ex = S[ex].max()
                star[n] = s_ex * (1 + ratio) / 2        # milieu [dernier nœud d'exercice, suivant]
            else:
                s_ex = S[ex].min()
                star[n] = s_ex * (1 + 1 / ratio) / 2
    if smooth:
        star[1:] = np.where(np.isnan(star[1:]) | np.isnan(star[:-1]), star[1:],
                            0.5 * (star[1:] + star[:-1]))
    star[-1] = K                                        # à l'échéance : exercice si S < K
    return t, star
