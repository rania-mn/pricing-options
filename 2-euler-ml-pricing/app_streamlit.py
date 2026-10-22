"""
=============================================================================
Interface Streamlit : Pricing d'Options Européennes
Schéma d'Euler + Machine Learning
Lancer avec :  streamlit run app_streamlit.py
=============================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Import du module principal
sys.path.insert(0, str(Path(__file__).parent))
from euler_ml_pricing import (
    black_scholes_call,
    estimer_prix_call,
    generer_donnees_euler_ml,
    entrainer_regression_poly,
    entrainer_mlp,
    predire_prix,
    fig1_trajectoires,
    fig3_machine_learning,
    fig4_comparaison,
)

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Pricing d'Options Europeennes",
    layout="wide",
    initial_sidebar_state="expanded",
)
# =============================================================================
# CSS 
# =============================================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #FFFFFF; }
[data-testid="stMain"]             { background-color: #FFFFFF; }
[data-testid="stSidebar"]          { background-color: #EBF4FC; }
[data-testid="stSidebar"] *        { color: #000000 !important; }
html, body, [class*="css"], p, span, div { color: #000000; }
h1, h2, h3, h4 { color: #1A3D6E !important; }
hr { border: 1.5px solid #1A3D6E; margin: 8px 0 16px 0; }

[data-testid="stTabs"] button {
    color: #1A3D6E; font-weight: 600; font-size: 0.95em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #1A3D6E !important;
}

.stButton > button {
    background-color: #1A3D6E; color: #FFFFFF !important;
    border-radius: 5px; font-size: 1.0em; font-weight: 700;
    padding: 10px 0; border: none; width: 100%;
}
.stButton > button:hover { background-color: #2563EB; }

.metric-card {
    background-color: #F0F7FF;
    border-left: 5px solid #1A3D6E;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
}
.metric-titre  { font-size: 0.82em; color: #1A3D6E; font-weight: 600; }
.metric-valeur { font-size: 1.55em; font-weight: 800; color: #000000; }
.metric-erreur { font-size: 0.80em; color: #555555; margin-top: 2px; }
.metric-ref    { font-size: 0.80em; color: #1A3D6E; font-weight: 600; margin-top: 2px; }

.info-box {
    background-color: #EBF4FC; border-left: 4px solid #2563EB;
    border-radius: 5px; padding: 12px 16px; margin: 10px 0;
    font-size: 0.95em; line-height: 1.7em;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def carte_metrique(titre, valeur, extra="", ref=False):
    classe = "metric-ref" if ref else "metric-erreur"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-titre">{titre}</div>
        <div class="metric-valeur">{valeur}</div>
        <div class="{classe}">{extra}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _generer_donnees(K, T, r, sigma, N, M_ml, n_points):
    return generer_donnees_euler_ml(
        K, T, r, sigma,
        N=N, M_par_point=M_ml,
        n_points=n_points,
        S_min=50.0, S_max=150.0,
        seed=42,
    )


@st.cache_data(show_spinner=False)
def _donnees_convergence(S0, K, T, r, sigma):
    """
    Cache le calcul des erreurs de convergence.

    Correction appliquee :
    - Forte : meme BM pour Euler et solution exacte (matrice G partagee)
    - Faible : difference pairee f(X_bar) - f(X_exact) sur les memes chemins
      => variance tres reduite => convergence visible avec M raisonnable
    """
    N_vals   = np.array([2, 5, 10, 20, 50, 100, 200])
    M_f, M_b = 20_000, 80_000
    seed     = 42

    # -- Erreur forte : meme BM pour Euler et exact --
    errs_f = []
    for N in N_vals:
        np.random.seed(seed)
        dt      = T / N
        sqrt_dt = np.sqrt(dt)
        G_all   = np.random.randn(N, M_f)
        W_T     = sqrt_dt * G_all.sum(axis=0)
        X_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        X = np.full(M_f, float(S0))
        for k in range(N):
            X = X + r * X * dt + sigma * X * sqrt_dt * G_all[k] 

        errs_f.append(np.sqrt(np.mean((X_exact - X) ** 2)))

    # -- Erreur faible : difference pairee sur les memes chemins --
    errs_b = []
    for N in N_vals:
        np.random.seed(seed)
        dt      = T / N
        sqrt_dt = np.sqrt(dt)
        G_all   = np.random.randn(N, M_b)
        W_T     = sqrt_dt * G_all.sum(axis=0)
        X_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        X = np.full(M_b, float(S0))
        for k in range(N):
            X = X + r * X * dt + sigma * X * sqrt_dt * G_all[k] 

        diff = np.exp(-r * T) * (np.maximum(X - K, 0) - np.maximum(X_exact - K, 0))
        errs_b.append(abs(np.mean(diff)))

    return N_vals, np.array(errs_f), np.array(errs_b)

def _fig_convergence(S0, K, T, r, sigma):
    N_vals, errs_f, errs_b = _donnees_convergence(S0, K, T, r, sigma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.loglog(N_vals, errs_f, 'bo-', ms=7, lw=2, label='Erreur forte $L^2$')
    C1 = errs_f[0] * np.sqrt(N_vals[0])
    ax.loglog(N_vals, C1 / np.sqrt(N_vals), 'r--', lw=2,
              label=r'$C / \sqrt{N}$  (pente $-1/2$)')
    ax.set_xlabel('Nombre de pas $N$', fontsize=11)
    ax.set_ylabel('Erreur forte', fontsize=11)
    ax.set_title(r'Convergence forte - ordre $N^{-1/2}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.loglog(N_vals, errs_b, 'gs-', ms=7, lw=2, label='Erreur faible')
    C2 = errs_b[0] * N_vals[0]
    ax.loglog(N_vals, C2 / N_vals, 'r--', lw=2,
              label=r'$C / N$  (pente $-1$)')
    ax.set_xlabel('Nombre de pas $N$', fontsize=11)
    ax.set_ylabel('Erreur faible', fontsize=11)
    ax.set_title(r'Convergence faible - ordre $N^{-1}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle("Vitesses de Convergence du Schema d'Euler  (echelle log-log)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# EN-TETE
# =============================================================================
st.markdown("""
<h1>Pricing des Options Européennes</h1>
<p style="font-size:1.05em; color:#333333;">
Schema d'Euler Stochastique et Machine Learning </p>
<hr>
""", unsafe_allow_html=True)


# =============================================================================
# BARRE LATERALE
# =============================================================================
with st.sidebar:
    st.markdown("## Parametres")

    st.markdown("**Modele Black-Scholes**")
    S0    = st.slider("Prix initial  S0",     50.0, 150.0, 100.0, 5.0)
    K     = st.slider("Strike  K",            50.0, 150.0, 100.0, 5.0)
    T     = st.slider("Maturité  T (ans)",     0.25,   2.0,   1.0, 0.25)
    r     = st.slider("Taux sans risque  r",   0.01,  0.10,  0.05, 0.01,
                      format="%.2f")
    sigma = st.slider("Volatilite  sigma",     0.10,  0.50,  0.20, 0.05,
                      format="%.2f")

    st.markdown("---")
    st.markdown("**Simulation numerique**")
    N = st.selectbox("Pas Euler  N", [10, 20, 50, 100, 200], index=2)
    M = st.selectbox("Simulations MC  M",
                     [5_000, 10_000, 50_000], index=1,
                     format_func=lambda x: f"{x:,}")

    st.markdown("---")
    st.markdown("**Entrainement ML**")
    n_pts = st.slider("Points d'entrainement", 20, 70, 40, 5)
    M_ml  = st.selectbox("Simulations par point",
                          [1_000, 3_000, 5_000], index=1,
                          format_func=lambda x: f"{x:,}")
    degre = st.selectbox("Degre polynomial", [4, 6, 8], index=1)

    st.markdown("---")
    lancer = st.button("Lancer la simulation")


# Apercu (toujours visible)
bs_preview = black_scholes_call(S0, K, T, r, sigma)
st.markdown(f"""
<div class="info-box">
<b>Parametres actuels :</b>
S0 = {S0} &nbsp;|&nbsp; K = {K} &nbsp;|&nbsp;
T = {T} an &nbsp;|&nbsp; r = {r*100:.0f} % &nbsp;|&nbsp; sigma = {sigma*100:.0f} %
&nbsp;&nbsp;&nbsp;&nbsp;

</div>
""", unsafe_allow_html=True)


# =============================================================================
# CONTENU PRINCIPAL
# =============================================================================
if not lancer:
    st.markdown("""
    <div style="text-align:center; margin-top:60px; color:#1A3D6E;">
        <h3>Ajustez les paramètres dans la barre de gauche,
        puis cliquez <em>Lancer la simulation</em>.</h3>
        <p style="color:#555555;">
       
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# 1. CALCULS
bs = black_scholes_call(S0, K, T, r, sigma)

with st.spinner("Generation des donnees Euler et entrainement ML..."):
    S_grille, prix_mc_train, prix_bs_train = _generer_donnees(
        K, T, r, sigma, N, M_ml, n_pts
    )
    modele_poly = entrainer_regression_poly(S_grille, prix_mc_train, degre=degre)
    modele_mlp  = entrainer_mlp(S_grille, prix_mc_train, couches=(64, 32))

with st.spinner("Estimation Monte Carlo..."):
    p_mc,    se_mc    = estimer_prix_call(S0, K, T, r, sigma, M, N=None, seed=0)
    p_euler, se_euler = estimer_prix_call(S0, K, T, r, sigma, M, N=N,    seed=0)
    p_poly            = float(predire_prix(modele_poly, np.array([S0]))[0])
    p_mlp             = float(predire_prix(modele_mlp,  np.array([S0]))[0])


# 2. CARTES METRIQUES 
st.markdown("## Prix du Call europeen par methode")

cols = st.columns(5)
with cols[0]:
    carte_metrique(
        "Black-Scholes (exact)",
        f"{bs:.4f}",
        "Reference analytique",
        ref=True,
    )
with cols[1]:
    carte_metrique(
        "Monte Carlo (exact)",
        f"{p_mc:.4f}",
        f"Erreur : {abs(p_mc - bs):.4f} &nbsp;|&nbsp; &plusmn;{se_mc:.4f}",
    )
with cols[2]:
    carte_metrique(
        f"Euler (N={N}) + MC",
        f"{p_euler:.4f}",
        f"Erreur : {abs(p_euler - bs):.4f} &nbsp;|&nbsp; &plusmn;{se_euler:.4f}",
    )
with cols[3]:
    carte_metrique(
        f"Euler + Poly (deg. {degre})",
        f"{p_poly:.4f}",
        f"Erreur : {abs(p_poly - bs):.4f}",
    )
with cols[4]:
    carte_metrique(
        "Euler + MLP [64, 32]",
        f"{p_mlp:.4f}",
        f"Erreur : {abs(p_mlp - bs):.4f}",
    )

st.markdown("<hr>", unsafe_allow_html=True)


# 3. ONGLETS
tab1, tab2= st.tabs([
   
    "Convergences",
    
    "Comparaison finale",
])


# Onglet 2 : Convergences 
with tab1:
    st.markdown("### Vitesses de Convergence du Schema d'Euler")
    st.markdown("""
    <div class="info-box">
    <b>Convergence forte</b> (Theoreme 1.5, Jourdain) :<br>
    &nbsp;&nbsp;&nbsp;
    &radic;E[|X<sub>T</sub> &minus; X&#773;<sub>T</sub>|<sup>2</sup>]
    &nbsp;&le;&nbsp; C / &radic;N &nbsp;&nbsp; (ordre 1/2)<br>
    <b>Convergence faible</b> (Theoreme 1.6, Jourdain) :<br>
    &nbsp;&nbsp;&nbsp;
    |E[f(X&#773;<sub>T</sub>)] &minus; E[f(X<sub>T</sub>)]|
    &nbsp;&le;&nbsp; C / N &nbsp;&nbsp; (ordre 1)<br>
    L'erreur sur le <em>prix</em> converge deux fois plus vite
    que l'erreur sur les <em>trajectoires</em>.
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Calcul des convergences (environ 10 secondes)..."):
        fig_c = _fig_convergence(S0, K, T, r, sigma)
    st.pyplot(fig_c, use_container_width=True)
    plt.close(fig_c)




# Onglet 4 : Comparaison finale
with tab2:
    st.markdown("### Comparaison Finale des Methodes de Pricing")
    fig_cmp = fig4_comparaison(
        S0, K, T, r, sigma, M, N, modele_poly, modele_mlp
    )
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    st.markdown("#### Tableau récapitulatif")
    donnees = {
        "Methode": [
            "Black-Scholes (exact)",
            "Monte Carlo (exact)",
            f"Euler (N={N}) + Monte Carlo",
            f"Euler + Regression poly. (deg. {degre})",
            "Euler + Reseau MLP [64, 32]",
        ],
        "Prix estime": [
            f"{bs:.6f}",
            f"{p_mc:.6f}",
            f"{p_euler:.6f}",
            f"{p_poly:.6f}",
            f"{p_mlp:.6f}",
        ],
        "Erreur absolue": [
            "---",
            f"{abs(p_mc    - bs):.6f}",
            f"{abs(p_euler - bs):.6f}",
            f"{abs(p_poly  - bs):.6f}",
            f"{abs(p_mlp   - bs):.6f}",
        ],
        "Erreur relative (%)": [
            "---",
            f"{abs(p_mc    - bs) / bs * 100:.3f}",
            f"{abs(p_euler - bs) / bs * 100:.3f}",
            f"{abs(p_poly  - bs) / bs * 100:.3f}",
            f"{abs(p_mlp   - bs) / bs * 100:.3f}",
        ],
        
    }
    st.dataframe(pd.DataFrame(donnees), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    <b>Lecture du tableau :</b><br>
    &bull; <em>BS exact</em> : formule analytique, sans erreur ni simulation.<br>
    &bull; <em>MC exact</em> : erreur statistique O(M<sup>&minus;1/2</sup>),
    sans erreur de discretisation.<br>
    &bull; <em>Euler + MC</em> : erreur mixte = discretisation O(N<sup>&minus;1</sup>)
    + statistique O(M<sup>&minus;1/2</sup>).<br>
    &bull; <em>Euler + ML</em> : entrainement couteux une seule fois,
    puis prediction en temps O(1) sans resimulation.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PIED DE PAGE
# =============================================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#888888; font-size:0.85em;">
Projet Calcul Stochastique pour la Finance - 2025-2026
</p>
""", unsafe_allow_html=True)