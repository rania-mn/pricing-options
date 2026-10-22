"""
Projet de Calcul Stochastique - 2eme annee CI FID ENSA-UIZ
Titre : Schema d'Euler Stochastique et Machine Learning
        pour le Pricing d'Options Europeennes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

# Repertoire de sauvegarde des figures
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

# Parametres de reference
S0_DEF    = 100.0   # prix initial de l'actif
K_DEF     = 100.0   # strike
T_DEF     = 1.0     # maturite (annees)
R_DEF     = 0.05    # taux sans risque
SIGMA_DEF = 0.20    # volatilite
N_DEF     = 50      # nombre de pas d'Euler
M_DEF     = 50_000  # nombre de trajectoires Monte Carlo


# =============================================================================
# 1. FORMULE DE BLACK-SCHOLES
# =============================================================================

def black_scholes_call(S0, K, T, r, sigma):
    """
    Prix exact d'un Call europeen (Black-Scholes) :
        C = S0 * N(d1) - K * exp(-rT) * N(d2)
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S0, K, T, r, sigma):
    """
    Prix du Put via la parite Call-Put : P = C - S0 + K * exp(-rT)
    """
    C = black_scholes_call(S0, K, T, r, sigma)
    return C - S0 + K * np.exp(-r * T)

# =============================================================================
# 2. SCHEMA D'EULER STOCHASTIQUE
# =============================================================================

def euler_schema(S0, r, sigma, T, N, M, seed=42):
    """
    Simule M trajectoires de l'EDS de Black-Scholes par le schema d'Euler.
    EDS : dX_t = sigma * X_t * dW_t + r * X_t * dt,  X_0 = S0

    Schema : X_{k+1} = X_k * (1 + sigma * sqrt(dt) * G_k + r * dt), G_k ~ N(0,1)

    Retour :
    X_T     : array(M,)      valeurs terminales
    chemins : array(N+1, M)  trajectoires completes
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N           # pas de temps
    sqrt_dt = np.sqrt(dt)

    # Toutes les trajectoires demarrent a S0
    X = np.full(M, float(S0))
    chemins = [X.copy()]

    for _ in range(N):

        # Increments gaussiens
        G = np.random.randn(M)

        # Mise a jour d'Euler
        X = X + sigma * X * sqrt_dt * G + r * X * dt

        # Correction de positivite
        X = np.maximum(X, 0.0)

        chemins.append(X.copy())

    return X, np.array(chemins)


def simulation_exacte_XT(S0, r, sigma, T, M, seed=42):
    """
    Simule M valeurs de X_T par la solution exacte (sans discretisation) :
        X_T = S0 * exp(sigma * W_T + (r - sigma^2/2) * T)
    """
    if seed is not None:
        np.random.seed(seed)
    W_T = np.sqrt(T) * np.random.randn(M)
    return S0 * np.exp(sigma * W_T + (r - 0.5 * sigma**2) * T)


# =============================================================================
# 3. ESTIMATION MONTE CARLO
# =============================================================================

def estimer_prix_call(S0, K, T, r, sigma, M, N=None, seed=42):
    """
    Estime le prix du Call par Monte Carlo :
        C_hat = exp(-rT) * moyenne( (X_T - K)+ )

    N fourni -> schema d'Euler ; N=None -> simulation exacte.
    Retour : (prix estime, ecart-type de l'estimateur)
    """
    if N is not None:
        X_T, _ = euler_schema(S0, r, sigma, T, N, M, seed=seed)
    else:
        X_T = simulation_exacte_XT(S0, r, sigma, T, M, seed=seed)

    payoffs = np.maximum(X_T - K, 0.0)
    prix    = np.exp(-r * T) * np.mean(payoffs)
    se      = np.exp(-r * T) * np.std(payoffs) / np.sqrt(M)
    return prix, se


# =============================================================================
# 4. MACHINE LEARNING (pipeline Euler -> donnees -> modele)
# =============================================================================

def generer_donnees_euler_ml(K, T, r, sigma, N, M_par_point, S_min=60.0, S_max=140.0, n_points=50, seed=0):
    """
    Genere le jeu d'entrainement ML via le schema d'Euler.

    Retour
    S_grille      : valeurs de S0 (features)
    prix_euler_mc : prix estimes par MC (cibles bruitees)
    prix_bs_exact : prix exacts Black-Scholes (comparaison)
    """
    S_grille    = np.linspace(S_min, S_max, n_points)
    prix_mc     = []
    prix_bs     = []
    rng         = np.random.default_rng(seed)

    for S0_i in S_grille:
        # Simulation Euler pour ce S0_i
        local_seed = int(rng.integers(0, 99999))
        X_T, _ = euler_schema(S0_i, r, sigma, T, N, M_par_point, seed=local_seed)

        # Estimation MC du prix
        payoffs = np.maximum(X_T - K, 0.0)
        prix_mc.append(np.exp(-r * T) * np.mean(payoffs))

        # Reference Black-Scholes
        prix_bs.append(black_scholes_call(S0_i, K, T, r, sigma))

    return S_grille, np.array(prix_mc), np.array(prix_bs)

def entrainer_regression_poly(S_train, y_train, degre=6):
    """
    Regression polynomiale de degre `degre` (moindres carres) :
        v(S0) = a_0 + a_1*S0 + ... + a_d*S0^d
    Retour : Pipeline scikit-learn entraine
    """
    modele = Pipeline([
        ('poly', PolynomialFeatures(degree=degre, include_bias=True)),
        ('reg',  LinearRegression())
    ])
    modele.fit(S_train.reshape(-1, 1), y_train)
    return modele


def entrainer_mlp(S_train, y_train, couches=(64, 32)):
    """
    Reseau MLP : entree S0 -> 2 couches cachees (64, 32, ReLU) -> sortie (prix).
    Entrainement par Adam (minimisation de la MSE), entrees normalisees.
    Retour : Pipeline scikit-learn entraine
    """
    modele = Pipeline([
        ('norm', StandardScaler()),          # normalisation des entrees
        ('mlp',  MLPRegressor(
            hidden_layer_sizes=couches,
            activation='relu',
            solver='adam',
            max_iter=2000,
            learning_rate_init=0.001,
            tol=1e-6,
            random_state=42
        ))
    ])
    modele.fit(S_train.reshape(-1, 1), y_train)
    return modele


def predire_prix(modele, S0_valeurs):
    """
    Prediction du prix pour tout nouveau S0, sans resimulation.
    Retour : array de prix predits
    """
    S0_arr = np.atleast_1d(S0_valeurs).reshape(-1, 1)
    return modele.predict(S0_arr)


# =============================================================================
# 5. FIGURES
# =============================================================================

def fig1_trajectoires(S0, r, sigma, T, K, n_traj=8, seed=7):
    """
    Figure 1 : trajectoires d'Euler vs solution exacte (N=5 et N=50),
    avec le meme mouvement brownien pour les deux courbes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, N in enumerate([5, 50]):
        ax  = axes[idx]
        dt  = T / N
        t   = np.linspace(0, T, N + 1)
        rng = np.random.default_rng(seed)

        for i in range(n_traj):
            # Increments browniens partages
            G  = rng.standard_normal(N)
            dW = np.sqrt(dt) * G

            # Mouvement brownien cumule
            W = np.concatenate([[0], np.cumsum(dW)])

            # Solution exacte
            X_exact = S0 * np.exp(sigma * W + (r - 0.5 * sigma**2) * t)

            # Schema d'Euler
            X_euler    = np.zeros(N + 1)
            X_euler[0] = S0
            for k in range(N):
                X_euler[k + 1] = max(X_euler[k] * (1 + sigma * dW[k] + r * dt), 0)

            c = plt.cm.tab10(i / 10)
            ax.plot(t, X_exact, '--', color=c, alpha=0.50, lw=1.2)
            ax.plot(t, X_euler, '-',  color=c, alpha=0.90, lw=1.5)

        ax.axhline(K, color='red', ls=':', lw=1.8, label=f'Strike K = {K}')
        ax.set_xlabel('Temps $t$', fontsize=11)
        ax.set_ylabel('$X_t$', fontsize=11)
        ax.set_title(f'$N = {N}$ pas  (- Euler,  -- Exact)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Trajectoires du Schema d'Euler vs Solution Exacte",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig2_convergence(S0, K, T, r, sigma,
                     N_vals=None, M_fort=20_000, M_faib=80_000, seed=42):
    """
    Figure 2 : vitesses de convergence du schema d'Euler.
    Erreur forte (trajectoire par trajectoire) : ordre 1/2.
    Erreur faible (sur le prix) : ordre 1.
    """
    if N_vals is None:
        N_vals = np.array([2, 5, 10, 20, 50, 100, 200])

    # Erreur forte
    errs_fort = []
    for N in N_vals:
        np.random.seed(seed)
        dt      = T / N
        sqrt_dt = np.sqrt(dt)
        G_all   = np.random.randn(N, M_fort)

        W_T     = sqrt_dt * G_all.sum(axis=0)
        X_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        # Schema d'Euler sans correction de positivite
        X = np.full(M_fort, float(S0))
        for k in range(N):
            X = X + r * X * dt + sigma * X * sqrt_dt * G_all[k]

        errs_fort.append(np.sqrt(np.mean((X_exact - X) ** 2)))

    # Erreur faible
    errs_faib = []
    for N in N_vals:
        np.random.seed(seed)
        dt      = T / N
        sqrt_dt = np.sqrt(dt)
        G_all   = np.random.randn(N, M_faib)

        W_T     = sqrt_dt * G_all.sum(axis=0)
        X_exact = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

        # Schema d'Euler sans correction de positivite
        X = np.full(M_faib, float(S0))
        for k in range(N):
            X = X + r * X * dt + sigma * X * sqrt_dt * G_all[k]

        # Meme BM : variance reduite
        diff = np.exp(-r * T) * (np.maximum(X - K, 0) - np.maximum(X_exact - K, 0))
        errs_faib.append(abs(np.mean(diff)))

    errs_fort = np.array(errs_fort)
    errs_faib = np.array(errs_faib)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Convergence forte
    ax = axes[0]
    ax.loglog(N_vals, errs_fort, 'bo-', ms=7, lw=2, label='Erreur forte $L^2$')
    C1 = errs_fort[0] * np.sqrt(N_vals[0])
    ax.loglog(N_vals, C1 / np.sqrt(N_vals), 'r--', lw=2,
              label=r'$C/\sqrt{N}$ (pente $-1/2$)')
    ax.set_xlabel('Nombre de pas $N$', fontsize=11)
    ax.set_ylabel('Erreur forte', fontsize=11)
    ax.set_title(r'Convergence forte | ordre $N^{-1/2}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Convergence faible
    ax = axes[1]
    ax.loglog(N_vals, errs_faib, 'gs-', ms=7, lw=2, label='Erreur faible')
    C2 = errs_faib[0] * N_vals[0]
    ax.loglog(N_vals, C2 / N_vals, 'r--', lw=2,
              label=r'$C/N$ (pente $-1$)')
    ax.set_xlabel('Nombre de pas $N$', fontsize=11)
    ax.set_ylabel('Erreur faible', fontsize=11)
    ax.set_title(r'Convergence faible |ordre $N^{-1}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle("Vitesses de Convergence du Schema d'Euler (echelle log-log)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

def fig3_machine_learning(S_grille, prix_mc_train, modele_poly, modele_mlp,
                          K, T, r, sigma):
    """
    Figure 3 : apprentissage ML de la fonction de prix.
    Gauche : donnees, courbe BS, regression polynomiale et MLP.
    Droite : erreurs relatives (%) vs Black-Scholes.
    """
    S_test  = np.linspace(S_grille.min() - 5, S_grille.max() + 5, 300)
    y_bs    = np.array([black_scholes_call(s, K, T, r, sigma) for s in S_test])
    y_poly  = predire_prix(modele_poly, S_test)
    y_mlp   = predire_prix(modele_mlp,  S_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Courbes de prix
    ax = axes[0]
    ax.scatter(S_grille, prix_mc_train,
               c='steelblue', s=28, zorder=5, alpha=0.85,
               label='Donnees Euler + MC (entrainement)')
    ax.plot(S_test, y_bs,   'k-',   lw=2.5, label='Black-Scholes (exact)')
    ax.plot(S_test, y_poly, 'r--',  lw=2.0, label='Regression poly. (deg. 6)')
    ax.plot(S_test, y_mlp,  'g-.',  lw=2.0, label='Reseau MLP [64, 32]')
    ax.set_xlabel('Prix initial $S_0$', fontsize=11)
    ax.set_ylabel('Prix du Call', fontsize=11)
    ax.set_title('Apprentissage de la fonction de prix', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Erreurs relatives
    ax2 = axes[1]
    denom    = np.maximum(y_bs, 1e-8)
    err_poly = np.abs(y_poly - y_bs) / denom * 100
    err_mlp  = np.abs(y_mlp  - y_bs) / denom * 100
    ax2.plot(S_test, err_poly, 'r-', lw=2, label='Erreur poly. (%)')
    ax2.plot(S_test, err_mlp,  'g-', lw=2, label='Erreur MLP (%)')
    ax2.set_xlabel('Prix initial $S_0$', fontsize=11)
    ax2.set_ylabel('Erreur relative (%)', fontsize=11)
    ax2.set_title('Erreur relative vs Black-Scholes', fontsize=11)
    ax2.set_ylim(0, 15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Machine Learning : Euler => Donnees => Modele => Prediction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig4_comparaison(S0, K, T, r, sigma, M, N, modele_poly, modele_mlp):
    """
    Figure 4 : comparaison finale des methodes de pricing.
    Gauche : prix par methode. Droite : erreur absolue vs Black-Scholes.
    """
    bs = black_scholes_call(S0, K, T, r, sigma)

    # Prix par chaque methode
    p_mc_exact,  _ = estimer_prix_call(S0, K, T, r, sigma, M, N=None, seed=0)
    p_euler_mc,  _ = estimer_prix_call(S0, K, T, r, sigma, M, N=N,    seed=0)
    p_poly          = float(predire_prix(modele_poly, np.array([S0]))[0])
    p_mlp           = float(predire_prix(modele_mlp,  np.array([S0]))[0])

    methodes  = ['BS (exact)', 'MC exact', f'Euler+MC\n(N={N})',
                 'Euler+Poly', 'Euler+MLP']
    prix      = [bs, p_mc_exact, p_euler_mc, p_poly, p_mlp]
    erreurs   = [0, abs(p_mc_exact-bs), abs(p_euler_mc-bs),
                 abs(p_poly-bs), abs(p_mlp-bs)]
    couleurs  = ['#1A4D8F', '#5B9BD5', '#70AD47', '#ED7D31', '#FFC000']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Barplot des prix
    ax = axes[0]
    bars = ax.bar(methodes, prix, color=couleurs, alpha=0.88, edgecolor='white',
                  linewidth=1.5)
    ax.axhline(bs, color='black', lw=2, ls='--', label=f'BS = {bs:.4f}')
    for bar, p in zip(bars, prix):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{p:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Prix du Call', fontsize=11)
    ax.set_title(f'Prix par methode  ($S_0={S0}$, $K={K}$)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Erreurs absolues
    ax2 = axes[1]
    methodes_approx = methodes[1:]
    erreurs_approx  = erreurs[1:]
    couleurs_approx = couleurs[1:]
    bars2 = ax2.bar(methodes_approx, erreurs_approx,
                    color=couleurs_approx, alpha=0.88,
                    edgecolor='white', linewidth=1.5)
    for bar, e in zip(bars2, erreurs_approx):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                 f'{e:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylabel('|Prix - Black-Scholes|', fontsize=11)
    ax2.set_title('Erreur absolue vs Black-Scholes', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Comparaison Finale des Methodes de Pricing',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# 6. PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("  Schema d'Euler + Machine Learning - Pricing d'Options Europeennes")
    print("=" * 70)

    S0, K, T, r, sigma = S0_DEF, K_DEF, T_DEF, R_DEF, SIGMA_DEF
    N,  M              = N_DEF, M_DEF

    # Reference analytique
    bs_call = black_scholes_call(S0, K, T, r, sigma)
    bs_put  = black_scholes_put(S0, K, T, r, sigma)
    print(f"\nParametres : S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Call BS (exact) : {bs_call:.6f}")
    print(f"Put  BS (exact) : {bs_put:.6f}")
    parite = bs_call - bs_put - S0 + K * np.exp(-r * T)
    print(f"Parite C-P-S0+K*e^(-rT) = {parite:.2e}  (approx 0, OK)")

    # Generation des donnees ML via Euler
    print(f"\n{'-'*70}")
    print("ETAPE 1 : Generation des donnees d'entrainement ML (Euler + MC)...")
    t0 = time.time()
    S_grille, prix_mc_train, prix_bs_train = generer_donnees_euler_ml(
        K, T, r, sigma, N=50, M_par_point=5000, n_points=50
    )
    print(f"  -> {len(S_grille)} points generes en {time.time() - t0:.1f} s")

    # Entrainement des modeles ML
    print("\nETAPE 2 : Entrainement des modeles ML...")
    modele_poly = entrainer_regression_poly(S_grille, prix_mc_train, degre=6)
    modele_mlp  = entrainer_mlp(S_grille, prix_mc_train, couches=(64, 32))
    print("  -> Regression polynomiale (deg. 6) : OK")
    print("  -> Reseau de neurones MLP [64, 32] : OK")

    # Tableau comparatif
    print(f"\n{'-'*70}")
    print("ETAPE 3 : Comparaison de toutes les methodes de pricing")
    print(f"{'-'*70}")
    print(f"  {'Methode':<38} {'Prix':>10}  {'|Erreur|':>10}")
    print(f"  {'-'*60}")

    methodes_prix = {
        'Black-Scholes (exact)':           (bs_call, None),
        'Monte Carlo (simulation exacte)': estimer_prix_call(S0, K, T, r, sigma, M=M, seed=0),
        f'Euler (N={N}) + Monte Carlo':    estimer_prix_call(S0, K, T, r, sigma, M=M, N=N, seed=0),
        'Euler + Regression polynomiale':  (float(predire_prix(modele_poly, np.array([S0]))[0]), None),
        'Euler + Reseau MLP [64,32]':      (float(predire_prix(modele_mlp,  np.array([S0]))[0]), None),
    }

    for methode, (prix, se) in methodes_prix.items():
        err = abs(prix - bs_call) if methode != 'Black-Scholes (exact)' else 0.0
        ic  = f"  +/-{se:.4f}" if se is not None else ""
        print(f"  {methode:<38} {prix:>10.6f}  {err:>10.6f}{ic}")

    # Generation et sauvegarde des figures
    print(f"\n{'-'*70}")
    print("Generation des figures...")

    f1 = fig1_trajectoires(S0, r, sigma, T, K)
    f1.savefig(FIGURES_DIR / 'fig1_trajectoires.jpg', dpi=200, bbox_inches='tight')
    plt.close(f1)
    print("  Fig 1 - Trajectoires Euler vs Exacte : OK")

    print("  Fig 2 - Convergences forte/faible (patience ~20s)...")
    f2 = fig2_convergence(S0, K, T, r, sigma)
    f2.savefig(FIGURES_DIR / 'fig2_convergence.jpg', dpi=200, bbox_inches='tight')
    plt.close(f2)
    print("  Fig 2 - Convergences forte/faible : OK")

    f3 = fig3_machine_learning(S_grille, prix_mc_train, modele_poly, modele_mlp,
                                K, T, r, sigma)
    f3.savefig(FIGURES_DIR / 'fig3_machine_learning.jpg', dpi=200, bbox_inches='tight')
    plt.close(f3)
    print("  Fig 3 - Machine Learning (donnees + modeles) : OK")

    f4 = fig4_comparaison(S0, K, T, r, sigma, M, N, modele_poly, modele_mlp)
    f4.savefig(FIGURES_DIR / 'fig4_comparaison.jpg', dpi=200, bbox_inches='tight')
    plt.close(f4)
    print("  Fig 4 - Comparaison finale des methodes : OK")

    print(f"\nFigures sauvegardees dans : {FIGURES_DIR.resolve()}/")
    print("=" * 70)