"""
Application de visualisation pour le pricing d'options européennes
Affiche les résultats: formules exactes, Monte Carlo, réduction de variance, Longstaff-Schwartz
"""
import sys
import os

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# Configuration matplotlib AVANT l'import de pyplot 
import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
from models.random_generators import RandomGenerator
from models.brownian import GeometricBrownianMotion
from pricing.analytical import BlackScholes
from pricing.monte_carlo import MonteCarloPricer
from pricing.variance_reduction import VarianceReduction
from pricing.longstaff_schwartz_european import LongstaffSchwartzEuropean

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paramètres de l'option
S0 = 100        # Prix initial
K = 100         # Strike
T = 1.0         # Maturité (1 an)
r = 0.05        # Taux sans risque (5%)
sigma = 0.2     # Volatilité (20%)
n_simulations = 50000

print("="*80)
print("PRICING D'OPTIONS EUROPEENNES - ANALYSE COMPLETE")
print("="*80)
print(f"\nParamètres:")
print(f"  S0 = {S0}, K = {K}, T = {T}, r = {r}, σ = {sigma}")
print(f"  Nombre de simulations: {n_simulations:,}")
print("="*80)

# Initialiser le générateur
random_gen = RandomGenerator(seed=42)


# 1. FORMULE EXACTE (BLACK-SCHOLES)
#==============================================================================
print("\n" + "="*80)
print("1. FORMULE EXACTE - BLACK-SCHOLES")
print("="*80)

bs = BlackScholes(S0, K, T, r, sigma)
bs_call = bs.price_call()
bs_put = bs.price_put()

print(f"\nCall européen: {bs_call:.4f}")
print(f"Put européen:  {bs_put:.4f}")

# Vérification Put-Call Parity
diff, lhs, rhs = bs.put_call_parity_check()
print(f"\nVérification Put-Call Parity:")
print(f"  C - P = {lhs:.6f}")
print(f"  S₀ - K*e^(-rT) = {rhs:.6f}")
print(f"  Différence: {diff:.2e}")

# Greeks
greeks_call = bs.greeks_call()
greeks_put = bs.greeks_put()

print(f"\nGreeks du Call:")
for greek, value in greeks_call.items():
    print(f"  {greek.capitalize()}: {value:.6f}")


# 2. COMPARAISON DES GENERATEURS ALEATOIRES
#==============================================================================
print("\n" + "="*80)
print("2. COMPARAISON DES GENERATEURS ALEATOIRES")
print("="*80)

n_test = 10000
comparison = random_gen.compare_methods(n_test)

print(f"\nGénération de {n_test} variables N(0,1):\n")
print(f"{'Méthode':<20} {'Moyenne':<12} {'Ecart-type':<12} {'Temps (s)':<12}")
print("-" * 60)
for method, results in comparison.items():
    print(f"{method:<20} {results['mean']:<12.6f} {results['std']:<12.6f} {results['time']:<12.6f}")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Comparaison des générateurs aléatoires', fontsize=14, fontweight='bold')

for idx, (method, results) in enumerate(comparison.items()):
    axes[idx].hist(results['samples'], bins=50, alpha=0.7, density=True, edgecolor='black')
    x = np.linspace(-4, 4, 100)
    axes[idx].plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2), 'r-', linewidth=2, label='N(0,1) théorique')
    axes[idx].set_title(f'{method}\nμ={results["mean"]:.3f}, σ={results["std"]:.3f}')
    axes[idx].set_xlabel('Valeur')
    axes[idx].set_ylabel('Densité')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'resultats_generateurs.png'), dpi=300, bbox_inches='tight')
print(f"\n Figure sauvegardée: resultats_generateurs.png")


# 3. MONTE CARLO STANDARD
#==============================================================================
print("\n" + "="*80)
print("3. MONTE CARLO STANDARD")
print("="*80)

mc = MonteCarloPricer(S0, K, T, r, sigma)

# Pricing Call
mc_call, mc_call_err, mc_call_payoffs = mc.price_call(n_simulations, random_gen)
print(f"\nCall Monte Carlo: {mc_call:.4f} ± {mc_call_err:.4f}")
print(f"  Erreur vs BS: {abs(mc_call - bs_call):.4f}")
print(f"  Intervalle 95%: [{mc_call - 1.96*mc_call_err:.4f}, {mc_call + 1.96*mc_call_err:.4f}]")

# Pricing Put
mc_put, mc_put_err, mc_put_payoffs = mc.price_put(n_simulations, random_gen)
print(f"\nPut Monte Carlo:  {mc_put:.4f} ± {mc_put_err:.4f}")
print(f"  Erreur vs BS: {abs(mc_put - bs_put):.4f}")
print(f"  Intervalle 95%: [{mc_put - 1.96*mc_put_err:.4f}, {mc_put + 1.96*mc_put_err:.4f}]")

# Analyse de convergence
print("\nAnalyse de convergence:")
n_sims_conv, prices_conv, errors_conv, conf_intervals_conv = mc.convergence_analysis(
    'call', n_max=n_simulations, n_steps=20, random_gen=random_gen
)

# Visualisation convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Convergence Monte Carlo - Call Européen', fontsize=14, fontweight='bold')

# Prix vs nombre de simulations
ax1.semilogx(n_sims_conv, prices_conv, 'b-', linewidth=2, label='Prix MC')
ax1.axhline(bs_call, color='r', linestyle='--', linewidth=2, label=f'BS exact = {bs_call:.4f}')
ax1.fill_between(n_sims_conv, 
                  [ci[0] for ci in conf_intervals_conv],
                  [ci[1] for ci in conf_intervals_conv],
                  alpha=0.3, label='IC 95%')
ax1.set_xlabel('Nombre de simulations')
ax1.set_ylabel('Prix du Call')
ax1.set_title('Convergence du prix')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Erreur standard vs nombre de simulations
ax2.loglog(n_sims_conv, errors_conv, 'g-', linewidth=2, label='Erreur standard')
# Théorique: erreur  1/√n
theoretical_error = errors_conv[0] * np.sqrt(n_sims_conv[0] / n_sims_conv)
ax2.loglog(n_sims_conv, theoretical_error, 'r--', linewidth=2, label='Théorique (1/√n)')
ax2.set_xlabel('Nombre de simulations')
ax2.set_ylabel('Erreur standard')
ax2.set_title('Décroissance de l\'erreur')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'convergence_monte_carlo.png'), dpi=300, bbox_inches='tight')
print(f"\n Figure sauvegardée: convergence_monte_carlo.png")


# 4. REDUCTION DE VARIANCE
#==============================================================================
print("\n" + "="*80)
print("4. TECHNIQUES DE REDUCTION DE VARIANCE")
print("="*80)

vr = VarianceReduction(S0, K, T, r, sigma)

# Call
print("\n--- CALL EUROPEEN ---")
results_call = vr.compare_variance_reduction('call', n_simulations, random_gen, mc_call, mc_call_err, mc_call_payoffs)

print(f"\n{'Méthode':<20} {'Prix':<12} {'Err. Std':<12} {'Variance':<12} {'Réduction':<12}")
print("-" * 75)

baseline_var = results_call['standard']['variance']
for method, res in results_call.items():
    reduction = (1 - res['variance'] / baseline_var) * 100 if method != 'standard' else 0
    print(f"{method:<20} {res['price']:<12.4f} {res['std_error']:<12.6f} {res['variance']:<12.6f} {reduction:<12.1f}%")

# Put
print("\n--- PUT EUROPEEN ---")
results_put = vr.compare_variance_reduction('put', n_simulations, random_gen, mc_put, mc_put_err, mc_put_payoffs)

print(f"\n{'Méthode':<20} {'Prix':<12} {'Err. Std':<12} {'Variance':<12} {'Réduction':<12}")
print("-" * 75)

baseline_var = results_put['standard']['variance']
for method, res in results_put.items():
    reduction = (1 - res['variance'] / baseline_var) * 100 if method != 'standard' else 0
    print(f"{method:<20} {res['price']:<12.4f} {res['std_error']:<12.6f} {res['variance']:<12.6f} {reduction:<12.1f}%")

# Visualisation réduction de variance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Réduction de Variance - Comparaison des Méthodes', fontsize=14, fontweight='bold')

option_types = ['call', 'put']
results_list = [results_call, results_put]

for idx, (opt_type, results) in enumerate(zip(option_types, results_list)):
    bs_price = bs_call if opt_type == 'call' else bs_put
    
    # Prix
    ax = axes[idx, 0]
    methods = list(results.keys())
    prices = [results[m]['price'] for m in methods]
    errors = [results[m]['std_error'] for m in methods]
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, prices, yerr=[1.96*e for e in errors], capsize=5, alpha=0.7, edgecolor='black')
    ax.axhline(bs_price, color='r', linestyle='--', linewidth=2, label=f'BS = {bs_price:.4f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Prix')
    ax.set_title(f'{opt_type.upper()} - Prix et IC 95%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Variance
    ax = axes[idx, 1]
    variances = [results[m]['variance'] for m in methods]
    colors = ['blue' if m == 'standard' else 'green' for m in methods]
    bars = ax.bar(x_pos, variances, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Variance')
    ax.set_title(f'{opt_type.upper()} - Variance des estimateurs')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'reduction_variance.png'), dpi=300, bbox_inches='tight')
print(f"\n Figure sauvegardée: reduction_variance.png")


# 5. LONGSTAFF-SCHWARTZ EUROPEEN (REGRESSION)
#==============================================================================
print("\n" + "="*80)
print("5. LONGSTAFF-SCHWARTZ EUROPEEN - ANALYSE PAR REGRESSION")
print("="*80)

ls = LongstaffSchwartzEuropean(S0, K, T, r, sigma, n_steps=50)

# Pricing avec analyse de régression
print("\nPricing avec analyse de régression:")
result_ls_call = ls.price_with_regression('call', n_simulations, random_gen, regression_step=25)
result_ls_put = ls.price_with_regression('put', n_simulations, random_gen, regression_step=25)

print(f"\nCall LS: {result_ls_call['price']:.4f} ± {result_ls_call['std_error']:.4f}")
print(f"  Erreur vs BS: {abs(result_ls_call['price'] - bs_call):.4f}")

print(f"\nPut LS:  {result_ls_put['price']:.4f} ± {result_ls_put['std_error']:.4f}")
print(f"  Erreur vs BS: {abs(result_ls_put['price'] - bs_put):.4f}")

# Analyse de régression
if result_ls_call['regression_analysis']:
    reg = result_ls_call['regression_analysis']
    print(f"\nAnalyse de régression à t = {reg['time']:.2f}:")
    print(f"  R² = {reg['R2']:.4f}")
    print(f"  Coefficients: {reg['coefficients']}")

# Analyse conditionnelle à plusieurs instants
print("\nAnalyse de l'espérance conditionnelle à plusieurs instants:")
analysis_call = ls.analyze_conditional_expectation('call', n_simulations, random_gen)

print(f"\nPrix final: {analysis_call['price']:.4f} ± {analysis_call['std_error']:.4f}")

# Visualisation Longstaff-Schwartz
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Longstaff-Schwartz Européen - Régression et Espérance Conditionnelle', 
             fontsize=14, fontweight='bold')

# Quelques trajectoires
ax1 = fig.add_subplot(gs[0, :])
n_paths_plot = 100
paths = result_ls_call['paths'][:n_paths_plot, :]
time_grid = np.linspace(0, T, ls.n_steps + 1)
for i in range(n_paths_plot):
    ax1.plot(time_grid, paths[i, :], alpha=0.3, linewidth=0.5)
ax1.axhline(K, color='r', linestyle='--', linewidth=2, label=f'Strike K={K}')
ax1.set_xlabel('Temps')
ax1.set_ylabel('Prix de l\'actif')
ax1.set_title(f'Trajectoires simulées (n={n_paths_plot})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Analyses de régression à différents instants
time_analyses = analysis_call['time_analyses']
plot_idx = 1
for t_step, analysis in sorted(time_analyses.items())[:4]:
    if plot_idx > 4:
        break
    
    row = 1 + (plot_idx - 1) // 2
    col = (plot_idx - 1) % 2
    ax = fig.add_subplot(gs[row, col])
    
    # Scatter: valeurs réelles vs prédites
    spot = analysis['spot_prices']
    actual = analysis['continuation_values']
    predicted = analysis['predicted_values']
    
    # Trier pour la courbe de régression
    sort_idx = np.argsort(spot)
    
    ax.scatter(spot, actual, alpha=0.3, s=1, label='Valeurs réelles')
    ax.plot(spot[sort_idx], predicted[sort_idx], 'r-', linewidth=2, label='Régression (degré 2)')
    ax.set_xlabel('Prix spot S(t)')
    ax.set_ylabel('Valeur de continuation E[V|S]')
    ax.set_title(f't = {analysis["time"]:.2f} | R² = {analysis["R2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1

plt.savefig(os.path.join(os.path.dirname(__file__), 'longstaff_schwartz.png'), dpi=300, bbox_inches='tight')
print(f"\n Figure sauvegardée: longstaff_schwartz.png")


# 6. TABLEAU RECAPITULATIF
#==============================================================================
print("\n" + "="*80)
print("6. TABLEAU RECAPITULATIF - COMPARAISON TOUTES METHODES")
print("="*80)

print("\n--- CALL EUROPEEN ---")
print(f"{'Méthode':<30} {'Prix':<12} {'Erreur':<12} {'Erreur vs BS':<15}")
print("-" * 75)
print(f"{'Black-Scholes (exact)':<30} {bs_call:<12.4f} {'-':<12} {0.0:<15.4f}")
print(f"{'Monte Carlo standard':<30} {mc_call:<12.4f} {mc_call_err:<12.6f} {abs(mc_call - bs_call):<15.4f}")
print(f"{'MC - Variables antithétiques':<30} {results_call['antithetic']['price']:<12.4f} {results_call['antithetic']['std_error']:<12.6f} {abs(results_call['antithetic']['price'] - bs_call):<15.4f}")
print(f"{'MC - Variables de contrôle':<30} {results_call['control']['price']:<12.4f} {results_call['control']['std_error']:<12.6f} {abs(results_call['control']['price'] - bs_call):<15.4f}")
print(f"{'Longstaff-Schwartz européen':<30} {result_ls_call['price']:<12.4f} {result_ls_call['std_error']:<12.6f} {abs(result_ls_call['price'] - bs_call):<15.4f}")

print("\n--- PUT EUROPEEN ---")
print(f"{'Méthode':<30} {'Prix':<12} {'Erreur':<12} {'Erreur vs BS':<15}")
print("-" * 75)
print(f"{'Black-Scholes (exact)':<30} {bs_put:<12.4f} {'-':<12} {0.0:<15.4f}")
print(f"{'Monte Carlo standard':<30} {mc_put:<12.4f} {mc_put_err:<12.6f} {abs(mc_put - bs_put):<15.4f}")
print(f"{'MC - Variables antithétiques':<30} {results_put['antithetic']['price']:<12.4f} {results_put['antithetic']['std_error']:<12.6f} {abs(results_put['antithetic']['price'] - bs_put):<15.4f}")
print(f"{'MC - Variables de contrôle':<30} {results_put['control']['price']:<12.4f} {results_put['control']['std_error']:<12.6f} {abs(results_put['control']['price'] - bs_put):<15.4f}")
print(f"{'Longstaff-Schwartz européen':<30} {result_ls_put['price']:<12.4f} {result_ls_put['std_error']:<12.6f} {abs(result_ls_put['price'] - bs_put):<15.4f}")


# 7. VISUALISATION FINALE - COMPARAISON GLOBALE
#==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparaison Globale des Méthodes de Pricing', fontsize=14, fontweight='bold')

methods = ['BS', 'MC std', 'MC antith.', 'MC control', 'L-S']
call_prices = [
    bs_call,
    mc_call,
    results_call['antithetic']['price'],
    results_call['control']['price'],
    result_ls_call['price']
]
call_errors = [
    0,
    mc_call_err,
    results_call['antithetic']['std_error'],
    results_call['control']['std_error'],
    result_ls_call['std_error']
]

put_prices = [
    bs_put,
    mc_put,
    results_put['antithetic']['price'],
    results_put['control']['price'],
    result_ls_put['price']
]
put_errors = [
    0,
    mc_put_err,
    results_put['antithetic']['std_error'],
    results_put['control']['std_error'],
    result_ls_put['std_error']
]

x_pos = np.arange(len(methods))

# Call
axes[0].bar(x_pos, call_prices, yerr=[1.96*e for e in call_errors], 
            capsize=5, alpha=0.7, edgecolor='black', color='steelblue')
axes[0].axhline(bs_call, color='r', linestyle='--', linewidth=2, label=f'BS = {bs_call:.4f}')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(methods, rotation=15)
axes[0].set_ylabel('Prix')
axes[0].set_title('Call Européen')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Put
axes[1].bar(x_pos, put_prices, yerr=[1.96*e for e in put_errors], 
            capsize=5, alpha=0.7, edgecolor='black', color='coral')
axes[1].axhline(bs_put, color='r', linestyle='--', linewidth=2, label=f'BS = {bs_put:.4f}')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(methods, rotation=15)
axes[1].set_ylabel('Prix')
axes[1].set_title('Put Européen')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'comparaison_globale.png'), dpi=300, bbox_inches='tight')
print(f"\nFigure sauvegardée: comparaison_globale.png")

plt.show()

print("\nTOUTES LES FIGURES SONT SAUVEGARDEES DANS LE DOSSIER app")
print("="*80)