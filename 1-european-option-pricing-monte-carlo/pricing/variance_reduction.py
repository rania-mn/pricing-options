"""
Techniques de réduction de variance pour les simulations Monte Carlo
"""
import numpy as np
from models.brownian import GeometricBrownianMotion

class VarianceReduction:
    """Classe pour appliquer les techniques de réduction de variance"""
    
    def __init__(self, S0, K, T, r, sigma):
        """
        Initialise les paramètres
            S0: Prix initial
            K: Strike
            T: Maturité
            r: Taux sans risque
            sigma: Volatilité
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def antithetic_variates_call(self, n_simulations, random_gen):
        """
        Technique des variables antithétiques pour un call
        Principe: Pour chaque Z, on utilise aussi -Z, ce qui réduit la variance
            n_simulations: Nombre de simulations (doit être pair)
            random_gen: Générateur de nombres aléatoires
        Returns:
            Tuple (prix, std_error, payoffs)
        """
        # S'assurer que n est pair
        n = n_simulations // 2  #si n_simulations impair on elimine la derniere
        
        # Générer les variables gaussiennes
        Z = random_gen.box_muller(n)
        
        # Prix terminaux avec Z
        S_T_pos = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        # Prix terminaux avec -Z (antithétiques)
        S_T_neg = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T - self.sigma * np.sqrt(self.T) * Z)
        
        # Payoffs
        payoff_pos = np.maximum(S_T_pos - self.K, 0)
        payoff_neg = np.maximum(S_T_neg - self.K, 0)
        
        # Combiner les payoffs
        all_payoffs = np.concatenate([payoff_pos, payoff_neg])
        
        # Actualiser
        discounted_payoffs = np.exp(-self.r * self.T) * all_payoffs
        
        # Prix et erreur
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(2 * n)
        
        return price, std_error, discounted_payoffs
    
    def antithetic_variates_put(self, n_simulations, random_gen):
        """
        Technique des variables antithétiques pour un put
            n_simulations: Nombre de simulations
            random_gen: Générateur de nombres aléatoires
        Returns:
            Tuple (prix, std_error, payoffs)
        """
        n = n_simulations // 2
        
        Z = random_gen.box_muller(n)
        
        # Prix terminaux
        S_T_pos = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        S_T_neg = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T - self.sigma * np.sqrt(self.T) * Z)
        
        # Payoffs
        payoff_pos = np.maximum(self.K - S_T_pos, 0)
        payoff_neg = np.maximum(self.K - S_T_neg, 0)
        
        all_payoffs = np.concatenate([payoff_pos, payoff_neg])
        discounted_payoffs = np.exp(-self.r * self.T) * all_payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(2 * n)
        
        return price, std_error, discounted_payoffs
    
    def control_variates_call(self, n_simulations, random_gen):
        """
        Technique des variables de contrôle pour un call
        Principe: Utiliser une variable corrélée dont on connaît l'espérance (ici, S_T)
            n_simulations: Nombre de simulations
            random_gen: Générateur de nombres aléatoires
        Returns:
            Tuple (prix, std_error, payoffs_bruts, payoffs_contrôle)
        """
        # Simuler les prix terminaux
        gbm = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T, 1)
        S_T = gbm.simulate_terminal(n_simulations, random_gen)
        
        # Payoffs du call
        payoffs = np.maximum(S_T - self.K, 0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Variable de contrôle: S_T
        # On sait que E[S_T] = S0 * exp(r*T)
        expected_S_T = self.S0 * np.exp(self.r * self.T)
        
        # Calculer la covariance et la variance
        cov = np.cov(discounted_payoffs, S_T)[0, 1]
        var_S_T = np.var(S_T)
        
        # Coefficient optimal de la variable de contrôle
        c = cov / var_S_T if var_S_T > 0 else 0
        
        # Estimateur avec variable de contrôle
        # Y_cv = Y - c*(X - E[X])
        controlled_payoffs = discounted_payoffs - c * (S_T - expected_S_T) #payoffs corrigés avec une variance plus faible
        
        # Prix et erreur
        price = np.mean(controlled_payoffs)
        std_error = np.std(controlled_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error, discounted_payoffs, controlled_payoffs
    
    def control_variates_put(self, n_simulations, random_gen):
        """
        Technique des variables de contrôle pour un put
            n_simulations: Nombre de simulations
            random_gen: Générateur de nombres aléatoires
        Returns:
            Tuple (prix, std_error, payoffs_bruts, payoffs_contrôle)
        """
        gbm = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T, 1)
        S_T = gbm.simulate_terminal(n_simulations, random_gen)
        
        payoffs = np.maximum(self.K - S_T, 0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        expected_S_T = self.S0 * np.exp(self.r * self.T)
        
        cov = np.cov(discounted_payoffs, S_T)[0, 1]
        var_S_T = np.var(S_T)
        c = cov / var_S_T if var_S_T > 0 else 0
        
        controlled_payoffs = discounted_payoffs - c * (S_T - expected_S_T)
        
        price = np.mean(controlled_payoffs)
        std_error = np.std(controlled_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error, discounted_payoffs, controlled_payoffs
    
    def compare_variance_reduction(self, option_type, n_simulations, random_gen, mc_price, mc_std_err, mc_payoffs):
        """
        Compare l'efficacité des différentes techniques de réduction de variance
            option_type: 'call' ou 'put'
            n_simulations: Nombre de simulations
            random_gen: Générateur de nombres aléatoires
        Returns:
            Dictionary avec les résultats de chaque méthode
        """
        results = {}

        # Monte Carlo standard
        # Utiliser les valeurs déjà calculées
        results['standard'] = {
                'price': mc_price,
                'std_error': mc_std_err,
                'variance': np.var(mc_payoffs)
            }

        if option_type == 'call':
            
            # Variables antithétiques
            price_av, std_err_av, payoffs_av = self.antithetic_variates_call(n_simulations, random_gen)
            results['antithetic'] = {
                'price': price_av,
                'std_error': std_err_av,
                'variance': np.var(payoffs_av)
            }
            
            # Variables de contrôle
            price_cv, std_err_cv, _, payoffs_cv = self.control_variates_call(n_simulations, random_gen)
            results['control'] = {
                'price': price_cv,
                'std_error': std_err_cv,
                'variance': np.var(payoffs_cv)
            }
        else:  # put     
            # Variables antithétiques
            price_av, std_err_av, payoffs_av = self.antithetic_variates_put(n_simulations, random_gen)
            results['antithetic'] = {
                'price': price_av,
                'std_error': std_err_av,
                'variance': np.var(payoffs_av)
            }
            
            # Variables de contrôle
            price_cv, std_err_cv, _, payoffs_cv = self.control_variates_put(n_simulations, random_gen)
            results['control'] = {
                'price': price_cv,
                'std_error': std_err_cv,
                'variance': np.var(payoffs_cv)
            }
        
        return results