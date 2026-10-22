"""
Simulations Monte Carlo pour le pricing d'option europÃ©enne
"""
import numpy as np
from models.brownian import GeometricBrownianMotion

class MonteCarloPricer:
    """Classe pour pricer les options par simulation Monte Carlo"""
    
    def __init__(self, S0, K, T, r, sigma):
        """
        Initialise le pricer Monte Carlo
            S0: Prix initial de l'actif
            K: Strike
            T: MaturitÃ©
            r: Taux sans risque (pour actualiser les payoffs)
            sigma: Volatilité
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def price_call(self, n_simulations, random_gen):
        """
        Calcule le prix d'un call europÃ©en par Monte Carlo
            n_simulations: Nombre de simulations
            random_gen: GÃ©nÃ©rateur de nombres alÃ©atoires
        Returns:
            Tuple (prix, std_error, payoffs)
        """
        # CrÃ©er un GBM avec drift = r (mesure risque-neutre)
        gbm = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T, 1) 
        
        # Simuler les prix terminaux
        S_T = gbm.simulate_terminal(n_simulations, random_gen)
        
        # Calculer les payoffs: max(S_T - K, 0)
        payoffs = np.maximum(S_T - self.K, 0)
        
        # Actualiser les payoffs
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Prix = moyenne des payoffs actualisÃ©s
        price = np.mean(discounted_payoffs)
        
        # Erreur standard = std / sqrt(n)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error, discounted_payoffs
    
    def price_put(self, n_simulations, random_gen):
        """
        Calcule le prix d'un put europÃ©en par Monte Carlo
            n_simulations: Nombre de simulations
            random_gen: GÃ©nÃ©rateur de nombres alÃ©atoires
        Returns:
            Tuple (prix, std_error, payoffs)
        """
        # CrÃ©er un GBM avec drift = r
        gbm = GeometricBrownianMotion(self.S0, self.r, self.sigma, self.T, 1)
        
        # Simuler les prix terminaux
        S_T = gbm.simulate_terminal(n_simulations, random_gen)
        
        # Calculer les payoffs: max(K - S_T, 0)
        payoffs = np.maximum(self.K - S_T, 0)
        
        # Actualiser les payoffs
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Prix et erreur standard
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error, discounted_payoffs
    
    def convergence_analysis(self, option_type, n_max, n_steps, random_gen):
        """
        Analyse la convergence du prix en fonction du nombre de simulations
            option_type: 'call' ou 'put'
            n_max: Nombre maximum de simulations
            n_steps: Nombre de points pour l'analyse
            random_gen: GÃ©nÃ©rateur de nombres alÃ©atoires
        Returns:
            Tuple (n_sims, prices, std_errors, confidence_intervals)
        """
        # Points de simulation (Ã©chelle logarithmique)
        n_sims = np.logspace(2, np.log10(n_max), n_steps).astype(int) 
        prices = []
        std_errors = []
        conf_intervals = []
        
        for n in n_sims:
            if option_type == 'call':
                price, std_err, _ = self.price_call(n, random_gen)
            else:
                price, std_err, _ = self.price_put(n, random_gen)
            
            prices.append(price)
            std_errors.append(std_err)
            
            # Intervalle de confiance Ã  95% : [prix - 1.96*std_err, prix + 1.96*std_err]
            conf_intervals.append((price - 1.96 * std_err, price + 1.96 * std_err))
        
        return n_sims, prices, std_errors, conf_intervals
    
    def calculate_confidence_interval(self, payoffs, confidence_level=0.95):
        """
        Calcule l'intervalle de confiance pour l'estimateur Monte Carlo
            payoffs: Array des payoffs actualisÃ©s
            confidence_level: Niveau de confiance (par dÃ©faut 95%)
        Returns:
            Tuple (lower_bound, upper_bound)
        """
        from scipy.stats import norm
        #Calculer la moyenne et lâ€™Ã©cart-type
        mean = np.mean(payoffs)
        std = np.std(payoffs)
        n = len(payoffs)
        
        # Quantile de la loi normale
        z = norm.ppf((1 + confidence_level) / 2)
        
        # Intervalle de confiance
        margin = z * std / np.sqrt(n)
        lower = mean - margin
        upper = mean + margin
        
        return lower, upper