"""
Pricing d'options européennes par la méthode de Longstaff-Schwartz modifiée
Utilise la régression pour approximer l'espérance conditionnelle

"""
import numpy as np
from typing import Tuple

class LongstaffSchwartzEuropean:
    """
    Algorithme de Longstaff-Schwartz adapté pour options européennes
    
    Principe: Utiliser la régression pour approximer l'espérance conditionnelle
    E[Payoff(S_T) | S_t] à différents instants t.
    
    Cette approche est pédagogique pour comprendre comment la régression
    peut approximer les valeurs futures conditionnelles.
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, n_steps: int):
        """
        Args:
            S0: Prix initial
            K: Strike
            T: Maturité
            r: Taux sans risque
            sigma: Volatilité
            n_steps: Nombre de pas de temps
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        
    def simulate_paths(self, n_paths: int, random_gen) -> np.ndarray:
        """
        Génère les trajectoires GBM
        Returns: Matrice (n_paths, n_steps+1) des prix
        """
        S = np.zeros((n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        
        # GBM : S(t+dt) = S(t) * exp((r - 0.5*σ²)*dt + σ*√dt*Z)
        for t in range(1, self.n_steps + 1):
            Z = random_gen.box_muller(n_paths)
            S[:, t] = S[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * Z
            )
        
        return S
    
    def regression(self, X: np.ndarray, Y: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Régression polynomiale
        E[Y|X] ≈ β₀ + β₁*X + β₂*X² + ... + βₙ*Xⁿ
        
        Args:
            X: Prix de l'actif (variable explicative)
            Y: Valeur future (variable à prédire)
            degree: Degré du polynôme
        
        Returns:
            (Valeurs prédites, Coefficients)
        """
        # Matrice de design: A = [1, X, X^2, ..., X^degree]
        A = np.column_stack([X**i for i in range(degree + 1)])
        
        try:
            # MCO: coeffs = (A^T A)^(-1) A^T Y
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            predictions = A @ coeffs
        except:
            # Si échec, retourner la moyenne
            coeffs = np.zeros(degree + 1)
            predictions = np.full_like(Y, np.mean(Y))
        
        return predictions, coeffs
    
    def price_with_regression(self, option_type: str, n_paths: int, 
                              random_gen, regression_step: int = None) -> dict:
        """
        Prix d'une option européenne avec analyse par régression
        
        ALGORITHME:
        1. Simuler N trajectoires complètes
        2. Calculer payoff à T
        3. Pour chaque instant t (backward):
           - Actualiser les payoffs futurs
           - Régresser: E[V_{t+1}|S_t] sur S_t
           - Comparer régression vs actualisation directe

            option_type: 'call' ou 'put'
            n_paths: Nombre de simulations
            random_gen: Générateur aléatoire
            regression_step: Step où analyser la régression (None = T-1)
            
        Returns:
            Dictionary avec prix, erreur, et détails de régression
        """
        # 1. Simuler trajectoires
        S = self.simulate_paths(n_paths, random_gen)
        
        # 2. Définir le payoff
        if option_type.lower() == 'call':
            payoff = lambda s: np.maximum(s - self.K, 0)
        else:
            payoff = lambda s: np.maximum(self.K - s, 0)
        
        # 3. Payoff à maturité
        V = payoff(S[:, -1])  # Valeur à T
        
        # 4. Stocker les résultats de régression
        regression_results = {}
        
        if regression_step is None:
            regression_step = self.n_steps - 1
        
        # 5. BACKWARD: actualiser et analyser
        for t in range(self.n_steps - 1, 0, -1):
            # Actualiser d'un pas
            V = V * np.exp(-self.r * self.dt)
            
            # À l'instant choisi, faire la régression
            if t == regression_step:
                # Régression: E[V_{t+1}|S_t]
                continuation_pred, coeffs = self.regression(S[:, t], V, degree=2)
                
                regression_results = {
                    'time_step': t,
                    'time': t * self.dt,
                    'spot_prices': S[:, t].copy(),
                    'actual_values': V.copy(),
                    'predicted_values': continuation_pred,
                    'coefficients': coeffs,
                    'R2': 1 - np.var(V - continuation_pred) / np.var(V)
                }
        
        # 6. Actualisation finale
        V = V * np.exp(-self.r * self.dt)
        
        # 7. Prix et erreur
        price = np.mean(V)
        std_error = np.std(V) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': V,
            'regression_analysis': regression_results,
            'paths': S
        }
    
    def analyze_conditional_expectation(self, option_type: str, n_paths: int, 
                                       random_gen, time_points: list = None) -> dict:
        """
        Analyse détaillée de l'espérance conditionnelle à plusieurs instants
            option_type: 'call' ou 'put'
            n_paths: Nombre de simulations
            random_gen: Générateur aléatoire
            time_points: Liste des pas de temps à analyser
            
        Returns:
            Dictionary avec analyses pour chaque instant
        """
        if time_points is None:
            # Par défaut: T/4, T/2, 3T/4
            time_points = [self.n_steps // 4, self.n_steps // 2, 
                          3 * self.n_steps // 4]
        
        # Simuler une seule fois
        S = self.simulate_paths(n_paths, random_gen)
        
        if option_type.lower() == 'call':
            payoff = lambda s: np.maximum(s - self.K, 0)
        else:
            payoff = lambda s: np.maximum(self.K - s, 0)
        
        V = payoff(S[:, -1])
        
        analyses = {}
        
        # Backward iteration
        for t in range(self.n_steps - 1, 0, -1):
            V = V * np.exp(-self.r * self.dt)
            
            if t in time_points:
                # Régression à cet instant
                continuation_pred, coeffs = self.regression(S[:, t], V, degree=2)
                
                analyses[t] = {
                    'time': t * self.dt,
                    'spot_prices': S[:, t].copy(),
                    'continuation_values': V.copy(),
                    'predicted_values': continuation_pred,
                    'coefficients': coeffs,
                    'R2': 1 - np.var(V - continuation_pred) / np.var(V),
                    'mean_spot': np.mean(S[:, t]),
                    'mean_value': np.mean(V)
                }
        
        V = V * np.exp(-self.r * self.dt)
        
        return {
            'price': np.mean(V),
            'std_error': np.std(V) / np.sqrt(n_paths),
            'time_analyses': analyses
        }