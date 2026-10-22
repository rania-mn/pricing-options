"""
Formules analytiques pour le pricing d'options européennes
Formule de Black-Scholes et calcul de la fonction de répartition
"""
import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    Formule de Black-Scholes pour options européennes
    
    Pour un call: C = S₀*N(d₁) - K*e^(-rT)*N(d₂)
    Pour un put:  P = K*e^(-rT)*N(-d₂) - S₀*N(-d₁)
    
    où:
    d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    N(x) = fonction de répartition de N(0,1)
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float):
        """
            S0: Prix initial de l'actif
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
    
    def _d1_d2(self):
        """
        Calcule d₁ et d₂
        Returns:
            Tuple (d1, d2)
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def price_call(self):
        """
        Prix exact d'un call européen
        C = S₀*N(d₁) - K*e^(-rT)*N(d₂)
        
        Returns:
            Prix du call
        """
        d1, d2 = self._d1_d2()
        
        # N(d) = CDF de la loi normale standard
        call_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
    def price_put(self):
        """
        Prix exact d'un put européen
        P = K*e^(-rT)*N(-d₂) - S₀*N(-d₁)
        Returns:
            Prix du put
        """
        d1, d2 = self._d1_d2()
        
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        
        return put_price
    
    def greeks_call(self):
        """
        Calcule les Greeks pour un call
        Returns:
            Dictionary avec Delta, Gamma, Vega, Theta, Rho
        """
        d1, d2 = self._d1_d2()
        sqrt_T = np.sqrt(self.T)
        
        # Delta: ∂C/∂S
        delta = norm.cdf(d1)
        
        # Gamma: ∂²C/∂S²
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * sqrt_T)
        
        # Vega: ∂C/∂σ
        vega = self.S0 * norm.pdf(d1) * sqrt_T
        
        # Theta: ∂C/∂T
        theta = (- (self.S0 * norm.pdf(d1) * self.sigma) / (2 * sqrt_T) 
                 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        
        # Rho: ∂C/∂r
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Vega par 1% de volatilité
            'theta': theta / 365,  # Theta par jour
            'rho': rho / 100  # Rho par 1% de taux
        }
    
    def greeks_put(self):
        """
        Calcule les Greeks pour un put
        Returns:
            Dictionary avec Delta, Gamma, Vega, Theta, Rho
        """
        d1, d2 = self._d1_d2()
        sqrt_T = np.sqrt(self.T)
        
        # Delta: ∂P/∂S
        delta = norm.cdf(d1) - 1
        
        # Gamma: ∂²P/∂S² (identique au call)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * sqrt_T)
        
        # Vega: ∂P/∂σ (identique au call)
        vega = self.S0 * norm.pdf(d1) * sqrt_T
        
        # Theta: ∂P/∂T
        theta = (- (self.S0 * norm.pdf(d1) * self.sigma) / (2 * sqrt_T) 
                 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        # Rho: ∂P/∂r
        rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,
            'theta': theta / 365,
            'rho': rho / 100
        }
    
    def put_call_parity_check(self):
        """
        Vérifie la parité put-call: C - P = S₀ - K*e^(-rT)
        
        Returns:
            Tuple (difference, should_be_zero)
        """
        call_price = self.price_call()
        put_price = self.price_put()
        
        lhs = call_price - put_price
        rhs = self.S0 - self.K * np.exp(-self.r * self.T)
        
        difference = abs(lhs - rhs)
        
        return difference, lhs, rhs