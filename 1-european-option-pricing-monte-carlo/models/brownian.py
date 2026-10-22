"""
Modélisation des mouvements browniens standard et géométrique
"""
import numpy as np

class BrownianMotion:
    """Classe pour simuler le mouvement brownien standard""" 
    def __init__(self, T, n_steps):
        """
        Initialise le mouvement brownien
            T: Horizon temporel
            n_steps: Nombre de pas de temps
        """
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps  
    
    def simulate(self, n_paths, random_gen):
        """
        Simule n_paths trajectoires de mouvement brownien
            n_paths: Nombre de trajectoires
            random_gen: Générateur de nombres aléatoires
        Returns:
            Array de dimension (n_paths, n_steps+1) contenant les trajectoires
        """
        # Initialisation
        W = np.zeros((n_paths, self.n_steps + 1)) #tableau lignes=paths colonnes=t
        
        # Générer les incréments gaussiens
        for i in range(1, self.n_steps + 1): #pour colonne i=0 les W sont initialisé = 0
            # Incréments: dW = sqrt(dt) * Z où Z ~ N(0,1)
            dW = np.sqrt(self.dt) * random_gen.box_muller(n_paths)    #dW = incrément aléatoire
            W[:, i] = W[:, i-1] + dW  
        
        return W


class GeometricBrownianMotion:
    """Classe pour simuler le mouvement brownien géométrique"""
    def __init__(self, S0, mu, sigma, T, n_steps):
        """
        Initialise le mouvement brownien géométrique
            S0: Prix initial de l'actif
            mu: Drift (rendement espéré)
            sigma: Volatilité
            T: Horizon temporel
            n_steps: Nombre de pas de temps
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
    
    def simulate(self, n_paths, random_gen):
        """
        Simule n_paths trajectoires du prix de l'actif selon un GBM
        Formule: S(t) = S0 * exp((mu - sigma²/2)*t + sigma*W(t))
            n_paths: Nombre de trajectoires
            random_gen: Générateur de nombres aléatoires
        Returns:
            Array de dimension (n_paths, n_steps+1) contenant les prix
        """
        # Initialisation
        S = np.zeros((n_paths, self.n_steps + 1)) 
        S[:, 0] = self.S0  #premiere colonne  = (S0,..,S0)t
        
        # Simulation des trajectoires
        for i in range(1, self.n_steps + 1):  #parcourir les pas
            # Générer les incréments gaussiens
            Z = random_gen.box_muller(n_paths)
            
            # Formule d'Euler pour le GBM
            # S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
            S[:, i] = S[:, i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        
        return S
    
    def simulate_terminal(self, n_paths, random_gen):
        """
        Simule uniquement les prix terminaux S(T) de manière directe
        Plus efficace pour le pricing d'options européennes
            n_paths: Nombre de simulations
            random_gen: Générateur de nombres aléatoires
        Returns:
            Array de dimension (n_paths,) contenant les prix terminaux
        """
        # Générer les variables gaussiennes
        Z = random_gen.box_muller(n_paths)
        
        # Formule directe: S(T) = S0 * exp((mu - sigma²/2)*T + sigma*sqrt(T)*Z)
        S_T = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        return S_T