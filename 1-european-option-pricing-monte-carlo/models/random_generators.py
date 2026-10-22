"""
Générateurs de nombres aléatoires pour les simulations Monte Carlo
Implémente: Box-Muller, Méthode d'inversion, Méthode de rejet
"""
import numpy as np

class RandomGenerator:
    """Classe pour générer des nombres aléatoires selon différentes méthodes""" 
    def __init__(self, seed=None):
        """
        Initialise le générateur
            seed: Graine aléatoire pour la reproductibilité
        """
        if seed is not None:
            np.random.seed(seed)
    
    def box_muller(self, n):
        """
        Génère des variables aléatoires gaussiennes avec la méthode de Box-Muller
        Transforme 2 uniformes en 2 normales: Z1 = sqrt(-2*ln(U1))*cos(2*pi*U2)

            n: Nombre de variables à générer
        Returns:
            Array de n variables N(0,1)
        """
        # Générer n/2 paires de variables uniformes (arrondir au supérieur)
        n_pairs = int(np.ceil(n / 2))
        
        # Variables uniformes sur (0,1)
        U1 = np.random.uniform(0, 1, n_pairs)
        U2 = np.random.uniform(0, 1, n_pairs)
        
        # Transformation de Box-Muller
        R = np.sqrt(-2 * np.log(U1))
        Theta = 2 * np.pi * U2
        
        # Génération de deux variables gaussiennes indépendantes
        Z1 = R * np.cos(Theta)
        Z2 = R * np.sin(Theta)
        
        # Combiner et retourner exactement n valeurs
        Z = np.concatenate([Z1, Z2])
        return Z[:n]
    
    def inversion_method(self, n):
        """
        Méthode d'inversion pour générer des normales N(0,1)
        Utilise la fonction quantile (inverse de la CDF) de la loi normale
        Z = g^(-1)(U) où U ~ U(0,1) et g est la CDF de N(0,1)

            n: Nombre de variables à générer
        Returns:
            Array de n variables N(0,1)
        """
        from scipy.stats import norm
        
        # Générer des uniformes sur (0,1)
        U = np.random.uniform(0, 1, n)
        
        # Appliquer la fonction quantile (inverse CDF)
        # norm.ppf = percent point function = inverse CDF
        Z = norm.ppf(U)
        
        return Z
    
    def rejection_method(self, n):
        """
        Méthode de rejet pour générer des normales N(0,1)
        Utilise une enveloppe exponentielle: f(x) ≤ M*g(x)
        où f est la densité normale et g est une densité facile à simuler
            n: Nombre de variables à générer
        Returns:
            Array de n variables N(0,1)
        """
        samples = []
        
        # Densité normale standard
        def normal_pdf(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        
        # Loi instrumentale: exponentielle décalée
        # g(x) = lambda * exp(-lambda * |x|) / 2
        lam = 1.0
        
        def instrumental_pdf(x):
            return (lam / 2) * np.exp(-lam * np.abs(x))
        
        # Constante M telle que f(x) ≤ M*g(x) pour tout x
        # Pour lambda = 1, M ≈ 1.315
        M = np.sqrt(2 * np.e / np.pi)
        
        while len(samples) < n:
            # Générer Y ~ g (exponentielle bilatérale)
            # Méthode: sign * Exp(lambda)
            sign = np.random.choice([-1, 1])
            Y = sign * np.random.exponential(1/lam)
            
            # Générer U ~ U(0,1)
            U = np.random.uniform(0, 1)
            
            # Test d'acceptation: U ≤ f(Y)/(M*g(Y))
            acceptance_ratio = normal_pdf(Y) / (M * instrumental_pdf(Y))
            
            if U <= acceptance_ratio:
                samples.append(Y)
        
        return np.array(samples[:n])
    
    def compare_methods(self, n, method_names=['box_muller', 'inversion', 'rejection']):
        """
        Compare les différentes méthodes de génération

            n: Nombre de variables à générer
            method_names: Liste des méthodes à comparer
        Returns:
            Dictionary avec les résultats de chaque méthode
        """
        import time
        results = {}
        
        for method in method_names:
            start_time = time.time()
            
            if method == 'box_muller':
                samples = self.box_muller(n)
            elif method == 'inversion':
                samples = self.inversion_method(n)
            elif method == 'rejection':
                samples = self.rejection_method(n)
            else:
                continue
            
            end_time = time.time()
            
            results[method] = {
                'samples': samples,
                'mean': np.mean(samples),
                'std': np.std(samples),
                'time': end_time - start_time
            }
        
        return results