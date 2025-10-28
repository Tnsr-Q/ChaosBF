#!/usr/bin/env python3
"""
Edge-Band Router for ChaosBF

Auto-tags elites based on Lyapunov CI and routes emitters to surf the edge.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TaggedElite:
    """Elite with Lyapunov edge-band tag."""
    genome: str
    fitness: float
    lyapunov: float
    ci_low: float
    ci_high: float
    tag: str  # 'critical-stable', 'critical-chaotic', or 'marginal'
    descriptors: np.ndarray


def tag_elite_by_lyapunov(lyapunov: float, ci_low: float, ci_high: float) -> str:
    """
    Tag elite based on Lyapunov exponent confidence interval.
    
    Args:
        lyapunov: Estimated Lyapunov exponent
        ci_low: Lower bound of 95% CI
        ci_high: Upper bound of 95% CI
    
    Returns:
        Tag: 'critical-stable', 'critical-chaotic', or 'marginal'
    """
    if ci_high < 0:
        return 'critical-stable'  # Definitely stable (CI entirely below 0)
    elif ci_low > 0:
        return 'critical-chaotic'  # Definitely chaotic (CI entirely above 0)
    else:
        return 'marginal'  # CI straddles 0 - true edge of chaos!


class EdgeBandRouter:
    """
    Routes emitters to spawn offspring around marginal elites.
    
    Marginal elites (CI straddles 0) are at the true edge of chaos.
    Focusing emitters here explores the most interesting dynamics.
    """
    
    def __init__(self, marginal_weight: float = 2.0):
        """
        Initialize edge-band router.
        
        Args:
            marginal_weight: Weight multiplier for marginal elites
        """
        self.marginal_weight = marginal_weight
        self.stats = {
            'critical_stable': 0,
            'critical_chaotic': 0,
            'marginal': 0
        }
    
    def add_elite(
        self,
        genome: str,
        fitness: float,
        lyapunov: float,
        ci_low: float,
        ci_high: float,
        descriptors: np.ndarray
    ) -> TaggedElite:
        """
        Add elite and tag it based on Lyapunov CI.
        
        Returns:
            Tagged elite
        """
        tag = tag_elite_by_lyapunov(lyapunov, ci_low, ci_high)
        self.stats[tag.replace('-', '_')] += 1
        
        return TaggedElite(
            genome=genome,
            fitness=fitness,
            lyapunov=lyapunov,
            ci_low=ci_low,
            ci_high=ci_high,
            tag=tag,
            descriptors=descriptors
        )
    
    def select_parent(
        self,
        elites: List[TaggedElite],
        strategy: str = 'marginal_weighted'
    ) -> TaggedElite:
        """
        Select parent elite for mutation.
        
        Args:
            elites: List of tagged elites
            strategy: Selection strategy
                - 'marginal_weighted': Prefer marginal elites
                - 'marginal_only': Only select marginal elites
                - 'uniform': Uniform selection
        
        Returns:
            Selected elite
        """
        if not elites:
            raise ValueError("No elites available")
        
        if strategy == 'marginal_only':
            marginal = [e for e in elites if e.tag == 'marginal']
            if marginal:
                return np.random.choice(marginal)
            # Fallback to all elites if no marginal
            return np.random.choice(elites)
        
        elif strategy == 'marginal_weighted':
            # Weight marginal elites higher
            weights = []
            for elite in elites:
                if elite.tag == 'marginal':
                    weights.append(self.marginal_weight)
                else:
                    weights.append(1.0)
            
            weights = np.array(weights)
            weights /= weights.sum()
            
            return np.random.choice(elites, p=weights)
        
        else:  # uniform
            return np.random.choice(elites)
    
    def get_stats(self) -> Dict:
        """Get edge-band statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total': total,
            'marginal_fraction': self.stats['marginal'] / total if total > 0 else 0.0
        }
    
    def print_stats(self):
        """Print edge-band statistics."""
        stats = self.get_stats()
        print("="*60)
        print("Edge-Band Statistics")
        print("="*60)
        print(f"Critical-Stable:  {stats['critical_stable']:4d}")
        print(f"Critical-Chaotic: {stats['critical_chaotic']:4d}")
        print(f"Marginal (Edge):  {stats['marginal']:4d}")
        print(f"Total:            {stats['total']:4d}")
        print(f"Marginal Fraction: {stats['marginal_fraction']:.2%}")
        print()


def main():
    """Demo of edge-band tagging."""
    import sys
    sys.path.insert(0, '/home/ubuntu/chaosbf/src')
    from lyapunov import LyapunovEstimator
    
    # Example genomes
    genomes = [
        '++[>+<-].',
        ':{;}{?}^=.',
        '*=@=.#',
        '+[>+<-];.'
    ]
    
    router = EdgeBandRouter(marginal_weight=2.0)
    estimator = LyapunovEstimator()
    
    print("Estimating Lyapunov exponents and tagging elites...")
    print()
    
    tagged_elites = []
    
    for genome in genomes:
        # Estimate Lyapunov with bootstrap CI
        result = estimator.estimate_with_bootstrap(
            genome,
            n_bootstrap=50,  # Reduced for demo
            steps=1000,
            verbose=False
        )
        
        # Create tagged elite
        elite = router.add_elite(
            genome=genome,
            fitness=1.0,  # Placeholder
            lyapunov=result['lambda_lyap'],
            ci_low=result['ci_low'],
            ci_high=result['ci_high'],
            descriptors=np.random.randn(4)  # Placeholder
        )
        
        tagged_elites.append(elite)
        
        print(f"Genome: {genome}")
        print(f"  λ_lyap: {elite.lyapunov:.4f} [{elite.ci_low:.4f}, {elite.ci_high:.4f}]")
        print(f"  Tag: {elite.tag}")
        print()
    
    # Print statistics
    router.print_stats()
    
    # Demo selection
    print("Parent selection (marginal-weighted):")
    for _ in range(5):
        parent = router.select_parent(tagged_elites, strategy='marginal_weighted')
        print(f"  Selected: {parent.genome} (tag: {parent.tag})")


if __name__ == '__main__':
    main()

