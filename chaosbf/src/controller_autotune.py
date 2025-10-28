#!/usr/bin/env python3
"""
CMA-ES Controller Autotune for ChaosBF PID Gains

Automatically tunes PID controller gains (kp, ki, kd) and variance shaping gain (γ)
to optimize criticality control performance.

Objective: Minimize deviation from λ = 1.0 while maintaining low volatility.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3
import numpy as np
from typing import Dict, Tuple, List
import json


class PIDAutotuner:
    """
    CMA-ES-based autotune for PID controller gains.
    
    Optimizes (kp, ki, kd, gamma) to minimize:
        objective = |λ - 1.0| + 0.1 * volatility
    """
    
    def __init__(
        self,
        genome: str,
        E: float = 200.0,
        T_initial: float = 0.5,
        steps_per_eval: int = 2000,
        seed: int = 42
    ):
        """
        Initialize autotuner.
        
        Args:
            genome: Test genome for tuning
            E: Initial energy
            T_initial: Initial temperature
            steps_per_eval: Steps per evaluation
            seed: RNG seed
        """
        self.genome = genome
        self.E = E
        self.T_initial = T_initial
        self.steps_per_eval = steps_per_eval
        self.seed = seed
        
        # CMA-ES parameters
        self.n_params = 4  # kp, ki, kd, gamma
        self.population_size = 8
        self.sigma = 0.3  # Initial step size
        
        # Search bounds
        self.bounds = {
            'kp': (0.01, 1.0),
            'ki': (0.0, 0.5),
            'kd': (0.0, 0.5),
            'gamma': (0.01, 0.5)
        }
        
        # Results
        self.best_params = None
        self.best_objective = float('inf')
        self.history = []
    
    def evaluate(self, params: np.ndarray) -> float:
        """
        Evaluate PID gains.
        
        Args:
            params: [kp, ki, kd, gamma]
        
        Returns:
            Objective value (lower is better)
        """
        kp, ki, kd, gamma = params
        
        # Clip to bounds
        kp = np.clip(kp, *self.bounds['kp'])
        ki = np.clip(ki, *self.bounds['ki'])
        kd = np.clip(kd, *self.bounds['kd'])
        gamma = np.clip(gamma, *self.bounds['gamma'])
        
        # Run ChaosBF with these gains
        cbf = ChaosBFv3(
            self.genome,
            E=self.E,
            T=self.T_initial,
            seed=self.seed,
            use_pid=True,
            use_variance_shaping=True,
            verbose=False
        )
        
        # Override PID gains
        cbf.pid_kp = kp
        cbf.pid_ki = ki
        cbf.pid_kd = kd
        cbf.variance_shaping_gamma = gamma
        
        # Run
        cbf.run(steps=self.steps_per_eval)
        
        # Compute objective
        stats = cbf.get_stats()
        lambda_dev = abs(stats['branching_factor'] - 1.0)
        volatility = stats['volatility']
        
        # Objective: minimize deviation + small penalty for volatility
        objective = lambda_dev + 0.1 * volatility
        
        return objective
    
    def run(self, generations: int = 20, verbose: bool = True) -> Dict:
        """
        Run CMA-ES optimization.
        
        Args:
            generations: Number of CMA-ES generations
            verbose: Print progress
        
        Returns:
            Best parameters and statistics
        """
        if verbose:
            print("="*80)
            print("CMA-ES PID Controller Autotune")
            print("="*80)
            print(f"Genome: {self.genome}")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {generations}")
            print(f"Steps per eval: {self.steps_per_eval}")
            print()
        
        # Initialize mean (starting point)
        mean = np.array([0.3, 0.05, 0.05, 0.1])  # Reasonable defaults
        
        # Covariance matrix
        C = np.eye(self.n_params)
        
        for gen in range(generations):
            # Generate population
            population = []
            objectives = []
            
            for _ in range(self.population_size):
                # Sample from multivariate Gaussian
                sample = mean + self.sigma * np.random.randn(self.n_params) @ np.linalg.cholesky(C).T
                
                # Evaluate
                obj = self.evaluate(sample)
                
                population.append(sample)
                objectives.append(obj)
            
            # Sort by objective
            sorted_indices = np.argsort(objectives)
            elite_indices = sorted_indices[:self.population_size // 2]
            
            # Update best
            best_idx = sorted_indices[0]
            if objectives[best_idx] < self.best_objective:
                self.best_objective = objectives[best_idx]
                self.best_params = population[best_idx]
            
            # Update mean (weighted recombination)
            weights = np.array([np.log(self.population_size / 2 + 0.5) - np.log(i + 1) 
                               for i in range(len(elite_indices))])
            weights /= weights.sum()
            
            mean = sum(w * population[i] for w, i in zip(weights, elite_indices))
            
            # Update covariance (simplified with regularization)
            elite_samples = np.array([population[i] for i in elite_indices])
            C = np.cov(elite_samples.T) + 1e-6 * np.eye(self.n_params)  # Regularize
            
            # Adapt sigma
            self.sigma *= 0.95  # Gradual decrease
            
            # Record history
            self.history.append({
                'generation': gen,
                'best_objective': float(objectives[best_idx]),
                'mean_objective': float(np.mean(objectives)),
                'best_params': population[best_idx].tolist()
            })
            
            if verbose:
                kp, ki, kd, gamma = population[best_idx]
                print(f"Gen {gen+1:2d}: obj={objectives[best_idx]:.4f} "
                      f"kp={kp:.3f} ki={ki:.3f} kd={kd:.3f} γ={gamma:.3f}")
        
        if verbose:
            print()
            print("="*80)
            print("BEST PARAMETERS")
            print("="*80)
            kp, ki, kd, gamma = self.best_params
            print(f"kp    = {kp:.4f}")
            print(f"ki    = {ki:.4f}")
            print(f"kd    = {kd:.4f}")
            print(f"gamma = {gamma:.4f}")
            print(f"Objective = {self.best_objective:.4f}")
            print()
        
        return {
            'best_params': {
                'kp': float(self.best_params[0]),
                'ki': float(self.best_params[1]),
                'kd': float(self.best_params[2]),
                'gamma': float(self.best_params[3])
            },
            'best_objective': float(self.best_objective),
            'history': self.history
        }
    
    def export(self, filepath: str):
        """Export results to JSON."""
        results = self.run(verbose=False) if self.best_params is None else {
            'best_params': {
                'kp': float(self.best_params[0]),
                'ki': float(self.best_params[1]),
                'kd': float(self.best_params[2]),
                'gamma': float(self.best_params[3])
            },
            'best_objective': float(self.best_objective),
            'history': self.history
        }
        
        data = {
            'config': {
                'genome': self.genome,
                'E': self.E,
                'T_initial': self.T_initial,
                'steps_per_eval': self.steps_per_eval,
                'seed': self.seed
            },
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported autotune results to {filepath}")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CMA-ES PID Autotune')
    parser.add_argument('--genome', default='++[>+<-].:{;}{?}^*=@=.#%',
                       help='Test genome')
    parser.add_argument('--energy', type=float, default=200.0,
                       help='Initial energy')
    parser.add_argument('--generations', type=int, default=20,
                       help='CMA-ES generations')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Steps per evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/autotune_results.json',
                       help='Output file')
    
    args = parser.parse_args()
    
    # Run autotune
    tuner = PIDAutotuner(
        genome=args.genome,
        E=args.energy,
        steps_per_eval=args.steps,
        seed=args.seed
    )
    
    results = tuner.run(generations=args.generations, verbose=True)
    
    # Export
    tuner.export(args.output)
    
    print("Autotune complete!")


if __name__ == '__main__':
    main()

