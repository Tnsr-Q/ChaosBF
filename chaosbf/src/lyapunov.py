#!/usr/bin/env python3
"""
Lyapunov Divergence Estimation for ChaosBF

Runs twin clones (A/B) with identical seeds + ε-perturbed RNG.
Tracks divergence in behavioral descriptor space to estimate λ_lyapunov.

This separates critical-stable (low λ_lyap) from critical-chaotic (high λ_lyap)
elites, even when both have λ_target ≈ 1.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3
import numpy as np
from typing import Tuple, Dict, List
import json


class LyapunovEstimator:
    """
    Estimate Lyapunov exponent via twin-run divergence.
    
    Runs two clones with slightly perturbed RNG seeds and tracks
    how quickly their behavioral descriptors diverge.
    """
    
    def __init__(
        self,
        perturbation: float = 1e-6,
        window_size: int = 100
    ):
        """
        Initialize Lyapunov estimator.
        
        Args:
            perturbation: RNG seed perturbation (as integer offset)
            window_size: Window for averaging divergence
        """
        self.perturbation = int(max(1, perturbation * 1e6))  # Convert to seed offset
        self.window_size = window_size
    
    def run_twin(
        self,
        genome: str,
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 2000,
        seed: int = 42,
        verbose: bool = False
    ) -> Tuple[ChaosBFv3, ChaosBFv3]:
        """
        Run twin clones with perturbed seeds.
        
        Returns:
            (clone_A, clone_B)
        """
        # Clone A: original seed
        clone_A = ChaosBFv3(
            genome,
            E=E,
            T=T,
            seed=seed,
            use_pid=True,
            use_variance_shaping=True,
            verbose=verbose
        )
        
        # Clone B: perturbed seed
        clone_B = ChaosBFv3(
            genome,
            E=E,
            T=T,
            seed=seed + self.perturbation,
            use_pid=True,
            use_variance_shaping=True,
            verbose=verbose
        )
        
        # Run both
        clone_A.run(steps=steps)
        clone_B.run(steps=steps)
        
        return clone_A, clone_B
    
    def compute_descriptor_divergence(
        self,
        clone_A: ChaosBFv3,
        clone_B: ChaosBFv3
    ) -> float:
        """
        Compute Euclidean distance between descriptors.
        
        Uses all available behavioral descriptors:
        - λ-deviation
        - Info-per-energy
        - Entropy slope
        - Volatility
        """
        stats_A = clone_A.get_stats()
        stats_B = clone_B.get_stats()
        
        # Extract descriptors
        desc_A = np.array([
            stats_A['lambda_dev'],
            stats_A['info_per_energy'],
            stats_A['entropy_slope'],
            stats_A['volatility']
        ])
        
        desc_B = np.array([
            stats_B['lambda_dev'],
            stats_B['info_per_energy'],
            stats_B['entropy_slope'],
            stats_B['volatility']
        ])
        
        # Euclidean distance
        divergence = float(np.linalg.norm(desc_A - desc_B))
        
        return divergence
    
    def estimate_lyapunov(
        self,
        genome: str,
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 2000,
        seed: int = 42,
        n_bootstrap: int = 200,
        verbose: bool = True
    ) -> Dict:
        """
        Estimate Lyapunov exponent for a genome with bootstrap CI.
        
        Returns:
            Dictionary with λ_lyapunov estimate, CI, and diagnostics
        """
        if verbose:
            print(f"Estimating Lyapunov exponent for genome: {genome}")
            print(f"  Perturbation: {self.perturbation} (seed offset)")
            print(f"  Running twin clones...")
        
        # Run twins
        clone_A, clone_B = self.run_twin(genome, E, T, steps, seed, verbose=False)
        
        # Compute divergence
        divergence = self.compute_descriptor_divergence(clone_A, clone_B)
        
        # Lyapunov exponent proxy
        # λ_lyap ≈ log(divergence) / time
        # Use steps as proxy for time
        time_proxy = min(clone_A.steps_executed, clone_B.steps_executed)
        
        if divergence > 1e-10 and time_proxy > 0:
            lambda_lyap = np.log(max(divergence, 1e-10)) / time_proxy
        else:
            lambda_lyap = 0.0
        
        # Bootstrap CI on divergence
        # Run multiple twin pairs with different seeds
        divergences = [divergence]
        for i in range(n_bootstrap - 1):
            try:
                clone_A_boot, clone_B_boot = self.run_twin(
                    genome, E, T, steps, seed + (i + 1) * 1000, verbose=False
                )
                div_boot = self.compute_descriptor_divergence(clone_A_boot, clone_B_boot)
                divergences.append(div_boot)
            except:
                pass
        
        # Compute CI
        divergences = np.array(divergences)
        lambda_lyaps = np.log(np.maximum(divergences, 1e-10)) / time_proxy
        
        ci_low, ci_high = np.percentile(lambda_lyaps, [2.5, 97.5])
        
        # Check if CI straddles zero (edge band)
        is_edge_band = (ci_low < 0) and (ci_high > 0)
        
        # Get stats
        stats_A = clone_A.get_stats()
        stats_B = clone_B.get_stats()
        
        result = {
            'genome': genome,
            'lambda_lyapunov': lambda_lyap,
            'lambda_lyap_ci_low': ci_low,
            'lambda_lyap_ci_high': ci_high,
            'is_edge_band': is_edge_band,
            'divergence': divergence,
            'divergence_mean': float(np.mean(divergences)),
            'divergence_std': float(np.std(divergences)),
            'n_bootstrap': len(divergences),
            'steps_A': clone_A.steps_executed,
            'steps_B': clone_B.steps_executed,
            'lambda_target_A': stats_A['branching_factor'],
            'lambda_target_B': stats_B['branching_factor'],
            'volatility_A': stats_A['volatility'],
            'volatility_B': stats_B['volatility'],
            'descriptors_A': {
                'lambda_dev': stats_A['lambda_dev'],
                'info_per_energy': stats_A['info_per_energy'],
                'entropy_slope': stats_A['entropy_slope'],
                'volatility': stats_A['volatility']
            },
            'descriptors_B': {
                'lambda_dev': stats_B['lambda_dev'],
                'info_per_energy': stats_B['info_per_energy'],
                'entropy_slope': stats_B['entropy_slope'],
                'volatility': stats_B['volatility']
            }
        }
        
        if verbose:
            print(f"\nResults:")
            print(f"  λ_lyapunov: {lambda_lyap:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            if is_edge_band:
                print(f"  *** EDGE BAND: CI straddles zero ***")
            print(f"  Divergence: {divergence:.6f} ± {np.std(divergences):.6f}")
            print(f"  λ_target (A): {stats_A['branching_factor']:.3f}")
            print(f"  λ_target (B): {stats_B['branching_factor']:.3f}")
            print(f"  Volatility (A): {stats_A['volatility']:.4f}")
            print(f"  Volatility (B): {stats_B['volatility']:.4f}")
        
        return result
    
    def batch_estimate(
        self,
        genomes: List[str],
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 2000,
        seed: int = 42,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Estimate Lyapunov exponents for multiple genomes.
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, genome in enumerate(genomes):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Genome {i+1}/{len(genomes)}")
                print(f"{'='*80}")
            
            result = self.estimate_lyapunov(
                genome, E, T, steps, seed + i * 1000, verbose
            )
            results.append(result)
        
        return results
    
    def plot_phase_space(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        Plot (λ_target, λ_lyap) phase space.
        
        This separates critical-stable from critical-chaotic elites.
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        lambda_targets = [r['lambda_target_A'] for r in results]
        lambda_lyaps = [r['lambda_lyapunov'] for r in results]
        volatilities = [r['volatility_A'] for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Lyapunov Analysis: Critical-Stable vs Critical-Chaotic', 
                     fontsize=14, fontweight='bold')
        
        # λ_target vs λ_lyap
        scatter1 = axes[0].scatter(
            lambda_targets, lambda_lyaps,
            c=volatilities, cmap='coolwarm',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='λ=1 (critical)')
        axes[0].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('λ_target (Branching Factor)')
        axes[0].set_ylabel('λ_lyapunov (Divergence Rate)')
        axes[0].set_title('Phase Space: Stability vs Chaos')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Volatility')
        
        # Volatility vs λ_lyap
        scatter2 = axes[1].scatter(
            volatilities, lambda_lyaps,
            c=lambda_targets, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[1].set_xlabel('Volatility (std(λ))')
        axes[1].set_ylabel('λ_lyapunov (Divergence Rate)')
        axes[1].set_title('Volatility vs Lyapunov Exponent')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='λ_target')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved Lyapunov phase space plot to {output_path}")
        plt.close()


def main():
    """Command-line interface for Lyapunov estimation."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Lyapunov Divergence Estimation for ChaosBF')
    parser.add_argument('--genomes', nargs='+', help='Genomes to analyze')
    parser.add_argument('--genome-file', help='File with one genome per line')
    parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per run')
    parser.add_argument('--perturbation', type=float, default=1e-6, help='RNG perturbation')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/lyapunov',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Get genomes
    if args.genome_file:
        with open(args.genome_file, 'r') as f:
            genomes = [line.strip() for line in f if line.strip()]
    elif args.genomes:
        genomes = args.genomes
    else:
        # Default test genomes
        genomes = [
            '++[>+<-].:{;}{?}^*=@=.#%',
            ':+[>+<-];.#',
            '{?}{?}{?}^=v=.#',
            '*=@=:{;}#%',
            '+++[>+++<-]>{?}*=@.#%'
        ]
    
    print(f"Analyzing {len(genomes)} genomes for Lyapunov divergence")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize estimator
    estimator = LyapunovEstimator(
        perturbation=args.perturbation
    )
    
    # Run batch estimation
    results = estimator.batch_estimate(
        genomes=genomes,
        E=args.energy,
        T=args.temp,
        steps=args.steps,
        seed=args.seed,
        verbose=True
    )
    
    # Save results
    results_file = f'{args.output}/lyapunov_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Plot phase space
    estimator.plot_phase_space(results, f'{args.output}/lyapunov_phase_space.png')
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    lambda_lyaps = [r['lambda_lyapunov'] for r in results]
    divergences = [r['divergence'] for r in results]
    
    print(f"λ_lyapunov range: [{min(lambda_lyaps):.6f}, {max(lambda_lyaps):.6f}]")
    print(f"Mean λ_lyapunov: {np.mean(lambda_lyaps):.6f}")
    print(f"Std λ_lyapunov: {np.std(lambda_lyaps):.6f}")
    print(f"\nDivergence range: [{min(divergences):.6f}, {max(divergences):.6f}]")
    print(f"Mean divergence: {np.mean(divergences):.6f}")
    
    # Classify elites
    critical_stable = sum(1 for r in results if abs(r['lambda_target_A'] - 1.0) < 0.2 and r['lambda_lyapunov'] < 0.0)
    critical_chaotic = sum(1 for r in results if abs(r['lambda_target_A'] - 1.0) < 0.2 and r['lambda_lyapunov'] > 0.0)
    
    print(f"\nCritical-stable elites (λ≈1, λ_lyap<0): {critical_stable}")
    print(f"Critical-chaotic elites (λ≈1, λ_lyap>0): {critical_chaotic}")


if __name__ == '__main__':
    main()

