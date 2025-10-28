#!/usr/bin/env python3
"""
AURORA-MAP-Elites: 6D Behavior Space with Learned + Hand-Crafted Descriptors

Combines:
- 2D learned descriptors from AURORA autoencoder
- 4D hand-crafted descriptors (λ-dev, info/energy, entropy-slope, volatility)

With increased bins (16 per dimension) and descriptor whitening for better coverage.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3
from aurora import AURORADescriptors
from map_elites_v31 import DescriptorWhitener, AdaptiveEmitterScheduler, Elite
import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AURORAMAPElites:
    """
    MAP-Elites with 6D behavior space:
    - 2D learned (AURORA latent dims)
    - 4D hand-crafted (λ-dev, info/E, dS/dt, volatility)
    
    With 16 bins per dimension for better granularity.
    """
    
    def __init__(
        self,
        aurora_model_path: Optional[str] = None,
        bins_per_dim: int = 16,
        steps_per_eval: int = 2000,
        seed: Optional[int] = None,
        use_whitening: bool = True,
        use_adaptive_emitters: bool = True
    ):
        """
        Initialize AURORA-MAP-Elites.
        
        Args:
            aurora_model_path: Path to trained AURORA model
            bins_per_dim: Number of bins per dimension (16 for better coverage)
            steps_per_eval: Steps per genome evaluation
            seed: RNG seed
            use_whitening: Enable descriptor whitening
            use_adaptive_emitters: Enable adaptive emitter scheduling
        """
        self.bins_per_dim = bins_per_dim
        self.steps_per_eval = steps_per_eval
        self.seed = seed
        self.use_whitening = use_whitening
        self.use_adaptive_emitters = use_adaptive_emitters
        
        # AURORA
        self.aurora = AURORADescriptors(latent_dim=2, seed=seed)
        if aurora_model_path:
            self.aurora.load(aurora_model_path)
        
        # 6D grid: (aurora1, aurora2, lambda_dev, info_E, entropy_slope, volatility)
        self.grid: Dict[Tuple[int, ...], Elite] = {}
        
        # Descriptor whitening (6D)
        if use_whitening:
            self.whitener = DescriptorWhitener(n_dims=6, alpha=0.1)
        
        # Adaptive emitters
        if use_adaptive_emitters:
            self.emitter_scheduler = AdaptiveEmitterScheduler(
                initial_novelty_ratio=0.3,
                tau_low=0.01,
                tau_high=0.05
            )
        
        # Lineage tracking
        self.next_ancestor_id = 0
        self.lineage_map = {}
        
        # Statistics
        self.evaluations = 0
        self.improvements = 0
        self.grid_spam_prevented = 0
    
    def _discretize(self, value: float, min_val: float = -3.0, max_val: float = 3.0) -> int:
        """Discretize a whitened value into a bin."""
        if value < min_val:
            return 0
        if value >= max_val:
            return self.bins_per_dim - 1
        
        normalized = (value - min_val) / (max_val - min_val)
        bin_idx = int(normalized * self.bins_per_dim)
        return min(bin_idx, self.bins_per_dim - 1)
    
    def _get_cell(self, descriptors: np.ndarray) -> Tuple[int, ...]:
        """
        Get 6D grid cell coordinates.
        
        Args:
            descriptors: (6,) array of whitened descriptors
        """
        return tuple(self._discretize(d) for d in descriptors)
    
    def evaluate(self, genome: str, E: float = 200.0, T: float = 0.5) -> Tuple[float, np.ndarray]:
        """
        Evaluate genome and return fitness + 6D descriptors.
        
        Returns:
            (fitness, raw_descriptors)
        """
        cbf = ChaosBFv3(
            genome,
            E=E,
            T=T,
            seed=self.seed,
            use_pid=True,
            use_variance_shaping=True,
            grammar_aware=True,
            verbose=False
        )
        
        cbf.run(steps=self.steps_per_eval)
        
        stats = cbf.get_stats()
        
        # Fitness
        fitness = stats['info_per_energy']
        
        # AURORA descriptors (2D)
        if self.aurora.is_trained:
            aurora_desc = self.aurora.compute_descriptors(cbf, track_coverage=True)
        else:
            aurora_desc = (0.0, 0.0)
        
        # Hand-crafted descriptors (4D)
        hand_crafted = np.array([
            stats['lambda_dev'],
            stats['info_per_energy'],
            stats['entropy_slope'],
            stats['volatility']
        ])
        
        # Combined 6D descriptors
        raw_descriptors = np.array([
            aurora_desc[0],
            aurora_desc[1],
            hand_crafted[0],
            hand_crafted[1],
            hand_crafted[2],
            hand_crafted[3]
        ])
        
        self.evaluations += 1
        
        return fitness, raw_descriptors
    
    def add_to_grid(self, genome: str, E: float = 200.0, T: float = 0.5) -> bool:
        """
        Evaluate and add genome to grid.
        
        Returns:
            True if added/improved, False otherwise
        """
        fitness, raw_descriptors = self.evaluate(genome, E, T)
        
        # Whiten descriptors
        if self.use_whitening:
            self.whitener.update(raw_descriptors.reshape(1, -1))
            whitened_descriptors = self.whitener.whiten(raw_descriptors)
        else:
            whitened_descriptors = raw_descriptors
        
        # Get cell
        cell = self._get_cell(whitened_descriptors)
        
        # Create elite
        elite = Elite(
            genome=genome,
            fitness=fitness,
            lambda_dev=raw_descriptors[2],
            info_per_energy=raw_descriptors[3],
            entropy_slope=raw_descriptors[4],
            volatility=raw_descriptors[5],
            stats={'aurora_desc': (raw_descriptors[0], raw_descriptors[1])},
            ancestor_id=self.next_ancestor_id
        )
        self.next_ancestor_id += 1
        
        # Check if improvement
        if cell not in self.grid or fitness > self.grid[cell].fitness:
            self.grid[cell] = elite
            self.improvements += 1
            return True
        
        return False
    
    def run(
        self,
        initial_genomes: List[str],
        iterations: int = 1000,
        E: float = 200.0,
        T: float = 0.5,
        verbose: bool = True
    ):
        """Run AURORA-MAP-Elites evolution."""
        import random
        
        if verbose:
            print("="*80)
            print("AURORA-MAP-Elites: 6D Behavior Space")
            print("="*80)
            print(f"Bins per dimension: {self.bins_per_dim}")
            print(f"Total cells: {self.bins_per_dim**6:,}")
            print(f"Descriptor whitening: {self.use_whitening}")
            print(f"Adaptive emitters: {self.use_adaptive_emitters}")
            print(f"AURORA trained: {self.aurora.is_trained}")
            print()
        
        # Initialize grid
        for genome in initial_genomes:
            self.add_to_grid(genome, E, T)
        
        if verbose:
            print(f"Initial grid size: {len(self.grid)}")
            print()
        
        # Main loop
        window_improvements = 0
        
        for iteration in range(iterations):
            # Get emitter config
            if self.use_adaptive_emitters:
                config = self.emitter_scheduler.get_config()
                novelty_ratio = config['novelty_ratio']
                mutation_rate = config['mutation_radius']
            else:
                novelty_ratio = 0.3
                mutation_rate = 0.1
            
            # Select parent
            if self.grid and random.random() < novelty_ratio:
                # Novelty selection
                parent = random.choice(list(self.grid.values())).genome
            elif self.grid:
                # Random selection
                parent = random.choice(list(self.grid.values())).genome
            else:
                parent = random.choice(initial_genomes)
            
            # Create offspring
            offspring = self._mutate(parent, mutation_rate)
            
            # Add to grid
            improved = self.add_to_grid(offspring, E, T)
            if improved:
                window_improvements += 1
            
            # Update adaptive scheduler
            if self.use_adaptive_emitters and (iteration + 1) % 100 == 0:
                self.emitter_scheduler.update(window_improvements, 100)
                window_improvements = 0
            
            # Progress
            if verbose and (iteration + 1) % 100 == 0:
                total_cells = self.bins_per_dim ** 6
                coverage = len(self.grid) / total_cells
                
                status = f"Iteration {iteration + 1}/{iterations}: "
                status += f"Grid={len(self.grid)}, "
                status += f"Coverage={coverage*100:.4f}%, "
                status += f"Improvements={self.improvements}"
                
                if self.use_adaptive_emitters:
                    status += f", Novelty={config['novelty_ratio']:.2f}"
                
                if self.aurora.is_trained:
                    kl = self.aurora.compute_latent_coverage_kl()
                    status += f", KL={kl:.3f}"
                
                print(status)
        
        if verbose:
            print()
            print(f"Final grid size: {len(self.grid)}")
            print(f"Coverage: {len(self.grid) / (self.bins_per_dim**6) * 100:.4f}%")
            print(f"Improvements: {self.improvements}")
            if self.aurora.is_trained:
                print(f"AURORA KL: {self.aurora.compute_latent_coverage_kl():.3f}")
    
    def _mutate(self, genome: str, rate: float = 0.1) -> str:
        """Mutate genome."""
        import random
        from chaosbf_v3 import SAFE_OPS
        
        code = list(genome)
        for i in range(len(code)):
            if random.random() < rate:
                code[i] = random.choice(SAFE_OPS)
        
        return ''.join(code)
    
    def get_best_elite(self) -> Optional[Elite]:
        """Get elite with highest fitness."""
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda e: e.fitness)
    
    def export_grid(self, filepath: str):
        """Export grid to JSON."""
        data = {
            'config': {
                'bins_per_dim': self.bins_per_dim,
                'total_cells': self.bins_per_dim ** 6,
                'use_whitening': self.use_whitening,
                'use_adaptive_emitters': self.use_adaptive_emitters,
                'aurora_trained': self.aurora.is_trained
            },
            'stats': {
                'evaluations': self.evaluations,
                'improvements': self.improvements,
                'grid_size': len(self.grid),
                'coverage': len(self.grid) / (self.bins_per_dim ** 6)
            },
            'elites': {
                f"{','.join(map(str, cell))}": elite.to_dict()
                for cell, elite in self.grid.items()
            }
        }
        
        if self.aurora.is_trained:
            data['stats']['aurora_kl'] = self.aurora.compute_latent_coverage_kl()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported grid to {filepath}")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AURORA-MAP-Elites 6D')
    parser.add_argument('--aurora-model', help='Path to trained AURORA model')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Number of iterations')
    parser.add_argument('--bins', type=int, default=16,
                       help='Bins per dimension')
    parser.add_argument('--seed', type=int, default=42,
                       help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/aurora_map_elites',
                       help='Output prefix')
    
    args = parser.parse_args()
    
    # Seed genomes
    seeds = [
        '++[>+<-].:{;}{?}^*=@=.#%',
        ':+[>+<-];.#',
        '{?}{?}{?}^=v=.#',
        '*=@=:{;}#%'
    ]
    
    # Run
    ame = AURORAMAPElites(
        aurora_model_path=args.aurora_model,
        bins_per_dim=args.bins,
        seed=args.seed
    )
    
    ame.run(
        initial_genomes=seeds,
        iterations=args.iterations,
        verbose=True
    )
    
    # Export
    ame.export_grid(f'{args.output}_grid.json')
    
    # Print best
    best = ame.get_best_elite()
    if best:
        print(f"\nBest elite:")
        print(f"  Genome: {best.genome}")
        print(f"  Fitness: {best.fitness:.4f}")


if __name__ == '__main__':
    main()

