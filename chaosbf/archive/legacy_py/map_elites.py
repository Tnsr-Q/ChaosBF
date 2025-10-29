#!/usr/bin/env python3
"""
MAP-Elites implementation for ChaosBF.

MAP-Elites is a quality-diversity algorithm that maintains a grid of elites
across multiple behavioral dimensions, enabling exploration of diverse solutions.

Behavioral dimensions for ChaosBF:
1. λ-deviation: How far from criticality (|λ - 1|)
2. Info-per-energy: K(O) / |ΔE|
3. Entropy slope: dS/dt
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v2 import ChaosBFv2, K
import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@dataclass
class Elite:
    """An elite individual in the MAP-Elites grid."""
    genome: str
    fitness: float
    lambda_dev: float
    info_per_energy: float
    entropy_slope: float
    stats: dict
    
    def to_dict(self) -> dict:
        return asdict(self)


class MAPElites:
    """
    MAP-Elites algorithm for ChaosBF.
    
    Maintains a grid of elites across behavioral dimensions.
    """
    
    def __init__(
        self,
        # Grid dimensions
        lambda_bins: int = 10,
        info_bins: int = 10,
        entropy_bins: int = 10,
        # Behavioral ranges
        lambda_range: Tuple[float, float] = (0.0, 2.0),
        info_range: Tuple[float, float] = (0.0, 10.0),
        entropy_range: Tuple[float, float] = (-1.0, 1.0),
        # Execution parameters
        steps_per_eval: int = 2000,
        seed: Optional[int] = None
    ):
        """
        Initialize MAP-Elites.
        
        Args:
            lambda_bins: Number of bins for λ-deviation dimension
            info_bins: Number of bins for info-per-energy dimension
            entropy_bins: Number of bins for entropy slope dimension
            lambda_range: Range for λ-deviation
            info_range: Range for info-per-energy
            entropy_range: Range for entropy slope
            steps_per_eval: Steps to run each genome
            seed: RNG seed
        """
        self.lambda_bins = lambda_bins
        self.info_bins = info_bins
        self.entropy_bins = entropy_bins
        
        self.lambda_range = lambda_range
        self.info_range = info_range
        self.entropy_range = entropy_range
        
        self.steps_per_eval = steps_per_eval
        self.seed = seed
        
        # Initialize grid
        self.grid: Dict[Tuple[int, int, int], Elite] = {}
        
        # Statistics
        self.evaluations = 0
        self.improvements = 0
        
    def _discretize(self, value: float, range_: Tuple[float, float], bins: int) -> int:
        """Discretize a continuous value into a bin index."""
        min_val, max_val = range_
        if value < min_val:
            return 0
        if value >= max_val:
            return bins - 1
        
        normalized = (value - min_val) / (max_val - min_val)
        bin_idx = int(normalized * bins)
        return min(bin_idx, bins - 1)
    
    def _get_cell(self, lambda_dev: float, info_per_energy: float, entropy_slope: float) -> Tuple[int, int, int]:
        """Get grid cell coordinates for behavioral descriptors."""
        lambda_bin = self._discretize(lambda_dev, self.lambda_range, self.lambda_bins)
        info_bin = self._discretize(info_per_energy, self.info_range, self.info_bins)
        entropy_bin = self._discretize(entropy_slope, self.entropy_range, self.entropy_bins)
        
        return (lambda_bin, info_bin, entropy_bin)
    
    def evaluate(self, genome: str, E: float = 200.0, T: float = 0.5) -> Tuple[float, Dict]:
        """
        Evaluate a genome and return fitness + behavioral descriptors.
        
        Returns:
            (fitness, behavioral_descriptors)
        """
        cbf = ChaosBFv2(
            genome,
            E=E,
            T=T,
            seed=self.seed,
            use_pid=True,
            grammar_aware=True,
            verbose=False
        )
        
        cbf.run(steps=self.steps_per_eval)
        
        stats = cbf.get_stats()
        
        # Fitness: info-per-energy
        energy_used = abs(E - stats['energy'])
        if energy_used < 1.0:
            energy_used = 1.0
        
        info = K(cbf.O)
        fitness = info / energy_used
        
        # Behavioral descriptors
        lambda_dev = abs(stats['branching_factor'] - 1.0)
        info_per_energy = fitness
        entropy_slope = stats['dS_dt']
        
        behavioral = {
            'lambda_dev': lambda_dev,
            'info_per_energy': info_per_energy,
            'entropy_slope': entropy_slope,
            'fitness': fitness,
            'stats': stats
        }
        
        self.evaluations += 1
        
        return fitness, behavioral
    
    def add_to_grid(self, genome: str, E: float = 200.0, T: float = 0.5) -> bool:
        """
        Evaluate a genome and add it to the grid if it's an improvement.
        
        Returns:
            True if genome was added/improved a cell
        """
        fitness, behavioral = self.evaluate(genome, E, T)
        
        cell = self._get_cell(
            behavioral['lambda_dev'],
            behavioral['info_per_energy'],
            behavioral['entropy_slope']
        )
        
        # Check if this cell is empty or if we have a better solution
        if cell not in self.grid or fitness > self.grid[cell].fitness:
            elite = Elite(
                genome=genome,
                fitness=fitness,
                lambda_dev=behavioral['lambda_dev'],
                info_per_energy=behavioral['info_per_energy'],
                entropy_slope=behavioral['entropy_slope'],
                stats=behavioral['stats']
            )
            
            self.grid[cell] = elite
            self.improvements += 1
            return True
        
        return False
    
    def random_mutation(self, genome: str, mutation_rate: float = 0.1) -> str:
        """Apply random mutations to a genome."""
        import random
        code = list(genome)
        
        for i in range(len(code)):
            if random.random() < mutation_rate:
                from chaosbf_v2 import SAFE_OPS
                code[i] = random.choice(SAFE_OPS)
        
        return ''.join(code)
    
    def random_crossover(self, genome1: str, genome2: str) -> str:
        """Perform crossover between two genomes."""
        import random
        k = min(len(genome1), len(genome2))
        
        if k < 2:
            return genome1
        
        cut = random.randint(1, k - 1)
        return genome1[:cut] + genome2[cut:]
    
    def run(
        self,
        initial_genomes: List[str],
        iterations: int = 1000,
        mutation_rate: float = 0.1,
        E: float = 200.0,
        T: float = 0.5,
        verbose: bool = True
    ):
        """
        Run MAP-Elites algorithm.
        
        Args:
            initial_genomes: List of seed genomes
            iterations: Number of iterations
            mutation_rate: Mutation rate for offspring
            E: Initial energy for evaluation
            T: Initial temperature for evaluation
            verbose: Print progress
        """
        import random
        
        # Add initial genomes to grid
        if verbose:
            print("Initializing grid with seed genomes...")
        
        for genome in initial_genomes:
            self.add_to_grid(genome, E, T)
        
        if verbose:
            print(f"Initial grid size: {len(self.grid)}")
            print(f"\nRunning {iterations} iterations...")
        
        # Main loop
        for iteration in range(iterations):
            # Select random elite from grid
            if not self.grid:
                # Grid is empty, use random initial genome
                parent = random.choice(initial_genomes)
            else:
                parent_cell = random.choice(list(self.grid.keys()))
                parent = self.grid[parent_cell].genome
            
            # Create offspring through mutation or crossover
            if random.random() < 0.5 and len(self.grid) > 1:
                # Crossover
                parent2_cell = random.choice(list(self.grid.keys()))
                parent2 = self.grid[parent2_cell].genome
                offspring = self.random_crossover(parent, parent2)
            else:
                # Mutation
                offspring = self.random_mutation(parent, mutation_rate)
            
            # Evaluate and add to grid
            self.add_to_grid(offspring, E, T)
            
            # Progress report
            if verbose and (iteration + 1) % 100 == 0:
                coverage = len(self.grid) / (self.lambda_bins * self.info_bins * self.entropy_bins)
                print(f"Iteration {iteration + 1}/{iterations}: "
                      f"Grid size={len(self.grid)}, "
                      f"Coverage={coverage*100:.1f}%, "
                      f"Improvements={self.improvements}")
        
        if verbose:
            print(f"\nFinal grid size: {len(self.grid)}")
            print(f"Total evaluations: {self.evaluations}")
            print(f"Total improvements: {self.improvements}")
    
    def get_best_elite(self) -> Optional[Elite]:
        """Get the elite with highest fitness."""
        if not self.grid:
            return None
        
        return max(self.grid.values(), key=lambda e: e.fitness)
    
    def export_grid(self, filepath: str):
        """Export grid to JSON file."""
        data = {
            'config': {
                'lambda_bins': self.lambda_bins,
                'info_bins': self.info_bins,
                'entropy_bins': self.entropy_bins,
                'lambda_range': self.lambda_range,
                'info_range': self.info_range,
                'entropy_range': self.entropy_range
            },
            'stats': {
                'evaluations': self.evaluations,
                'improvements': self.improvements,
                'grid_size': len(self.grid)
            },
            'elites': {
                f"{cell[0]},{cell[1]},{cell[2]}": elite.to_dict()
                for cell, elite in self.grid.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported grid to {filepath}")
    
    def plot_grid(self, output_path: str):
        """
        Visualize the MAP-Elites grid.
        
        Creates 2D projections of the 3D grid.
        """
        if not self.grid:
            print("Grid is empty, cannot plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('MAP-Elites Behavior Space', fontsize=16, fontweight='bold')
        
        # Extract data
        cells = list(self.grid.keys())
        elites = list(self.grid.values())
        
        lambda_indices = [c[0] for c in cells]
        info_indices = [c[1] for c in cells]
        entropy_indices = [c[2] for c in cells]
        fitnesses = [e.fitness for e in elites]
        
        # Lambda vs Info (colored by fitness)
        scatter1 = axes[0, 0].scatter(
            lambda_indices, info_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 0].set_xlabel('λ-deviation bin')
        axes[0, 0].set_ylabel('Info-per-energy bin')
        axes[0, 0].set_title('λ-deviation vs Info-per-energy')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Fitness')
        
        # Lambda vs Entropy (colored by fitness)
        scatter2 = axes[0, 1].scatter(
            lambda_indices, entropy_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 1].set_xlabel('λ-deviation bin')
        axes[0, 1].set_ylabel('Entropy slope bin')
        axes[0, 1].set_title('λ-deviation vs Entropy slope')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Fitness')
        
        # Info vs Entropy (colored by fitness)
        scatter3 = axes[1, 0].scatter(
            info_indices, entropy_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[1, 0].set_xlabel('Info-per-energy bin')
        axes[1, 0].set_ylabel('Entropy slope bin')
        axes[1, 0].set_title('Info-per-energy vs Entropy slope')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[1, 0], label='Fitness')
        
        # Fitness distribution
        axes[1, 1].hist(fitnesses, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Fitness')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Fitness Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved MAP-Elites visualization to {output_path}")
        plt.close()


def main():
    """Command-line interface for MAP-Elites."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAP-Elites for ChaosBF')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per evaluation')
    parser.add_argument('--seed', type=int, help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/map_elites', help='Output prefix')
    
    args = parser.parse_args()
    
    # Seed genomes
    seeds = [
        '++[>+<-].:{;}{?}^*=@=.#%',
        ':+[>+<-];.#',
        '{?}{?}{?}^=v=.#',
        '*=@=:{;}#%',
        '+++[>+++<-]>{?}*=@.#%',
        '++[>+<-]^v^v:{;}#.%',
        '{+}{?}^{-}{?}v*=@:{;}#%'
    ]
    
    # Run MAP-Elites
    map_elites = MAPElites(
        lambda_bins=10,
        info_bins=10,
        entropy_bins=10,
        steps_per_eval=args.steps,
        seed=args.seed
    )
    
    map_elites.run(
        initial_genomes=seeds,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        E=args.energy,
        T=args.temp,
        verbose=True
    )
    
    # Export results
    map_elites.export_grid(f'{args.output}_grid.json')
    map_elites.plot_grid(f'{args.output}_visualization.png')
    
    # Print best elite
    best = map_elites.get_best_elite()
    if best:
        print(f"\nBest elite:")
        print(f"  Genome: {best.genome}")
        print(f"  Fitness: {best.fitness:.4f}")
        print(f"  λ-deviation: {best.lambda_dev:.4f}")
        print(f"  Info-per-energy: {best.info_per_energy:.4f}")
        print(f"  Entropy slope: {best.entropy_slope:.4f}")


if __name__ == '__main__':
    main()

