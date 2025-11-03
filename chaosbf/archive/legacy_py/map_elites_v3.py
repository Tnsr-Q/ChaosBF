#!/usr/bin/env python3
"""
MAP-Elites v3.0 for ChaosBF with mixed emitters and enhanced behavior space.

Key upgrades:
- 4D behavior space: λ-deviation, info-per-energy, entropy-slope, volatility
- Mixed emitters: random-direction + lineage-novelty
- Min-change guard to prevent grid spam
- Proper descriptor computation (post-dump)
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
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
    volatility: float
    stats: dict
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def descriptor_vector(self) -> np.ndarray:
        """Return descriptor as numpy array for distance calculations."""
        return np.array([
            self.lambda_dev,
            self.info_per_energy,
            self.entropy_slope,
            self.volatility
        ])


class MAPElitesv3:
    """
    MAP-Elites v3.0 with mixed emitters and 4D behavior space.
    """
    
    def __init__(
        self,
        # Grid dimensions (4D)
        lambda_bins: int = 8,
        info_bins: int = 8,
        entropy_bins: int = 8,
        volatility_bins: int = 8,
        # Behavioral ranges
        lambda_range: Tuple[float, float] = (0.0, 2.0),
        info_range: Tuple[float, float] = (-1.0, 5.0),
        entropy_range: Tuple[float, float] = (-1.0, 1.0),
        volatility_range: Tuple[float, float] = (0.0, 0.5),
        # Min-change guard
        min_descriptor_change: float = 0.05,
        min_fitness_improvement: float = 0.01,
        # Execution parameters
        steps_per_eval: int = 2000,
        seed: Optional[int] = None
    ):
        """Initialize MAP-Elites v3.0."""
        self.lambda_bins = lambda_bins
        self.info_bins = info_bins
        self.entropy_bins = entropy_bins
        self.volatility_bins = volatility_bins
        
        self.lambda_range = lambda_range
        self.info_range = info_range
        self.entropy_range = entropy_range
        self.volatility_range = volatility_range
        
        self.min_descriptor_change = min_descriptor_change
        self.min_fitness_improvement = min_fitness_improvement
        
        self.steps_per_eval = steps_per_eval
        self.seed = seed
        
        # Initialize 4D grid
        self.grid: Dict[Tuple[int, int, int, int], Elite] = {}
        
        # Statistics
        self.evaluations = 0
        self.improvements = 0
        self.grid_spam_prevented = 0
        
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
    
    def _get_cell(self, lambda_dev: float, info_per_energy: float, 
                  entropy_slope: float, volatility: float) -> Tuple[int, int, int, int]:
        """Get grid cell coordinates for behavioral descriptors."""
        lambda_bin = self._discretize(lambda_dev, self.lambda_range, self.lambda_bins)
        info_bin = self._discretize(info_per_energy, self.info_range, self.info_bins)
        entropy_bin = self._discretize(entropy_slope, self.entropy_range, self.entropy_bins)
        volatility_bin = self._discretize(volatility, self.volatility_range, self.volatility_bins)
        
        return (lambda_bin, info_bin, entropy_bin, volatility_bin)
    
    def evaluate(self, genome: str, E: float = 200.0, T: float = 0.5) -> Tuple[float, Dict]:
        """
        Evaluate a genome and return fitness + behavioral descriptors.
        
        CRITICAL: Descriptors computed AFTER phenotype operations.
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
        
        # Fitness: info-per-energy (from properly computed descriptors)
        fitness = stats['info_per_energy']
        
        # Behavioral descriptors (already computed correctly in v3)
        behavioral = {
            'lambda_dev': stats['lambda_dev'],
            'info_per_energy': stats['info_per_energy'],
            'entropy_slope': stats['entropy_slope'],
            'volatility': stats['volatility'],
            'fitness': fitness,
            'stats': stats
        }
        
        self.evaluations += 1
        
        return fitness, behavioral
    
    def _descriptor_distance(self, elite1: Elite, elite2: Elite) -> float:
        """Calculate Euclidean distance between two elites in descriptor space."""
        v1 = elite1.descriptor_vector()
        v2 = elite2.descriptor_vector()
        return float(np.linalg.norm(v1 - v2))
    
    def _check_min_change_guard(self, cell: Tuple, new_elite: Elite, new_fitness: float) -> bool:
        """
        Check if new elite passes min-change guard.
        
        Prevents grid spam by requiring either:
        1. Significant descriptor change from incumbent
        2. Significant fitness improvement
        """
        if cell not in self.grid:
            return True  # Cell is empty, always accept
        
        incumbent = self.grid[cell]
        
        # Check fitness improvement
        fitness_improvement = new_fitness - incumbent.fitness
        if fitness_improvement >= self.min_fitness_improvement:
            return True
        
        # Check descriptor distance
        descriptor_dist = self._descriptor_distance(new_elite, incumbent)
        if descriptor_dist >= self.min_descriptor_change:
            return True
        
        # Neither condition met, reject
        self.grid_spam_prevented += 1
        return False
    
    def add_to_grid(self, genome: str, E: float = 200.0, T: float = 0.5) -> bool:
        """
        Evaluate a genome and add it to the grid if it passes guards.
        """
        fitness, behavioral = self.evaluate(genome, E, T)
        
        cell = self._get_cell(
            behavioral['lambda_dev'],
            behavioral['info_per_energy'],
            behavioral['entropy_slope'],
            behavioral['volatility']
        )
        
        # Create elite
        elite = Elite(
            genome=genome,
            fitness=fitness,
            lambda_dev=behavioral['lambda_dev'],
            info_per_energy=behavioral['info_per_energy'],
            entropy_slope=behavioral['entropy_slope'],
            volatility=behavioral['volatility'],
            stats=behavioral['stats']
        )
        
        # Check min-change guard
        if not self._check_min_change_guard(cell, elite, fitness):
            return False
        
        # Check if this is an improvement
        if cell not in self.grid or fitness > self.grid[cell].fitness:
            self.grid[cell] = elite
            self.improvements += 1
            return True
        
        return False
    
    def select_parent_random(self) -> Optional[str]:
        """Random parent selection."""
        if not self.grid:
            return None
        
        import random
        parent_cell = random.choice(list(self.grid.keys()))
        return self.grid[parent_cell].genome
    
    def select_parent_novelty(self) -> Optional[str]:
        """
        Lineage-novelty parent selection.
        
        Pick parents far in descriptor space to encourage exploration.
        """
        if len(self.grid) < 2:
            return self.select_parent_random()
        
        import random
        
        # Pick a random reference elite
        ref_cell = random.choice(list(self.grid.keys()))
        ref_elite = self.grid[ref_cell]
        
        # Find elite farthest from reference
        max_dist = 0.0
        farthest_genome = None
        
        for elite in self.grid.values():
            dist = self._descriptor_distance(ref_elite, elite)
            if dist > max_dist:
                max_dist = dist
                farthest_genome = elite.genome
        
        return farthest_genome if farthest_genome else self.select_parent_random()
    
    def mutate(self, genome: str, mutation_rate: float = 0.1) -> str:
        """Apply random mutations to a genome."""
        import random
        code = list(genome)
        
        for i in range(len(code)):
            if random.random() < mutation_rate:
                from chaosbf_v3 import SAFE_OPS
                code[i] = random.choice(SAFE_OPS)
        
        return ''.join(code)
    
    def crossover(self, genome1: str, genome2: str) -> str:
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
        novelty_ratio: float = 0.3,
        E: float = 200.0,
        T: float = 0.5,
        verbose: bool = True
    ):
        """
        Run MAP-Elites with mixed emitters.
        
        Args:
            initial_genomes: Seed genomes
            iterations: Number of iterations
            mutation_rate: Mutation rate for offspring
            novelty_ratio: Fraction of iterations using novelty emitter
            E: Initial energy
            T: Initial temperature
            verbose: Print progress
        """
        import random
        
        # Add initial genomes
        if verbose:
            print("Initializing grid with seed genomes...")
        
        for genome in initial_genomes:
            self.add_to_grid(genome, E, T)
        
        if verbose:
            print(f"Initial grid size: {len(self.grid)}")
            print(f"\nRunning {iterations} iterations with mixed emitters...")
            print(f"  Novelty ratio: {novelty_ratio:.1%}")
            print(f"  Min descriptor change: {self.min_descriptor_change}")
            print(f"  Min fitness improvement: {self.min_fitness_improvement}")
            print()
        
        # Main loop with mixed emitters
        for iteration in range(iterations):
            # Select parent using mixed emitters
            if random.random() < novelty_ratio:
                parent = self.select_parent_novelty()
            else:
                parent = self.select_parent_random()
            
            if not parent:
                parent = random.choice(initial_genomes)
            
            # Create offspring through mutation or crossover
            if random.random() < 0.5 and len(self.grid) > 1:
                # Crossover
                if random.random() < novelty_ratio:
                    parent2 = self.select_parent_novelty()
                else:
                    parent2 = self.select_parent_random()
                
                if not parent2:
                    parent2 = random.choice(initial_genomes)
                
                offspring = self.crossover(parent, parent2)
            else:
                # Mutation
                offspring = self.mutate(parent, mutation_rate)
            
            # Evaluate and add to grid
            self.add_to_grid(offspring, E, T)
            
            # Progress report
            if verbose and (iteration + 1) % 100 == 0:
                total_cells = (self.lambda_bins * self.info_bins * 
                              self.entropy_bins * self.volatility_bins)
                coverage = len(self.grid) / total_cells
                print(f"Iteration {iteration + 1}/{iterations}: "
                      f"Grid size={len(self.grid)}, "
                      f"Coverage={coverage*100:.2f}%, "
                      f"Improvements={self.improvements}, "
                      f"Spam prevented={self.grid_spam_prevented}")
        
        if verbose:
            print(f"\nFinal grid size: {len(self.grid)}")
            print(f"Total evaluations: {self.evaluations}")
            print(f"Total improvements: {self.improvements}")
            print(f"Grid spam prevented: {self.grid_spam_prevented}")
    
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
                'volatility_bins': self.volatility_bins,
                'lambda_range': self.lambda_range,
                'info_range': self.info_range,
                'entropy_range': self.entropy_range,
                'volatility_range': self.volatility_range
            },
            'stats': {
                'evaluations': self.evaluations,
                'improvements': self.improvements,
                'grid_spam_prevented': self.grid_spam_prevented,
                'grid_size': len(self.grid)
            },
            'elites': {
                f"{cell[0]},{cell[1]},{cell[2]},{cell[3]}": elite.to_dict()
                for cell, elite in self.grid.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported grid to {filepath}")
    
    def plot_grid(self, output_path: str):
        """Visualize the MAP-Elites grid with 2D projections."""
        if not self.grid:
            print("Grid is empty, cannot plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MAP-Elites v3.0: 4D Behavior Space (2D Projections)', 
                     fontsize=16, fontweight='bold')
        
        # Extract data
        cells = list(self.grid.keys())
        elites = list(self.grid.values())
        
        lambda_indices = [c[0] for c in cells]
        info_indices = [c[1] for c in cells]
        entropy_indices = [c[2] for c in cells]
        volatility_indices = [c[3] for c in cells]
        fitnesses = [e.fitness for e in elites]
        
        # Lambda vs Info
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
        
        # Lambda vs Entropy
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
        
        # Lambda vs Volatility
        scatter3 = axes[0, 2].scatter(
            lambda_indices, volatility_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 2].set_xlabel('λ-deviation bin')
        axes[0, 2].set_ylabel('Volatility bin')
        axes[0, 2].set_title('λ-deviation vs Volatility')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[0, 2], label='Fitness')
        
        # Info vs Entropy
        scatter4 = axes[1, 0].scatter(
            info_indices, entropy_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[1, 0].set_xlabel('Info-per-energy bin')
        axes[1, 0].set_ylabel('Entropy slope bin')
        axes[1, 0].set_title('Info-per-energy vs Entropy slope')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1, 0], label='Fitness')
        
        # Info vs Volatility
        scatter5 = axes[1, 1].scatter(
            info_indices, volatility_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[1, 1].set_xlabel('Info-per-energy bin')
        axes[1, 1].set_ylabel('Volatility bin')
        axes[1, 1].set_title('Info-per-energy vs Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter5, ax=axes[1, 1], label='Fitness')
        
        # Fitness distribution
        axes[1, 2].hist(fitnesses, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 2].set_xlabel('Fitness')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Fitness Distribution')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved MAP-Elites visualization to {output_path}")
        plt.close()


def main():
    """Command-line interface for MAP-Elites v3.0."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAP-Elites v3.0 for ChaosBF')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--novelty-ratio', type=float, default=0.3, help='Novelty emitter ratio')
    parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per evaluation')
    parser.add_argument('--seed', type=int, help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/map_elites_v3', 
                       help='Output prefix')
    
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
    
    # Run MAP-Elites v3.0
    map_elites = MAPElitesv3(
        lambda_bins=8,
        info_bins=8,
        entropy_bins=8,
        volatility_bins=8,
        steps_per_eval=args.steps,
        seed=args.seed
    )
    
    map_elites.run(
        initial_genomes=seeds,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        novelty_ratio=args.novelty_ratio,
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
        print(f"  Volatility: {best.volatility:.4f}")


if __name__ == '__main__':
    main()

