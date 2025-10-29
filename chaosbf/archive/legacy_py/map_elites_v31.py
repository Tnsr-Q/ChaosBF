#!/usr/bin/env python3
"""
MAP-Elites v3.1 for ChaosBF with critical bolts tightened:
- Descriptor whitening (z-score before binning)
- Adaptive emitter scheduler (novelty pressure)
- Lineage entropy tracking

These small changes unlock grid interior and prevent stalling.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
import numpy as np
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
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
    ancestor_id: int = 0  # For lineage tracking
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def descriptor_vector(self) -> np.ndarray:
        """Return descriptor as numpy array (before whitening)."""
        return np.array([
            self.lambda_dev,
            self.info_per_energy,
            self.entropy_slope,
            self.volatility
        ])


class DescriptorWhitener:
    """
    Online descriptor whitening using exponential moving average.
    
    Maintains running statistics to z-score descriptors before binning.
    """
    
    def __init__(self, n_dims: int, alpha: float = 0.1, eps: float = 1e-8):
        """
        Initialize whitener.
        
        Args:
            n_dims: Number of descriptor dimensions
            alpha: EMA decay rate
            eps: Small constant for numerical stability
        """
        self.n_dims = n_dims
        self.alpha = alpha
        self.eps = eps
        
        # Running statistics
        self.mean = np.zeros(n_dims)
        self.var = np.ones(n_dims)
        self.n_samples = 0
    
    def update(self, descriptors: np.ndarray):
        """
        Update running statistics with new descriptors.
        
        Args:
            descriptors: (n_samples, n_dims) array
        """
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        
        batch_mean = np.mean(descriptors, axis=0)
        batch_var = np.var(descriptors, axis=0)
        
        if self.n_samples == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            # EMA update
            self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
            self.var = (1 - self.alpha) * self.var + self.alpha * batch_var
        
        self.n_samples += descriptors.shape[0]
    
    def whiten(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Z-score normalize descriptors.
        
        Args:
            descriptors: (n_samples, n_dims) or (n_dims,) array
        
        Returns:
            Whitened descriptors
        """
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        std = np.sqrt(self.var + self.eps)
        whitened = (descriptors - self.mean) / std
        
        if squeeze:
            whitened = whitened.squeeze()
        
        return whitened


class AdaptiveEmitterScheduler:
    """
    Adaptive emitter budget based on discovery rate.
    
    Increases novelty pressure when grid stalls, decreases when discovery rebounds.
    """
    
    def __init__(
        self,
        initial_novelty_ratio: float = 0.3,
        tau_low: float = 0.01,
        tau_high: float = 0.05,
        window_size: int = 100,
        delta_ratio: float = 0.1,
        delta_mutation: float = 0.1
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            initial_novelty_ratio: Starting ratio of novelty emitters
            tau_low: Discovery rate threshold for increasing novelty
            tau_high: Discovery rate threshold for decreasing novelty
            window_size: Window for computing discovery rate
            delta_ratio: Change in novelty ratio per adjustment
            delta_mutation: Change in mutation radius per adjustment
        """
        self.novelty_ratio = initial_novelty_ratio
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.window_size = window_size
        self.delta_ratio = delta_ratio
        self.delta_mutation = delta_mutation
        
        # History
        self.improvements_history = []
        self.mutation_radius = 0.1
    
    def update(self, improvements: int, iterations: int):
        """
        Update emitter budget based on discovery rate.
        
        Args:
            improvements: Number of improvements in window
            iterations: Number of iterations in window
        """
        discovery_rate = improvements / max(1, iterations)
        self.improvements_history.append(improvements)
        
        # Keep window
        if len(self.improvements_history) > self.window_size:
            self.improvements_history.pop(0)
        
        # Adjust based on discovery rate
        if discovery_rate < self.tau_low:
            # Stalling: increase novelty pressure
            self.novelty_ratio = min(0.7, self.novelty_ratio + self.delta_ratio)
            self.mutation_radius = min(0.3, self.mutation_radius * (1 + self.delta_mutation))
        elif discovery_rate > self.tau_high:
            # Rebounding: decrease novelty pressure
            self.novelty_ratio = max(0.1, self.novelty_ratio - self.delta_ratio)
            self.mutation_radius = max(0.05, self.mutation_radius * (1 - self.delta_mutation))
    
    def get_config(self) -> Dict:
        """Get current emitter configuration."""
        return {
            'novelty_ratio': self.novelty_ratio,
            'random_ratio': 1.0 - self.novelty_ratio,
            'mutation_radius': self.mutation_radius
        }


class MAPElitesv31:
    """
    MAP-Elites v3.1 with descriptor whitening and adaptive emitters.
    """
    
    def __init__(
        self,
        # Grid dimensions (4D)
        lambda_bins: int = 8,
        info_bins: int = 8,
        entropy_bins: int = 8,
        volatility_bins: int = 8,
        # Behavioral ranges (will be adjusted by whitening)
        lambda_range: Tuple[float, float] = (-3.0, 3.0),
        info_range: Tuple[float, float] = (-3.0, 3.0),
        entropy_range: Tuple[float, float] = (-3.0, 3.0),
        volatility_range: Tuple[float, float] = (-3.0, 3.0),
        # Min-change guard
        min_descriptor_change: float = 0.05,
        min_fitness_improvement: float = 0.01,
        # Execution parameters
        steps_per_eval: int = 2000,
        seed: Optional[int] = None,
        # Whitening
        use_whitening: bool = True,
        # Adaptive emitters
        use_adaptive_emitters: bool = True
    ):
        """Initialize MAP-Elites v3.1."""
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
        
        # Whitening
        self.use_whitening = use_whitening
        if use_whitening:
            self.whitener = DescriptorWhitener(n_dims=4, alpha=0.1)
        
        # Adaptive emitters
        self.use_adaptive_emitters = use_adaptive_emitters
        if use_adaptive_emitters:
            self.emitter_scheduler = AdaptiveEmitterScheduler(
                initial_novelty_ratio=0.3,
                tau_low=0.01,
                tau_high=0.05
            )
        else:
            self.emitter_scheduler = None
        
        # Lineage tracking
        self.next_ancestor_id = 0
        self.lineage_map = {}  # cell -> ancestor_id
        
        # Statistics
        self.evaluations = 0
        self.improvements = 0
        self.grid_spam_prevented = 0
        self.improvements_per_window = []
    
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
    
    def _get_cell(self, descriptors: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get grid cell coordinates for behavioral descriptors.
        
        Args:
            descriptors: (4,) array of [lambda_dev, info_per_energy, entropy_slope, volatility]
                        ALREADY WHITENED if whitening is enabled
        """
        lambda_bin = self._discretize(descriptors[0], self.lambda_range, self.lambda_bins)
        info_bin = self._discretize(descriptors[1], self.info_range, self.info_bins)
        entropy_bin = self._discretize(descriptors[2], self.entropy_range, self.entropy_bins)
        volatility_bin = self._discretize(descriptors[3], self.volatility_range, self.volatility_bins)
        
        return (lambda_bin, info_bin, entropy_bin, volatility_bin)
    
    def evaluate(self, genome: str, E: float = 200.0, T: float = 0.5) -> Tuple[float, Dict, np.ndarray]:
        """
        Evaluate a genome and return fitness + behavioral descriptors.
        
        Returns:
            (fitness, behavioral_dict, raw_descriptors)
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
        
        # Raw descriptors (before whitening)
        raw_descriptors = np.array([
            stats['lambda_dev'],
            stats['info_per_energy'],
            stats['entropy_slope'],
            stats['volatility']
        ])
        
        # Behavioral dict
        behavioral = {
            'lambda_dev': stats['lambda_dev'],
            'info_per_energy': stats['info_per_energy'],
            'entropy_slope': stats['entropy_slope'],
            'volatility': stats['volatility'],
            'fitness': fitness,
            'stats': stats
        }
        
        self.evaluations += 1
        
        return fitness, behavioral, raw_descriptors
    
    def _descriptor_distance(self, elite1: Elite, elite2: Elite) -> float:
        """Calculate Euclidean distance between two elites in descriptor space."""
        v1 = elite1.descriptor_vector()
        v2 = elite2.descriptor_vector()
        return float(np.linalg.norm(v1 - v2))
    
    def _check_min_change_guard(self, cell: Tuple, new_elite: Elite, new_fitness: float) -> bool:
        """Check if new elite passes min-change guard."""
        if cell not in self.grid:
            return True
        
        incumbent = self.grid[cell]
        
        # Check fitness improvement
        fitness_improvement = new_fitness - incumbent.fitness
        if fitness_improvement >= self.min_fitness_improvement:
            return True
        
        # Check descriptor distance
        descriptor_dist = self._descriptor_distance(new_elite, incumbent)
        if descriptor_dist >= self.min_descriptor_change:
            return True
        
        self.grid_spam_prevented += 1
        return False
    
    def add_to_grid(self, genome: str, E: float = 200.0, T: float = 0.5, parent_cell: Optional[Tuple] = None) -> bool:
        """
        Evaluate a genome and add it to the grid if it passes guards.
        
        Args:
            genome: Genome to evaluate
            E: Initial energy
            T: Initial temperature
            parent_cell: Parent cell for lineage tracking
        """
        fitness, behavioral, raw_descriptors = self.evaluate(genome, E, T)
        
        # Update whitener if enabled
        if self.use_whitening:
            self.whitener.update(raw_descriptors.reshape(1, -1))
            whitened_descriptors = self.whitener.whiten(raw_descriptors)
        else:
            whitened_descriptors = raw_descriptors
        
        # Get cell using whitened descriptors
        cell = self._get_cell(whitened_descriptors)
        
        # Determine ancestor ID
        if parent_cell is not None and parent_cell in self.lineage_map:
            ancestor_id = self.lineage_map[parent_cell]
        else:
            ancestor_id = self.next_ancestor_id
            self.next_ancestor_id += 1
        
        # Create elite
        elite = Elite(
            genome=genome,
            fitness=fitness,
            lambda_dev=behavioral['lambda_dev'],
            info_per_energy=behavioral['info_per_energy'],
            entropy_slope=behavioral['entropy_slope'],
            volatility=behavioral['volatility'],
            stats=behavioral['stats'],
            ancestor_id=ancestor_id
        )
        
        # Check min-change guard
        if not self._check_min_change_guard(cell, elite, fitness):
            return False
        
        # Check if this is an improvement
        if cell not in self.grid or fitness > self.grid[cell].fitness:
            self.grid[cell] = elite
            self.lineage_map[cell] = ancestor_id
            self.improvements += 1
            return True
        
        return False
    
    def compute_lineage_entropy(self) -> float:
        """
        Compute Shannon entropy over ancestor IDs.
        
        Detects monoculture creep even when fitness rises.
        """
        if not self.grid:
            return 0.0
        
        # Count ancestor IDs
        ancestor_counts = defaultdict(int)
        for elite in self.grid.values():
            ancestor_counts[elite.ancestor_id] += 1
        
        # Compute Shannon entropy
        total = len(self.grid)
        entropy = 0.0
        for count in ancestor_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def select_parent_random(self) -> Optional[Tuple[str, Tuple]]:
        """Random parent selection. Returns (genome, cell)."""
        if not self.grid:
            return None
        
        import random
        parent_cell = random.choice(list(self.grid.keys()))
        return self.grid[parent_cell].genome, parent_cell
    
    def select_parent_novelty(self) -> Optional[Tuple[str, Tuple]]:
        """Lineage-novelty parent selection. Returns (genome, cell)."""
        if len(self.grid) < 2:
            return self.select_parent_random()
        
        import random
        
        # Pick a random reference elite
        ref_cell = random.choice(list(self.grid.keys()))
        ref_elite = self.grid[ref_cell]
        
        # Find elite farthest from reference
        max_dist = 0.0
        farthest_cell = None
        farthest_genome = None
        
        for cell, elite in self.grid.items():
            dist = self._descriptor_distance(ref_elite, elite)
            if dist > max_dist:
                max_dist = dist
                farthest_genome = elite.genome
                farthest_cell = cell
        
        if farthest_genome:
            return farthest_genome, farthest_cell
        else:
            return self.select_parent_random()
    
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
        E: float = 200.0,
        T: float = 0.5,
        verbose: bool = True
    ):
        """
        Run MAP-Elites with adaptive emitters and whitening.
        """
        import random
        
        # Add initial genomes
        if verbose:
            print("Initializing grid with seed genomes...")
        
        for genome in initial_genomes:
            self.add_to_grid(genome, E, T)
        
        if verbose:
            print(f"Initial grid size: {len(self.grid)}")
            print(f"Initial lineage entropy: {self.compute_lineage_entropy():.3f}")
            if self.use_whitening:
                print(f"Descriptor whitening: ENABLED")
            if self.use_adaptive_emitters:
                print(f"Adaptive emitters: ENABLED")
            print()
        
        # Main loop
        window_improvements = 0
        
        for iteration in range(iterations):
            # Get emitter config
            if self.use_adaptive_emitters:
                emitter_config = self.emitter_scheduler.get_config()
                novelty_ratio = emitter_config['novelty_ratio']
                mutation_rate = emitter_config['mutation_radius']
            else:
                novelty_ratio = 0.3
                mutation_rate = 0.1
            
            # Select parent using mixed emitters
            if random.random() < novelty_ratio:
                parent_result = self.select_parent_novelty()
            else:
                parent_result = self.select_parent_random()
            
            if not parent_result:
                parent = random.choice(initial_genomes)
                parent_cell = None
            else:
                parent, parent_cell = parent_result
            
            # Create offspring
            if random.random() < 0.5 and len(self.grid) > 1:
                # Crossover
                if random.random() < novelty_ratio:
                    parent2_result = self.select_parent_novelty()
                else:
                    parent2_result = self.select_parent_random()
                
                if not parent2_result:
                    parent2 = random.choice(initial_genomes)
                else:
                    parent2, _ = parent2_result
                
                offspring = self.crossover(parent, parent2)
            else:
                # Mutation
                offspring = self.mutate(parent, mutation_rate)
            
            # Evaluate and add to grid
            improved = self.add_to_grid(offspring, E, T, parent_cell)
            if improved:
                window_improvements += 1
            
            # Update adaptive emitter scheduler
            if self.use_adaptive_emitters and (iteration + 1) % 100 == 0:
                self.emitter_scheduler.update(window_improvements, 100)
                self.improvements_per_window.append(window_improvements)
                window_improvements = 0
            
            # Progress report
            if verbose and (iteration + 1) % 100 == 0:
                total_cells = (self.lambda_bins * self.info_bins * 
                              self.entropy_bins * self.volatility_bins)
                coverage = len(self.grid) / total_cells
                lineage_entropy = self.compute_lineage_entropy()
                
                status = f"Iteration {iteration + 1}/{iterations}: "
                status += f"Grid={len(self.grid)}, "
                status += f"Coverage={coverage*100:.2f}%, "
                status += f"Improvements={self.improvements}, "
                status += f"Lineage H={lineage_entropy:.3f}"
                
                if self.use_adaptive_emitters:
                    config = self.emitter_scheduler.get_config()
                    status += f", Novelty={config['novelty_ratio']:.2f}"
                
                print(status)
        
        if verbose:
            print(f"\nFinal grid size: {len(self.grid)}")
            print(f"Final lineage entropy: {self.compute_lineage_entropy():.3f}")
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
                'use_whitening': self.use_whitening,
                'use_adaptive_emitters': self.use_adaptive_emitters
            },
            'stats': {
                'evaluations': self.evaluations,
                'improvements': self.improvements,
                'grid_spam_prevented': self.grid_spam_prevented,
                'grid_size': len(self.grid),
                'lineage_entropy': self.compute_lineage_entropy()
            },
            'elites': {
                f"{cell[0]},{cell[1]},{cell[2]},{cell[3]}": elite.to_dict()
                for cell, elite in self.grid.items()
            }
        }
        
        if self.use_whitening:
            data['whitener'] = {
                'mean': self.whitener.mean.tolist(),
                'var': self.whitener.var.tolist(),
                'n_samples': self.whitener.n_samples
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported grid to {filepath}")
    
    def plot_grid(self, output_path: str):
        """Visualize the MAP-Elites grid."""
        if not self.grid:
            print("Grid is empty, cannot plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MAP-Elites v3.1: Whitened 4D Behavior Space', 
                     fontsize=16, fontweight='bold')
        
        # Extract data
        cells = list(self.grid.keys())
        elites = list(self.grid.values())
        
        lambda_indices = [c[0] for c in cells]
        info_indices = [c[1] for c in cells]
        entropy_indices = [c[2] for c in cells]
        volatility_indices = [c[3] for c in cells]
        fitnesses = [e.fitness for e in elites]
        ancestor_ids = [e.ancestor_id for e in elites]
        
        # Lambda vs Info
        scatter1 = axes[0, 0].scatter(
            lambda_indices, info_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 0].set_xlabel('λ-deviation bin (whitened)')
        axes[0, 0].set_ylabel('Info-per-energy bin (whitened)')
        axes[0, 0].set_title('λ-deviation vs Info-per-energy')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Fitness')
        
        # Lambda vs Entropy
        scatter2 = axes[0, 1].scatter(
            lambda_indices, entropy_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 1].set_xlabel('λ-deviation bin (whitened)')
        axes[0, 1].set_ylabel('Entropy slope bin (whitened)')
        axes[0, 1].set_title('λ-deviation vs Entropy slope')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Fitness')
        
        # Lambda vs Volatility
        scatter3 = axes[0, 2].scatter(
            lambda_indices, volatility_indices,
            c=fitnesses, cmap='viridis',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[0, 2].set_xlabel('λ-deviation bin (whitened)')
        axes[0, 2].set_ylabel('Volatility bin (whitened)')
        axes[0, 2].set_title('λ-deviation vs Volatility')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[0, 2], label='Fitness')
        
        # Info vs Entropy (colored by lineage)
        scatter4 = axes[1, 0].scatter(
            info_indices, entropy_indices,
            c=ancestor_ids, cmap='tab20',
            s=100, alpha=0.7, edgecolors='black'
        )
        axes[1, 0].set_xlabel('Info-per-energy bin (whitened)')
        axes[1, 0].set_ylabel('Entropy slope bin (whitened)')
        axes[1, 0].set_title('Info vs Entropy (colored by lineage)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1, 0], label='Ancestor ID')
        
        # Fitness distribution
        axes[1, 1].hist(fitnesses, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Fitness')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Fitness Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Lineage diversity
        lineage_entropy = self.compute_lineage_entropy()
        axes[1, 2].text(0.5, 0.6, f'Lineage Entropy:\n{lineage_entropy:.3f}', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
        axes[1, 2].text(0.5, 0.4, f'Unique lineages:\n{len(set(ancestor_ids))}',
                       ha='center', va='center', fontsize=16)
        axes[1, 2].text(0.5, 0.2, f'Grid size:\n{len(self.grid)}',
                       ha='center', va='center', fontsize=16)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Lineage Diversity')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved MAP-Elites visualization to {output_path}")
        plt.close()


def main():
    """Command-line interface for MAP-Elites v3.1."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAP-Elites v3.1 for ChaosBF')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per evaluation')
    parser.add_argument('--seed', type=int, help='RNG seed')
    parser.add_argument('--no-whitening', action='store_true', help='Disable descriptor whitening')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive emitters')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/map_elites_v31', 
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
    
    # Run MAP-Elites v3.1
    map_elites = MAPElitesv31(
        lambda_bins=8,
        info_bins=8,
        entropy_bins=8,
        volatility_bins=8,
        steps_per_eval=args.steps,
        seed=args.seed,
        use_whitening=not args.no_whitening,
        use_adaptive_emitters=not args.no_adaptive
    )
    
    map_elites.run(
        initial_genomes=seeds,
        iterations=args.iterations,
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
        print(f"  Ancestor ID: {best.ancestor_id}")


if __name__ == '__main__':
    main()

