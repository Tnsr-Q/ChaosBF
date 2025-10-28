#!/usr/bin/env python3
"""
Island Ecology for ChaosBF

4-8 islands with distinct (E₀, T₀, θ_rep) priors.
Migration based on novelty deficit.
Observe speciation and cross-fertilization.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class IslandConfig:
    """Configuration for a single island."""
    id: int
    E0: float  # Initial energy
    T0: float  # Initial temperature
    theta_rep: float  # Replication threshold
    name: str


@dataclass
class Migrant:
    """Elite migrating between islands."""
    genome: str
    fitness: float
    source_island: int
    descriptors: np.ndarray


class Island:
    """
    Single island in the ecology.
    
    Each island has distinct thermodynamic conditions and evolves
    independently with occasional migration.
    """
    
    def __init__(
        self,
        config: IslandConfig,
        seed_genomes: List[str],
        seed: int = 42
    ):
        """
        Initialize island.
        
        Args:
            config: Island configuration
            seed_genomes: Initial genomes
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.RandomState(seed + config.id)
        
        # Population
        self.population = []
        for genome in seed_genomes:
            cbf = ChaosBFv3(
                genome,
                E=config.E0,
                T=config.T0,
                seed=seed + config.id + len(self.population),
                verbose=False
            )
            self.population.append(cbf)
        
        # Statistics
        self.generation = 0
        self.immigrants = 0
        self.emigrants = 0
        self.best_fitness = 0.0
        self.diversity = 0.0
    
    def evolve(self, steps: int = 1000):
        """Evolve island population."""
        for cbf in self.population:
            cbf.run(steps=steps)
        
        self.generation += 1
        self._update_stats()
    
    def _update_stats(self):
        """Update island statistics."""
        fitnesses = []
        for cbf in self.population:
            # Simple fitness: energy efficiency
            fitness = cbf.E / (cbf.steps + 1)
            fitnesses.append(fitness)
        
        if fitnesses:
            self.best_fitness = max(fitnesses)
            self.diversity = np.std(fitnesses)
    
    def get_elites(self, n: int = 5) -> List[Migrant]:
        """
        Get top elites for potential migration.
        
        Args:
            n: Number of elites to return
        
        Returns:
            List of migrants
        """
        # Compute fitness for all
        candidates = []
        for cbf in self.population:
            fitness = cbf.E / (cbf.steps + 1)
            descriptors = np.array([
                cbf.lambda_estimate,
                cbf.S,
                cbf.F,
                cbf.complexity_estimate
            ])
            candidates.append((cbf.code, fitness, descriptors))
        
        # Sort by fitness
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n as migrants
        migrants = []
        for i in range(min(n, len(candidates))):
            genome, fitness, descriptors = candidates[i]
            migrants.append(Migrant(
                genome=genome,
                fitness=fitness,
                source_island=self.config.id,
                descriptors=descriptors
            ))
        
        return migrants
    
    def accept_immigrant(self, migrant: Migrant):
        """
        Accept immigrant from another island.
        
        Args:
            migrant: Migrant to accept
        """
        # Create new CBF with island's conditions
        cbf = ChaosBFv3(
            migrant.genome,
            E=self.config.E0,
            T=self.config.T0,
            seed=self.rng.randint(0, 1000000),
            verbose=False
        )
        
        # Replace weakest individual
        if len(self.population) > 0:
            fitnesses = [c.E / (c.steps + 1) for c in self.population]
            weakest_idx = np.argmin(fitnesses)
            self.population[weakest_idx] = cbf
        else:
            self.population.append(cbf)
        
        self.immigrants += 1
    
    def compute_novelty_deficit(self, global_descriptors: List[np.ndarray]) -> float:
        """
        Compute novelty deficit relative to global population.
        
        Lower deficit = more novel = less likely to receive migrants.
        Higher deficit = less novel = more likely to receive migrants.
        
        Args:
            global_descriptors: Descriptors from all islands
        
        Returns:
            Novelty deficit score
        """
        if not self.population or not global_descriptors:
            return 0.0
        
        # Get local descriptors
        local_descriptors = []
        for cbf in self.population:
            desc = np.array([
                cbf.lambda_estimate,
                cbf.S,
                cbf.F,
                cbf.complexity_estimate
            ])
            local_descriptors.append(desc)
        
        # Compute average distance to global population
        total_dist = 0.0
        count = 0
        
        for local_desc in local_descriptors:
            for global_desc in global_descriptors:
                dist = np.linalg.norm(local_desc - global_desc)
                total_dist += dist
                count += 1
        
        avg_dist = total_dist / count if count > 0 else 0.0
        
        # Deficit = inverse of novelty (low distance = high deficit)
        deficit = 1.0 / (avg_dist + 0.1)
        
        return deficit
    
    def get_stats(self) -> Dict:
        """Get island statistics."""
        return {
            'id': self.config.id,
            'name': self.config.name,
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'diversity': self.diversity,
            'immigrants': self.immigrants,
            'emigrants': self.emigrants,
            'E0': self.config.E0,
            'T0': self.config.T0,
            'theta_rep': self.config.theta_rep
        }


class IslandEcology:
    """
    Manages multiple islands with migration.
    
    Observes speciation and cross-fertilization across thermal niches.
    """
    
    def __init__(
        self,
        n_islands: int = 4,
        seed_genomes: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize island ecology.
        
        Args:
            n_islands: Number of islands (4-8)
            seed_genomes: Initial genomes for all islands
            seed: Random seed
        """
        self.n_islands = n_islands
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Default seed genomes
        if seed_genomes is None:
            seed_genomes = [
                '++[>+<-].',
                ':{;}{?}^=.',
                '*=@=.#',
                '+[>+<-];.'
            ]
        
        # Create island configurations
        self.islands = []
        configs = self._create_island_configs(n_islands)
        
        for config in configs:
            island = Island(config, seed_genomes, seed)
            self.islands.append(island)
        
        self.generation = 0
    
    def _create_island_configs(self, n: int) -> List[IslandConfig]:
        """Create diverse island configurations."""
        configs = []
        
        # Thermal gradient
        E_range = np.linspace(150, 250, n)
        T_range = np.linspace(0.3, 0.7, n)
        theta_range = np.linspace(5.0, 15.0, n)
        
        names = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta']
        
        for i in range(n):
            config = IslandConfig(
                id=i,
                E0=E_range[i],
                T0=T_range[i],
                theta_rep=theta_range[i],
                name=names[i] if i < len(names) else f'Island-{i}'
            )
            configs.append(config)
        
        return configs
    
    def evolve(self, steps: int = 1000, migration_interval: int = 5):
        """
        Evolve all islands with periodic migration.
        
        Args:
            steps: Steps per generation
            migration_interval: Generations between migrations
        """
        # Evolve all islands
        for island in self.islands:
            island.evolve(steps)
        
        self.generation += 1
        
        # Periodic migration
        if self.generation % migration_interval == 0:
            self._migrate()
    
    def _migrate(self):
        """
        Perform migration between islands.
        
        Migration probability proportional to novelty deficit.
        """
        # Collect all descriptors
        global_descriptors = []
        for island in self.islands:
            for cbf in island.population:
                desc = np.array([
                    cbf.lambda_estimate,
                    cbf.S,
                    cbf.F,
                    cbf.complexity_estimate
                ])
                global_descriptors.append(desc)
        
        # Compute novelty deficits
        deficits = []
        for island in self.islands:
            deficit = island.compute_novelty_deficit(global_descriptors)
            deficits.append(deficit)
        
        # Normalize to probabilities
        total_deficit = sum(deficits)
        if total_deficit > 0:
            migration_probs = [d / total_deficit for d in deficits]
        else:
            migration_probs = [1.0 / len(self.islands)] * len(self.islands)
        
        # Each island sends migrants
        for source_island in self.islands:
            elites = source_island.get_elites(n=2)
            
            for migrant in elites:
                # Select destination based on novelty deficit
                dest_idx = self.rng.choice(len(self.islands), p=migration_probs)
                dest_island = self.islands[dest_idx]
                
                # Don't migrate to self
                if dest_island.config.id != source_island.config.id:
                    dest_island.accept_immigrant(migrant)
                    source_island.emigrants += 1
    
    def get_stats(self) -> Dict:
        """Get ecology statistics."""
        island_stats = [island.get_stats() for island in self.islands]
        
        return {
            'generation': self.generation,
            'n_islands': self.n_islands,
            'islands': island_stats,
            'total_population': sum(s['population_size'] for s in island_stats),
            'total_immigrants': sum(s['immigrants'] for s in island_stats),
            'total_emigrants': sum(s['emigrants'] for s in island_stats)
        }
    
    def print_stats(self):
        """Print ecology statistics."""
        stats = self.get_stats()
        
        print("="*80)
        print(f"Island Ecology - Generation {stats['generation']}")
        print("="*80)
        
        for island_stats in stats['islands']:
            print(f"\n{island_stats['name']} (ID {island_stats['id']}):")
            print(f"  E₀={island_stats['E0']:.1f}, T₀={island_stats['T0']:.2f}, θ_rep={island_stats['theta_rep']:.1f}")
            print(f"  Population: {island_stats['population_size']}")
            print(f"  Best Fitness: {island_stats['best_fitness']:.4f}")
            print(f"  Diversity: {island_stats['diversity']:.4f}")
            print(f"  Immigrants: {island_stats['immigrants']}, Emigrants: {island_stats['emigrants']}")
        
        print(f"\nTotal Population: {stats['total_population']}")
        print(f"Total Migration: {stats['total_immigrants']} immigrants, {stats['total_emigrants']} emigrants")
        print()


def main():
    """Demo of island ecology."""
    print("Initializing island ecology...")
    
    ecology = IslandEcology(n_islands=4, seed=42)
    
    print(f"Created {ecology.n_islands} islands with diverse thermal conditions.")
    print()
    
    # Evolve for several generations
    for gen in range(10):
        ecology.evolve(steps=500, migration_interval=3)
        
        if (gen + 1) % 3 == 0:
            ecology.print_stats()
    
    # Final statistics
    print("="*80)
    print("Final Ecology State")
    print("="*80)
    ecology.print_stats()


if __name__ == '__main__':
    main()

