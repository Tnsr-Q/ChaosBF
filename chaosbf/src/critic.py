#!/usr/bin/env python3
"""
Critic-in-the-Loop for ChaosBF

Tiny regex/grammar critic predicts phenotype tokens.
Fitness bonus for surprising the critic.
Bootstraps semantics without external models.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


class PhenotypeCritic:
    """
    Learns to predict phenotype patterns.
    
    Gives fitness bonus to programs that surprise the critic
    (high prediction error = novel behavior).
    """
    
    def __init__(
        self,
        ngram_size: int = 3,
        learning_rate: float = 0.1,
        surprise_weight: float = 1.0
    ):
        """
        Initialize critic.
        
        Args:
            ngram_size: Size of n-grams for prediction
            learning_rate: Learning rate for updating predictions
            surprise_weight: Weight for surprise bonus
        """
        self.ngram_size = ngram_size
        self.learning_rate = learning_rate
        self.surprise_weight = surprise_weight
        
        # N-gram model: context -> next token probabilities
        self.ngram_counts = defaultdict(Counter)
        self.total_observations = 0
        
        # Pattern library (regex-like)
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Initialize pattern library."""
        patterns = [
            ('digits', re.compile(r'\d+')),
            ('letters', re.compile(r'[a-zA-Z]+')),
            ('whitespace', re.compile(r'\s+')),
            ('punctuation', re.compile(r'[.,!?;:]+')),
            ('brackets', re.compile(r'[\[\]{}()]+')),
            ('operators', re.compile(r'[+\-*/=<>]+')),
            ('special', re.compile(r'[^a-zA-Z0-9\s]+')),
        ]
        return patterns
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        return list(text)
    
    def _get_context(self, tokens: List[str], pos: int) -> str:
        """Get n-gram context at position."""
        start = max(0, pos - self.ngram_size + 1)
        context = ''.join(tokens[start:pos])
        return context
    
    def predict(self, context: str) -> Dict[str, float]:
        """
        Predict next token probabilities given context.
        
        Args:
            context: Context string
        
        Returns:
            Dictionary of token -> probability
        """
        if context not in self.ngram_counts:
            # Uniform prior
            return {}
        
        counts = self.ngram_counts[context]
        total = sum(counts.values())
        
        if total == 0:
            return {}
        
        probs = {token: count / total for token, count in counts.items()}
        return probs
    
    def surprise(self, context: str, actual_token: str) -> float:
        """
        Compute surprise (negative log probability) of actual token.
        
        Higher surprise = more novel/unexpected.
        
        Args:
            context: Context string
            actual_token: Actual next token
        
        Returns:
            Surprise value
        """
        probs = self.predict(context)
        
        if not probs or actual_token not in probs:
            # Maximum surprise for unseen tokens
            return 10.0
        
        prob = probs[actual_token]
        surprise = -np.log(prob + 1e-10)
        
        return surprise
    
    def learn(self, text: str):
        """
        Learn from observed phenotype.
        
        Args:
            text: Phenotype text to learn from
        """
        tokens = self._tokenize(text)
        
        for i in range(len(tokens)):
            context = self._get_context(tokens, i)
            token = tokens[i]
            
            # Update n-gram counts
            self.ngram_counts[context][token] += 1
            self.total_observations += 1
    
    def evaluate(self, text: str) -> float:
        """
        Evaluate phenotype and return surprise score.
        
        Higher score = more surprising/novel.
        
        Args:
            text: Phenotype text to evaluate
        
        Returns:
            Average surprise score
        """
        if not text:
            return 0.0
        
        tokens = self._tokenize(text)
        total_surprise = 0.0
        count = 0
        
        for i in range(len(tokens)):
            context = self._get_context(tokens, i)
            token = tokens[i]
            
            surprise = self.surprise(context, token)
            total_surprise += surprise
            count += 1
        
        avg_surprise = total_surprise / count if count > 0 else 0.0
        return avg_surprise
    
    def fitness_bonus(self, text: str) -> float:
        """
        Compute fitness bonus for surprising the critic.
        
        Args:
            text: Phenotype text
        
        Returns:
            Fitness bonus
        """
        surprise_score = self.evaluate(text)
        bonus = self.surprise_weight * surprise_score
        
        return bonus
    
    def get_stats(self) -> Dict:
        """Get critic statistics."""
        return {
            'ngram_size': self.ngram_size,
            'total_observations': self.total_observations,
            'unique_contexts': len(self.ngram_counts),
            'avg_tokens_per_context': np.mean([len(counts) for counts in self.ngram_counts.values()]) if self.ngram_counts else 0.0
        }
    
    def print_stats(self):
        """Print critic statistics."""
        stats = self.get_stats()
        print("="*60)
        print("Critic Statistics")
        print("="*60)
        print(f"N-gram size: {stats['ngram_size']}")
        print(f"Total observations: {stats['total_observations']}")
        print(f"Unique contexts: {stats['unique_contexts']}")
        print(f"Avg tokens per context: {stats['avg_tokens_per_context']:.2f}")
        print()


class CriticEvolution:
    """
    Co-evolution of programs and critic.
    
    Programs evolve to surprise the critic.
    Critic learns from observed phenotypes.
    """
    
    def __init__(
        self,
        critic: PhenotypeCritic,
        population_size: int = 10,
        seed: int = 42
    ):
        """
        Initialize critic evolution.
        
        Args:
            critic: Phenotype critic
            population_size: Size of program population
            seed: Random seed
        """
        self.critic = critic
        self.population_size = population_size
        self.rng = np.random.RandomState(seed)
        
        # Population
        self.population = []
        self.generation = 0
    
    def initialize_population(self, seed_genomes: List[str]):
        """Initialize population with seed genomes."""
        for genome in seed_genomes[:self.population_size]:
            cbf = ChaosBFv3(
                genome,
                E=200,
                T=0.5,
                seed=self.rng.randint(0, 1000000),
                verbose=False
            )
            self.population.append(cbf)
    
    def evolve_generation(self, steps: int = 1000):
        """
        Evolve one generation with critic feedback.
        
        Args:
            steps: Steps per program
        """
        # Run all programs
        for cbf in self.population:
            cbf.run(steps=steps)
        
        # Evaluate with critic
        fitnesses = []
        for cbf in self.population:
            # Base fitness: energy efficiency
            base_fitness = cbf.E / (cbf.steps + 1)
            
            # Critic bonus: surprise score
            surprise_bonus = self.critic.fitness_bonus(cbf.output)
            
            total_fitness = base_fitness + surprise_bonus
            fitnesses.append((cbf, total_fitness, surprise_bonus))
        
        # Learn from all phenotypes
        for cbf in self.population:
            self.critic.learn(cbf.output)
        
        # Select and reproduce
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top half, mutate to create new population
        elite_size = self.population_size // 2
        elites = [f[0] for f in fitnesses[:elite_size]]
        
        new_population = elites.copy()
        
        # Create offspring by mutation
        while len(new_population) < self.population_size:
            parent = self.rng.choice(elites)
            
            # Create mutated offspring
            offspring = ChaosBFv3(
                parent.code,
                E=200,
                T=0.5,
                seed=self.rng.randint(0, 1000000),
                verbose=False
            )
            offspring.mutate_one()
            
            new_population.append(offspring)
        
        self.population = new_population
        self.generation += 1
        
        # Return best fitness and surprise
        best_fitness = fitnesses[0][1]
        best_surprise = fitnesses[0][2]
        
        return best_fitness, best_surprise
    
    def get_stats(self) -> Dict:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'critic_stats': self.critic.get_stats()
        }


def main():
    """Demo of critic-in-the-loop."""
    print("="*80)
    print("Critic-in-the-Loop Demo")
    print("="*80)
    print()
    
    # Create critic
    critic = PhenotypeCritic(ngram_size=3, surprise_weight=0.5)
    
    # Test surprise evaluation
    print("Testing surprise evaluation...")
    
    # Learn from some examples
    critic.learn("Hello world")
    critic.learn("Hello there")
    critic.learn("Hello friend")
    
    # Evaluate novel vs familiar
    familiar = "Hello world"
    novel = "Goodbye universe"
    
    surprise_familiar = critic.evaluate(familiar)
    surprise_novel = critic.evaluate(novel)
    
    print(f"Familiar text: '{familiar}'")
    print(f"  Surprise: {surprise_familiar:.4f}")
    print()
    print(f"Novel text: '{novel}'")
    print(f"  Surprise: {surprise_novel:.4f}")
    print()
    
    critic.print_stats()
    
    # Test co-evolution
    print("Testing co-evolution...")
    
    evolution = CriticEvolution(critic, population_size=5, seed=42)
    
    seed_genomes = [
        '++[>+<-].',
        ':{;}{?}^=.',
        '*=@=.#',
        '+[>+<-];.',
        '++.'
    ]
    
    evolution.initialize_population(seed_genomes)
    
    print(f"Initialized population with {len(evolution.population)} programs")
    print()
    
    # Evolve for a few generations
    for gen in range(3):
        best_fitness, best_surprise = evolution.evolve_generation(steps=500)
        print(f"Generation {gen + 1}: Best fitness = {best_fitness:.4f}, Best surprise = {best_surprise:.4f}")
    
    print()
    critic.print_stats()


if __name__ == '__main__':
    main()

