#!/usr/bin/env python3
"""
Example seed programs for ChaosBF demonstrating various features.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf import ChaosBF


# Example programs from the specification
EXAMPLES = {
    'hot_seed': {
        'code': '++[>+<-].:{;}{?}^*=@=.#%{?}{?}v=.#',
        'description': 'Hot, minimal, self-editing seed with entropy sensing, mutation, replication, crossover, and learning',
        'params': {'E': 180, 'T': 0.6, 'steps': 4000}
    },
    
    'entropy_harvester': {
        'code': ':+[>+<-];.#',
        'description': 'Senses entropy, increments, loops, accrues entropy, outputs, and measures',
        'params': {'E': 100, 'T': 0.3, 'steps': 1000}
    },
    
    'critical_mutator': {
        'code': '{?}{?}{?}^=v=.#',
        'description': 'Critical branching with multiple mutations, annealing, and learning',
        'params': {'E': 120, 'T': 0.7, 'steps': 2000}
    },
    
    'replicate_learn_loop': {
        'code': '*=@=:{;}#%',
        'description': 'Replication, learning, crossover, entropy sensing/accrual, measurement, and dump',
        'params': {'E': 150, 'T': 0.5, 'steps': 3000}
    },
    
    'hello_world': {
        'code': '+++++++++[>++++++++<-]>.#%',
        'description': 'Classic Hello World (outputs "H") with measurement and dump',
        'params': {'E': 100, 'T': 0.2, 'steps': 500}
    },
    
    'evolving_counter': {
        'code': '++++[>++++<-]>{?}*=@.#%',
        'description': 'Counter that mutates, replicates, learns, and outputs',
        'params': {'E': 150, 'T': 0.6, 'steps': 2000}
    },
    
    'thermodynamic_oscillator': {
        'code': '++[>+<-]^v^v:{;}#.%',
        'description': 'Oscillates temperature while sensing entropy',
        'params': {'E': 120, 'T': 0.4, 'steps': 1500}
    },
    
    'edge_of_chaos': {
        'code': '{+}{?}^{-}{?}v*=@:{;}#%',
        'description': 'Operates at edge-of-chaos with critical branching, mutation, replication, and learning',
        'params': {'E': 200, 'T': 0.5, 'steps': 5000}
    },
    
    'self_optimizer': {
        'code': '+++[>+++<-]===.{?}*@=.#%',
        'description': 'Repeatedly learns to optimize itself while mutating and replicating',
        'params': {'E': 180, 'T': 0.55, 'steps': 3500}
    },
    
    'population_seed': {
        'code': '++{+}*{?}@*{-}=:{;}!#%',
        'description': 'Builds population through replication, mutation, crossover, with elite selection',
        'params': {'E': 250, 'T': 0.65, 'steps': 6000}
    },
    
    'quench_survivor': {
        'code': '+++[>++<-]{?}*@~.#%',
        'description': 'Mutates and replicates, then uses quench to survive low free energy',
        'params': {'E': 100, 'T': 0.8, 'steps': 2500}
    },
    
    'complex_evolver': {
        'code': '++[>++<-]:{;}^{?}*=@{?}v=!#%~',
        'description': 'Complex program with all major features: entropy, mutation, replication, learning, selection, quench',
        'params': {'E': 300, 'T': 0.6, 'steps': 8000}
    }
}


def run_example(name: str, verbose: bool = False):
    """Run a specific example by name."""
    if name not in EXAMPLES:
        print(f"Unknown example: {name}")
        print(f"Available examples: {', '.join(EXAMPLES.keys())}")
        return
    
    ex = EXAMPLES[name]
    print(f"=== {name.upper().replace('_', ' ')} ===")
    print(f"Description: {ex['description']}")
    print(f"Code: {ex['code']}")
    print(f"Parameters: E={ex['params']['E']}, T={ex['params']['T']}, steps={ex['params']['steps']}")
    print()
    
    cbf = ChaosBF(
        ex['code'],
        E=ex['params']['E'],
        T=ex['params']['T'],
        verbose=verbose
    )
    
    cbf.run(steps=ex['params']['steps'])
    print()


def run_all_examples(verbose: bool = False):
    """Run all example programs."""
    for name in EXAMPLES.keys():
        run_example(name, verbose)
        print("\n" + "="*80 + "\n")


def main():
    """Command-line interface for running examples."""
    if len(sys.argv) < 2:
        print("Usage: python seed_programs.py <example_name|all> [--verbose]")
        print("\nAvailable examples:")
        for name, ex in EXAMPLES.items():
            print(f"  {name:25s} - {ex['description']}")
        sys.exit(1)
    
    name = sys.argv[1]
    verbose = '--verbose' in sys.argv
    
    if name == 'all':
        run_all_examples(verbose)
    else:
        run_example(name, verbose)


if __name__ == '__main__':
    main()

