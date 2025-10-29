#!/usr/bin/env python3
"""
ChaosBF v2.0: Production-grade thermodynamic, evolvable Brainfuck

Major improvements:
- Fixed thermodynamic accounting (entropy operator no longer double-penalizes)
- PID controller for criticality (λ ≈ 1)
- Grammar-aware mutation preserving bracket balance
- Improved learning with ΔF calculation
- Windowed λ estimator for stability
- Reproducible RNG seeding
- Enhanced observability and logging
"""

import random
import zlib
from math import log2
from typing import List, Optional, Dict, Tuple
import sys
import json
from dataclasses import dataclass, asdict
from collections import deque


OPS = "><+-[].,^v:;?*@=!{}#%~"
SAFE_OPS = "><+-.,^v:?*@=!#%~"  # Exclude brackets for safe mutation


def K(b: str) -> float:
    """Kolmogorov complexity proxy using compression."""
    if not b:
        return 0.0
    try:
        c = zlib.compress(b.encode('latin1'))
        return max(0.0, len(b) - len(c))
    except:
        return float(len(b))


@dataclass
class ThermodynamicState:
    """Snapshot of thermodynamic state for analysis."""
    step: int
    energy: float
    temperature: float
    entropy: float
    free_energy: float
    branching_factor: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class ChaosBFv2:
    """
    ChaosBF v2.0 interpreter with production-grade features.
    """
    
    def __init__(
        self,
        code: str,
        E: float = 200.0,
        T: float = 0.5,
        tau: float = 0.1,
        theta: float = 5.0,
        w: int = 8,
        tape_size: int = 65536,
        seed: Optional[int] = None,
        # PID controller parameters
        use_pid: bool = True,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        pid_interval: int = 50,
        T_min: float = 0.01,
        T_max: float = 2.0,
        # Mutation parameters
        grammar_aware: bool = True,
        wild_mutation_rate: float = 0.1,
        # Learning parameters
        learning_cap: float = 0.5,
        # Leak parameters
        leak_max: float = 10.0,
        # Lambda estimation
        lambda_window: int = 100,
        verbose: bool = False
    ):
        """
        Initialize ChaosBF v2.0 runtime.
        
        Args:
            code: Initial program code
            E: Initial energy
            T: Initial temperature
            tau: Temperature adjustment step (manual)
            theta: Free energy threshold for replication
            w: Window size for entropy sensing
            tape_size: Memory tape size
            seed: RNG seed for reproducibility
            use_pid: Enable PID controller for criticality
            kp, ki, kd: PID controller gains
            pid_interval: Steps between PID updates
            T_min, T_max: Temperature bounds
            grammar_aware: Use grammar-aware mutation
            wild_mutation_rate: Probability of wild (bracket-breaking) mutation
            learning_cap: Maximum fraction of ΔF to credit as energy
            leak_max: Maximum energy leak per loop iteration
            lambda_window: Window size for λ estimation
            verbose: Print detailed execution info
        """
        # Set random seed for reproducibility
        self.seed = seed if seed is not None else random.randint(0, 2**32-1)
        self.rng = random.Random(self.seed)
        
        self.code = list(code)
        self.ip = 0
        self.stack = []
        
        # Memory tape
        self.M = [0] * tape_size
        self.p = 0
        self.tape_size = tape_size
        
        # Output buffer
        self.O = ""
        
        # Thermodynamic state
        self.E = E
        self.E_initial = E
        self.T = T
        self.S = 0.0
        self.tau = tau
        self.theta = theta
        self.w = w
        self.leak_max = leak_max
        
        # Evolution state
        self.G = []  # Genome bank
        self.elite = []  # Elite genomes
        
        # Criticality tracking with windowed estimator
        self.lambda_window = lambda_window
        self.branch_children = deque(maxlen=lambda_window)
        self.branch_parents = deque(maxlen=lambda_window)
        
        # Local entropy cache
        self.slocal = 0.0
        
        # PID controller state
        self.use_pid = use_pid
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pid_interval = pid_interval
        self.T_min = T_min
        self.T_max = T_max
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        
        # Mutation parameters
        self.grammar_aware = grammar_aware
        self.wild_mutation_rate = wild_mutation_rate
        
        # Learning parameters
        self.learning_cap = learning_cap
        
        # Statistics
        self.steps_executed = 0
        self.mutations = 0
        self.mutations_wild = 0
        self.replications = 0
        self.crossovers = 0
        self.learns = 0
        self.pid_updates = 0
        
        # Temporal derivatives for analysis
        self.prev_S = 0.0
        self.prev_K = 0.0
        self.dS_dt = 0.0
        self.dK_dt = 0.0
        
        # PID controller state for logging
        self.pid_p = 0.0
        self.pid_i = 0.0
        self.pid_d = 0.0
        
        self.verbose = verbose
        
    def sense_entropy(self) -> float:
        """Calculate local entropy using Shannon entropy."""
        a = max(0, self.p - self.w)
        b = min(len(self.M), self.p + self.w + 1)
        window = self.M[a:b]
        
        if not window:
            return 0.0
        
        total = len(window)
        hist = {}
        for v in window:
            hist[v] = hist.get(v, 0) + 1
        
        H = 0.0
        for count in hist.values():
            if count > 0:
                p = count / total
                H -= p * log2(p)
        
        return H
    
    def deltaE(self, op: str, depth: int = 0) -> float:
        """
        Calculate energy change for an operation.
        
        CORRECTNESS FIX: ';' now has ΔE=0 (only affects S, not E)
        """
        # Depth-dependent leak with cap
        leak = 0.0
        if op in '[]{}':
            leak = min(self.leak_max, 1.0 + depth / 3.0)
        
        base_costs = {
            '>': -1, '<': -1,
            '+': -2, '-': +1,
            '.': -1, ',': -1,
            '^': -1, 'v': -1,
            ':': 0,
            ';': 0,  # FIXED: No energy cost, only entropy change
            '?': -2, '*': -10, '@': -6,
            '=': 0,  # Energy gain calculated separately based on ΔF
            '!': -1,
            '{': -2, '}': 0,
            '#': 0, '%': -1, '~': +5
        }
        
        return base_costs.get(op, 0) - leak
    
    def mutate_one(self):
        """
        Mutate a random instruction.
        
        UPGRADE: Grammar-aware mutation preserving bracket balance.
        """
        if not self.code:
            return
        
        i = self.rng.randrange(len(self.code))
        c = self.code[i]
        
        # Grammar-aware mutation (90% of time)
        if self.grammar_aware and self.rng.random() > self.wild_mutation_rate:
            # Preserve bracket structure
            if c in '[]{}':
                # Don't mutate brackets in safe mode
                return
            else:
                # Mutate to non-bracket operator
                self.code[i] = self.rng.choice(SAFE_OPS)
        else:
            # Wild mutation (can break brackets)
            self.code[i] = self.rng.choice(OPS)
            self.mutations_wild += 1
        
        self.mutations += 1
    
    def crossover(self):
        """Perform single-point crossover with energy-aware cut point selection."""
        if not self.G or not self.code:
            return
        
        g = self.rng.choice(self.G)
        k = min(len(g), len(self.code))
        
        if k > 2:
            cut = self.rng.randrange(1, k - 1)
        else:
            cut = 1
        
        self.code = self.code[:cut] + list(g)[cut:]
        self.crossovers += 1
    
    def replicate(self):
        """Replicate current genome with mutation burst."""
        g = ''.join(self.code)
        self.G.append(g)
        
        # Mutation burst proportional to T
        burst_size = max(1, int(self.T * 3))
        for _ in range(burst_size):
            if self.code:
                self.mutate_one()
        
        self.replications += 1
        return g
    
    def learn(self):
        """
        Peephole optimization with thermodynamically correct ΔF calculation.
        
        UPGRADE: Only credit energy if ΔF > 0, capped at learning_cap * ΔF.
        """
        s_before = ''.join(self.code)
        
        # Calculate energy cost before optimization
        E_before = self._estimate_energy_cost(s_before)
        
        # Apply peephole optimizations
        s_after = s_before.replace('+-', '').replace('-+', '')
        s_after = s_after.replace('><', '').replace('<>', '')
        
        if s_before == s_after:
            return  # No change
        
        # Calculate energy cost after optimization
        E_after = self._estimate_energy_cost(s_after)
        
        # Energy saved
        dE = E_before - E_after
        
        # Entropy change (simplification reduces entropy)
        # Estimate: shorter code = lower entropy
        dS = max(0.0, K(s_before) - K(s_after)) * 0.1  # Scale factor
        
        # Free energy gain
        dF = dE - self.T * dS
        
        # Credit only capped fraction of positive ΔF
        if dF > 0:
            energy_gain = self.learning_cap * dF
            self.E += energy_gain
            self.learns += 1
        
        self.code = list(s_after)
    
    def _estimate_energy_cost(self, code: str) -> float:
        """Estimate total energy cost of executing a code segment."""
        cost = 0.0
        depth = 0
        for c in code:
            if c == '[' or c == '{':
                depth += 1
            elif c == ']' or c == '}':
                depth = max(0, depth - 1)
            cost += abs(self.deltaE(c, depth))
        return cost
    
    def free_energy(self) -> float:
        """Calculate free energy F = E - T·S"""
        return self.E - self.T * self.S
    
    def branching_factor(self) -> float:
        """
        Estimate branching factor using windowed offspring/parent ratio.
        
        UPGRADE: More stable estimator than simple mean.
        """
        if not self.branch_children or not self.branch_parents:
            return 1.0
        
        total_children = sum(self.branch_children)
        total_parents = sum(self.branch_parents) + 1e-6  # Avoid division by zero
        
        return total_children / total_parents
    
    def update_pid_controller(self):
        """
        Update temperature using PID controller to maintain λ ≈ 1.
        
        UPGRADE: Closed-loop criticality control.
        """
        if not self.use_pid:
            return
        
        lambda_hat = self.branching_factor()
        error = lambda_hat - 1.0
        
        # PID terms
        self.pid_p = self.kp * error
        self.pid_integral += error
        self.pid_i = self.ki * self.pid_integral
        self.pid_d = self.kd * (error - self.pid_prev_error)
        
        # Update temperature
        dT = self.pid_p + self.pid_i + self.pid_d
        self.T = max(self.T_min, min(self.T_max, self.T + dT))
        
        self.pid_prev_error = error
        self.pid_updates += 1
        
        if self.verbose:
            print(f"PID: λ={lambda_hat:.3f}, err={error:.3f}, P={self.pid_p:.3f}, I={self.pid_i:.3f}, D={self.pid_d:.3f}, T={self.T:.3f}")
    
    def update_temporal_derivatives(self):
        """Update dS/dt and dK/dt for analysis."""
        current_K = K(self.O)
        
        self.dS_dt = self.S - self.prev_S
        self.dK_dt = current_K - self.prev_K
        
        self.prev_S = self.S
        self.prev_K = current_K
    
    def step(self):
        """Execute one instruction."""
        if not self.code:
            return False
        
        # Wrap instruction pointer
        self.ip = self.ip % len(self.code)
        
        op = self.code[self.ip]
        depth = len(self.stack)
        
        # Update local entropy if needed
        if op == ':':
            self.slocal = self.sense_entropy()
        
        # Apply energy cost
        self.E += self.deltaE(op, depth)
        
        # Execute operation
        if op == '>':
            self.p = (self.p + 1) % self.tape_size
            
        elif op == '<':
            self.p = (self.p - 1) % self.tape_size
            
        elif op == '+':
            self.M[self.p] = (self.M[self.p] + 1) & 0xFF
            
        elif op == '-':
            self.M[self.p] = (self.M[self.p] - 1) & 0xFF
            
        elif op == '.':
            self.O += chr(self.M[self.p])
            
        elif op == ',':
            self.M[self.p] = 0
            
        elif op == '[':
            if self.M[self.p] == 0:
                bal = 1
                while bal and self.ip + 1 < len(self.code):
                    self.ip += 1
                    c = self.code[self.ip]
                    if c == '[':
                        bal += 1
                    elif c == ']':
                        bal -= 1
            else:
                self.stack.append(self.ip)
                
        elif op == ']':
            if self.M[self.p] != 0 and self.stack:
                self.ip = self.stack[-1]
            else:
                if self.stack:
                    self.stack.pop()
                    
        elif op == '^':
            self.T += self.tau
            self.T = min(self.T_max, self.T)
            
        elif op == 'v':
            self.T = max(self.T_min, self.T - self.tau)
            
        elif op == ':':
            # Entropy sensing (already done above)
            pass
            
        elif op == ';':
            # FIXED: Only update entropy, no energy cost
            self.S += self.slocal
            
        elif op == '?':
            # Thermal mutation
            p_mut = min(0.95, 0.2 + 0.6 * self.T)
            if self.rng.random() < p_mut:
                self.mutate_one()
                
        elif op == '*':
            # Replicate if free energy allows
            if self.free_energy() > self.theta:
                self.replicate()
                
        elif op == '@':
            self.crossover()
            
        elif op == '=':
            self.learn()
            
        elif op == '!':
            # Selection
            score = K(self.O) / (1 + abs(self.E - self.E_initial))
            if score > 1.0:
                self.elite.append(''.join(self.code))
                
        elif op == '{':
            # Critical loop: record branching
            p_branch = min(0.9, 0.3 + 0.4 * self.T)
            if self.rng.random() < p_branch:
                self.M[self.p] ^= 1
                self.branch_children.append(1)
            else:
                self.branch_children.append(0)
            self.branch_parents.append(1)
            
        elif op == '}':
            pass
            
        elif op == '#':
            # Measure
            F = self.free_energy()
            lam = self.branching_factor()
            print(f"E={self.E:.1f} T={self.T:.2f} S={self.S:.2f} F={F:.2f} λ~{lam:.2f}")
            
        elif op == '%':
            # Phenotype dump
            print(self.O)
            self.O = ""
            
        elif op == '~':
            # Quench
            if self.free_energy() < 0 and self.elite:
                self.code = list(self.rng.choice(self.elite))
                self.T *= 0.8
        
        # Advance instruction pointer
        self.ip = self.ip + 1
        self.steps_executed += 1
        
        # PID controller update
        if self.use_pid and self.steps_executed % self.pid_interval == 0:
            self.update_pid_controller()
        
        # Update temporal derivatives periodically
        if self.steps_executed % 10 == 0:
            self.update_temporal_derivatives()
        
        return True
    
    def run(self, steps: int = 20000, halt_at_end: bool = False):
        """Run the interpreter."""
        if self.verbose:
            print(f"Starting ChaosBF v2.0")
            print(f"  Seed: {self.seed}")
            print(f"  E={self.E}, T={self.T}")
            print(f"  PID: {'enabled' if self.use_pid else 'disabled'}")
            print(f"  Grammar-aware mutation: {self.grammar_aware}")
            print(f"  Code: {''.join(self.code)}")
            print()
        
        while self.E > 0 and steps > 0:
            if halt_at_end and self.ip >= len(self.code):
                break
            self.step()
            steps -= 1
        
        print("\n=== HALT ===")
        self.print_stats()
        
        if self.O:
            print(f"\nFinal Output: {self.O}")
        
        return self.O
    
    def print_stats(self):
        """Print final statistics."""
        print(f"Final state:")
        print(f"  Seed: {self.seed}")
        print(f"  Energy: {self.E:.2f}")
        print(f"  Temperature: {self.T:.2f}")
        print(f"  Entropy: {self.S:.2f}")
        print(f"  Free Energy: {self.free_energy():.2f}")
        print(f"  Branching Factor: {self.branching_factor():.2f}")
        print(f"  Steps: {self.steps_executed}")
        print(f"  Mutations: {self.mutations} (wild: {self.mutations_wild})")
        print(f"  Replications: {self.replications}")
        print(f"  Crossovers: {self.crossovers}")
        print(f"  Learns: {self.learns}")
        print(f"  PID Updates: {self.pid_updates}")
        print(f"  Genome Bank Size: {len(self.G)}")
        print(f"  Elite Count: {len(self.elite)}")
        print(f"  Output Complexity: {K(self.O):.2f}")
        print(f"  dS/dt: {self.dS_dt:.3f}")
        print(f"  dK/dt: {self.dK_dt:.3f}")
    
    def get_stats(self) -> dict:
        """Return current statistics as a dictionary."""
        return {
            'seed': self.seed,
            'energy': self.E,
            'temperature': self.T,
            'entropy': self.S,
            'free_energy': self.free_energy(),
            'branching_factor': self.branching_factor(),
            'steps': self.steps_executed,
            'mutations': self.mutations,
            'mutations_wild': self.mutations_wild,
            'replications': self.replications,
            'crossovers': self.crossovers,
            'learns': self.learns,
            'pid_updates': self.pid_updates,
            'genome_bank_size': len(self.G),
            'elite_count': len(self.elite),
            'output_length': len(self.O),
            'output_complexity': K(self.O),
            'dS_dt': self.dS_dt,
            'dK_dt': self.dK_dt,
            'pid_p': self.pid_p,
            'pid_i': self.pid_i,
            'pid_d': self.pid_d
        }
    
    def snapshot(self) -> dict:
        """Create a complete snapshot of the current state."""
        return {
            'code': ''.join(self.code),
            'ip': self.ip,
            'tape': self.M[:100],  # First 100 cells
            'pointer': self.p,
            'output': self.O,
            'stats': self.get_stats(),
            'genome_bank': self.G.copy(),
            'elite': self.elite.copy()
        }


def main():
    """Command-line interface for ChaosBF v2.0."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaosBF v2.0: Thermodynamic, evolvable Brainfuck')
    parser.add_argument('code', help='Program code')
    parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    parser.add_argument('--steps', type=int, default=20000, help='Max steps')
    parser.add_argument('--seed', type=int, help='RNG seed for reproducibility')
    parser.add_argument('--no-pid', action='store_true', help='Disable PID controller')
    parser.add_argument('--no-grammar', action='store_true', help='Disable grammar-aware mutation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    cbf = ChaosBFv2(
        args.code,
        E=args.energy,
        T=args.temp,
        seed=args.seed,
        use_pid=not args.no_pid,
        grammar_aware=not args.no_grammar,
        verbose=args.verbose
    )
    
    cbf.run(steps=args.steps)


if __name__ == '__main__':
    main()

