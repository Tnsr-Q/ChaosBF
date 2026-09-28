#!/usr/bin/env python3
"""
ChaosBF (CBF): A thermodynamic, evolvable Brainfuck derivative

A language that operates at the edge-of-chaos with:
- Thermodynamic energy/entropy/free-energy model
- Self-replication and mutation
- Crossover and learning
- Criticality control (λ≈1)
- Open-ended evolution
"""

import random
import zlib
from math import log2
from typing import List, Optional
import sys


OPS = "><+-[].,^v:;?*@=!{}#%~"


def K(b: str) -> float:
    """
    Complexity proxy using compression.
    K(x) ≈ len(x) - len(compress(x))
    """
    if not b:
        return 0.0
    try:
        c = zlib.compress(b.encode('latin1'))
        return max(0.0, len(b) - len(c))
    except:
        return float(len(b))


class ChaosBF:
    """
    ChaosBF interpreter with thermodynamic state and evolutionary operators.
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
        verbose: bool = False
    ):
        """
        Initialize ChaosBF runtime.
        
        Args:
            code: Initial program code
            E: Initial energy
            T: Initial temperature
            tau: Temperature adjustment step
            theta: Free energy threshold for replication
            w: Window size for entropy sensing
            tape_size: Memory tape size
            verbose: Print detailed execution info
        """
        self.code = list(code)
        self.ip = 0  # instruction pointer
        self.stack = []  # loop stack
        
        # Memory tape
        self.M = [0] * tape_size
        self.p = 0  # tape pointer
        self.tape_size = tape_size
        
        # Output buffer
        self.O = ""
        
        # Thermodynamic state
        self.E = E  # Energy
        self.T = T  # Temperature
        self.S = 0.0  # Entropy
        self.tau = tau  # Temperature step
        self.theta = theta  # Replication threshold
        self.w = w  # Entropy window
        
        # Evolution state
        self.G = []  # Genome bank
        self.elite = []  # Elite genomes
        
        # Criticality tracking
        self.branch_hist = [1]
        
        # Local entropy cache
        self.slocal = 0.0
        
        # Statistics
        self.steps_executed = 0
        self.mutations = 0
        self.replications = 0
        self.crossovers = 0
        self.learns = 0
        
        self.verbose = verbose
        
    def sense_entropy(self) -> float:
        """
        Calculate local entropy in a window around the tape pointer.
        Uses Shannon entropy over byte histogram.
        """
        a = max(0, self.p - self.w)
        b = min(len(self.M), self.p + self.w + 1)
        window = self.M[a:b]
        
        if not window:
            return 0.0
        
        total = len(window)
        hist = {}
        for v in window:
            hist[v] = hist.get(v, 0) + 1
        
        # Shannon entropy
        H = 0.0
        for count in hist.values():
            if count > 0:
                p = count / total
                H -= p * log2(p)
        
        return H
    
    def deltaE(self, op: str, depth: int = 0) -> float:
        """
        Calculate energy change for an operation.
        Includes depth-dependent leak for loops.
        """
        leak = 0.0
        if op in '[]{}':
            leak = 1.0 + depth / 3.0
        
        base_costs = {
            '>': -1, '<': -1,
            '+': -2, '-': +1,
            '.': -1, ',': -1,
            '^': -1, 'v': -1,
            ':': 0, ';': -self.slocal,
            '?': -2, '*': -10, '@': -6,
            '=': 0, '!': -1,
            '{': -2, '}': 0,
            '#': 0, '%': -1, '~': +5
        }
        
        return base_costs.get(op, 0) - leak
    
    def mutate_one(self):
        """Mutate a random instruction in the code."""
        if not self.code:
            return
        i = random.randrange(len(self.code))
        self.code[i] = random.choice(OPS)
        self.mutations += 1
    
    def crossover(self):
        """Perform single-point crossover with a random genome from the bank."""
        if not self.G or not self.code:
            return
        
        g = random.choice(self.G)
        k = min(len(g), len(self.code))
        
        if k > 2:
            cut = random.randrange(1, k - 1)
        else:
            cut = 1
        
        self.code = self.code[:cut] + list(g)[cut:]
        self.crossovers += 1
    
    def replicate(self):
        """
        Replicate current genome to the bank.
        Applies light mutation burst proportional to temperature.
        """
        g = ''.join(self.code)
        self.G.append(g)
        
        # Mutation burst proportional to T
        burst_size = max(1, int(self.T * 3))
        for _ in range(burst_size):
            if self.code:
                j = random.randrange(len(self.code))
                self.code[j] = random.choice(OPS)
        
        self.replications += 1
        return g
    
    def learn(self):
        """
        Peephole optimization: remove redundant instruction pairs.
        Converts simplification to energy gain.
        """
        s = ''.join(self.code)
        before = len(s)
        
        # Remove canceling pairs
        s = s.replace('+-', '').replace('-+', '')
        s = s.replace('><', '').replace('<>', '')
        
        gain = max(0, before - len(s))
        self.code = list(s)
        
        # Convert simplification to energy
        self.E += gain * 0.5
        
        if gain > 0:
            self.learns += 1
    
    def free_energy(self) -> float:
        """Calculate free energy F = E - T·S"""
        return self.E - self.T * self.S
    
    def branching_factor(self) -> float:
        """Estimate current branching factor λ"""
        if not self.branch_hist:
            return 1.0
        return max(1e-6, sum(self.branch_hist) / len(self.branch_hist))
    
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
            # Read input (simplified: just set to 0)
            self.M[self.p] = 0
            
        elif op == '[':
            if self.M[self.p] == 0:
                # Jump forward to matching ]
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
            # Anneal up
            self.T += self.tau
            
        elif op == 'v':
            # Anneal down
            self.T = max(0.01, self.T - self.tau)
            
        elif op == ':':
            # Sense entropy (already done above)
            pass
            
        elif op == ';':
            # Accrue entropy to global
            self.S += self.slocal
            
        elif op == '?':
            # Thermal mutation
            p_mut = min(0.95, 0.2 + 0.6 * self.T)
            if random.random() < p_mut:
                self.mutate_one()
                
        elif op == '*':
            # Replicate if free energy allows
            if self.free_energy() > self.theta:
                self.replicate()
                
        elif op == '@':
            # Crossover
            self.crossover()
            
        elif op == '=':
            # Learn (peephole optimization)
            self.learn()
            
        elif op == '!':
            # Select: promote to elite if score is high
            score = K(self.O) / (1 + abs(self.E - 200))
            if score > 1.0:
                self.elite.append(''.join(self.code))
                
        elif op == '{':
            # Critical loop: force branching near λ≈1
            p_branch = min(0.9, 0.3 + 0.4 * self.T)
            if random.random() < p_branch:
                self.M[self.p] ^= 1
            self.branch_hist.append(1 if self.M[self.p] else 0)
            
        elif op == '}':
            # End critical loop
            pass
            
        elif op == '#':
            # Measure: print state
            F = self.free_energy()
            lam = self.branching_factor()
            print(f"E={self.E:.1f} T={self.T:.2f} S={self.S:.2f} F={F:.2f} λ~{lam:.2f}")
            
        elif op == '%':
            # Phenotype dump
            print(self.O)
            self.O = ""
            
        elif op == '~':
            # Quench: revert to elite if free energy is negative
            if self.free_energy() < 0 and self.elite:
                self.code = list(random.choice(self.elite))
                self.T *= 0.8
        
        # Advance instruction pointer
        self.ip = self.ip + 1
        self.steps_executed += 1
        
        return True
    
    def run(self, steps: int = 20000, halt_at_end: bool = False):
        """
        Run the interpreter for up to `steps` instructions or until energy depleted.
        
        Args:
            steps: Maximum number of steps to execute
            halt_at_end: If True, stop when reaching end of code (non-looping mode)
        """
        if self.verbose:
            print(f"Starting ChaosBF with E={self.E}, T={self.T}")
            print(f"Code: {''.join(self.code)}")
            print()
        
        while self.E > 0 and steps > 0:
            if halt_at_end and self.ip >= len(self.code):
                break
            self.step()
            steps -= 1
        
        print("\n=== HALT ===")
        print(f"Final state:")
        print(f"  Energy: {self.E:.2f}")
        print(f"  Temperature: {self.T:.2f}")
        print(f"  Entropy: {self.S:.2f}")
        print(f"  Free Energy: {self.free_energy():.2f}")
        print(f"  Branching Factor: {self.branching_factor():.2f}")
        print(f"  Steps: {self.steps_executed}")
        print(f"  Mutations: {self.mutations}")
        print(f"  Replications: {self.replications}")
        print(f"  Crossovers: {self.crossovers}")
        print(f"  Learns: {self.learns}")
        print(f"  Genome Bank Size: {len(self.G)}")
        print(f"  Elite Count: {len(self.elite)}")
        
        if self.O:
            print(f"\nFinal Output: {self.O}")
        
        return self.O
    
    def get_stats(self) -> dict:
        """Return current statistics as a dictionary."""
        return {
            'energy': self.E,
            'temperature': self.T,
            'entropy': self.S,
            'free_energy': self.free_energy(),
            'branching_factor': self.branching_factor(),
            'steps': self.steps_executed,
            'mutations': self.mutations,
            'replications': self.replications,
            'crossovers': self.crossovers,
            'learns': self.learns,
            'genome_bank_size': len(self.G),
            'elite_count': len(self.elite),
            'output_length': len(self.O),
            'output_complexity': K(self.O)
        }


def main():
    """Command-line interface for ChaosBF."""
    if len(sys.argv) < 2:
        print("Usage: python chaosbf.py <code> [options]")
        print("Options:")
        print("  --energy E       Initial energy (default: 200)")
        print("  --temp T         Initial temperature (default: 0.5)")
        print("  --steps N        Max steps (default: 20000)")
        print("  --verbose        Verbose output")
        sys.exit(1)
    
    code = sys.argv[1]
    
    # Parse options
    E = 200.0
    T = 0.5
    steps = 20000
    verbose = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--energy' and i + 1 < len(sys.argv):
            E = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--temp' and i + 1 < len(sys.argv):
            T = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--steps' and i + 1 < len(sys.argv):
            steps = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--verbose':
            verbose = True
            i += 1
        else:
            i += 1
    
    # Run
    cbf = ChaosBF(code, E=E, T=T, verbose=verbose)
    cbf.run(steps=steps)


if __name__ == '__main__':
    main()

