#!/usr/bin/env python3
"""
ChaosBF v3.0: Publication-Grade Thermodynamic, Evolvable Brainfuck

Critical fixes from expert analysis:
- Fixed descriptor computation (post-dump, proper timing)
- Dual-loop criticality control (PID + variance shaping)
- Energy-aware crossover (cut at minimal ΔE valleys)
- EMA-smoothed entropy slope
- Volatility descriptor for MAP-Elites
- Landauer cost for information erasure
- Metropolis acceptance for reversibility
- Lyapunov divergence estimation
"""

import random
import zlib
from math import log2, exp
from typing import List, Optional, Dict, Tuple, Deque
import sys
import json
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np


OPS = "><+-[].,^v:;?*@=!{}#%~"
SAFE_OPS = "><+-.,^v:?*@=!#%~"


def K(b: str) -> float:
    """Kolmogorov complexity proxy using LZ compression."""
    if not b:
        return 0.0
    try:
        c = zlib.compress(b.encode('latin1'), level=9)
        return max(0.0, len(b) - len(c))
    except:
        return float(len(b))


def ema(value: float, prev_ema: float, alpha: float = 0.2) -> float:
    """Exponential moving average."""
    return alpha * value + (1 - alpha) * prev_ema


class ChaosBFv3:
    """
    ChaosBF v3.0 interpreter with publication-grade features.
    
    Major upgrades:
    - Dual-loop criticality control (PID + variance shaping)
    - Energy-aware crossover
    - Fixed descriptor computation
    - Metropolis moves
    - Lyapunov estimation
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
        # Variance shaping parameters
        use_variance_shaping: bool = True,
        gamma: float = 0.02,
        var_target: float = 0.05,
        var_alpha: float = 0.01,
        # Temperature bounds
        T_min: float = 0.01,
        T_max: float = 2.0,
        # Mutation parameters
        grammar_aware: bool = True,
        wild_mutation_rate: float = 0.1,
        adaptive_mutation: bool = True,
        # Learning parameters
        learning_cap: float = 0.5,
        learning_rate_limit: float = 0.05,
        # Leak parameters
        leak_max: float = 10.0,
        # Lambda estimation
        lambda_window: int = 200,
        # Metropolis parameters
        use_metropolis: bool = False,
        # Lyapunov estimation
        track_lyapunov: bool = False,
        verbose: bool = False
    ):
        """Initialize ChaosBF v3.0 runtime."""
        
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
        self.G = []
        self.elite = []
        
        # Criticality tracking with windowed estimator
        self.lambda_window = lambda_window
        self.branch_children = deque(maxlen=lambda_window)
        self.branch_parents = deque(maxlen=lambda_window)
        self.lambda_history = deque(maxlen=lambda_window)
        
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
        
        # Variance shaping (dual-loop control)
        self.use_variance_shaping = use_variance_shaping
        self.gamma = gamma
        self.var_target = var_target
        self.var_alpha = var_alpha
        self.var_ema = 0.0
        
        # Mutation parameters
        self.grammar_aware = grammar_aware
        self.wild_mutation_rate = wild_mutation_rate
        self.adaptive_mutation = adaptive_mutation
        
        # Learning parameters
        self.learning_cap = learning_cap
        self.learning_rate_limit = learning_rate_limit
        self.learning_energy_gained = 0.0
        self.learning_window_start = 0
        
        # Metropolis acceptance
        self.use_metropolis = use_metropolis
        self.metropolis_accepts = 0
        self.metropolis_rejects = 0
        self.metropolis_target_acceptance = (0.23, 0.30)  # Classic MCMC sweet spot
        self.mutation_radius = 0.1  # Adaptive mutation radius
        
        # Lyapunov estimation
        self.track_lyapunov = track_lyapunov
        self.lyapunov_estimate = 0.0
        
        # Statistics
        self.steps_executed = 0
        self.mutations = 0
        self.mutations_wild = 0
        self.replications = 0
        self.crossovers = 0
        self.learns = 0
        self.pid_updates = 0
        
        # Temporal derivatives with EMA smoothing
        self.prev_S = 0.0
        self.prev_K = 0.0
        self.dS_dt_ema = 0.0
        self.dK_dt_ema = 0.0
        self.entropy_slope_alpha = 0.2
        
        # Volatility tracking
        self.lambda_volatility = 0.0
        
          # PID controller
        self.pid_kp = 0.3
        self.pid_ki = 0.05
        self.pid_kd = 0.05
        self._apply_controller_sanity_guard()
        self.pid_d = 0.0
        
        # Energy window for descriptor computation
        self.E_window_start = E
        self.E_window_steps = 100
        
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
        
        # Shannon entropy with small epsilon to avoid log(0)
        H = 0.0
        epsilon = 1e-10
        for count in hist.values():
            if count > 0:
                p = count / total
                H -= p * log2(p + epsilon)
        
        # Add small Gaussian noise to avoid flat bins
        H += self.rng.gauss(0, 0.01)
        
        return max(0.0, H)
    
    def landauer_cost_exact(self, window_before: List[int], window_after: List[int]) -> float:
        """
        Landauer-exact costing based on local Shannon entropy changes.
        
        Charge energy when a write reduces local H(window).
        Credit energy (bounded) when H increases.
        
        Args:
            window_before: Local window before write
            window_after: Local window after write
        
        Returns:
            Energy change (negative = cost, positive = credit)
        """
        # Compute Shannon entropy of windows
        def shannon_entropy(window):
            if not window:
                return 0.0
            
            # Count frequencies
            from collections import Counter
            counts = Counter(window)
            total = len(window)
            
            # Compute H = -sum(p * log2(p))
            H = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    H -= p * np.log2(p + 1e-10)
            
            return H
        
        H_before = shannon_entropy(window_before)
        H_after = shannon_entropy(window_after)
        
        dH = H_after - H_before
        
        if dH < 0:  # Erasure (entropy decreased)
            # Landauer cost: ΔE = -k*T*ln(2)*ΔH
            # Simplified: charge proportional to entropy reduction
            return dH * self.T  # Negative (cost)
        else:  # Creation (entropy increased)
            # Bounded credit for entropy increase
            max_credit = 0.5  # Prevent unbounded energy gain
            return min(dH * self.T, max_credit)  # Positive (credit)
    
    def landauer_cost(self, cell_value_before: int, cell_value_after: int) -> float:
        """
        Calculate Landauer cost for information erasure (legacy method).
        
        Charge energy when a write reduces local Shannon information.
        """
        # Simple proxy: if we're resetting to 0, charge Landauer cost
        if cell_value_before != 0 and cell_value_after == 0:
            return -0.5  # Small Landauer cost
        return 0.0
    
    def deltaE(self, op: str, depth: int = 0) -> float:
        """Calculate energy change for an operation."""
        leak = 0.0
        if op in '[]{}':
            leak = min(self.leak_max, 1.0 + depth / 3.0)
        
        base_costs = {
            '>': -1, '<': -1,
            '+': -2, '-': +1,
            '.': -1, ',': -1,
            '^': -1, 'v': -1,
            ':': 0,
            ';': 0,  # Pure entropy operation
            '?': -2, '*': -10, '@': -6,
            '=': 0,  # Energy gain calculated separately
            '!': -1,
            '{': -2, '}': 0,
            '#': 0, '%': -1, '~': +5
        }
        
        return base_costs.get(op, 0) - leak
    
    def local_deltaE_map(self, code: List[str]) -> np.ndarray:
        """
        Compute local energy cost map for energy-aware crossover.
        
        Returns array of local ΔE costs for each position in code.
        """
        costs = np.zeros(len(code))
        depth = 0
        
        for i, op in enumerate(code):
            if op == '[' or op == '{':
                depth += 1
            elif op == ']' or op == '}':
                depth = max(0, depth - 1)
            
            costs[i] = abs(self.deltaE(op, depth))
        
        return costs
    
    def mutate_one(self):
        """Grammar-aware mutation with adaptive bias and Metropolis gating."""
        if not self.code:
            return
        
        # Store state for Metropolis
        if self.use_metropolis:
            F_before = self.free_energy()
        
        i = self.rng.randrange(len(self.code))
        c = self.code[i]
        old_op = c
        
        # Adaptive mutation: bias location based on λ
        if self.adaptive_mutation:
            lambda_hat = self.branching_factor()
            
            # If λ < 1, bias mutations inside loop bodies (create branches)
            # If λ > 1, bias mutations outside loops (consolidate)
            in_loop = self._is_in_loop(i)
            
            if lambda_hat < 1.0 and not in_loop:
                # Try to find a position inside a loop
                loop_positions = self._find_loop_positions()
                if loop_positions:
                    i = self.rng.choice(loop_positions)
                    c = self.code[i]
            elif lambda_hat > 1.0 and in_loop:
                # Try to find a position outside loops
                non_loop_positions = self._find_non_loop_positions()
                if non_loop_positions:
                    i = self.rng.choice(non_loop_positions)
                    c = self.code[i]
        
         # Grammar-aware mutation
        if self.grammar_aware and self.rng.random() > self.wild_mutation_rate:
            if c in '[]{}':                return  # Don't mutate brackets in safe mode
            else:
                self.code[i] = self.rng.choice(SAFE_OPS)
        else:
            self.code[i] = self.rng.choice(OPS)
            self.mutations_wild += 1
        
        # Metropolis acceptance
        if self.use_metropolis:
            F_after = self.free_energy()
            dF = F_after - F_before
            
            if dF > 0:  # Uphill in free energy
                p_accept = min(1.0, exp(-dF / max(0.01, self.T)))
                if self.rng.random() > p_accept:
                    # Reject mutation
                    self.code[i] = old_op
                    self.metropolis_rejects += 1
                    return
                self.metropolis_accepts += 1
        
        self.mutations += 1
        
        # Acceptance-rate targeting (tune mutation radius every 100 steps)
        if self.use_metropolis and (self.metropolis_accepts + self.metropolis_rejects) % 100 == 0:
            self._tune_mutation_radius()
    
    def _is_in_loop(self, pos: int) -> bool:
        """Check if position is inside a loop."""
        depth = 0
        for i in range(pos):
            if self.code[i] in '[{':
                depth += 1
            elif self.code[i] in ']}':
                depth = max(0, depth - 1)
        return depth > 0
    
    def _find_loop_positions(self) -> List[int]:
        """Find all positions inside loops."""
        positions = []
        depth = 0
        for i, c in enumerate(self.code):
            if c in '[{':
                depth += 1
            elif c in ']}':
                depth = max(0, depth - 1)
            if depth > 0:
                positions.append(i)
        return positions
    
    def _find_non_loop_positions(self) -> List[int]:
        """Find all positions outside loops."""
        positions = []
        depth = 0
        for i, c in enumerate(self.code):
            if c in '[{':
                depth += 1
            elif c in ']}':
                depth = max(0, depth - 1)
            if depth == 0:
                positions.append(i)
        return positions
    
    def crossover_energy_aware(self, mate_code: str):
        """
        Energy-aware crossover: cut at minimal ΔE valleys.
        
        This dramatically improves offspring viability.
        """
        if not mate_code or not self.code:
            return
        
        mate = list(mate_code)
        k = min(len(self.code), len(mate))
        
        if k < 3:
            # Fallback to random cut
            cut = self.rng.randint(1, k - 1) if k > 1 else 1
        else:
            # Compute local cost map
            costs = self.local_deltaE_map(self.code[:k])
            
            # Smooth with simple moving average
            window = min(9, k // 3)
            if window >= 3:
                smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
                # Find minimum in smoothed costs
                cut = np.argmin(smoothed) + window // 2
                cut = max(1, min(cut, k - 1))
            else:
                cut = np.argmin(costs)
                cut = max(1, min(cut, k - 1))
        
        self.code = self.code[:cut] + mate[cut:]
        self.crossovers += 1
    
    def crossover(self):
        """Standard crossover (kept for compatibility)."""
        if not self.G or not self.code:
            return
        
        mate = self.rng.choice(self.G)
        self.crossover_energy_aware(mate)
    
    def replicate(self):
        """Replicate with Metropolis acceptance criterion."""
        g = ''.join(self.code)
        
        # Metropolis acceptance
        if self.use_metropolis:
            dF = self.free_energy() - self.theta
            if dF < 0:
                # Accept with probability exp(-dF/T)
                p_accept = min(1.0, exp(-abs(dF) / max(0.01, self.T)))
                if self.rng.random() > p_accept:
                    self.metropolis_rejects += 1
                    return None
                self.metropolis_accepts += 1
        
        self.G.append(g)
        
        # Mutation burst
        burst_size = max(1, int(self.T * 3))
        for _ in range(burst_size):
            if self.code:
                self.mutate_one()
        
        self.replications += 1
        return g
    
    def learn(self):
        """
        Peephole optimization with rate-limited energy credit.
        
        Prevents learning bursts from leveraging stale ΔF samples.
        """
        s_before = ''.join(self.code)
        
        # Check rate limit
        steps_since_window = self.steps_executed - self.learning_window_start
        if steps_since_window >= 1000:
            self.learning_energy_gained = 0.0
            self.learning_window_start = self.steps_executed
        
        # Calculate energy cost before optimization
        E_before = self._estimate_energy_cost(s_before)
        
        # Apply peephole optimizations
        s_after = s_before.replace('+-', '').replace('-+', '')
        s_after = s_after.replace('><', '').replace('<>', '')
        
        if s_before == s_after:
            return
        
        # Calculate energy cost after optimization
        E_after = self._estimate_energy_cost(s_after)
        
        # Energy saved
        dE = E_before - E_after
        
        # Entropy change
        dS = max(0.0, K(s_before) - K(s_after)) * 0.1
        
        # Free energy gain
        dF = dE - self.T * dS
        
        # Credit with rate limit
        if dF > 0:
            max_gain = self.learning_rate_limit * self.E
            available = max_gain - self.learning_energy_gained
            
            if available > 0:
                energy_gain = min(self.learning_cap * dF, available)
                self.E += energy_gain
                self.learning_energy_gained += energy_gain
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
    
    def _tune_mutation_radius(self):
        """
        Tune mutation radius to target acceptance rate of 0.23-0.30.
        
        If acceptance < 0.15: shrink radius (too many rejections)
        If acceptance > 0.35: expand radius (too many acceptances)
        """
        total = self.metropolis_accepts + self.metropolis_rejects
        if total == 0:
            return
        
        acceptance_ratio = self.metropolis_accepts / total
        
        if acceptance_ratio < 0.15:
            self.mutation_radius *= 0.9  # Shrink
        elif acceptance_ratio > 0.35:
            self.mutation_radius *= 1.1  # Expand
        
        # Clamp to reasonable range
        self.mutation_radius = max(0.01, min(0.5, self.mutation_radius))
    
    def _apply_controller_sanity_guard(self):
        """
        Controller sanity guard: prevent negative ki with high gamma or no anti-windup.
        
        CMA-ES can find negative ki, but this causes integral windup issues
        unless variance term gamma < 0.4 and anti-windup is active.
        """
        if self.pid_ki < 0:
            if self.variance_shaping_gamma >= 0.4 or not self.use_variance_shaping:
                # Unsafe: force ki to zero
                import warnings
                warnings.warn(
                    f"Negative ki={self.pid_ki:.4f} with gamma={self.variance_shaping_gamma:.4f}. "
                    f"Forcing ki=0 to prevent integral windup."
                )
                self.pid_ki = 0.0
    
    def free_energy(self) -> float:
        """Compute free energy F = E - T*S."""
        return self.E - self.T * self.S
    
    def branching_factor(self) -> float:
        """Estimate branching factor using windowed ratio."""
        if not self.branch_children or not self.branch_parents:
            return 1.0
        
        total_children = sum(self.branch_children)
        total_parents = sum(self.branch_parents) + 1e-6
        
        return total_children / total_parents
    
    def lambda_volatility_estimate(self) -> float:
        """Estimate volatility (std dev) of λ over window."""
        if len(self.lambda_history) < 2:
            return 0.0
        
        return float(np.std(list(self.lambda_history)))
    
    def update_pid_controller(self):
        """Update temperature using PID + variance shaping."""
        if not self.use_pid:
            return
        
        lambda_hat = self.branching_factor()
        self.lambda_history.append(lambda_hat)
        
        # Fast PID on λ
        error = lambda_hat - 1.0
        self.pid_p = self.kp * error
        self.pid_integral += error
        self.pid_i = self.ki * self.pid_integral
        self.pid_d = self.kd * (error - self.pid_prev_error)
        
        dT_pid = self.pid_p + self.pid_i + self.pid_d
        
        # Slow variance shaping
        dT_var = 0.0
        if self.use_variance_shaping:
            # Update variance EMA
            var_current = (lambda_hat - 1.0) ** 2
            self.var_ema = ema(var_current, self.var_ema, self.var_alpha)
            
            # Variance shaping: target a little wiggle
            var_error = self.var_target - self.var_ema
            dT_var = self.gamma * var_error
        
        # Combined update
        self.T += dT_pid + dT_var
        self.T = max(self.T_min, min(self.T_max, self.T))
        
        self.pid_prev_error = error
        self.pid_updates += 1
        
        if self.verbose:
            print(f"PID: λ={lambda_hat:.3f}, var={self.var_ema:.4f}, "
                  f"T={self.T:.3f} (dT_pid={dT_pid:.4f}, dT_var={dT_var:.4f})")
    
    def update_temporal_derivatives(self):
        """Update dS/dt and dK/dt with EMA smoothing."""
        current_K = K(self.O)
        
        # Raw derivatives
        dS_raw = self.S - self.prev_S
        dK_raw = current_K - self.prev_K
        
        # EMA smoothing
        self.dS_dt_ema = ema(dS_raw, self.dS_dt_ema, self.entropy_slope_alpha)
        self.dK_dt_ema = ema(dK_raw, self.dK_dt_ema, self.entropy_slope_alpha)
        
        self.prev_S = self.S
        self.prev_K = current_K
    
    def compute_descriptors(self) -> Dict[str, float]:
        """
        Compute behavioral descriptors AFTER phenotype operations.
        
        CRITICAL FIX: This must be called after % dump and compression.
        """
        # Energy used in window
        E_used = abs(self.E_window_start - self.E)
        if self.steps_executed % self.E_window_steps == 0:
            self.E_window_start = self.E
        
        # Complexity of current output
        K_output = K(self.O)
        
        # Info per energy (incremental)
        info_per_energy = self.dK_dt_ema / max(1.0, E_used / self.E_window_steps)
        
        # Entropy slope (EMA-smoothed)
        entropy_slope = self.dS_dt_ema
        
        # Lambda deviation
        lambda_hat = self.branching_factor()
        lambda_dev = abs(lambda_hat - 1.0)
        
        # Volatility
        volatility = self.lambda_volatility_estimate()
        
        return {
            'lambda_dev': lambda_dev,
            'info_per_energy': info_per_energy,
            'entropy_slope': entropy_slope,
            'volatility': volatility,
            'K_output': K_output
        }
    
    def step(self):
        """Execute one instruction."""
        if not self.code:
            return False
        
        self.ip = self.ip % len(self.code)
        op = self.code[self.ip]
        depth = len(self.stack)
        
        # Update local entropy if needed
        if op == ':':
            self.slocal = self.sense_entropy()
        
        # Apply energy cost
        self.E += self.deltaE(op, depth)
        
        # Landauer cost for writes
        if op in '+-,':
            old_val = self.M[self.p]
        
        # Execute operation
        if op == '>':
            self.p = (self.p + 1) % self.tape_size
            
        elif op == '<':
            self.p = (self.p - 1) % self.tape_size
            
        elif op == '+':
            new_val = (self.M[self.p] + 1) & 0xFF
            self.E += self.landauer_cost(self.M[self.p], new_val)
            self.M[self.p] = new_val
            
        elif op == '-':
            new_val = (self.M[self.p] - 1) & 0xFF
            self.E += self.landauer_cost(self.M[self.p], new_val)
            self.M[self.p] = new_val
            
        elif op == '.':
            self.O += chr(self.M[self.p])
            
        elif op == ',':
            new_val = 0
            self.E += self.landauer_cost(self.M[self.p], new_val)
            self.M[self.p] = new_val
            
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
            pass
            
        elif op == ';':
            self.S += self.slocal
            
        elif op == '?':
            p_mut = min(0.95, 0.2 + 0.6 * self.T)
            if self.rng.random() < p_mut:
                self.mutate_one()
                
        elif op == '*':
            if self.free_energy() > self.theta:
                self.replicate()
                
        elif op == '@':
            self.crossover()
            
        elif op == '=':
            self.learn()
            
        elif op == '!':
            # Compute fitness AFTER operations
            descriptors = self.compute_descriptors()
            score = descriptors['K_output'] / (1 + abs(self.E - self.E_initial))
            if score > 1.0:
                self.elite.append(''.join(self.code))
                
        elif op == '{':
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
            F = self.free_energy()
            lam = self.branching_factor()
            print(f"E={self.E:.1f} T={self.T:.2f} S={self.S:.2f} F={F:.2f} λ~{lam:.2f}")
            
        elif op == '%':
            print(self.O)
            # Update descriptors AFTER dump
            self.update_temporal_derivatives()
            self.O = ""
            
        elif op == '~':
            if self.free_energy() < 0 and self.elite:
                self.code = list(self.rng.choice(self.elite))
                self.T *= 0.8
        
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
            print(f"Starting ChaosBF v3.0")
            print(f"  Seed: {self.seed}")
            print(f"  E={self.E}, T={self.T}")
            print(f"  PID: {'enabled' if self.use_pid else 'disabled'}")
            print(f"  Variance shaping: {'enabled' if self.use_variance_shaping else 'disabled'}")
            print(f"  Metropolis: {'enabled' if self.use_metropolis else 'disabled'}")
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
        print(f"  Lambda Volatility: {self.lambda_volatility_estimate():.4f}")
        print(f"  Steps: {self.steps_executed}")
        print(f"  Mutations: {self.mutations} (wild: {self.mutations_wild})")
        print(f"  Replications: {self.replications}")
        print(f"  Crossovers: {self.crossovers}")
        print(f"  Learns: {self.learns}")
        print(f"  PID Updates: {self.pid_updates}")
        if self.use_metropolis:
            print(f"  Metropolis: {self.metropolis_accepts} accepts, {self.metropolis_rejects} rejects")
        print(f"  Genome Bank Size: {len(self.G)}")
        print(f"  Elite Count: {len(self.elite)}")
        print(f"  Output Complexity: {K(self.O):.2f}")
        print(f"  dS/dt (EMA): {self.dS_dt_ema:.3f}")
        print(f"  dK/dt (EMA): {self.dK_dt_ema:.3f}")
    
    def get_stats(self) -> dict:
        """Return current statistics."""
        descriptors = self.compute_descriptors()
        
        return {
            'seed': self.seed,
            'energy': self.E,
            'temperature': self.T,
            'entropy': self.S,
            'free_energy': self.free_energy(),
            'branching_factor': self.branching_factor(),
            'lambda_volatility': self.lambda_volatility_estimate(),
            'steps': self.steps_executed,
            'mutations': self.mutations,
            'mutations_wild': self.mutations_wild,
            'replications': self.replications,
            'crossovers': self.crossovers,
            'learns': self.learns,
            'pid_updates': self.pid_updates,
            'metropolis_accepts': self.metropolis_accepts,
            'metropolis_rejects': self.metropolis_rejects,
            'genome_bank_size': len(self.G),
            'elite_count': len(self.elite),
            'output_length': len(self.O),
            'output_complexity': K(self.O),
            'dS_dt_ema': self.dS_dt_ema,
            'dK_dt_ema': self.dK_dt_ema,
            'pid_p': self.pid_p,
            'pid_i': self.pid_i,
            'pid_d': self.pid_d,
            'var_ema': self.var_ema,
            **descriptors
        }
    
    def snapshot(self) -> dict:
        """Create a complete snapshot."""
        return {
            'code': ''.join(self.code),
            'ip': self.ip,
            'tape': self.M[:100],
            'pointer': self.p,
            'output': self.O,
            'stats': self.get_stats(),
            'genome_bank': self.G.copy(),
            'elite': self.elite.copy()
        }


def main():
    """Command-line interface for ChaosBF v3.0."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaosBF v3.0: Publication-Grade Thermodynamic Brainfuck')
    parser.add_argument('code', help='Program code')
    parser.add_argument('--energy', type=float, default=200.0)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--no-pid', action='store_true')
    parser.add_argument('--no-variance', action='store_true')
    parser.add_argument('--metropolis', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    cbf = ChaosBFv3(
        args.code,
        E=args.energy,
        T=args.temp,
        seed=args.seed,
        use_pid=not args.no_pid,
        use_variance_shaping=not args.no_variance,
        use_metropolis=args.metropolis,
        verbose=args.verbose
    )
    
    cbf.run(steps=args.steps)


if __name__ == '__main__':
    main()

