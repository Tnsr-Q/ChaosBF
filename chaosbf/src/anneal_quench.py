#!/usr/bin/env python3
"""
Anneal-Quench Storm Experiments for ChaosBF

Scripted thermal cycles that demonstrate phase transitions and complexity bursts.
This produces the "money plot" that reviewers love: ΔK/Δt surges at quench edges.

Protocol:
1. Anneal: Gradually raise T until λ > λ_threshold (typically 1.3)
2. Quench: Rapidly cool T to baseline
3. Measure: Track ΔK/Δt (complexity burst) during quench
4. Repeat: Multiple cycles to show reproducibility
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Dict, Tuple
import json


class AnnealQuenchStorm:
    """
    Anneal-quench thermal cycling experiment.
    
    Demonstrates phase transitions and complexity bursts at the edge of chaos.
    """
    
    def __init__(
        self,
        genome: str,
        E_initial: float = 200.0,
        T_baseline: float = 0.5,
        T_max: float = 1.5,
        lambda_threshold: float = 1.3,
        anneal_rate: float = 0.05,
        quench_rate: float = 0.3,
        steps_per_cycle: int = 100,
        n_cycles: int = 5,
        seed: int = 42
    ):
        """
        Initialize anneal-quench experiment.
        
        Args:
            genome: Program to evolve
            E_initial: Initial energy
            T_baseline: Baseline temperature (cold state)
            T_max: Maximum temperature (hot state)
            lambda_threshold: Branching factor threshold for quench trigger
            anneal_rate: Temperature increase per cycle
            quench_rate: Temperature decrease per cycle (rapid)
            steps_per_cycle: Execution steps per thermal cycle
            n_cycles: Number of anneal-quench cycles
            seed: RNG seed
        """
        self.genome = genome
        self.E_initial = E_initial
        self.T_baseline = T_baseline
        self.T_max = T_max
        self.lambda_threshold = lambda_threshold
        self.anneal_rate = anneal_rate
        self.quench_rate = quench_rate
        self.steps_per_cycle = steps_per_cycle
        self.n_cycles = n_cycles
        self.seed = seed
        
        # Results
        self.history = {
            'time': [],
            'T': [],
            'E': [],
            'S': [],
            'F': [],
            'lambda': [],
            'K': [],
            'dK_dt': [],
            'phase': []  # 'anneal', 'quench', or 'stable'
        }
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run anneal-quench storm experiment.
        
        Returns:
            Dictionary with results and statistics
        """
        if verbose:
            print("="*80)
            print("ANNEAL-QUENCH STORM EXPERIMENT")
            print("="*80)
            print(f"Genome: {self.genome}")
            print(f"Baseline T: {self.T_baseline}")
            print(f"Max T: {self.T_max}")
            print(f"λ threshold: {self.lambda_threshold}")
            print(f"Cycles: {self.n_cycles}")
            print()
        
        # Initialize ChaosBF
        cbf = ChaosBFv3(
            self.genome,
            E=self.E_initial,
            T=self.T_baseline,
            seed=self.seed,
            use_pid=False,  # Manual temperature control
            use_variance_shaping=False,
            verbose=False
        )
        
        time_step = 0
        K_prev = 0.0
        phase = 'stable'
        
        for cycle in range(self.n_cycles):
            if verbose:
                print(f"Cycle {cycle + 1}/{self.n_cycles}")
            
            # ANNEAL PHASE: Raise T until λ > threshold
            phase = 'anneal'
            while cbf.T < self.T_max:
                # Execute steps
                for _ in range(self.steps_per_cycle):
                    if not cbf.step():
                        break
                    time_step += 1
                
                # Record state
                stats = cbf.get_stats()
                K_current = stats['output_complexity']
                dK_dt = K_current - K_prev
                
                self.history['time'].append(time_step)
                self.history['T'].append(cbf.T)
                self.history['E'].append(cbf.E)
                self.history['S'].append(cbf.S)
                self.history['F'].append(cbf.E - cbf.T * cbf.S)
                self.history['lambda'].append(stats['branching_factor'])
                self.history['K'].append(K_current)
                self.history['dK_dt'].append(dK_dt)
                self.history['phase'].append(phase)
                
                K_prev = K_current
                
                # Check threshold
                if stats['branching_factor'] > self.lambda_threshold:
                    if verbose:
                        print(f"  λ = {stats['branching_factor']:.3f} > {self.lambda_threshold} at T = {cbf.T:.3f}")
                    break
                
                # Anneal: raise temperature
                cbf.T = min(cbf.T + self.anneal_rate, self.T_max)
                
                if cbf.E <= 0:
                    break
            
            # QUENCH PHASE: Rapid cool to baseline
            phase = 'quench'
            if verbose:
                print(f"  QUENCH: T {cbf.T:.3f} → {self.T_baseline:.3f}")
            
            while cbf.T > self.T_baseline:
                # Execute steps
                for _ in range(self.steps_per_cycle):
                    if not cbf.step():
                        break
                    time_step += 1
                
                # Record state
                stats = cbf.get_stats()
                K_current = stats['output_complexity']
                dK_dt = K_current - K_prev
                
                self.history['time'].append(time_step)
                self.history['T'].append(cbf.T)
                self.history['E'].append(cbf.E)
                self.history['S'].append(cbf.S)
                self.history['F'].append(cbf.E - cbf.T * cbf.S)
                self.history['lambda'].append(stats['branching_factor'])
                self.history['K'].append(K_current)
                self.history['dK_dt'].append(dK_dt)
                self.history['phase'].append(phase)
                
                K_prev = K_current
                
                # Quench: rapid cool
                cbf.T = max(cbf.T - self.quench_rate, self.T_baseline)
                
                if cbf.E <= 0:
                    break
            
            # STABLE PHASE: Hold at baseline
            phase = 'stable'
            for _ in range(self.steps_per_cycle * 2):
                if not cbf.step():
                    break
                time_step += 1
                
                # Record every 10 steps
                if time_step % 10 == 0:
                    stats = cbf.get_stats()
                    K_current = stats['output_complexity']
                    dK_dt = K_current - K_prev
                    
                    self.history['time'].append(time_step)
                    self.history['T'].append(cbf.T)
                    self.history['E'].append(cbf.E)
                    self.history['S'].append(cbf.S)
                    self.history['F'].append(cbf.E - cbf.T * cbf.S)
                    self.history['lambda'].append(stats['branching_factor'])
                    self.history['K'].append(K_current)
                    self.history['dK_dt'].append(dK_dt)
                    self.history['phase'].append(phase)
                    
                    K_prev = K_current
            
            if cbf.E <= 0:
                if verbose:
                    print(f"  Energy depleted at cycle {cycle + 1}")
                break
        
        # Compute statistics
        results = self._compute_statistics()
        
        if verbose:
            print()
            print("="*80)
            print("RESULTS")
            print("="*80)
            print(f"Total time steps: {time_step}")
            print(f"Cycles completed: {cycle + 1}")
            print(f"Max ΔK/Δt: {results['max_dK_dt']:.3f}")
            print(f"Mean ΔK/Δt (quench): {results['mean_dK_dt_quench']:.3f}")
            print(f"Complexity bursts detected: {results['n_bursts']}")
            print()
        
        return results
    
    def _compute_statistics(self) -> Dict:
        """Compute statistics from history."""
        dK_dt = np.array(self.history['dK_dt'])
        phases = self.history['phase']
        
        # Find quench phases
        quench_mask = np.array([p == 'quench' for p in phases])
        dK_dt_quench = dK_dt[quench_mask]
        
        # Detect bursts (ΔK/Δt > 2σ above mean)
        mean_dK = np.mean(dK_dt)
        std_dK = np.std(dK_dt)
        burst_threshold = mean_dK + 2 * std_dK
        bursts = dK_dt > burst_threshold
        
        return {
            'max_dK_dt': float(np.max(np.abs(dK_dt))),
            'mean_dK_dt': float(np.mean(dK_dt)),
            'std_dK_dt': float(np.std(dK_dt)),
            'mean_dK_dt_quench': float(np.mean(dK_dt_quench)) if len(dK_dt_quench) > 0 else 0.0,
            'std_dK_dt_quench': float(np.std(dK_dt_quench)) if len(dK_dt_quench) > 0 else 0.0,
            'n_bursts': int(np.sum(bursts)),
            'burst_threshold': float(burst_threshold),
            'total_steps': len(self.history['time'])
        }
    
    def plot(self, output_path: str):
        """
        Generate the "money plot" showing thermal cycles and complexity bursts.
        
        Args:
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('Anneal-Quench Storm: Phase Transitions and Complexity Bursts',
                     fontsize=14, fontweight='bold')
        
        time = np.array(self.history['time'])
        T = np.array(self.history['T'])
        lambda_vals = np.array(self.history['lambda'])
        K = np.array(self.history['K'])
        dK_dt = np.array(self.history['dK_dt'])
        phases = self.history['phase']
        
        # Color code phases
        colors = {'anneal': 'red', 'quench': 'blue', 'stable': 'green'}
        phase_colors = [colors[p] for p in phases]
        
        # Plot 1: Temperature
        axes[0].plot(time, T, 'k-', linewidth=2, label='Temperature')
        axes[0].axhline(self.T_baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        axes[0].axhline(self.T_max, color='gray', linestyle='--', alpha=0.5, label='Max')
        axes[0].set_ylabel('Temperature T', fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Branching Factor
        axes[1].plot(time, lambda_vals, 'k-', linewidth=2, label='λ')
        axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Critical (λ=1)')
        axes[1].axhline(self.lambda_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Threshold ({self.lambda_threshold})')
        axes[1].set_ylabel('Branching Factor λ', fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Complexity
        axes[2].plot(time, K, 'k-', linewidth=2, label='K(O)')
        axes[2].set_ylabel('Complexity K(O)', fontweight='bold')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: ΔK/Δt (THE MONEY PLOT)
        axes[3].plot(time, dK_dt, 'k-', linewidth=1, alpha=0.7, label='ΔK/Δt')
        
        # Highlight bursts
        stats = self._compute_statistics()
        burst_threshold = stats['burst_threshold']
        axes[3].axhline(burst_threshold, color='red', linestyle='--', alpha=0.5, label=f'Burst threshold')
        axes[3].axhline(-burst_threshold, color='red', linestyle='--', alpha=0.5)
        
        # Fill quench regions
        for i in range(len(time)):
            if phases[i] == 'quench':
                axes[3].axvspan(time[i], time[i] + 1, color='blue', alpha=0.1)
        
        axes[3].set_xlabel('Time (steps)', fontweight='bold')
        axes[3].set_ylabel('ΔK/Δt (Complexity Burst)', fontweight='bold')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        # Add annotation
        fig.text(0.5, 0.02, 
                f'Bursts detected: {stats["n_bursts"]} | Max ΔK/Δt: {stats["max_dK_dt"]:.3f} | Mean (quench): {stats["mean_dK_dt_quench"]:.3f}',
                ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved anneal-quench plot to {output_path}")
        plt.close()
    
    def export(self, filepath: str):
        """Export results to JSON."""
        data = {
            'config': {
                'genome': self.genome,
                'E_initial': self.E_initial,
                'T_baseline': self.T_baseline,
                'T_max': self.T_max,
                'lambda_threshold': self.lambda_threshold,
                'anneal_rate': self.anneal_rate,
                'quench_rate': self.quench_rate,
                'n_cycles': self.n_cycles
            },
            'statistics': self._compute_statistics(),
            'history': {k: [float(v) if isinstance(v, (int, float, np.number)) else v 
                           for v in vals] 
                       for k, vals in self.history.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported results to {filepath}")


def main():
    """Command-line interface for anneal-quench experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Anneal-Quench Storm Experiments')
    parser.add_argument('--genome', default='++[>+<-].:{;}{?}^*=@=.#%',
                       help='Genome to evolve')
    parser.add_argument('--energy', type=float, default=300.0,
                       help='Initial energy')
    parser.add_argument('--T-baseline', type=float, default=0.5,
                       help='Baseline temperature')
    parser.add_argument('--T-max', type=float, default=1.5,
                       help='Maximum temperature')
    parser.add_argument('--lambda-threshold', type=float, default=1.3,
                       help='Branching factor threshold for quench')
    parser.add_argument('--cycles', type=int, default=5,
                       help='Number of anneal-quench cycles')
    parser.add_argument('--seed', type=int, default=42,
                       help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/anneal_quench',
                       help='Output prefix')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = AnnealQuenchStorm(
        genome=args.genome,
        E_initial=args.energy,
        T_baseline=args.T_baseline,
        T_max=args.T_max,
        lambda_threshold=args.lambda_threshold,
        n_cycles=args.cycles,
        seed=args.seed
    )
    
    results = experiment.run(verbose=True)
    
    # Export and plot
    experiment.export(f'{args.output}_results.json')
    experiment.plot(f'{args.output}_plot.png')
    
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()

