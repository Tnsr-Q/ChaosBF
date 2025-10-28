#!/usr/bin/env python3
"""
Experiment protocols for ChaosBF research.

Implements systematic experiments as suggested in the upgrade notes:
1. Criticality sweep
2. Energy-budget ablation
3. Operator knockout
4. Population vs solo comparison
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v2 import ChaosBFv2, K
from map_elites import MAPElites
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from typing import List, Dict, Tuple
import time


class ExperimentRunner:
    """Runner for systematic ChaosBF experiments."""
    
    def __init__(self, output_dir: str = '/home/ubuntu/chaosbf/output/experiments'):
        self.output_dir = output_dir
        self.results = {}
    
    def criticality_sweep(
        self,
        genome: str,
        T_range: Tuple[float, float] = (0.1, 1.2),
        T_steps: int = 12,
        E: float = 200.0,
        steps: int = 5000,
        seed: int = 42
    ) -> Dict:
        """
        Experiment 1: Criticality Sweep
        
        Sweep initial temperature T0 and measure:
        - Time to halt
        - Mean λ
        - Final F
        - K(O)
        
        Expect a ridge at λ≈1 (the "golden trail").
        """
        print("="*80)
        print("EXPERIMENT 1: Criticality Sweep")
        print("="*80)
        print(f"Genome: {genome}")
        print(f"T range: {T_range}, steps: {T_steps}")
        print(f"Energy: {E}, max steps: {steps}")
        print()
        
        T_values = np.linspace(T_range[0], T_range[1], T_steps)
        results = {
            'T_values': T_values.tolist(),
            'time_to_halt': [],
            'mean_lambda': [],
            'final_F': [],
            'K_output': [],
            'final_energy': [],
            'mutations': [],
            'replications': []
        }
        
        for i, T in enumerate(T_values):
            print(f"Run {i+1}/{T_steps}: T={T:.3f}...", end=' ')
            
            cbf = ChaosBFv2(
                genome,
                E=E,
                T=T,
                seed=seed + i,
                use_pid=False,  # No PID for this experiment
                verbose=False
            )
            
            start = time.time()
            cbf.run(steps=steps)
            elapsed = time.time() - start
            
            stats = cbf.get_stats()
            
            results['time_to_halt'].append(stats['steps'])
            results['mean_lambda'].append(stats['branching_factor'])
            results['final_F'].append(stats['free_energy'])
            results['K_output'].append(stats['output_complexity'])
            results['final_energy'].append(stats['energy'])
            results['mutations'].append(stats['mutations'])
            results['replications'].append(stats['replications'])
            
            print(f"steps={stats['steps']}, λ={stats['branching_factor']:.3f}, "
                  f"F={stats['free_energy']:.2f}, K={stats['output_complexity']:.2f}")
        
        # Plot results
        self._plot_criticality_sweep(results)
        
        # Save results
        with open(f'{self.output_dir}/criticality_sweep.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results['criticality_sweep'] = results
        return results
    
    def _plot_criticality_sweep(self, results: Dict):
        """Plot criticality sweep results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Criticality Sweep: Temperature vs Outcomes', fontsize=16, fontweight='bold')
        
        T_values = results['T_values']
        
        # Time to halt
        axes[0, 0].plot(T_values, results['time_to_halt'], 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Initial Temperature (T₀)')
        axes[0, 0].set_ylabel('Steps to Halt')
        axes[0, 0].set_title('Execution Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean lambda
        axes[0, 1].plot(T_values, results['mean_lambda'], 'r-o', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='black', linestyle='--', label='λ=1 (critical)')
        axes[0, 1].set_xlabel('Initial Temperature (T₀)')
        axes[0, 1].set_ylabel('Mean Branching Factor (λ)')
        axes[0, 1].set_title('Criticality')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final free energy
        axes[1, 0].plot(T_values, results['final_F'], 'g-o', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Initial Temperature (T₀)')
        axes[1, 0].set_ylabel('Final Free Energy (F)')
        axes[1, 0].set_title('Free Energy at Halt')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Output complexity
        axes[1, 1].plot(T_values, results['K_output'], 'm-o', linewidth=2)
        axes[1, 1].set_xlabel('Initial Temperature (T₀)')
        axes[1, 1].set_ylabel('Output Complexity K(O)')
        axes[1, 1].set_title('Information Production')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/criticality_sweep.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {self.output_dir}/criticality_sweep.png")
        plt.close()
    
    def energy_budget_ablation(
        self,
        genome: str,
        E_values: List[float] = [80, 120, 160, 200, 250],
        T: float = 0.6,
        steps: int = 10000,
        seed: int = 42
    ) -> Dict:
        """
        Experiment 2: Energy-Budget Ablation
        
        Run with different initial energies to show that higher E doesn't just
        prolong life but enables structural transitions (e.g., replication bursts).
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: Energy-Budget Ablation")
        print("="*80)
        print(f"Genome: {genome}")
        print(f"E values: {E_values}")
        print(f"Temperature: {T}, max steps: {steps}")
        print()
        
        results = {
            'E_values': E_values,
            'time_to_halt': [],
            'final_F': [],
            'K_output': [],
            'mutations': [],
            'replications': [],
            'crossovers': [],
            'learns': [],
            'genome_bank_size': []
        }
        
        for i, E in enumerate(E_values):
            print(f"Run {i+1}/{len(E_values)}: E={E:.1f}...", end=' ')
            
            cbf = ChaosBFv2(
                genome,
                E=E,
                T=T,
                seed=seed + i,
                use_pid=True,
                verbose=False
            )
            
            cbf.run(steps=steps)
            stats = cbf.get_stats()
            
            results['time_to_halt'].append(stats['steps'])
            results['final_F'].append(stats['free_energy'])
            results['K_output'].append(stats['output_complexity'])
            results['mutations'].append(stats['mutations'])
            results['replications'].append(stats['replications'])
            results['crossovers'].append(stats['crossovers'])
            results['learns'].append(stats['learns'])
            results['genome_bank_size'].append(stats['genome_bank_size'])
            
            print(f"steps={stats['steps']}, reps={stats['replications']}, "
                  f"K={stats['output_complexity']:.2f}")
        
        # Plot results
        self._plot_energy_ablation(results)
        
        # Save results
        with open(f'{self.output_dir}/energy_ablation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results['energy_ablation'] = results
        return results
    
    def _plot_energy_ablation(self, results: Dict):
        """Plot energy ablation results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Energy Budget Ablation', fontsize=16, fontweight='bold')
        
        E_values = results['E_values']
        
        # Steps vs Energy
        axes[0, 0].plot(E_values, results['time_to_halt'], 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Initial Energy (E₀)')
        axes[0, 0].set_ylabel('Steps to Halt')
        axes[0, 0].set_title('Execution Time vs Energy Budget')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Replications vs Energy
        axes[0, 1].plot(E_values, results['replications'], 'r-o', linewidth=2)
        axes[0, 1].set_xlabel('Initial Energy (E₀)')
        axes[0, 1].set_ylabel('Replications')
        axes[0, 1].set_title('Evolutionary Events vs Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Complexity vs Energy
        axes[1, 0].plot(E_values, results['K_output'], 'g-o', linewidth=2)
        axes[1, 0].set_xlabel('Initial Energy (E₀)')
        axes[1, 0].set_ylabel('Output Complexity K(O)')
        axes[1, 0].set_title('Information Production vs Energy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Genome bank vs Energy
        axes[1, 1].plot(E_values, results['genome_bank_size'], 'm-o', linewidth=2)
        axes[1, 1].set_xlabel('Initial Energy (E₀)')
        axes[1, 1].set_ylabel('Genome Bank Size')
        axes[1, 1].set_title('Population Growth vs Energy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/energy_ablation.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {self.output_dir}/energy_ablation.png")
        plt.close()
    
    def operator_knockout(
        self,
        genome: str,
        operators_to_test: List[str] = ['?', '*', '@', '=', '~', '{'],
        E: float = 200.0,
        T: float = 0.6,
        steps: int = 5000,
        seed: int = 42
    ) -> Dict:
        """
        Experiment 3: Operator Knockout
        
        Disable one operator at a time and measure impact on:
        - K(O) / ΔE (info per energy)
        - Elite diversity
        
        This quantifies each operator's causal contribution.
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: Operator Knockout")
        print("="*80)
        print(f"Genome: {genome}")
        print(f"Operators to test: {operators_to_test}")
        print()
        
        results = {
            'operators': ['baseline'] + operators_to_test,
            'info_per_energy': [],
            'K_output': [],
            'energy_used': [],
            'mutations': [],
            'replications': [],
            'crossovers': [],
            'learns': [],
            'genome_bank_size': [],
            'elite_count': []
        }
        
        # Baseline (no knockout)
        print("Run baseline (no knockout)...", end=' ')
        cbf = ChaosBFv2(genome, E=E, T=T, seed=seed, use_pid=True, verbose=False)
        cbf.run(steps=steps)
        stats = cbf.get_stats()
        
        energy_used = abs(E - stats['energy'])
        info_per_energy = stats['output_complexity'] / max(1.0, energy_used)
        
        results['info_per_energy'].append(info_per_energy)
        results['K_output'].append(stats['output_complexity'])
        results['energy_used'].append(energy_used)
        results['mutations'].append(stats['mutations'])
        results['replications'].append(stats['replications'])
        results['crossovers'].append(stats['crossovers'])
        results['learns'].append(stats['learns'])
        results['genome_bank_size'].append(stats['genome_bank_size'])
        results['elite_count'].append(stats['elite_count'])
        
        print(f"K/E={info_per_energy:.4f}, K={stats['output_complexity']:.2f}")
        
        # Knockout experiments
        for op in operators_to_test:
            print(f"Run knockout '{op}'...", end=' ')
            
            # Remove operator from genome
            knocked_genome = genome.replace(op, '')
            if not knocked_genome:
                knocked_genome = '+'  # Minimal valid program
            
            cbf = ChaosBFv2(knocked_genome, E=E, T=T, seed=seed, use_pid=True, verbose=False)
            cbf.run(steps=steps)
            stats = cbf.get_stats()
            
            energy_used = abs(E - stats['energy'])
            info_per_energy = stats['output_complexity'] / max(1.0, energy_used)
            
            results['info_per_energy'].append(info_per_energy)
            results['K_output'].append(stats['output_complexity'])
            results['energy_used'].append(energy_used)
            results['mutations'].append(stats['mutations'])
            results['replications'].append(stats['replications'])
            results['crossovers'].append(stats['crossovers'])
            results['learns'].append(stats['learns'])
            results['genome_bank_size'].append(stats['genome_bank_size'])
            results['elite_count'].append(stats['elite_count'])
            
            print(f"K/E={info_per_energy:.4f}, K={stats['output_complexity']:.2f}")
        
        # Plot results
        self._plot_operator_knockout(results)
        
        # Save results
        with open(f'{self.output_dir}/operator_knockout.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.results['operator_knockout'] = results
        return results
    
    def _plot_operator_knockout(self, results: Dict):
        """Plot operator knockout results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Operator Knockout Analysis', fontsize=16, fontweight='bold')
        
        operators = results['operators']
        x_pos = np.arange(len(operators))
        
        # Info per energy
        axes[0, 0].bar(x_pos, results['info_per_energy'], color='steelblue', edgecolor='black')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(operators)
        axes[0, 0].set_ylabel('Info per Energy (K/ΔE)')
        axes[0, 0].set_title('Information Efficiency')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Output complexity
        axes[0, 1].bar(x_pos, results['K_output'], color='coral', edgecolor='black')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(operators)
        axes[0, 1].set_ylabel('Output Complexity K(O)')
        axes[0, 1].set_title('Information Production')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Evolutionary events
        width = 0.25
        axes[1, 0].bar(x_pos - width, results['mutations'], width, label='Mutations', edgecolor='black')
        axes[1, 0].bar(x_pos, results['replications'], width, label='Replications', edgecolor='black')
        axes[1, 0].bar(x_pos + width, results['crossovers'], width, label='Crossovers', edgecolor='black')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(operators)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Evolutionary Events')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Population metrics
        axes[1, 1].bar(x_pos - width/2, results['genome_bank_size'], width, 
                      label='Genome Bank', edgecolor='black')
        axes[1, 1].bar(x_pos + width/2, results['elite_count'], width, 
                      label='Elites', edgecolor='black')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(operators)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Population Diversity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/operator_knockout.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {self.output_dir}/operator_knockout.png")
        plt.close()
    
    def export_all_results(self):
        """Export all experiment results to a single JSON file."""
        filepath = f'{self.output_dir}/all_experiments.json'
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nExported all results to {filepath}")


def main():
    """Run all experiments."""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='ChaosBF Experiment Runner')
    parser.add_argument('--genome', default='++[>+<-].:{;}{?}^*=@=.#%', 
                       help='Genome to test')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/experiments',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['criticality', 'energy', 'knockout', 'all'],
                       default=['all'],
                       help='Experiments to run')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    runner = ExperimentRunner(output_dir=args.output)
    
    experiments = args.experiments
    if 'all' in experiments:
        experiments = ['criticality', 'energy', 'knockout']
    
    if 'criticality' in experiments:
        runner.criticality_sweep(args.genome)
    
    if 'energy' in experiments:
        runner.energy_budget_ablation(args.genome)
    
    if 'knockout' in experiments:
        runner.operator_knockout(args.genome)
    
    runner.export_all_results()
    
    print("\n" + "="*80)
    print("All experiments complete!")
    print("="*80)


if __name__ == '__main__':
    main()

