#!/usr/bin/env python3
"""
Phase diagram analysis for ChaosBF v3.0.

Grid sweep over (E0, T0) with heatmaps of:
- Time to halt
- Mean λ
- QD score
- Replication count

Identifies phase boundaries:
- Subcritical (λ < 1)
- Critical band (λ ≈ 1)
- Supercritical blowup (λ > 1)
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from typing import List, Dict, Tuple


class PhaseDiagram:
    """Phase diagram analyzer for ChaosBF."""
    
    def __init__(self, output_dir: str = '/home/ubuntu/chaosbf/output/phase_diagram'):
        self.output_dir = output_dir
        self.results = {}
    
    def sweep_phase_space(
        self,
        genome: str,
        E_range: Tuple[float, float] = (50, 300),
        T_range: Tuple[float, float] = (0.1, 1.5),
        E_steps: int = 10,
        T_steps: int = 10,
        max_steps: int = 5000,
        seed: int = 42
    ) -> Dict:
        """
        Sweep (E0, T0) phase space and measure key metrics.
        
        Returns:
            Dictionary with 2D arrays for each metric
        """
        print("="*80)
        print("PHASE DIAGRAM ANALYSIS")
        print("="*80)
        print(f"Genome: {genome}")
        print(f"E range: {E_range}, steps: {E_steps}")
        print(f"T range: {T_range}, steps: {T_steps}")
        print(f"Max execution steps: {max_steps}")
        print()
        
        E_values = np.linspace(E_range[0], E_range[1], E_steps)
        T_values = np.linspace(T_range[0], T_range[1], T_steps)
        
        # Initialize result arrays
        time_to_halt = np.zeros((T_steps, E_steps))
        mean_lambda = np.zeros((T_steps, E_steps))
        qd_score = np.zeros((T_steps, E_steps))
        replication_count = np.zeros((T_steps, E_steps))
        final_complexity = np.zeros((T_steps, E_steps))
        volatility = np.zeros((T_steps, E_steps))
        
        total_runs = E_steps * T_steps
        run_count = 0
        
        for i, T0 in enumerate(T_values):
            for j, E0 in enumerate(E_values):
                run_count += 1
                print(f"Run {run_count}/{total_runs}: E0={E0:.1f}, T0={T0:.3f}...", end=' ')
                
                cbf = ChaosBFv3(
                    genome,
                    E=E0,
                    T=T0,
                    seed=seed + run_count,
                    use_pid=False,  # No PID for phase diagram
                    use_variance_shaping=False,
                    verbose=False
                )
                
                cbf.run(steps=max_steps)
                stats = cbf.get_stats()
                
                # Record metrics
                time_to_halt[i, j] = stats['steps']
                mean_lambda[i, j] = stats['branching_factor']
                replication_count[i, j] = stats['replications']
                final_complexity[i, j] = stats['output_complexity']
                volatility[i, j] = stats['lambda_volatility']
                
                # QD score: diversity × quality
                # Simple proxy: replications × complexity
                qd_score[i, j] = stats['replications'] * stats['output_complexity']
                
                print(f"steps={stats['steps']}, λ={stats['branching_factor']:.3f}, "
                      f"reps={stats['replications']}, K={stats['output_complexity']:.1f}")
        
        results = {
            'E_values': E_values.tolist(),
            'T_values': T_values.tolist(),
            'time_to_halt': time_to_halt.tolist(),
            'mean_lambda': mean_lambda.tolist(),
            'qd_score': qd_score.tolist(),
            'replication_count': replication_count.tolist(),
            'final_complexity': final_complexity.tolist(),
            'volatility': volatility.tolist()
        }
        
        # Save results
        with open(f'{self.output_dir}/phase_diagram.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot phase diagram
        self._plot_phase_diagram(E_values, T_values, time_to_halt, mean_lambda,
                                qd_score, replication_count, final_complexity, volatility)
        
        self.results = results
        return results
    
    def _plot_phase_diagram(
        self,
        E_values: np.ndarray,
        T_values: np.ndarray,
        time_to_halt: np.ndarray,
        mean_lambda: np.ndarray,
        qd_score: np.ndarray,
        replication_count: np.ndarray,
        final_complexity: np.ndarray,
        volatility: np.ndarray
    ):
        """Plot phase diagram heatmaps."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ChaosBF Phase Diagram: (E₀, T₀) Parameter Space', 
                     fontsize=16, fontweight='bold')
        
        extent = [E_values[0], E_values[-1], T_values[0], T_values[-1]]
        
        # Time to halt
        im1 = axes[0, 0].imshow(time_to_halt, aspect='auto', origin='lower',
                                extent=extent, cmap='viridis')
        axes[0, 0].set_xlabel('Initial Energy (E₀)')
        axes[0, 0].set_ylabel('Initial Temperature (T₀)')
        axes[0, 0].set_title('Time to Halt (steps)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Mean lambda with critical line
        im2 = axes[0, 1].imshow(mean_lambda, aspect='auto', origin='lower',
                                extent=extent, cmap='RdYlGn_r', vmin=0.5, vmax=1.5)
        axes[0, 1].contour(E_values, T_values, mean_lambda, levels=[1.0], 
                          colors='black', linewidths=2)
        axes[0, 1].set_xlabel('Initial Energy (E₀)')
        axes[0, 1].set_ylabel('Initial Temperature (T₀)')
        axes[0, 1].set_title('Mean Branching Factor (λ) - Critical Line at λ=1')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # QD score
        im3 = axes[0, 2].imshow(qd_score, aspect='auto', origin='lower',
                                extent=extent, cmap='plasma')
        axes[0, 2].set_xlabel('Initial Energy (E₀)')
        axes[0, 2].set_ylabel('Initial Temperature (T₀)')
        axes[0, 2].set_title('QD Score (replications × complexity)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Replication count
        im4 = axes[1, 0].imshow(replication_count, aspect='auto', origin='lower',
                                extent=extent, cmap='hot')
        axes[1, 0].set_xlabel('Initial Energy (E₀)')
        axes[1, 0].set_ylabel('Initial Temperature (T₀)')
        axes[1, 0].set_title('Replication Count')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Final complexity
        im5 = axes[1, 1].imshow(final_complexity, aspect='auto', origin='lower',
                                extent=extent, cmap='inferno')
        axes[1, 1].set_xlabel('Initial Energy (E₀)')
        axes[1, 1].set_ylabel('Initial Temperature (T₀)')
        axes[1, 1].set_title('Output Complexity K(O)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Volatility
        im6 = axes[1, 2].imshow(volatility, aspect='auto', origin='lower',
                                extent=extent, cmap='coolwarm')
        axes[1, 2].set_xlabel('Initial Energy (E₀)')
        axes[1, 2].set_ylabel('Initial Temperature (T₀)')
        axes[1, 2].set_title('Lambda Volatility (std dev)')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/phase_diagram.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved phase diagram to {self.output_dir}/phase_diagram.png")
        plt.close()
    
    def identify_phase_boundaries(self) -> Dict:
        """
        Identify phase boundaries from phase diagram data.
        
        Returns:
            Dictionary with boundary coordinates
        """
        if not self.results:
            print("No results available. Run sweep_phase_space first.")
            return {}
        
        mean_lambda = np.array(self.results['mean_lambda'])
        E_values = np.array(self.results['E_values'])
        T_values = np.array(self.results['T_values'])
        
        # Find critical band (0.9 < λ < 1.1)
        critical_mask = (mean_lambda > 0.9) & (mean_lambda < 1.1)
        
        # Find subcritical region (λ < 0.9)
        subcritical_mask = mean_lambda < 0.9
        
        # Find supercritical region (λ > 1.1)
        supercritical_mask = mean_lambda > 1.1
        
        boundaries = {
            'critical_band': {
                'count': int(np.sum(critical_mask)),
                'mean_lambda': float(np.mean(mean_lambda[critical_mask])) if np.any(critical_mask) else 0.0
            },
            'subcritical': {
                'count': int(np.sum(subcritical_mask)),
                'mean_lambda': float(np.mean(mean_lambda[subcritical_mask])) if np.any(subcritical_mask) else 0.0
            },
            'supercritical': {
                'count': int(np.sum(supercritical_mask)),
                'mean_lambda': float(np.mean(mean_lambda[supercritical_mask])) if np.any(supercritical_mask) else 0.0
            }
        }
        
        print("\nPhase Boundaries:")
        print(f"  Critical band (0.9 < λ < 1.1): {boundaries['critical_band']['count']} points")
        print(f"  Subcritical (λ < 0.9): {boundaries['subcritical']['count']} points")
        print(f"  Supercritical (λ > 1.1): {boundaries['supercritical']['count']} points")
        
        return boundaries


def main():
    """Run phase diagram analysis."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='ChaosBF Phase Diagram Analysis')
    parser.add_argument('--genome', default='++[>+<-].:{;}{?}^*=@=.#%',
                       help='Genome to analyze')
    parser.add_argument('--E-min', type=float, default=50, help='Minimum energy')
    parser.add_argument('--E-max', type=float, default=300, help='Maximum energy')
    parser.add_argument('--T-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--T-max', type=float, default=1.5, help='Maximum temperature')
    parser.add_argument('--E-steps', type=int, default=10, help='Energy steps')
    parser.add_argument('--T-steps', type=int, default=10, help='Temperature steps')
    parser.add_argument('--max-steps', type=int, default=5000, help='Max execution steps')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/phase_diagram',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run phase diagram analysis
    analyzer = PhaseDiagram(output_dir=args.output)
    
    analyzer.sweep_phase_space(
        genome=args.genome,
        E_range=(args.E_min, args.E_max),
        T_range=(args.T_min, args.T_max),
        E_steps=args.E_steps,
        T_steps=args.T_steps,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    # Identify phase boundaries
    analyzer.identify_phase_boundaries()
    
    print("\n" + "="*80)
    print("Phase diagram analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

