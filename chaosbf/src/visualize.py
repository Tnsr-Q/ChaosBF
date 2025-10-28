#!/usr/bin/env python3
"""
Visualization and analysis tools for ChaosBF.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf import ChaosBF, K
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class ChaosBFTracer(ChaosBF):
    """Extended ChaosBF with execution tracing for visualization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = {
            'step': [],
            'energy': [],
            'temperature': [],
            'entropy': [],
            'free_energy': [],
            'branching_factor': [],
            'output_length': [],
            'output_complexity': [],
            'genome_bank_size': [],
            'elite_count': [],
            'mutations': [],
            'replications': [],
            'crossovers': [],
            'learns': []
        }
    
    def step(self):
        """Execute one step and record trace data."""
        result = super().step()
        
        # Record state
        self.trace['step'].append(self.steps_executed)
        self.trace['energy'].append(self.E)
        self.trace['temperature'].append(self.T)
        self.trace['entropy'].append(self.S)
        self.trace['free_energy'].append(self.free_energy())
        self.trace['branching_factor'].append(self.branching_factor())
        self.trace['output_length'].append(len(self.O))
        self.trace['output_complexity'].append(K(self.O))
        self.trace['genome_bank_size'].append(len(self.G))
        self.trace['elite_count'].append(len(self.elite))
        self.trace['mutations'].append(self.mutations)
        self.trace['replications'].append(self.replications)
        self.trace['crossovers'].append(self.crossovers)
        self.trace['learns'].append(self.learns)
        
        return result
    
    def plot_thermodynamics(self, output_path: str):
        """Plot thermodynamic state variables over time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ChaosBF Thermodynamic State Evolution', fontsize=16, fontweight='bold')
        
        # Energy
        axes[0, 0].plot(self.trace['step'], self.trace['energy'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Energy (E)')
        axes[0, 0].set_title('Energy Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Temperature
        axes[0, 1].plot(self.trace['step'], self.trace['temperature'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Temperature (T)')
        axes[0, 1].set_title('Temperature Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 0].plot(self.trace['step'], self.trace['entropy'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Entropy (S)')
        axes[1, 0].set_title('Entropy Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Free Energy
        axes[1, 1].plot(self.trace['step'], self.trace['free_energy'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Free Energy (F = E - T·S)')
        axes[1, 1].set_title('Free Energy Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved thermodynamics plot to {output_path}")
        plt.close()
    
    def plot_evolution(self, output_path: str):
        """Plot evolutionary dynamics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ChaosBF Evolutionary Dynamics', fontsize=16, fontweight='bold')
        
        # Mutations
        axes[0, 0].plot(self.trace['step'], self.trace['mutations'], 'c-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Cumulative Mutations')
        axes[0, 0].set_title('Mutations Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Replications
        axes[0, 1].plot(self.trace['step'], self.trace['replications'], 'orange', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Replications')
        axes[0, 1].set_title('Replications Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Genome Bank & Elite
        axes[1, 0].plot(self.trace['step'], self.trace['genome_bank_size'], 'b-', 
                       linewidth=2, label='Genome Bank')
        axes[1, 0].plot(self.trace['step'], self.trace['elite_count'], 'r-', 
                       linewidth=2, label='Elite')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Population Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Crossovers & Learns
        axes[1, 1].plot(self.trace['step'], self.trace['crossovers'], 'purple', 
                       linewidth=2, label='Crossovers')
        axes[1, 1].plot(self.trace['step'], self.trace['learns'], 'green', 
                       linewidth=2, label='Learns')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Cumulative Count')
        axes[1, 1].set_title('Crossovers & Learning Events')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved evolution plot to {output_path}")
        plt.close()
    
    def plot_criticality(self, output_path: str):
        """Plot criticality and complexity metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ChaosBF Criticality & Complexity', fontsize=16, fontweight='bold')
        
        # Branching Factor
        axes[0, 0].plot(self.trace['step'], self.trace['branching_factor'], 'darkblue', linewidth=2)
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='λ = 1 (critical)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Branching Factor (λ)')
        axes[0, 0].set_title('Criticality: Branching Factor')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 2])
        
        # Output Length
        axes[0, 1].plot(self.trace['step'], self.trace['output_length'], 'teal', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Output Length')
        axes[0, 1].set_title('Output Buffer Growth')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Output Complexity
        axes[1, 0].plot(self.trace['step'], self.trace['output_complexity'], 'darkgreen', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Complexity K(O)')
        axes[1, 0].set_title('Output Complexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase Space: Free Energy vs Temperature
        scatter = axes[1, 1].scatter(
            self.trace['temperature'], 
            self.trace['free_energy'],
            c=self.trace['step'],
            cmap='viridis',
            alpha=0.6,
            s=10
        )
        axes[1, 1].set_xlabel('Temperature (T)')
        axes[1, 1].set_ylabel('Free Energy (F)')
        axes[1, 1].set_title('Phase Space: F vs T')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Step')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved criticality plot to {output_path}")
        plt.close()
    
    def plot_all(self, prefix: str = '/home/ubuntu/chaosbf/output/trace'):
        """Generate all visualization plots."""
        self.plot_thermodynamics(f'{prefix}_thermodynamics.png')
        self.plot_evolution(f'{prefix}_evolution.png')
        self.plot_criticality(f'{prefix}_criticality.png')
    
    def export_trace(self, output_path: str):
        """Export trace data to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = list(self.trace.keys())
            writer.writerow(headers)
            
            # Data rows
            n = len(self.trace['step'])
            for i in range(n):
                row = [self.trace[key][i] for key in headers]
                writer.writerow(row)
        
        print(f"Saved trace data to {output_path}")


def compare_programs(programs: dict, steps: int = 5000, output_prefix: str = '/home/ubuntu/chaosbf/output/comparison'):
    """
    Compare multiple ChaosBF programs side-by-side.
    
    Args:
        programs: Dict of {name: {'code': code, 'E': E, 'T': T}}
        steps: Number of steps to run each program
        output_prefix: Prefix for output files
    """
    results = {}
    
    print("Running programs for comparison...")
    for name, config in programs.items():
        print(f"\nRunning {name}...")
        cbf = ChaosBFTracer(
            config['code'],
            E=config.get('E', 200),
            T=config.get('T', 0.5)
        )
        cbf.run(steps=steps)
        results[name] = cbf
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ChaosBF Program Comparison', fontsize=16, fontweight='bold')
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    for idx, (name, cbf) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # Energy
        axes[0, 0].plot(cbf.trace['step'], cbf.trace['energy'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
        
        # Temperature
        axes[0, 1].plot(cbf.trace['step'], cbf.trace['temperature'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
        
        # Free Energy
        axes[0, 2].plot(cbf.trace['step'], cbf.trace['free_energy'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
        
        # Branching Factor
        axes[1, 0].plot(cbf.trace['step'], cbf.trace['branching_factor'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
        
        # Mutations
        axes[1, 1].plot(cbf.trace['step'], cbf.trace['mutations'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
        
        # Genome Bank
        axes[1, 2].plot(cbf.trace['step'], cbf.trace['genome_bank_size'], 
                       color=color, linewidth=2, label=name, alpha=0.7)
    
    # Configure axes
    axes[0, 0].set_title('Energy')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('E')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Temperature')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('T')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Free Energy')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('F')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Branching Factor')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('λ')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Mutations')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_title('Genome Bank Size')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Size')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_prefix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()
    
    return results


def main():
    """Command-line interface for visualization."""
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <code> [options]")
        print("Options:")
        print("  --energy E       Initial energy (default: 200)")
        print("  --temp T         Initial temperature (default: 0.5)")
        print("  --steps N        Max steps (default: 5000)")
        print("  --output PATH    Output prefix (default: /home/ubuntu/chaosbf/output/trace)")
        sys.exit(1)
    
    code = sys.argv[1]
    
    # Parse options
    E = 200.0
    T = 0.5
    steps = 5000
    output = '/home/ubuntu/chaosbf/output/trace'
    
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
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Run with tracing
    print(f"Running ChaosBF with tracing...")
    cbf = ChaosBFTracer(code, E=E, T=T)
    cbf.run(steps=steps)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    cbf.plot_all(prefix=output)
    cbf.export_trace(f'{output}_data.csv')
    
    print(f"\nDone!")


if __name__ == '__main__':
    main()

