#!/usr/bin/env python3
"""
Enhanced CLI for ChaosBF v2 with logging, manifest generation, and experiment management.

Features:
- JSONL run logging
- Run manifest with metadata (seed, version, commit hash)
- Checkpoint/snapshot support
- Trace export with extended schema
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v2 import ChaosBFv2, K
import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional, List
import hashlib


VERSION = "2.0.0"


def get_code_hash(code: str) -> str:
    """Get hash of the code for reproducibility."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class ChaosBFRunner:
    """Enhanced runner with logging and checkpointing."""
    
    def __init__(
        self,
        output_dir: str = '/home/ubuntu/chaosbf/output',
        trace_interval: int = 100,
        checkpoint_interval: int = 1000
    ):
        self.output_dir = output_dir
        self.trace_interval = trace_interval
        self.checkpoint_interval = checkpoint_interval
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_file = None
        self.manifest = {}
    
    def create_manifest(
        self,
        genome: str,
        E: float,
        T: float,
        seed: Optional[int],
        **kwargs
    ) -> dict:
        """Create run manifest with metadata."""
        manifest = {
            'run_id': self.run_id,
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'genome': genome,
            'genome_hash': get_code_hash(genome),
            'parameters': {
                'E': E,
                'T': T,
                'seed': seed,
                **kwargs
            },
            'system': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        return manifest
    
    def save_manifest(self, manifest: dict):
        """Save run manifest to JSON."""
        filepath = f'{self.output_dir}/manifest_{self.run_id}.json'
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {filepath}")
    
    def init_trace_log(self):
        """Initialize JSONL trace log."""
        filepath = f'{self.output_dir}/trace_{self.run_id}.jsonl'
        self.trace_file = open(filepath, 'w')
        print(f"Logging trace to {filepath}")
    
    def log_trace(self, cbf: ChaosBFv2):
        """Log current state to JSONL trace."""
        if not self.trace_file:
            return
        
        stats = cbf.get_stats()
        
        # Extended trace with PID and derivatives
        trace_entry = {
            'step': stats['steps'],
            'E': stats['energy'],
            'T': stats['temperature'],
            'S': stats['entropy'],
            'F': stats['free_energy'],
            'lambda': stats['branching_factor'],
            'K_output': stats['output_complexity'],
            'dS_dt': stats['dS_dt'],
            'dK_dt': stats['dK_dt'],
            'pid_p': stats['pid_p'],
            'pid_i': stats['pid_i'],
            'pid_d': stats['pid_d'],
            'mutations': stats['mutations'],
            'replications': stats['replications'],
            'crossovers': stats['crossovers'],
            'learns': stats['learns'],
            'genome_bank_size': stats['genome_bank_size'],
            'elite_count': stats['elite_count']
        }
        
        self.trace_file.write(json.dumps(trace_entry) + '\n')
        self.trace_file.flush()
    
    def save_checkpoint(self, cbf: ChaosBFv2, step: int):
        """Save checkpoint snapshot."""
        snapshot = cbf.snapshot()
        filepath = f'{self.output_dir}/checkpoint_{self.run_id}_step{step}.json'
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"Saved checkpoint to {filepath}")
    
    def run(
        self,
        genome: str,
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 20000,
        seed: Optional[int] = None,
        use_pid: bool = True,
        grammar_aware: bool = True,
        save_checkpoints: bool = True,
        verbose: bool = False
    ):
        """Run ChaosBF with full logging and checkpointing."""
        
        # Create manifest
        manifest = self.create_manifest(
            genome, E, T, seed,
            steps=steps,
            use_pid=use_pid,
            grammar_aware=grammar_aware
        )
        self.save_manifest(manifest)
        
        # Initialize trace log
        self.init_trace_log()
        
        # Create interpreter
        cbf = ChaosBFv2(
            genome,
            E=E,
            T=T,
            seed=seed,
            use_pid=use_pid,
            grammar_aware=grammar_aware,
            verbose=verbose
        )
        
        print(f"\nRunning ChaosBF v{VERSION}")
        print(f"Run ID: {self.run_id}")
        print(f"Genome: {genome}")
        print(f"Seed: {cbf.seed}")
        print()
        
        # Run with periodic logging and checkpointing
        step_count = 0
        while cbf.E > 0 and step_count < steps:
            cbf.step()
            step_count += 1
            
            # Trace logging
            if step_count % self.trace_interval == 0:
                self.log_trace(cbf)
            
            # Checkpointing
            if save_checkpoints and step_count % self.checkpoint_interval == 0:
                self.save_checkpoint(cbf, step_count)
        
        # Final trace
        self.log_trace(cbf)
        
        # Close trace file
        if self.trace_file:
            self.trace_file.close()
        
        # Print final stats
        print("\n" + "="*80)
        print("RUN COMPLETE")
        print("="*80)
        cbf.print_stats()
        
        # Save final snapshot
        final_snapshot = cbf.snapshot()
        final_path = f'{self.output_dir}/final_{self.run_id}.json'
        with open(final_path, 'w') as f:
            json.dump(final_snapshot, f, indent=2)
        print(f"\nSaved final snapshot to {final_path}")
        
        return cbf


def main():
    """Enhanced CLI for ChaosBF v2."""
    parser = argparse.ArgumentParser(
        description=f'ChaosBF v{VERSION}: Thermodynamic, evolvable Brainfuck',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  chaosbf run "++[>+<-].#" --energy 200 --steps 5000
  
  # With seed for reproducibility
  chaosbf run "++[>+<-].:{;}{?}^*=@=.#%" --seed 42 --steps 10000
  
  # Export trace every 50 steps
  chaosbf run "++[>+<-].#" --trace 50 --export output/my_run/
  
  # Run from examples
  chaosbf run examples/hot_seed --steps 5000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a ChaosBF program')
    run_parser.add_argument('code', help='Program code or path to file')
    run_parser.add_argument('--energy', type=float, default=200.0, help='Initial energy')
    run_parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature')
    run_parser.add_argument('--steps', type=int, default=20000, help='Max steps')
    run_parser.add_argument('--seed', type=int, help='RNG seed for reproducibility')
    run_parser.add_argument('--no-pid', action='store_true', help='Disable PID controller')
    run_parser.add_argument('--no-grammar', action='store_true', help='Disable grammar-aware mutation')
    run_parser.add_argument('--trace', type=int, default=100, help='Trace logging interval')
    run_parser.add_argument('--checkpoint', type=int, default=1000, help='Checkpoint interval')
    run_parser.add_argument('--no-checkpoints', action='store_true', help='Disable checkpoints')
    run_parser.add_argument('--export', default='/home/ubuntu/chaosbf/output', help='Output directory')
    run_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'run':
        # Check if code is a file path
        code = args.code
        if os.path.isfile(args.code):
            with open(args.code, 'r') as f:
                code = f.read().strip()
        
        runner = ChaosBFRunner(
            output_dir=args.export,
            trace_interval=args.trace,
            checkpoint_interval=args.checkpoint
        )
        
        runner.run(
            genome=code,
            E=args.energy,
            T=args.temp,
            steps=args.steps,
            seed=args.seed,
            use_pid=not args.no_pid,
            grammar_aware=not args.no_grammar,
            save_checkpoints=not args.no_checkpoints,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()

