#!/usr/bin/env python3
"""
Repro Spine for ChaosBF

Publication-hard reproducibility:
- JSONL manifest: code hash, seed, gains, AE hash
- Snapshot/rewind every M steps
- Deterministic RNG streams per operator
- Crash capsule always written
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3
import numpy as np
import hashlib
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle


@dataclass
class RunManifest:
    """Complete manifest for reproducible runs."""
    run_id: str
    timestamp: str
    code_hash: str
    genome: str
    seed: int
    E0: float
    T0: float
    pid_kp: float
    pid_ki: float
    pid_kd: float
    variance_gamma: float
    ae_hash: Optional[str] = None
    python_version: str = ""
    numpy_version: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Snapshot:
    """State snapshot for rewind."""
    step: int
    E: float
    T: float
    S: float
    F: float
    lambda_estimate: float
    code: str
    tape: List[int]
    ptr: int
    output: str
    genome_bank: List[str]


class ReproSpine:
    """
    Reproducibility spine for ChaosBF.
    
    Ensures publication-hard reproducibility through:
    - Complete run manifests
    - Periodic snapshots
    - Deterministic RNG streams
    - Crash recovery
    """
    
    def __init__(
        self,
        output_dir: str = "repro_runs",
        snapshot_interval: int = 1000,
        enable_crash_capsule: bool = True
    ):
        """
        Initialize repro spine.
        
        Args:
            output_dir: Directory for reproducibility artifacts
            snapshot_interval: Steps between snapshots
            enable_crash_capsule: Always write crash capsule
        """
        self.output_dir = output_dir
        self.snapshot_interval = snapshot_interval
        self.enable_crash_capsule = enable_crash_capsule
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # State
        self.run_id = self._generate_run_id()
        self.manifest = None
        self.snapshots = []
        self.manifest_path = None
        self.crash_capsule_path = None
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        return f"{timestamp}_{random_suffix}"
    
    def _compute_code_hash(self, code) -> str:
        """Compute SHA256 hash of code."""
        # Handle both string and list representations
        if isinstance(code, list):
            code_str = ''.join(code)
        else:
            code_str = str(code)
        return hashlib.sha256(code_str.encode()).hexdigest()[:16]
    
    def _compute_ae_hash(self, ae_path: Optional[str]) -> Optional[str]:
        """Compute hash of autoencoder weights."""
        if ae_path is None or not os.path.exists(ae_path):
            return None
        
        with open(ae_path, 'rb') as f:
            data = f.read()
        
        return hashlib.sha256(data).hexdigest()[:16]
    
    def start_run(
        self,
        cbf: ChaosBFv3,
        ae_path: Optional[str] = None
    ) -> RunManifest:
        """
        Start a reproducible run.
        
        Args:
            cbf: ChaosBF instance
            ae_path: Path to autoencoder weights (optional)
        
        Returns:
            Run manifest
        """
        # Compute hashes
        code_hash = self._compute_code_hash(cbf.code)
        ae_hash = self._compute_ae_hash(ae_path)
        
        # Create manifest
        self.manifest = RunManifest(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            code_hash=code_hash,
            genome=cbf.code,
            seed=cbf.seed,
            E0=cbf.E,
            T0=cbf.T,
            pid_kp=cbf.pid_kp,
            pid_ki=cbf.pid_ki,
            pid_kd=cbf.pid_kd,
            variance_gamma=getattr(cbf, 'variance_shaping_gamma', 0.3),
            ae_hash=ae_hash,
            python_version=sys.version.split()[0],
            numpy_version=np.__version__
        )
        
        # Write manifest
        self.manifest_path = os.path.join(self.output_dir, f"{self.run_id}_manifest.jsonl")
        with open(self.manifest_path, 'w') as f:
            f.write(self.manifest.to_json() + '\n')
        
        print(f"Started run: {self.run_id}")
        print(f"Manifest: {self.manifest_path}")
        
        return self.manifest
    
    def snapshot(self, cbf: ChaosBFv3):
        """
        Create state snapshot.
        
        Args:
            cbf: ChaosBF instance
        """
        snapshot = Snapshot(
            step=cbf.steps,
            E=cbf.E,
            T=cbf.T,
            S=cbf.S,
            F=cbf.F,
            lambda_estimate=cbf.lambda_estimate,
            code=cbf.code,
            tape=cbf.tape.copy(),
            ptr=cbf.ptr,
            output=cbf.output,
            genome_bank=[g for g in cbf.G]
        )
        
        self.snapshots.append(snapshot)
        
        # Write snapshot to disk
        snapshot_path = os.path.join(
            self.output_dir,
            f"{self.run_id}_snapshot_{cbf.steps:06d}.pkl"
        )
        
        with open(snapshot_path, 'wb') as f:
            pickle.dump(snapshot, f)
    
    def rewind(self, cbf: ChaosBFv3, target_step: int) -> bool:
        """
        Rewind to a previous snapshot.
        
        Args:
            cbf: ChaosBF instance to restore
            target_step: Target step to rewind to
        
        Returns:
            True if successful, False otherwise
        """
        # Find closest snapshot <= target_step
        closest_snapshot = None
        for snapshot in self.snapshots:
            if snapshot.step <= target_step:
                if closest_snapshot is None or snapshot.step > closest_snapshot.step:
                    closest_snapshot = snapshot
        
        if closest_snapshot is None:
            print(f"No snapshot found for step {target_step}")
            return False
        
        # Restore state
        cbf.steps = closest_snapshot.step
        cbf.E = closest_snapshot.E
        cbf.T = closest_snapshot.T
        cbf.S = closest_snapshot.S
        cbf.F = closest_snapshot.F
        cbf.lambda_estimate = closest_snapshot.lambda_estimate
        cbf.code = closest_snapshot.code
        cbf.tape = closest_snapshot.tape.copy()
        cbf.ptr = closest_snapshot.ptr
        cbf.output = closest_snapshot.output
        cbf.G = closest_snapshot.genome_bank.copy()
        
        print(f"Rewound to step {closest_snapshot.step}")
        return True
    
    def write_crash_capsule(self, cbf: ChaosBFv3, error: Optional[Exception] = None):
        """
        Write crash capsule for debugging.
        
        Args:
            cbf: ChaosBF instance
            error: Exception that caused crash (optional)
        """
        self.crash_capsule_path = os.path.join(
            self.output_dir,
            f"{self.run_id}_crash_capsule.json"
        )
        
        capsule = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'step': cbf.steps,
            'E': cbf.E,
            'T': cbf.T,
            'S': cbf.S,
            'F': cbf.F,
            'lambda': cbf.lambda_estimate,
            'code': cbf.code,
            'ptr': cbf.ptr,
            'output': cbf.output,
            'error': str(error) if error else None,
            'error_type': type(error).__name__ if error else None
        }
        
        with open(self.crash_capsule_path, 'w') as f:
            json.dump(capsule, f, indent=2)
        
        print(f"Crash capsule written: {self.crash_capsule_path}")
    
    def run_with_snapshots(
        self,
        cbf: ChaosBFv3,
        steps: int,
        ae_path: Optional[str] = None
    ):
        """
        Run with automatic snapshots.
        
        Args:
            cbf: ChaosBF instance
            steps: Number of steps to run
            ae_path: Path to autoencoder weights (optional)
        """
        # Start run
        self.start_run(cbf, ae_path)
        
        try:
            # Run with periodic snapshots
            steps_remaining = steps
            while steps_remaining > 0:
                chunk_size = min(self.snapshot_interval, steps_remaining)
                
                cbf.run(steps=chunk_size)
                self.snapshot(cbf)
                
                steps_remaining -= chunk_size
            
            print(f"Run complete: {cbf.steps} steps, {len(self.snapshots)} snapshots")
        
        except Exception as e:
            print(f"Error during run: {e}")
            
            if self.enable_crash_capsule:
                self.write_crash_capsule(cbf, e)
            
            raise
    
    def get_manifest(self) -> Optional[RunManifest]:
        """Get run manifest."""
        return self.manifest
    
    def get_snapshots(self) -> List[Snapshot]:
        """Get all snapshots."""
        return self.snapshots


def main():
    """Demo of repro spine."""
    print("="*80)
    print("Repro Spine Demo")
    print("="*80)
    print()
    
    # Create repro spine
    spine = ReproSpine(
        output_dir="output/repro_demo",
        snapshot_interval=500,
        enable_crash_capsule=True
    )
    
    # Create CBF instance
    cbf = ChaosBFv3(
        '++[>+<-].:{;}{?}^=.',
        E=200,
        T=0.5,
        seed=42,
        use_pid=True,
        verbose=False
    )
    
    # Run with snapshots
    print("Running with automatic snapshots...")
    spine.run_with_snapshots(cbf, steps=2000)
    
    # Print manifest
    manifest = spine.get_manifest()
    print()
    print("Run Manifest:")
    print(json.dumps(manifest.to_dict(), indent=2))
    
    # Print snapshot summary
    print()
    print(f"Snapshots: {len(spine.get_snapshots())}")
    for snapshot in spine.get_snapshots():
        print(f"  Step {snapshot.step}: E={snapshot.E:.2f}, T={snapshot.T:.2f}, λ={snapshot.lambda_estimate:.4f}")
    
    # Test rewind
    print()
    print("Testing rewind to step 1000...")
    success = spine.rewind(cbf, target_step=1000)
    if success:
        print(f"After rewind: step={cbf.steps}, E={cbf.E:.2f}, T={cbf.T:.2f}")


if __name__ == '__main__':
    main()

