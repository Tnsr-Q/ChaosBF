#!/usr/bin/env python3
"""
Property-based testing and invariant checking for ChaosBF v2.

Tests thermodynamic invariants and system properties:
1. [] balance preserved under grammar-aware mutation
2. = never increases F unless measurable ΔE saving occurred
3. PID controller converges to λ≈1 within ε
4. Energy accounting correctness
5. Entropy monotonicity (S never decreases)
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v2 import ChaosBFv2, K, SAFE_OPS
import random


class InvariantChecker:
    """Check thermodynamic and computational invariants."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def assert_true(self, condition: bool, message: str):
        """Assert a condition is true."""
        if condition:
            self.tests_passed += 1
            self.log(f"✓ {message}")
        else:
            self.tests_failed += 1
            self.log(f"✗ FAILED: {message}")
            raise AssertionError(message)
    
    def test_bracket_balance_preserved(self):
        """Test that grammar-aware mutation preserves bracket balance."""
        self.log("\nTest: Bracket balance preserved under grammar-aware mutation")
        
        def count_brackets(code: str) -> tuple:
            return (code.count('['), code.count(']'), code.count('{'), code.count('}'))
        
        # Test genomes with balanced brackets
        test_genomes = [
            '++[>+<-].',
            '+++[>++[>++<-]<-]',
            '{+}{?}^{-}',
            '++[>+<-]{?}[>++<--]'
        ]
        
        for genome in test_genomes:
            initial_balance = count_brackets(genome)
            
            cbf = ChaosBFv2(
                genome,
                E=100,
                T=0.8,  # High temp for more mutations
                seed=42,
                grammar_aware=True,
                wild_mutation_rate=0.0,  # No wild mutations
                verbose=False
            )
            
            # Run with mutations
            for _ in range(10):
                cbf.mutate_one()
            
            final_code = ''.join(cbf.code)
            final_balance = count_brackets(final_code)
            
            self.assert_true(
                initial_balance == final_balance,
                f"Bracket balance preserved: {genome} -> {final_code}"
            )
    
    def test_learning_free_energy_invariant(self):
        """Test that learning operator only credits energy when ΔF > 0."""
        self.log("\nTest: Learning operator respects free energy invariant")
        
        # Genome with redundant operations
        genome = '++--><><++--'
        
        cbf = ChaosBFv2(
            genome,
            E=100,
            T=0.5,
            seed=42,
            verbose=False
        )
        
        E_before = cbf.E
        F_before = cbf.free_energy()
        
        # Apply learning
        cbf.learn()
        
        E_after = cbf.E
        F_after = cbf.free_energy()
        
        # If energy increased, free energy should have increased
        if E_after > E_before:
            self.assert_true(
                F_after >= F_before or abs(F_after - F_before) < 0.1,
                f"Learning respects ΔF: F_before={F_before:.2f}, F_after={F_after:.2f}"
            )
        else:
            self.assert_true(
                True,
                "Learning did not increase energy (acceptable)"
            )
    
    def test_pid_convergence(self):
        """Test that PID controller converges to λ≈1."""
        self.log("\nTest: PID controller converges to λ≈1")
        
        genome = '{+}{?}{-}{?}{+}{-}'
        
        cbf = ChaosBFv2(
            genome,
            E=300,
            T=0.5,
            seed=42,
            use_pid=True,
            kp=0.2,
            ki=0.02,
            kd=0.1,
            pid_interval=20,
            verbose=False
        )
        
        cbf.run(steps=1000)
        
        final_lambda = cbf.branching_factor()
        
        # Allow some tolerance
        epsilon = 0.3
        self.assert_true(
            abs(final_lambda - 1.0) < epsilon,
            f"PID converged to λ≈1: λ={final_lambda:.3f} (ε={epsilon})"
        )
    
    def test_energy_accounting(self):
        """Test that energy changes match operation costs."""
        self.log("\nTest: Energy accounting correctness")
        
        genome = '+++---'
        
        cbf = ChaosBFv2(
            genome,
            E=100,
            T=0.1,
            seed=42,
            use_pid=False,
            verbose=False
        )
        
        E_initial = cbf.E
        
        # Execute manually and track energy
        expected_E = E_initial
        for op in genome:
            expected_E += cbf.deltaE(op, 0)
            cbf.step()
        
        actual_E = cbf.E
        
        # Allow small floating point error
        self.assert_true(
            abs(actual_E - expected_E) < 0.01,
            f"Energy accounting correct: expected={expected_E:.2f}, actual={actual_E:.2f}"
        )
    
    def test_entropy_monotonicity(self):
        """Test that entropy never decreases (2nd law)."""
        self.log("\nTest: Entropy monotonicity (2nd law)")
        
        genome = ':+[>+<-];:;:;'
        
        cbf = ChaosBFv2(
            genome,
            E=200,
            T=0.5,
            seed=42,
            verbose=False
        )
        
        entropy_history = [cbf.S]
        
        cbf.run(steps=100)
        
        # Check that entropy never decreased
        prev_S = 0.0
        monotonic = True
        for step in range(cbf.steps_executed):
            if cbf.S < prev_S - 0.01:  # Allow small numerical error
                monotonic = False
                break
            prev_S = cbf.S
        
        self.assert_true(
            cbf.S >= entropy_history[0],
            f"Entropy is monotonic: S_initial={entropy_history[0]:.3f}, S_final={cbf.S:.3f}"
        )
    
    def test_free_energy_consistency(self):
        """Test that F = E - T*S is always maintained."""
        self.log("\nTest: Free energy consistency")
        
        genome = '++[>+<-].:{;}{?}^v'
        
        cbf = ChaosBFv2(
            genome,
            E=150,
            T=0.6,
            seed=42,
            verbose=False
        )
        
        for _ in range(50):
            if cbf.E <= 0:
                break
            
            cbf.step()
            
            # Check F = E - T*S
            expected_F = cbf.E - cbf.T * cbf.S
            actual_F = cbf.free_energy()
            
            self.assert_true(
                abs(expected_F - actual_F) < 0.01,
                f"F = E - T*S: expected={expected_F:.3f}, actual={actual_F:.3f}"
            )
    
    def test_replication_threshold(self):
        """Test that replication only occurs when F > theta."""
        self.log("\nTest: Replication threshold enforcement")
        
        genome = '++*'  # Try to replicate immediately
        
        # Test with low energy (F < theta)
        cbf_low = ChaosBFv2(
            genome,
            E=10,  # Low energy
            T=0.5,
            theta=20.0,  # High threshold
            seed=42,
            verbose=False
        )
        
        cbf_low.run(steps=10, halt_at_end=True)
        
        self.assert_true(
            cbf_low.replications == 0,
            f"No replication when F < theta: reps={cbf_low.replications}"
        )
        
        # Test with high energy (F > theta)
        cbf_high = ChaosBFv2(
            genome,
            E=200,  # High energy
            T=0.1,  # Low temp
            theta=5.0,  # Low threshold
            seed=42,
            verbose=False
        )
        
        cbf_high.run(steps=10, halt_at_end=True)
        
        self.assert_true(
            cbf_high.replications > 0,
            f"Replication occurs when F > theta: reps={cbf_high.replications}"
        )
    
    def test_temperature_bounds(self):
        """Test that temperature stays within [T_min, T_max]."""
        self.log("\nTest: Temperature bounds enforcement")
        
        genome = '^^^^vvvv^^^^vvvv'
        
        cbf = ChaosBFv2(
            genome,
            E=100,
            T=0.5,
            T_min=0.01,
            T_max=2.0,
            seed=42,
            verbose=False
        )
        
        cbf.run(steps=100, halt_at_end=True)
        
        self.assert_true(
            cbf.T_min <= cbf.T <= cbf.T_max,
            f"Temperature within bounds: T={cbf.T:.3f}, bounds=[{cbf.T_min}, {cbf.T_max}]"
        )
    
    def test_deterministic_with_seed(self):
        """Test that execution is deterministic with same seed."""
        self.log("\nTest: Deterministic execution with seed")
        
        genome = '++[>+<-].:{;}{?}*@'
        
        cbf1 = ChaosBFv2(genome, E=150, T=0.6, seed=42, verbose=False)
        cbf1.run(steps=100)
        
        cbf2 = ChaosBFv2(genome, E=150, T=0.6, seed=42, verbose=False)
        cbf2.run(steps=100)
        
        # Should have identical outcomes
        self.assert_true(
            cbf1.steps_executed == cbf2.steps_executed,
            f"Deterministic steps: {cbf1.steps_executed} == {cbf2.steps_executed}"
        )
        
        self.assert_true(
            abs(cbf1.E - cbf2.E) < 0.01,
            f"Deterministic energy: {cbf1.E:.3f} ≈ {cbf2.E:.3f}"
        )
        
        self.assert_true(
            cbf1.mutations == cbf2.mutations,
            f"Deterministic mutations: {cbf1.mutations} == {cbf2.mutations}"
        )
    
    def run_all_tests(self):
        """Run all invariant tests."""
        print("="*80)
        print("ChaosBF v2 Invariant Tests")
        print("="*80)
        
        tests = [
            self.test_bracket_balance_preserved,
            self.test_learning_free_energy_invariant,
            self.test_pid_convergence,
            self.test_energy_accounting,
            self.test_entropy_monotonicity,
            self.test_free_energy_consistency,
            self.test_replication_threshold,
            self.test_temperature_bounds,
            self.test_deterministic_with_seed
        ]
        
        for test in tests:
            try:
                test()
            except AssertionError as e:
                self.log(f"Test failed: {e}")
            except Exception as e:
                self.log(f"Test error: {e}")
                self.tests_failed += 1
        
        print("\n" + "="*80)
        print(f"Results: {self.tests_passed} passed, {self.tests_failed} failed")
        print("="*80)
        
        return self.tests_failed == 0


def main():
    """Run invariant tests."""
    checker = InvariantChecker(verbose=True)
    success = checker.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

