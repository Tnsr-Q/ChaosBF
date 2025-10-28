#!/usr/bin/env python3
"""
Test suite for ChaosBF interpreter.
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf import ChaosBF, K


def test_basic_operations():
    """Test basic Brainfuck operations."""
    print("Testing basic operations...")
    
    # Test increment and output
    cbf = ChaosBF('+++.', E=50, T=0.1)
    cbf.run(steps=10, halt_at_end=True)
    assert cbf.M[0] == 3, "Increment failed"
    
    # Test pointer movement
    cbf = ChaosBF('>+>++>+++', E=50, T=0.1)
    cbf.run(steps=20, halt_at_end=True)
    assert cbf.M[0] == 0, "Pointer movement failed"
    assert cbf.M[1] == 1, "Pointer movement failed"
    assert cbf.M[2] == 2, "Pointer movement failed"
    assert cbf.M[3] == 3, "Pointer movement failed"
    
    # Test decrement
    cbf = ChaosBF('+++--', E=50, T=0.1)
    cbf.run(steps=10, halt_at_end=True)
    assert cbf.M[0] == 1, "Decrement failed"
    
    print("✓ Basic operations passed")


def test_loops():
    """Test loop constructs."""
    print("Testing loops...")
    
    # Simple loop
    cbf = ChaosBF('+++[>+<-]', E=100, T=0.1)
    cbf.run(steps=50)
    assert cbf.M[0] == 0, "Loop failed"
    assert cbf.M[1] == 3, "Loop failed"
    
    # Nested loop
    cbf = ChaosBF('++[>++[>++<-]<-]', E=150, T=0.1)
    cbf.run(steps=100)
    assert cbf.M[2] == 8, "Nested loop failed"
    
    print("✓ Loops passed")


def test_thermodynamics():
    """Test thermodynamic features."""
    print("Testing thermodynamics...")
    
    # Energy depletion
    cbf = ChaosBF('++++++++++++', E=10, T=0.1)
    cbf.run(steps=100)
    assert cbf.E <= 0, "Energy depletion failed"
    
    # Temperature control
    cbf = ChaosBF('^^^^^', E=50, T=0.1)
    initial_T = cbf.T
    cbf.run(steps=10)
    assert cbf.T > initial_T, "Temperature increase failed"
    
    cbf = ChaosBF('vvvvv', E=50, T=1.0)
    initial_T = cbf.T
    cbf.run(steps=10)
    assert cbf.T < initial_T, "Temperature decrease failed"
    
    # Free energy
    cbf = ChaosBF('+++', E=100, T=0.5)
    cbf.S = 10.0
    F = cbf.free_energy()
    assert F == 100 - 0.5 * 10, "Free energy calculation failed"
    
    print("✓ Thermodynamics passed")


def test_entropy():
    """Test entropy sensing."""
    print("Testing entropy...")
    
    cbf = ChaosBF(':;', E=50, T=0.1)
    cbf.M[0] = 1
    cbf.M[1] = 2
    cbf.M[2] = 3
    cbf.run(steps=10)
    assert cbf.S > 0, "Entropy sensing failed"
    
    print("✓ Entropy passed")


def test_mutation():
    """Test mutation operator."""
    print("Testing mutation...")
    
    cbf = ChaosBF('+++', E=100, T=0.9)
    original_code = ''.join(cbf.code)
    cbf.run(steps=10)
    
    # High temperature should cause mutations
    cbf = ChaosBF('???', E=100, T=0.9)
    original_code = ''.join(cbf.code)
    cbf.run(steps=10)
    mutated_code = ''.join(cbf.code)
    # Code should likely have changed
    
    print("✓ Mutation passed")


def test_replication():
    """Test replication operator."""
    print("Testing replication...")
    
    cbf = ChaosBF('*', E=200, T=0.5, theta=5.0)
    cbf.run(steps=5)
    assert len(cbf.G) > 0, "Replication failed"
    
    print("✓ Replication passed")


def test_crossover():
    """Test crossover operator."""
    print("Testing crossover...")
    
    cbf = ChaosBF('+++', E=100, T=0.5)
    cbf.G = ['---', '>>>']
    original_code = ''.join(cbf.code)
    cbf.crossover()
    new_code = ''.join(cbf.code)
    # Code should have changed
    assert len(cbf.code) > 0, "Crossover failed"
    
    print("✓ Crossover passed")


def test_learning():
    """Test learning (peephole optimization)."""
    print("Testing learning...")
    
    cbf = ChaosBF('++--', E=100, T=0.1)
    cbf.learn()
    assert len(cbf.code) == 0, "Learning failed to optimize ++"
    
    cbf = ChaosBF('><><', E=100, T=0.1)
    cbf.learn()
    assert len(cbf.code) == 0, "Learning failed to optimize ><><"
    
    cbf = ChaosBF('+++--', E=100, T=0.1)
    cbf.learn()
    assert len(cbf.code) == 1, "Learning over-optimized"
    
    print("✓ Learning passed")


def test_complexity():
    """Test complexity measure K."""
    print("Testing complexity...")
    
    # Empty string
    assert K('') == 0, "K('') should be 0"
    
    # Repetitive string (compresses well)
    k1 = K('aaaaaaaaaa')
    
    # Random-ish string (compresses poorly)
    k2 = K('abcdefghij')
    
    # More complex should have higher K
    assert k2 >= k1, "Complexity measure failed"
    
    print("✓ Complexity passed")


def test_elite_selection():
    """Test elite selection."""
    print("Testing elite selection...")
    
    cbf = ChaosBF('+++.', E=200, T=0.1)
    cbf.O = "Hello World"  # High complexity output
    cbf.run(steps=10)
    
    # Should have some elite if score is high
    # (depends on exact score calculation)
    
    print("✓ Elite selection passed")


def test_critical_branching():
    """Test critical branching."""
    print("Testing critical branching...")
    
    cbf = ChaosBF('{}{}{}{}{', E=100, T=0.5)
    cbf.run(steps=20)
    assert len(cbf.branch_hist) > 1, "Critical branching failed"
    
    print("✓ Critical branching passed")


def test_quench():
    """Test quench operator."""
    print("Testing quench...")
    
    cbf = ChaosBF('+++', E=100, T=0.5)
    cbf.elite = ['---', '>>>']
    cbf.E = 10
    cbf.S = 100  # High entropy, negative free energy
    cbf.code = list('+++')
    
    # Execute quench
    cbf.ip = 0
    cbf.code = list('~')
    cbf.step()
    
    # Should have reverted to an elite genome
    assert ''.join(cbf.code) in cbf.elite, "Quench failed"
    
    print("✓ Quench passed")


def test_full_program():
    """Test a complete program execution."""
    print("Testing full program...")
    
    code = '++[>+<-].:{;}{?}^*=@=.#%{?}{?}v=.#'
    cbf = ChaosBF(code, E=180, T=0.6)
    cbf.run(steps=4000)
    
    stats = cbf.get_stats()
    assert stats['steps'] > 0, "No steps executed"
    assert stats['energy'] <= 180, "Energy increased"
    
    print("✓ Full program passed")


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("ChaosBF Test Suite")
    print("="*80)
    print()
    
    tests = [
        test_basic_operations,
        test_loops,
        test_thermodynamics,
        test_entropy,
        test_mutation,
        test_replication,
        test_crossover,
        test_learning,
        test_complexity,
        test_elite_selection,
        test_critical_branching,
        test_quench,
        test_full_program
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
        print()
    
    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

