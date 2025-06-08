"""
Equivalence test to verify refactored components produce identical results to original simulation_main.py.

This test compares outputs from the original functions vs. the new modular components
to ensure we haven't introduced any bugs during refactoring.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the current directory to the path so we can import both original and new modules
sys.path.append(str(Path(__file__).parent))

# Import original functions from simulation_main.py
from simulation_main import (
    load_params as original_load_params,
    get_sigma_list as original_get_sigma_list,
    get_m_grid as original_get_m_grid,
    check_matrix_rank as original_check_matrix_rank
)

# Import new modular functions
from simulation.config.parameter_loader import load_params, get_sigma_list, get_m_grid
from simulation.data.matrix_utils import check_matrix_rank
from simulation.data.data_generator import DataGenerator

# Import data generation functions for direct comparison
from rgs_experiments.utils.sim_util_dgs import generate_banded_X, generate_exact_sparsity_example


def test_parameter_loading_equivalence():
    """Test that parameter loading functions produce identical results."""
    print("=" * 60)
    print("Testing Parameter Loading Equivalence")
    print("=" * 60)
    
    param_path = Path(__file__).parent.parent / "sim_params_test.json"
    
    # Load with both methods
    original_params = original_load_params(param_path)
    new_params = load_params(param_path)
    
    # Compare results
    if original_params == new_params:
        print("✓ Parameter loading: IDENTICAL")
    else:
        print("✗ Parameter loading: DIFFERENT")
        return False
    
    # Test sigma list generation
    original_sigmas = original_get_sigma_list(
        original_params['simulation']['sigma'],
        original_params['data']['signal_proportion'],
        original_params['data']['n_predictors']
    )
    
    new_sigmas = get_sigma_list(
        new_params['simulation']['sigma'],
        new_params['data']['signal_proportion'],
        new_params['data']['n_predictors']
    )
    
    if np.allclose(original_sigmas, new_sigmas):
        print("✓ Sigma list generation: IDENTICAL")
    else:
        print("✗ Sigma list generation: DIFFERENT")
        print(f"  Original: {original_sigmas}")
        print(f"  New:      {new_sigmas}")
        return False
    
    # Test m_grid generation
    original_m_grid = original_get_m_grid(
        original_params['model']['m_grid'],
        original_params['data']['n_predictors']
    )
    
    new_m_grid = get_m_grid(
        new_params['model']['m_grid'],
        new_params['data']['n_predictors']
    )
    
    if original_m_grid == new_m_grid:
        print("✓ M-grid generation: IDENTICAL")
    else:
        print("✗ M-grid generation: DIFFERENT")
        print(f"  Original: {original_m_grid}")
        print(f"  New:      {new_m_grid}")
        return False
    
    return True


def test_matrix_utilities_equivalence():
    """Test that matrix utility functions produce identical results."""
    print("\n" + "=" * 60)
    print("Testing Matrix Utilities Equivalence")
    print("=" * 60)
    
    # Create test matrices with fixed seeds
    np.random.seed(42)
    test_matrices = [
        np.random.randn(50, 10),  # Full rank
        np.random.randn(20, 30),  # More rows than columns
    ]
    
    # Create a singular matrix
    singular_matrix = np.random.randn(50, 10)
    singular_matrix[:, -1] = singular_matrix[:, 0]  # Make last column = first column
    test_matrices.append(singular_matrix)
    
    for i, X in enumerate(test_matrices):
        original_result = original_check_matrix_rank(X)
        new_result = check_matrix_rank(X)
        
        # Compare all fields
        if original_result == new_result:
            print(f"✓ Matrix {i+1} rank check: IDENTICAL")
        else:
            print(f"✗ Matrix {i+1} rank check: DIFFERENT")
            print(f"  Original: {original_result}")
            print(f"  New:      {new_result}")
            return False
    
    return True


def test_data_generation_equivalence():
    """Test that data generation produces identical results."""
    print("\n" + "=" * 60)
    print("Testing Data Generation Equivalence")
    print("=" * 60)
    
    # Load test parameters
    param_path = Path(__file__).parent.parent / "sim_params_test.json"
    params = load_params(param_path)
    
    # Test design matrix generation
    seed = 42
    n_predictors = params['data']['n_predictors']
    n_train = params['data']['n_train']
    gamma = params['data']['banded_params']['gamma']
    
    # Generate with original function
    X_original, cov_original = generate_banded_X(
        n_predictors=n_predictors,
        n_train=n_train,
        gamma=gamma,
        seed=seed
    )
    
    # Generate with new DataGenerator
    data_gen = DataGenerator(params)
    X_new, cov_new = data_gen.generate_design_matrix(seed=seed)
    
    if np.allclose(X_original, X_new, rtol=1e-10, atol=1e-10):
        print("✓ Design matrix generation: IDENTICAL")
    else:
        print("✗ Design matrix generation: DIFFERENT")
        print(f"  Max absolute difference: {np.max(np.abs(X_original - X_new))}")
        return False
    
    if np.allclose(cov_original, cov_new, rtol=1e-10, atol=1e-10):
        print("✓ Covariance matrix generation: IDENTICAL")
    else:
        print("✗ Covariance matrix generation: DIFFERENT")
        print(f"  Max absolute difference: {np.max(np.abs(cov_original - cov_new))}")
        return False
    
    # Test response generation
    sigma = 0.5
    signal_proportion = params['data']['signal_proportion']
    response_seed = 123
    
    # Generate with original function
    _, y_original, y_true_original, beta_original, p_original, sigma_original = generate_exact_sparsity_example(
        X_original, signal_proportion, sigma, seed=response_seed
    )
    
    # Generate with new DataGenerator 
    train_data = data_gen.generate_train_data(X_original, sigma, seed=response_seed)
    
    if np.allclose(y_original, train_data['y'], rtol=1e-10, atol=1e-10):
        print("✓ Response generation: IDENTICAL")
    else:
        print("✗ Response generation: DIFFERENT")
        print(f"  Max absolute difference: {np.max(np.abs(y_original - train_data['y']))}")
        return False
    
    if np.allclose(y_true_original, train_data['y_true'], rtol=1e-10, atol=1e-10):
        print("✓ True response generation: IDENTICAL")
    else:
        print("✗ True response generation: DIFFERENT")
        return False
    
    if np.allclose(beta_original, train_data['beta_true'], rtol=1e-10, atol=1e-10):
        print("✓ Beta coefficients: IDENTICAL")
    else:
        print("✗ Beta coefficients: DIFFERENT")
        return False
    
    return True


def test_deterministic_behavior():
    """Test that both implementations produce identical results across multiple runs with same seeds."""
    print("\n" + "=" * 60)
    print("Testing Deterministic Behavior")
    print("=" * 60)
    
    param_path = Path(__file__).parent.parent / "sim_params_test.json"
    params = load_params(param_path)
    
    # Test multiple runs with same parameters
    seeds = [42, 123, 456]
    sigmas = [0.1, 0.5, 1.0]
    
    for seed in seeds:
        for sigma in sigmas:
            # Run 1
            data_gen1 = DataGenerator(params)
            X1, _ = data_gen1.generate_design_matrix(seed=seed)
            train_data1 = data_gen1.generate_train_data(X1, sigma, seed=seed+1000)
            
            # Run 2 
            data_gen2 = DataGenerator(params)
            X2, _ = data_gen2.generate_design_matrix(seed=seed)
            train_data2 = data_gen2.generate_train_data(X2, sigma, seed=seed+1000)
            
            if not (np.allclose(X1, X2) and np.allclose(train_data1['y'], train_data2['y'])):
                print(f"✗ Deterministic behavior failed for seed={seed}, sigma={sigma}")
                return False
    
    print("✓ Deterministic behavior: IDENTICAL across multiple runs")
    return True


def main():
    """Run all equivalence tests."""
    print("Testing Equivalence Between Original and Refactored Components")
    print("=" * 70)
    
    tests = [
        ("Parameter Loading", test_parameter_loading_equivalence),
        ("Matrix Utilities", test_matrix_utilities_equivalence),
        ("Data Generation", test_data_generation_equivalence),
        ("Deterministic Behavior", test_deterministic_behavior)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("EQUIVALENCE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL EQUIVALENCE TESTS PASSED!")
        print("   The refactored components produce identical results to the original code.")
        print("   It's safe to proceed with further refactoring.")
    else:
        print("⚠️  EQUIVALENCE TESTS FAILED!")
        print("   The refactored components DO NOT match the original behavior.")
        print("   DO NOT proceed until all differences are resolved.")
    
    return passed == len(results)


if __name__ == "__main__":
    main() 