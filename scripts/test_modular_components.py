"""
Test script to verify the new modular simulation components work correctly.

This script tests the extracted utilities against the test parameter file
to ensure we haven't broken any functionality during the refactoring.
"""

import sys
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import the simulation modules
sys.path.append(str(Path(__file__).parent))

from simulation.config.parameter_loader import load_params, get_sigma_list, get_m_grid
from simulation.config.parameter_validator import validate_params
from simulation.data.matrix_utils import check_matrix_rank, print_matrix_diagnostics
from simulation.data.data_generator import DataGenerator


def test_parameter_loading():
    """Test parameter loading and validation."""
    print("=" * 50)
    print("Testing Parameter Loading and Validation")
    print("=" * 50)
    
    # Test with the existing test parameters
    param_path = Path(__file__).parent.parent / "sim_params_test.json"
    print(f"Loading parameters from: {param_path}")
    
    # Load parameters
    params = load_params(param_path)
    print(f"✓ Successfully loaded parameters")
    
    # Validate parameters
    try:
        validated_params = validate_params(params)
        print(f"✓ Parameters passed validation")
    except Exception as e:
        print(f"✗ Parameter validation failed: {e}")
        return False
    
    # Test sigma list generation
    sigmas = get_sigma_list(
        params['simulation']['sigma'],
        params['data']['signal_proportion'],
        params['data']['n_predictors']
    )
    print(f"✓ Generated {len(sigmas)} sigma values: {sigmas[:3]}...")
    
    # Test m_grid generation
    m_grid = get_m_grid(
        params['model']['m_grid'],
        params['data']['n_predictors']
    )
    print(f"✓ Generated m_grid: {m_grid}")
    
    return True


def test_data_generation():
    """Test data generation components."""
    print("\n" + "=" * 50)
    print("Testing Data Generation")
    print("=" * 50)
    
    # Load test parameters
    param_path = Path(__file__).parent.parent / "sim_params_test.json"
    params = load_params(param_path)
    
    # Create data generator
    data_gen = DataGenerator(params)
    print(f"✓ Created DataGenerator for {params['data']['covariance_type']} covariance")
    
    # Generate base design matrix
    try:
        X, cov_matrix = data_gen.generate_design_matrix(seed=42)
        print(f"✓ Generated design matrix with shape: {X.shape}")
        
        # Test matrix diagnostics
        print("\nMatrix Diagnostics:")
        print_matrix_diagnostics(X, "Test Design Matrix")
        
    except Exception as e:
        print(f"✗ Design matrix generation failed: {e}")
        return False
    
    # Test response data generation
    try:
        sigma = 0.5
        train_data = data_gen.generate_train_data(X, sigma, seed=123)
        print(f"✓ Generated training data:")
        print(f"  - X_train shape: {train_data['X'].shape}")
        print(f"  - y_train shape: {train_data['y'].shape}")
        print(f"  - beta_true nonzeros: {np.sum(train_data['beta_true'] != 0)}")
        
        test_data = data_gen.generate_test_data(X, sigma, seed=456)
        print(f"✓ Generated test data:")
        print(f"  - X_test shape: {test_data['X'].shape}")
        print(f"  - y_test shape: {test_data['y'].shape}")
        
    except Exception as e:
        print(f"✗ Response data generation failed: {e}")
        return False
    
    return True


def test_matrix_utilities():
    """Test matrix utility functions."""
    print("\n" + "=" * 50)
    print("Testing Matrix Utilities")
    print("=" * 50)
    
    # Create a test matrix
    np.random.seed(42)
    X = np.random.randn(50, 10)
    
    # Test rank checking
    rank_info = check_matrix_rank(X)
    print(f"✓ Rank check completed:")
    print(f"  - Full rank: {rank_info['is_full_rank']}")
    print(f"  - Condition number: {rank_info['condition_number']:.2e}")
    
    # Test with a singular matrix
    X_singular = np.random.randn(50, 10)
    X_singular[:, -1] = X_singular[:, 0]  # Make last column same as first
    
    rank_info_singular = check_matrix_rank(X_singular)
    print(f"✓ Singular matrix rank check:")
    print(f"  - Full rank: {rank_info_singular['is_full_rank']}")
    print(f"  - Rank: {rank_info_singular['rank']}")
    
    return True


def main():
    """Run all tests."""
    print("Testing Modular Simulation Components")
    print("=" * 60)
    
    tests = [
        ("Parameter Loading", test_parameter_loading),
        ("Data Generation", test_data_generation),
        ("Matrix Utilities", test_matrix_utilities)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The modular components are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")


if __name__ == "__main__":
    main() 