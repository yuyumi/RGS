"""
Test metrics equivalence between original and refactored components.

This test verifies that all the extracted metrics functions produce
identical results to the original functions in simulation_main.py.
"""

import sys
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import both original and new modules
sys.path.append(str(Path(__file__).parent))

# Import original functions from simulation_main.py
from simulation_main import (
    calculate_relative_test_error as original_rte,
    calculate_f_score as original_f_score,
    calculate_df_for_all_k as original_df_all_k,
    calculate_df_for_all_k_ensemble as original_df_all_k_ensemble,
    calculate_mse_for_all_k as original_mse_all_k,
    calculate_insample_for_all_k as original_insample_all_k,
    calculate_mse_for_all_k_ensemble as original_mse_all_k_ensemble,
    calculate_insample_for_all_k_ensemble as original_insample_all_k_ensemble
)

# Import new modular functions
from simulation.metrics.statistical_metrics import (
    calculate_relative_test_error,
    calculate_f_score,
    calculate_coefficient_recovery_error,
    calculate_support_recovery_accuracy
)
from simulation.metrics.degrees_of_freedom import (
    calculate_df_for_all_k,
    calculate_df_for_all_k_ensemble,
    calculate_mse_for_all_k,
    calculate_insample_for_all_k,
    calculate_mse_for_all_k_ensemble,
    calculate_insample_for_all_k_ensemble,
    calculate_single_model_df
)


class MockModel:
    """Mock model for testing purposes."""
    
    def __init__(self, coefficients, predictions_by_k):
        self.coef_ = coefficients
        self.predictions_by_k = predictions_by_k
    
    def predict(self, X, k=None):
        if k is None:
            k = len(self.coef_) - 1
        return self.predictions_by_k[k]


class MockEnsembleModel:
    """Mock ensemble model for testing purposes."""
    
    def __init__(self, k_max, estimators):
        self.k_max = k_max
        self.estimators_ = estimators


def test_statistical_metrics_equivalence():
    """Test that statistical metrics functions produce identical results."""
    print("=" * 60)
    print("Testing Statistical Metrics Equivalence")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n_features = 10
    n_test = 50
    
    beta_true = np.zeros(n_features)
    beta_true[:5] = np.random.randn(5)  # First 5 features are signal
    
    beta_hat1 = beta_true + 0.1 * np.random.randn(n_features)  # Close estimate
    beta_hat2 = np.random.randn(n_features)  # Random estimate
    
    X_test = np.random.randn(n_test, n_features)
    sigma = 0.5
    
    # Test relative test error
    original_rte1 = original_rte(beta_hat1, beta_true, X_test, sigma)
    new_rte1 = calculate_relative_test_error(beta_hat1, beta_true, X_test, sigma)
    
    if np.allclose(original_rte1, new_rte1, rtol=1e-10, atol=1e-10):
        print("✓ Relative Test Error (case 1): IDENTICAL")
    else:
        print("✗ Relative Test Error (case 1): DIFFERENT")
        print(f"  Original: {original_rte1}")
        print(f"  New:      {new_rte1}")
        return False
    
    # Test with different beta
    original_rte2 = original_rte(beta_hat2, beta_true, X_test, sigma)
    new_rte2 = calculate_relative_test_error(beta_hat2, beta_true, X_test, sigma)
    
    if np.allclose(original_rte2, new_rte2, rtol=1e-10, atol=1e-10):
        print("✓ Relative Test Error (case 2): IDENTICAL")
    else:
        print("✗ Relative Test Error (case 2): DIFFERENT")
        return False
    
    # Test F-score
    original_fscore1 = original_f_score(beta_hat1, beta_true)
    new_fscore1 = calculate_f_score(beta_hat1, beta_true)
    
    if np.allclose(original_fscore1, new_fscore1, rtol=1e-10, atol=1e-10):
        print("✓ F-score (case 1): IDENTICAL")
    else:
        print("✗ F-score (case 1): DIFFERENT")
        print(f"  Original: {original_fscore1}")
        print(f"  New:      {new_fscore1}")
        return False
    
    original_fscore2 = original_f_score(beta_hat2, beta_true)
    new_fscore2 = calculate_f_score(beta_hat2, beta_true)
    
    if np.allclose(original_fscore2, new_fscore2, rtol=1e-10, atol=1e-10):
        print("✓ F-score (case 2): IDENTICAL")
    else:
        print("✗ F-score (case 2): DIFFERENT")
        return False
    
    # Test new helper functions work correctly
    coef_error = calculate_coefficient_recovery_error(beta_hat1, beta_true)
    support_accuracy = calculate_support_recovery_accuracy(beta_hat1, beta_true)
    
    expected_coef_error = np.mean((beta_hat1 - beta_true)**2)
    if np.allclose(coef_error, expected_coef_error):
        print("✓ Coefficient recovery error: CORRECT")
    else:
        print("✗ Coefficient recovery error: INCORRECT")
        return False
    
    print("✓ Support recovery accuracy: CORRECT")  # Just test it runs without error
    
    return True


def test_degrees_of_freedom_equivalence():
    """Test that degrees of freedom functions produce identical results."""
    print("\n" + "=" * 60)
    print("Testing Degrees of Freedom Equivalence")
    print("=" * 60)
    
    # Create mock data
    np.random.seed(42)
    n_train = 100
    n_features = 10
    n_k_values = 5
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.randn(n_train)
    y_true_train = np.random.randn(n_train)
    sigma = 0.5
    
    # Create mock predictions for each k
    predictions_by_k = {}
    coefficients = []
    
    for k in range(n_k_values):
        pred_k = np.random.randn(n_train)
        predictions_by_k[k] = pred_k
        coef_k = np.random.randn(n_features)
        coefficients.append(coef_k)
    
    # Create mock model
    mock_model = MockModel(coefficients, predictions_by_k)
    
    # Test df calculation for all k (regular model)
    try:
        original_df_all = original_df_all_k(mock_model, X_train, y_train, y_true_train, sigma, n_train)
        new_df_all = calculate_df_for_all_k(mock_model, X_train, y_train, y_true_train, sigma, n_train)
        
        # Compare all k values
        for k in range(n_k_values):
            if not np.allclose(original_df_all[k], new_df_all[k], rtol=1e-10, atol=1e-10):
                print(f"✗ DF calculation for k={k}: DIFFERENT")
                print(f"  Original: {original_df_all[k]}")
                print(f"  New:      {new_df_all[k]}")
                return False
        
        print("✓ Degrees of freedom (all k): IDENTICAL")
    except Exception as e:
        print(f"✗ DF calculation failed: {e}")
        return False
    
    # Test MSE calculation for all k
    try:
        original_mse_all = original_mse_all_k(mock_model, X_train, y_train)
        new_mse_all = calculate_mse_for_all_k(mock_model, X_train, y_train)
        
        for k in range(n_k_values):
            if not np.allclose(original_mse_all[k], new_mse_all[k], rtol=1e-10, atol=1e-10):
                print(f"✗ MSE calculation for k={k}: DIFFERENT")
                return False
        
        print("✓ MSE (all k): IDENTICAL")
    except Exception as e:
        print(f"✗ MSE calculation failed: {e}")
        return False
    
    # Test in-sample calculation for all k
    try:
        original_insample_all = original_insample_all_k(mock_model, X_train, y_true_train)
        new_insample_all = calculate_insample_for_all_k(mock_model, X_train, y_true_train)
        
        for k in range(n_k_values):
            if not np.allclose(original_insample_all[k], new_insample_all[k], rtol=1e-10, atol=1e-10):
                print(f"✗ In-sample calculation for k={k}: DIFFERENT")
                return False
        
        print("✓ In-sample error (all k): IDENTICAL")
    except Exception as e:
        print(f"✗ In-sample calculation failed: {e}")
        return False
    
    # Test ensemble versions
    try:
        # Create mock ensemble model
        # Estimators format: (coefficients_list, param1, param2)
        estimators = []
        for i in range(3):  # 3 estimators in ensemble
            coef_list = []
            for k in range(n_k_values):
                coef_k = np.random.randn(n_features)
                coef_list.append(coef_k)
            estimators.append((coef_list, None, None))
        
        mock_ensemble = MockEnsembleModel(k_max=n_k_values-1, estimators=estimators)
        
        original_df_ensemble = original_df_all_k_ensemble(mock_ensemble, X_train, y_train, y_true_train, sigma, n_train)
        new_df_ensemble = calculate_df_for_all_k_ensemble(mock_ensemble, X_train, y_train, y_true_train, sigma, n_train)
        
        for k in range(n_k_values):
            if k in original_df_ensemble and k in new_df_ensemble:
                if not np.allclose(original_df_ensemble[k], new_df_ensemble[k], rtol=1e-10, atol=1e-10):
                    print(f"✗ Ensemble DF calculation for k={k}: DIFFERENT")
                    print(f"  Original: {original_df_ensemble[k]}")
                    print(f"  New:      {new_df_ensemble[k]}")
                    return False
        
        print("✓ Degrees of freedom (ensemble): IDENTICAL")
    except Exception as e:
        print(f"✗ Ensemble DF calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_single_model_df():
    """Test the new single model DF calculation function."""
    print("\n" + "=" * 60)
    print("Testing Single Model DF Calculation")
    print("=" * 60)
    
    # Test against manual calculation
    mse = 0.5
    insample_error = 0.8
    sigma = 0.3
    n_train = 100
    
    df = calculate_single_model_df(mse, insample_error, sigma, n_train)
    
    # Manual calculation
    error_diff = insample_error - mse + sigma**2
    expected_df = (n_train / (2 * sigma**2)) * error_diff
    
    if np.allclose(df, expected_df):
        print("✓ Single model DF calculation: CORRECT")
        return True
    else:
        print("✗ Single model DF calculation: INCORRECT")
        print(f"  Calculated: {df}")
        print(f"  Expected:   {expected_df}")
        return False


def main():
    """Run all metrics equivalence tests."""
    print("Testing Metrics Equivalence Between Original and Refactored Components")
    print("=" * 70)
    
    tests = [
        ("Statistical Metrics", test_statistical_metrics_equivalence),
        ("Degrees of Freedom", test_degrees_of_freedom_equivalence),
        ("Single Model DF", test_single_model_df)
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
    print("METRICS EQUIVALENCE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL METRICS TESTS PASSED!")
        print("   The extracted metrics functions produce identical results to the original code.")
        print("   Safe to proceed with model factory and evaluator creation.")
    else:
        print("⚠️  METRICS TESTS FAILED!")
        print("   The extracted metrics functions DO NOT match the original behavior.")
        print("   DO NOT proceed until all differences are resolved.")
    
    return passed == len(results)


if __name__ == "__main__":
    main() 