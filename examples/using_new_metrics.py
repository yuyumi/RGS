"""
Example: Using the original metrics in the reorganized structure.

This script demonstrates how to use the original metrics from
simulation_main.py that have been reorganized into focused modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
from simulation.metrics import (
    # Original metrics from simulation_main.py
    calculate_relative_test_error,
    calculate_f_score,
    
    # Degrees of freedom metrics (original)
    calculate_df_for_all_k,
    calculate_df_for_all_k_ensemble,
    calculate_mse_for_all_k,
    calculate_insample_for_all_k,
    calculate_mse_for_all_k_ensemble,
    calculate_insample_for_all_k_ensemble
)


def example_original_metrics():
    """Demonstrate usage of the original metrics."""
    
    # Generate some example data
    np.random.seed(42)
    n_features = 10
    
    # True coefficients (sparse)
    beta_true = np.zeros(n_features)
    beta_true[:5] = np.random.randn(5) * 2  # First 5 features are non-zero
    
    # Estimated coefficients (with some errors)
    beta_hat = beta_true + np.random.randn(n_features) * 0.1
    beta_hat[np.abs(beta_hat) < 0.2] = 0  # Threshold small values
    
    # Generate some prediction data
    n_samples = 50
    X_test = np.random.randn(n_samples, n_features)
    sigma = 0.5
    
    print("=== ORIGINAL METRICS ===")
    print("These are the metrics that existed in simulation_main.py")
    print()
    
    # Original error metric
    rte = calculate_relative_test_error(beta_hat, beta_true, X_test, sigma)
    print(f"Relative Test Error: {rte:.3f}")
    
    # Original support recovery metric
    f_score = calculate_f_score(beta_hat, beta_true)
    print(f"F-Score: {f_score:.3f}")
    
    print("\n=== DEGREES OF FREEDOM METRICS ===")
    print("These were originally in simulation_main.py and are now organized")
    print("Note: These require actual fitted models to work properly")
    print("(This is just showing the API)")


def show_original_metrics_organization():
    """Show the reorganized structure for original metrics only."""
    
    print("\n" + "="*50)
    print("ORIGINAL METRICS ORGANIZATION")
    print("="*50)
    
    print("\n📊 ERROR_METRICS.PY")
    print("  • calculate_relative_test_error (original)")
    
    print("\n🎯 SUPPORT_RECOVERY_METRICS.PY")
    print("  • calculate_f_score (original)")
    
    print("\n🔢 DEGREES_OF_FREEDOM.PY")
    print("  • calculate_df_for_all_k (original)")
    print("  • calculate_df_for_all_k_ensemble (original)")
    print("  • calculate_mse_for_all_k (original)")
    print("  • calculate_insample_for_all_k (original)")
    print("  • calculate_mse_for_all_k_ensemble (original)")
    print("  • calculate_insample_for_all_k_ensemble (original)")
    
    print("\n✅ BENEFITS OF REORGANIZATION:")
    print("  • Semantic grouping by metric type")
    print("  • Easier to find the right metric")
    print("  • Cleaner imports")
    print("  • Better maintainability")
    print("  • Ready for adding new metrics")


if __name__ == "__main__":
    example_original_metrics()
    show_original_metrics_organization() 