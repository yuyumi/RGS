# Metrics Module Organization

This directory contains evaluation metrics organized by their semantic purpose and similarity. Each metric type is separated into focused modules for better maintainability and discoverability.

## Module Structure

### 📊 **error_metrics.py**
Error and loss metrics for model evaluation.
- `calculate_relative_test_error()` - Relative Test Error (RTE) calculation (original from simulation_main.py)

### 🎯 **support_recovery_metrics.py**
Metrics for evaluating how well models recover the true support set of non-zero coefficients.
- `calculate_f_score()` - F-score for support recovery (original from simulation_main.py)

### 🔢 **degrees_of_freedom.py**
Degrees of freedom calculations for model evaluation using the Stein formula.
- `calculate_df_for_all_k()` - DF for each k value in RGS models
- `calculate_df_for_all_k_ensemble()` - DF for each k value in ensemble models
- `calculate_mse_for_all_k()` - MSE for each k value in RGS models
- `calculate_insample_for_all_k()` - In-sample error for each k value in RGS models
- `calculate_mse_for_all_k_ensemble()` - MSE for each k value in ensemble models
- `calculate_insample_for_all_k_ensemble()` - In-sample error for each k value in ensemble models

## Usage

All metrics are imported through the main `__init__.py` file:

```python
from simulation.metrics import (
    # Original metrics from simulation_main.py
    calculate_relative_test_error,
    calculate_f_score,
    
    # Degrees of freedom metrics
    calculate_df_for_all_k,
    calculate_df_for_all_k_ensemble,
    calculate_mse_for_all_k,
    calculate_insample_for_all_k,
    calculate_mse_for_all_k_ensemble,
    calculate_insample_for_all_k_ensemble
)
```

## Adding New Metrics

To add new metrics:

1. **Determine the appropriate module** based on semantic similarity to existing metrics
2. **Add the function** to the relevant `.py` file with proper docstring and type hints
3. **Update `__init__.py`** to export the new function
4. **Follow the existing patterns** for consistency

If your metric doesn't fit into existing categories, create a new module following the naming convention `{category}_metrics.py`.

## Design Principles

- **Semantic Grouping**: Metrics are grouped by their purpose and similarity
- **Single Responsibility**: Each module has a focused purpose
- **Consistent API**: All metrics follow similar parameter and return patterns  
- **Type Safety**: Full type hints for all functions
- **Documentation**: Comprehensive docstrings with parameter descriptions
- **Preserve Originals**: Only original metrics from the codebase are included 