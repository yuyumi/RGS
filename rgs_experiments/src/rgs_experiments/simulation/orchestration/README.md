# Experiment Orchestration

This module provides the `ExperimentOrchestrator` class that coordinates all the refactored components to run complete experiment iterations.

## Overview

The `ExperimentOrchestrator` is the refactored version of the monolithic `run_one_dgp_iter` function from `simulation_main.py`. It provides a clean, modular approach to running single experiment iterations by coordinating:

- **Data Generation**: Using the `DataGenerator` for all data creation
- **Model Creation**: Using the `ModelFactory` for consistent model instantiation  
- **Model Evaluation**: Using the `ModelEvaluator` for standardized evaluation
- **Baseline Models**: Integrating sklearn baseline models (Lasso, Ridge, ElasticNet)

## Key Features

### 🎯 **Complete Experiment Coordination**
- Manages the entire experiment lifecycle from data generation to result collection
- Handles both baseline models and RFS models in a unified workflow
- Supports both cross-validation and validation set approaches

### 🔧 **Flexible Configuration**
- Supports validation set approach (`use_validation_set` parameter)
- Automatically configures cross-validation strategies
- Handles different data generator types and parameters

### 📊 **Comprehensive Results**
- Returns all metrics from baseline and RFS models
- Includes detailed k-wise evaluation for all models
- Captures timing information for performance analysis
- Maintains compatibility with original result format

## Architecture

```
ExperimentOrchestrator
├── _setup_cross_validation()     # CV strategy configuration
├── _fit_baseline_models()        # Lasso, Ridge, ElasticNet
├── _fit_rfs_models()             # BaggedGS, SmearedGS, RGSCV, GS
└── run_single_experiment()       # Main orchestration method
```

## Usage Example

```python
from simulation.orchestration import ExperimentOrchestrator
from simulation.config.parameter_loader import load_params
from simulation.data import DataGenerator

# Load parameters
params = load_params('params.json')

# Generate base design matrix
data_gen = DataGenerator(params)
X, cov_matrix = data_gen.generate_design_matrix(seed=42)

# Initialize orchestrator
orchestrator = ExperimentOrchestrator(params)

# Run single experiment
results = orchestrator.run_single_experiment(
    X=X,
    cov_matrix=cov_matrix,
    sigma=0.5,
    seed=42,
    sim_num=0
)

# Access results
print(f"Lasso RTE: {results['rte_lasso']:.3f}")
print(f"RGSCV RTE: {results['rte_rgs']:.3f}")
print(f"BaggedGS best k: {results['best_k_bagged_gs']}")
```

## Supported Models

### Baseline Models
- **Lasso**: L1-regularized linear regression
- **Ridge**: L2-regularized linear regression  
- **ElasticNet**: Combined L1/L2 regularization

### RFS Models
- **BaggedGS**: Bagged Greedy Selection ensemble
- **SmearedGS**: Data Smearing with Greedy Selection
- **RGSCV**: Random Greedy Selection with cross-validation
- **Greedy Selection**: Standard greedy feature selection

## Results Format

The orchestrator returns a comprehensive dictionary containing:

```python
{
    # Experiment metadata
    'simulation': sim_num,
    'sigma': sigma,
    'method': 'fs',  # or 'omp'
    
    # Model performance (for each model: lasso, ridge, elastic, bagged_gs, smeared_gs, rgs, gs)
    'rte_{model}': relative_test_error,
    'f_score_{model}': f_score,
    'mse_{model}': training_mse,
    'insample_{model}': insample_error,
    'df_{model}': degrees_of_freedom,
    'coef_recovery_{model}': coefficient_recovery_error,
    'support_recovery_{model}': support_recovery_accuracy,
    'outsample_mse_{model}': test_mse,
    
    # Model-specific parameters
    'best_k_{model}': optimal_k_value,
    'best_m': optimal_m_value,  # for RGSCV
    'best_noise_scale': optimal_noise_scale,  # for SmearedGS
    
    # Detailed k-wise metrics (for each k and model)
    'mse_by_k_{model}_{k}': mse_for_k,
    'insample_by_k_{model}_{k}': insample_for_k,
    'df_by_k_{model}_{k}': df_for_k,
    
    # Timing information
    'time_{model}': fitting_time_seconds,
}
```

## Integration Benefits

### 🏗️ **Modular Design**
- Each component can be tested and modified independently
- Clear separation between data generation, model fitting, and evaluation
- Easy to add new models or modify existing ones

### 🔄 **Backward Compatibility**
- Results format matches original `run_one_dgp_iter` output
- Existing analysis code works without modification
- Same random seed behavior for reproducible results

### 🚀 **Performance**
- Leverages optimized components (vectorized data generation, efficient metrics)
- Consistent cross-validation strategies across all models
- Parallel-ready design for future scaling

### 🧪 **Testability**
- Each method can be tested independently
- Clear interfaces make mocking easy for unit tests
- Comprehensive integration testing included

## Future Extensions

The orchestrator design makes it easy to:
- Add new model types through the `ModelFactory`
- Implement new evaluation metrics via the `ModelEvaluator`
- Support additional data generation strategies
- Add experiment-level optimizations (caching, parallelization)

This modular approach transforms what was previously a 700+ line monolithic function into a clean, maintainable, and extensible system while preserving all original functionality. 