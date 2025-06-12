# Simulation Parameter Guide

This guide documents all parameters used in the simulation pipeline and when they are required.

## Parameter Structure Overview

```json
{
    "simulation": {...},  // Simulation control parameters
    "data": {...},        // Data generation parameters  
    "model": {...},       // Model training parameters
    "output": {...}       // Output configuration
}
```

## Simulation Parameters

```json
"simulation": {
    "n_sim": 100,               // Number of simulation runs (required)
    "base_seed": 42,            // Base random seed (required, integer)
    "use_validation_set": false, // Whether to use validation data (optional, default: false)
    "sigma": {                  // Noise level specification (required)
        "type": "sigma",        // Type of sigma specification (required)
        "style": "list",        // How sigma values are specified (required: "list", "pve", etc.)
        "values": [0.5, 1.0]    // Sigma values or PVE values (required)
    }
}
```

## Data Generation Parameters

```json
"data": {
    "n_predictors": 100,        // Number of predictors/features (required, positive integer)
    "n_train": 500,             // Training sample size (required, positive integer)
    "n_val": 200,              // Validation sample size (optional, positive integer)
    "n_test": 300,             // Test sample size (optional, positive integer)
    "signal_proportion": 0.1,   // Proportion of predictors that are signals (required, 0 < value <= 1)
    "covariance_type": "block", // Type of covariance structure (required)
    "generator_type": "exact",  // Type of response generator (required)
    "generator_params": {...},  // Generator-specific parameters (conditional)
    "block_params": {...},      // Block covariance parameters (conditional)
    "banded_params": {...}      // Banded covariance parameters (conditional)
}
```

### Covariance Types and Required Parameters

#### 1. Orthogonal Covariance (`"covariance_type": "orthogonal"`)
- **No additional parameters required**
- Generates orthogonal design matrix

#### 2. Banded Covariance (`"covariance_type": "banded"`)
- **Requires**: `banded_params`
```json
"banded_params": {
    "gamma": 0.5              // Correlation decay parameter (required, 0 <= gamma <= 1)
}
```

#### 3. Block Covariance (`"covariance_type": "block"`)
- **Requires**: `block_params`
```json
"block_params": {
    "block_size": 20,         // Size of each block (required, positive integer)
    "within_correlation": 0.7, // Correlation within blocks (required, number)
    "fixed_design": true      // Whether to use fixed design matrix (optional, default: true)
}
```
- **Constraint**: `n_predictors` must be divisible by `block_size`

### Generator Types and Required Parameters

#### 1. Exact Sparsity (`"generator_type": "exact"`)
- **No additional parameters required**
- Creates exactly sparse coefficient vector

#### 2. Spaced Sparsity (`"generator_type": "spaced"`)
- **No additional parameters required**
- Creates spaced sparse coefficient pattern

#### 3. Inexact Sparsity (`"generator_type": "inexact"`)
- **Requires**: `generator_params` with `eta`
```json
"generator_params": {
    "eta": 0.5              // Decay parameter for weak signals (required, 0 <= eta <= 1)
}
```

#### 4. Nonlinear (`"generator_type": "nonlinear"`)
- **Requires**: `generator_params` with `eta`
```json
"generator_params": {
    "eta": 0.5              // Mixing parameter for linear/nonlinear (required, 0 <= eta <= 1)
}
```

#### 5. Laplace (`"generator_type": "laplace"`)
- **No additional parameters required**
- Generates coefficients from Laplace distribution

#### 6. Cauchy (`"generator_type": "cauchy"`)
- **No additional parameters required**
- Generates coefficients from Cauchy distribution

## Model Parameters

```json
"model": {
    "method": "fs",             // Selection method (optional, default: "fs")
    "k_max": 25,               // Maximum number of features to select (required, positive)
    "m_grid": {                // Grid search parameters (required)
        "type": "geometric",   // Type of grid (required)
        "params": {            // Grid-specific parameters (required)
            "base": 1.5,       // Base for geometric progression
            "num_points": 8    // Number of grid points
        }
    },
    "rgscv": {...},           // RGSCV-specific parameters (optional)
    "bagged_gs": {...},       // Bagged GS parameters (optional)
    "smeared_gs": {...},      // Smeared GS parameters (optional)
    "baseline": {...}         // Baseline method parameters (optional)
}
```

## Output Parameters

```json
"output": {
    "save_path": "results/raw/"  // Directory to save results (required, string)
}
```

## Parameter Validation Rules

### Cross-Parameter Constraints
1. `k_max` should not exceed `n_predictors` (warning if violated)
2. `n_train` < `n_predictors` indicates high-dimensional setting (warning)
3. `n_predictors` must be divisible by `block_size` for block covariance
4. `generator_params` with `eta` required for `inexact` and `nonlinear` generators

### Value Constraints
- `signal_proportion`: 0 < value <= 1
- `gamma` (banded): 0 <= value <= 1  
- `eta` (generators): 0 <= value <= 1
- `fixed_design` (block): boolean value
- All sample sizes: positive integers
- All seeds: integers

## Common Parameter Combinations

### Example 1: Block Covariance with Exact Sparsity
```json
{
    "data": {
        "covariance_type": "block",
        "generator_type": "exact",
        "block_params": {
            "block_size": 20,
            "within_correlation": 0.7,
            "fixed_design": true
        }
    }
}
```

### Example 2: Banded Covariance with Inexact Sparsity
```json
{
    "data": {
        "covariance_type": "banded", 
        "generator_type": "inexact",
        "banded_params": {
            "gamma": 0.5
        },
        "generator_params": {
            "eta": 0.3
        }
    }
}
```

### Example 3: Random Block Design with Nonlinear Response
```json
{
    "data": {
        "covariance_type": "block",
        "generator_type": "nonlinear", 
        "block_params": {
            "block_size": 25,
            "within_correlation": 0.8,
            "fixed_design": false
        },
        "generator_params": {
            "eta": 0.6
        }
    }
}
``` 