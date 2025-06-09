# Randomized Greedy Selection (RGS)

A research framework for Randomized Greedy Selection algorithms with comprehensive simulation and comparison tools.

## Quick Start

### Installation

1. **Set up environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Unix/MacOS
source venv/bin/activate
```

2. **Install packages:**
```bash
# Install core RGS package
pip install ./rgs

# Install experiments package  
pip install ./rgs_experiments
```

### Running Simulations

**Single simulation:**
```bash
# Run with specific parameter file
python scripts/run_simulation.py params/your_params.json

# Validate parameters only
python scripts/run_simulation.py --validate-only params/your_params.json
```

**Batch simulations:**
```bash
# Run all parameter files in params/
python scripts/run_simulation.py

# Run files matching pattern
python scripts/run_simulation.py --pattern "banded"

# Use different parameter directory
python scripts/run_simulation.py --params-dir old_params
```

### Generate Plots

```bash
# Generate plots for all results
python scripts/plotting_runner.py

# Plot specific results
python scripts/plotting_runner.py --pattern "orthogonal"
```

## Configuration

### Parameter Files

Create JSON parameter files in `params/` directory. Key parameters:

- **Data generation:**
  - `covariance_type`: "orthogonal", "banded", "block"
  - `signal_type`: "exact", "inexact", "nonlinear", "laplace", "cauchy"
  - `n_train`, `n_test`, `n_predictors`: Sample sizes and dimensions

- **Simulation settings:**
  - `n_sim`: Number of simulation replications
  - `sigma_vals`: Noise levels to test
  - `seed`: Random seed for reproducibility

- **Methods:**
  - `k_max`, `m_grid`: RGS parameters
  - `n_replications`, `n_resample_iter`: Bootstrap parameters

See `templates/` for example parameter files.

## Package Usage

### Core RGS Algorithm

```python
from rgs import RGSCV
from rgs.penalized_score import create_penalized_scorer

# Create penalized scorer
scorer = create_penalized_scorer(sigma2=sigma**2, n=n_train, p=n_features)

# Fit RGS model
model = RGSCV(
    k_max=50,
    m_grid=[5, 10, 20],
    n_replications=100,
    n_resample_iter=50,
    cv=5,
    scoring=scorer(k=25),  # Example k value
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Simulation Utilities

```python
from rgs_experiments.utils.sim_util_dgs import (
    generate_banded_X,
    generate_exact_sparsity_example
)

# Generate correlated design matrix
X = generate_banded_X(n_features=100, n_samples=500)

# Generate sparse response with known structure
X, y, y_true, beta_true, p, sigma = generate_exact_sparsity_example(
    X, signal_proportion=0.2, sigma=1.0, seed=42
)
```

## Output

- **Simulation results:** Saved to `results/raw/` as CSV files
- **Plots:** Generated in `results/figures/` including:
  - MSE vs noise level
  - MSE vs proportion of variance explained
  - Degrees of freedom comparisons
  - Performance across different k values

## Version

**v1.1.0** - Major performance improvements and critical bug fixes. Now supports large-scale problems (p=2000+) with significantly improved computational efficiency.

## License

MIT License - see LICENSE file for details.