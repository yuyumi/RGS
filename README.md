# Randomized Greedy Selection (RGS)

A research framework for Randomized Greedy Selection algorithms with comprehensive simulation and comparison tools.

## Project Structure

```
RFS/
├── scripts/              # Execution scripts
│   ├── run_simulation.py
│   ├── plotting_runner.py
│   └── csv_combiner_runner.py
├── rgs_experiments/      # Experiments package
│   └── src/rgs_experiments/
│       ├── simulation/   # Simulation utilities
│       ├── plotting/     # Plotting functions
│       └── utils/        # SNR and other utilities
├── rgs/                  # Core RGS algorithm package
├── results/              # Simulation outputs
│   ├── raw/             # Raw CSV results
│   └── figures/         # Generated plots
├── params/              # Parameter configuration files
└── templates/           # Example configurations
```

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
python scripts/plotting_runner.py --pattern "banded"
```

## Configuration

### Parameter Files

Create JSON parameter files in `params/` directory. Key parameters:

- **Data generation:**
  - `covariance_type`: "banded", "block"
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

### SNR and Signal Strength Analysis

```python
from rgs_experiments.utils.snr_utils import (
    get_signal_strength_from_results,
    compute_snr_from_signal_strength
)

# Compute signal strength from simulation results
signal_strength = get_signal_strength_from_results(
    results_path="results/raw/simulation_results.csv",
    method="from_params"  # Uses parameter files for true beta
)

# Convert to SNR
snr = compute_snr_from_signal_strength(signal_strength, sigma=1.0)
```

### Advanced Plotting

```python
from rgs_experiments.plotting.plotting import (
    plot_metric_by_variance_explained,
    plot_mse_vs_df_by_k
)

# Plot performance vs variance explained
fig = plot_metric_by_variance_explained(
    results_path="results/raw/simulation_results.csv",
    metric='mse',
    show_std=True,
    log_scale=False
)

# Plot MSE vs degrees of freedom trade-offs
fig = plot_mse_vs_df_by_k(
    results_path="results/raw/simulation_results.csv",
    target_sigma=1.0,
    show_std=True
)
```

## Versioning

### v1.2.0 - Enhanced Robustness

- Resolved shape mismatch error in QR factorization updates that occurred with small orthogonal components
- Added comprehensive signal-to-noise ratio computation from parameter files
- Enhanced plotting functions to handle edge cases (single data points, NaN standard deviations)
- Eliminated fallback values

### v1.1.0 - Performance and Scalability

- Major performance improvements for large-scale problems (p=2000+)
- Significantly improved computational efficiency
- Enhanced memory management

## Output

- **Simulation results:** Saved to `results/raw/` as CSV files with comprehensive metrics
- **Parameter files:** Corresponding JSON files with simulation configurations
- **Plots:** Generated in `results/figures/` including:
  - MSE vs noise level (SNR)
  - MSE vs proportion of variance explained (PVE)
  - Degrees of freedom comparisons
  - Performance across different k values
  - Method comparison visualizations


## License

MIT License - see LICENSE file for details.