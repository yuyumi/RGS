# Randomized Greedy Selection (RGS)

This repository contains two Python packages:
1. `rgs`: Core implementation of the Randomized Greedy Selection algorithm
2. `rgs_experiments`: Experimental framework and utilities for comparing RGS with other methods

## Repository Structure
```
repository-root/
├── rgs/                    # Core RGS package
│   └── src/
│       └── rgs/
│           ├── __init__.py
│           ├── rgs.py
│           └── penalized_score.py
│
├── rgs_experiments/        # Experiments package
│   └── src/
│       └── rgs_experiments/
│           ├── plotting/   
│           │   └── plotting.py
│           └── utils/      
│               ├── sim_util_dgs.py
│               └── sim_util.py
│
├── scripts/               # Experiment scripts
│   ├── simulation_main.py
│   ├── simulation_runner.py
│   └── plotting_runner.py
│
├── params/               # Parameter files
│   └── sim_params.json
│
└── results/              # Results and figures
    ├── raw/             # Raw simulation results
    └── figures/         # Generated plots
```

## Installation

### Requirements
- Python >=3.8
- pip
- virtualenv (recommended)

### Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Unix/MacOS
source venv/bin/activate
```

2. Install the packages:
```bash
# Install core RGS package
pip install ./rgs

# Install experiments package
pip install ./rgs_experiments
```

## Running Experiments

### 1. Configure Parameters

Copy and modify the parameter template in params/:
```bash
# Copy template
cp templates/template_params.json params/sim_params.json

# Edit parameters as needed
```

Key parameters include:
- Covariance type (orthogonal, banded, block)
- Signal type (exact, inexact, nonlinear, laplace, cauchy)
- Number of simulations
- Sigma values (noise levels)

### 2. Run Simulations

Single parameter file:
```bash
python scripts/simulation_main.py
```

Multiple parameter files:
```bash
# Run all parameter files
python scripts/simulation_runner.py

# Run specific patterns
python scripts/simulation_runner.py --pattern "banded"
```

Results will be saved in `results/raw/` with naming pattern:
`simulation_results_{covariance}_{signal}_{timestamp}.csv`

### 3. Generate Plots

Generate plots for all results:
```bash
python scripts/plotting_runner.py
```

Generate plots for specific results:
```bash
python scripts/plotting_runner.py --pattern "banded"
```

Plots will be saved in `results/figures/` and include:
- MSE vs sigma
- MSE vs proportion of variance explained (PVE)
- In-sample vs sigma
- In-sample vs PVE
- Degrees of freedom vs sigma
- Degrees of freedom vs PVE
- MSE vs k for different sigma values
- Degrees of freedom vs k for different sigma values

## Package Usage

### Using RGS Package
```python
from rgs import RGSCV
from rgs.penalized_score import create_penalized_scorer

# First create a base scorer function with the known parameters
base_scorer = create_penalized_scorer(sigma2=sigma**2, 
                                    n=n_train, 
                                    p=p)

# Then create a scorer for a specific k value
# Note: You'll typically let RGSCV handle this internally
scorer_k = base_scorer(k=some_k_value)

# Create and fit model
rgscv = RGSCV(
    k_max=k_max,
    m_grid=m_grid,
    n_replications=n_replications,
    n_resample_iter=n_resample_iter,
    random_state=seed,
    cv=cv,
    scoring=scorer_k
)
rgscv.fit(X, y)

# Make predictions
y_pred = rgscv.predict(X)
```

### Using Experiment Utilities
```python
from rgs_experiments.utils.sim_util_dgs import (
    generate_banded_X,
    generate_exact_sparsity_example
)

# Generate design matrix
X = generate_banded_X(n_predictors, n_train)

# Generate response
X, y, y_true, beta_true, p, sigma = generate_exact_sparsity_example(
    X, 
    signal_proportion, 
    sigma,
    seed=seed
)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.