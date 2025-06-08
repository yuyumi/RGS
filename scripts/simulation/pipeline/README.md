# Simulation Pipeline Module

This module provides the complete simulation pipeline that coordinates simulation runs across multiple parameter combinations.

## Overview

The `SimulationPipeline` class is the top-level orchestrator that coordinates all components to run complete simulations efficiently.

## Key Features

- **Complete Workflow**: Manages the entire simulation from parameter loading to result saving
- **Progress Tracking**: Real-time progress bars showing simulation progress
- **Result Management**: Automatic saving of results, summaries, and parameters
- **Modular Design**: Uses all components (DataGenerator, ModelFactory, etc.)

## Usage

### Basic Usage

```python
from simulation.pipeline import SimulationPipeline

# Run from parameter file
results_df, summary_df, timing_df = SimulationPipeline.run_from_config("params/sim_params.json")
```

### Advanced Usage

```python
# Create pipeline instance for more control
pipeline = SimulationPipeline("params/sim_params.json")

# Access pipeline parameters
print(f"Running {pipeline.params['simulation']['n_sim']} simulations")

# Run the simulation
results_df, summary_df, timing_df = pipeline.run_full_simulation()

# Results are automatically saved to the specified output directory
```

## Pipeline Workflow

The simulation pipeline follows this sequence:

1. **Parameter Loading**: Load and validate simulation parameters
2. **Design Matrix Setup**: Generate base design matrix and check rank
3. **Sigma Computation**: Calculate noise levels based on parameters
4. **Experiment Grid**: Run experiments across all simulation × sigma combinations
5. **Result Processing**: Convert results to DataFrames and create summaries
6. **File Output**: Save results, summaries, and parameters to files

## Output Files

The pipeline automatically saves:

- `simulation_results_[timestamp].csv`: Raw results from all experiments
- `simulation_summary_[timestamp].csv`: Summary statistics grouped by sigma
- `simulation_timing_summary_[timestamp].csv`: Timing statistics
- `simulation_params_[timestamp].json`: Parameters used for the simulation

## Components Used

The pipeline coordinates these components:

- **DataGenerator**: Generates design matrices and synthetic data
- **ExperimentOrchestrator**: Runs individual experiments
- **ModelFactory**: Creates and fits all model types
- **ModelEvaluator**: Evaluates models and computes metrics
- **Metrics Modules**: Calculate error metrics, F-scores, and degrees of freedom

## Progress Tracking

The pipeline provides detailed progress information:

```
Starting Simulation Pipeline
============================================================

1. Setting up design matrix...
Design Matrix Information:
  - Dimensions: (50, 100)
  - Full rank: True
  - Rank: 50 / 50
  - Condition number: 1.2345e+02

2. Computing sigma values...
   Using 3 sigma values: [0.5, 1.0, 2.0]

3. Running experiment grid...
   Total experiments: 100 simulations × 3 sigma values = 300

Sim 1/100, σ=0.500, exact, banded: 100%|██████████| 300/300 [15:32<00:00,  3.1s/it]

4. Processing results...
5. Creating summary statistics...
6. Saving results...

============================================================
SIMULATION COMPLETED
============================================================
Total runtime: 15.5 minutes
Results saved with base filename: banded_exact_20241201_143022
Results shape: (300, 203)
Save location: results/
```

## Configuration

The pipeline uses the same parameter files as the original simulation:

```json
{
  "simulation": {
    "n_sim": 100,
    "base_seed": 42,
    "sigma": {
      "type": "pve",
      "values": [0.1, 0.3, 0.5]
    }
  },
  "data": {
    "n_predictors": 100,
    "n_train": 50,
    "covariance_type": "banded",
    "generator_type": "exact"
  },
  "output": {
    "save_path": "results/"
  }
}
```

## Testing

Test the pipeline with:

```bash
cd scripts
python test_pipeline.py
```

This runs a small-scale test to verify all components work correctly together.

## Performance

The pipeline maintains the same performance characteristics as the original implementation:

- **Memory Usage**: Efficient memory management through streaming results
- **Speed**: Parallel operations where possible, optimized model fitting
- **Scalability**: Handles large parameter grids efficiently

## Error Handling

The pipeline includes robust error handling:

- **Parameter Validation**: Validates all parameters before starting
- **Matrix Rank Checking**: Warns about ill-conditioned design matrices  
- **Progress Recovery**: Clear error messages with simulation state information
- **Resource Cleanup**: Proper cleanup of temporary resources

## Reproducibility

The pipeline produces consistent results when using the same parameters and random seeds, ensuring reproducibility for research. 