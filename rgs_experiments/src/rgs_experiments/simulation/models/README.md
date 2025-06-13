# Model Management Components

This module provides a clean, organized approach to model creation and evaluation for the RFS simulation experiments.

## Components

### ModelFactory
**File**: `model_factory.py`

The `ModelFactory` class encapsulates the model creation logic that was previously scattered throughout `simulation_main.py`. It provides a consistent interface for creating different types of feature selection models.

**Supported Models**:
- **BaggedGS**: Bagged Greedy Selection with cross-validation
- **SmearedGS**: Data Smearing with Greedy Selection 
- **RGSCV**: Random Greedy Selection with cross-validation
- **Greedy Selection**: Standard greedy selection (RGSCV with specific parameters)

**Key Methods**:
- `create_bagged_gs(seed, sim_num)`: Creates a BaggedGS model
- `create_smeared_gs(seed, sim_num)`: Creates a SmearedGS model
- `create_rgscv(seed, sim_num)`: Creates an RGSCV model
- `create_greedy_selection(seed, sim_num)`: Creates a Greedy Selection model
- `fit_model(model, X_train, y_train, start_time)`: Fits a model and tracks timing

### ModelEvaluator
**File**: `model_evaluator.py`

The `ModelEvaluator` class handles the evaluation of fitted models, computing various performance metrics and organizing results in a consistent format.

**Key Methods**:
- `extract_coefficients_regular(model)`: Extracts coefficients from regular models (RGSCV)
- `extract_coefficients_ensemble(model, n_features)`: Extracts average coefficients from ensemble models
- `evaluate_model_basic(...)`: Computes basic evaluation metrics
- `evaluate_model_detailed(...)`: Computes detailed k-wise metrics
- `evaluate_model_complete(...)`: Performs both basic and detailed evaluation

**Computed Metrics**:
- Basic metrics: MSE, in-sample error, degrees of freedom, RTE, F-score
- Recovery metrics: Coefficient recovery error, support recovery accuracy
- Test metrics: Out-of-sample MSE
- Detailed metrics: All metrics computed for each k value (k-wise evaluation)

## Usage Example

```python
from simulation.models import ModelFactory, ModelEvaluator
from simulation.config.parameter_loader import load_params
from rgs.mse import create_mse_scorer

# Load parameters and create scorer
params = load_params('params.json')
make_k_scorer = create_mse_scorer(sigma=0.5, n=100, p=50)

# Initialize components
model_factory = ModelFactory(params, cv_value=5, make_k_scorer=make_k_scorer)
model_evaluator = ModelEvaluator(sigma=0.5, n_train=100)

# Create and fit a model
seed, sim_num = 42, 0
bagged_model, start_time = model_factory.create_bagged_gs(seed, sim_num)
bagged_model, fitting_time = model_factory.fit_model(bagged_model, X_train, y_train, start_time)

# Evaluate the model
results = model_evaluator.evaluate_model_complete(
    model=bagged_model,
    X_train=X_train, y_train=y_train, y_true_train=y_true_train,
    X_test=X_test, y_test=y_test,
    beta_true=beta_true, cov_matrix=cov_matrix,
    model_name="bagged_gs", is_ensemble=True
)
```

## Benefits

1. **Separation of Concerns**: Model creation and evaluation are now cleanly separated
2. **Reusability**: Both components can be used independently in different contexts
3. **Consistency**: All models are created and evaluated using the same interface
4. **Testability**: Each component can be tested independently
5. **Maintainability**: Changes to model creation or evaluation logic are localized

## Integration

These components are designed to integrate seamlessly with:
- **Parameter Loading**: Works with the parameter management system
- **Data Generation**: Uses data from the data generation pipeline
- **Metrics**: Leverages the refactored metrics module
- **Main Simulation**: Will be integrated into the main simulation loop

This refactoring significantly reduces the complexity of the main simulation function while improving code organization and reusability. 