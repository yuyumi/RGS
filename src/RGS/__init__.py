from .core.rgs import RGS, RGSCV
from .penalized_score import create_penalized_scorer
from .utils.sim_util_dgs import *  # This will import everything in __all__

# Update the package's __all__ to include everything
__all__ = [
    'RGS',
    'RGSCV',
    'create_penalized_scorer',
    'generate_orthogonal_X',
    'generate_banded_X',
    'generate_block_X',
    'generate_exact_sparsity_example',
    'generate_inexact_sparsity_example',
    'generate_nonlinear_example',
    'generate_laplace_example',
    'generate_cauchy_example'
]