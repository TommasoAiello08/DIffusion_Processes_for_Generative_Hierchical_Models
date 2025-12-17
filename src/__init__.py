"""
Diffusion Processes for Generative Hierarchical Models

This package contains modules for generating synthetic hierarchical data
and training diffusion models on them.
"""

from .grammar import HierarchicalGrammar, encode_dataset, prepare_tensors, reconstruct_labels
from .diffusion import DiffusionProcess, generate_diffusion_dataset
from .models import (
    TransformerDenoiser_for_classification,
    TransformerDenoiser_for_denoise,
    PositionalEncoding,
)
from .training import (
    train_for_classification,
    train_unmasked,
    masked_train_for_denoise,
)
from .evaluation import (
    bin_accuracy,
    reverse_diffusion,
    masked_reverse_diffusion,
    batched_reverse_diffusion,
)
from .utils import (
    build_allowed_matrix,
    log_softmax_mask,
    build_inverse_map,
    tensor_to_string,
    recover_root,
)

__version__ = "0.1.0"
