"""
Configuration file for diffusion experiments.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for data generation."""
    n_samples: int = 5000
    test_split: float = 0.2
    random_seed: int = 42
    

@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    noise_scale: float = 0.3
    max_time: float = 10.0
    mode: Literal["exp", "cos"] = "cos"
    time_distribution: Literal["uniform", "gaussian"] = "uniform"


@dataclass
class ModelConfig:
    """Configuration for Transformer model."""
    embed_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 512
    num_encoders: int = 3
    num_decoders: int = 3
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    warmup_steps: int = 2000
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    early_patience: int = 7
    min_delta: float = 1e-4
    

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str = "binary_diffusion"
    grammar_type: Literal["binary_3_level", "binary_5_level"] = "binary_3_level"
    task_type: Literal["classification", "denoising"] = "denoising"
    
    data: DataConfig = DataConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    save_dir: str = "./outputs"
    device: str = "cuda"  # or "cpu"


# Predefined experiment configurations
SMALL_EXPERIMENT = ExperimentConfig(
    experiment_name="small_test",
    grammar_type="binary_3_level",
    data=DataConfig(n_samples=1000),
    training=TrainingConfig(epochs=20, batch_size=128),
)

MEMORIZATION_EXPERIMENT = ExperimentConfig(
    experiment_name="memorization_study",
    grammar_type="binary_5_level",
    data=DataConfig(n_samples=10000),
    training=TrainingConfig(epochs=100, batch_size=512),
)

GENERALIZATION_EXPERIMENT = ExperimentConfig(
    experiment_name="generalization_study",
    grammar_type="binary_5_level",
    data=DataConfig(n_samples=5000, test_split=0.3),
    training=TrainingConfig(epochs=50, batch_size=256),
)
