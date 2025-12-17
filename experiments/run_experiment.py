"""
Main experiment runner for diffusion on hierarchical data.

This script runs end-to-end experiments: data generation, model training, and evaluation.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grammar import HierarchicalGrammar, encode_dataset, prepare_tensors
from src.diffusion import DiffusionProcess, generate_diffusion_dataset
from src.models import TransformerDenoiser_for_denoise
from src.utils import build_allowed_matrix, init_weights_xavier
from configs.grammars import BINARY_3_LEVEL_GRAMMAR, BINARY_5_LEVEL_GRAMMAR, get_leaf_alphabet, get_sequence_length
from configs.experiment_config import ExperimentConfig, SMALL_EXPERIMENT


def setup_experiment(config: ExperimentConfig):
    """Setup experiment directories and logging."""
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.save_dir) / f"{config.experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = {
        "experiment_name": config.experiment_name,
        "grammar_type": config.grammar_type,
        "task_type": config.task_type,
        "data": config.data.__dict__,
        "diffusion": config.diffusion.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    return exp_dir


def generate_data(config: ExperimentConfig):
    """Generate synthetic hierarchical data."""
    print("=" * 60)
    print("GENERATING DATA")
    print("=" * 60)
    
    # Select grammar
    if config.grammar_type == "binary_3_level":
        grammar_spec = BINARY_3_LEVEL_GRAMMAR
    elif config.grammar_type == "binary_5_level":
        grammar_spec = BINARY_5_LEVEL_GRAMMAR
    else:
        raise ValueError(f"Unknown grammar type: {config.grammar_type}")
    
    # Create grammar and generate data
    grammar = HierarchicalGrammar(grammar_spec)
    df = grammar.generate_dataset(n_samples=config.data.n_samples)
    
    print(f"Generated {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Sample sequence: {df.iloc[0]['sequence']}")
    
    # Encode dataset
    leaf_alphabet = get_leaf_alphabet(grammar_spec)
    df = encode_dataset(df, leaf_alphabet)
    
    # Prepare tensors
    data_tensor, label_tensor = prepare_tensors(df)
    
    print(f"Data tensor shape: {data_tensor.shape}")
    print(f"Label tensor shape: {label_tensor.shape}")
    
    return data_tensor, label_tensor, grammar_spec, leaf_alphabet


def setup_model(config: ExperimentConfig, d, n):
    """Initialize model and optimizer."""
    print("\n" + "=" * 60)
    print("SETTING UP MODEL")
    print("=" * 60)
    
    model = TransformerDenoiser_for_denoise(
        d=d,
        n=n,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        num_encoders=config.model.num_encoders,
        num_decoders=config.model.num_decoders,
        dropout=config.model.dropout,
    )
    
    # Initialize weights
    model.apply(init_weights_xavier)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    return model


def train_model(model, clean_data, noisy_data, timesteps, labels, config, exp_dir, allowed=None):
    """Train the diffusion model."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Simple training loop (simplified version)
    from sklearn.model_selection import train_test_split
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    # Split data
    x_tr, x_te, y_tr, y_te, t_tr, t_te, lbl_tr, lbl_te = train_test_split(
        noisy_data, clean_data, timesteps, labels,
        test_size=config.data.test_split,
        random_state=config.data.random_seed
    )
    
    tr_loader = DataLoader(
        TensorDataset(x_tr, y_tr, t_tr, lbl_tr),
        batch_size=config.training.batch_size,
        shuffle=True
    )
    te_loader = DataLoader(
        TensorDataset(x_te, y_te, t_te, lbl_te),
        batch_size=config.training.batch_size
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.epochs):
        # Training
        model.train()
        train_loss = 0
        for x_noisy, y_clean, t, lbl in tr_loader:
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)
            t = t.to(device)
            
            optimizer.zero_grad()
            logits = model(x_noisy, t)
            logp = F.log_softmax(logits, dim=2)
            
            # Target indices
            tgt = y_clean.squeeze(1).argmax(dim=1)
            
            # Compute loss
            loss = loss_fn(
                logp.squeeze(1).transpose(1, 2).reshape(-1, logp.size(2)),
                tgt.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(tr_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_noisy, y_clean, t, lbl in te_loader:
                x_noisy = x_noisy.to(device)
                y_clean = y_clean.to(device)
                t = t.to(device)
                
                logits = model(x_noisy, t)
                logp = F.log_softmax(logits, dim=2)
                tgt = y_clean.squeeze(1).argmax(dim=1)
                
                loss = loss_fn(
                    logp.squeeze(1).transpose(1, 2).reshape(-1, logp.size(2)),
                    tgt.reshape(-1)
                )
                val_loss += loss.item()
        
        val_loss /= len(te_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - config.training.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{config.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Patience: {patience_counter}/{config.training.early_patience}")
        
        if patience_counter >= config.training.early_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save losses
    losses_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    losses_df.to_csv(exp_dir / "losses.csv", index=False)
    
    # Load best model
    model.load_state_dict(torch.load(exp_dir / "best_model.pt"))
    
    return model, losses_df


def main():
    """Main experiment pipeline."""
    # Load configuration
    config = SMALL_EXPERIMENT  # Can be changed to other configs
    
    print("Starting experiment:", config.experiment_name)
    print("Grammar:", config.grammar_type)
    print("Task:", config.task_type)
    
    # Setup experiment
    exp_dir = setup_experiment(config)
    print(f"Experiment directory: {exp_dir}")
    
    # Set random seeds
    torch.manual_seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)
    
    # Generate data
    clean_data, labels, grammar_spec, leaf_alphabet = generate_data(config)
    
    # Get data dimensions
    _, _, d, n = clean_data.shape
    
    # Create diffusion process
    diffusion = DiffusionProcess(
        noise_scale=config.diffusion.noise_scale,
        max_time=config.diffusion.max_time,
        mode=config.diffusion.mode
    )
    
    # Generate noisy data
    print("\nGenerating diffusion dataset...")
    noisy_data, timesteps, labels_expanded = generate_diffusion_dataset(
        clean_data, labels, diffusion,
        time_distribution=config.diffusion.time_distribution,
        max_time=config.diffusion.max_time
    )
    
    print(f"Noisy data shape: {noisy_data.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    
    # Build allowed matrix for grammar constraints
    allowed = build_allowed_matrix(grammar_spec['rules'], leaf_alphabet)
    
    # Setup model
    model = setup_model(config, d, n)
    
    # Train model
    model, losses = train_model(
        model, clean_data, noisy_data, timesteps, labels_expanded,
        config, exp_dir, allowed
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {exp_dir}")
    print(f"Best validation loss: {losses['val_loss'].min():.4f}")


if __name__ == "__main__":
    main()
