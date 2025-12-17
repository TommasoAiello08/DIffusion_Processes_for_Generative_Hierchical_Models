"""
Diffusion process module for adding noise to hierarchical data.

Implements forward diffusion with multiple noise schedules (exponential, cosine).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


class DiffusionProcess:
    """
    Forward diffusion process for adding noise to data.
    
    Supports exponential and cosine noise schedules for controlled
    noise injection at different timesteps.
    """
    
    def __init__(self, noise_scale=0.3, max_time=10.0, mode="exp"):
        """
        Initialize diffusion process.
        
        Args:
            noise_scale (float): Scale of Gaussian noise injection
            max_time (float): Maximum timestep for diffusion
            mode (str): Noise schedule mode - "exp" or "cos"
        """
        self.noise_scale = noise_scale
        self.max_time = max_time
        self.mode = mode

    def alpha_bar(self, t):
        """
        Cosine-based noise schedule: monotonic decay from 1 → 0.
        
        Maps t ∈ [0, max_time] to α_bar ∈ [1, 0] using half-cosine.
        
        Args:
            t: Timestep(s), float or tensor
            
        Returns:
            torch.Tensor: Alpha bar values
        """
        t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
        theta = (t / self.max_time) * (np.pi / 2)
        return torch.cos(theta).pow(2)

    def add_noise(self, one_hot_matrix, t):
        """
        Add noise to data at timestep t.
        
        Args:
            one_hot_matrix (torch.Tensor): Input data, shape (B, 1, d, n)
            t: Timestep, float or tensor (B, 1) with t ∈ [0, max_time]
            
        Returns:
            torch.Tensor: Noised data, shape (B, 1, d, n)
        """
        if self.mode == "exp":
            # Exponential noise schedule
            alpha_t = np.sqrt(np.exp(-0.5 * t))
            beta_t = np.sqrt(1 - np.exp(-0.5 * t))
            noise = torch.randn_like(one_hot_matrix) * 0.2
            return alpha_t * one_hot_matrix + beta_t * noise
        
        elif self.mode == "cos":
            # Cosine noise schedule
            B, _, d, n = one_hot_matrix.shape

            if not isinstance(t, torch.Tensor):
                t = torch.tensor([[t]], dtype=torch.float32, device=one_hot_matrix.device)
            elif t.ndim == 1:
                t = t.unsqueeze(1)
            t = t.to(one_hot_matrix.device)

            alpha_bar_t = self.alpha_bar(t).view(-1, 1, 1, 1).to(one_hot_matrix.device)
            alpha_t = torch.sqrt(alpha_bar_t)
            beta_t = torch.sqrt(1.0 - alpha_bar_t)

            noise = torch.randn_like(one_hot_matrix) * self.noise_scale
            return alpha_t * one_hot_matrix + beta_t * noise
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'exp' or 'cos'.")


def generate_diffusion_dataset(data, labels, diffusion_process, 
                               time_distribution='uniform', max_time=1):
    """
    Generate training dataset by applying diffusion at random timesteps.

    Args:
        data (torch.Tensor): Clean data, shape (batch_size, 1, d, n)
        labels (torch.Tensor): Labels for each sample
        diffusion_process (DiffusionProcess): Diffusion process instance
        time_distribution (str): 'uniform' or 'gaussian' timestep sampling
        max_time (float): Maximum timestep for diffusion

    Returns:
        tuple: (noisy_data, timesteps, labels)
            - noisy_data: shape (batch_size, 1, d, n)
            - timesteps: shape (batch_size, 1, 1), normalized to [0, 1]
            - labels: shape (batch_size,)
    """
    data = data.squeeze(1)  # (batch_size, 1, d, n) -> (batch_size, d, n)
    batch_size, d, n = data.shape

    noisy_data_list = []
    timesteps_list = []
    expanded_labels_list = []

    for i in range(batch_size):
        # Sample a timestep
        if time_distribution == 'uniform':
            t = np.random.uniform(0, max_time)
        elif time_distribution == 'gaussian':
            t = np.clip(np.random.normal(loc=max_time / 4, scale=max_time / 2), 0, max_time)
        else:
            raise ValueError("Invalid time_distribution. Choose 'uniform' or 'gaussian'.")

        t_tensor = torch.tensor([[t]], dtype=torch.float32)

        # Add noise to sample
        noised_sample = diffusion_process.add_noise(
            data[i].unsqueeze(0).unsqueeze(0), t
        )

        noisy_data_list.append(noised_sample)
        timesteps_list.append(t_tensor)
        expanded_labels_list.append(
            labels[i].unsqueeze(0) if isinstance(labels, torch.Tensor) 
            else torch.tensor([labels[i]])
        )

    # Stack into tensors
    noisy_data = torch.cat(noisy_data_list, dim=0)  # (batch_size, 1, d, n)
    timesteps = torch.cat(timesteps_list, dim=0)  # (batch_size, 1, 1)
    expanded_labels = torch.cat(expanded_labels_list, dim=0)

    timesteps /= max_time  # Normalize to [0, 1]

    return noisy_data, timesteps, expanded_labels


def visualize_noised_data(data, diffusion_process, max_time=10, num_samples=10):
    """
    Visualize the forward diffusion process at different timesteps.
    
    Args:
        data (torch.Tensor): Clean data to visualize
        diffusion_process (DiffusionProcess): Diffusion process instance
        max_time (float): Maximum timestep
        num_samples (int): Number of timesteps to visualize
    """
    for j in range(5):
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            t = i * (max_time / (num_samples - 1))
            noised_sample = diffusion_process.add_noise(data[50 + j].unsqueeze(0), t)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(noised_sample.squeeze().numpy(), cmap="viridis")
            plt.title(f"t = {t:.2f}")
        plt.show()
