"""
Neural network models for diffusion-based denoising.

Implements Transformer-based architectures for both classification and denoising tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding with timestep conditioning.
    
    Adds sinusoidal positional encodings to sequence embeddings and
    concatenates normalized timestep information.
    """
    
    def __init__(self, embed_dim, max_len=8):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim (int): Embedding dimension (final dim after concatenating timestep)
            max_len (int): Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding = torch.zeros(max_len, embed_dim - 1)  # Reserve 1 dim for timestep
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim - 1, 2).float() * (-math.log(10000.0) / (embed_dim - 1))
        )

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term[: (embed_dim - 1) // 2])
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, embed_dim-1)

    def forward(self, x, timesteps):
        """
        Add positional encoding and timestep information.
        
        Args:
            x (torch.Tensor): Input embeddings, shape (batch_size, seq_len, embed_dim-1)
            timesteps (torch.Tensor): Normalized timesteps, shape (batch_size, 1), values in [0, 1]
            
        Returns:
            torch.Tensor: Encoded input, shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        pe = self.encoding[:, :seq_len, :].to(x.device) * 0.5

        # Expand timesteps to match sequence length
        timesteps = timesteps.unsqueeze(1).expand(batch_size, seq_len, 1)

        return torch.cat([x + pe, timesteps], dim=-1)


class TransformerEncoder(nn.Module):
    """Single Transformer encoder block with self-attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize Transformer encoder.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass with residual connections."""
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class TransformerDecoder(nn.Module):
    """Single Transformer decoder block with self and cross-attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initialize Transformer decoder.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        """Forward pass with self-attention, cross-attention, and feed-forward."""
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        cross_attn_output, _ = self.cross_attention(x, memory, memory)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        return x


class ClassificationHead(nn.Module):
    """Classification head for sequence-level prediction."""
    
    def __init__(self, embed_dim, ff_dim, n, num_classes):
        """
        Initialize classification head.
        
        Args:
            embed_dim (int): Embedding dimension
            ff_dim (int): Hidden layer dimension
            n (int): Sequence length
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * n, ff_dim)
        self.fc2 = nn.Linear(ff_dim, num_classes)
        self.relu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Forward pass through classification layers."""
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class TransformerDenoiser_for_classification(nn.Module):
    """
    Transformer-based denoiser for classification tasks.
    
    Predicts class labels from noised hierarchical sequences.
    """
    
    def __init__(self, d, n, embed_dim=128, num_heads=4, ff_dim=256, num_classes=2):
        """
        Initialize classification denoiser.
        
        Args:
            d (int): Vocabulary size (number of symbols)
            n (int): Sequence length
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.embedding = nn.Linear(d, embed_dim - 1)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=n)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim)
        self.classifier = ClassificationHead(embed_dim, ff_dim, n, num_classes)

    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Noised input, shape (batch_size, 1, d, n)
            t (torch.Tensor): Timesteps, shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Class probabilities, shape (batch_size, num_classes)
        """
        x = x.squeeze(1).transpose(1, 2)  # (batch_size, n, d)
        x = self.embedding(x)  # (batch_size, n, embed_dim-1)
        x = self.pos_encoder(x, t)  # (batch_size, n, embed_dim)
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        return logits


class TransformerDenoiser_for_denoise(nn.Module):
    """
    Transformer-based denoiser for sequence reconstruction.
    
    Encoder-decoder architecture for predicting clean sequences from noised input.
    """
    
    def __init__(self, d, n, embed_dim=128, num_heads=8, ff_dim=512, 
                 num_encoders=3, num_decoders=3, dropout=0.1):
        """
        Initialize denoising model.
        
        Args:
            d (int): Vocabulary size
            n (int): Sequence length
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            num_encoders (int): Number of encoder layers
            num_decoders (int): Number of decoder layers
            dropout (float): Dropout rate
        """
        super().__init__()
        self.embedding = nn.Linear(d, embed_dim - 1)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=n)

        # Stack encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoders)
        ])

        # Stack decoders
        self.decoders = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoders)
        ])

        # Output projection
        self.output_layer = nn.Linear(embed_dim, d)

    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Noised input, shape (batch_size, 1, d, n)
            t (torch.Tensor): Timesteps, shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Denoised output, shape (batch_size, 1, d, n)
        """
        # Reshape and embed
        x = x.squeeze(1).transpose(1, 2)  # (batch_size, n, d)
        x = self.embedding(x)  # (batch_size, n, embed_dim-1)
        x = self.dropout(x)
        x = self.pos_encoder(x, t)  # (batch_size, n, embed_dim)

        # Encode
        for encoder in self.encoders:
            x = encoder(x)

        memory = x

        # Decode
        for decoder in self.decoders:
            x = decoder(x, memory)

        # Project to output
        output = self.output_layer(x)  # (batch_size, n, d)

        # Reshape to match input
        return output.transpose(1, 2).unsqueeze(1)  # (batch_size, 1, d, n)
