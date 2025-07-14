# %%
from RHM2 import HierarchicalGrammar, encode_dataset, prepare_tensors, reconstruct_labels
import torch

# %%
rules = {
    'root_symbols': ['1', '2'],
    'terminal_symbols': ['i', 'j', 'k', 'l'],  # Symbols that don't appear on LHS of rules
    'rules': {
        # Level 1 rules
        '1': [('aa', 1.0), ('ab', 1.0), ('ac', 1.0), ('ad', 1.0)],
        '2': [('ba', 1.0), ('bb', 1.0), ('bc', 1.0), ('bd', 1.0)],
        
        # Level 2 rules
        'a': [ ('ef', 1.0), ('eg', 1.0), ('eh', 1.0)],
        'b': [('fe', 1.0), ('fg', 1.0), ('fh', 1.0)],
        'c': [('ge', 1.0), ('gf', 1.0), ('gh', 1.0)],
        'd': [('he', 1.0), ('hf', 1.0), ('hg', 1.0)],
        
        # Level 3 rules
        'e': [('ij', 1.0), ('ik', 1.0), ('il', 1.0)],
        'f': [('ji', 1.0), ('jk', 1.0), ('jl', 1.0)],
        'g': [('ki', 1.0), ('kl', 1.0), ('kl', 1.0)],
        'h': [('li', 1.0), ('lj', 1.0), ('lk', 1.0)],
        
        # Level 4 rules
        'i': [('mm', 1.0), ('mo', 1.0), ('mp', 1.0)],
        'j': [('nm', 1.0), ('nn', 1.0), ('no', 1.0)],
        'k': [('om', 1.0), ('on', 1.0), ('op', 1.0)],
        'l': [('pm', 1.0), ('po', 1.0), ('pp', 1.0)],
        
        # Level 5 rules
        'm': [('qr', 1.0), ('qs', 1.0), ('qt', 1.0)],
        'n': [('rq', 1.0), ('rr', 1.0), ('rt', 1.0)],
        'o': [('sq', 1.0), ('ss', 1.0), ('st', 1.0)],
        'p': [('tq', 1.0), ('ts', 1.0), ('tt', 1.0)],
    }
}

# %%


# %%
# rules = {
#     'root_symbols'   : ['1','2','3','4'],  # now 4 classes
#     'terminal_symbols': ['e',"f", "g", "h"],         # still 2 leaf tokens
#     'rules': {
#         # ─── level 1 ─── choose one of four roots
#         '1': [('aa',1.0), ('ab',1.0)],
#         '2': [('ba',1.0), ('bb',1.0)],
#         '3': [('cd',1.0), ('dc',1.0)],
#         '4': [('dc',1.0), ('dd',1.0)],

#         # ─── level 2 ─── expand a/b/c/d
#         'a': [('ee',1.0), ('ef',1.0)],
#         'b': [('fe',1.0), ('ff',1.0)],
#         'c': [('gg',1.0), ('gh',1.0)],  # mixing in i/j leaves
#         'd': [('hg',1.0), ('hh',1.0)],

#         # ─── level 3 ─── terminal level: map c/d→i/j
#         'e': [('ii',1.0), ('ij',1.0)],
#         'f': [('ji',1.0), ('jj',1.0)],
#         'g': [('kk',1.0), ('kl',1.0)],
#         'h': [('lk',1.0), ('ll',1.0)],
#     }
# }


# %%
# if False:
#     x = syntetic_data(6561,rules, max_levels=max_levels, branching_rate=branching_rate, dim_vocabulary=dim_vocabulary)
#     x.to_csv("synthetic_data_250.csv", index=False)
# else:
#     x = pd.read_csv("synthetic_data_25k.csv")
# p = possible_characters(x, dim_vocabulary=dim_vocabulary)
# encode_dataset(x,possible_characters(x, dim_vocabulary=dim_vocabulary))
# data, label = prepare_tensors(x)

# %%
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# build the (d × d) legality matrix once
# ------------------------------------------------------------
def build_allowed_matrix(prod_rules, leaf_alphabet):
    """
    prod_rules   : dict mapping parent_symbol -> list of (expansion_str, weight)
    leaf_alphabet: list of single‐char terminal symbols, e.g. ['i','j','k','l']
    returns a (d×d) bool Tensor `allowed` where allowed[l,r]=True iff the pair
    (leaf_alphabet[l], leaf_alphabet[r]) is legal in your grammar.
    """
    d = len(leaf_alphabet)
    char2idx = {c: i for i, c in enumerate(leaf_alphabet)}

    allowed = torch.zeros(d, d, dtype=torch.bool)
    for parent, exp_list in prod_rules.items():
        for exp_str, _ in exp_list:
            # exp_str should be length‐2, e.g. "ij", "hl", etc.
            if len(exp_str) != 2:
                continue
            c1, c2 = exp_str[0], exp_str[1]
            if c1 in char2idx and c2 in char2idx:
                allowed[char2idx[c1], char2idx[c2]] = True

    return allowed

# ------------------------------------------------------------
# masked log-softmax that **re-normalises** per position
# ------------------------------------------------------------
def log_softmax_mask(logits, allowed):
    """
    logits  : (B, 1, d, n)        – raw logits per leaf symbol
    allowed : (d, d) bool         – legality matrix for sibling pairs

    returns : (B, 1, d, n)        – log-probs re-normalised w.r.t. the mask
    """
    B, _, d, n = logits.shape
    assert n % 2 == 0, "tree leaves come in (left,right) pairs"

    # first, ordinary log-softmax along the symbol axis
    logp = F.log_softmax(logits, dim=2)                         # (B,1,d,n)

    # we’ll overwrite each pair (k,k+1) in place
    out = logp.clone()
    illegal = (~allowed).to(logp.device)                        # (d,d)

    for k in range(0, n, 2):
        L = out[..., k  ]                                       # (B,1,d)
        R = out[..., k+1]                                       # (B,1,d)

        # joint log-prob  log p(l,r) = log p(l) + log p(r)
        joint = L.unsqueeze(-1) + R.unsqueeze(-2)               # (B,1,d,d)
        joint = joint.masked_fill(illegal, -1e9)                # zap illegal combos

        # renormalise joint so Σₗᵣ exp = 1  (numeric stable)
        joint = joint - torch.logsumexp(joint.view(B, -1), dim=1).view(B,1,1,1)

        # new marginals  p(l) = Σᵣ p(l,r) ,  p(r) = Σₗ p(l,r)
        Lm = torch.logsumexp(joint, dim=-1)                     # (B,1,d)
        Rm = torch.logsumexp(joint, dim=-2)                     # (B,1,d)

        out[..., k  ] = Lm
        out[..., k+1] = Rm

    return out



# %% [markdown]
# ## COSINE SCHEDULE TRIAL
# 

# %%
import torch
import numpy as np

class DiffusionProcess():
    def __init__(self, noise_scale=0.3, max_time=10.0, mode = "exp"):
        """
        Proper cosine-based noise schedule without oscillation.
        Args:
            noise_scale: how much gaussian noise to inject
            max_time: upper bound for timestep t
        """
        self.noise_scale = noise_scale
        self.max_time = max_time
        self.mode = mode

    def alpha_bar(self, t):
        """
        Monotonic decay from 1 → 0 using half-cosine on domain [0, π/2].
        Maps t ∈ [0, max_time] to θ ∈ [0, π/2]
        """
        t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
        theta = (t / self.max_time) * (np.pi / 2)
        return torch.cos(theta).pow(2)

    def add_noise(self, one_hot_matrix, t):
        """
        Add noise using alpha_bar(t). No oscillation.
        Args:
            one_hot_matrix: (B, 1, d, n)
            t: float or tensor (B, 1) with t ∈ [0, max_time]
        Returns:
            Noised tensor: (B, 1, d, n)
        """
        if self.mode == "exp":
            batch_size, _, d, n= one_hot_matrix.shape  # d = number of possible states, n = number of tokens

            alpha_t = np.sqrt(np.exp(-0.5 * t))
            beta_t = np.sqrt(1 - np.exp(-0.5 * t))

            noise = torch.randn_like(one_hot_matrix) * 0.2

            return alpha_t * one_hot_matrix + beta_t * noise
        
        elif self.mode == "cos":
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


# %%
def generate_diffusion_dataset(data, labels, diffusion_process, time_distribution='uniform', max_time=1):
    """
    Generates a dataset by applying the diffusion process to input data at random timesteps.

    Parameters:
    - data: Tensor of shape (batch_size, 1, d, n), the original clean data.
    - labels: Corresponding labels for each data sample.
    - diffusion_process: An instance of DiffusionProcess.
    - time_distribution: 'uniform' or 'gaussian' (defines how timesteps are sampled).
    - max_time: Maximum timestep for diffusion.

    Returns:
    - noisy_data: Tensor of shape (batch_size, 1, d, n), noised data.
    - timesteps: Tensor of shape (batch_size, 1, 1), corresponding timesteps.
    - clean_data: Tensor of shape (batch_size, 1, d, n), original clean data.
    - labels: Expanded labels matching noisy_data.
    """

    data = data.squeeze(1)  # Convert (batch_size, 1, d, n) -> (batch_size, d, n) 
    batch_size, d, n = data.shape

    noisy_data_list = []
    timesteps_list = []
    expanded_labels_list = []

    for i in range(batch_size):
        # Sample a time step
        if time_distribution == 'uniform':
            t = np.random.uniform(0, max_time)
        elif time_distribution == 'gaussian':
            t = np.clip(np.random.normal(loc=max_time / 4, scale=max_time/2), 0, max_time)
        else:
            raise ValueError("Invalid time_distribution. Choose 'uniform' or 'gaussian'.")

        t_tensor = torch.tensor([[t]], dtype=torch.float32)  # Shape (1, 1)

        noised_sample = diffusion_process.add_noise(data[i].unsqueeze(0).unsqueeze(0), t)  # (1, 1, d, n)

        # Store results
        noisy_data_list.append(noised_sample)
        timesteps_list.append(t_tensor)  # (1, 1)
        expanded_labels_list.append(labels[i].unsqueeze(0) if isinstance(labels, torch.Tensor) else torch.tensor([labels[i]]))  # Ensure tensor format

    # Stack all samples into tensors
    noisy_data = torch.cat(noisy_data_list, dim=0)  # (batch_size, 1, d, n)
    timesteps = torch.cat(timesteps_list, dim=0)  # (batch_size, 1, 1)
    expanded_labels = torch.cat(expanded_labels_list, dim=0)  # (batch_size, ...)

    timesteps /= max_time  # Normalize timesteps to [0, 1]

    return noisy_data, timesteps, expanded_labels


# %%
# Generate the dataset



# %%
import numpy as np

# if False:

#     np.save("data.npy",  data.detach().cpu().numpy())
#     np.save("noisy_data.npy",  noisy_data.detach().cpu().numpy())
#     np.save("timesteps.npy",   timesteps.detach().cpu().numpy())
#     np.save("labels.npy",      labels.detach().cpu().numpy())
# else:
#     data = torch.from_numpy(np.load("/Users/tommasoaiello/Desktop/TesiVS/2Label3Layer4Dim2Rate/data.npy"))
#     noisy_data  = torch.from_numpy(np.load("/Users/tommasoaiello/Desktop/TesiVS/2Label3Layer4Dim2Rate/noisy_data.npy"))
#     timesteps   = torch.from_numpy(np.load("/Users/tommasoaiello/Desktop/TesiVS/2Label3Layer4Dim2Rate/timesteps.npy"))
#     labels      = torch.from_numpy(np.load("/Users/tommasoaiello/Desktop/TesiVS/2Label3Layer4Dim2Rate/labels.npy"))




# %%
import matplotlib.pyplot as plt

def visualize_noised_data(data, diffusion_process, max_time=10, num_samples=10):
    
    for j in range(5):
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            t = i * (max_time / (num_samples - 1))
            noised_sample = diffusion_process.add_noise(data[50+j].unsqueeze(0), t)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(noised_sample.squeeze().numpy(), cmap="viridis")
            plt.title(f"t = {t:.2f}")
        plt.show()




# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding = torch.zeros(max_len, embed_dim - 1)  # Only for embed_dim - 1
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim - 1, 2).float() * (-math.log(10000.0) / (embed_dim - 1)))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term[: (embed_dim - 1) // 2])
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, embed_dim-1)

    def forward(self, x, timesteps):
        """
        x: Tensor of shape (batch_size, seq_len, embed_dim-1)
        timesteps: Tensor of shape (batch_size, 1), values in range [0, 1]
        """
        batch_size, seq_len, _ = x.shape
        pe = self.encoding[:, :seq_len, :].to(x.device) * 0.5  # Ensure it’s on the correct device

        timesteps = timesteps.unsqueeze(1).expand(batch_size, seq_len, 1)  # Expand to match sequence length

        return torch.cat([x + pe, timesteps], dim=-1)  # Concatenate along embedding dimension

# %%
# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Add dropout to residual connection
        x = self.norm1(x)
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)  # Add dropout to residual connection
        x = self.norm2(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, ff_dim, n, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * n, ff_dim)  # First hidden layer
        self.fc2 = nn.Linear(ff_dim, num_classes)  # Output layer
        self.relu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax for classification

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten sequence dimension
        x = self.fc1(x)  # Pass through first linear layer
        x = self.relu(x)  # Activation function
        x = self.fc2(x)  # Output layer
        return self.softmax(x)  # Apply softmax

class TransformerDenoiser_for_classification(nn.Module):
    def __init__(self, d, n, embed_dim=128, num_heads=4, ff_dim=256):
        super().__init__()
        self.embedding = nn.Linear(d, embed_dim-1)  # Encode (d) → (128 -1 = 12 !!!!!!CAREFUL ABOUT THE -1
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=n)   # Positional Encoding along `n`
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim)  
        self.decoder = TransformerDecoder(embed_dim, num_heads, ff_dim)
        self.classifier = ClassificationHead(embed_dim, ff_dim, n, 2)#add number labels)

    def forward(self, x, t):
        print(x.shape,t.shape)
        x = x.squeeze(1).transpose(1, 2)  # (batch_size, 1, d, n) → (batch_size, n, d)
        print(x.shape)
        x = self.embedding(x)  # (batch_size, n, 128 - 1 = 127)
        print(x.shape, "first embedding")
        x = self.pos_encoder(x, t)  # Add positional encoding along `n`
        print(x.shape, "after positional encoding")
        print(x[0])
        encoded = self.encoder(x)
        print(x.shape)
        logits = self.classifier(encoded)  # (batch_size, num_classes)
        return logits  # (batch_size, n, d) 

def train_for_classification(model, noisy_data, labels, timesteps, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(model, noisy_data.shape, timesteps.shape, labels.shape)

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(noisy_data, timesteps)  # (batch_size, num_classes)
    
        loss = loss_fn(predictions, labels)
        print(loss)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# %%
from torch.optim.lr_scheduler import ExponentialLR
# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Add dropout to residual connection
        x = self.norm1(x)
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)  # Add dropout to residual connection
        x = self.norm2(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # Replace ReLU with GELU
            nn.Dropout(dropout),  # Add dropout
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Add dropout to residual connection
        x = self.norm1(x)
        cross_attn_output, _ = self.cross_attention(x, memory, memory)
        x = x + self.dropout(cross_attn_output)  # Add dropout to residual connection
        x = self.norm2(x)
        ff_output = self.ffn(x)
        x = x + self.dropout(ff_output)  # Add dropout to residual connection
        x = self.norm3(x)
        return x
    
class TransformerDenoiser_for_denoise(nn.Module):
    def __init__(self, d, n, embed_dim=128, num_heads=8, ff_dim=512, num_encoders=3, num_decoders=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(d, embed_dim - 1)  # Embed input features
        self.dropout = nn.Dropout(dropout)  # Add dropout
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=n)   # Add positional encoding

        # Encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoders)
        ])

        # Decoders
        self.decoders = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoders)
        ])

        # Output layer
        self.output_layer = nn.Linear(embed_dim, d)  # Map back to original feature size

    def forward(self, x, t):
        # Reshape and embed input
        x = x.squeeze(1).transpose(1, 2)  # (batch_size, n, d)
        x = self.embedding(x)  # (batch_size, n, embed_dim - 1)
        x = self.dropout(x)  # Apply dropout
        x = self.pos_encoder(x, t)  # Add positional encoding

        # Pass through encoders
        for encoder in self.encoders:
            x = encoder(x)  # (batch_size, n, embed_dim)

        # Store encoder output as memory
        memory = x

        # Pass through decoders
        for decoder in self.decoders:
            x = decoder(x, memory)  # (batch_size, n, embed_dim)

        # Output layer
        output = self.output_layer(x)  # (batch_size, n, d)

        # Reshape output to match input shape
        return output.transpose(1, 2).unsqueeze(1)  # (batch_size, 1, d, n)

# %%
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)  # Initialize LayerNorm weights to 1
        nn.init.zeros_(m.bias)  # Initialize LayerNorm biases to 0

# %%
import torch, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

_EDGES = torch.linspace(0., 1., 11)          # [0.0, 0.1, …, 1.0]

def bin_accuracy(pred_labels, gold_labels, timesteps):


    """
    pred_labels, gold_labels : 1-D tensors (N,)  • class index per sample
    timesteps               : 1-D tensor  (N,)   • in [0,1]
    returns: tensor (10,)    • mean accuracy for each 0.1 bucket (nan if empty)
    """
    bin_id = torch.bucketize(timesteps, _EDGES, right=False) - 1   # (N,)
    acc = torch.full((10,), torch.nan)
    for b in range(10):
        mask = (bin_id == b)
        if mask.any():
            acc[b] = (pred_labels[mask] == gold_labels[mask]).float().mean()
    return acc.cpu()

# ---------- 3. pretty plot ----------
def plot_evolution(A):  
    """
    A : tensor / ndarray (E, 10) with accuracies ∈ [0,1] or nan
    """
    A = np.asarray(A)
    E, B = A.shape
    # x-coords = bucket mid-points: 0.05, 0.15, … , 0.95
    x = np.linspace(0.05, 0.95, B)

    # figure style mimicking your cosine-sim screenshot
    FIG_BG = "#f7f5f2"
    AX_BG  = "#faf9f6"
    mpl.rcParams.update({
        "axes.facecolor": AX_BG,
        "figure.facecolor": FIG_BG,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.8,
    })

    cmap   = plt.get_cmap("plasma")         # colourful like your demo
    norm   = mpl.colors.Normalize(vmin=0, vmax=E-1)
    sm     = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    for e in range(E):
        color = cmap(norm(e))
        ax.plot(x, A[e], marker='o', linewidth=2.5, color=color, label=f"epoch {e+1}")

    # aesthetics
    ax.set_xlabel("t / T  (bucket mid-point)")
    ax.set_ylabel("accuracy (non-zero labels)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.set_title("evolution of allowed-sequence accuracy over training", pad=14)
    ax.grid(axis="y")

    # spines: keep left + bottom, drop the rest
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # colour bar instead of legend (cleaner if E > 5)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("epoch", rotation=270, labelpad=12)

    plt.tight_layout()
    plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def train_unmasked(model, clean, noisy, timesteps, labels, *,
                   epochs=50, batch_size=256, test_size=0.1,
                   lr=1e-3, warmup_steps=2000, device=None):
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import torch, torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) split ---------------------------------------------------------------
    x_tr, x_te, y_tr, y_te, t_tr, t_te, lbl_tr, lbl_test = train_test_split(
        noisy, clean, timesteps, labels, test_size=test_size, random_state=42
    )
    tr_loader = DataLoader(TensorDataset(x_tr, y_tr, t_tr, lbl_tr),
                           batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te, y_te, t_te, lbl_test),
                           batch_size=batch_size)

    # 2) optimiser -----------------------------------------------------------
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = LambdaLR(opt, lambda step: (128**-0.5) *
                    min((step+1)**-0.5, (step+1)*warmup_steps**-1.5))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {"train": [], "val": []}
    acc_log = []                       # 🌈 NEW — per-epoch bucket accuracies

    model.to(device)

    for epoch in range(1, epochs+1):
        # —–––––––– training –––––––—
        model.train()
        total_loss = 0
        for x_noisy, y_clean, t, lbl in tr_loader:
            x_noisy, y_clean, t = x_noisy.to(device), y_clean.to(device), t.to(device)
            opt.zero_grad()
            logits = model(x_noisy, t)               # (B,1,d,n)
            logp   = F.log_softmax(logits, dim=2)
            tgt    = y_clean.squeeze(1).argmax(dim=1)
            loss   = loss_fn(
                logp.squeeze(1).transpose(1, 2).reshape(-1, logp.size(2)),
                tgt.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total_loss += loss.item()
        hist["train"].append(total_loss / len(tr_loader))

        # —–––––––– validation –––––––—
        model.eval()
        total_loss = 0

        # 🌈 NEW — collectors for bucket accuracy
        preds_epoch, labels_epoch, ts_epoch = [], [], []

        with torch.no_grad():
            for x_noisy, y_clean, t, lbl in te_loader:
                x_noisy, y_clean, t = x_noisy.to(device), y_clean.to(device), t.to(device)
                logits = model(x_noisy, t)
                logp   = F.log_softmax(logits, dim=2)
                tgt    = y_clean.squeeze(1).argmax(dim=1)
                loss   = loss_fn(
                    logp.squeeze(1).transpose(1, 2).reshape(-1, logp.size(2)),
                    tgt.reshape(-1)
                )
                total_loss += loss.item()

                # 🌈 grab a single label per sample for bucket-acc.
                # here we just take the VERY FIRST character (pos 0);
                # tweak if your “sequence-level” label lives elsewhere.
                pred_class  = logits[:, 0, :, 0].argmax(dim=1).cpu()  # (B,)
                gold_class  = lbl.cpu()                  # (B,)
                preds_epoch.append(pred_class)
                labels_epoch.append(gold_class)
                ts_epoch.append(t.reshape(-1).cpu())   # ← flat & future-proof

        hist["val"].append(total_loss / len(te_loader))

        # 🌈 compute & store bucket-wise accuracy for this epoch
        acc_epoch = bin_accuracy(
            torch.cat(preds_epoch),
            torch.cat(labels_epoch),
            torch.cat(ts_epoch)
        )
        acc_log.append(acc_epoch)

        print(f"epoch {epoch:02d} | train {hist['train'][-1]:.4f}"
              f" | val {hist['val'][-1]:.4f}"
              f" | acc 0-0.1 {acc_epoch[0]:.2f}")

    # —–––––––– loss curves –––––––—
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"],   label="val")
    plt.xlabel("epoch"); plt.ylabel("CE loss")
    plt.legend(); plt.grid(); plt.show()

    # —–––––––– rainbow evolution –––––––—
    A = torch.stack(acc_log)           # (E, 10)
    plot_evolution(A)          

# %%
def pair_penalty(probs, allowed, p=2): #SQUAREDDD
    """
    probs   : (B,1,d,n)   – *probabilities* (must already sum to 1 along d)
    allowed : (d,d) bool  – True for every legal (left,right) leaf pair

    returns : scalar      – mean probability mass assigned to *illegal* pairs
    """
    B, _, d, n = probs.shape

    num_pairs  = n // 2

    illegal = (~allowed).float().to(probs.device)     # (d,d)
    pen_b   = 0.0

    for k in range(0, n, 2):
        pL = probs[..., k  ].squeeze(1)               # (B,d)
        pR = probs[..., k+1].squeeze(1)               # (B,d)
        joint = torch.einsum('bd,be->bde', pL, pR)    # (B,d,d)
        pen_b += (joint * illegal).pow(p).sum(dim=(1,2))     # (B,)

    return (pen_b / num_pairs).mean()                 # scalar



def onehot_to_idx(batch_onehot):
    """
    batch_onehot : (B,1,d,n) or (B,d,n)
    returns      : (B,n) long
    """
    if batch_onehot.dim() == 4:
        batch_onehot = batch_onehot.squeeze(1)        # (B,d,n)
    return batch_onehot.argmax(dim=1)                 # (B,n)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

def masked_train_for_denoise(
    model, clean_data, noisy_data, timesteps, allowed,
    *, epochs=50, lr=3e-4, batch_size=256, test_size=0.2,
    device=None, λ_pen=1.0,
    early_patience=7,                # 🛑 new: how many epochs w/o improvement
    min_delta=1e-4,                  # 🛑 new: minimum improvement to reset counter
    save_path="best_model.pt"        # 🛑 new: where to save best weights
):
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    acc_log = []

    # split
    x_tr, x_te, y_tr_oh, y_te_oh, t_tr, t_te = train_test_split(
        noisy_data, clean_data, timesteps,
        test_size=test_size, random_state=42
    )
    y_tr = onehot_to_idx(y_tr_oh)
    y_te = onehot_to_idx(y_te_oh)

    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr, t_tr),
        batch_size=batch_size, shuffle=True
    )
    te_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te, t_te),
        batch_size=batch_size
    )

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.4)

    hist_train, hist_val, hist_illegal = [], [], []

    # 🛑 early-stopping bookkeeping
    best_val      = float("inf")
    epochs_bad    = 0

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        # =============== train ==================
        model.train()
        running_loss, running_illegal = 0.0, 0.0
        for x_noisy, y_idx, t in tr_loader:
            x_noisy, y_idx, t = x_noisy.to(device), y_idx.to(device), t.to(device)
            opt.zero_grad()

            logits  = model(x_noisy, t)                    # (B,1,d,n)
            logp_m  = log_softmax_mask(logits, allowed)
            probs_m = logp_m.exp()

            loss_ce  = F.nll_loss(
                logp_m.squeeze(1).transpose(1, 2).reshape(-1, logits.size(2)),
                y_idx.reshape(-1)
            )
            loss_pen = pair_penalty(probs_m, allowed)
            loss     = loss_ce + λ_pen * loss_pen

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            running_loss    += loss_ce.item()
            running_illegal += loss_pen.item()

        hist_train.append(running_loss / len(tr_loader))
        hist_illegal.append(running_illegal / len(tr_loader))

        # =============== validation ==================
        model.eval()
        val_loss = 0.0
        
        preds_epoch  = []
        labels_epoch = []
        ts_epoch     = []

        with torch.no_grad():
            for x_noisy, y_idx, t in te_loader:
                x_noisy, y_idx, t = x_noisy.to(device), y_idx.to(device), t.to(device)
                logits  = model(x_noisy, t)
                logp_m  = log_softmax_mask(logits, allowed)
                probs_m = logp_m.exp()

                loss_ce = F.nll_loss(
                    logp_m.squeeze(1).transpose(1, 2).reshape(-1, logits.size(2)),
                    y_idx.reshape(-1)
                )
                loss_pen = pair_penalty(probs_m, allowed)
                val_loss += (loss_ce + λ_pen * loss_pen).item()

                                # ----- grab per-sample prediction for accuracy ----------
                # 👇 adapt this one-liner to however you define "label":
                pred_class = logits[:, 0, :, 0].argmax(dim=1)   # (B,)
                gold_class = y_idx[:, 0]                        # (B,)
                preds_epoch.append(pred_class.cpu())
                labels_epoch.append(gold_class.cpu())
                ts_epoch.append(t.squeeze(1).cpu())
                # --------------------------------------------------------

        val_loss /= len(te_loader)
        hist_val.append(val_loss)
        sched.step()

        acc_epoch = bin_accuracy(
            torch.cat(preds_epoch),
            torch.cat(labels_epoch),
            torch.cat(ts_epoch)
        )
        acc_log.append(acc_epoch)            ### stash for plotting later


        tqdm.write(
            f"epoch {epoch:02d} │ "
            f"tr_ce {hist_train[-1]:.4f} │ "
            f"illeg {hist_illegal[-1]:.3e} │ "
            f"val {val_loss:.4f} │ "
            f"lr {opt.param_groups[0]['lr']:.2e}"
        )

        # =============== early-stopping ==================
        if val_loss < best_val - min_delta:          # improvement
            best_val = val_loss
            epochs_bad = 0
            torch.save(model.state_dict(), save_path)  # 🛑 keep best weights
        else:
            epochs_bad += 1
            if epochs_bad >= early_patience:
                tqdm.write(
                    f"⏹️ early stop: no val improvement ≥{min_delta} "
                    f"for {early_patience} epochs."
                )
                break

    
    # ------- (after training finishes / before the history plot) ----------
    A = torch.stack(acc_log)      # (E, 10)
    torch.save(A, "bucket_acc.pt")

    plot_evolution(A)             ### call the pretty rainbow plot

    model.load_state_dict(torch.load(save_path))

    # ---- plot history -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(hist_train, label="train CE")
    plt.plot(hist_val,   label="val CE+pen")
    plt.plot(hist_illegal, label="illegal mass (train)")
    plt.axvline(len(hist_val) - 1, color="red", ls="--", lw=1,
                label="last epoch")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()



# %%
# ---------- 1. binning utility (reuse your edges logic) ----------
edges = torch.linspace(0., 1., 11)  # cpu is fine for bookkeeping

def bin_accuracy(pred_labels, labels, timesteps):
    """return a (10,) tensor of accuracies for the fixed 0.1 buckets."""
    B = pred_labels.size(0)
    bin_id = torch.bucketize(timesteps.reshape(-1), edges, right=False) - 1  # (B,)
    acc = torch.full((10,), torch.nan)  # init with nans for empty bins
    for b in range(10):
        mask = (bin_id == b)
        if mask.any():
            acc[b] = (pred_labels[mask] == labels[mask]).float().mean()
    return acc.cpu()  # keep on cpu for later plotting


# %%
# ---------- 3. pretty plot ----------
def plot_evolution(A):
    """
    A : tensor / ndarray (E, 10) with accuracies ∈ [0,1] or nan
    """
    A = np.asarray(A)
    E, B = A.shape
    # x-coords = bucket mid-points: 0.05, 0.15, … , 0.95
    x = np.linspace(0.05, 0.95, B)

    # figure style mimicking your cosine-sim screenshot
    FIG_BG = "#f7f5f2"
    AX_BG  = "#faf9f6"
    mpl.rcParams.update({
        "axes.facecolor": AX_BG,
        "figure.facecolor": FIG_BG,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.8,
    })

    cmap   = plt.get_cmap("plasma")         # colourful like your demo
    norm   = mpl.colors.Normalize(vmin=0, vmax=E-1)
    sm     = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    for e in range(E):
        color = cmap(norm(e))
        ax.plot(x, A[e], marker='o', linewidth=2.5, color=color, label=f"epoch {e+1}")

    # aesthetics
    ax.set_xlabel("t / T  (bucket mid-point)")
    ax.set_ylabel("accuracy (non-zero labels)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.set_title("evolution of allowed-sequence accuracy over training", pad=14)
    ax.grid(axis="y")

    # spines: keep left + bottom, drop the rest
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # colour bar instead of legend (cleaner if E > 5)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("epoch", rotation=270, labelpad=12)

    plt.tight_layout()
    plt.show()

# call it


# %%
def print_gradient_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            grad_min = param.grad.abs().min().item()
            print(f"Layer: {name}")
            print(f"  Mean gradient: {grad_mean:.6f}")
            print(f"  Max gradient:  {grad_max:.6f}")
            print(f"  Min gradient:  {grad_min:.6f}")
        else:
            print(f"Layer: {name} - No gradient")

# %%
#DiffusionLM paper for reverse diffusion
#Implement diffusion, careful about proper implementation of the formula
#split losses into timestep zones (especcialy to track the early and more informative samples)
#implement the reverse diffusion
#!!!Mask elements of the dataset to make the structure not complete. Moreover one could make
#the children/synonims choosing process not uniform to add complexity
#the size of the dataset is crucial, one should stay in 5-10K out of 65K to avoid memorization
#Check if the reconstructed samples from the reverse process are valid sequences, one can do it with 
#simple deterministc dynamic programming algortihm that checks if the sequence is valid
#Moreover, measure the distance between eleemnts of the dataset and the reconstructed ones
#one can use the overlap of character. 

# %% [markdown]
# ![Screenshot 2025-03-22 alle 15.44.32.png](<attachment:Screenshot 2025-03-22 alle 15.44.32.png>)

# %%
import torch, math

def masked_reverse_diffusion(
    model,
    noisy_image,        # (b,1,d,n)
    timesteps_norm,     # (b,1) or (b,1,1)  already 0‑1
    diffusion,          # your DiffusionProcess (for alpha_bar)
    T: int = 1000       # discrete steps
):
    device = noisy_image.device

    # -------- timetable wrangling ---------------------------------
    t_norm = timesteps_norm.reshape(-1)        # -> (b,)  1‑D flat
    t_idx  = (t_norm * T).long().clamp(max=T)  # integer indices 0…T

    # -------- cosine schedule (discretised) ------------------------
    t_grid        = torch.linspace(0., diffusion.max_time, T + 1, device=device)
    alpha_bar     = diffusion.alpha_bar(t_grid)                       # ᾱ₀…ᾱ_T
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    betas = 1.0 - (alpha_bar[1:] / alpha_bar_prev[:-1])
    betas = betas.clamp(max=0.999)

    x_t = noisy_image.clone()  # (b,1,d,n)

    # -------- reverse loop -----------------------------------------
    for b, start in enumerate(t_idx):
        for j in range(start.item(), 0, -1):
            beta_j  = betas[j-1]
            alpha_j = 1.0 - beta_j
            a_bar_j   = alpha_bar[j]
            a_bar_jm1 = alpha_bar_prev[j]

            # model wants (batch,1) timestep in [0,1]
            t_in = torch.full((1, 1), j / T, device=device)


            raw_logits = F.log_softmax(model(x_t[b:b+1], t_in), dim=2)
            masked_log = log_softmax_mask(raw_logits,allowed)
            x0_hat     = masked_log.exp()  # (1,1,d,n)  p(x₀|xₜ)
            coef1 = torch.sqrt(a_bar_jm1) * beta_j / (1 - a_bar_j)
            coef2 = torch.sqrt(alpha_j)   * (1 - a_bar_jm1) / (1 - a_bar_j)
            mean  = coef1 * x0_hat + coef2 * x_t[b:b+1]

            if j > 1:
                x_t[b:b+1] = mean + torch.randn_like(mean) * torch.sqrt(beta_j)
            else:
                x_t[b:b+1] = mean

    return x_t


# %%
import torch
import torch.nn.functional as F

@torch.no_grad()
def reverse_diffusion(
    model,
    noisy_image,        # (B,1,d,n)
    timesteps_norm,     # (B,1) in [0,1]
    diffusion,          # your DiffusionProcess instance
    *, T = 1000
):
    device = noisy_image.device
    B, _, d, n = noisy_image.shape

    # build β schedule
    t_grid     = torch.linspace(0., diffusion.max_time, T+1, device=device)
    a_bar      = diffusion.alpha_bar(t_grid)
    a_bar_prev = torch.cat([torch.ones(1, device=device), a_bar[:-1]])
    betas      = (1 - a_bar[1:] / a_bar_prev[:-1]).clamp(max=0.999)

    # map continuous t→integer
    t_idx = (timesteps_norm.view(-1) * T).long().clamp_max(T)

    x_t = noisy_image.clone()
    for b, start in enumerate(t_idx):
        for j in range(start.item(), 0, -1):
            β  = betas[j-1]
            α  = 1 - β
            ā  = a_bar[j]
            ām = a_bar_prev[j]

            t_in   = torch.tensor([[j / T]], device=device)
            logits = model(x_t[b:b+1], t_in)    # (1,1,d,n)
            p0     = F.softmax(logits, dim=2)

            coef1 = torch.sqrt(ām) * β  / (1-ā)
            coef2 = torch.sqrt(α)  * (1-ām) / (1-ā)
            mean  = coef1 * p0 + coef2 * x_t[b:b+1]

            if j > 1:
                noise = torch.randn_like(mean) * torch.sqrt(β)
                x_t[b:b+1] = mean + noise
            else:
                x_t[b:b+1] = mean

    return x_t


# %%
@torch.no_grad()
def batched_reverse_diffusion(
    model,               # your denoiser
    x_t,                 # (B,1,d,n)  – starting noisy imgs
    t_norm,              # (B,1) in [0,1]
    diffusion,           # has .alpha_bar(t)
    *, T=100, amp=True
):
    """
    run all samples in *one* loop over timesteps.
    each sample only updates until its own start index.
    """
    device = x_t.device
    B = x_t.size(0)

    # ---- schedule pre-compute ------------------------------------------------
    t_grid        = torch.linspace(0., diffusion.max_time, T+1, device=device)
    a_bar         = diffusion.alpha_bar(t_grid)                    # (T+1,)
    a_bar_prev    = torch.cat([torch.ones(1, device=device), a_bar[:-1]])
    beta          = 1.0 - (a_bar[1:] / a_bar_prev[:-1])
    beta.clamp_(max=0.999)

    # map each sample’s continuous t → int idx (0..T)
    start_idx = (t_norm.view(-1) * T).long().clamp_max(T)          # (B,)

    # we keep all samples in the tensor but at each j we mask out the ones
    # that are already finished (idx==j)
    for j in range(T, 0, -1):
        active = start_idx >= j                                    # (B,) bool
        if not active.any():
            continue

        β  = beta[j-1]
        α  = 1.0 - β
        ā  = a_bar[j]
        ām = a_bar_prev[j]

        # ---------------------------------------------------------------------
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x_t[active], torch.full((active.sum(),1), j/T,
                                                   device=device))
        p0 = torch.softmax(logits, dim=2)                          # (b,1,d,n)

        coef1 = torch.sqrt(ām) * β / (1-ā)
        coef2 = torch.sqrt(α) * (1-ām) / (1-ā)
        mean  = coef1 * p0 + coef2 * x_t[active]

        if j > 1:
            noise = torch.randn_like(mean) * torch.sqrt(β)
            x_t[active] = mean + noise
        else:
            x_t[active] = mean

    return x_t



# %%
def build_inverse_map(prod_rules):
    """
    prod_rules: dict mapping parent_symbol -> list of (expansion_str, weight)
    returns: dict { (left_char, right_char) : parent_symbol }
    """
    inv = {}
    for parent, exp_list in prod_rules.items():
        for exp_str, _ in exp_list:
            # exp_str is e.g. "ij"
            if len(exp_str) != 2:
                continue
            left, right = exp_str[0], exp_str[1]
            inv[(left, right)] = parent
    return inv

# 2) tensor → string of leaf chars
def tensor_to_string(mat, index_to_char):
    """
    mat: (d, n) tensor of probs or one-hot
    index_to_char: dict { idx : char }
    """
    idxs = mat.argmax(dim=0)    # (n,)
    return ''.join(index_to_char[int(i)] for i in idxs)

# 3) collapse siblings until only the root remains
def recover_root(seq, inv_map, branching_rate=2):
    """
    seq: string of leaf chars, e.g. "ikjlfehp"
    inv_map: from build_inverse_map
    returns: root symbol as an int (1 or 2), or 0 if illegal
    """
    level = list(seq)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), branching_rate):
            pair = (level[i], level[i+1])
            parent = inv_map.get(pair)
            if parent is None:
                return 0
            nxt.append(parent)
        level = nxt
    root = level[0]
    # assume root is '1' or '2'
    return int(root)

# 4) batch wrapper
def predict_labels(denoised_batch, leaf_chars, prod_rules, branching_rate=2):
    """
    denoised_batch: (B,1,d,n) logits or probs
    leaf_chars: list of chars in exactly the order you one-hot encoded them
    prod_rules: grammar['rules'] dictionary
    """
    inv_map = build_inverse_map(prod_rules)
    # build idx→char map
    index_to_char = {i: c for i, c in enumerate(leaf_chars)}

    B, _, d, n = denoised_batch.shape
    preds = []
    for b in range(B):
        mat = denoised_batch[b,0]           # (d,n)
        seq = tensor_to_string(mat, index_to_char)
        lbl = recover_root(seq, inv_map, branching_rate)
        preds.append(lbl)
    return torch.tensor(preds, dtype=torch.long)

# %%
import torch, pandas as pd, matplotlib.pyplot as plt

# # ---------------------------------------------------------------
# # 0. tensors that belong together
# # ---------------------------------------------------------------
# B           = pred_labels.size(0)          # 1000
# t_used      = timesteps[:B].squeeze(1)     # (1000,)
# labels_used = labels[:B]                   # (1000,)

# assert t_used.numel() == labels_used.numel() == B

# # ---------------------------------------------------------------
# # 1. bucketise timesteps into 0-0.1, 0.1-0.2, … , 0.9-1.0
# # ---------------------------------------------------------------
# edges  = torch.linspace(0., 1., 11, device=t_used.device)   # 11 edges
# bin_id = torch.bucketize(t_used, edges, right=False) - 1    # (B,)

# # ---------------------------------------------------------------
# # 2. accuracy per bin
# # ---------------------------------------------------------------
# acc, count = [], []
# for b in range(10):
#     mask  = (bin_id == b)
#     cnt   = int(mask.sum())
#     count.append(cnt)
#     if cnt:
#         acc.append((pred_labels[mask] == labels_used[mask]).float().mean().item())
#     else:
#         acc.append(float('nan'))

# # ---------------------------------------------------------------
# # 3. bar-plot
# # ---------------------------------------------------------------
# def plot_paperish(df: pd.DataFrame) -> None:
#     FIG_BG = "#f7f5f2"
#     AX_BG = "#faf9f6"
#     BAR_EDGE = "black"
#     BAR_FC = "#d8d2c8"

#     fig, ax = plt.subplots(figsize=(9, 5), facecolor=FIG_BG)
#     ax.set_facecolor(AX_BG)

#     bars = ax.bar(
#         df.index, df["accuracy"], edgecolor=BAR_EDGE, linewidth=2.2, color=BAR_FC
#     )

#     ax.set_xticks(df.index)
#     ax.set_xticklabels(
#         [f"{a:.1f}-{b:.1f}" for a, b in zip(df.t_min, df.t_max)],
#         rotation=45,
#         ha="right",
#     )
#     ax.set_xlabel("timestep interval")
#     ax.set_ylabel("accuracy (non‑zero labels)")
#     ax.set_ylim(0, 1)
#     ax.set_title("Percentage of allowed sequences over time", pad=14)

#     ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

#     for spine in ["top", "right", "left"]:
#         ax.spines[spine].set_visible(False)
#     ax.spines["bottom"].set_linewidth(1.2)

#     for bar, val in zip(bars, df["accuracy"]):
#         if not np.isnan(val):
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height() + 0.02,
#                 f"{val:.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=9,
#             )

#     plt.tight_layout()
#     plt.show()
# # %%

# # %%
# # if torch isn't available in runtime, fallback to numpy for demonstration
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# try:
#     pred_labels, labels, timesteps  # noqa: F821
#     use_torch = True
# except NameError:
#     use_torch = False

# if use_torch:
#     import torch

#     B = pred_labels.size(0)
#     t_used = timesteps[:B].squeeze(1)
#     labels_used = labels[:B]
#     edges = torch.linspace(0.0, 1.0, 11, device=t_used.device)
#     bin_id = torch.bucketize(t_used, edges, right=False) - 1

#     acc, n_total = [], []
#     for b in range(10):
#         mask_bin = bin_id == b
#         mask_valid = (labels_used != 0) & (pred_labels != 0)
#         mask = mask_bin & mask_valid
#         cnt = int(mask.sum())
#         n_total.append(cnt)
#         if cnt:
#             acc.append((pred_labels[mask] == labels_used[mask]).float().mean().item())
#         else:
#             acc.append(np.nan)

# else:
#     # minimal demo data
#     np.random.seed(0)
#     B = 1000
#     timesteps = np.random.rand(B)
#     labels = np.random.randint(0, 4, size=B)
#     pred_labels = labels.copy()
#     flip_idx = np.random.choice(B, 200, replace=False)
#     pred_labels[flip_idx] = np.random.randint(1, 4, size=200)

#     bin_id = np.minimum((timesteps * 10).astype(int), 9)

#     acc, n_total = [], []
#     for b in range(10):
#         mask_bin = bin_id == b
#         mask_valid = (labels != 0) & (pred_labels != 0)
#         mask = mask_bin & mask_valid
#         cnt = int(mask.sum())
#         n_total.append(cnt)

#         if cnt:
#             acc.append(np.mean(pred_labels[mask] == labels[mask]))
#         else:
#             acc.append(np.nan)
# df = pd.DataFrame(
#     {
#         "t_min": [0.1 * i for i in range(10)],
#         "t_max": [0.1 * (i + 1) for i in range(10)],
#         "n_total": n_total,
#         "accuracy": acc,
#     }
# )




