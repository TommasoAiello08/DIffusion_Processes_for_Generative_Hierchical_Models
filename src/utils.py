"""
Utility functions for grammar-constrained diffusion models.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict


def build_allowed_matrix(prod_rules, leaf_alphabet):
    """
    Build a legality matrix for leaf symbol pairs.
    
    Args:
        prod_rules (dict): Production rules mapping parent -> [(expansion, weight), ...]
        leaf_alphabet (list): List of terminal symbols (single characters)
        
    Returns:
        torch.Tensor: Boolean matrix (d×d) where allowed[i,j]=True iff
                     (leaf_alphabet[i], leaf_alphabet[j]) is a legal pair
    """
    d = len(leaf_alphabet)
    char2idx = {c: i for i, c in enumerate(leaf_alphabet)}

    allowed = torch.zeros(d, d, dtype=torch.bool)
    for parent, exp_list in prod_rules.items():
        for exp_str, _ in exp_list:
            # exp_str should be length-2, e.g. "ij", "hl", etc.
            if len(exp_str) != 2:
                continue
            c1, c2 = exp_str[0], exp_str[1]
            if c1 in char2idx and c2 in char2idx:
                allowed[char2idx[c1], char2idx[c2]] = True

    return allowed


def log_softmax_mask(logits, allowed):
    """
    Masked log-softmax that re-normalizes per position pair.
    
    Ensures that sibling pairs conform to grammar rules by masking
    illegal combinations and renormalizing probabilities.
    
    Args:
        logits (torch.Tensor): Raw logits, shape (B, 1, d, n)
        allowed (torch.Tensor): Legality matrix, shape (d, d)
        
    Returns:
        torch.Tensor: Masked log-probabilities, shape (B, 1, d, n)
    """
    B, _, d, n = logits.shape
    assert n % 2 == 0, "Tree leaves come in (left,right) pairs"

    # First, ordinary log-softmax along the symbol axis
    logp = F.log_softmax(logits, dim=2)  # (B,1,d,n)

    # Overwrite each pair (k,k+1) in place
    out = logp.clone()
    illegal = (~allowed).to(logp.device)  # (d,d)

    for k in range(0, n, 2):
        L = out[..., k]      # (B,1,d)
        R = out[..., k + 1]  # (B,1,d)

        # Joint log-prob: log p(l,r) = log p(l) + log p(r)
        joint = L.unsqueeze(-1) + R.unsqueeze(-2)  # (B,1,d,d)
        joint = joint.masked_fill(illegal, -1e9)   # Zap illegal combos

        # Renormalize joint so Σ_lr exp = 1 (numerically stable)
        joint = joint - torch.logsumexp(joint.view(B, -1), dim=1).view(B, 1, 1, 1)

        # New marginals: p(l) = Σ_r p(l,r), p(r) = Σ_l p(l,r)
        Lm = torch.logsumexp(joint, dim=-1)  # (B,1,d)
        Rm = torch.logsumexp(joint, dim=-2)  # (B,1,d)

        out[..., k] = Lm
        out[..., k + 1] = Rm

    return out


def pair_penalty(probs, allowed, p=2):
    """
    Compute penalty for illegal symbol pairs.
    
    Args:
        probs (torch.Tensor): Probability distributions, shape (B,1,d,n)
        allowed (torch.Tensor): Legality matrix, shape (d,d)
        p (int): Power for penalty (default: 2 for squared penalty)
        
    Returns:
        torch.Tensor: Scalar penalty - mean probability mass on illegal pairs
    """
    B, _, d, n = probs.shape
    num_pairs = n // 2

    illegal = (~allowed).float().to(probs.device)  # (d,d)
    pen_b = 0.0

    for k in range(0, n, 2):
        pL = probs[..., k].squeeze(1)      # (B,d)
        pR = probs[..., k + 1].squeeze(1)  # (B,d)
        joint = torch.einsum('bd,be->bde', pL, pR)  # (B,d,d)
        pen_b += (joint * illegal).pow(p).sum(dim=(1, 2))  # (B,)

    return (pen_b / num_pairs).mean()  # scalar


def onehot_to_idx(batch_onehot):
    """
    Convert one-hot encoding to indices.
    
    Args:
        batch_onehot (torch.Tensor): One-hot encoded data, shape (B,1,d,n) or (B,d,n)
        
    Returns:
        torch.Tensor: Integer indices, shape (B,n)
    """
    if batch_onehot.dim() == 4:
        batch_onehot = batch_onehot.squeeze(1)  # (B,d,n)
    return batch_onehot.argmax(dim=1)  # (B,n)


def build_inverse_map(prod_rules):
    """
    Build inverse production rule map.
    
    Args:
        prod_rules (dict): Production rules mapping parent -> [(expansion, weight), ...]
        
    Returns:
        dict: Mapping from expansion string to list of possible parent symbols
    """
    inv_map = defaultdict(list)
    for parent, exp_list in prod_rules.items():
        for exp_str, _ in exp_list:
            inv_map[exp_str].append(parent)
    return inv_map


def tensor_to_string(mat, index_to_char):
    """
    Convert one-hot tensor to string.
    
    Args:
        mat (torch.Tensor): One-hot encoded matrix
        index_to_char (dict): Mapping from indices to characters
        
    Returns:
        str: Decoded string sequence
    """
    if mat.dim() == 3:  # (1, d, n)
        mat = mat.squeeze(0)  # (d, n)
    indices = mat.argmax(dim=0)
    return ''.join([index_to_char[idx.item()] for idx in indices])


def recover_root(seq, inv_map, branching_rate=2):
    """
    Bottom-up parsing to recover root symbol from leaf sequence.
    
    Args:
        seq (str): Leaf symbol sequence
        inv_map (dict): Inverse production rule mapping
        branching_rate (int): Branching factor (default: 2)
        
    Returns:
        str: Root symbol ('1', '2', etc.) or '?' if parsing fails
    """
    current = list(seq)
    while len(current) > 1:
        next_level = []
        j = 0
        while j < len(current):
            # Try different segmentation sizes
            for k in [branching_rate, 3, 1]:
                if j + k <= len(current):
                    segment = ''.join(current[j:j + k])
                    if segment in inv_map:
                        import random
                        next_level.append(random.choice(inv_map[segment]))
                        j += k
                        break
            else:
                next_level.append('?')
                j += 1
        current = next_level
    
    return current[0] if current else '?'


def init_weights_xavier(m):
    """
    Initialize model weights using Xavier uniform initialization.
    
    Args:
        m (nn.Module): Module to initialize
    """
    import torch.nn as nn
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
