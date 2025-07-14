# %%
import pandas as pd
import numpy as np
import torch
from collections import defaultdict


class HierarchicalGrammar:
    def __init__(self, grammar_spec):
        """
        Initialize with a grammar specification.
        
        Args:
            grammar_spec (dict): Contains root_symbols, rules, and terminal_symbols
        """
        self.grammar = grammar_spec
        self._validate_grammar()
        
        # Precompute probabilities for faster sampling
        self.prob_rules = {}
        for nt, prods in self.grammar['rules'].items():
            productions, weights = zip(*prods)
            probs = np.array(weights) / sum(weights)
            self.prob_rules[nt] = (productions, probs)
    
    def _validate_grammar(self):
        """Check the grammar specification is valid."""
        required_keys = ['root_symbols', 'rules', 'terminal_symbols']
        for key in required_keys:
            if key not in self.grammar:
                raise ValueError(f"Grammar must specify {key}")
                
        for root in self.grammar['root_symbols']:
            if root not in self.grammar['rules']:
                raise ValueError(f"Root symbol {root} has no production rules")

    def generate_sample(self, max_depth=5, start_symbol=None):
        """
        completely ignore max_depth and just expand until
        you hit one of your true terminals
        """
        if start_symbol is None:
            start_symbol = np.random.choice(self.grammar['root_symbols'])
        return self._expand_symbol(start_symbol)

    def _expand_symbol(self, symbol):
        if symbol in self.grammar['terminal_symbols']:
            return {'label': symbol, 'children': []}
        productions, probs = self.prob_rules[symbol]
        choice = productions[np.random.choice(len(productions), p=probs)]
        return {'label': symbol,
                'children': [self._expand_symbol(c) for c in choice]}
    # drop max_depth completely

    
    def generate_dataset(self, n_samples=100, max_depth=5):
        """
        Generate a dataset of samples.
        
        Args:
            n_samples (int): Number of samples to generate (default: 100)
            max_depth (int): Maximum expansion depth (default: 5)
            
        Returns:
            pd.DataFrame: Contains label and sequence columns
        """
        samples = []
        for _ in range(n_samples):
            tree = self.generate_sample(max_depth)
            leaf_string = self.tree_to_string(tree)
            samples.append({'label': tree['label'], 'sequence': leaf_string})
        
        return pd.DataFrame(samples)
    
    @staticmethod
    def tree_to_string(tree):
        """Convert tree structure to leaf string."""
        if not tree['children']:
            return tree['label']
        return ''.join(HierarchicalGrammar.tree_to_string(child) 
                      for child in tree['children'])


def encode_dataset(df, chars):
    char_to_index = {c: i for i, c in enumerate(chars)}
    d = len(chars)
    n = len(df.iloc[0]['sequence'])  # leaf count

    encoded = np.zeros((len(df), d, n), dtype=np.uint8)
    for row, seq in enumerate(df['sequence']):
        for col, ch in enumerate(seq):
            encoded[row, char_to_index[ch], col] = 1
    df['sequence'] = list(encoded)
    return df



# Prepare tensors
def prepare_tensors(df):
    # Convert string labels to integers first
    labels = df['label'].astype(int).tolist()
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Stack the sequence data
    data_tensor = torch.tensor(np.stack(df['sequence'].values), 
                            dtype=torch.float32).unsqueeze(1)
    
    return data_tensor, label_tensor

def reconstruct_labels(one_hot_matrices, grammar_spec, index_to_char):
    inv_map = defaultdict(list)
    for parent, prods in grammar_spec['rules'].items():
        for prod, _ in prods:
            inv_map[prod].append(parent)
    
    batch_size = one_hot_matrices.shape[0]
    labels = []
    
    for i in range(batch_size):
        seq_indices = one_hot_matrices[i].argmax(dim=0)
        sequence = ''.join([index_to_char[idx.item()] for idx in seq_indices])
        
        current = list(sequence)
        while len(current) > 1:
            next_level = []
            j = 0
            while j < len(current):
                # Try different possible segmentations
                for k in [2, 3, 1]:  # Try 2 first, then 3, then single char
                    if j + k <= len(current):
                        segment = ''.join(current[j:j+k])
                        if segment in inv_map:
                            next_level.append(np.random.choice(inv_map[segment]))
                            j += k
                            break
                else:
                    next_level.append('?')
                    j += 1
            current = next_level
        
        root = current[0] if current else '?'
        label = 1 if root == '1' else 2 if root == '2' else -1
        labels.append(label)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # print out which indices failed (label == -1)
    fail_idxs = (labels_tensor == -1).nonzero(as_tuple=True)[0].tolist()
    if fail_idxs:
        print("failed reconstruction at indices:", fail_idxs)
    return labels_tensor
