"""
Example grammar configurations for Random Hierarchical Models.

These grammars define different hierarchical structures for experimentation.
"""

# Binary grammar with 2 root symbols and 5 hierarchical levels
BINARY_5_LEVEL_GRAMMAR = {
    'root_symbols': ['1', '2'],
    'terminal_symbols': ['q', 'r', 's', 't'],
    'rules': {
        # Level 1 rules
        '1': [('aa', 1.0), ('ab', 1.0), ('ac', 1.0), ('ad', 1.0)],
        '2': [('ba', 1.0), ('bb', 1.0), ('bc', 1.0), ('bd', 1.0)],
        
        # Level 2 rules
        'a': [('ef', 1.0), ('eg', 1.0), ('eh', 1.0)],
        'b': [('fe', 1.0), ('fg', 1.0), ('fh', 1.0)],
        'c': [('ge', 1.0), ('gf', 1.0), ('gh', 1.0)],
        'd': [('he', 1.0), ('hf', 1.0), ('hg', 1.0)],
        
        # Level 3 rules
        'e': [('ij', 1.0), ('ik', 1.0), ('il', 1.0)],
        'f': [('ji', 1.0), ('jk', 1.0), ('jl', 1.0)],
        'g': [('ki', 1.0), ('kj', 1.0), ('kl', 1.0)],
        'h': [('li', 1.0), ('lj', 1.0), ('lk', 1.0)],
        
        # Level 4 rules
        'i': [('mm', 1.0), ('mn', 1.0), ('mo', 1.0)],
        'j': [('nm', 1.0), ('nn', 1.0), ('no', 1.0)],
        'k': [('om', 1.0), ('on', 1.0), ('oo', 1.0)],
        'l': [('pm', 1.0), ('pn', 1.0), ('po', 1.0)],
        
        # Level 5 rules
        'm': [('qr', 1.0), ('qs', 1.0), ('qt', 1.0)],
        'n': [('rq', 1.0), ('rr', 1.0), ('rs', 1.0)],
        'o': [('sq', 1.0), ('sr', 1.0), ('st', 1.0)],
        'p': [('tq', 1.0), ('ts', 1.0), ('tt', 1.0)],
    }
}

# Simpler 3-level binary grammar
BINARY_3_LEVEL_GRAMMAR = {
    'root_symbols': ['1', '2', '3', '4'],
    'terminal_symbols': ['i', 'j', 'k', 'l'],
    'rules': {
        # Level 1 - choose one of four roots
        '1': [('aa', 1.0), ('ab', 1.0)],
        '2': [('ba', 1.0), ('bb', 1.0)],
        '3': [('cc', 1.0), ('cd', 1.0)],
        '4': [('dc', 1.0), ('dd', 1.0)],

        # Level 2 - expand a/b/c/d
        'a': [('ee', 1.0), ('ef', 1.0)],
        'b': [('fe', 1.0), ('ff', 1.0)],
        'c': [('gg', 1.0), ('gh', 1.0)],
        'd': [('hg', 1.0), ('hh', 1.0)],

        # Level 3 - terminal level
        'e': [('ii', 1.0), ('ij', 1.0)],
        'f': [('ji', 1.0), ('jj', 1.0)],
        'g': [('kk', 1.0), ('kl', 1.0)],
        'h': [('lk', 1.0), ('ll', 1.0)],
    }
}


def get_leaf_alphabet(grammar_spec):
    """Extract the leaf alphabet from a grammar specification."""
    return grammar_spec['terminal_symbols']


def get_sequence_length(grammar_spec):
    """
    Compute expected sequence length by following expansion rules.
    Assumes uniform branching.
    """
    # Start from a root symbol
    root = grammar_spec['root_symbols'][0]
    
    def count_leaves(symbol):
        if symbol in grammar_spec['terminal_symbols']:
            return 1
        # Take first production as representative
        production = grammar_spec['rules'][symbol][0][0]
        return sum(count_leaves(c) for c in production)
    
    return count_leaves(root)
