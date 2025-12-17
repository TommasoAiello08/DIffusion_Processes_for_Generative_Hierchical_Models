#!/usr/bin/env python3
"""
Setup verification script for the reorganized repository.

Run this to verify that everything is properly set up.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_imports():
    """Check if all required packages can be imported."""
    print("\nChecking imports...")
    required = [
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "tqdm",
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found - run: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def check_src_modules():
    """Check if src modules can be imported."""
    print("\nChecking src modules...")
    modules = [
        "src.grammar",
        "src.diffusion",
        "src.models",
        "src.utils",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            all_ok = False
    
    return all_ok

def check_directory_structure():
    """Check if all required directories exist."""
    print("\nChecking directory structure...")
    required_dirs = [
        "src",
        "experiments",
        "configs",
        "notebooks",
        "data",
        "outputs",
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ not found")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    required_files = [
        "requirements.txt",
        ".gitignore",
        "README.md",
        "src/__init__.py",
        "src/grammar.py",
        "src/diffusion.py",
        "src/models.py",
        "src/utils.py",
        "experiments/run_experiment.py",
        "configs/grammars.py",
        "configs/experiment_config.py",
    ]
    
    all_ok = True
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} not found")
            all_ok = False
    
    return all_ok

def quick_functionality_test():
    """Quick test of basic functionality."""
    print("\nRunning quick functionality tests...")
    
    try:
        from src.grammar import HierarchicalGrammar
        from configs.grammars import BINARY_3_LEVEL_GRAMMAR
        
        print("  Testing grammar generation...", end=" ")
        grammar = HierarchicalGrammar(BINARY_3_LEVEL_GRAMMAR)
        df = grammar.generate_dataset(n_samples=10)
        assert len(df) == 10
        print("✅")
        
        print("  Testing data encoding...", end=" ")
        from src.grammar import encode_dataset, prepare_tensors
        from configs.grammars import get_leaf_alphabet
        
        leaf_alphabet = get_leaf_alphabet(BINARY_3_LEVEL_GRAMMAR)
        df_encoded = encode_dataset(df, leaf_alphabet)
        data_tensor, label_tensor = prepare_tensors(df_encoded)
        assert data_tensor.shape[0] == 10
        print("✅")
        
        print("  Testing diffusion...", end=" ")
        from src.diffusion import DiffusionProcess
        
        diffusion = DiffusionProcess(mode="cos")
        noisy = diffusion.add_noise(data_tensor[:1], t=5.0)
        assert noisy.shape == data_tensor[:1].shape
        print("✅")
        
        print("  Testing model initialization...", end=" ")
        from src.models import TransformerDenoiser_for_denoise
        
        _, _, d, n = data_tensor.shape
        model = TransformerDenoiser_for_denoise(d=d, n=n, embed_dim=64, num_heads=4)
        print("✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 60)
    print("REPOSITORY SETUP VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_files),
        ("Source Modules", check_src_modules),
        ("Functionality", quick_functionality_test),
    ]
    
    results = []
    for name, check_func in checks:
        results.append(check_func())
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ All checks passed!")
        print("\nYou're ready to run experiments:")
        print("  python experiments/run_experiment.py")
        print("\nOr explore the notebooks:")
        print("  jupyter notebook notebooks/01_getting_started.ipynb")
        return 0
    else:
        print("❌ Some checks failed")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Ensure you're in the repository root directory")
        print("3. Check Python version (3.9+ required)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
