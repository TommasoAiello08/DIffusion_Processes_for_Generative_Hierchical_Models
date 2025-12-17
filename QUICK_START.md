# 🚀 Quick Start Guide

## Your Repository is Now Ready!

### What Just Happened?

Your messy thesis repo has been transformed into a **professional, modular research codebase**. Here's how to use it:

---

## 🔍 First Steps

### 1. Verify Everything Works

```bash
python setup_check.py
```

This will check:
- ✅ Python version
- ✅ Required packages
- ✅ Directory structure
- ✅ Module imports
- ✅ Basic functionality

**Expected output**: All checks passed ✅

---

### 2. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Experiments

### Quick Test Experiment

```bash
python experiments/run_experiment.py
```

This will:
1. Generate 1000 synthetic samples
2. Apply diffusion (add noise)
3. Train a Transformer denoiser
4. Save results to `outputs/small_test_[timestamp]/`

**Expected time**: 5-10 minutes (depending on hardware)

### What Gets Saved?

```
outputs/small_test_20251217_143045/
├── config.json          # Experiment configuration
├── best_model.pt        # Best model checkpoint
└── losses.csv          # Training/validation losses
```

---

## 📊 Exploring the Code

### Interactive Tutorial

```bash
jupyter notebook Notebooks/01_getting_started.ipynb
```

This notebook shows:
- How to generate hierarchical data
- How to encode sequences
- How to visualize samples
- How to prepare data for training

### Using the Modules Directly

```python
# In Python or Jupyter
import sys
sys.path.append('..')  # If in notebooks/

from src.grammar import HierarchicalGrammar
from configs.grammars import BINARY_3_LEVEL_GRAMMAR

# Generate data
grammar = HierarchicalGrammar(BINARY_3_LEVEL_GRAMMAR)
df = grammar.generate_dataset(n_samples=100)

print(df.head())
```

---

## ⚙️ Customizing Experiments

### Option 1: Modify Existing Config

Edit `configs/experiment_config.py`:

```python
SMALL_EXPERIMENT = ExperimentConfig(
    experiment_name="my_experiment",
    data=DataConfig(n_samples=5000),  # ← Change this
    training=TrainingConfig(epochs=50),  # ← And this
)
```

### Option 2: Create New Config

In `experiments/run_experiment.py`, change:

```python
def main():
    # config = SMALL_EXPERIMENT
    config = ExperimentConfig(
        experiment_name="custom_run",
        grammar_type="binary_5_level",  # Use deeper grammar
        data=DataConfig(n_samples=10000),
        training=TrainingConfig(epochs=100, batch_size=512),
    )
    # ... rest of main()
```

---

## 📁 Directory Guide

| Directory | Purpose | Gitignored? |
|-----------|---------|-------------|
| `src/` | Core source code | ❌ No |
| `experiments/` | Experiment runners | ❌ No |
| `configs/` | Configuration files | ❌ No |
| `Notebooks/` | Tutorial notebooks | ❌ No |
| `data/` | Generated datasets | ✅ Yes |
| `outputs/` | Experiment results | ✅ Yes |

---

## 🔧 Common Tasks

### Add a New Grammar

Edit `configs/grammars.py`:

```python
MY_CUSTOM_GRAMMAR = {
    'root_symbols': ['A', 'B'],
    'terminal_symbols': ['x', 'y'],
    'rules': {
        'A': [('ab', 1.0)],
        'B': [('ba', 1.0)],
        'a': [('xy', 1.0)],
        'b': [('yx', 1.0)],
    }
}
```

### Run a Custom Experiment

```bash
python experiments/run_experiment.py
```

(After modifying the config as shown above)

### Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load losses
losses = pd.read_csv('outputs/my_experiment_*/losses.csv')

# Plot
plt.plot(losses['train_loss'], label='Train')
plt.plot(losses['val_loss'], label='Val')
plt.legend()
plt.show()
```

---

## 📚 File Reference

### Core Modules

- **`src/grammar.py`** - Generate hierarchical data
- **`src/diffusion.py`** - Forward diffusion (add noise)
- **`src/models.py`** - Transformer models
- **`src/utils.py`** - Grammar constraints, utilities

### Configuration

- **`configs/grammars.py`** - Grammar definitions
- **`configs/experiment_config.py`** - Experiment settings

### Scripts

- **`experiments/run_experiment.py`** - Main experiment pipeline
- **`setup_check.py`** - Verify installation

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure you're in the repo root
cd /path/to/DIffusion_Processes_for_Generative_Hierchical_Models

# Reinstall packages
pip install -r requirements.txt
```

### Module Not Found

```python
# In notebooks, add parent to path
import sys
sys.path.append('..')
```

### CUDA Errors

Edit `configs/experiment_config.py`:

```python
@dataclass
class ExperimentConfig:
    device: str = "cpu"  # ← Change from "cuda" to "cpu"
```

---

## 🎯 Next Steps

1. ✅ Run `python setup_check.py`
2. ✅ Run `python experiments/run_experiment.py`
3. ✅ Explore `Notebooks/01_getting_started.ipynb`
4. ⏳ Customize configs for your thesis experiments
5. ⏳ Archive old notebooks to `Notebooks/old/`
6. ⏳ Push to GitHub:
   ```bash
   git add .
   git commit -m "Reorganize into modular structure"
   git push
   ```

---

## 📖 Documentation

- **README.md** - Main documentation
- **MIGRATION_GUIDE.md** - Detailed migration info
- **REORGANIZATION_SUMMARY.md** - What changed
- **This file** - Quick reference

---

## 💡 Tips

- Keep old notebooks in `Notebooks/` for reference
- Use `outputs/` for experiment results (gitignored)
- Commit code changes, not data or models
- Create new configs instead of hardcoding parameters
- Use meaningful experiment names

---

## ✨ You're All Set!

Your repository is now:
- ✅ Professional and organized
- ✅ Reproducible and documented
- ✅ Ready for thesis work
- ✅ Easy to share and collaborate

**Happy experimenting! 🎓**
