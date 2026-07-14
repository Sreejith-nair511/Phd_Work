# VS Code Complete Setup Guide

Run the entire Multimodal Depression Detection Framework in VS Code with full debugging, IntelliSense, and integrated development features.

---

## Installation & Environment Setup

### 1. Install Required Tools

```bash
# Install Python 3.10 or later (verify with python --version)
# Download from: https://www.python.org/downloads/

# Install Git for Windows
# Download from: https://git-scm.com/download/win

# Install Visual Studio Code
# Download from: https://code.visualstudio.com/
```

### 2. Clone Repository

Open PowerShell and run:

```powershell
cd "C:\path\to\your\workspace"
git clone https://github.com/Sreejith-nair511/Phd_Work.git
cd Phd_Work
```

### 3. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Install Dependencies

```powershell
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Install dev tools for VS Code integration
pip install black flake8 pytest pylint autopep8
```

### 5. Install VS Code Extensions

Open VS Code and install these extensions:

1. **Python** (Microsoft)
   - Ctrl+Shift+X → search "Python" → install
   - Provides IntelliSense, debugging, linting

2. **Pylance** (Microsoft)
   - Advanced Python IntelliSense and type checking

3. **PyTorch** (official)
   - PyTorch code snippets and documentation

4. **Jupyter** (Microsoft)
   - For running notebook experiments

5. **JSON** (Microsoft)
   - YAML/JSON syntax highlighting (already built-in)

6. **GitLens** (Eric Amodio)
   - Git integration and blame information

7. **Better Comments** (Aaron Bond)
   - Highlight TODO, FIXME, etc.

---

## VS Code Workspace Configuration

### 6. Create Workspace File

Create `.vscode/settings.json` in project root:

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.pylintArgs": [
    "--disable=C0111",
    "--disable=C0103",
    "--max-line-length=100"
  ],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": [
    "--line-length=100"
  ],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "editor.rulers": [100],
  "editor.insertSpaces": true,
  "editor.tabSize": 4,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.git": true
  },
  "search.exclude": {
    "**/__pycache__": true,
    "**/.git": true,
    "**/models/checkpoints/**": true,
    "**/dataset/raw/**": true
  }
}
```

### 7. Create VS Code Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Run Experiment",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/run_experiments.py",
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": ["--exp", "C_speech_text"]
    },
    {
      "name": "Python: Debug Single Experiment",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/run_experiments.py",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": ["--exp", "A_speech_only"],
      "stopOnEntry": false,
      "showReturnValue": true
    },
    {
      "name": "Python: Test Dataset",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "console": "integratedTerminal",
      "args": [
        "tests/test_dataset.py",
        "-v",
        "--tb=short"
      ]
    },
    {
      "name": "Python: Interactive Console",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

### 8. Create VS Code Tasks Configuration

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Install Dependencies",
      "type": "shell",
      "command": "pip",
      "args": ["install", "-r", "requirements.txt"],
      "problemMatcher": []
    },
    {
      "label": "Run All Experiments",
      "type": "shell",
      "command": "python",
      "args": ["run_experiments.py"],
      "problemMatcher": [],
      "group": {
        "kind": "build",
        "isDefault": true
      }
    },
    {
      "label": "Run Experiment A (Speech Only)",
      "type": "shell",
      "command": "python",
      "args": ["run_experiments.py", "--exp", "A_speech_only"],
      "problemMatcher": []
    },
    {
      "label": "Run Experiment C (Speech + Text)",
      "type": "shell",
      "command": "python",
      "args": ["run_experiments.py", "--exp", "C_speech_text"],
      "problemMatcher": []
    },
    {
      "label": "Run Linter (pylint)",
      "type": "shell",
      "command": "pylint",
      "args": ["encoders/", "models/", "training/", "evaluation/"],
      "problemMatcher": {
        "pattern": {
          "regexp": "^(.+?):(\\d+):(\\d+): ([A-Z][0-9]+) (.+)$",
          "file": 1,
          "line": 2,
          "column": 3,
          "code": 4,
          "message": 5
        }
      }
    },
    {
      "label": "Format Code (black)",
      "type": "shell",
      "command": "black",
      "args": [
        ".",
        "--line-length=100",
        "--exclude=venv"
      ],
      "problemMatcher": []
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest",
      "args": [
        "tests/",
        "-v",
        "--tb=short"
      ],
      "problemMatcher": []
    }
  ]
}
```

---

## Running Experiments

### Quick Start (No Debugging)

**Method 1: VS Code Terminal**

Open VS Code integrated terminal (Ctrl+`):

```powershell
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Run all experiments
python run_experiments.py

# Run single experiment
python run_experiments.py --exp A_speech_only
python run_experiments.py --exp C_speech_text
python run_experiments.py --exp F_all_modalities
```

**Method 2: VS Code Tasks**

1. Press Ctrl+Shift+B (Run Build Task)
2. Select from dropdown:
   - "Run All Experiments"
   - "Run Experiment A (Speech Only)"
   - "Run Experiment C (Speech + Text)"

### Debug Mode

1. Set breakpoint by clicking line number
2. Press F5 or select Debug configuration from dropdown
3. Choose "Python: Debug Single Experiment"
4. Step through code with:
   - F10: Step over
   - F11: Step into
   - Shift+F11: Step out
   - F5: Continue

### Create Test Script

Create `test_quick.py` in workspace root:

```python
#!/usr/bin/env python
"""Quick test script for VS Code debugging."""
import sys
sys.path.insert(0, ".")

from utils.config import load_config
from models.multimodal_model import MultimodalDepressionModel
from dataset.dataset_factory import build_dataloaders
import torch

# Load config
config = load_config("configs/experiments/speech_text.yaml")

# Build model
model = MultimodalDepressionModel(config)
print(f"Model parameters: {model.count_parameters()}")

# Build dataloaders
train_loader, val_loader, test_loader = build_dataloaders(config)
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Single forward pass
batch = next(iter(train_loader))
batch_inputs, labels = batch
print(f"Batch inputs keys: {batch_inputs.keys()}")
print(f"Labels shape: {labels.shape}")

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Forward pass
with torch.no_grad():
    logits, embeddings = model(batch_inputs)
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings keys: {embeddings.keys()}")

print("Test passed!")
```

Run with: Press F5 → choose "Python: Interactive Console" → it runs `test_quick.py`

---

## Key Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open terminal | Ctrl + ` |
| Run/Debug | F5 |
| Step over | F10 |
| Step into | F11 |
| Continue | F5 |
| Stop debugging | Shift + F5 |
| Run task | Ctrl + Shift + B |
| Format document | Shift + Alt + F |
| Go to definition | F12 |
| Find references | Shift + F12 |
| Quick fix | Ctrl + . |
| Command palette | Ctrl + Shift + P |
| Search in folder | Ctrl + Shift + F |

---

## Useful Commands in Command Palette (Ctrl+Shift+P)

```
Python: Select Interpreter
  → Choose your venv from the list
  
Python: Run Python File In Terminal
  → Quickly run current file
  
Python: Debug Python File
  → Run with debugger
  
Test: Run All Tests
  → Run pytest
  
Format Document
  → Auto-format with black
  
Sort Imports
  → Auto-organize imports
```

---

## Monitoring & Outputs

### View Training Logs

Files created during training:

```
runs/
  └─ exp_name.log          # Training log file
visualization/outputs/
  ├─ exp_name/
  │  ├─ metrics.json
  │  ├─ classification_report.txt
  │  ├─ confusion_matrix.png
  │  ├─ training_curves.png
  │  ├─ roc_curve.png
  │  └─ tsne.png
```

### View in VS Code

1. Open integrated terminal (Ctrl+`)
2. View live logs:
   ```powershell
   Get-Content -Path "runs/exp_name.log" -Wait
   ```

3. Open image files:
   - Double-click `.png` files to view in VS Code

4. Open JSON metrics:
   - Double-click `.json` files for formatted view

---

## Debugging Common Issues

### Issue 1: Python interpreter not found

```powershell
# Check Python path
python --version

# If command not found, add Python to PATH:
# Control Panel → System → Environment Variables → 
# Edit PATH → Add C:\Users\YourUser\AppData\Local\Programs\Python\Python310
```

### Issue 2: Virtual environment not activating

```powershell
# Check if venv exists
ls venv

# If not, create it:
python -m venv venv

# Activate (not cd, use dot-source):
. .\venv\Scripts\Activate.ps1

# Verify activation (should show venv in prompt):
# (venv) PS C:\path\to\project>
```

### Issue 3: CUDA/GPU not available

```python
# In Python console:
import torch
print(torch.cuda.is_available())       # Should be True if GPU available
print(torch.cuda.get_device_name(0))   # GPU name
```

If False, edit config to use CPU:

```yaml
# configs/base_config.yaml
project:
  device: "cpu"
```

### Issue 4: Import errors

```powershell
# Ensure PYTHONPATH is set correctly
$env:PYTHONPATH = "C:\path\to\Phd_Work"

# Verify imports work
python -c "from models.multimodal_model import MultimodalDepressionModel; print('OK')"
```

### Issue 5: Module not found (transformers, torch, etc.)

```powershell
# Reinstall requirements
pip install --upgrade --force-reinstall -r requirements.txt

# Or install individually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.35.0
```

---

## Performance Optimization

### Enable GPU Acceleration

```yaml
# configs/base_config.yaml
project:
  device: "cuda"
  mixed_precision: true

training:
  batch_size: 32  # Increase if GPU memory allows
```

### Reduce Memory Usage

```yaml
training:
  batch_size: 8
encoders:
  embedding_dim: 128
```

### Profile Training Speed

Add to `run_experiments.py`:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your training code here
run_experiment(config_path)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Useful Notebooks for Exploration

Create `notebooks/exploration.ipynb`:

```python
# Cell 1: Setup
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '/path/to/Phd_Work')

from utils.config import load_config
from models.multimodal_model import MultimodalDepressionModel
from dataset.dataset_factory import build_dataloaders
import torch
import matplotlib.pyplot as plt

# Cell 2: Load config and model
config = load_config("configs/experiments/speech_text.yaml")
model = MultimodalDepressionModel(config)
print(model.count_parameters())

# Cell 3: Check dataset
train_loader, _, _ = build_dataloaders(config)
batch = next(iter(train_loader))
print(f"Batch shape: {batch[0]['speech'].shape}")

# Cell 4: Forward pass
logits, embeddings = model(batch[0])
print(f"Output shape: {logits.shape}")
```

---

## CI/CD with GitHub Actions (Optional)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## Summary

**You now have:**

1. Full Python environment in VS Code
2. Debugging capabilities with breakpoints
3. Task runners for experiments
4. IntelliSense and code completion
5. Integrated terminal for live logs
6. Configuration for all experiments A-F
7. Performance profiling tools

**To start working:**

```powershell
# Terminal 1: Activate and run
.\venv\Scripts\Activate.ps1
python run_experiments.py --exp C_speech_text

# Terminal 2: Monitor logs
Get-Content "runs/*.log" -Wait
```

**Expected output:**

```
Starting training: exp_C_speech_text | Epochs=50 | Device=cuda
Epoch [001/050] train_loss=0.6234 train_acc=0.6320 val_loss=0.5821 val_acc=0.7140 ...
...
Training complete.
All outputs saved to visualization/outputs/exp_C_speech_text/
```

Visualize results in VS Code by opening PNG files from `visualization/outputs/`.
