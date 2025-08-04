# Installation & Environment Setup

## System Requirements

- Python 3.8 or higher
- 64-bit operating system
- At least 4GB RAM (8GB+ recommended)
- For GPU: NVIDIA GPU with CUDA 11.8+ or 12.x

## Environment Setup

### Using Conda (Recommended)

```bash
# Create new environment
conda create -n xgboost-env python=3.11
conda activate xgboost-env

# Install XGBoost with all dependencies
conda install -c conda-forge xgboost=2.1.0
```

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv xgboost-env

# Activate (Linux/Mac)
source xgboost-env/bin/activate

# Activate (Windows)
xgboost-env\Scripts\activate

# Install XGBoost
pip install xgboost==2.1.0
```

## Installation Options

### CPU-Only Installation

```bash
pip install xgboost==2.1.0
```

### GPU Installation (CUDA)

```bash
# For CUDA 12.x
pip install xgboost[gpu]==2.1.0

# Or using conda
conda install -c conda-forge xgboost-gpu=2.1.0
```

### Development Installation

```bash
# From source (latest features)
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
python setup.py install
```

## Core Dependencies

### Required Dependencies

```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.5.0
pip install pandas>=1.5.0
```

### Optional Dependencies

```bash
# For plotting
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0

# For data processing
pip install pandas>=2.0.0

# For model export
pip install joblib>=1.3.0
pip install onnx>=1.14.0
pip install onnxmltools>=1.11.0

# For hyperparameter tuning
pip install optuna>=3.3.0
pip install scikit-optimize>=0.9.0
```

## Complete Environment Setup

### requirements.txt

```text
xgboost==2.1.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.5.0
pandas>=2.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.3.0
jupyter>=1.0.0
optuna>=3.3.0
onnx>=1.14.0
onnxmltools>=1.11.0
```

### One-command Installation

```bash
pip install -r requirements.txt
```

## Verification

### Test CPU Installation

```python
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Test basic functionality
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
print("CPU installation successful!")
```

### Test GPU Installation

```python
import xgboost as xgb

# Check GPU availability
print(f"XGBoost built with GPU support: {xgb.build_info()}")

# Test GPU training
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
model = xgb.train(params, dtrain, num_boost_round=10)
print("GPU installation successful!")
```

## Version Management

### Check Current Version

```python
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
print(f"Build info: {xgb.build_info()}")
```

### Upgrade XGBoost

```bash
# Upgrade to latest version
pip install --upgrade xgboost

# Upgrade to specific version
pip install xgboost==2.1.0 --upgrade
```

## Docker Setup

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "your_script.py"]
```

### Docker Compose (with GPU)

```yaml
version: '3.8'
services:
  xgboost-app:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
```

## Next Steps

Once installation is complete, proceed to [XGBoost Basics](02-xgboost-basics.md) to understand the fundamentals.