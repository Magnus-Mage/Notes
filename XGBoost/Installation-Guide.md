# Installation and Environment Setup

## Overview

Setting up XGBoost properly is crucial for optimal performance and avoiding compatibility issues. This guide covers various installation methods, environment configurations, and troubleshooting common setup problems.

## Creating Virtual Environments

### Using Conda (Recommended)

Conda is the preferred method for managing XGBoost environments, especially when working with GPU versions or complex dependencies.

```bash
# Create a new environment with Python 3.9
conda create -n xgboost-env python=3.9
conda activate xgboost-env

# Create environment with common data science packages
conda create -n xgboost-env python=3.9 \
  numpy pandas scikit-learn matplotlib jupyter notebook

# Activate the environment
conda activate xgboost-env
```

### Using venv (Standard Python)

For pure Python environments without conda:

```bash
# Create virtual environment
python -m venv xgboost-env

# Activate environment
# On Windows:
xgboost-env\Scripts\activate
# On macOS/Linux:
source xgboost-env/bin/activate

# Upgrade pip to latest version
python -m pip install --upgrade pip
```

### Using pyenv (Version Management)

For managing multiple Python versions:

```bash
# Install specific Python version
pyenv install 3.9.16
pyenv install 3.10.11

# Set global Python version
pyenv global 3.9.16

# Create virtual environment with specific Python version
pyenv virtualenv 3.9.16 xgboost-env
pyenv activate xgboost-env
```

## Installation Methods

### Standard Installation (CPU Only)

#### Using pip
```bash
# Basic installation
pip install xgboost

# Install with additional dependencies
pip install xgboost[plotting]

# Install specific version
pip install xgboost==1.7.5

# Install from wheel (faster)
pip install --only-binary=all xgboost
```

#### Using conda
```bash
# Install from conda-forge (recommended)
conda install -c conda-forge xgboost

# Install specific version
conda install -c conda-forge xgboost=1.7.5

# Install with additional packages
conda install -c conda-forge xgboost scikit-learn pandas matplotlib
```

### GPU Installation

#### Prerequisites for GPU Support
Before installing GPU-enabled XGBoost, ensure you have:
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit (version 10.1 or later)
- cuDNN library
- Appropriate GPU drivers

#### Check CUDA Installation
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Check GPU compute capability
nvidia-ml-py3 or deviceQuery
```

#### Install GPU-enabled XGBoost
```bash
# Using pip
pip install xgboost[gpu]

# Using conda (recommended for GPU)
conda install -c conda-forge xgboost-gpu

# For specific CUDA version
conda install -c conda-forge xgboost-gpu cudatoolkit=11.2
```

### Development Installation

For contributors or users who need the latest features:

```bash
# Clone repository
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost

# Build from source (CPU)
python setup.py build
python setup.py install

# Build with GPU support
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
make -j4
cd ..
python setup.py install
```

### Installation Verification

Create a verification script to ensure everything is working:

```python
# verify_installation.py
import sys
import pkg_resources

def check_installation():
    """Comprehensive installation verification."""
    
    print("System Information:")
    print("=" * 40)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check package versions
    packages = [
        'xgboost',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'scipy'
    ]
    
    print("\nPackage Versions:")
    print("=" * 20)
    
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package:15}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package:15}: NOT INSTALLED")
    
    # Test XGBoost functionality
    print("\nXGBoost Tests:")
    print("=" * 15)
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost imported successfully")
        print(f"  Version: {xgb.__version__}")
        
        # Test build info
        build_info = xgb.build_info()
        print(f"  Build Info: {build_info}")
        
        # Test basic functionality
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        # Test CPU training
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        pred = model.predict(X[:5])
        print(f"✓ CPU training successful")
        
        # Test GPU if available
        try:
            gpu_model = xgb.XGBClassifier(
                n_estimators=10, 
                tree_method='gpu_hist',
                random_state=42
            )
            gpu_model.fit(X, y)
            print(f"✓ GPU training successful")
        except Exception as e:
            print(f"⚠ GPU training not available: {e}")
        
    except Exception as e:
        print(f"✗ XGBoost test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = check_installation()
    if success:
        print("\n🎉 Installation verification completed successfully!")
    else:
        print("\n❌ Installation verification failed!")
```

## Docker Setup

### Basic XGBoost Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["python", "main.py"]
```

### GPU-enabled Dockerfile

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install XGBoost with GPU support
RUN pip install xgboost[gpu]

# Install additional packages
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  xgboost-dev:
    build: .
    container_name: xgboost-dev
    volumes:
      - .:/app
      - ./data:/app/data
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
    environment:
      - PYTHONPATH=/app
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  xgboost-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: xgboost-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - .:/app
    command: python train_gpu.py
```

## Environment Configuration Files

### requirements.txt
```txt
# Core XGBoost requirements
xgboost>=1.7.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0

# Development
jupyter>=1.0.0
notebook>=6.0.0
ipykernel>=6.0.0

# Hyperparameter tuning
optuna>=3.0.0
hyperopt>=0.2.7
scikit-optimize>=0.9.0

# Model interpretation
shap>=0.40.0
lime>=0.2.0

# Utilities
tqdm>=4.60.0
joblib>=1.0.0
```

### environment.yml (Conda)
```yaml
# environment.yml
name: xgboost-env
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.9
  - xgboost>=1.7.0
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - matplotlib>=3.3.0
  - seaborn>=0.11.0
  - jupyter
  - notebook
  - pip
  - pip:
    - optuna>=3.0.0
    - shap>=0.40.0
    - plotly>=5.0.0
```

```bash
# Create environment from file
conda env create -f environment.yml
conda activate xgboost-env
```

## Platform-Specific Setup

### Windows Setup

```bash
# Install Microsoft C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install via conda (recommended on Windows)
conda install -c conda-forge xgboost

# Alternative: Use pre-compiled wheels
pip install --only-binary=all xgboost
```

### macOS Setup

```bash
# Install Xcode command line tools
xcode-select --install

# Install using Homebrew Python (optional)
brew install python
pip3 install xgboost

# Or use conda
conda install -c conda-forge xgboost
```

### Linux Setup

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake

# Install XGBoost
pip install xgboost
```

## Cloud Platform Setup

### Google Colab
```python
# Install in Colab
!pip install xgboost

# For GPU support
!pip install xgboost[gpu]

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### AWS SageMaker
```python
# SageMaker notebook instance
import sagemaker
import xgboost as xgb

# Use built-in XGBoost container
from sagemaker.xgboost.estimator import XGBoost

xgb_estimator = XGBoost(
    entry_point='train.py',
    framework_version='1.5-1',
    instance_type='ml.m5.large'
)
```

### Azure ML
```python
# Azure ML environment
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment(name="xgboost-env")
conda_dep = CondaDependencies()
conda_dep.add_pip_package("xgboost>=1.7.0")
env.python.conda_dependencies = conda_dep
```

## Troubleshooting Common Issues

### Import Errors
```bash
# If import fails, try:
pip uninstall xgboost
pip install xgboost --no-cache-dir

# For conda environments:
conda remove xgboost
conda install -c conda-forge xgboost
```

### GPU Issues
```bash
# Check CUDA compatibility
python -c "import xgboost as xgb; print(xgb.build_info())"

# Reinstall with proper CUDA version
pip uninstall xgboost
conda install -c conda-forge xgboost-gpu cudatoolkit=11.2
```

### Memory Issues
```python
# For large datasets, consider:
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Limit CPU threads

# Use single precision
model = xgb.XGBRegressor(tree_method='approx')
```

### Build Issues (from source)
```bash
# Ensure all dependencies
sudo apt-get install build-essential cmake ninja-build

# Clean build
rm -rf build/
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Performance Optimization Tips

### Environment Variables
```bash
# Optimize for CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Optimize for GPU
export CUDA_VISIBLE_DEVICES=0
```

### System Configuration
```python
# In Python scripts
import os
import multiprocessing

# Set optimal thread count
n_threads = min(4, multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(n_threads)
```

## Next Steps

After successful installation:

1. **Verify Installation**: Run the verification script
2. **Test Examples**: Try basic classification/regression examples
3. **Benchmark Performance**: Test on your typical dataset sizes
4. **Configure IDE**: Set up your development environment
5. **Learn Dependencies**: Understand version compatibility requirements

This completes the installation and environment setup. Next, we'll cover dependencies and version management in detail.
