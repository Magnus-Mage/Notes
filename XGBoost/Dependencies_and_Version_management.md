# Dependencies and Version Management

## Overview

Managing dependencies correctly is crucial for XGBoost projects, especially when working across different environments, teams, or deployment scenarios. This guide covers core dependencies, version compatibility, and best practices for dependency management.

## Core Dependencies

### Required Dependencies

XGBoost has minimal core requirements, making it lightweight and easy to integrate:

```txt
# Absolutely required
numpy>=1.16.0
scipy>=1.0.0
```

### Recommended Dependencies

For most machine learning workflows, you'll want these additional packages:

```txt
# Machine Learning Ecosystem
scikit-learn>=0.22.0    # Model selection, metrics, preprocessing
pandas>=1.0.0           # Data manipulation and analysis

# Visualization
matplotlib>=3.1.0       # Basic plotting
seaborn>=0.9.0         # Statistical visualizations
plotly>=4.0.0          # Interactive plots

# Jupyter Environment
jupyter>=1.0.0         # Jupyter notebook
ipykernel>=5.0.0       # Kernel support
notebook>=6.0.0        # Notebook interface

# Development & Testing
pytest>=6.0.0          # Testing framework
black>=21.0.0          # Code formatting
flake8>=3.8.0          # Linting
```

### Optional Enhanced Dependencies

For advanced workflows and specialized use cases:

```txt
# Hyperparameter Optimization
optuna>=3.0.0          # Advanced hyperparameter tuning
hyperopt>=0.2.7        # Bayesian optimization
scikit-optimize>=0.9.0 # Gaussian process optimization
ray[tune]>=2.0.0       # Distributed hyperparameter tuning

# Model Interpretation
shap>=0.40.0           # SHAP values for interpretability
lime>=0.2.0            # Local interpretable explanations
eli5>=0.11.0           # Model interpretation utilities

# Performance & Scaling
dask>=2021.0.0         # Distributed computing
modin>=0.15.0          # Parallel pandas
joblib>=1.0.0          # Parallel processing utilities

# Model Deployment
onnx>=1.12.0           # ONNX model format
onnxruntime>=1.12.0    # ONNX inference engine
mlflow>=1.20.0         # ML lifecycle management
bentoml>=1.0.0         # Model serving framework

# Data Processing
polars>=0.14.0         # Fast DataFrame library
pyarrow>=5.0.0         # Columnar data format
featuretools>=1.0.0    # Automated feature engineering

# Monitoring & Logging
wandb>=0.12.0          # Experiment tracking
tensorboard>=2.8.0     # Visualization for experiments
```

## Version Compatibility Matrix

### XGBoost Version Compatibility

| XGBoost Version | Python Support | NumPy Support | Scikit-learn Support | Release Date |
|----------------|----------------|---------------|---------------------|--------------|
| 2.0.x          | 3.8-3.11       | ≥1.21.0       | ≥1.0.0              | 2023-12     |
| 1.7.x          | 3.7-3.11       | ≥1.16.0       | ≥0.22.0             | 2023-01     |
| 1.6.x          | 3.7-3.10       | ≥1.16.0       | ≥0.22.0             | 2022-05     |
| 1.5.x          | 3.6-3.9        | ≥1.15.0       | ≥0.22.0             | 2021-11     |
| 1.4.x          | 3.6-3.9        | ≥1.15.0       | ≥0.20.0             | 2021-05     |

### GPU Support Compatibility

| XGBoost Version | CUDA Support | cuDNN Support | Compute Capability |
|----------------|--------------|---------------|-------------------|
| 2.0.x          | 11.0-12.x    | 8.0+          | 3.5+              |
| 1.7.x          | 10.1-11.8    | 7.6+          | 3.5+              |
| 1.6.x          | 10.1-11.7    | 7.6+          | 3.5+              |

### Operating System Support

| OS               | XGBoost Version | Notes                    |
|------------------|----------------|--------------------------|
| Windows 10/11    | All versions   | Requires Visual C++      |
| macOS 10.14+     | All versions   | Intel and Apple Silicon  |
| Linux (Ubuntu)   | All versions   | Recommended platform     |
| Linux (CentOS)   | All versions   | May need dev tools       |

## Dependency Management Strategies

### 1. Using requirements.txt

Create specific requirement files for different scenarios:

#### requirements-minimal.txt
```txt
# Minimal installation for production
xgboost==1.7.5
numpy==1.21.6
scikit-learn==1.0.2
```

#### requirements-dev.txt
```txt
# Development environment
-r requirements-minimal.txt

# Development tools
jupyter==1.0.0
matplotlib==3.5.3
seaborn==0.11.2
pandas==1.4.4
plotly==5.10.0

# Testing and quality
pytest==7.1.2
black==22.6.0
flake8==5.0.4
mypy==0.971

# Hyperparameter tuning
optuna==3.0.2
```

#### requirements-full.txt
```txt
# Complete environment with all optional dependencies
-r requirements-dev.txt

# Advanced features
shap==0.41.0
lime==0.2.0.1
dask==2022.8.1
mlflow==1.28.0
onnx==1.12.0
onnxruntime==1.12.1
wandb==0.13.2
```

### 2. Using Conda Environment Files

#### environment-base.yml
```yaml
name: xgboost-base
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.9
  - xgboost=1.7.5
  - numpy=1.21.6
  - scikit-learn=1.0.2
  - pandas=1.4.4
  - matplotlib=3.5.3
  - jupyter
```

#### environment-gpu.yml
```yaml
name: xgboost-gpu
channels:
  - conda-forge
  - nvidia
  - defaults

dependencies:
  - python=3.9
  - xgboost-gpu=1.7.5
  - cudatoolkit=11.2
  - numpy=1.21.6
  - scikit-learn=1.0.2
  - pandas=1.4.4
  - matplotlib=3.5.3
  - jupyter
  
  # Additional GPU-optimized packages
  - cupy
  - cudf  # GPU-accelerated pandas
```

### 3. Using Poetry (Modern Dependency Management)

#### pyproject.toml
```toml
[tool.poetry]
name = "xgboost-project"
version = "0.1.0"
description = "XGBoost machine learning project"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
xgboost = "^1.7.0"
numpy = "^1.21.0"
scikit-learn = "^1.0.0"
pandas = "^1.4.0"
matplotlib = "^3.5.0"

# Optional dependencies
optuna = {version = "^3.0.0", optional = true}
shap = {version = "^0.40.0", optional = true}
mlflow = {version = "^1.20.0", optional = true}

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^22.0.0"
flake8 = "^5.0.0"
jupyter = "^1.0.0"

[tool.poetry.extras]
tuning = ["optuna"]
interpretation = ["shap", "lime"]
deployment = ["mlflow", "onnx", "onnxruntime"]
all = ["optuna", "shap", "lime", "mlflow", "onnx", "onnxruntime"]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## Version Pinning Strategies

### 1. Exact Pinning (Most Restrictive)
```txt
# Exact versions - guarantees reproducibility
xgboost==1.7.5
numpy==1.21.6
scikit-learn==1.0.2
pandas==1.4.4
```

### 2. Compatible Release Pinning (Recommended)
```txt
# Compatible release - allows patch updates
xgboost~=1.7.0    # Allows 1.7.x but not 1.8.x
numpy~=1.21.0     # Allows 1.21.x but not 1.22.x
scikit-learn~=1.0.0
pandas~=1.4.0
```

### 3. Minimum Version Pinning (Most Flexible)
```txt
# Minimum versions - more flexible but less predictable
xgboost>=1.7.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.4.0
```

## Dependency Resolution Tools

### 1. pip-tools for Dependency Resolution

```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in
echo "xgboost~=1.7.0" > requirements.in
echo "scikit-learn~=1.0.0" >> requirements.in
echo "pandas~=1.4.0" >> requirements.in

# Generate locked requirements.txt
pip-compile requirements.in

# Update dependencies
pip-compile --upgrade requirements.in

# Install exact versions
pip-sync requirements.txt
```

### 2. Conda-lock for Conda Environments

```bash
# Install conda-lock
conda install -c conda-forge conda-lock

# Generate lock file from environment.yml
conda-lock -f environment.yml

# Install from lock file
conda-lock install conda-lock.yml
```

## Environment Testing and Validation

### Automated Dependency Verification

```python
# dependency_checker.py
import sys
import pkg_resources
import importlib
from typing import Dict, List, Tuple, Optional
import warnings

class DependencyChecker:
    """Comprehensive dependency verification for XGBoost projects."""
    
    def __init__(self):
        self.required_packages = {
            'xgboost': '>=1.6.0',
            'numpy': '>=1.18.0',
            'scikit-learn': '>=0.22.0'
        }
        
        self.recommended_packages = {
            'pandas': '>=1.0.0',
            'matplotlib': '>=3.1.0',
            'seaborn': '>=0.9.0',
            'jupyter': '>=1.0.0'
        }
        
        self.optional_packages = {
            'optuna': '>=2.0.0',
            'shap': '>=0.35.0',
            'mlflow': '>=1.15.0',
            'onnx': '>=1.8.0'
        }
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 7:
            return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.7+)"
    
    def check_package_version(self, package: str, min_version: str) -> Tuple[bool, str, Optional[str]]:
        """Check if package meets minimum version requirement."""
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version):
                return True, f"✓ {package} {installed_version}", installed_version
            else:
                return False, f"✗ {package} {installed_version} (requires {min_version}+)", installed_version
        except pkg_resources.DistributionNotFound:
            return False, f"✗ {package} not installed", None
    
    def check_xgboost_features(self) -> Dict[str, bool]:
        """Check XGBoost specific features and capabilities."""
        features = {}
        
        try:
            import xgboost as xgb
            
            # Check GPU support
            try:
                build_info = xgb.build_info()
                features['gpu_support'] = 'USE_CUDA' in build_info and build_info.get('USE_CUDA') == 'ON'
            except:
                features['gpu_support'] = False
            
            # Test basic functionality
            try:
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=100, n_features=4, random_state=42)
                model = xgb.XGBClassifier(n_estimators=5, random_state=42)
                model.fit(X, y)
                features['basic_training'] = True
            except:
                features['basic_training'] = False
            
            # Test GPU training if available
            if features['gpu_support']:
                try:
                    gpu_model = xgb.XGBClassifier(
                        n_estimators=5, 
                        tree_method='gpu_hist',
                        random_state=42
                    )
                    gpu_model.fit(X, y)
                    features['gpu_training'] = True
                except:
                    features['gpu_training'] = False
            else:
                features['gpu_training'] = False
                
        except ImportError:
            features = {
                'gpu_support': False,
                'basic_training': False,
                'gpu_training': False
            }
        
        return features
    
    def generate_report(self) -> None:
        """Generate comprehensive dependency report."""
        print("XGBoost Dependency Check Report")
        print("=" * 50)
        
        # Python version check
        py_ok, py_msg = self.check_python_version()
        print(f"\nPython Version: {py_msg}")
        
        # Required packages
        print(f"\nRequired Packages:")
        print("-" * 20)
        all_required_ok = True
        for package, min_version in self.required_packages.items():
            ok, msg, version = self.check_package_version(package, min_version)
            print(f"  {msg}")
            if not ok:
                all_required_ok = False
        
        # Recommended packages
        print(f"\nRecommended Packages:")
        print("-" * 25)
        for package, min_version in self.recommended_packages.items():
            ok, msg, version = self.check_package_version(package, min_version)
            print(f"  {msg}")
        
        # Optional packages
        print(f"\nOptional Packages:")
        print("-" * 20)
        for package, min_version in self.optional_packages.items():
            ok, msg, version = self.check_package_version(package, min_version)
            print(f"  {msg}")
        
        # XGBoost features
        print(f"\nXGBoost Features:")
        print("-" * 20)
        features = self.check_xgboost_features()
        for feature, available in features.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature.replace('_', ' ').title()}")
        
        # Overall status
        print(f"\nOverall Status:")
        print("-" * 15)
        if py_ok and all_required_ok:
            print("🎉 Environment is ready for XGBoost development!")
        else:
            print("❌ Environment setup incomplete. Please install missing dependencies.")
            
        return py_ok and all_required_ok

def main():
    checker = DependencyChecker()
    return checker.generate_report()

if __name__ == "__main__":
    main()
```

## Conflict Resolution

### Common Dependency Conflicts

#### NumPy Version Conflicts
```bash
# Issue: Multiple packages requiring different NumPy versions
# Solution: Find compatible versions
pip install pipdeptree
pipdeptree --packages numpy

# Force specific version if needed
pip install numpy==1.21.6 --force-reinstall
```

#### Scikit-learn Compatibility Issues
```bash
# Check scikit-learn dependent packages
pipdeptree --packages scikit-learn

# Update all related packages
pip install --upgrade scikit-learn xgboost optuna
```

#### CUDA/GPU Conflicts
```bash
# Check CUDA version compatibility
nvidia-smi
nvcc --version

# Install compatible versions
conda install -c conda-forge xgboost-gpu cudatoolkit=11.2
```

### Dependency Conflict Resolution Script

```python
# resolve_conflicts.py
import pkg_resources
import subprocess
import sys
from typing import Dict, List

class ConflictResolver:
    """Resolve common XGBoost dependency conflicts."""
    
    def __init__(self):
        self.known_conflicts = {
            'numpy': {
                'xgboost': '>=1.16.0',
                'scikit-learn': '>=1.16.0',
                'pandas': '>=1.16.0'
            },
            'scipy': {
                'xgboost': '>=1.0.0',
                'scikit-learn': '>=1.0.0'
            }
        }
    
    def check_conflicts(self) -> Dict[str, List[str]]:
        """Check for version conflicts."""
        conflicts = {}
        
        for base_package, dependents in self.known_conflicts.items():
            try:
                base_version = pkg_resources.get_distribution(base_package).version
                package_conflicts = []
                
                for dependent, min_version in dependents.items():
                    try:
                        dep_dist = pkg_resources.get_distribution(dependent)
                        # Check if dependent requires higher version
                        requirements = [str(req) for req in dep_dist.requires()]
                        for req in requirements:
                            if base_package in req:
                                package_conflicts.append(f"{dependent}: {req}")
                    except pkg_resources.DistributionNotFound:
                        continue
                
                if package_conflicts:
                    conflicts[base_package] = package_conflicts
                    
            except pkg_resources.DistributionNotFound:
                continue
        
        return conflicts
    
    def suggest_resolution(self, conflicts: Dict[str, List[str]]) -> None:
        """Suggest resolution for conflicts."""
        if not conflicts:
            print("✓ No dependency conflicts detected!")
            return
        
        print("Dependency Conflicts Detected:")
        print("=" * 35)
        
        for package, issues in conflicts.items():
            print(f"\n{package.upper()} conflicts:")
            for issue in issues:
                print(f"  - {issue}")
            
            # Suggest resolution
            print(f"\nSuggested resolution for {package}:")
            if package == 'numpy':
                print("  pip install 'numpy>=1.21.0,<1.25.0' --upgrade")
            elif package == 'scipy':
                print("  pip install 'scipy>=1.7.0' --upgrade")

def main():
    resolver = ConflictResolver()
    conflicts = resolver.check_conflicts()
    resolver.suggest_resolution(conflicts)

if __name__ == "__main__":
    main()
```

## Production Deployment Considerations

### Docker Multi-stage Builds

```dockerfile
# Dockerfile.production
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Production stage
FROM python:3.9-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

CMD ["python", "main.py"]
```

### Minimal Production Requirements

```txt
# requirements-prod.txt - Minimal production dependencies
xgboost==1.7.5
numpy==1.21.6
scikit-learn==1.0.2

# Only include what's actually used in production
# Exclude development, testing, and visualization packages
```

## Dependency Security

### Security Scanning

```bash
# Install security scanner
pip install safety bandit

# Check for known vulnerabilities
safety check -r requirements.txt

# Check for security issues in code
bandit -r your_project/
```

### Keeping Dependencies Updated

```bash
# Check outdated packages
pip list --outdated

# Update specific packages
pip install --upgrade xgboost scikit-learn

# Update all packages (be careful!)
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U
```

## Best Practices Summary

### Development Environment
1. **Use virtual environments** - Always isolate project dependencies
2. **Pin major versions** - Use compatible release pins (`~=`) for stability
3. **Document dependencies** - Maintain clear requirements files
4. **Test regularly** - Run dependency checks in CI/CD

### Production Environment
1. **Exact pinning** - Use exact versions for reproducibility
2. **Minimal dependencies** - Only include what's needed
3. **Security scanning** - Regular vulnerability checks
4. **Multi-stage builds** - Optimize Docker images

### Version Management
1. **Semantic versioning** - Understand version implications
2. **Compatibility testing** - Test with different versions
3. **Gradual updates** - Update dependencies incrementally
4. **Rollback plan** - Always have a way to revert changes

This comprehensive guide to dependencies and version management ensures reliable and maintainable XGBoost projects across different environments and deployment scenarios.
