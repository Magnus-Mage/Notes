# XGBoost Complete Documentation 2025

**A comprehensive guide covering XGBoost 3.x installation, deployment, algorithm comparisons, sklearn integration, export formats, and benchmarking scripts with the latest 2025 features.**

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [What is XGBoost](#what-is-xgboost)
3. [Algorithm Comparisons](#algorithm-comparisons)
4. [Deployment with Scikit-Learn](#deployment-with-scikit-learn)
5. [Alternative Deployment Methods](#alternative-deployment-methods)
6. [Export and Serialization Formats](#export-and-serialization-formats)
7. [Benchmarking Scripts](#benchmarking-scripts)
8. [Latest Features (2025)](#latest-features-2025)
9. [Best Practices](#best-practices)

---

## Installation and Setup

### Basic Installation

```bash
# Install via pip (recommended)
pip install xgboost

# Install with optional dependencies
pip install xgboost[pandas,scikit-learn,plotting]

# Install development version
pip install git+https://github.com/dmlc/xgboost.git
```

### System Requirements

- **Python**: 3.10 or newer
- **Operating System**: Linux (glibc 2.28+), macOS, Windows
- **GPU Support**: CUDA 11.0+ (optional)

### Package Variants (2025 Update)

Starting from XGBoost 2.1.0, two variants are distributed:

- **manylinux_2_28**: For recent Linux distros with glibc 2.28+
  - Full features including GPU algorithms and federated learning
- **manylinux2014**: For older Linux distros (deprecated May 31, 2025)
  - No GPU algorithms or federated learning support

### GPU Installation

```bash
# For CUDA support
pip install xgboost[gpu]

# Verify GPU installation
python -c "import xgboost as xgb; print(xgb.XGBClassifier().get_params())"
```

### Conda Installation

```bash
conda install -c conda-forge xgboost
```

---

## What is XGBoost

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed for high efficiency, flexibility, and portability. It implements machine learning algorithms under the Gradient Boosting framework, providing parallel tree boosting (GBDT/GBM) that solves data science problems quickly and accurately.

### Key Features

- **Scalability**: Handles datasets beyond billions of examples
- **Distributed Computing**: Runs on Hadoop, SGE, MPI environments
- **GPU Acceleration**: CUDA support for faster training
- **Memory Efficiency**: External memory training capabilities
- **Model Interpretability**: Feature importance and SHAP values

---

## Algorithm Comparisons

### XGBoost vs Other Gradient Boosting Methods

| Feature | XGBoost | LightGBM | CatBoost | Scikit-Learn GBM |
|---------|---------|----------|----------|------------------|
| **Speed** | Very Fast | Fastest | Fast | Slow |
| **Memory Usage** | Efficient | Very Efficient | Moderate | High |
| **Accuracy** | Excellent | Excellent | Excellent | Good |
| **Overfitting** | Moderate Risk | High Risk | Low Risk | Low Risk |
| **Categorical Features** | Manual Encoding | Automatic | Automatic | Manual Encoding |
| **GPU Support** | Yes | Yes | Yes | No |
| **Distributed Training** | Yes | Yes | Yes | No |

### Performance Characteristics

```python
# Typical performance comparison on tabular data
Algorithm_Performance = {
    'XGBoost': {
        'training_speed': 'Fast',
        'prediction_speed': 'Fast',
        'memory_efficiency': 'High',
        'hyperparameter_sensitivity': 'Moderate',
        'best_for': 'General purpose, competitions'
    },
    'Random_Forest': {
        'training_speed': 'Fast',
        'prediction_speed': 'Moderate',
        'memory_efficiency': 'Moderate',
        'hyperparameter_sensitivity': 'Low',
        'best_for': 'Baseline models, interpretability'
    },
    'Neural_Networks': {
        'training_speed': 'Slow',
        'prediction_speed': 'Fast',
        'memory_efficiency': 'Low',
        'hyperparameter_sensitivity': 'High',
        'best_for': 'Complex patterns, large datasets'
    }
}
```

---

## Deployment with Scikit-Learn

### Basic XGBoost with Sklearn Interface

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

# Load sample data
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize XGBoost classifier with sklearn interface
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1  # Avoid thread thrashing with sklearn
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### Regression Example

```python
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

# For regression tasks
reg_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Fit and predict
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
```

### Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Grid search
grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
```

---

## Alternative Deployment Methods

### Native XGBoost API

```python
import xgboost as xgb

# Create DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# Train model
num_rounds = 100
model_native = xgb.train(
    params, 
    dtrain, 
    num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Predict
predictions = model_native.predict(dtest)
```

### Flask REST API Deployment

```python
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load pre-trained model
model = xgb.Booster()
model.load_model('xgboost_model.json')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(features)
        
        # Make prediction
        prediction = model.predict(dmatrix)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### FastAPI Deployment

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
from typing import List

app = FastAPI(title="XGBoost Prediction API")

# Load model
model = xgb.Booster()
model.load_model('xgboost_model.json')

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(features)
        
        # Make prediction
        prediction = model.predict(dmatrix)[0]
        
        return PredictionResponse(
            prediction=float(prediction > 0.5),
            probability=float(prediction)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## Export and Serialization Formats

### 1. JSON Format (Recommended)

```python
# Save model as JSON
model.save_model('xgboost_model.json')

# Load JSON model
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')

# For sklearn interface
model.save_model('sklearn_xgb_model.json')
```

### 2. Joblib Format

```python
import joblib

# Save with joblib (sklearn interface)
joblib.dump(model, 'xgboost_model.pkl')

# Load with joblib
loaded_model = joblib.load('xgboost_model.pkl')

# Compressed version
joblib.dump(model, 'xgboost_model.pkl.gz', compress=3)
```

### 3. Pickle Format

```python
import pickle

# Save with pickle
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load with pickle
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### 4. ONNX Format

```python
# Install required packages first:
# pip install onnxmltools skl2onnx

from skl2onnx import to_onnx
import numpy as np

# Convert to ONNX (sklearn interface required)
initial_types = [('float_input', np.float32, [None, X.shape[1]])]
onnx_model = to_onnx(model, initial_types=initial_types)

# Save ONNX model
with open('xgboost_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Load and use ONNX model
import onnxruntime as rt

session = rt.InferenceSession('xgboost_model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Predict with ONNX
result = session.run([output_name], {input_name: X_test.astype(np.float32)})
```

### 5. Native XGBoost Binary Format

```python
# Save in native binary format
model.save_model('xgboost_model.bin')

# Load native binary format
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.bin')
```

### Format Comparison Table

| Format | Size | Load Speed | Compatibility | Cross-platform | Use Case |
|--------|------|------------|---------------|----------------|----------|
| JSON | Medium | Fast | XGBoost only | Yes | Production, debugging |
| Joblib | Small | Very Fast | Sklearn ecosystem | Yes | Python-only deployment |
| Pickle | Small | Fast | Python-only | No | Python prototyping |
| ONNX | Medium | Fast | Multi-language | Yes | Cross-platform inference |
| Binary | Small | Very Fast | XGBoost native | Yes | High-performance serving |

---

## Benchmarking Scripts

### 1. Model Training Benchmark

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

def benchmark_classification(n_samples=10000, n_features=20, n_classes=2):
    """Benchmark classification algorithms"""
    
    # Generate dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(n_features * 0.7),
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Models to benchmark
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'train_time': train_time,
            'pred_time': pred_time,
            'accuracy': accuracy
        }
        
        print(f"  Train time: {train_time:.3f}s")
        print(f"  Pred time: {pred_time:.3f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print()
    
    return results

def benchmark_regression(n_samples=10000, n_features=20):
    """Benchmark regression algorithms"""
    
    # Generate dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            'train_time': train_time,
            'pred_time': pred_time,
            'mse': mse
        }
    
    return results

# Run benchmarks
if __name__ == "__main__":
    print("=== Classification Benchmark ===")
    class_results = benchmark_classification()
    
    print("=== Regression Benchmark ===")
    reg_results = benchmark_regression()
```

### 2. Scalability Benchmark

```python
def scalability_benchmark():
    """Test XGBoost performance across different dataset sizes"""
    
    sample_sizes = [1000, 5000, 10000, 50000, 100000]
    results = []
    
    for n_samples in sample_sizes:
        print(f"Testing with {n_samples} samples...")
        
        # Generate data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'n_samples': n_samples,
            'train_time': train_time,
            'pred_time': pred_time,
            'accuracy': accuracy
        })
    
    # Plot results
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(df_results['n_samples'], df_results['train_time'], 'b-o')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('XGBoost Training Time vs Dataset Size')
    ax1.grid(True)
    
    ax2.plot(df_results['n_samples'], df_results['accuracy'], 'r-o')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('XGBoost Accuracy vs Dataset Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df_results
```

### 3. Memory Usage Benchmark

```python
import psutil
import os

def memory_benchmark():
    """Monitor memory usage during XGBoost training"""
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate large dataset
    X, y = make_classification(
        n_samples=50000,
        n_features=100,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    memory_usage = []
    
    # Initial memory
    initial_memory = get_memory_usage()
    memory_usage.append(('Initial', initial_memory))
    
    # After data loading
    after_data = get_memory_usage()
    memory_usage.append(('After Data Loading', after_data))
    
    # During training
    model = xgb.XGBClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    after_training = get_memory_usage()
    memory_usage.append(('After Training', after_training))
    
    # During prediction
    y_pred = model.predict(X_test)
    after_prediction = get_memory_usage()
    memory_usage.append(('After Prediction', after_prediction))
    
    print("Memory Usage Benchmark:")
    for stage, memory in memory_usage:
        print(f"{stage}: {memory:.2f} MB")
    
    return memory_usage

# Run memory benchmark
memory_results = memory_benchmark()
```

### 4. GPU vs CPU Benchmark

```python
def gpu_cpu_benchmark():
    """Compare GPU vs CPU training times (requires GPU setup)"""
    
    # Generate large dataset
    X, y = make_classification(
        n_samples=100000,
        n_features=50,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # CPU Training
    print("Training on CPU...")
    cpu_model = xgb.XGBClassifier(
        n_estimators=200,
        tree_method='hist',
        random_state=42
    )
    
    start_time = time.time()
    cpu_model.fit(X_train, y_train)
    cpu_time = time.time() - start_time
    
    cpu_pred = cpu_model.predict(X_test)
    cpu_accuracy = accuracy_score(y_test, cpu_pred)
    
    results['CPU'] = {'time': cpu_time, 'accuracy': cpu_accuracy}
    
    # GPU Training (if available)
    try:
        print("Training on GPU...")
        gpu_model = xgb.XGBClassifier(
            n_estimators=200,
            tree_method='gpu_hist',
            gpu_id=0,
            random_state=42
        )
        
        start_time = time.time()
        gpu_model.fit(X_train, y_train)
        gpu_time = time.time() - start_time
        
        gpu_pred = gpu_model.predict(X_test)
        gpu_accuracy = accuracy_score(y_test, gpu_pred)
        
        results['GPU'] = {'time': gpu_time, 'accuracy': gpu_accuracy}
        
        speedup = cpu_time / gpu_time
        print(f"GPU Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"GPU training failed: {e}")
        results['GPU'] = {'time': None, 'accuracy': None}
    
    return results
```

---

## Latest Features (2025)

### New in XGBoost 3.0.0 (February 27, 2025)

#### 1. Enhanced GPU Memory Training
```python
# Utilize NVLink-C2C for GPU-based external memory training
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    max_bin=256,
    enable_categorical=True,
    # New: Handle terabytes of data with GPU external memory
    external_memory=True
)
```

#### 2. Prediction Cache Support
```python
# New prediction cache feature for improved inference speed
model = xgb.XGBClassifier(prediction_cache=True)
model.fit(X_train, y_train)

# Cached predictions for repeated inference
predictions = model.predict(X_test)  # First call - creates cache
predictions = model.predict(X_test)  # Second call - uses cache (faster)
```

#### 3. Improved Quantile Sketching
```python
# Enhanced quantile sketching for better batch processing
model = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.5,
    # Improved sketching algorithm automatically enabled
)
```

#### 4. Native DataFrame Support
```python
# Direct pandas DataFrame support with improved performance
import pandas as pd

# No need for manual conversion - direct DataFrame support
df = pd.DataFrame(X_train)
model = xgb.XGBClassifier()
model.fit(df, y_train)  # Direct DataFrame input
```

#### 5. Enhanced LambdaRank Features
```python
# New ranking parameters
ranker = xgb.XGBRanker(
    objective='rank:pairwise',
    # New parameters for ranking tasks
    lambdarank_pair_method='mean',  # Options: 'mean', 'topk'
    lambdarank_num_pair_per_sample=8,  # Control sampling
    lambdarank_max_pair_per_sample=16
)
```

### Performance Improvements

```python
# Automatic page concatenation for better GPU utilization
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    # Automatic GPU optimization - no manual tuning needed
    gpu_page_size='auto',  # New automatic setting
    max_bin=256
)
```

---

## Best Practices

### 1. Hyperparameter Tuning Strategy

```python
# Recommended hyperparameter tuning order
def tune_xgboost_parameters(X_train, y_train):
    
    # Step 1: Fix learning rate and number of estimators
    base_params = {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # Step 2: Tune max_depth and min_child_weight
    depth_params = {
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5]
    }
    
    # Step 3: Tune gamma
    gamma_params = {
        'gamma': [0, 0.1, 0.2, 0.3]
    }
    
    # Step 4: Tune subsample and colsample_bytree
    sampling_params = {
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Step 5: Tune regularization
    reg_params = {
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 10, 100]
    }
    
    # Step 6: Lower learning rate and increase estimators
    final_params = {
        'learning_rate': [0.01, 0.05],
        'n_estimators': [200, 500, 1000]
    }
    
    return {
        'base': base_params,
        'depth': depth_params,
        'gamma': gamma_params,
        'sampling': sampling_params,
        'regularization': reg_params,
        'final': final_params
    }
```

### 2. Memory Optimization

```python
# For large datasets
def optimize_memory_usage(X, y):
    
    # Use appropriate data types
    for col in X.columns:
        if X[col].dtype == 'int64':
            X[col] = X[col].astype('int32')
        elif X[col].dtype == 'float64':
            X[col] = X[col].astype('float32')
    
    # Use external memory for very large datasets
    model = xgb.XGBClassifier(
        tree_method='hist',  # Memory efficient
        max_depth=6,         # Limit depth to control memory
        subsample=0.8,       # Reduce memory by subsampling
        external_memory=True # Use external memory for huge datasets
    )
    
    return model
```

### 3. Cross-Validation Best Practices

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import numpy as np

def robust_cross_validation(X, y, task_type='classification'):
    
    if task_type == 'classification':
        # Use stratified CV for classification
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif task_type == 'time_series':
        # Use time series split for temporal data
        cv = TimeSeriesSplit(n_splits=5)
    else:
        # Standard CV for regression
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        early_stopping_rounds=10,
        eval_metric='logloss'
    )
    
    # Custom CV with early stopping
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False
        )
        
        score = model.score(X_val_cv, y_val_cv)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# Example usage
mean_score, std_score = robust_cross_validation(X, y, 'classification')
print(f"CV Score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
```

### 4. Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(model, feature_names):
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# SHAP values for better interpretability
try:
    import shap
    
    def shap_analysis(model, X_test):
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        
        # Waterfall plot for single prediction
        shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
        
        return shap_values
        
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
```

### 5. Production Deployment Checklist

```python
class XGBoostProductionModel:
    """Production-ready XGBoost model wrapper"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model safely with error handling"""
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.is_loaded = True
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
    
    def predict(self, X, return_probabilities=False):
        """Safe prediction with input validation"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Input validation
        if X.shape[1] != self.model.num_features:
            raise ValueError(f"Expected {self.model.num_features} features, got {X.shape[1]}")
        
        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # Make prediction
        predictions = self.model.predict(dmatrix)
        
        if return_probabilities:
            return predictions
        else:
            return (predictions > 0.5).astype(int)
    
    def get_model_info(self):
        """Return model metadata"""
        if not self.is_loaded:
            return {}
        
        return {
            'num_features': self.model.num_features,
            'num_boosted_rounds': self.model.num_boosted_rounds(),
            'feature_names': self.model.feature_names,
            'feature_types': self.model.feature_types
        }

# Usage example
production_model = XGBoostProductionModel('xgboost_model.json')
predictions = production_model.predict(X_test)
model_info = production_model.get_model_info()
```

---

## Advanced Use Cases

### 1. Multi-class Classification

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix

# Generate multi-class dataset
X_multi, y_multi = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=5,
    n_informative=15,
    random_state=42
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Multi-class XGBoost
multi_model = xgb.XGBClassifier(
    objective='multi:softprob',  # For probability output
    num_class=5,                 # Number of classes
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

multi_model.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_model.predict(X_test_multi)
y_pred_proba_multi = multi_model.predict_proba(X_test_multi)

print("Multi-class Classification Report:")
print(classification_report(y_test_multi, y_pred_multi))
```

### 2. Ranking with XGBoost

```python
# Example ranking dataset
def create_ranking_dataset():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create relevance scores (0-4)
    y = np.random.randint(0, 5, n_samples)
    
    # Create query groups (each query has multiple documents)
    group_sizes = np.random.randint(10, 50, 50)  # 50 queries
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
    
    return X[:len(groups)], y[:len(groups)], group_sizes

X_rank, y_rank, groups = create_ranking_dataset()

# XGBoost Ranker
ranker = xgb.XGBRanker(
    objective='rank:ndcg',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    random_state=42
)

ranker.fit(X_rank, y_rank, group=groups)
ranking_scores = ranker.predict(X_rank)
```

### 3. Time Series Forecasting

```python
def create_time_series_features(data, window_size=5):
    """Create lagged features for time series"""
    features = []
    targets = []
    
    for i in range(window_size, len(data)):
        # Use previous window_size values as features
        features.append(data[i-window_size:i])
        targets.append(data[i])
    
    return np.array(features), np.array(targets)

# Generate sample time series
np.random.seed(42)
time_series = np.cumsum(np.random.randn(1000)) + np.sin(np.arange(1000) * 0.1)

# Create features
X_ts, y_ts = create_time_series_features(time_series, window_size=10)

# Split maintaining temporal order
split_point = int(0.8 * len(X_ts))
X_train_ts, X_test_ts = X_ts[:split_point], X_ts[split_point:]
y_train_ts, y_test_ts = y_ts[:split_point], y_ts[split_point:]

# Train time series model
ts_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

ts_model.fit(X_train_ts, y_train_ts)
y_pred_ts = ts_model.predict(X_test_ts)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_ts, label='Actual', alpha=0.7)
plt.plot(y_pred_ts, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Time Series Forecasting with XGBoost')
plt.show()
```

### 4. Imbalanced Dataset Handling

```python
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

# Create imbalanced dataset
X_imb, y_imb = make_classification(
    n_samples=10000,
    n_features=20,
    n_redundant=0,
    n_informative=20,
    weights=[0.95, 0.05],  # 5% positive class
    random_state=42
)

print(f"Class distribution: {np.bincount(y_imb)}")

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)

# Method 1: Use scale_pos_weight
pos_weight = len(y_train_imb[y_train_imb == 0]) / len(y_train_imb[y_train_imb == 1])

model_weighted = xgb.XGBClassifier(
    scale_pos_weight=pos_weight,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model_weighted.fit(X_train_imb, y_train_imb)

# Method 2: Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_imb, y_train_imb)

model_smote = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model_smote.fit(X_train_smote, y_train_smote)

# Compare results
models = {
    'Weighted': model_weighted,
    'SMOTE': model_smote
}

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_imb)[:, 1]
    auc_score = roc_auc_score(y_test_imb, y_pred_proba)
    print(f"{name} AUC: {auc_score:.4f}")
```

---

## Troubleshooting Common Issues

### 1. Memory Issues

```python
def handle_memory_issues():
    """Solutions for common memory problems"""
    
    # Problem: Out of memory during training
    # Solution 1: Use external memory
    model = xgb.XGBClassifier(
        tree_method='hist',
        max_bin=256,  # Reduce number of bins
        external_memory=True
    )
    
    # Solution 2: Reduce batch size
    model = xgb.XGBClassifier(
        tree_method='hist',
        subsample=0.5,  # Use only 50% of data per tree
        max_depth=4     # Reduce tree depth
    )
    
    # Solution 3: Use streaming for large datasets
    def train_in_batches(X, y, batch_size=10000):
        model = xgb.XGBClassifier()
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            if i == 0:
                model.fit(batch_X, batch_y)
            else:
                # Incremental training (if supported)
                model.fit(batch_X, batch_y, xgb_model=model.get_booster())
        
        return model
```

### 2. Performance Issues

```python
def optimize_performance():
    """Performance optimization tips"""
    
    # Use histogram-based algorithm for speed
    fast_model = xgb.XGBClassifier(
        tree_method='hist',
        max_bin=256,
        n_jobs=-1  # Use all CPU cores
    )
    
    # GPU acceleration (if available)
    gpu_model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0
    )
    
    # Reduce precision for speed (if acceptable)
    fast_model = xgb.XGBClassifier(
        tree_method='approx',  # Approximate algorithm
        sketch_eps=0.1         # Approximation level
    )
```

### 3. Overfitting Issues

```python
def prevent_overfitting():
    """Strategies to prevent overfitting"""
    
    # Regularization parameters
    regularized_model = xgb.XGBClassifier(
        reg_alpha=10,      # L1 regularization
        reg_lambda=10,     # L2 regularization
        max_depth=4,       # Limit tree depth
        min_child_weight=5, # Minimum samples in leaf
        subsample=0.8,     # Row sampling
        colsample_bytree=0.8, # Column sampling
        early_stopping_rounds=10  # Early stopping
    )
    
    # Cross-validation for parameter tuning
    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(
        regularized_model,
        X_train, y_train,
        cv=5,
        scoring='accuracy'
    )
    
    print(f"CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

---

## Conclusion

XGBoost remains one of the most powerful and versatile machine learning algorithms available in 2025. With continuous improvements in GPU support, memory efficiency, and new features like prediction caching and enhanced ranking capabilities, it continues to be the go-to choice for many data science competitions and production systems.

### Key Takeaways:

1. **Installation**: Use the latest XGBoost 3.x with proper variant selection
2. **Deployment**: Multiple options from sklearn integration to standalone APIs
3. **Export Formats**: JSON recommended for production, ONNX for cross-platform
4. **Performance**: GPU acceleration and memory optimization crucial for large datasets
5. **Best Practices**: Proper hyperparameter tuning and cross-validation essential

### Resources for Further Learning:

- **Official Documentation**: https://xgboost.readthedocs.io/
- **GitHub Repository**: https://github.com/dmlc/xgboost
- **Tutorials**: XGBoost Python tutorials and examples
- **Community**: Stack Overflow and XGBoost discussions

For the most up-to-date information and features, always refer to the official XGBoost documentation and release notes.