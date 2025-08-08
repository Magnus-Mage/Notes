# XGBoost Programming Basics

Now that you understand XGBoost's mathematical foundation, let's dive into the practical programming aspects. This guide covers the essential APIs, interfaces, and programming patterns you'll need to effectively use XGBoost.

## XGBoost APIs Overview

XGBoost provides multiple interfaces to accommodate different programming preferences and use cases:

### 1. Scikit-learn Interface (Recommended for Beginners)
```python
from xgboost import XGBRegressor, XGBClassifier, XGBRanker

# Most familiar interface for sklearn users
model = XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. Native XGBoost API (More Control)
```python
import xgboost as xgb

# Direct control over data structures and parameters
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror'}
model = xgb.train(params, dtrain, num_boost_round=100)
```

### 3. Dask Interface (Distributed Computing)
```python
import xgboost as xgb
from dask.distributed import Client

# Distributed training on clusters
client = Client('scheduler-address:8786')
dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
model = xgb.dask.train(client, params, dtrain)
```

## Core Data Structures

### DMatrix - XGBoost's Internal Data Structure

```python
import xgboost as xgb
import numpy as np
import pandas as pd

# Creating DMatrix from different sources
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

# From NumPy arrays
dtrain = xgb.DMatrix(X, label=y)

# From pandas DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
dtrain = xgb.DMatrix(df, label=y)

# From sparse matrix
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)
dtrain = xgb.DMatrix(X_sparse, label=y)

# From file (libsvm format)
# dtrain = xgb.DMatrix('train.libsvm')

print(f"DMatrix shape: {dtrain.num_row()} x {dtrain.num_col()}")
print(f"Feature names: {dtrain.feature_names}")
print(f"Feature types: {dtrain.feature_types}")
```

### DMatrix Advanced Features

```python
# Setting feature names and types
feature_names = [f'feature_{i}' for i in range(10)]
feature_types = ['float'] * 10  # or 'int', 'categorical'

dtrain.feature_names = feature_names
dtrain.feature_types = feature_types

# Adding weights (for weighted learning)
weights = np.random.uniform(0.5, 2.0, 1000)
dtrain.set_weight(weights)

# Adding base margin (for continuing training)
base_margin = np.random.randn(1000)
dtrain.set_base_margin(base_margin)

# Group information (for ranking tasks)
groups = np.array([100, 200, 300, 400])  # Query group sizes
dtrain.set_group(groups)

# Accessing DMatrix properties
print(f"Number of rows: {dtrain.num_row()}")
print(f"Number of columns: {dtrain.num_col()}")
print(f"Labels: {dtrain.get_label()[:5]}")  # First 5 labels
print(f"Weights: {dtrain.get_weight()[:5]}")  # First 5 weights
```

## Model Interfaces

### Scikit-learn Style Interface

```python
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Regression example
X, y = np.random.randn(1000, 10), np.random.randn(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize regressor
regressor = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Fit model
regressor.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=False
)

# Make predictions
y_pred = regressor.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.4f}")

# Classification example
from sklearn.datasets import make_classification

X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_classes=3)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2)

classifier = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

classifier.fit(X_train_cls, y_train_cls)
y_pred_cls = classifier.predict(X_test_cls)
y_pred_proba = classifier.predict_proba(X_test_cls)

accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"Accuracy: {accuracy:.4f}")
```

### Native XGBoost Interface

```python
# Native interface provides more control
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Make predictions
predictions = model.predict(dtest)
print(f"Native API RMSE: {mean_squared_error(y_test, predictions, squared=False):.4f}")

# Save and load model
model.save_model('native_model.json')
loaded_model = xgb.Booster()
loaded_model.load_model('native_model.json')
```

## Parameter Configuration

### Core Parameters

```python
# Tree Parameters
tree_params = {
    'max_depth': 6,                    # Maximum tree depth
    'min_child_weight': 1,             # Minimum sum of weights in child
    'gamma': 0,                        # Minimum loss reduction for split
    'subsample': 1.0,                  # Fraction of samples per tree
    'colsample_bytree': 1.0,           # Fraction of features per tree
    'colsample_bylevel': 1.0,          # Fraction of features per level
    'colsample_bynode': 1.0,           # Fraction of features per node
}

# Boosting Parameters
boosting_params = {
    'learning_rate': 0.1,              # Step size shrinkage
    'n_estimators': 100,               # Number of boosting rounds
    'objective': 'reg:squarederror',   # Loss function
    'booster': 'gbtree',               # Booster type
    'tree_method': 'auto',             # Tree construction algorithm
}

# Regularization Parameters
regularization_params = {
    'reg_alpha': 0,                    # L1 regularization
    'reg_lambda': 1,                   # L2 regularization
    'scale_pos_weight': 1,             # Balance positive/negative weights
}

# Combine all parameters
all_params = {**tree_params, **boosting_params, **regularization_params}
```

### Parameter Validation

```python
def validate_parameters(params):
    """Validate XGBoost parameters"""
    
    # Check for common parameter mistakes
    checks = {
        'max_depth': (1, 20, 'Max depth should be between 1 and 20'),
        'learning_rate': (0.001, 1.0, 'Learning rate should be between 0.001 and 1.0'),
        'n_estimators': (1, 10000, 'Number of estimators should be between 1 and 10000'),
        'subsample': (0.1, 1.0, 'Subsample should be between 0.1 and 1.0'),
        'colsample_bytree': (0.1, 1.0, 'Column sample by tree should be between 0.1 and 1.0'),
    }
    
    for param, (min_val, max_val, message) in checks.items():
        if param in params:
            value = params[param]
            if not (min_val <= value <= max_val):
                print(f"Warning: {message}. Current value: {value}")
    
    # Check for conflicting parameters
    if params.get('booster') == 'gblinear' and 'max_depth' in params:
        print("Warning: max_depth is ignored when using gblinear booster")
    
    return params

# Example usage
validated_params = validate_parameters(all_params)
```

## Objective Functions and Metrics

### Built-in Objectives

```python
# Regression objectives
regression_objectives = {
    'reg:squarederror': 'Mean squared error',
    'reg:squaredlogerror': 'Mean squared log error',
    'reg:logistic': 'Logistic regression',
    'reg:pseudohubererror': 'Pseudo-Huber loss',
    'reg:absoluteerror': 'Mean absolute error',
    'reg:quantileerror': 'Quantile regression',
}

# Classification objectives
classification_objectives = {
    'binary:logistic': 'Binary logistic regression',
    'binary:logitraw': 'Binary logistic regression (raw output)',
    'binary:hinge': 'Binary hinge loss',
    'multi:softmax': 'Multiclass classification (returns class)',
    'multi:softprob': 'Multiclass classification (returns probability)',
}

# Ranking objectives
ranking_objectives = {
    'rank:pairwise': 'Pairwise ranking',
    'rank:ndcg': 'NDCG ranking',
    'rank:map': 'Mean average precision ranking',
}

# Example: Using different objectives
def demonstrate_objectives():
    """Show how to use different objective functions"""
    
    # Binary classification
    binary_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    # Multiclass classification
    multiclass_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 3,  # Required for multiclass
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    # Regression with custom metric
    regression_params = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],  # Multiple metrics
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    return binary_params, multiclass_params, regression_params

binary_params, multiclass_params, regression_params = demonstrate_objectives()
```

### Custom Evaluation Metrics

```python
def custom_rmse(y_pred, y_true):
    """Custom RMSE evaluation metric"""
    y_label = y_true.get_label()
    rmse = np.sqrt(np.mean((y_label - y_pred) ** 2))
    return 'custom_rmse', rmse

def custom_accuracy(y_pred, y_true):
    """Custom accuracy for binary classification"""
    y_label = y_true.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_label == y_pred_binary)
    return 'custom_accuracy', accuracy

# Usage with custom metrics
def train_with_custom_metrics():
    """Train model with custom evaluation metrics"""
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        feval=custom_rmse,  # Custom evaluation function
        verbose_eval=20
    )
    
    return model

# custom_model = train_with_custom_metrics()
```

## Cross-Validation

### Built-in Cross-Validation

```python
def xgboost_cross_validation():
    """Demonstrate XGBoost's built-in cross-validation"""
    
    # Prepare data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # CV parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Run cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,                    # 5-fold CV
        stratified=False,           # For regression
        shuffle=True,
        seed=42,
        early_stopping_rounds=10,
        verbose_eval=50,
        show_stdv=True
    )
    
    print("CV Results Summary:")
    print(cv_results.tail())
    
    # Find best iteration
    best_iteration = cv_results.shape[0]
    best_score = cv_results.iloc[-1]['test-rmse-mean']
    
    print(f"Best iteration: {best_iteration}")
    print(f"Best CV RMSE: {best_score:.4f} ± {cv_results.iloc[-1]['test-rmse-std']:.4f}")
    
    return cv_results

cv_results = xgboost_cross_validation()
```

### Scikit-learn Cross-Validation Integration

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

def sklearn_cv_integration():
    """Integration with scikit-learn cross-validation"""
    
    # Basic cross-validation
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    rmse_scores = np.sqrt(-cv_scores)
    print(f"CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]
    }
    
    grid_search = GridSearchCV(
        XGBRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_

best_model = sklearn_cv_integration()
```

## Model Introspection

### Feature Importance

```python
def analyze_feature_importance(model, feature_names=None):
    """Analyze and visualize feature importance"""
    
    # Get different types of importance
    importance_types = ['weight', 'gain', 'cover']
    
    if hasattr(model, 'feature_importances_'):
        # Scikit-learn interface
        importance = model.feature_importances_
        print("Feature importance (gain):")
        for i, imp in enumerate(importance):
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            print(f"{feature_name}: {imp:.4f}")
    
    elif hasattr(model, 'get_score'):
        # Native XGBoost interface
        for imp_type in importance_types:
            importance_dict = model.get_score(importance_type=imp_type)
            print(f"\nFeature importance ({imp_type}):")
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features[:10]:  # Top 10
                print(f"{feature}: {score:.4f}")
    
    return importance

# Example usage
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
importance = analyze_feature_importance(regressor, feature_names)
```

### Model Visualization

```python
def visualize_model(model, num_trees=1, feature_names=None):
    """Visualize XGBoost model structure"""
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot importance
        if hasattr(model, 'feature_importances_'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Feature importance plot
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:10]  # Top 10
            
            ax1.bar(range(len(indices)), importance[indices])
            ax1.set_title('Feature Importance')
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Importance')
            
            if feature_names:
                ax1.set_xticks(range(len(indices)))
                ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            
            # Tree plot (requires graphviz)
            try:
                import xgboost as xgb
                xgb.plot_tree(model, num_trees=num_trees, ax=ax2)
                ax2.set_title(f'Tree {num_trees}')
            except ImportError:
                ax2.text(0.5, 0.5, 'graphviz required for tree plot', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            plt.show()
    
    except ImportError:
        print("Matplotlib required for visualization")

# Example usage
# visualize_model(regressor, num_trees=0, feature_names=feature_names)
```

## Memory Management

### Efficient Data Loading

```python
def efficient_data_handling():
    """Demonstrate efficient data handling techniques"""
    
    # Memory-efficient data loading
    def load_large_dataset_chunks(file_path, chunk_size=10000):
        """Load large dataset in chunks"""
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Process chunk if needed
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    
    # DMatrix from iterator (memory efficient)
    def create_dmatrix_from_chunks(file_path):
        """Create DMatrix from chunked data"""
        def data_iterator():
            for chunk in pd.read_csv(file_path, chunksize=1000):
                yield chunk.iloc[:, :-1].values, chunk.iloc[:, -1].values
        
        # This is a conceptual example - actual implementation varies
        # return xgb.DMatrix(data_iterator())
    
    # Sparse matrix handling
    from scipy.sparse import csr_matrix
    
    # Convert dense to sparse if many zeros
    def optimize_sparse_data(X, threshold=0.1):
        """Convert to sparse if sparsity > threshold"""
        sparsity = 1.0 - np.count_nonzero(X) / X.size
        if sparsity > threshold:
            print(f"Converting to sparse matrix (sparsity: {sparsity:.2%})")
            return csr_matrix(X)
        return X
    
    # Memory usage monitoring
    def monitor_memory_usage():
        """Monitor memory usage during training"""
        import psutil
        process = psutil.Process()
        
        def get_memory_usage():
            return process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {get_memory_usage():.1f} MB")
        return get_memory_usage
    
    return load_large_dataset_chunks, optimize_sparse_data, monitor_memory_usage

# Get utility functions
load_chunks, optimize_sparse, monitor_memory = efficient_data_handling()
```

## Error Handling

```python
def robust_xgboost_training():
    """Demonstrate robust XGBoost training with error handling"""
    
    def safe_train_model(X, y, params=None):
        """Train XGBoost model with comprehensive error handling"""
        
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1
            }
        
        try:
            # Validate input data
            if X.shape[0] != len(y):
                raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {len(y)}")
            
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print("Warning: NaN values detected in data")
                # XGBoost handles NaN automatically, but you might want to log this
            
            # Create DMatrix with error handling
            dtrain = xgb.DMatrix(X, label=y)
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
            
            return model, None
            
        except xgb.core.XGBoostError as e:
            return None, f"XGBoost error: {str(e)}"
        except MemoryError as e:
            return None, f"Memory error: {str(e)}. Try reducing dataset size or using sparse matrices."
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    # Example usage
    model, error = safe_train_model(X_train, y_train)
    
    if error:
        print(f"Training failed: {error}")
    else:
        print("Training successful!")
        predictions = model.predict(xgb.DMatrix(X_test))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f"RMSE: {rmse:.4f}")
    
    return model

robust_model = robust_xgboost_training()
```

## Best Practices Summary

### Code Organization

```python
class XGBoostTrainer:
    """Organized XGBoost training class"""
    
    def __init__(self, params=None):
        self.par
