# CPU Usage Examples

This guide covers practical XGBoost usage on CPU with sample code and datasets.

## Basic Regression Example

### Sample Dataset: House Prices

```python
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load sample dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Create DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(feature_names)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create XGBoost regressor
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Basic Classification Example

### Sample Dataset: Wine Quality

```python
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

## Advanced CPU Configuration

### Multi-threading Optimization

```python
import xgboost as xgb
import multiprocessing

# Get number of CPU cores
n_cores = multiprocessing.cpu_count()
print(f"Available CPU cores: {n_cores}")

# Configure for maximum CPU utilization
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    n_jobs=n_cores,  # Use all available cores
    tree_method='hist',  # Faster histogram-based algorithm
    random_state=42
)

# Train with verbose output
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=True
)
```

### Memory-Efficient Training

```python
# For large datasets that might not fit in memory
def train_large_dataset(X_train, y_train, X_test, y_test):
    # Convert to DMatrix for memory efficiency
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters optimized for memory usage
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,  # Reduce depth to save memory
        'learning_rate': 0.1,
        'subsample': 0.8,  # Use subset of data
        'colsample_bytree': 0.8,  # Use subset of features
        'tree_method': 'approx',  # Approximate algorithm
        'sketch_eps': 0.1  # Approximation level
    }
    
    # Train with evaluation
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    return model
```

## Cross-Validation Example

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Regression cross-validation
def cv_regression_example():
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(
        model, X, y, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.4f} (+/- {np.sqrt(-cv_scores.std()) * 2:.4f})")
    return cv_scores

# Classification cross-validation
def cv_classification_example():
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # Stratified 5-fold cross-validation
    cv_scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return cv_scores
```

## Feature Importance Analysis

```python
import matplotlib.pyplot as plt

def analyze_feature_importance(model, feature_names):
    # Get feature importance
    importance = model.feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Usage
importance_df = analyze_feature_importance(model, feature_names)
print(importance_df.head(10))
```

## Model Validation

```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(X, y, param_name, param_range):
    model = xgb.XGBRegressor(random_state=42)
    
    train_scores, test_scores = validation_curve(
        model, X, y, 
        param_name=param_name, 
        param_range=param_range,
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    train_mean = np.sqrt(-train_scores.mean(axis=1))
    train_std = np.sqrt(-train_scores.std(axis=1))
    test_mean = np.sqrt(-test_scores.mean(axis=1))
    test_std = np.sqrt(-test_scores.std(axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training RMSE')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(param_range, test_mean, 'o-', color='red', label='Validation RMSE')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(f'Validation Curve for {param_name}')
    plt.show()

# Example: Analyze max_depth
max_depth_range = range(1, 11)
plot_validation_curve(X, y, 'max_depth', max_depth_range)
```

## Performance Monitoring

```python
import time

def benchmark_training(X_train, y_train, X_test, y_test):
    # Different configurations to benchmark
    configs = [
        {'name': 'Default', 'params': {}},
        {'name': 'Fast', 'params': {'tree_method': 'hist', 'max_depth': 4}},
        {'name': 'Accurate', 'params': {'n_estimators': 500, 'learning_rate': 0.05}},
        {'name': 'Memory Efficient', 'params': {'tree_method': 'approx', 'subsample': 0.8}}
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        # Create model with configuration
        model = xgb.XGBRegressor(random_state=42, **config['params'])
        
        # Time the training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Time the prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate accuracy
        mse = mean_squared_error(y_test, y_pred)
        
        results.append({
            'Configuration': config['name'],
            'Training Time (s)': training_time,
            'Prediction Time (s)': prediction_time,
            'MSE': mse
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df
```

## Custom Evaluation Metrics

```python
def custom_evaluation_example():
    # Custom evaluation metric
    def custom_eval(y_pred, dtrain):
        y_true = dtrain.get_label()
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return 'mape', mape
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    # Train with custom evaluation
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        feval=custom_eval,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    return model
```

## Next Steps

- For GPU acceleration, see [GPU Usage Examples](04-gpu-examples.md)
- For cloud deployment, check [Cloud Providers Integration](05-cloud-providers.md)
- For hyperparameter tuning, visit [Hyperparameter Tuning](08-hyperparameter-tuning.md)