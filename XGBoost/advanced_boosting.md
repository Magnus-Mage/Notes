# Advanced Boosting Techniques

This guide covers advanced XGBoost boosting methods, custom objectives, and specialized configurations.

## Boosting Algorithms

### Gradient Boosting with Trees (Default)

```python
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Generate sample data
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Standard gradient boosting
standard_gbm = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    booster='gbtree',  # Default tree booster
    random_state=42
)

standard_gbm.fit(X_train_reg, y_train_reg)
pred_standard = standard_gbm.predict(X_test_reg)
print(f"Standard GBM RMSE: {np.sqrt(mean_squared_error(y_test_reg, pred_standard)):.4f}")
```

### DART (Dropouts meet Multiple Additive Regression Trees)

```python
# DART booster - adds dropout regularization
dart_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    booster='dart',
    # DART specific parameters
    sample_type='uniform',  # How to sample trees for dropout
    normalize_type='tree',  # How to normalize dropout
    rate_drop=0.1,  # Dropout rate (fraction of trees to drop)
    one_drop=0,  # Whether to drop at least one tree
    skip_drop=0.5,  # Probability of skipping dropout
    random_state=42
)

dart_model.fit(X_train_reg, y_train_reg)
pred_dart = dart_model.predict(X_test_reg)
print(f"DART RMSE: {np.sqrt(mean_squared_error(y_test_reg, pred_dart)):.4f}")
```

### Linear Booster (GBLINEAR)

```python
# Linear booster - uses linear models instead of trees
linear_model = xgb.XGBRegressor(
    n_estimators=1000,  # More iterations needed for linear
    learning_rate=0.1,
    booster='gblinear',
    # Linear booster parameters
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)

linear_model.fit(X_train_reg, y_train_reg)
pred_linear = linear_model.predict(X_test_reg)
print(f"GBLINEAR RMSE: {np.sqrt(mean_squared_error(y_test_reg, pred_linear)):.4f}")
```

## Advanced Tree Parameters

### Tree Growing Policies

```python
def compare_grow_policies():
    """Compare different tree growing policies"""
    
    policies = ['depthwise', 'lossguide']
    results = {}
    
    for policy in policies:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            grow_policy=policy,
            max_leaves=0 if policy == 'depthwise' else 31,  # Only for lossguide
            random_state=42
        )
        
        model.fit(X_train_reg, y_train_reg)
        pred = model.predict(X_test_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred))
        
        results[policy] = rmse
        print(f"{policy.capitalize()} RMSE: {rmse:.4f}")
    
    return results

grow_policy_results = compare_grow_policies()
```

### Tree Construction Methods

```python
def compare_tree_methods():
    """Compare different tree construction methods"""
    
    methods = ['exact', 'approx', 'hist']
    
    for method in methods:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method=method,
            # Method-specific parameters
            sketch_eps=0.03 if method == 'approx' else None,  # Approximation level
            max_bin=256 if method == 'hist' else None,  # Histogram bins
            random_state=42
        )
        
        import time
        start_time = time.time()
        model.fit(X_train_reg, y_train_reg)
        training_time = time.time() - start_time
        
        pred = model.predict(X_test_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred))
        
        print(f"{method.upper():<6} | RMSE: {rmse:.4f} | Time: {training_time:.2f}s")

print("Tree Method Comparison:")
compare_tree_methods()
```

## Custom Objective Functions

### Custom Regression Objective

```python
def custom_huber_objective(y_pred, y_true, delta=1.0):
    """Custom Huber loss objective function"""
    residual = y_true.get_label() - y_pred
    
    # Huber loss gradient and hessian
    gradient = np.where(
        np.abs(residual) <= delta,
        -residual,  # L2 region
        -delta * np.sign(residual)  # L1 region
    )
    
    hessian = np.where(
        np.abs(residual) <= delta,
        np.ones_like(residual),  # L2 region
        np.zeros_like(residual)  # L1 region (approximation)
    )
    
    # Add small epsilon to avoid division by zero
    hessian = np.maximum(hessian, 1e-6)
    
    return gradient, hessian

# Train with custom objective
dtrain = xgb.DMatrix(X_train_reg, label=y_train_reg)
dtest = xgb.DMatrix(X_test_reg, label=y_test_reg)

params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': custom_huber_objective,  # Custom objective
    'eval_metric': 'rmse'
}

custom_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=False
)

custom_pred = custom_model.predict(dtest)
print(f"Custom Huber Loss RMSE: {np.sqrt(mean_squared_error(y_test_reg, custom_pred)):.4f}")
```

### Custom Classification Objective

```python
def custom_focal_loss_objective(y_pred, y_true, alpha=0.25, gamma=2.0):
    """Custom focal loss for imbalanced classification"""
    y_label = y_true.get_label()
    
    # Convert to probabilities
    prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Focal loss components
    pt = np.where(y_label == 1, prob, 1 - prob)
    alpha_t = np.where(y_label == 1, alpha, 1 - alpha)
    
    # Gradient
    gradient = alpha_t * (gamma * pt**(gamma-1) * np.log(np.maximum(pt, 1e-15)) + pt**gamma / np.maximum(pt, 1e-15)) * (prob - y_label)
    
    # Hessian (approximation)
    hessian = alpha_t * pt**gamma * prob * (1 - prob)
    hessian = np.maximum(hessian, 1e-6)
    
    return gradient, hessian

# Binary classification with focal loss
dtrain_cls = xgb.DMatrix(X_train_cls, label=(y_train_cls == 1).astype(int))
dtest_cls = xgb.DMatrix(X_test_cls, label=(y_test_cls == 1).astype(int))

focal_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': custom_focal_loss_objective,
    'eval_metric': 'logloss'
}

focal_model = xgb.train(
    focal_params,
    dtrain_cls,
    num_boost_round=100,
    evals=[(dtrain_cls, 'train'), (dtest_cls, 'test')],
    verbose_eval=False
)
```

## Custom Evaluation Metrics

```python
def custom_evaluation_metrics():
    """Define custom evaluation metrics"""
    
    def mean_absolute_percentage_error(y_pred, y_true):
        """MAPE evaluation metric"""
        y_label = y_true.get_label()
        mape = np.mean(np.abs((y_label - y_pred) / np.maximum(np.abs(y_label), 1e-6))) * 100
        return 'mape', mape
    
    def weighted_rmse(y_pred, y_true, weights=None):
        """Weighted RMSE evaluation metric"""
        y_label = y_true.get_label()
        if weights is None:
            weights = np.ones_like(y_label)
        
        weighted_squared_errors = weights * (y_label - y_pred) ** 2
        wrmse = np.sqrt(np.sum(weighted_squared_errors) / np.sum(weights))
        return 'wrmse', wrmse
    
    def top_k_accuracy(y_pred, y_true, k=3):
        """Top-k accuracy for multi-class"""
        y_label = y_true.get_label().astype(int)
        n_classes = len(np.unique(y_label))
        
        # Reshape predictions for multi-class
        y_pred_reshaped = y_pred.reshape(-1, n_classes)
        
        # Get top-k predictions
        top_k_pred = np.argsort(y_pred_reshaped, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = np.array([label in top_k_pred[i] for i, label in enumerate(y_label)])
        top_k_acc = np.mean(correct)
        
        return f'top_{k}_accuracy', top_k_acc
    
    # Train with multiple custom metrics
    dtrain = xgb.DMatrix(X_train_reg, label=y_train_reg)
    dtest = xgb.DMatrix(X_test_reg, label=y_test_reg)
    
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
        feval=mean_absolute_percentage_error,
        verbose_eval=20
    )
    
    return model

custom_metrics_model = custom_evaluation_metrics()
```

## Regularization Techniques

### Advanced Regularization

```python
def advanced_regularization():
    """Demonstrate advanced regularization techniques"""
    
    # L1 and L2 regularization
    regularized_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        # Regularization parameters
        reg_alpha=0.1,  # L1 regularization (Lasso)
        reg_lambda=1.0,  # L2 regularization (Ridge)
        # Tree-specific regularization
        gamma=0.1,  # Minimum loss reduction for split
        min_child_weight=3,  # Minimum sum of weights in child
        max_delta_step=1,  # Maximum delta step for updates
        random_state=42
    )
    
    regularized_model.fit(X_train_reg, y_train_reg)
    
    # Dropout regularization (subsampling)
    dropout_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        # Dropout parameters
        subsample=0.8,  # Row subsampling
        colsample_bytree=0.8,  # Column subsampling per tree
        colsample_bylevel=0.8,  # Column subsampling per level
    # Dropout regularization (subsampling)
    dropout_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        # Dropout parameters
        subsample=0.8,  # Row subsampling
        colsample_bytree=0.8,  # Column subsampling per tree
        colsample_bylevel=0.8,  # Column subsampling per level
        colsample_bynode=0.8,  # Column subsampling per node
        random_state=42
    )
    
    dropout_model.fit(X_train_reg, y_train_reg)
    
    # Compare regularization effects
    models = {
        'No Regularization': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42),
        'L1/L2 Regularization': regularized_model,
        'Dropout Regularization': dropout_model
    }
    
    results = {}
    for name, model in models.items():
        if name == 'No Regularization':
            model.fit(X_train_reg, y_train_reg)
        
        train_pred = model.predict(X_train_reg)
        test_pred = model.predict(X_test_reg)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_reg, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_reg, test_pred))
        
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfitting': train_rmse - test_rmse
        }
    
    # Display results
    print("Regularization Comparison:")
    print("=" * 70)
    print(f"{'Method':<20} {'Train RMSE':<12} {'Test RMSE':<12} {'Overfitting':<12}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['train_rmse']:<12.4f} {metrics['test_rmse']:<12.4f} {metrics['overfitting']:<12.4f}")
    
    return results

regularization_results = advanced_regularization()
```

## Monotonic Constraints

```python
def monotonic_constraints_example():
    """Demonstrate monotonic constraints"""
    
    # Create dataset where we want monotonic relationships
    np.random.seed(42)
    n_samples = 1000
    
    # Features with known monotonic relationships
    age = np.random.uniform(18, 80, n_samples)  # Should increase price
    mileage = np.random.uniform(0, 200000, n_samples)  # Should decrease price
    horsepower = np.random.uniform(100, 500, n_samples)  # Should increase price
    
    # Create target with monotonic relationships + noise
    price = (age * 100 - mileage * 0.1 + horsepower * 50 + 
             np.random.normal(0, 1000, n_samples))
    
    X_mono = np.column_stack([age, mileage, horsepower])
    feature_names = ['age', 'mileage', 'horsepower']
    
    X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(
        X_mono, price, test_size=0.2, random_state=42
    )
    
    # Model without constraints
    model_unconstrained = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    # Model with monotonic constraints
    model_constrained = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        monotone_constraints=(1, -1, 1),  # age↑, mileage↓, horsepower↑
        random_state=42
    )
    
    # Train both models
    model_unconstrained.fit(X_train_mono, y_train_mono)
    model_constrained.fit(X_train_mono, y_train_mono)
    
    # Test monotonicity
    def test_monotonicity(model, feature_idx, feature_name):
        """Test if model respects monotonic constraint"""
        test_point = np.mean(X_test_mono, axis=0).reshape(1, -1)
        
        feature_range = np.linspace(
            X_test_mono[:, feature_idx].min(),
            X_test_mono[:, feature_idx].max(),
            50
        )
        
        predictions = []
        for value in feature_range:
            point = test_point.copy()
            point[0, feature_idx] = value
            pred = model.predict(point)[0]
            predictions.append(pred)
        
        # Check monotonicity
        diffs = np.diff(predictions)
        if feature_idx == 1:  # mileage (should decrease)
            monotonic = np.all(diffs <= 0.1)  # Allow small violations
        else:  # age, horsepower (should increase)
            monotonic = np.all(diffs >= -0.1)  # Allow small violations
        
        return monotonic, predictions, feature_range
    
    print("Monotonicity Test Results:")
    print("=" * 50)
    
    for i, feature_name in enumerate(feature_names):
        unconstrained_mono, _, _ = test_monotonicity(model_unconstrained, i, feature_name)
        constrained_mono, _, _ = test_monotonicity(model_constrained, i, feature_name)
        
        print(f"{feature_name:<12} | Unconstrained: {'✅' if unconstrained_mono else '❌'} | Constrained: {'✅' if constrained_mono else '❌'}")
    
    return model_constrained, model_unconstrained

constrained_model, unconstrained_model = monotonic_constraints_example()
```

## Feature Interaction Constraints

```python
def feature_interaction_constraints():
    """Demonstrate feature interaction constraints"""
    
    # Create interaction constraint groups
    # Features in the same group can interact, features in different groups cannot
    
    # Example: [0, 1] can interact with each other, [2, 3] can interact with each other
    # but group [0, 1] cannot interact with group [2, 3]
    
    interaction_constraints = [[0, 1], [2, 3]]  # Two interaction groups
    
    model_with_constraints = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        interaction_constraints=interaction_constraints,
        random_state=42
    )
    
    model_with_constraints.fit(X_train_reg, y_train_reg)
    
    # Compare with unconstrained model
    model_no_constraints = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model_no_constraints.fit(X_train_reg, y_train_reg)
    
    # Evaluate both models
    constrained_pred = model_with_constraints.predict(X_test_reg)
    unconstrained_pred = model_no_constraints.predict(X_test_reg)
    
    constrained_rmse = np.sqrt(mean_squared_error(y_test_reg, constrained_pred))
    unconstrained_rmse = np.sqrt(mean_squared_error(y_test_reg, unconstrained_pred))
    
    print("Feature Interaction Constraints:")
    print(f"Constrained Model RMSE: {constrained_rmse:.4f}")
    print(f"Unconstrained Model RMSE: {unconstrained_rmse:.4f}")
    
    return model_with_constraints

interaction_model = feature_interaction_constraints()
```

## Multi-target Regression

```python
def multi_target_regression():
    """Demonstrate multi-target regression with XGBoost"""
    
    # Create multi-target dataset
    from sklearn.datasets import make_regression
    
    X, y_multi = make_regression(
        n_samples=1000,
        n_features=10,
        n_targets=3,  # 3 targets
        noise=0.1,
        random_state=42
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42
    )
    
    # Method 1: Train separate models for each target
    separate_models = {}
    separate_predictions = {}
    
    for i in range(y_multi.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train_multi, y_train_multi[:, i])
        pred = model.predict(X_test_multi)
        
        separate_models[f'target_{i}'] = model
        separate_predictions[f'target_{i}'] = pred
    
    # Method 2: Multi-output wrapper
    from sklearn.multioutput import MultiOutputRegressor
    
    multi_output_model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    )
    
    multi_output_model.fit(X_train_multi, y_train_multi)
    multi_output_pred = multi_output_model.predict(X_test_multi)
    
    # Evaluate both approaches
    print("Multi-target Regression Results:")
    print("=" * 40)
    
    for i in range(y_multi.shape[1]):
        separate_rmse = np.sqrt(mean_squared_error(y_test_multi[:, i], separate_predictions[f'target_{i}']))
        multi_rmse = np.sqrt(mean_squared_error(y_test_multi[:, i], multi_output_pred[:, i]))
        
        print(f"Target {i}: Separate={separate_rmse:.4f}, Multi-output={multi_rmse:.4f}")
    
    return separate_models, multi_output_model

separate_models, multi_output_model = multi_target_regression()
```

## Early Stopping Strategies

```python
def advanced_early_stopping():
    """Demonstrate advanced early stopping strategies"""
    
    # Strategy 1: Standard early stopping
    model_standard_es = xgb.XGBRegressor(
        n_estimators=1000,  # Large number
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=10,
        random_state=42
    )
    
    model_standard_es.fit(
        X_train_reg, y_train_reg,
        eval_set=[(X_test_reg, y_test_reg)],
        verbose=False
    )
    
    # Strategy 2: Custom early stopping with multiple metrics
    class MultiMetricEarlyStopping:
        def __init__(self, patience=10, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.best_scores = {}
            self.wait_counts = {}
        
        def __call__(self, env):
            """Custom early stopping callback"""
            # Get current evaluation results
            for eval_name, eval_metric, score, _ in env.evaluation_result_list:
                key = f"{eval_name}_{eval_metric}"
                
                if key not in self.best_scores:
                    self.best_scores[key] = float('inf')
                    self.wait_counts[key] = 0
                
                # Check if score improved
                if score < self.best_scores[key] - self.min_delta:
                    self.best_scores[key] = score
                    self.wait_counts[key] = 0
                else:
                    self.wait_counts[key] += 1
                
                # Stop if any metric hasn't improved for patience rounds
                if self.wait_counts[key] >= self.patience:
                    print(f"Early stopping at iteration {env.iteration} due to {key}")
                    raise xgb.core.EarlyStopException(env.iteration)
    
    # Strategy 3: Learning rate scheduling with early stopping
    class LearningRateScheduler:
        def __init__(self, base_lr=0.1, decay_factor=0.9, patience=5):
            self.base_lr = base_lr
            self.decay_factor = decay_factor
            self.patience = patience
            self.best_score = float('inf')
            self.wait = 0
            self.current_lr = base_lr
        
        def __call__(self, env):
            """Learning rate scheduler callback"""
            if len(env.evaluation_result_list) > 0:
                current_score = env.evaluation_result_list[0][2]  # First metric score
                
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.wait = 0
                else:
                    self.wait += 1
                    
                    if self.wait >= self.patience:
                        self.current_lr *= self.decay_factor
                        env.model.set_param('learning_rate', self.current_lr)
                        print(f"Reducing learning rate to {self.current_lr:.6f}")
                        self.wait = 0
    
    # Train with custom callbacks
    dtrain = xgb.DMatrix(X_train_reg, label=y_train_reg)
    dtest = xgb.DMatrix(X_test_reg, label=y_test_reg)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': ['rmse', 'mae']
    }
    
    custom_early_stop = MultiMetricEarlyStopping(patience=15)
    lr_scheduler = LearningRateScheduler()
    
    try:
        model_custom = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            callbacks=[custom_early_stop, lr_scheduler],
            verbose_eval=False
        )
    except xgb.core.EarlyStopException as e:
        print(f"Training stopped early at iteration {e.best_iteration}")
    
    print(f"Standard early stopping: {model_standard_es.best_iteration} iterations")
    
    return model_standard_es

early_stopping_model = advanced_early_stopping()
```

## Model Stacking and Ensembling

```python
def xgboost_ensemble():
    """Create ensemble of XGBoost models with different configurations"""
    
    # Define different model configurations
    model_configs = [
        {'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'max_depth': 8, 'learning_rate': 0.02, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'booster': 'dart', 'max_depth': 6, 'learning_rate': 0.1, 'rate_drop': 0.1},
        {'booster': 'gblinear', 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
    ]
    
    # Train ensemble models
    ensemble_models = []
    ensemble_predictions = []
    
    for i, config in enumerate(model_configs):
        print(f"Training model {i+1}/5...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            **config
        )
        
        model.fit(X_train_reg, y_train_reg)
        pred = model.predict(X_test_reg)
        
        ensemble_models.append(model)
        ensemble_predictions.append(pred)
    
    # Simple averaging ensemble
    avg_ensemble_pred = np.mean(ensemble_predictions, axis=0)
    
    # Weighted ensemble (weights based on individual performance)
    weights = []
    for pred in ensemble_predictions:
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred))
        weight = 1.0 / rmse  # Better models get higher weights
        weights.append(weight)
    
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    weighted_ensemble_pred = np.average(ensemble_predictions, axis=0, weights=weights)
    
    # Stack ensemble (train meta-learner)
    from sklearn.linear_model import LinearRegression
    
    # Create meta-features (predictions from base models)
    meta_features = np.column_stack(ensemble_predictions)
    
    # Train meta-learner
    meta_learner = LinearRegression()
    meta_learner.fit(meta_features, y_test_reg)
    
    # Meta-learner predictions (this would be on a separate holdout set in practice)
    stack_ensemble_pred = meta_learner.predict(meta_features)
    
    # Evaluate ensemble methods
    single_rmse = [np.sqrt(mean_squared_error(y_test_reg, pred)) for pred in ensemble_predictions]
    avg_rmse = np.sqrt(mean_squared_error(y_test_reg, avg_ensemble_pred))
    weighted_rmse = np.sqrt(mean_squared_error(y_test_reg, weighted_ensemble_pred))
    stack_rmse = np.sqrt(mean_squared_error(y_test_reg, stack_ensemble_pred))
    
    print("\nEnsemble Results:")
    print("=" * 40)
    print(f"Individual models RMSE: {single_rmse}")
    print(f"Best single model RMSE: {min(single_rmse):.4f}")
    print(f"Average ensemble RMSE: {avg_rmse:.4f}")
    print(f"Weighted ensemble RMSE: {weighted_rmse:.4f}")
    print(f"Stacked ensemble RMSE: {stack_rmse:.4f}")
    
    return ensemble_models, weights, meta_learner

ensemble_models, ensemble_weights, meta_learner = xgboost_ensemble()
```

## Advanced Feature Engineering

```python
def automatic_feature_engineering():
    """Demonstrate automatic feature engineering with XGBoost"""
    
    # Original features
    X_original = X_train_reg.copy()
    
    # Polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_original[:, :5])  # Only first 5 features to avoid explosion
    
    # Statistical features
    X_stats = np.column_stack([
        np.mean(X_original, axis=1),      # Row means
        np.std(X_original, axis=1),       # Row standard deviations
        np.max(X_original, axis=1),       # Row maximums
        np.min(X_original, axis=1),       # Row minimums
        np.median(X_original, axis=1)     # Row medians
    ])
    
    # Combine all features
    X_engineered = np.column_stack([X_original, X_poly, X_stats])
    
    print(f"Original features: {X_original.shape[1]}")
    print(f"With polynomial features: {X_poly.shape[1]}")
    print(f"With statistical features: {X_stats.shape[1]}")
    print(f"Total engineered features: {X_engineered.shape[1]}")
    
    # Train model with feature selection
    model_feature_selection = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        colsample_bytree=0.5,  # Randomly select 50% of features
        random_state=42
    )
    
    # Split engineered data
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
        X_engineered, y_train_reg, test_size=0.2, random_state=42
    )
    
    model_feature_selection.fit(X_train_eng, y_train_eng)
    pred_engineered = model_feature_selection.predict(X_test_eng)
    
    # Compare with original features
    model_original = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    X_train_orig, X_test_orig, _, _ = train_test_split(
        X_original, y_train_reg, test_size=0.2, random_state=42
    )
    
    model_original.fit(X_train_orig, y_train_eng)
    pred_original = model_original.predict(X_test_orig)
    
    rmse_original = np.sqrt(mean_squared_error(y_test_eng, pred_original))
    rmse_engineered = np.sqrt(mean_squared_error(y_test_eng, pred_engineered))
    
    print(f"\nFeature Engineering Results:")
    print(f"Original features RMSE: {rmse_original:.4f}")
    print(f"Engineered features RMSE: {rmse_engineered:.4f}")
    print(f"Improvement: {((rmse_original - rmse_engineered) / rmse_original * 100):.2f}%")
    
    return model_feature_selection, model_original

feature_eng_model, original_model = automatic_feature_engineering()
```

## Next Steps

- For hyperparameter optimization of these advanced techniques, see [Hyperparameter Tuning](08-hyperparameter-tuning.md)
- For production deployment of advanced models, check [Model Inference](09-model-inference.md)
- For troubleshooting advanced configurations, visit [Troubleshooting Guide](10-troubleshooting.md)