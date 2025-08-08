# Hyperparameter Tuning

This comprehensive guide covers modern hyperparameter tuning techniques for XGBoost, including automated optimization methods and best practices.

## Understanding XGBoost Hyperparameters

### Core Parameter Categories

```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, accuracy_score

# Load sample dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter categories and their impacts
parameter_guide = {
    'tree_structure': {
        'max_depth': {
            'range': [3, 4, 5, 6, 7, 8, 9, 10],
            'default': 6,
            'impact': 'Controls tree depth - higher values can overfit',
            'tuning_priority': 'high'
        },
        'min_child_weight': {
            'range': [1, 3, 5, 7, 10, 15, 20],
            'default': 1,
            'impact': 'Minimum sum of instance weight in a child',
            'tuning_priority': 'medium'
        },
        'gamma': {
            'range': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'default': 0,
            'impact': 'Minimum loss reduction for split',
            'tuning_priority': 'low'
        }
    },
    'boosting_params': {
        'learning_rate': {
            'range': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'default': 0.1,
            'impact': 'Step size shrinkage - lower needs more estimators',
            'tuning_priority': 'high'
        },
        'n_estimators': {
            'range': [100, 200, 500, 1000, 1500, 2000],
            'default': 100,
            'impact': 'Number of boosting rounds',
            'tuning_priority': 'high'
        }
    },
    'regularization': {
        'reg_alpha': {
            'range': [0, 0.01, 0.1, 0.5, 1, 2, 5],
            'default': 0,
            'impact': 'L1 regularization - promotes sparsity',
            'tuning_priority': 'medium'
        },
        'reg_lambda': {
            'range': [1, 1.5, 2, 3, 4, 5, 10],
            'default': 1,
            'impact': 'L2 regularization - prevents overfitting',
            'tuning_priority': 'medium'
        }
    },
    'sampling': {
        'subsample': {
            'range': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'default': 1.0,
            'impact': 'Fraction of samples per tree',
            'tuning_priority': 'medium'
        },
        'colsample_bytree': {
            'range': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'default': 1.0,
            'impact': 'Fraction of features per tree',
            'tuning_priority': 'medium'
        }
    }
}

def print_parameter_guide():
    """Display parameter tuning guide"""
    for category, params in parameter_guide.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("=" * 50)
        for param_name, info in params.items():
            print(f"{param_name}:")
            print(f"  Default: {info['default']}")
            print(f"  Range: {info['range']}")
            print(f"  Impact: {info['impact']}")
            print(f"  Priority: {info['tuning_priority']}")
            print()

print_parameter_guide()
```

## Manual Grid Search

### Basic Grid Search

```python
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer
import time

def basic_grid_search():
    """Perform basic grid search with cross-validation"""
    
    # Define parameter grid
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create XGBoost model
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    # Perform grid search
    print("Starting grid search...")
    start_time = time.time()
    
    grid_search.fit(X_train, y_train)
    
    search_time = time.time() - start_time
    
    # Display results
    print(f"\nGrid Search completed in {search_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Test on holdout set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return grid_search

# Run basic grid search
grid_results = basic_grid_search()
```

### Staged Grid Search (Coarse-to-Fine)

```python
def staged_grid_search():
    """Perform staged grid search: coarse then fine-tuning"""
    
    print("Stage 1: Coarse Grid Search")
    print("=" * 40)
    
    # Stage 1: Coarse search
    coarse_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 500, 1000],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0]
    }
    
    coarse_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        coarse_grid,
        cv=3,  # Fewer folds for speed
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    coarse_search.fit(X_train, y_train)
    coarse_best = coarse_search.best_params_
    
    print(f"Coarse search best params: {coarse_best}")
    print(f"Coarse search best score: {np.sqrt(-coarse_search.best_score_):.4f}")
    
    print("\nStage 2: Fine Grid Search")
    print("=" * 40)
    
    # Stage 2: Fine search around best parameters
    fine_grid = {}
    
    # Create fine ranges around best parameters
    for param, value in coarse_best.items():
        if param == 'max_depth':
            fine_grid[param] = [max(1, value-1), value, min(15, value+1)]
        elif param == 'learning_rate':
            fine_grid[param] = [max(0.01, value-0.05), value, min(0.3, value+0.05)]
        elif param == 'n_estimators':
            fine_grid[param] = [max(50, value-100), value, value+100]
        elif param in ['subsample', 'colsample_bytree']:
            fine_grid[param] = [max(0.5, value-0.1), value, min(1.0, value+0.1)]
    
    fine_search = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        fine_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    fine_search.fit(X_train, y_train)
    
    print(f"Fine search best params: {fine_search.best_params_}")
    print(f"Fine search best score: {np.sqrt(-fine_search.best_score_):.4f}")
    
    return coarse_search, fine_search

# coarse_results, fine_results = staged_grid_search()
```

## Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def random_search_optimization():
    """Perform randomized search for hyperparameter optimization"""
    
    # Define parameter distributions
    param_distributions = {
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'n_estimators': randint(50, 2000),
        'subsample': uniform(0.6, 0.4),        # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4), # 0.6 to 1.0
        'reg_alpha': uniform(0, 5),            # 0 to 5
        'reg_lambda': uniform(1, 9),           # 1 to 10
        'min_child_weight': randint(1, 20),
        'gamma': uniform(0, 0.5)
    }
    
    # Create model
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    # Setup random search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter combinations to try
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    # Perform search
    print("Starting randomized search...")
    start_time = time.time()
    
    random_search.fit(X_train, y_train)
    
    search_time = time.time() - start_time
    
    print(f"\nRandomized search completed in {search_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score (RMSE): {np.sqrt(-random_search.best_score_):.4f}")
    
    # Test on holdout set
    y_pred = random_search.best_estimator_.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return random_search

# random_results = random_search_optimization()
```

## Bayesian Optimization with Optuna

```python
# Install optuna: pip install optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    
    def optuna_optimization():
        """Advanced hyperparameter tuning using Optuna"""
        
        def objective(trial):
            # Suggest parameters
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Create model
            model = xgb.XGBRegressor(**params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Return RMSE (Optuna minimizes the objective)
            return np.sqrt(-cv_scores.mean())
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Optimize
        print("Starting Optuna optimization...")
        study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout
        
        # Results
        print(f"\nOptuna optimization completed!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best CV score (RMSE): {study.best_value:.4f}")
        
        # Train final model
        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(X_train, y_train)
        
        # Test performance
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test RMSE: {test_rmse:.4f}")
        
        return study, best_model
    
    # optuna_study, optuna_model = optuna_optimization()
    
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    
    def placeholder_optuna():
        print("Optuna optimization would be performed here")
        return None, None
    
    optuna_study, optuna_model = placeholder_optuna()
```

## Advanced Optuna Features

```python
try:
    import optuna
    import plotly.graph_objects as go
    
    def advanced_optuna_optimization():
        """Advanced Optuna features with visualization and pruning"""
        
        def objective(trial):
            # Suggest parameters with conditions
            max_depth = trial.suggest_int('max_depth', 3, 12)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            
            # Conditional parameters
            if max_depth <= 6:
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            else:
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
            
            # Tree method suggestion
            tree_method = trial.suggest_categorical('tree_method', ['hist', 'approx'])
            
            params = {
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'tree_method': tree_method,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Progressive validation with pruning
            model = xgb.XGBRegressor(**params)
            
            # Use callback for early stopping in CV
            scores = []
            for fold, (train_idx, val_idx) in enumerate(
                sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42).split(X_train)
            ):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                score = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(np.mean(scores), fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Advanced study configuration
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(
                seed=42,
                n_startup_trials=10,
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1
            ),
            study_name='xgboost_advanced_tuning'
        )
        
        # Add callbacks
        def callback(study, trial):
            if trial.number % 10 == 0:
                print(f"Trial {trial.number}: Best value so far: {study.best_value:.4f}")
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=200,
            timeout=7200,  # 2 hours
            callbacks=[callback]
        )
        
        # Analysis
        print(f"\nStudy statistics:")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best parameters: {study.best_params}")
        print(f"Best value: {study.best_value:.4f}")
        
        return study
    
    # advanced_study = advanced_optuna_optimization()
    
except ImportError:
    print("Advanced Optuna features require plotly: pip install plotly")
```

## Early Stopping Strategies

```python
def early_stopping_tuning():
    """Hyperparameter tuning with early stopping"""
    
    # Split training data for early stopping
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    def tune_with_early_stopping(params):
        """Train model with early stopping and return validation score"""
        
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit with early stopping
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Get validation score
        y_val_pred = model.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        return val_score, model.best_iteration
    
    # Parameter grid for early stopping
    early_stop_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [2000],  # High value, will be pruned by early stopping
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    best_score = float('inf')
    best_params = None
    best_n_estimators = None
    
    results = []
    
    for params in ParameterGrid(early_stop_grid):
        try:
            val_score, best_iter = tune_with_early_stopping(params)
            
            results.append({
                'params': params.copy(),
                'val_score': val_score,
                'best_n_estimators': best_iter
            })
            
            if val_score < best_score:
                best_score = val_score
                best_params = params.copy()
                best_n_estimators = best_iter
            
            print(f"Params: {params}")
            print(f"Val RMSE: {val_score:.4f}, Best n_estimators: {best_iter}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
    
    # Update best parameters with optimal n_estimators
    best_params['n_estimators'] = best_n_estimators
    
    print(f"\nBest early stopping results:")
    print(f"Best parameters: {best_params}")
    print(f"Best validation RMSE: {best_score:.4f}")
    print(f"Optimal n_estimators: {best_n_estimators}")
    
    return results, best_params

# early_stop_results, early_stop_best = early_stopping_tuning()
```

## Learning Curve Analysis

```python
import matplotlib.pyplot as plt

def analyze_learning_curves(params_list):
    """Analyze learning curves for different parameter sets"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, params in enumerate(params_list):
        if i >= 4:  # Only plot first 4 parameter sets
            break
            
        # Create model
        model = xgb.XGBRegressor(**params, random_state=42)
        
        # Fit with evaluation
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='rmse',
            verbose=False
        )
        
        # Get evaluation results
        results = model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        
        # Plot learning curves
        axes[i].plot(x_axis, results['validation_0']['rmse'], label='Train')
        axes[i].plot(x_axis, results['validation_1']['rmse'], label='Test')
        axes[i].set_xlabel('Boosting Round')
        axes[i].set_ylabel('RMSE')
        axes[i].set_title(f"Learning Curve - Set {i+1}")
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Example parameter sets for comparison
example_params = [
    {'max_depth': 3, 'learning_rate':
