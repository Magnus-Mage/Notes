# XGBoost Basics and Use Cases

## Core Concepts Deep Dive

### Understanding Gradient Boosting

XGBoost implements gradient boosting, which builds models sequentially where each new model corrects the errors of previous models.

```python
# conceptual_demonstration.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Generate synthetic data for demonstration
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X = X.reshape(-1, 1)

# Sort for visualization
sort_idx = np.argsort(X.flatten())
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# Demonstrate boosting concept step by step
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Step 1: Initial prediction (mean)
initial_pred = np.full_like(y_sorted, np.mean(y_sorted))
residuals_1 = y_sorted - initial_pred

axes[0,0].scatter(X_sorted, y_sorted, alpha=0.6, label='True values')
axes[0,0].plot(X_sorted, initial_pred, 'r-', label='Initial prediction (mean)')
axes[0,0].set_title('Step 1: Initial Prediction')
axes[0,0].legend()

# Step 2: First tree fits residuals
tree1 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree1.fit(X_sorted, residuals_1)
pred1 = tree1.predict(X_sorted)
combined_pred1 = initial_pred + 0.3 * pred1  # Learning rate = 0.3
residuals_2 = y_sorted - combined_pred1

axes[0,1].scatter(X_sorted, residuals_1, alpha=0.6, label='Residuals')
axes[0,1].plot(X_sorted, pred1, 'g-', label='Tree 1 prediction')
axes[0,1].set_title('Step 2: First Tree Fits Residuals')
axes[0,1].legend()

# Step 3: Second tree fits new residuals
tree2 = DecisionTreeRegressor(max_depth=3, random_state=43)
tree2.fit(X_sorted, residuals_2)
pred2 = tree2.predict(X_sorted)
combined_pred2 = combined_pred1 + 0.3 * pred2

axes[1,0].scatter(X_sorted, y_sorted, alpha=0.6, label='True values')
axes[1,0].plot(X_sorted, combined_pred1, 'orange', label='After tree 1')
axes[1,0].plot(X_sorted, combined_pred2, 'purple', label='After tree 2')
axes[1,0].set_title('Step 3: Sequential Improvement')
axes[1,0].legend()

# Step 4: Compare with XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.3, random_state=42)
xgb_model.fit(X_sorted, y_sorted)
xgb_pred = xgb_model.predict(X_sorted)

axes[1,1].scatter(X_sorted, y_sorted, alpha=0.6, label='True values')
axes[1,1].plot(X_sorted, xgb_pred, 'red', linewidth=2, label='XGBoost prediction')
axes[1,1].set_title('Step 4: XGBoost Final Result')
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("Gradient Boosting Demonstration:")
print("=" * 35)
print(f"Initial MSE (mean prediction): {np.mean((y_sorted - initial_pred)**2):.2f}")
print(f"After 1 tree: {np.mean((y_sorted - combined_pred1)**2):.2f}")
print(f"After 2 trees: {np.mean((y_sorted - combined_pred2)**2):.2f}")
print(f"XGBoost (50 trees): {np.mean((y_sorted - xgb_pred)**2):.2f}")
```

### XGBoost Mathematical Foundation

```python
# mathematical_foundation.py
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def demonstrate_objective_function():
    """Demonstrate XGBoost objective function components."""
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    
    # Train model with different regularization parameters
    regularization_configs = [
        {'reg_alpha': 0, 'reg_lambda': 0, 'name': 'No Regularization'},
        {'reg_alpha': 0.1, 'reg_lambda': 0, 'name': 'L1 Regularization'},
        {'reg_alpha': 0, 'reg_lambda': 0.1, 'name': 'L2 Regularization'},
        {'reg_alpha': 0.1, 'reg_lambda': 0.1, 'name': 'L1 + L2 Regularization'}
    ]
    
    print("Regularization Impact on XGBoost:")
    print("=" * 40)
    
    for config in regularization_configs:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=42
        )
        
        model.fit(X, y)
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)
        
        print(f"{config['name']:20}: MSE = {mse:.6f}")
        
        # Show feature importance changes
        importance = model.feature_importances_
        print(f"{'':20}  Feature importance std: {np.std(importance):.4f}")
    
    return regularization_configs

def demonstrate_learning_rate_impact():
    """Show how learning rate affects convergence."""
    
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    
    learning_rates = [0.01, 0.1, 0.3, 1.0]
    
    print("\nLearning Rate Impact:")
    print("=" * 25)
    
    results = {}
    for lr in learning_rates:
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=lr,
            random_state=42,
            eval_metric='rmse'
        )
        
        # Track training progress
        eval_result = {}
        model.fit(X, y, eval_set=[(X, y)], eval_metric='rmse', verbose=False)
        
        final_score = model.best_score
        best_iteration = model.best_iteration
        
        results[lr] = {
            'final_score': final_score,
            'best_iteration': best_iteration
        }
        
        print(f"Learning Rate {lr:4.2f}: RMSE = {final_score:.4f}, Best iter = {best_iteration}")
    
    return results

if __name__ == "__main__":
    demonstrate_objective_function()
    demonstrate_learning_rate_impact()
```

## Algorithm Comparison Framework

```python
# algorithm_comparison.py
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

class AlgorithmBenchmark:
    """Comprehensive algorithm comparison framework."""
    
    def __init__(self):
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        self.classification_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
    
    def benchmark_regression(self, n_samples=5000, n_features=20, noise=0.1):
        """Benchmark regression algorithms."""
        
        print("Regression Algorithm Benchmark")
        print("=" * 50)
        
        # Generate dataset
        X, y = make_regression(
            n_samples=n_samples, 
            n_features=n_features, 
            noise=noise, 
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        
        for name, model in self.regression_models.items():
            print(f"\nTesting {name}...")
            
            # Training time
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Prediction time
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results.append({
                    'Algorithm': name,
                    'Training Time (s)': training_time,
                    'Prediction Time (s)': prediction_time,
                    'MSE': mse,
                    'R²': r2,
                    'CV R² Mean': cv_mean,
                    'CV R² Std': cv_std
                })
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R²', ascending=False)
        
        print("\nRegression Results Summary:")
        print(results_df.round(4))
        
        return results_df
    
    def benchmark_classification(self, n_samples=5000, n_features=20, n_classes=3):
        """Benchmark classification algorithms."""
        
        print("\nClassification Algorithm Benchmark")
        print("=" * 50)
        
        # Generate dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_classes=n_classes,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        
        for name, model in self.classification_models.items():
            print(f"\nTesting {name}...")
            
            start_time = time.time()
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                accuracy = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results.append({
                    'Algorithm': name,
                    'Training Time (s)': training_time,
                    'Prediction Time (s)': prediction_time,
                    'Accuracy': accuracy,
                    'CV Accuracy Mean': cv_scores.mean(),
                    'CV Accuracy Std': cv_scores.std()
                })
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\nClassification Results Summary:")
        print(results_df.round(4))
        
        return results_df
    
    def visualize_results(self, regression_results, classification_results):
        """Visualize benchmark results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Regression R² scores
        reg_sorted = regression_results.sort_values('R²', ascending=True)
        axes[0,0].barh(reg_sorted['Algorithm'], reg_sorted['R²'])
        axes[0,0].set_xlabel('R² Score')
        axes[0,0].set_title('Regression Performance (R²)')
        
        # Regression training times
        axes[0,1].barh(reg_sorted['Algorithm'], reg_sorted['Training Time (s)'])
        axes[0,1].set_xlabel('Training Time (seconds)')
        axes[0,1].set_title('Regression Training Time')
        axes[0,1].set_xscale('log')
        
        # Classification accuracy
        if not classification_results.empty:
            cls_sorted = classification_results.sort_values('Accuracy', ascending=True)
            axes[1,0].barh(cls_sorted['Algorithm'], cls_sorted['Accuracy'])
            axes[1,0].set_xlabel('Accuracy')
            axes[1,0].set_title('Classification Performance')
            
            # Classification training times
            axes[1,1].barh(cls_sorted['Algorithm'], cls_sorted['Training Time (s)'])
            axes[1,1].set_xlabel('Training Time (seconds)')
            axes[1,1].set_title('Classification Training Time')
            axes[1,1].set_xscale('log')
        
        plt.tight_layout()
        plt.show()

def main():
    """Run comprehensive algorithm benchmark."""
    benchmark = AlgorithmBenchmark()
    
    # Run benchmarks
    regression_results = benchmark.benchmark_regression()
    classification_results = benchmark.benchmark_classification()
    
    # Visualize results
    benchmark.visualize_results(regression_results, classification_results)
    
    # XGBoost specific analysis
    print("\nXGBoost Specific Insights:")
    print("=" * 30)
    
    xgb_reg_idx = regression_results[regression_results['Algorithm'] == 'XGBoost'].index[0]
    xgb_reg_row = regression_results.loc[xgb_reg_idx]
    
    print(f"XGBoost Regression Ranking: {regression_results.index.get_loc(xgb_reg_idx) + 1} out of {len(regression_results)}")
    print(f"XGBoost R² Score: {xgb_reg_row['R²']:.4f}")
    print(f"XGBoost Training Time: {xgb_reg_row['Training Time (s)']:.4f}s")
    
    if not classification_results.empty:
        xgb_cls_idx = classification_results[classification_results['Algorithm'] == 'XGBoost'].index[0]
        xgb_cls_row = classification_results.loc[xgb_cls_idx]
        print(f"XGBoost Classification Ranking: {classification_results.index.get_loc(xgb_cls_idx) + 1} out of {len(classification_results)}")
        print(f"XGBoost Accuracy: {xgb_cls_row['Accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

## Real-world Use Case Examples

### 1. Financial Credit Scoring

```python
# credit_scoring_example.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def create_synthetic_credit_data(n_samples=10000):
    """Create synthetic credit scoring dataset."""
    np.random.seed(42)
    
    # Demographics
    age = np.random.normal(40, 12, n_samples)
    age = np.clip(age, 18, 80)
    
    income = np.random.lognormal(10, 0.5, n_samples)
    income = np.clip(income, 20000, 200000)
    
    # Credit history
    credit_history_length = np.random.exponential(5, n_samples)
    credit_history_length = np.clip(credit_history_length, 0, 30)
    
    existing_loans = np.random.poisson(2, n_samples)
    existing_loans = np.clip(existing_loans, 0, 10)
    
    # Employment
    employment_length = np.random.exponential(3, n_samples)
    employment_length = np.clip(employment_length, 0, 20)
    
    # Debt ratios
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8
    
    # Credit utilization
    credit_utilization = np.random.beta(2, 3, n_samples)
    
    # Categorical variables
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples, p=[0.4, 0.35, 0.2, 0.05])
    
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    employment_type = np.random.choice(employment_types, n_samples, p=[0.7, 0.15, 0.1, 0.05])
    
    # Target variable (default probability based on features)
    default_prob = (
        0.1 +
        0.3 * (debt_to_income > 0.4) +
        0.2 * (credit_utilization > 0.8) +
        0.15 * (existing_loans > 3) +
        0.1 * (employment_type == 'Unemployed') +
        -0.1 * (income > 60000) +
        -0.05 * (credit_history_length > 5)
    )
    
    default_prob = np.clip(default_prob, 0.01, 0.99)
    default = np.random.binomial(1, default_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_history_length': credit_history_length,
        'existing_loans': existing_loans,
        'employment_length': employment_length,
        'debt_to_income_ratio': debt_to_income,
        'credit_utilization': credit_utilization,
        'education': education,
        'employment_type': employment_type,
        'default': default
    })
    
    return data

def build_credit_scoring_model():
    """Build and evaluate credit scoring model."""
    
    print("Credit Scoring with XGBoost")
    print("=" * 30)
    
    # Create synthetic data
    data = create_synthetic_credit_data(10000)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Default rate: {data['default'].mean():.2%}")
    
    # Encode categorical variables
    le_education = LabelEncoder()
    le_employment = LabelEncoder()
    
    data['education_encoded'] = le_education.fit_transform(data['education'])
    data['employment_type_encoded'] = le_employment.fit_transform(data['employment_type'])
    
    # Prepare features
    feature_columns = [
        'age', 'income', 'credit_history_length', 'existing_loans',
        'employment_length', 'debt_to_income_ratio', 'credit_utilization',
        'education_encoded', 'employment_type_encoded'
    ]
    
    X = data[feature_columns]
    y = data['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
        random_state=42
    )
    
    # Train with early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='auc',
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Evaluation
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print("-" * 20)
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature importance
    top_features = feature_importance.head(8)
    axes[0,0].barh(top_features['feature'], top_features['importance'])
    axes[0,0].set_title('Feature Importance')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_p
