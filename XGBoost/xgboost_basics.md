# XGBoost Basics

## What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is an optimized distributed gradient boosting library designed for high performance, flexibility, and portability. It implements machine learning algorithms under the Gradient Boosting framework.

### Key Features

- **High Performance**: Optimized for speed and memory efficiency
- **Flexibility**: Supports various objective functions and evaluation metrics
- **Portability**: Runs on major distributed environments (Hadoop, SGE, MPI)
- **Language Support**: Available in Python, R, Java, Scala, Julia, and more

## Core Concepts

### Gradient Boosting

XGBoost builds models sequentially, where each new model corrects errors made by previous models:

1. Start with an initial prediction
2. Calculate residuals (errors)
3. Train a new model to predict these residuals
4. Add the new model's predictions to the ensemble
5. Repeat until convergence or maximum iterations

### Key Algorithms

- **Tree Boosting**: Uses decision trees as base learners
- **Linear Boosting**: Uses linear models as base learners
- **DART**: Dropout regularization for tree boosting
- **GBLINEAR**: Linear gradient boosting

## When to Use XGBoost

### Ideal Use Cases

- **Structured/Tabular Data**: Excel sheets, CSV files, database tables
- **Competition Datasets**: Kaggle, ML competitions
- **Mixed Data Types**: Numerical and categorical features
- **Medium-sized Datasets**: 1K to 1M+ samples
- **Prediction Accuracy**: When high accuracy is crucial

### Application Areas

#### Finance
- Credit scoring
- Fraud detection
- Risk assessment
- Algorithmic trading

#### Healthcare
- Disease prediction
- Drug discovery
- Medical diagnosis
- Patient outcome prediction

#### E-commerce
- Recommendation systems
- Price optimization
- Customer churn prediction
- Demand forecasting

#### Marketing
- Customer segmentation
- Lead scoring
- Campaign optimization
- Lifetime value prediction

#### Technology
- Click-through rate prediction
- Search ranking
- Ad targeting
- System monitoring

## XGBoost vs Other Algorithms

### Advantages over Traditional ML

| Aspect | XGBoost | Random Forest | SVM | Linear Models |
|--------|---------|---------------|-----|---------------|
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Interpretability | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| Overfitting Control | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### When NOT to Use XGBoost

- **Deep Learning Tasks**: Images, text, audio (use neural networks)
- **Very Small Datasets**: < 1000 samples (simple models work better)
- **Real-time Inference**: Microsecond latency requirements
- **Linear Relationships**: When data is perfectly linear
- **Sparse High-dimensional Data**: Text classification with TF-IDF

## Types of Problems

### Supervised Learning

#### Regression
```python
# Continuous target variable
xgb.XGBRegressor()
```

#### Classification
```python
# Binary classification
xgb.XGBClassifier()

# Multi-class classification
xgb.XGBClassifier(num_class=n_classes)
```

#### Ranking
```python
# Learning to rank
xgb.XGBRanker()
```

### Model Interfaces

#### Scikit-learn Interface (Recommended for beginners)
```python
from xgboost import XGBRegressor, XGBClassifier

model = XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Native Interface (More control)
```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror'}
model = xgb.train(params, dtrain, num_boost_round=100)
```

## Key Parameters Overview

### Tree Parameters
- `max_depth`: Maximum depth of trees (default: 6)
- `min_child_weight`: Minimum sum of weights in child nodes (default: 1)
- `subsample`: Fraction of samples for each tree (default: 1.0)
- `colsample_bytree`: Fraction of features for each tree (default: 1.0)

### Boosting Parameters
- `learning_rate`: Step size shrinkage (default: 0.3)
- `n_estimators`: Number of boosting rounds (default: 100)
- `objective`: Loss function to minimize
- `eval_metric`: Evaluation metric for validation

### Regularization
- `reg_alpha`: L1 regularization (default: 0)
- `reg_lambda`: L2 regularization (default: 1)

## Performance Characteristics

### Memory Usage
- **DMatrix**: Efficient sparse matrix storage
- **Feature Engineering**: Automatic handling of missing values
- **Memory Mapping**: Large datasets that don't fit in memory

### Speed Optimizations
- **Parallel Processing**: Multi-threading support
- **GPU Acceleration**: CUDA support for training
- **Approximate Learning**: Histogram-based algorithms
- **Early Stopping**: Prevents overfitting and saves time

## Common Workflows

### Basic Workflow
1. Data preprocessing
2. Feature engineering
3. Train-validation split
4. Model training
5. Hyperparameter tuning
6. Model evaluation
7. Final prediction

### Advanced Workflow
1. Exploratory data analysis
2. Feature selection and engineering
3. Cross-validation setup
4. Multi-objective optimization
5. Model stacking/ensembling
6. Model interpretation
7. Production deployment

## Next Steps

Now that you understand the basics, let's move to practical examples:
- [CPU Usage Examples](03-cpu-examples.md) - Start with basic CPU training
- [GPU Usage Examples](04-gpu-examples.md) - Accelerate with GPU computing