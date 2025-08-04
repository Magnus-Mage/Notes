# GPU Usage Examples

This guide covers XGBoost GPU acceleration for faster training and inference.

## GPU Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 6.0+ (Pascal architecture or newer)
- At least 2GB GPU memory (4GB+ recommended)
- CUDA 11.8+ or 12.x

### Software Requirements
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Install GPU-enabled XGBoost
pip install xgboost[gpu]==2.1.0
```

## GPU Verification

```python
import xgboost as xgb
import numpy as np

def check_gpu_support():
    """Check if XGBoost was built with GPU support"""
    build_info = xgb.build_info()
    print("XGBoost Build Information:")
    print(f"  CUDA Support: {'USE_CUDA' in build_info}")
    print(f"  NCCL Support: {'USE_NCCL' in build_info}")
    
    # Test GPU availability
    try:
        # Create a simple test dataset
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)
        
        # Try GPU training
        dtrain = xgb.DMatrix(X, label=y)
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        model = xgb.train(params, dtrain, num_boost_round=1)
        print("✅ GPU training successful!")
        return True
    except Exception as e:
        print(f"❌ GPU training failed: {e}")
        return False

check_gpu_support()
```

## Basic GPU Training

### GPU Regression Example

```python
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# GPU-accelerated XGBoost regressor
gpu_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.1,
    tree_method='gpu_hist',  # GPU histogram algorithm
    gpu_id=0,  # Use first GPU
    random_state=42
)

# Time GPU training
start_time = time.time()
gpu_model.fit(X_train, y_train)
gpu_training_time = time.time() - start_time

# Make predictions
y_pred_gpu = gpu_model.predict(X_test)
gpu_mse = mean_squared_error(y_test, y_pred_gpu)

print(f"GPU Training Time: {gpu_training_time:.2f} seconds")
print(f"GPU MSE: {gpu_mse:.4f}")
```

### GPU vs CPU Performance Comparison

```python
def compare_gpu_cpu_performance(X_train, y_train, X_test, y_test):
    """Compare GPU vs CPU training performance"""
    
    # Common parameters
    common_params = {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    results = {}
    
    # CPU Training
    print("Training on CPU...")
    cpu_model = xgb.XGBRegressor(
        tree_method='hist',  # Fast CPU algorithm
        **common_params
    )
    
    start_time = time.time()
    cpu_model.fit(X_train, y_train)
    cpu_time = time.time() - start_time
    
    cpu_pred = cpu_model.predict(X_test)
    cpu_mse = mean_squared_error(y_test, cpu_pred)
    
    results['CPU'] = {
        'training_time': cpu_time,
        'mse': cpu_mse
    }
    
    # GPU Training
    print("Training on GPU...")
    gpu_model = xgb.XGBRegressor(
        tree_method='gpu_hist',  # GPU algorithm
        gpu_id=0,
        **common_params
    )
    
    start_time = time.time()
    gpu_model.fit(X_train, y_train)
    gpu_time = time.time() - start_time
    
    gpu_pred = gpu_model.predict(X_test)
    gpu_mse = mean_squared_error(y_test, gpu_pred)
    
    results['GPU'] = {
        'training_time': gpu_time,
        'mse': gpu_mse
    }
    
    # Display comparison
    speedup = cpu_time / gpu_time
    print(f"\n{'Method':<6} {'Time (s)':<10} {'MSE':<10} {'Speedup':<8}")
    print("-" * 40)
    print(f"{'CPU':<6} {cpu_time:<10.2f} {cpu_mse:<10.4f} {'1.0x':<8}")
    print(f"{'GPU':<6} {gpu_time:<10.2f} {gpu_mse:<10.4f} {speedup:<8.2f}x")
    
    return results

# Run comparison
results = compare_gpu_cpu_performance(X_train, y_train, X_test, y_test)
```

## Advanced GPU Configuration

### Multi-GPU Setup

```python
def multi_gpu_training():
    """Train using multiple GPUs"""
    import subprocess
    
    # Check available GPUs
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
        print(f"Available GPUs: {gpu_count}")
    except:
        print("nvidia-smi not available")
        gpu_count = 1
    
    if gpu_count > 1:
        # Multi-GPU training with Dask (requires dask-cuda)
        try:
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            import dask.dataframe as dd
            
            # Setup Dask cluster
            cluster = LocalCUDACluster()
            client = Client(cluster)
            
            # Convert to Dask DataFrame
            X_dask = dd.from_pandas(pd.DataFrame(X_train), npartitions=gpu_count)
            y_dask = dd.from_pandas(pd.Series(y_train), npartitions=gpu_count)
            
            # Train with multiple GPUs
            model = xgb.dask.DaskXGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.1,
                tree_method='gpu_hist'
            )
            
            model.fit(X_dask, y_dask)
            
            # Clean up
            client.close()
            cluster.close()
            
            print("Multi-GPU training completed!")
            return model
            
        except ImportError:
            print("Dask-CUDA not available. Install with: pip install dask-cuda")
    
    else:
        print("Single GPU training...")
        return xgb.XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            n_estimators=1000
        )

# multi_gpu_model = multi_gpu_training()
```

### GPU Memory Optimization

```python
def gpu_memory_optimization():
    """Optimize GPU memory usage for large datasets"""
    
    # Check GPU memory
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        memory_info = result.stdout.strip().split('\n')[0].split(', ')
        total_memory = int(memory_info[0])
        used_memory = int(memory_info[1])
        available_memory = total_memory - used_memory
        
        print(f"GPU Memory - Total: {total_memory}MB, Used: {used_memory}MB, Available: {available_memory}MB")
    except:
        print("Could not get GPU memory info")
        available_memory = 4000  # Assume 4GB
    
    # Optimize parameters based on available memory
    if available_memory > 8000:  # > 8GB
        max_depth = 10
        n_estimators = 2000
        subsample = 1.0
    elif available_memory > 4000:  # > 4GB
        max_depth = 8
        n_estimators = 1000
        subsample = 0.8
    else:  # < 4GB
        max_depth = 6
        n_estimators = 500
        subsample = 0.6
    
    # Memory-optimized GPU model
    model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=0.8,
        single_precision_histogram=True,  # Use less memory
        random_state=42
    )
    
    print(f"Optimized parameters: max_depth={max_depth}, n_estimators={n_estimators}, subsample={subsample}")
    return model

# Create memory-optimized model
optimized_model = gpu_memory_optimization()
```

## GPU-Specific Parameters

### Core GPU Parameters

```python
def demonstrate_gpu_parameters():
    """Show different GPU-specific parameters"""
    
    # Basic GPU configuration
    basic_gpu = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=100
    )
    
    # Advanced GPU configuration
    advanced_gpu = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        max_bin=256,  # Number of histogram bins (more = better accuracy, slower)
        single_precision_histogram=True,  # Use float32 (faster, less memory)
        deterministic_histogram=True,  # Reproducible results (slower)
        grow_policy='depthwise',  # Tree growing policy
        n_estimators=100
    )
    
    # Gradient-based one-side sampling (GOSS) - works well with GPU
    goss_gpu = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        boosting_type='goss',
        top_rate=0.2,  # Retain top 20% of gradients
        other_rate=0.1,  # Random sample 10% of remaining
        n_estimators=100
    )
    
    return basic_gpu, advanced_gpu, goss_gpu

basic, advanced, goss = demonstrate_gpu_parameters()
```

## GPU Inference Optimization

```python
def gpu_inference_benchmark():
    """Compare CPU vs GPU inference performance"""
    
    # Train models
    cpu_model = xgb.XGBRegressor(tree_method='hist', n_estimators=100)
    gpu_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=100)
    
    cpu_model.fit(X_train, y_train)
    gpu_model.fit(X_train, y_train)
    
    # Benchmark inference
    import time
    n_iterations = 100
    
    # CPU inference
    cpu_times = []
    for _ in range(n_iterations):
        start = time.time()
        _ = cpu_model.predict(X_test)
        cpu_times.append(time.time() - start)
    
    # GPU inference
    gpu_times = []
    for _ in range(n_iterations):
        start = time.time()
        _ = gpu_model.predict(X_test)
        gpu_times.append(time.time() - start)
    
    cpu_avg = np.mean(cpu_times) * 1000  # Convert to ms
    gpu_avg = np.mean(gpu_times) * 1000
    
    print(f"Average Inference Time:")
    print(f"  CPU: {cpu_avg:.2f} ms")
    print(f"  GPU: {gpu_avg:.2f} ms")
    print(f"  GPU Speedup: {cpu_avg/gpu_avg:.2f}x")

gpu_inference_benchmark()
```

## Large Dataset GPU Training

```python
def train_large_dataset_gpu(file_path=None):
    """Handle large datasets that don't fit in memory"""
    
    if file_path is None:
        # Create synthetic large dataset
        n_samples = 1000000  # 1M samples
        n_features = 100
        
        print(f"Creating synthetic dataset: {n_samples} samples, {n_features} features")
        X_large = np.random.randn(n_samples, n_features).astype(np.float32)
        y_large = np.random.randn(n_samples).astype(np.float32)
    else:
        # Load from file (chunked loading for very large files)
        import pandas as pd
        chunks = pd.read_csv(file_path, chunksize=100000)
        X_large = np.vstack([chunk.iloc[:, :-1].values for chunk in chunks])
        y_large = np.hstack([chunk.iloc[:, -1].values for chunk in chunks])
    
    # Convert to GPU-optimized DMatrix
    print("Converting to DMatrix...")
    dtrain = xgb.DMatrix(X_large, label=y_large)
    
    # GPU parameters optimized for large datasets
    params = {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'single_precision_histogram': True,
        'max_bin': 128  # Reduce bins for memory efficiency
    }
    
    print("Training large dataset on GPU...")
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        verbose_eval=10
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

# Train on large synthetic dataset
# large_model = train_large_dataset_gpu()
```

## GPU Cross-Validation

```python
def gpu_cross_validation():
    """Perform cross-validation using GPU"""
    
    # Convert to DMatrix for native XGBoost CV
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # GPU CV parameters
    params = {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    print("Running GPU cross-validation...")
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=10,
        verbose_eval=50,
        show_stdv=True,
        seed=42
    )
    
    print("\nCV Results Summary:")
    print(cv_results.tail())
    
    # Find best iteration
    best_iteration = cv_results.shape[0]
    best_score = cv_results.iloc[-1]['test-rmse-mean']
    
    print(f"Best iteration: {best_iteration}")
    print(f"Best CV score: {best_score:.4f}")
    
    return cv_results

# cv_results = gpu_cross_validation()
```

## Troubleshooting GPU Issues

```python
def diagnose_gpu_issues():
    """Diagnose common GPU-related issues"""
    
    print("🔍 GPU Diagnostics")
    print("=" * 50)
    
    # 1. Check CUDA availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
        else:
            print("❌ nvidia-smi failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - CUDA drivers may not be installed")
    
    # 2. Check XGBoost GPU build
    build_info = str(xgb.build_info())
    if 'USE_CUDA' in build_info:
        print("✅ XGBoost built with CUDA support")
    else:
        print("❌ XGBoost not built with CUDA support")
        print("   Install with: pip install xgboost[gpu]")
    
    # 3. Test GPU training
    try:
        X_test = np.random.randn(100, 10)
        y_test = np.random.randn(100)
        
        model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
        model.fit(X_test, y_test)
        print("✅ GPU training test successful")
    except Exception as e:
        print(f"❌ GPU training test failed: {e}")
    
    # 4. Memory check
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        memory = int(result.stdout.strip())
        if memory < 2000:
            print(f"⚠️  Limited GPU memory: {memory}MB (recommended: 4GB+)")
        else:
            print(f"✅ GPU memory: {memory}MB")
    except:
        print("⚠️  Could not check GPU memory")

diagnose_gpu_issues()
```

## Performance Tips

### GPU Optimization Checklist

1. **Use appropriate tree_method**: `gpu_hist` for GPU
2. **Set gpu_id**: Specify which GPU to use (0 for first GPU)
3. **Optimize max_bin**: Balance between accuracy and memory (64-512)
4. **Use single_precision_histogram**: For memory efficiency
5. **Adjust subsample**: Reduce for large datasets (0.6-0.8)
6. **Monitor memory usage**: Use nvidia-smi to check memory consumption

### Best Practices

```python
def gpu_best_practices_example():
    """Example showing GPU best practices"""
    
    # Optimal GPU configuration for most cases
    model = xgb.XGBRegressor(
        # GPU-specific parameters
        tree_method='gpu_hist',
        gpu_id=0,
        single_precision_histogram=True,
        
        # Performance parameters
        max_depth=6,
        n_estimators=1000,
        learning_rate=0.1,
        
        # Memory efficiency
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=256,
        
        # Reproducibility
        random_state=42
    )
    
    return model

optimal_gpu_model = gpu_best_practices_example()
```

## Next Steps

- For cloud GPU deployment, see [Cloud Providers Integration](05-cloud-providers.md)
- For model export formats, check [Model Export Formats](06-model-exports.md)
- For hyperparameter tuning on GPU, visit [Hyperparameter Tuning](08-hyperparameter-tuning.md)