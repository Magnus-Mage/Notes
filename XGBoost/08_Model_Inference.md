# Model Inference

This guide covers efficient XGBoost model inference patterns, optimization techniques, and deployment strategies for production environments.

## Basic Inference Patterns

### Single Prediction

```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import time

# Load and prepare sample data
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a sample model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Single sample prediction
def single_prediction_example():
    """Demonstrate single sample prediction"""
    
    # Single sample from test set
    single_sample = X_test[0:1]  # Keep as 2D array
    
    # Method 1: Direct prediction
    prediction = model.predict(single_sample)
    print(f"Single prediction: {prediction[0]:.4f}")
    
    # Method 2: With probability (for classification)
    if hasattr(model, 'predict_proba'):
        # This would be for classification models
        # proba = model.predict_proba(single_sample)
        pass
    
    # Method 3: Using native XGBoost API
    dmatrix = xgb.DMatrix(single_sample)
    native_pred = model.get_booster().predict(dmatrix)
    print(f"Native API prediction: {native_pred[0]:.4f}")
    
    return prediction[0]

single_pred = single_prediction_example()
```

### Batch Prediction

```python
def batch_prediction_patterns():
    """Demonstrate different batch prediction patterns"""
    
    batch_sizes = [1, 10, 100, 1000]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size <= len(X_test):
            batch_data = X_test[:batch_size]
            
            # Time batch prediction
            start_time = time.time()
            predictions = model.predict(batch_data)
            inference_time = time.time() - start_time
            
            results[batch_size] = {
                'predictions': predictions,
                'time': inference_time,
                'time_per_sample': inference_time / batch_size,
                'throughput': batch_size / inference_time
            }
            
            print(f"Batch size {batch_size:>4}: "
                  f"{inference_time*1000:>7.2f}ms total, "
                  f"{results[batch_size]['time_per_sample']*1000:>6.3f}ms/sample, "
                  f"{results[batch_size]['throughput']:>8.1f} samples/sec")
    
    return results

batch_results = batch_prediction_patterns()
```

### Streaming Inference

```python
import queue
import threading
from typing import Generator, List, Tuple

class StreamingPredictor:
    """Handle streaming inference with buffering and batching"""
    
    def __init__(self, model, batch_size=32, max_wait_time=0.1):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start the inference worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_worker)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the inference worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _inference_worker(self):
        """Background worker for batch inference"""
        batch_buffer = []
        batch_ids = []
        last_batch_time = time.time()
        
        while self.running or not self.input_queue.empty():
            try:
                # Try to get new item with timeout
                item_id, data = self.input_queue.get(timeout=0.01)
                batch_buffer.append(data)
                batch_ids.append(item_id)
                
                # Process batch if full or timeout exceeded
                current_time = time.time()
                should_process = (
                    len(batch_buffer) >= self.batch_size or
                    (batch_buffer and current_time - last_batch_time > self.max_wait_time)
                )
                
                if should_process:
                    self._process_batch(batch_buffer, batch_ids)
                    batch_buffer = []
                    batch_ids = []
                    last_batch_time = current_time
                    
            except queue.Empty:
                # Process remaining items if timeout exceeded
                if batch_buffer and time.time() - last_batch_time > self.max_wait_time:
                    self._process_batch(batch_buffer, batch_ids)
                    batch_buffer = []
                    batch_ids = []
                    last_batch_time = time.time()
                continue
        
        # Process any remaining items
        if batch_buffer:
            self._process_batch(batch_buffer, batch_ids)
    
    def _process_batch(self, batch_data: List, batch_ids: List):
        """Process a batch of data"""
        try:
            # Stack batch data
            batch_array = np.vstack(batch_data)
            
            # Make predictions
            predictions = self.model.predict(batch_array)
            
            # Put results in output queue
            for item_id, prediction in zip(batch_ids, predictions):
                self.output_queue.put((item_id, prediction))
                
        except Exception as e:
            # Handle errors gracefully
            for item_id in batch_ids:
                self.output_queue.put((item_id, f"Error: {str(e)}"))
    
    def predict_async(self, data: np.ndarray, item_id: str = None) -> str:
        """Submit data for asynchronous prediction"""
        if item_id is None:
            item_id = f"item_{time.time()}"
        
        self.input_queue.put((item_id, data))
        return item_id
    
    def get_result(self, timeout=None) -> Tuple[str, float]:
        """Get prediction result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

# Example usage
def streaming_inference_example():
    """Demonstrate streaming inference"""
    
    predictor = StreamingPredictor(model, batch_size=16, max_wait_time=0.05)
    predictor.start()
    
    # Submit predictions
    item_ids = []
    for i in range(50):
        data = X_test[i:i+1]
        item_id = predictor.predict_async(data, f"sample_{i}")
        item_ids.append(item_id)
        
        # Simulate streaming delay
        time.sleep(0.01)
    
    # Collect results
    results = {}
    for _ in range(50):
        item_id, prediction = predictor.get_result(timeout=1.0)
        if item_id:
            results[item_id] = prediction
    
    predictor.stop()
    
    print(f"Processed {len(results)} streaming predictions")
    return results

streaming_results = streaming_inference_example()
```

## Memory-Efficient Inference

### Large Dataset Inference

```python
def memory_efficient_inference(model, X_large, chunk_size=1000):
    """Handle inference on large datasets that don't fit in memory"""
    
    n_samples = len(X_large)
    all_predictions = []
    
    print(f"Processing {n_samples} samples in chunks of {chunk_size}")
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = X_large[start_idx:end_idx]
        
        # Process chunk
        chunk_predictions = model.predict(chunk)
        all_predictions.extend(chunk_predictions)
        
        # Progress update
        if (start_idx // chunk_size + 1) % 10 == 0:
            progress = (end_idx / n_samples) * 100
            print(f"Progress: {progress:.1f}% ({end_idx}/{n_samples})")
    
    return np.array(all_predictions)

# Example with large synthetic dataset
def large_dataset_example():
    """Create and process large dataset"""
    
    # Create large synthetic dataset
    n_large = 10000
    X_large = np.random.randn(n_large, X_train.shape[1])
    
    # Memory-efficient inference
    start_time = time.time()
    predictions = memory_efficient_inference(model, X_large, chunk_size=500)
    total_time = time.time() - start_time
    
    print(f"Large dataset inference:")
    print(f"  Samples: {len(predictions)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {len(predictions)/total_time:.1f} samples/sec")
    
    return predictions

large_predictions = large_dataset_example()
```

### Sparse Data Inference

```python
from scipy.sparse import csr_matrix, random as sparse_random

def sparse_inference_optimization():
    """Optimize inference for sparse data"""
    
    # Create sparse test data
    sparse_density = 0.1  # 10% non-zero elements
    n_samples, n_features = 1000, X_train.shape[1]
    
    X_sparse = sparse_random(n_samples, n_features, density=sparse_density, format='csr')
    
    # Convert dense test data to sparse for comparison
    X_test_sparse = csr_matrix(X_test)
    
    print(f"Sparse matrix density: {X_test_sparse.nnz / X_test_sparse.size:.3f}")
    print(f"Memory usage reduction: {1 - X_test_sparse.nnz / X_test_sparse.size:.2%}")
    
    # Benchmark dense vs sparse inference
    def benchmark_sparse_vs_dense():
        n_iterations = 100
        
        # Dense inference timing
        start_time = time.time()
        for _ in range(n_iterations):
            _ = model.predict(X_test[:100])
        dense_time = time.time() - start_time
        
        # Sparse inference timing
        start_time = time.time()
        for _ in range(n_iterations):
            _ = model.predict(X_test_sparse[:100])
        sparse_time = time.time() - start_time
        
        print(f"Dense inference: {dense_time*1000:.2f}ms")
        print(f"Sparse inference: {sparse_time*1000:.2f}ms")
        print(f"Speedup: {dense_time/sparse_time:.2f}x")
        
        return dense_time, sparse_time
    
    return benchmark_sparse_vs_dense()

dense_time, sparse_time = sparse_inference_optimization()
```

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

class InferenceFeatureProcessor:
    """Feature processing pipeline optimized for inference"""
    
    def __init__(self, numerical_features=None, categorical_features=None):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.preprocessor = None
        self.is_fitted = False
    
    def fit(self, X_train):
        """Fit preprocessing pipeline on training data"""
        
        transformers = []
        
        if self.numerical_features:
            transformers.append(
                ('num', StandardScaler(), self.numerical_features)
            )
        
        if self.categorical_features:
            # For categorical features (would need actual categorical data)
            from sklearn.preprocessing import OneHotEncoder
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            )
        
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            self.preprocessor.fit(X_train)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform features for inference"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.preprocessor:
            return self.preprocessor.transform(X)
        return X
    
    def transform_single(self, x):
        """Transform single sample (optimized)"""
        if isinstance(x, dict):
            # Convert dict to array format
            x = pd.DataFrame([x])
        elif isinstance(x, (list, np.ndarray)):
            x = np.array(x).reshape(1, -1)
        
        return self.transform(x)

class XGBoostInferencePipeline:
    """Complete inference pipeline with preprocessing"""
    
    def __init__(self, model, feature_processor=None):
        self.model = model
        self.feature_processor = feature_processor
    
    def predict(self, X):
        """Full pipeline prediction"""
        if self.feature_processor:
            X = self.feature_processor.transform(X)
        return self.model.predict(X)
    
    def predict_single(self, x):
        """Optimized single sample prediction"""
        if self.feature_processor:
            x = self.feature_processor.transform_single(x)
        return self.model.predict(x)[0]
    
    def predict_proba(self, X):
        """Probability prediction (for classification)"""
        if hasattr(self.model, 'predict_proba'):
            if self.feature_processor:
                X = self.feature_processor.transform(X)
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability prediction")

# Example usage
def inference_pipeline_example():
    """Demonstrate complete inference pipeline"""
    
    # Create feature processor (using all features as numerical for this example)
    feature_processor = InferenceFeatureProcessor(
        numerical_features=list(range(X_train.shape[1]))
    )
    feature_processor.fit(X_train)
    
    # Create inference pipeline
    inference_pipeline = XGBoostInferencePipeline(model, feature_processor)
    
    # Test different prediction methods
    batch_predictions = inference_pipeline.predict(X_test[:10])
    single_prediction = inference_pipeline.predict_single(X_test[0])
    
    print(f"Batch predictions shape: {batch_predictions.shape}")
    print(f"Single prediction: {single_prediction:.4f}")
    print(f"Pipeline predictions match model: {np.allclose(batch_predictions, model.predict(X_test[:10]))}")
    
    return inference_pipeline

inference_pipeline = inference_pipeline_example()
```

## Inference Optimization Techniques

### Model Compilation and Caching

```python
import joblib
import pickle
from functools import lru_cache

class OptimizedInferenceModel:
    """Optimized model wrapper for fast inference"""
    
    def __init__(self, model_path=None, model=None, cache_size=1000):
        if model_path:
            self.model = self._load_model(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Must provide either model_path or model")
        
        # Enable prediction caching for repeated inputs
        self.predict_cached = lru_cache(maxsize=cache_size)(self._predict_internal)
        
        # Pre-compile model (warm up)
        self._warm_up()
    
    def _load_model(self, model_path):
        """Load model with optimal format detection"""
        if model_path.endswith('.json'):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif model_path.endswith('.joblib'):
            return joblib.load(model_path)
        elif model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _warm_up(self):
        """Warm up model with dummy prediction"""
        dummy_input = np.random.randn(1, 13)  # Assuming 13 features
        _ = self.model.predict(dummy_input)
    
    def _predict_internal(self, X_tuple):
        """Internal prediction method for caching"""
        X = np.array(X_tuple).reshape(1, -1)
        return self.model.predict(X)[0]
    
    def predict(self, X, use_cache=True):
        """Optimized prediction with optional caching"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if use_cache and X.shape[0] == 1:
            # Use cached prediction for single samples
            return np.array([self.predict_cached(tuple(X[0]))])
        else:
            # Direct prediction for batch
            return self.model.predict(X)
    
    def get_cache_info(self):
        """Get cache statistics"""
        return self.predict_cached.cache_info()
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.predict_cached.cache_clear()

# Example usage
def optimization_example():
    """Demonstrate inference optimizations"""
    
    # Create optimized model
    opt_model = OptimizedInferenceModel(model=model, cache_size=1000)
    
    # Test with repeated predictions (should hit cache)
    repeated_sample = X_test[0]
    
    # Benchmark cached vs uncached
    n_iterations = 1000
    
    # Cached predictions
    start_time = time.time()
    for _ in range(n_iterations):
        _ = opt_model.predict(repeated_sample, use_cache=True)
    cached_time = time.time() - start_time
    
    # Clear cache and test uncached
    opt_model.clear_cache()
    start_time = time.time()
    for _ in range(n_iterations):
        _ = opt_model.predict(repeated_sample, use_cache=False)
    uncached_time = time.time() - start_time
    
    cache_info = opt_model.get_cache_info()
    
    print(f"Cached predictions: {cached_time*1000:.2f}ms")
    print(f"Uncached predictions: {uncached_time*1000:.2f}ms")
    print(f"Cache speedup: {uncached_time/cached_time:.2f}x")
    print(f"Cache info: {cache_info}")
    
    return opt_model

optimized_model = optimization_example()
```

### Parallel Inference

```python
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def parallel_inference_patterns():
    """Demonstrate different parallel inference approaches"""
    
    # Prepare large test dataset
    n_samples = 5000
    X_large = np.random.randn(n_samples, X_train.shape[1])
    chunk_size = 500
    chunks = [X_large[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]
    
    def predict_chunk(model_and_chunk):
        """Helper function for parallel prediction"""
        model, chunk = model_and_chunk
        return model.predict(chunk)
    
    # Sequential baseline
    start_time = time.time()
    sequential_results = [model.predict(chunk) for chunk in chunks]
    sequential_time = time.time() - start_time
    
    # Thread-based parallelism (I/O bound tasks)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        thread_results = list(executor.map(
            partial(predict_chunk),
            [(model, chunk) for chunk in chunks]
        ))
    thread_time = time.time() - start_time
    
    # Process-based parallelism (CPU bound tasks)
    # Note: This requires model to be pickleable
    try:
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()//2) as executor:
            process_results = list(executor.map(
                partial(predict_chunk),
                [(model, chunk) for chunk in chunks]
            ))
        process_time = time.time() - start_time
    except Exception as e:
        print(f"Process-based parallelism failed: {e}")
        process_time = float('inf')
    
    print(f"Inference Performance Comparison:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Threaded:   {thread_time:.2f}s (speedup: {sequential_time/thread_time:.2f}x)")
    if process_time != float('inf'):
        print(f"  Processes:  {process_time:.2f}s (speedup: {sequential_time/process_time:.2f}x)")
    
    return sequential_results, thread_results

sequential_results, thread_results = parallel_inferen
