# Model Export Formats

This guide covers different formats for saving and loading XGBoost models for various deployment scenarios.

## Native XGBoost Formats

### JSON Format (Recommended)

```python
import xgboost as xgb
import json
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Train a sample model
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Save as JSON (XGBoost native format)
model.save_model('xgboost_model.json')

# Load JSON model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('xgboost_model.json')

# Verify predictions match
original_pred = model.predict(X_test)
loaded_pred = loaded_model.predict(X_test)

print(f"Predictions match: {(original_pred == loaded_pred).all()}")
```

### Binary Format (.model)

```python
# Save as binary format (more compact)
model.save_model('xgboost_model.model')

# Load binary model
loaded_binary_model = xgb.XGBRegressor()
loaded_binary_model.load_model('xgboost_model.model')

# Check file sizes
import os
json_size = os.path.getsize('xgboost_model.json')
binary_size = os.path.getsize('xgboost_model.model')

print(f"JSON size: {json_size} bytes")
print(f"Binary size: {binary_size} bytes")
print(f"Binary is {json_size/binary_size:.1f}x smaller")
```

### Native XGBoost API Format

```python
# Using native XGBoost API for more control
dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train native model
native_model = xgb.train(params, dtrain, num_boost_round=100)

# Save native model
native_model.save_model('native_xgboost.json')

# Load native model
loaded_native = xgb.Booster()
loaded_native.load_model('native_xgboost.json')

# Make predictions with native model
dtest = xgb.DMatrix(X_test)
native_predictions = loaded_native.predict(dtest)
```

## Joblib Serialization

```python
import joblib
import pickle

# Save entire model with joblib (includes preprocessing pipelines)
joblib.dump(model, 'xgboost_model.joblib')

# Load joblib model
loaded_joblib_model = joblib.load('xgboost_model.joblib')

# Joblib is great for sklearn pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=6))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'xgboost_pipeline.joblib')

# Load and use pipeline
loaded_pipeline = joblib.load('xgboost_pipeline.joblib')
pipeline_predictions = loaded_pipeline.predict(X_test)

print(f"Pipeline predictions shape: {pipeline_predictions.shape}")
```

### Joblib vs Pickle Comparison

```python
import time

# Compare joblib vs pickle performance
def benchmark_serialization():
    methods = {
        'joblib': (joblib.dump, joblib.load),
        'pickle': (pickle.dump, pickle.load)
    }
    
    results = {}
    
    for method_name, (dump_func, load_func) in methods.items():
        # Save timing
        start_time = time.time()
        if method_name == 'joblib':
            dump_func(model, f'model_{method_name}.pkl')
        else:
            with open(f'model_{method_name}.pkl', 'wb') as f:
                dump_func(model, f)
        save_time = time.time() - start_time
        
        # Load timing
        start_time = time.time()
        if method_name == 'joblib':
            loaded = load_func(f'model_{method_name}.pkl')
        else:
            with open(f'model_{method_name}.pkl', 'rb') as f:
                loaded = load_func(f)
        load_time = time.time() - start_time
        
        # File size
        file_size = os.path.getsize(f'model_{method_name}.pkl')
        
        results[method_name] = {
            'save_time': save_time,
            'load_time': load_time,
            'file_size': file_size
        }
    
    return results

# Run benchmark
benchmark_results = benchmark_serialization()
for method, metrics in benchmark_results.items():
    print(f"{method.upper()}:")
    print(f"  Save time: {metrics['save_time']:.4f}s")
    print(f"  Load time: {metrics['load_time']:.4f}s")
    print(f"  File size: {metrics['file_size']} bytes")
```

## ONNX Export

### Basic ONNX Export

```python
# Install required packages first:
# pip install onnx onnxmltools skl2onnx

try:
    import onnx
    import onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.utils import save_model as save_onnx_model
    
    def export_to_onnx(xgb_model, X_sample, output_path):
        """Export XGBoost model to ONNX format"""
        
        # Convert to ONNX
        onnx_model = convert_xgboost(
            xgb_model,
            initial_types=[('input', onnx.TensorProto.FLOAT, X_sample.shape)]
        )
        
        # Save ONNX model
        save_onnx_model(onnx_model, output_path)
        
        return onnx_model
    
    # Export model to ONNX
    onnx_model = export_to_onnx(model, X_test[:1], 'xgboost_model.onnx')
    print("Model exported to ONNX successfully!")
    
except ImportError:
    print("ONNX export requires: pip install onnx onnxmltools skl2onnx")
```

### ONNX Runtime Inference

```python
try:
    import onnxruntime as ort
    import numpy as np
    
    def onnx_inference(onnx_model_path, X_test):
        """Run inference using ONNX Runtime"""
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_model_path)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        predictions = session.run(
            [output_name],
            {input_name: X_test.astype(np.float32)}
        )[0]
        
        return predictions
    
    # Run ONNX inference
    onnx_predictions = onnx_inference('xgboost_model.onnx', X_test)
    
    # Compare with original predictions
    original_predictions = model.predict(X_test)
    print(f"ONNX predictions match: {np.allclose(onnx_predictions.flatten(), original_predictions)}")
    
except ImportError:
    print("ONNX Runtime requires: pip install onnxruntime")
```

### ONNX Performance Comparison

```python
def compare_onnx_performance():
    """Compare ONNX vs native XGBoost inference performance"""
    
    try:
        import onnxruntime as ort
        
        # Prepare test data
        test_samples = 1000
        X_perf_test = np.random.randn(test_samples, X_train.shape[1]).astype(np.float32)
        
        # Native XGBoost timing
        start_time = time.time()
        for _ in range(100):
            _ = model.predict(X_perf_test)
        native_time = time.time() - start_time
        
        # ONNX timing
        session = ort.InferenceSession('xgboost_model.onnx')
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        start_time = time.time()
        for _ in range(100):
            _ = session.run([output_name], {input_name: X_perf_test})[0]
        onnx_time = time.time() - start_time
        
        print(f"Native XGBoost: {native_time:.4f}s")
        print(f"ONNX Runtime: {onnx_time:.4f}s")
        print(f"ONNX Speedup: {native_time/onnx_time:.2f}x")
        
    except ImportError:
        print("ONNX performance test requires onnxruntime")

compare_onnx_performance()
```

## Cloud-Specific Formats

### AWS SageMaker Format

```python
def export_for_sagemaker():
    """Export model for AWS SageMaker deployment"""
    
    import tarfile
    import boto3
    
    # Save model in SageMaker format
    model.save_model('model.tar.gz')
    
    # Create tar.gz archive (SageMaker requirement)
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add('xgboost_model.json', arcname='xgboost-model')
    
    # Upload to S3 (optional)
    try:
        s3 = boto3.client('s3')
        bucket_name = 'your-sagemaker-bucket'
        s3.upload_file('model.tar.gz', bucket_name, 'models/xgboost-model.tar.gz')
        print(f"Model uploaded to s3://{bucket_name}/models/xgboost-model.tar.gz")
    except Exception as e:
        print(f"S3 upload failed: {e}")
    
    return 'model.tar.gz'

# sagemaker_model = export_for_sagemaker()
```

### Google Cloud AI Platform Format

```python
def export_for_gcp():
    """Export model for Google Cloud AI Platform"""
    
    from google.cloud import storage
    import os
    
    # Save in GCP-compatible format
    model.save_model('xgboost_model.bst')
    
    # Create model version directory structure
    os.makedirs('model/1', exist_ok=True)
    
    # Copy model to versioned directory
    import shutil
    shutil.copy('xgboost_model.bst', 'model/1/model.bst')
    
    # Upload to Google Cloud Storage (optional)
    try:
        client = storage.Client()
        bucket = client.bucket('your-gcp-bucket')
        
        blob = bucket.blob('models/xgboost/1/model.bst')
        blob.upload_from_filename('model/1/model.bst')
        
        print("Model uploaded to GCS")
    except Exception as e:
        print(f"GCS upload failed: {e}")
    
    return 'model/1/model.bst'

# gcp_model = export_for_gcp()
```

### Azure Machine Learning Format

```python
def export_for_azure():
    """Export model for Azure Machine Learning"""
    
    try:
        from azureml.core import Model, Workspace
        
        # Save model locally
        model.save_model('azure_xgboost_model.json')
        
        # Register model in Azure ML (requires Azure ML workspace)
        # ws = Workspace.from_config()
        # azure_model = Model.register(
        #     workspace=ws,
        #     model_path='azure_xgboost_model.json',
        #     model_name='xgboost-model',
        #     description='XGBoost regression model'
        # )
        
        print("Model prepared for Azure ML deployment")
        return 'azure_xgboost_model.json'
        
    except ImportError:
        print("Azure ML export requires: pip install azureml-sdk")
        return None

# azure_model = export_for_azure()
```

## Format Comparison

```python
def compare_all_formats():
    """Compare all export formats"""
    
    formats = []
    
    # Test each format
    test_predictions = model.predict(X_test[:10])  # First 10 samples for testing
    
    # JSON format
    model.save_model('test_json.json')
    json_model = xgb.XGBRegressor()
    json_model.load_model('test_json.json')
    json_pred = json_model.predict(X_test[:10])
    json_size = os.path.getsize('test_json.json')
    
    formats.append({
        'format': 'JSON',
        'file_size': json_size,
        'predictions_match': np.allclose(test_predictions, json_pred),
        'cross_platform': True,
        'human_readable': True,
        'use_case': 'General deployment, debugging'
    })
    
    # Binary format
    model.save_model('test_binary.model')
    binary_model = xgb.XGBRegressor()
    binary_model.load_model('test_binary.model')
    binary_pred = binary_model.predict(X_test[:10])
    binary_size = os.path.getsize('test_binary.model')
    
    formats.append({
        'format': 'Binary',
        'file_size': binary_size,
        'predictions_match': np.allclose(test_predictions, binary_pred),
        'cross_platform': True,
        'human_readable': False,
        'use_case': 'Production, storage efficiency'
    })
    
    # Joblib format
    joblib.dump(model, 'test_joblib.pkl')
    joblib_model = joblib.load('test_joblib.pkl')
    joblib_pred = joblib_model.predict(X_test[:10])
    joblib_size = os.path.getsize('test_joblib.pkl')
    
    formats.append({
        'format': 'Joblib',
        'file_size': joblib_size,
        'predictions_match': np.allclose(test_predictions, joblib_pred),
        'cross_platform': False,
        'human_readable': False,
        'use_case': 'Python-only, with preprocessing pipelines'
    })
    
    # Display comparison table
    import pandas as pd
    
    df = pd.DataFrame(formats)
    print("Format Comparison:")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"Format: {row['format']}")
        print(f"  File Size: {row['file_size']} bytes")
        print(f"  Predictions Match: {row['predictions_match']}")
        print(f"  Cross-Platform: {row['cross_platform']}")
        print(f"  Human Readable: {row['human_readable']}")
        print(f"  Use Case: {row['use_case']}")
        print()
    
    return df

comparison_df = compare_all_formats()
```

## Advanced Export Options

### Model Metadata and Versioning

```python
import json
from datetime import datetime

def export_with_metadata(model, model_path, metadata=None):
    """Export model with comprehensive metadata"""
    
    # Default metadata
    default_metadata = {
        'model_type': 'XGBoost',
        'version': xgb.__version__,
        'created_at': datetime.now().isoformat(),
        'framework': 'scikit-learn' if hasattr(model, 'fit') else 'native',
        'objective': getattr(model, 'objective', 'unknown'),
        'n_estimators': getattr(model, 'n_estimators', 'unknown'),
        'max_depth': getattr(model, 'max_depth', 'unknown'),
        'learning_rate': getattr(model, 'learning_rate', 'unknown')
    }
    
    # Merge with provided metadata
    if metadata:
        default_metadata.update(metadata)
    
    # Save model
    model.save_model(model_path)
    
    # Save metadata
    metadata_path = model_path.replace('.json', '_metadata.json').replace('.model', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(default_metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path, metadata_path

# Export with custom metadata
custom_metadata = {
    'dataset': 'Boston Housing',
    'features': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
    'target': 'MEDV',
    'training_samples': len(X_train),
    'validation_score': 0.85,
    'author': 'Data Science Team'
}

model_path, metadata_path = export_with_metadata(
    model, 
    'production_model.json', 
    custom_metadata
)
```

### Feature Importance Export

```python
def export_feature_importance(model, feature_names, output_path):
    """Export feature importance as separate file"""
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance dictionary
    importance_dict = {
        'feature_importance': dict(zip(feature_names, importance.tolist())),
        'importance_type': 'gain',  # XGBoost default
        'sorted_features': sorted(
            zip(feature_names, importance), 
            key=lambda x: x[1], 
            reverse=True
        )
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print(f"Feature importance saved to: {output_path}")
    return importance_dict

# Export feature importance
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
importance_data = export_feature_importance(
    model, 
    feature_names, 
    'feature_importance.json'
)
```

### Multi-format Export Pipeline

```python
def export_all_formats(model, base_name, X_sample):
    """Export model in all supported formats"""
    
    export_results = {}
    
    # JSON format
    json_path = f"{base_name}.json"
    model.save_model(json_path)
    export_results['json'] = {
        'path': json_path,
        'size': os.path.getsize(json_path),
        'type': 'XGBoost native'
    }
    
    # Binary format
    binary_path = f"{base_name}.model"
    model.save_model(binary_path)
    export_results['binary'] = {
        'path': binary_path,
        'size': os.path.getsize(binary_path),
        'type': 'XGBoost native'
    }
    
    # Joblib format
    joblib_path = f"{base_name}.joblib"
    joblib.dump(model, joblib_path)
    export_results['joblib'] = {
        'path': joblib_path,
        'size': os.path.getsize(joblib_path),
        'type': 'Python pickle'
    }
    
    # Pickle format
    pickle_path = f"{base_name}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    export_results['pickle'] = {
        'path': pickle_path,
        'size': os.path.getsize(pickle_path),
        'type': 'Python pickle'
    }
    
    # ONNX format (if available)
    try:
        import onnxmltools
        onnx_path = f"{base_name}.onnx"
        onnx_model = export_to_onnx(model, X_sample, onnx_path)
        export_results['onnx'] = {
            'path': onnx_path,
            'size': os.path.getsize(onnx_path),
            'type': 'ONNX'
        }
    except ImportError:
        print("ONNX export skipped (onnxmltools not available)")
    
    # Display results
    print("Export Summary:")
    print("=" * 60)
    for format_name, info in export_results.items():
        print(f"{format_name.upper():<8} | {info['size']:>8} bytes | {info['type']}")
    
    return export_results

# Export model in all formats
all_exports = export_all_formats(model, 'multi_format_model', X_test[:1])
```

## Model Validation After Export

```python
def validate_exported_models(original_model, X_test, export_results):
    """Validate that all exported models produce identical results"""
    
    # Get original predictions
    original_predictions = original_model.predict(X_test)
    
    validation_results = {}
    
    # Test JSON format
    if 'json' in export_results:
        json_model = xgb.XGBRegressor()
        json_model.load_model(export_results['json']['path'])
        json_predictions = json_model.predict(X_test)
        validation_results['json'] = np.allclose(original_predictions, json_predictions)
    
    # Test binary format
    if 'binary' in export_results:
        binary_model = xgb.XGBRegressor()
        binary_model.load_model(export_results['binary']['path'])
        binary_predictions = binary_model.predict(X_test)
        validation_results['binary'] = np.allclose(original_predictions, binary_predictions)
    
    # Test joblib format
    if 'joblib' in export_results:
        joblib_model = joblib.load(export_results['joblib']['path'])
        joblib_predictions = joblib_model.predict(X_test)
        validation_results['joblib'] = np.allclose(original_predictions, joblib_predictions)
    
    # Test pickle format
    if 'pickle' in export_results:
        with open(export_results['pickle']['path'], 'rb') as f:
            pickle_model = pickle.load(f)
        pickle_predictions = pickle_model.predict(X_test)
        validation_results['pickle'] = np.allclose(original_predictions, pickle_predictions)
    
    # Test ONNX format
    if 'onnx' in export_results:
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(export_results['onnx']['path'])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            onnx_predictions = session.run(
                [output_name], 
                {input_name: X_test.astype(np.float32)}
            )[0].flatten()
            validation_results['onnx'] = np.allclose(original_predictions, onnx_predictions, rtol=1e-5)
        except ImportError:
            validation_results['onnx'] = "ONNX Runtime not available"
    
    # Display validation results
    print("\nValidation Results:")
    print("=" * 40)
    for format_name, is_valid in validation_results.items():
        status = "✅ PASS" if is_valid is True else "❌ FAIL" if is_valid is False else "⚠️ SKIP"
        print(f"{format_name.upper():<8} | {status}")
    
    return validation_results

# Validate all exported models
validation_results = validate_exported_models(model, X_test, all_exports)
```

## Production Deployment Helpers

```python
def create_deployment_package(model, model_name, format_type='json'):
    """Create a complete deployment package"""
    
    import zipfile
    import shutil
    from pathlib import Path
    
    # Create deployment directory
    deploy_dir = Path(f"deployment_{model_name}")
    deploy_dir.mkdir(exist_ok=True)
    
    # Save model in specified format
    if format_type == 'json':
        model_path = deploy_dir / f"{model_name}.json"
        model.save_model(str(model_path))
    elif format_type == 'binary':
        model_path = deploy_dir / f"{model_name}.model"
        model.save_model(str(model_path))
    elif format_type == 'joblib':
        model_path = deploy_dir / f"{model_name}.joblib"
        joblib.dump(model, str(model_path))
    
    # Create inference script
    inference_script = f"""
import xgboost as xgb
import numpy as np
import json

class XGBoostPredictor:
    def __init__(self, model_path):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
    
    def predict(self, X):
        '''Make predictions on input data'''
        return self.model.predict(X)
    
    def predict_batch(self, X_batch):
        '''Make batch predictions'''
        return self.model.predict(X_batch)

# Example usage
if __name__ == "__main__":
    predictor = XGBoostPredictor('{model_path.name}')
    
    # Example prediction
    sample_input = np.random.randn(1, {X_train.shape[1]})
    prediction = predictor.predict(sample_input)
    print(f"Prediction: {{prediction[0]}}")
"""
    
    with open(deploy_dir / "predictor.py", 'w') as f:
        f.write(inference_script)
    
    # Create requirements.txt
    requirements = [
        f"xgboost=={xgb.__version__}",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0"
    ]
    
    if format_type == 'joblib':
        requirements.append("joblib>=1.0.0")
    
    with open(deploy_dir / "requirements.txt", 'w') as f:
        f.write('\n'.join(requirements))
    
    # Create README
    readme_content = f"""
# {model_name} Deployment Package

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from predictor import XGBoostPredictor

predictor = XGBoostPredictor('{model_path.name}')
predictions = predictor.predict(your_data)
```

## Model Information
- Format: {format_type}
- XGBoost Version: {xgb.__version__}
- Model File: {model_path.name}
"""
    
    with open(deploy_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Create ZIP package
    zip_path = f"{model_name}_deployment.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in deploy_dir.rglob('*'):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(deploy_dir))
    
    print(f"Deployment package created: {zip_path}")
    print(f"Package contents: {list(deploy_dir.glob('*'))}")
    
    return zip_path, deploy_dir

# Create deployment package
deployment_zip, deployment_dir = create_deployment_package(
    model, 
    "boston_housing_model", 
    format_type='json'
)
```

## Format Selection Guide

```python
def recommend_format(use_case):
    """Recommend best format based on use case"""
    
    recommendations = {
        'development': {
            'format': 'JSON',
            'reason': 'Human readable, easy debugging, cross-platform'
        },
        'production_python': {
            'format': 'Binary',
            'reason': 'Compact size, fast loading, native XGBoost'
        },
        'production_cross_platform': {
            'format': 'ONNX',
            'reason': 'Cross-language support, optimized runtime'
        },
        'with_preprocessing': {
            'format': 'Joblib',
            'reason': 'Supports sklearn pipelines and preprocessing'
        },
        'cloud_deployment': {
            'format': 'JSON',
            'reason': 'Platform agnostic, easy integration'
        },
        'edge_deployment': {
            'format': 'ONNX',
            'reason': 'Optimized inference, minimal dependencies'
        },
        'model_serving': {
            'format': 'Binary',
            'reason': 'Fast loading, compact storage'
        }
    }
    
    if use_case in recommendations:
        rec = recommendations[use_case]
        print(f"For {use_case}:")
        print(f"  Recommended format: {rec['format']}")
        print(f"  Reason: {rec['reason']}")
        return rec['format']
    else:
        print("Available use cases:")
        for case in recommendations.keys():
            print(f"  - {case}")
        return None

# Get recommendations
recommend_format('production_python')
recommend_format('cloud_deployment')
recommend_format('edge_deployment')
```

## Clean Up

```python
def cleanup_export_files():
    """Clean up all generated export files"""
    
    import glob
    
    # List of patterns to clean up
    cleanup_patterns = [
        '*.json',
        '*.model', 
        '*.joblib',
        '*.pkl',
        '*.onnx',
        '*_metadata.json',
        'feature_importance.json',
        '*_deployment.zip',
        'deployment_*'
    ]
    
    cleaned_files = []
    
    for pattern in cleanup_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    cleaned_files.append(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    cleaned_files.append(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    print(f"Cleaned up {len(cleaned_files)} files/directories")
    return cleaned_files

# Uncomment to clean up generated files
# cleanup_export_files()
```

## Next Steps

- For advanced boosting techniques, see [Advanced Boosting Techniques](06_Advanced_Boosting.md)
- For hyperparameter tuning, check [Hyperparameter Tuning](08-hyperparameter-tuning.md)
- For production inference, visit [Model Inference](09-model-inference.md)
