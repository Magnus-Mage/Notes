# Cloud Providers Integration

This guide covers deploying and running XGBoost on major cloud platforms.

## AWS (Amazon Web Services)

### AWS SageMaker

#### Built-in XGBoost Algorithm

```python
import boto3
import sagemaker
from sagemaker.xgboost import XGBoost

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# XGBoost estimator using SageMaker's built-in algorithm
xgb_estimator = XGBoost(
    entry_point='train.py',  # Your training script
    framework_version='1.7-1',  # XGBoost version
    py_version='py3',
    instance_type='ml.m5.xlarge',
    instance_count=1,
    output_path='s3://your-bucket/output',
    role=role,
    hyperparameters={
        'max_depth': 5,
        'eta': 0.2,
        'gamma': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'objective': 'reg:squarederror',
        'num_round': 100
    }
)

# Train the model
xgb_estimator.fit({'train': 's3://your-bucket/train.csv'})
```

#### Custom XGBoost Training Script (train.py)

```python
import argparse
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--num_round', type=int, default=100)
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(f"{args.train}/train.csv", header=None)
    
    # Prepare features and target
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Set parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round
    )
    
    # Save model
    model.save_model(f"{args.model_dir}/xgboost-model")
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
```

#### SageMaker Deployment

```python
# Deploy the trained model
predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Make predictions
import numpy as np
test_data = np.random.randn(10, 13)  # 10 samples, 13 features
predictions = predictor.predict(test_data)

# Clean up
predictor.delete_endpoint()
```

### AWS EC2 with GPU

```python
# EC2 GPU instance setup script
"""
# Install CUDA and XGBoost on EC2 GPU instance
sudo apt update
sudo apt install -y python3-pip

# Install CUDA (example for Ubuntu 20.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda

# Install XGBoost with GPU support
pip3 install xgboost[gpu]
"""

# Python code for EC2 GPU training
import xgboost as xgb
import boto3

def train_on_ec2_gpu(s3_bucket, data_key):
    # Download data from S3
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, data_key, 'training_data.csv')
    
    # Load and prepare data
    import pandas as pd
    data = pd.read_csv('training_data.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # GPU training
    model = xgb.XGBRegressor(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1
    )
    
    model.fit(X, y)
    
    # Save model to S3
    model.save_model('model.json')
    s3.upload_file('model.json', s3_bucket, 'models/xgboost_model.json')
    
    return model
```

## Google Cloud Platform (GCP)

### Vertex AI (AI Platform)

#### Custom Training Job

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

# Initialize Vertex AI
aiplatform.init(project='your-project-id', location='us-central1')

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='xgboost-training',
    script_path='trainer/train.py',
    container_uri='gcr.io/cloud-aiplatform/training/xgboost-cpu.1-4:latest',
    requirements=['xgboost==2.1.0', 'pandas', 'scikit-learn'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-4:latest'
)

# Run training
model = job.run(
    dataset=None,
    replica_count=1,
    machine_type='n1-standard-4',
    args=['--epochs=100', '--learning-rate=0.1']
)
```

#### Training Script for Vertex AI (trainer/train.py)

```python
import argparse
import os
import pandas as pd
import xgboost as xgb
from google.cloud import storage
import joblib

def download_data_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download data from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload model to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--data-bucket', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-bucket', type=str, required=True)
    
    args = parser.parse_args()
    
    # Download training data
    download_data_from_gcs(args.data_bucket, args.data_path, 'train_data.csv')
    
    # Load and prepare data
    data = pd.read_csv('train_data.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=args.epochs,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save model
    model.save_model('xgboost_model.json')
    
    # Upload model to GCS
    upload_model_to_gcs(
        args.model_bucket, 
        'xgboost_model.json', 
        'models/xgboost_model.json'
    )
    
    print("Training completed and model uploaded to GCS!")

if __name__ == '__main__':
    main()
```

### Google Colab with GPU

```python
# Google Colab GPU setup
"""
1. Go to Runtime > Change runtime type
2. Select GPU as hardware accelerator
3. Run the following code:
"""

# Install XGBoost with GPU support
!pip install xgboost

# Verify GPU availability
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# XGBoost GPU training in Colab
import xgboost as xgb
import numpy as np

# Generate sample data
X = np.random.randn(10000, 20)
y = np.random.randn(10000)

# Train with GPU
model_gpu = xgb.XGBRegressor(
    tree_method='gpu_hist',
    gpu_id=0,
    n_estimators=1000,
    max_depth=8
)

model_gpu.fit(X, y)
print("GPU training completed in Google Colab!")
```

## Microsoft Azure

### Azure Machine Learning

#### Azure ML Training Script

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to workspace
ws = Workspace.from_config()

# Create or get compute target
compute_name = 'xgboost-cluster'
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC6',  # GPU instance
        max_nodes=1
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Create environment
env = Environment.from_pip_requirements('xgboost-env', 'requirements.txt')
env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04'

# Configure training run
src = ScriptRunConfig(
    source_directory='.',
    script='train_azure.py',
    compute_target=compute_target,
    environment=env,
    arguments=[
        '--data-path', 'datasets/training_data',
        '--epochs', 1000,
        '--learning-rate', 0.1
    ]
)

# Submit experiment
experiment = Experiment(ws, 'xgboost-experiment')
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
```

#### Azure Training Script (train_azure.py)

```python
import argparse
import os
import pandas as pd
import xgboost as xgb
from azureml.core import Run
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--max-depth', type=int, default=6)
    
    args = parser.parse_args()
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Load data
    data_path = os.path.join(args.data_path, 'train.csv')
    data = pd.read_csv(data_path)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Train model with GPU if available
    try:
        model = xgb.XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            n_estimators=args.epochs,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42
        )
        print("Training with GPU")
    except:
        model = xgb.XGBRegressor(
            tree_method='hist',
            n_estimators=args.epochs,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42
        )
        print("Training with CPU")
    
    # Train the model
    model.fit(X, y)
    
    # Log metrics to Azure ML
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    run.log('mse', mse)
    run.log('r2_score', r2)
    
    # Save model
    os.makedirs('outputs', exist_ok=True)
    model.save_model('outputs/xgboost_model.json')
    
    print(f"Training completed. MSE: {mse:.4f}, R²: {r2:.4f}")

if __name__ == '__main__':
    main()
```

## Other Cloud Providers

### Databricks

```python
# Databricks notebook cell
# Install XGBoost with GPU support
%pip install xgboost

# MLflow integration for model tracking
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from Databricks File System (DBFS)
data = spark.read.format("csv").option("header", "true").load("/FileStore/shared_uploads/training_data.csv")
pandas_df = data.toPandas()

X = pandas_df.drop('target', axis=1)
y = pandas_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("mse", mse)
    
    # Log model
    mlflow.xgboost.log_model(model, "xgboost_model")
    
    print(f"Model MSE: {mse:.4f}")
```

### Kubernetes Deployment

#### Kubernetes YAML Configuration

```yaml
# xgboost-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xgboost-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: xgboost-server
  template:
    metadata:
      labels:
        app: xgboost-server
    spec:
      containers:
      - name: xgboost-container
        image: your-registry/xgboost-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/app/models/xgboost_model.json"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: xgboost-service
spec:
  selector:
    app: xgboost-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Kubernetes Inference Server (Flask)

```python
# app.py - XGBoost inference server
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

# Load model on startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'xgboost_model.json')
model = xgb.Booster()
model.load_model(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert to numpy array
        features = np.array(data['features']).reshape(1, -1)
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(features)
        
        # Make prediction
        prediction = model.predict(dmatrix)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        
        dmatrix = xgb.DMatrix(features)
        predictions = model.predict(dmatrix)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'count': len(predictions),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

#### Dockerfile for Kubernetes

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

## Cloud-Specific Optimizations

### AWS Optimization

```python
def optimize_for_aws():
    """AWS-specific optimizations"""
    
    # Use AWS instance storage for temporary files
    import tempfile
    temp_dir = '/tmp' if os.path.exists('/tmp') else tempfile.gettempdir()
    
    # Optimize for AWS instance types
    instance_type = os.environ.get('AWS_INSTANCE_TYPE', 'unknown')
    
    if 'p3' in instance_type or 'p4' in instance_type:  # GPU instances
        config = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 8,
            'n_estimators': 2000
        }
    elif 'c5' in instance_type:  # CPU optimized
        config = {
            'tree_method': 'hist',
            'n_jobs': -1,
            'max_depth': 6,
            'n_estimators': 1000
        }
    else:  # General purpose
        config = {
            'tree_method': 'hist',
            'max_depth': 6,
            'n_estimators': 500
        }
    
    return config

aws_config = optimize_for_aws()
```

### GCP Optimization

```python
def optimize_for_gcp():
    """GCP-specific optimizations"""
    
    # Detect GCP machine type
    try:
        import requests
        metadata = requests.get(
            'http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
            headers={'Metadata-Flavor': 'Google'},
            timeout=1
        ).text
        machine_type = metadata.split('/')[-1]
    except:
        machine_type = 'unknown'
    
    # Configure based on machine type
    if 'gpu' in machine_type:
        config = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'single_precision_histogram': True
        }
    elif 'highmem' in machine_type:
        config = {
            'tree_method': 'hist',
            'max_depth': 10,
            'n_estimators': 2000
        }
    else:
        config = {
            'tree_method': 'approx',
            'max_depth': 6,
            'subsample': 0.8
        }
    
    return config

gcp_config = optimize_for_gcp()
```

### Azure Optimization

```python
def optimize_for_azure():
    """Azure-specific optimizations"""
    
    # Use Azure ML context if available
    try:
        from azureml.core import Run
        run = Run.get_context()
        
        # Log metrics during training
        def log_metric(name, value):
            run.log(name, value)
            
        config = {
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'early_stopping_rounds': 10,
            'callbacks': [lambda env: log_metric('train_rmse', env.evaluation_result_list[0][1])]
        }
    except ImportError:
        # Fallback configuration
        config = {
            'tree_method': 'hist',
            'max_depth': 6,
            'n_estimators': 1000
        }
    
    return config

azure_config = optimize_for_azure()
```

## Cost Optimization Strategies

### Spot Instances / Preemptible VMs

```python
def handle_preemption():
    """Handle preemptible instance interruptions"""
    
    import signal
    import pickle
    import time
    
    class PreemptionHandler:
        def __init__(self, model, checkpoint_path):
            self.model = model
            self.checkpoint_path = checkpoint_path
            self.interrupted = False
            
            # Register signal handlers
            signal.signal(signal.SIGTERM, self._handle_preemption)
            signal.signal(signal.SIGINT, self._handle_preemption)
        
        def _handle_preemption(self, signum, frame):
            print("Preemption signal received. Saving checkpoint...")
            self.interrupted = True
            self._save_checkpoint()
        
        def _save_checkpoint(self):
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Checkpoint saved to {self.checkpoint_path}")
        
        def _load_checkpoint(self):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return None
    
    # Usage example
    checkpoint_path = '/tmp/xgboost_checkpoint.pkl'
    handler = PreemptionHandler(None, checkpoint_path)
    
    # Try to load existing checkpoint
    model = handler._load_checkpoint()
    if model is None:
        model = xgb.XGBRegressor(n_estimators=1000, max_depth=6)
        print("Starting training from scratch")
    else:
        print("Resuming training from checkpoint")
    
    # Continue training with periodic checkpoints
    return model, handler

# Example usage
# model, handler = handle_preemption()
```

### Auto-scaling Configuration

```python
def create_autoscaling_config():
    """Create auto-scaling configuration for cloud deployment"""
    
    config = {
        'min_replicas': 1,
        'max_replicas': 10,
        'target_cpu_utilization': 70,
        'target_memory_utilization': 80,
        
        # Scale-up settings
        'scale_up_stabilization_window': 60,  # seconds
        'scale_up_policies': [
            {
                'type': 'percent',
                'value': 100,  # Double the replicas
                'period_seconds': 15
            }
        ],
        
        # Scale-down settings
        'scale_down_stabilization_window': 300,  # seconds
        'scale_down_policies': [
            {
                'type': 'percent', 
                'value': 10,  # Reduce by 10%
                'period_seconds': 60
            }
        ]
    }
    
    return config

autoscaling_config = create_autoscaling_config()
```

## Monitoring and Logging

```python
def setup_cloud_monitoring():
    """Setup monitoring for cloud deployments"""
    
    import logging
    import time
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/var/log/xgboost.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('xgboost_cloud')
    
    class ModelMonitor:
        def __init__(self):
            self.prediction_count = 0
            self.error_count = 0
            self.start_time = time.time()
        
        def log_prediction(self, prediction_time, input_size):
            self.prediction_count += 1
            logger.info(f"Prediction #{self.prediction_count} - Time: {prediction_time:.3f}s, Input size: {input_size}")
        
        def log_error(self, error_msg):
            self.error_count += 1
            logger.error(f"Error #{self.error_count}: {error_msg}")
        
        def get_stats(self):
            uptime = time.time() - self.start_time
            return {
                'uptime_seconds': uptime,
                'total_predictions': self.prediction_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / max(self.prediction_count, 1),
                'predictions_per_second': self.prediction_count / uptime
            }
    
    return ModelMonitor()

# monitor = setup_cloud_monitoring()
```

## Next Steps

- For model export formats compatible with cloud services, see [Model Export Formats](06-model-exports.md)
- For production inference optimization, check [Model Inference](09-model-inference.md)
- For troubleshooting cloud deployments, visit [Troubleshooting Guide](10-troubleshooting.md)