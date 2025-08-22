# Ultralytics YOLO Guide 2025

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Different Tasks & Capabilities](#different-tasks--capabilities)
5. [Supported Platforms](#supported-platforms)
6. [Model Storage & Management](#model-storage--management)
7. [Model Reuse & Loading](#model-reuse--loading)
8. [Inference](#inference)
9. [Open Source Integrations](#open-source-integrations)
10. [Official Documentation Links](#official-documentation-links)

## Overview

Ultralytics YOLO11 is the latest version of the acclaimed real-time object detection and image segmentation model, built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

### Key Features
- **Real-time object detection** with state-of-the-art accuracy
- **Multiple computer vision tasks** (detection, segmentation, pose estimation, classification, tracking)
- **Cross-platform support** (edge devices to cloud APIs)
- **Easy integration** with popular ML platforms
- **Comprehensive model family** (YOLO11, YOLO12 with attention-centric architecture)

## Installation

### Method 1: Standard Installation (Recommended)
Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8:

```bash
pip install ultralytics
```

### Method 2: Conda Installation
```bash
conda install -c conda-forge ultralytics
```

### Method 3: Docker Installation
```bash
# Pull the latest Ultralytics Docker image
docker pull ultralytics/ultralytics:latest

# Run container with GPU support
docker run --gpus all -it ultralytics/ultralytics:latest
```

### Method 4: Installation without Dependencies
You can install the ultralytics package core without any dependencies using pip's --no-deps flag:

```bash
pip install ultralytics --no-deps
```

### Method 5: Development Installation
```bash
# Clone repository
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# Install in editable mode
pip install -e .
```

## Usage

### Basic Usage Examples

#### Python API
```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("yolo11n.pt")  # nano version for speed
# model = YOLO("yolo11s.pt")  # small
# model = YOLO("yolo11m.pt")  # medium  
# model = YOLO("yolo11l.pt")  # large
# model = YOLO("yolo11x.pt")  # extra large

# Predict on an image
results = model("path/to/image.jpg")

# Predict on multiple images
results = model(["image1.jpg", "image2.jpg"])

# Process results
for r in results:
    print(r.boxes)  # print detection bounding boxes
    r.show()        # display to screen
    r.save(filename='result.jpg')  # save to disk
```

#### Command Line Interface
```bash
# Predict with pre-trained model
yolo predict model=yolo11n.pt source="path/to/image.jpg"

# Predict on video
yolo predict model=yolo11n.pt source="path/to/video.mp4"

# Predict with webcam
yolo predict model=yolo11n.pt source=0
```

### Training Custom Models

#### Python Training
```python
from ultralytics import YOLO

# Load a pre-trained model for transfer learning
model = YOLO("yolo11n.pt")

# Train the model on custom dataset
model.train(
    data="path/to/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU device
)
```

#### CLI Training
```bash
yolo detect train data=path/to/dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## Different Tasks & Capabilities

Ultralytics offers support for a wide range of models, each tailored to specific tasks like object detection, instance segmentation, image classification, pose estimation, and multi-object tracking.

### 1. Object Detection
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")

# Access detection results
for r in results:
    boxes = r.boxes.xyxy  # bounding boxes
    conf = r.boxes.conf   # confidence scores
    cls = r.boxes.cls     # class labels
```

### 2. Instance Segmentation  
```python
model = YOLO("yolo11n-seg.pt")
results = model("image.jpg")

# Access segmentation masks
for r in results:
    masks = r.masks.data  # segmentation masks
    boxes = r.boxes.xyxy  # bounding boxes
```

### 3. Image Classification
```python
model = YOLO("yolo11n-cls.pt")
results = model("image.jpg")

# Access classification results
for r in results:
    probs = r.probs  # class probabilities
    top1 = r.probs.top1  # top-1 class index
```

### 4. Pose Estimation
```python
model = YOLO("yolo11n-pose.pt")
results = model("image.jpg")

# Access pose keypoints
for r in results:
    keypoints = r.keypoints.xy  # keypoint coordinates
    conf = r.keypoints.conf     # keypoint confidence
```

### 5. Object Tracking
```python
model = YOLO("yolo11n.pt")
results = model.track("video.mp4")

# Access tracking results
for r in results:
    boxes = r.boxes.xyxy
    track_ids = r.boxes.id  # object IDs across frames
```

### 6. Oriented Bounding Boxes (OBB)
```python
model = YOLO("yolo11n-obb.pt")
results = model("image.jpg")

# Access OBB results
for r in results:
    obb = r.obb  # oriented bounding boxes
```

## Supported Platforms

### Hardware Platforms
- **CPU**: Intel, AMD, Apple Silicon (M1/M2)
- **GPU**: NVIDIA CUDA, Apple Metal, AMD ROCm
- **Edge Devices**: Raspberry Pi, NVIDIA Jetson, Google Coral
- **Mobile**: iOS (Core ML), Android (TensorFlow Lite)

### Cloud Platforms
- **AWS**: EC2, SageMaker, Lambda
- **Google Cloud**: Compute Engine, AI Platform, Cloud Functions
- **Azure**: Virtual Machines, Machine Learning Studio
- **Docker**: Containerized deployment

### Export Formats
```python
# Export to different formats
model = YOLO("yolo11n.pt")

model.export(format="onnx")        # ONNX
model.export(format="torchscript") # TorchScript
model.export(format="coreml")      # Core ML (iOS)
model.export(format="tflite")      # TensorFlow Lite (mobile)
model.export(format="openvino")    # Intel OpenVINO
model.export(format="tensorrt")    # NVIDIA TensorRT
```

## Model Storage & Management

### Local Storage
```python
# Save trained model
model.save("my_custom_model.pt")

# Models are automatically saved during training
model.train(data="dataset.yaml", name="my_experiment")
# Saved to: runs/detect/my_experiment/weights/best.pt
```

### Cloud Storage Integration
```python
# Save to cloud storage (example with AWS S3)
import boto3

# After training
model_path = "runs/detect/train/weights/best.pt"

# Upload to S3
s3_client = boto3.client('s3')
s3_client.upload_file(model_path, 'my-bucket', 'models/yolo11_custom.pt')
```

### Model Versioning
```python
# Version your models with timestamps
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"yolo11_custom_{timestamp}.pt"
model.save(model_name)
```

## Model Reuse & Loading

### Loading Pre-trained Models
```python
from ultralytics import YOLO

# Load official pre-trained models
model = YOLO("yolo11n.pt")    # Downloads automatically first time
model = YOLO("yolo11s.pt")    
model = YOLO("yolo11m.pt")

# Load custom trained models
model = YOLO("path/to/my_model.pt")

# Load from URL
model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt")
```

### Transfer Learning
```python
# Start with pre-trained model for better results
base_model = YOLO("yolo11n.pt")

# Fine-tune on your dataset
base_model.train(
    data="custom_dataset.yaml",
    epochs=50,
    lr0=0.001,  # Lower learning rate for fine-tuning
    freeze=10   # Freeze first 10 layers
)
```

### Model Checkpoints
```python
# Resume training from checkpoint
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)

# Load specific checkpoint
model = YOLO("runs/detect/train/weights/epoch_50.pt")
```

## Inference

### Batch Inference
```python
model = YOLO("yolo11n.pt")

# Single image
results = model("image.jpg")

# Multiple images
results = model(["img1.jpg", "img2.jpg", "img3.jpg"])

# Entire directory
results = model("path/to/images/")

# Video file
results = model("video.mp4")

# YouTube video
results = model("https://youtu.be/dQw4w9WgXcQ")

# Live stream
results = model("rtsp://192.168.1.100:554/stream")
```

### Inference Configuration
```python
model = YOLO("yolo11n.pt")

results = model.predict(
    source="image.jpg",
    conf=0.25,      # confidence threshold
    iou=0.45,       # NMS IOU threshold
    imgsz=640,      # image size
    device="0",     # GPU device
    save=True,      # save results
    save_txt=True,  # save as txt
    save_conf=True  # save confidence scores
)
```

### Real-time Inference
```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Visualize results
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('YOLO11 Inference', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Open Source Integrations

### MLflow Integration

MLflow logging integration with Ultralytics YOLO offers a streamlined way to keep track of your machine learning experiments. It empowers you to monitor performance metrics and manage artifacts effectively, thus aiding in robust model development and deployment.

```python
from ultralytics import YOLO

# Enable MLflow logging
model = YOLO("yolo11n.pt")
model.train(
    data="dataset.yaml",
    epochs=100,
    # MLflow will automatically log metrics, parameters, and artifacts
)
```

**MLflow Configuration:**
```python
import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("yolo11_experiment")

with mlflow.start_run():
    model = YOLO("yolo11n.pt")
    
    # Training automatically logs to MLflow
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640
    )
    
    # Log additional custom metrics
    mlflow.log_metric("custom_score", 0.95)
    
    # Log model
    mlflow.pytorch.log_model(model.model, "yolo11_model")
```

### Weights & Biases (wandb)
```python
from ultralytics import YOLO
import wandb

# Initialize wandb
wandb.init(project="yolo11-project")

model = YOLO("yolo11n.pt")
model.train(
    data="dataset.yaml",
    epochs=100,
    # wandb logging is automatically enabled
)
```

### TensorBoard Integration
```python
model = YOLO("yolo11n.pt")
model.train(
    data="dataset.yaml",
    epochs=100,
    # TensorBoard logs saved to runs/detect/train/
)

# View in TensorBoard
# tensorboard --logdir runs/detect
```

### Comet ML
```python
from ultralytics import YOLO
import comet_ml

# Initialize Comet
comet_ml.init(project_name="yolo11-detection")

model = YOLO("yolo11n.pt")
model.train(data="dataset.yaml", epochs=100)
```

### ClearML Integration
```python
from ultralytics import YOLO
from clearml import Task

# Initialize ClearML task
task = Task.init(project_name="YOLOv11", task_name="object_detection")

model = YOLO("yolo11n.pt")
model.train(data="dataset.yaml", epochs=100)
```

### Ray Tune for Hyperparameter Optimization
```python
from ultralytics import YOLO
from ray import tune

def train_yolo(config):
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset.yaml",
        epochs=config["epochs"],
        lr0=config["lr0"],
        batch=config["batch_size"]
    )

# Hyperparameter search
analysis = tune.run(
    train_yolo,
    config={
        "epochs": tune.choice([50, 100, 200]),
        "lr0": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64])
    },
    num_samples=10
)
```

### DVC (Data Version Control)
```yaml
# dvc.yaml
stages:
  train:
    cmd: python train_yolo.py
    deps:
      - data/
      - train_yolo.py
    outs:
      - models/yolo11_trained.pt
    metrics:
      - metrics.json
```

### Hydra Configuration Management
```python
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

@hydra.main(config_path="conf", config_name="config")
def train_yolo(cfg: DictConfig):
    model = YOLO(cfg.model.name)
    model.train(
        data=cfg.data.path,
        epochs=cfg.training.epochs,
        lr0=cfg.training.lr0,
        batch=cfg.training.batch_size
    )

if __name__ == "__main__":
    train_yolo()
```

## Official Documentation Links

### Main Documentation
- **Homepage**: https://docs.ultralytics.com/
- **Quickstart Guide**: https://docs.ultralytics.com/quickstart/
- **GitHub Repository**: https://github.com/ultralytics/ultralytics
- **PyPI Package**: https://pypi.org/project/ultralytics/

### Models & Tasks
- **YOLO11 Models**: https://docs.ultralytics.com/models/yolo11/
- **YOLO12 Models**: https://docs.ultralytics.com/models/yolo12/
- **All Supported Models**: https://docs.ultralytics.com/models/
- **Object Detection**: https://docs.ultralytics.com/tasks/detect/
- **Instance Segmentation**: https://docs.ultralytics.com/tasks/segment/
- **Image Classification**: https://docs.ultralytics.com/tasks/classify/
- **Pose Estimation**: https://docs.ultralytics.com/tasks/pose/
- **Object Tracking**: https://docs.ultralytics.com/modes/track/

### Training & Modes
- **Training Guide**: https://docs.ultralytics.com/modes/train/
- **Validation**: https://docs.ultralytics.com/modes/val/
- **Prediction**: https://docs.ultralytics.com/modes/predict/
- **Export Models**: https://docs.ultralytics.com/modes/export/

### Integrations
- **All Integrations**: https://docs.ultralytics.com/integrations/
- **MLflow Integration**: https://docs.ultralytics.com/integrations/mlflow/
- **Weights & Biases**: https://docs.ultralytics.com/integrations/weights-biases/
- **Comet ML**: https://docs.ultralytics.com/integrations/comet/
- **ClearML**: https://docs.ultralytics.com/integrations/clearml/

### Support & Community
- **FAQ**: https://docs.ultralytics.com/help/FAQ/
- **Discussions**: https://github.com/orgs/ultralytics/discussions
- **Issues**: https://github.com/ultralytics/ultralytics/issues
- **Discord Community**: https://discord.com/invite/ultralytics

### Commercial & Licensing
- **Ultralytics Licensing**: https://www.ultralytics.com/license
- **Enterprise Solutions**: https://www.ultralytics.com/enterprise

---

**Last Updated**: August 2025 | **Version**: Based on YOLO11/YOLO12 and Ultralytics 8.x series