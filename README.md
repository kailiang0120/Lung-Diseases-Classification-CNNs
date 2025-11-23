# 🩺 X-Ray Multi-Class Classification System

A production-ready deep learning system for classifying chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. Built with PyTorch and featuring a user-friendly Gradio web interface for real-time inference.

---

## 📋 Table of Contents

1. [Quick Start (Inference Only)](#-quick-start-inference-only)
2. [Project Overview](#-project-overview)
3. [Installation Guide](#-installation-guide)
4. [Running the Inference UI](#-running-the-inference-ui)
5. [Training Your Own Model](#-training-your-own-model)
6. [Project Structure](#-project-structure)
7. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start (Inference Only)

**Want to test the model immediately? Follow these steps:**

### Prerequisites
- Python 3.10+
- 4GB+ RAM
- GPU (optional, but recommended)

### Setup & Run

```bash
# 1. Clone the repository
git clone <repository-url>
cd X-Ray

# 2. Create conda environment
conda create -n xray-ml python=3.10 -y
conda activate xray-ml

# 3. Install dependencies
# For GPU (CUDA 12.4)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# For CPU only
pip install torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt

# 4. Launch the web interface
python app/inference_ui.py
```

**The interface will open at:** `http://127.0.0.1:7860`

Upload a chest X-ray image and get instant predictions!

---

## 🎯 Project Overview

### What It Does
This system uses state-of-the-art deep learning (EfficientNetV2-S) to analyze chest X-ray images and classify them into:
- **Normal** - Healthy lungs
- **Pneumonia** - Bacterial/viral lung infection
- **Tuberculosis** - TB infection

### Key Features
- ✅ **Web-based Interface** - Easy-to-use Gradio UI for instant predictions
- ✅ **GPU & CPU Support** - Runs on both CUDA-enabled GPUs and CPUs
- ✅ **Pre-trained Models** - Ready-to-use EfficientNetV2 and DenseNet121 models
- ✅ **Production Ready** - Automated data cleaning, augmentation, and validation
- ✅ **Extensible** - Full training pipeline for custom datasets

### Tech Stack
- **Framework:** PyTorch 2.4.1
- **Model:** EfficientNetV2-S (via timm library)
- **UI:** Gradio 4.44.1
- **Dataset:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)

---

## 💻 Installation Guide

### Option 1: Using Conda (Recommended)

#### Step 1: Create Environment
```bash
# Create new conda environment
conda create -n xray-ml python=3.10 -y

# Activate environment
conda activate xray-ml
```

#### Step 2: Install PyTorch

**For GPU (CUDA 12.4):**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

**For GPU (CUDA 11.8):**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch==2.4.1 torchvision==0.19.1
```

**Check your CUDA version:**
```bash
nvidia-smi  # Look for "CUDA Version: XX.X"
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check Gradio installation
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.4.1+cu124  (or 2.4.1 for CPU)
CUDA Available: True  (or False for CPU)
Gradio: 4.44.1
```

---

### Option 2: Using pip + venv

```bash
# Create virtual environment
python -m venv xray-env

# Activate environment
# Windows:
xray-env\Scripts\activate
# Linux/Mac:
source xray-env/bin/activate

# Install PyTorch (GPU)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

---

## 🌐 Running the Inference UI

### Basic Usage

```bash
# Activate your environment
conda activate xray-ml  # or: source xray-env/bin/activate

# Run the inference UI
python app/inference_ui.py
```

**The web interface will launch at:** `http://127.0.0.1:7860`

### UI Features

1. **Upload Image** - Drag & drop or click to upload chest X-ray (JPG/PNG)
2. **Instant Prediction** - Get probabilities for all three classes
3. **Confidence Scores** - See model confidence for each diagnosis

### Advanced Options

**Change Port:**
```python
# Edit app/inference_ui.py, line 110:
interface.launch(share=False, server_port=8080)
```

**Enable Public Link (Temporary):**
```python
# Edit app/inference_ui.py, line 110:
interface.launch(share=True)  # Creates a public gradio.live link
```

**Force CPU Mode:**
```python
# Edit app/inference_ui.py, line 10:
device = torch.device("cpu")
```

### Using Different Models

The project includes two pre-trained models:
- `models/efficientnetv2s_best.pth` (default, faster)
- `models/densenet121_best.pth` (alternative)

**To switch models:**

```python
# Edit app/inference_ui.py, lines 12-16:
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "densenet121_best.pth",  # Change this filename
)

# Also update line 33-34 to match the model architecture:
model = timm.create_model(
    "densenet121", pretrained=False, num_classes=len(class_names)
)
```

---

## 🎓 Training Your Own Model

### Dataset Setup

#### Step 1: Download Dataset

Download the [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset) and organize as:

```
data/
├── train/
│   ├── normal/          # Normal X-ray images
│   ├── pneumonia/       # Pneumonia X-ray images
│   └── tuberculosis/    # TB X-ray images
├── val/
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
└── test/
    ├── normal/
    ├── pneumonia/
    └── tuberculosis/
```

**Supported formats:** `.jpg`, `.jpeg`, `.png`

#### Step 2: Data Cleaning & Validation

```bash
# Audit dataset for issues
python scripts/data_audit.py --config configs/main_config.yaml

# Clean dataset (removes duplicates, corrupt files)
python scripts/clean.py \
  --config configs/main_config.yaml \
  --output-root data_clean \
  --copy-clean

# Verify cleaned dataset
python scripts/data_audit.py --data-root data_clean
```

**What cleaning does:**
- Removes duplicate images (perceptual hashing)
- Detects and excludes corrupt files
- Converts all images to RGB
- Resizes to consistent dimensions (325×325)
- Generates detailed report: `reports/cleaning_report.csv`

#### Step 3: Inspect Data Pipeline

```bash
# Verify data loaders work correctly
python scripts/inspect_dataloaders.py \
  --config configs/training_config.yaml \
  --num-workers 0
```

### Training Configuration

Edit `configs/training_config.yaml` to customize:

```yaml
paths:
  data_root: data_clean  # Point to cleaned dataset

training:
  model_name: tf_efficientnetv2_s  # Model architecture
  pretrained: true                 # Use ImageNet weights
  epochs: 50                       # Training epochs
  learning_rate: 0.001             # Initial learning rate
  batch_size: 32                   # Batch size (reduce if OOM)
  
data_module:
  batch_size: 32                   # DataLoader batch size
  num_workers: 4                   # Parallel data loading
  pin_memory: true                 # GPU optimization
```

**Available models:** `densenet121`, `resnet50`, `tf_efficientnetv2_s`, `convnext_tiny`, etc.  
See [timm documentation](https://huggingface.co/docs/timm/index) for full list.

### Launch Training

```bash
# Start training
python scripts/train.py \
  --config configs/training_config.yaml \
  --run-name my_xray_model \
  --num-workers 4

# Monitor training (optional)
tensorboard --logdir logs
```

**Training outputs:**
- `models/<run_name>/best.pth` - Best model checkpoint
- `models/<run_name>/epoch_XXX.pth` - Per-epoch checkpoints
- `logs/<run_name>/` - TensorBoard logs & metrics

### Resume Training

```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --run-name my_xray_model_resume \
  --resume models/my_xray_model/best.pth \
  --num-workers 4
```

---

## 📁 Project Structure

```
X-Ray/
├── app/
│   └── inference_ui.py          # Gradio web interface
│
├── configs/
│   ├── main_config.yaml         # Data & augmentation config
│   └── training_config.yaml     # Training hyperparameters
│
├── data/                        # Raw dataset (download separately)
│   ├── train/
│   ├── val/
│   └── test/
│
├── data_clean/                  # Cleaned dataset (auto-generated)
│
├── models/                      # Trained model checkpoints
│   ├── efficientnetv2s_best.pth
│   └── densenet121_best.pth
│
├── logs/                        # Training logs & TensorBoard
│
├── reports/                     # Data quality reports
│   └── cleaning_report.csv
│
├── scripts/                     # Training & data processing
│   ├── train.py                 # Main training script
│   ├── clean.py                 # Data cleaning
│   ├── data_audit.py            # Dataset validation
│   └── inspect_dataloaders.py  # Data pipeline testing
│
├── src/                         # Core modules
│   ├── data/                    # Dataset & transforms
│   ├── models/                  # Model builders
│   ├── training/                # Training loop & metrics
│   └── utils/                   # Helper functions
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solution:** Reduce batch size

```yaml
# Edit configs/training_config.yaml
data_module:
  batch_size: 16  # Reduce from 32

training:
  batch_size: 16  # Keep both in sync
```

Or force CPU mode:
```bash
python scripts/train.py --config configs/training_config.yaml --cpu
```

#### 2. "ModuleNotFoundError: No module named 'gradio'"

**Solution:** Reinstall dependencies
```bash
conda activate xray-ml
pip install --upgrade -r requirements.txt
```

#### 3. "Internal Server Error" in Gradio UI

**Solution:** The code includes a runtime patch for Gradio 4.44.1. If issues persist:

```bash
# Verify versions
python -c "import gradio, gradio_client; print(gradio.__version__, gradio_client.__version__)"
# Expected: 4.44.1 1.3.0

# Reinstall if needed
pip install --upgrade gradio==4.44.1 gradio_client>=1.0.2
```

#### 4. "Model file not found"

**Solution:** Ensure model checkpoint exists
```bash
# Check if model exists
ls models/efficientnetv2s_best.pth

# If missing, train a model first or download pre-trained weights
```

#### 5. Slow inference on CPU

**Expected behavior:** CPU inference is 5-10× slower than GPU. For production, use GPU or consider:
- Reducing image size (edit `transform` in `inference_ui.py`)
- Using a smaller model (e.g., `mobilenetv3_small`)

#### 6. Dataset download issues

**Solution:** Download from Kaggle:
1. Go to https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset
2. Click "Download" (requires Kaggle account)
3. Extract to `data/` folder following structure above

### Performance Tips

**GPU Optimization:**
```yaml
# configs/training_config.yaml
data_module:
  pin_memory: true      # Faster GPU transfer
  num_workers: 4        # Parallel data loading
```

**CPU Optimization:**
```yaml
data_module:
  pin_memory: false
  num_workers: 2        # Reduce for CPU
```

### Getting Help

1. **Check logs:** Training logs are in `logs/<run_name>/history.json`
2. **Verify data:** Run `python scripts/data_audit.py` to check dataset
3. **Test pipeline:** Run `python scripts/inspect_dataloaders.py` to verify data loading

---

## 📊 Model Performance

### Pre-trained Models

| Model | Accuracy | Inference Speed (GPU) | Size |
|-------|----------|----------------------|------|
| EfficientNetV2-S | ~92% | ~50ms/image | 82MB |
| DenseNet121 | ~90% | ~40ms/image | 31MB |

*Tested on NVIDIA RTX 3060, batch size 1*

### Dataset Statistics

| Split | Normal | Pneumonia | Tuberculosis | Total |
|-------|--------|-----------|--------------|-------|
| Train | 7,263 | 4,674 | 8,513 | 20,450 |
| Val | 900 | 570 | 1,064 | 2,534 |
| Test | 925 | 580 | 1,064 | 2,569 |

---

## 📝 License & Citation

### Dataset
The dataset is from Kaggle: [Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)

Please cite the original dataset authors if you use this in research.

### Code
This project is for educational and research purposes. For medical applications, always consult qualified healthcare professionals.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional model architectures
- Multi-GPU training support
- REST API for inference
- Docker containerization
- Model quantization for mobile deployment

---

## 📧 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review training logs in `logs/`
3. Verify environment with `pip list`

---

**Quick Commands Reference:**

```bash
# Setup
conda create -n xray-ml python=3.10 -y
conda activate xray-ml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Inference
python app/inference_ui.py

# Training
python scripts/clean.py --config configs/main_config.yaml --output-root data_clean
python scripts/train.py --config configs/training_config.yaml --run-name my_model

# Monitoring
tensorboard --logdir logs
```

---

**Note:** This system is a research/educational tool. Always consult qualified medical professionals for actual diagnoses.
