# X-Ray Multi-Class Classification

Classifies chest X-ray images as normal, pneumonia, or tuberculosis. Ships with ready-to-use PyTorch checkpoints and a Gradio web UI.

## Quick Start: Inference (Local)

1. Clone and enter the project:
   ```bash
   git clone <repository-url>
   cd Lung-Diseases-Classification-CNNs
   ```
2. Create and activate a virtual environment.
3. Install PyTorch first, then the rest of the requirements.

### Option A: GPU (CUDA 12.4 wheels)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-deps
```

### Option B: CPU only

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

4. Launch the UI:
   ```bash
   python app/inference_ui.py
   ```
   Open http://127.0.0.1:7860 in your browser, upload an image, and view the predicted class probabilities.

**Switching models:** `models/efficientnetv2s_best.pth` is loaded by default. To try `models/densenet121_best.pth` (or your own checkpoint), change `MODEL_PATH` and the `timm.create_model(...)` call in `app/inference_ui.py` to the matching architecture name.

## Training: What I Run

I train in Colab using `notebook/cnn_train.ipynb`. Datasets available at: https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset

```
data/
  train/
    normal/
    pneumonia/
    tuberculosis/
  val/
    normal/
    pneumonia/
    tuberculosis/
  test/
    normal/
    pneumonia/
    tuberculosis/
```

How I train:
- Open the notebook in Colab, mount Drive, and set `DATASET_ZIP_PATH` to the zipped `data/` folder on Drive.
- The notebook unzips to `/content/data`, builds a `CONFIG` dict (batch size 16, image size 512, EfficientNetV2-S, AdamW, ReduceLROnPlateau), trains for 30 epochs with light augments (resize/crop, small rotation, color jitter), and logs to TensorBoard.
- Best checkpoint is saved as `/content/drive/MyDrive/CNN/models/best_model.pth`. I download that file, drop it into `models/`, and update `MODEL_PATH` in `app/inference_ui.py` if needed.

To run locally instead of Colab, change the `CONFIG["paths"]` at the top of the notebook to point to your local `data`, `models`, and `logs` folders and ensure a GPU is available.

## Optional: Clean or Audit Your Dataset Locally

`scripts/clean.py` expects a YAML config. Save the snippet below as `configs/main_config.yaml` (create the folder if needed):

```yaml
paths:
  data_root: data
  reports_dir: reports
  cleaned_root: data_clean
dataset:
  splits: [train, val, test]
  classes: [normal, pneumonia, tuberculosis]
  image_exts: [.png, .jpg, .jpeg]
  target_image_size: [512, 512] # (Adjust based on the model or the GPU resource available)
cleaning:
  min_width: 10
  min_height: 10
  hash_algorithm: phash
  duplicate_threshold: 0
  convert_to_rgb: true
  copy_cleaned_dataset: true
  report_filename: cleaning_report.csv
```

Then run:

```bash
python scripts/clean.py --config configs/main_config.yaml --copy-clean
```

The script reports corrupt files, potential duplicates, and can write a cleaned copy to `data_clean/` plus a CSV report in `reports/`.

## Repository Map

- `app/inference_ui.py` — Gradio app for inference.
- `models/` — Pretrained checkpoints (EfficientNetV2-S default, DenseNet121 alternative).
- `notebook/cnn_train.ipynb` — End-to-end training notebook (Colab-friendly).
- `scripts/clean.py`, `scripts/data_audit.py` — Dataset QA helpers.
- `reports/cleaning_report.csv` — Example cleaning output.
- `requirements.txt` — Dependency list; install PyTorch separately to match your hardware.

## Troubleshooting

- Torch install errors: use the CPU install if you do not have CUDA 12.4, or choose the CUDA wheel that matches your driver.
- Slow inference: CPU mode is slower; use a GPU or reduce the resize in `transform` inside `app/inference_ui.py`.
- "Model file not found": confirm the path in `MODEL_PATH` points to a real `.pth` file in `models/`.
- Gradio import/schema issues: the app pins `gradio==4.44.1` and patches schema parsing at startup; reinstall the pinned version if you changed it.

## Next Steps

- Monitor Colab training with `tensorboard --logdir /content/drive/MyDrive/CNN/logs`.
- Experiment with different architectures by changing `CONFIG["training"]["model_name"]` in the notebook and updating `MODEL_PATH` for inference.
