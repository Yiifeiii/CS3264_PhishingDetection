# CS3264 Phishing Detection

This project benchmarks multiple image deepfake / AI-generated image detectors from a single CLI.

`app.py` is the single entrypoint for prediction, benchmarking, and SAFE fine-tuning.

## Models

- `faceforge`: Xception-based FaceForge checkpoint
- `safe`: SAFE checkpoint from the official repository
- `distildire`: DistilDIRE CelebA-HQ checkpoint with ADM first-step noise

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

The benchmark CLI expects labeled images under folders named either:

- `real` and `fake`
- `0_real` and `1_fake`

Example:

```text
data/raw/
  real/
    image1.jpg
    image2.jpg
  fake/
    image3.jpg
    image4.jpg
```

## CLI

### Single-image inference

Run one model on one image:

```bash
python app.py predict --model safe --image data/raw/fake/test1.png
```

Options:

- `--model`: `faceforge`, `safe`, or `distildire`
- `--ocr`: also run OCR and print extracted text

Example:

```bash
python app.py predict --model faceforge --image data/raw/real/image1.jpg --ocr
```

### Benchmark one or more models

Run all models on a labeled dataset:

```bash
python app.py benchmark --models faceforge safe distildire --dataset-root data/raw
```

Save per-image results to CSV:

```bash
python app.py benchmark \
  --models faceforge safe distildire \
  --dataset-root data/raw \
  --output-csv results/benchmark.csv
```

Show per-image predictions:

```bash
python app.py benchmark \
  --models faceforge safe distildire \
  --dataset-root data/raw \
  --show-details
```

Useful options:

- `--models`: choose any subset of `faceforge`, `safe`, `distildire`
- `--dataset-root`: root folder containing labeled images
- `--limit`: only benchmark the first N discovered images
- `--output-csv`: save per-image predictions

## Fine-tuning SAFE

SAFE can be fine-tuned directly from this repo:

```bash
python app.py finetune-safe \
  --train-data-path data/your_train \
  --val-data-path data/your_val \
  --output-dir results/SAFE
```

Optional useful flags:

- `--epochs`
- `--batch-size`
- `--lr`
- `--freeze-backbone`

## Notes

- `safe` and `distildire` rely on the official upstream repositories cloned under `external/`.
- `distildire` is much slower on CPU than the other models.
- The default model backend in config is `safe`, but the CLI lets you override this per run.
