# CS3264 Phishing Detection

This project benchmarks multiple image deepfake / AI-generated image detectors from a single CLI.

`app.py` is the single entrypoint for prediction, benchmarking, SAFE fine-tuning, and DistilDIRE fine-tuning.

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

Use a fine-tuned DistilDIRE checkpoint without editing config:

```bash
python app.py predict \
  --model distildire \
  --image data/raw/fake/test1.png \
  --distildire-model-path results/DISTILDIRE/best.pth
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

Benchmark a fine-tuned DistilDIRE checkpoint directly:

```bash
python app.py benchmark \
  --models distildire \
  --dataset-root data/raw \
  --distildire-model-path results/DISTILDIRE/best.pth
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
  --train-data-path data/raw \
  --val-data-path data/raw \
  --output-dir results/SAFE
```

Optional useful flags:

- `--epochs`
- `--batch-size`
- `--lr`
- `--val-ratio`
- `--num-train`
- `--num-val`
- `--freeze-backbone`

Notes:

- If `--train-data-path` and `--val-data-path` are the same root, SAFE fine-tuning now performs the same deterministic stratified 9:1 split as DistilDIRE.
- The exact SAFE split is saved to `split.json` in the output directory.
- Use different train and val roots if you already have a fixed split and do not want auto-splitting.

## Fine-tuning DistilDIRE

DistilDIRE can also be fine-tuned directly from the same CLI:

```bash
python app.py finetune-distildire \
  --train-data-path data/raw \
  --val-data-path data/raw \
  --output-dir results/DISTILDIRE
```

Useful options:

- `--epochs`
- `--batch-size`
- `--lr`
- `--val-ratio`
- `--fake-threshold`
- `--num-train`
- `--num-val`
- `--freeze-backbone`

Notes:

- The local trainer expects the same labeled folder layout used by benchmarking.
- If `--train-data-path` and `--val-data-path` are the same root, the trainer now performs a deterministic stratified split using a 9:1 train/val ratio by default.
- The split metadata is written to `split.json` in the output directory.
- `--num-train` and `--num-val` use balanced label sampling, which is useful for smoke runs on a very small dataset.
- The default config points both train and val to `data/raw` for convenience; with the default `--val-ratio 0.1`, that becomes an internal 90/10 split.

## Notes

- `safe` and `distildire` rely on the official upstream repositories cloned under `external/`.
- `distildire` is much slower on CPU than the other models.
- The default model backend in config is `safe`, but the CLI lets you override this per run.
