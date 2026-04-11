# SigLIP Baseline

This folder contains a frozen-image-embedding baseline for binary classification:

- `real` = non-scam / non-fake
- `fake` = scam / fake

The pipeline uses a pretrained SigLIP image encoder to turn screenshots into embeddings, then trains a shallow classifier on top of those embeddings.

## Data Layout

The current code expects:

```text
data/sglip/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

The class mapping is:

- `0 = real`
- `1 = fake`

This mapping is defined in `config.py`.

## What Each Script Does

- `../scripts/split_siglip_dataset.py`
  - Splits `data/raw/{real,fake}` into `data/sglip/{train,val,test}/{real,fake}`
  - Copies images instead of moving them
- `extract_embeddings.py`
  - Runs SigLIP on `train`, `val`, and `test`
  - Saves one `.npz` embedding file per split
- `train_classifier.py`
  - Trains one or more classifiers on the `train` embeddings
  - Reports validation metrics on `val`
- `evaluate.py`
  - Loads saved classifier checkpoints
  - Evaluates them on `test`
  - Supports `--model all` to compare all saved models in one run
- `inference.py`
  - Runs prediction on one image using a saved classifier

## Full Workflow

If you have not created the split yet:

```bash
./venv/bin/python scripts/split_siglip_dataset.py
```

Extract embeddings:

```bash
./venv/bin/python siglip/extract_embeddings.py
```

Train all classifiers:

```bash
./venv/bin/python siglip/train_classifier.py --model all
```

Evaluate all classifiers on the test set:

```bash
./venv/bin/python siglip/evaluate.py --model all
```

Run single-image inference:

```bash
./venv/bin/python siglip/inference.py --image path/to/image.png --model xgboost
```

## How Train, Val, and Test Are Used

These 3 splits have different jobs:

- `train`
  - Used to fit the classifier weights
- `val`
  - Used during development to compare models
  - This is where you should choose the best model or tune thresholds
- `test`
  - Used only after model selection
  - This is the final benchmark number

Current behavior:

- `extract_embeddings.py` processes all 3 splits separately
- `train_classifier.py` trains on `train` and reports on `val`
- `evaluate.py` reports on `test`

If you use the test set to make model decisions repeatedly, it stops being a true final benchmark.

## What The Metrics Mean

This is still one binary classification task: `real` vs `fake`.

The classification report shows metrics for both classes:

- `real recall`
  - Of all truly real images, how many were predicted as real
- `fake recall`
  - Of all truly fake images, how many were predicted as fake

The top-level `precision`, `recall`, and `f1` printed by the scripts refer to the positive class.

In this project, the positive class is `fake` because:

- `real = 0`
- `fake = 1`

So:

- top-level `precision` = fake precision
- top-level `recall` = fake recall
- top-level `f1` = fake f1

For scam detection, `fake recall` is usually the most important metric because it tells you how many scam images you actually catch.

## Current Classifiers

Supported classifier heads:

- `logreg`
- `lightgbm`
- `xgboost`

They all use the same SigLIP embeddings. The only thing that changes is the classifier on top.

## Output Files

Embeddings:

- `outputs/embeddings/train_embeddings.npz`
- `outputs/embeddings/val_embeddings.npz`
- `outputs/embeddings/test_embeddings.npz`

Saved models:

- `outputs/models/logreg.joblib`
- `outputs/models/lightgbm.joblib`
- `outputs/models/xgboost.joblib`

Reports:

- `outputs/reports/val_metrics_logreg.txt`
- `outputs/reports/val_metrics_lightgbm.txt`
- `outputs/reports/val_metrics_xgboost.txt`
- `outputs/reports/test_metrics_logreg.txt`
- `outputs/reports/test_metrics_lightgbm.txt`
- `outputs/reports/test_metrics_xgboost.txt`

## Practical Interpretation

Use validation metrics to choose the candidate model.

Use test metrics once for the final comparison.

For your use case, the most useful things to inspect are:

- fake recall
- fake f1
- roc_auc
- confusion matrix

Accuracy alone can look good even when the model still misses too many fake images.

## Troubleshooting

If SigLIP loading fails because Hugging Face files are not cached yet, connect to the internet once and run:

```bash
./venv/bin/python -c "from transformers import AutoProcessor, AutoModel; AutoProcessor.from_pretrained('google/siglip2-base-patch16-224'); AutoModel.from_pretrained('google/siglip2-base-patch16-224')"
```

Then rerun:

```bash
./venv/bin/python siglip/extract_embeddings.py
```
