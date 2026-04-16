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
  - Splits one or more labeled roots into `data/sglip/{train,val,test}/{real,fake}`
  - Example roots: `data/chat`, `data/social`
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
./venv/bin/python scripts/split_siglip_dataset.py \
  --source-root data/chat \
  --output-root data/sglip \
  --clear-output
```

To merge `chat` and `social` into one experiment:

```bash
./venv/bin/python scripts/split_siglip_dataset.py \
  --source-root data/chat \
  --source-root data/social \
  --output-root data/sglip \
  --clear-output
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

## Concatenated Global + Crop Fusion

If you already have one set of global-image SigLIP embeddings and one set of crop-image
SigLIP embeddings for the same train/val/test splits, you can concatenate them and train
the same shallow classifier heads on the fused representation.

The fusion script aligns samples by image path before concatenation, so the global and
crop `.npz` files must describe the same images within each split.

Example:

```bash
./venv/bin/python scripts/run_siglip_concat_fusion.py \
  --global-train outputs/global/embeddings/train_embeddings.npz \
  --crop-train outputs/crop/embeddings/train_embeddings.npz \
  --global-val outputs/global/embeddings/val_embeddings.npz \
  --crop-val outputs/crop/embeddings/val_embeddings.npz \
  --global-test outputs/global/embeddings/test_embeddings.npz \
  --crop-test outputs/crop/embeddings/test_embeddings.npz \
  --output-root outputs/siglip_concat_fusion \
  --model all \
  --l2-normalize-concat
```

Outputs:

- fused embeddings under `outputs/siglip_concat_fusion/embeddings/`
- trained models under `outputs/siglip_concat_fusion/models/`
- validation/test reports under `outputs/siglip_concat_fusion/reports/`
- run summary in `outputs/siglip_concat_fusion/summary.json`

## Fixed Holdout With GroundingDINO Crops + Concat Fusion

For the `chat` + `social` custom datasets, there is a single runner that:

- creates one fixed final test holdout
- extracts full-image SigLIP embeddings
- extracts GroundingDINO crop proposals, saves every crop, and pools crop embeddings per image
- uses source-aware GroundingDINO prompts:
  - `chat`: message bubble / text-message style regions
  - `social`: post / caption / logo / face style regions
- concatenates the global and crop embeddings
- trains all classifier heads (`logreg`, `lightgbm`, `xgboost`)
- evaluates `global`, `crop-only`, and `fusion` on the same untouched test split

Example:

```bash
./venv/bin/python scripts/run_siglip_grounding_dino_fusion_holdout.py \
  --source-root data/chat \
  --source-root data/social \
  --test-total 80 \
  --output-root outputs/chat_social_grounding_dino_fusion \
  --model-name artifacts/siglip2-base-patch16-224 \
  --detector-model-name IDEA-Research/grounding-dino-tiny \
  --device auto \
  --batch-size 16 \
  --max-crops-per-image 4 \
  --crop-pooling avg \
  --l2-normalize-concat \
  --clear-output
```

You can override the source-aware prompt sets if needed:

```bash
./venv/bin/python scripts/run_siglip_grounding_dino_fusion_holdout.py \
  --source-root data/chat \
  --source-root data/social \
  --chat-prompt-labels "message bubble,chat message,text message,sms,message block,conversation text,notification banner,link preview,button,qr code" \
  --social-prompt-labels "social media post,post card,caption text,text overlay,headline,news card,logo,profile picture,person,face,button,qr code"
```

With `chat` and `social`, `--test-total 80` means:

- `20` test images from `chat/real`
- `20` test images from `chat/fake`
- `20` test images from `social/real`
- `20` test images from `social/fake`

Main artifacts:

- split manifest: `outputs/chat_social_grounding_dino_fusion/manifest.json`
- split CSVs: `outputs/chat_social_grounding_dino_fusion/split/`
- global embeddings: `outputs/chat_social_grounding_dino_fusion/global_siglip/embeddings/`
- GroundingDINO crops: `outputs/chat_social_grounding_dino_fusion/grounding_dino_crop_siglip/crops/`
- crop embeddings: `outputs/chat_social_grounding_dino_fusion/grounding_dino_crop_siglip/embeddings/`
- fusion embeddings: `outputs/chat_social_grounding_dino_fusion/fusion_concat_siglip/embeddings/`
- saved joblib models:
  - `.../global_siglip/models/`
  - `.../grounding_dino_crop_siglip/models/`
  - `.../fusion_concat_siglip/models/`
- final metrics table: `outputs/chat_social_grounding_dino_fusion/summary/final_test_metrics.csv`
- final report: `outputs/chat_social_grounding_dino_fusion/summary/final_report.md`

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
