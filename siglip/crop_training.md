1. Create a fixed raw holdout split

Use this once to materialize `train/` and `test/` folders from `data/crop`:

```bash
./venv/bin/python scripts/split_siglip_holdout.py \
  --source-root data/crop \
  --output-root data/crop_split \
  --test-total 20 \
  --clear-output
```

This writes:
- `data/crop_split/train/{real,fake}`
- `data/crop_split/test/{real,fake}`
- `data/crop_split/manifest.json`

`data/crop` can be either:
- `data/crop/{real,fake}`
- `data/crop/raw/{real,fake}`

The splitter handles both layouts.

2. Run k-fold validation on the train split and evaluate once on the fixed test split

Use the fixed-split CV runner:

```bash
./venv/bin/python scripts/run_siglip_fixed_split_cv.py \
  --train-root data/crop_split/train \
  --test-root data/crop_split/test \
  --train-sizes 8,16,24,32,40,48,56,64,72,76 \
  --folds 5 \
  --repeats 1 \
  --final-train-mode full_train \
  --output-root outputs/fixed_split_cv_crop \
  --model-name artifacts/siglip2-base-patch16-224 \
  --num-workers 0 \
  --clear-output
```

What this does:
- extracts SigLIP embeddings for `train` and `test`
- runs 5-fold CV on `train`
- compares `logreg`, `lightgbm`, and `xgboost`
- trains final models on the full train split
- evaluates final models once on the untouched test split

Important:
- do not use `scripts/run_siglip_learning_curve_cv.py` on `data/crop_split`
- that older script creates a second test split, which is not what you want here

3. Current train-size constraint for `data/crop_split`

With the current split:
- `train/real = 94`
- `train/fake = 48`
- `test/real = 10`
- `test/fake = 10`

For `5` folds, the largest valid total training size is `76`.

Reason:
- the script samples equally from `real` and `fake`
- each training size must be divisible by `2`
- the minority class (`fake`) limits the per-fold train pool

4. Plot the validation learning curves

This plots the averaged k-fold CV metrics from:
- `summary/cv_summary_metrics.csv`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ./venv/bin/python scripts/plot_siglip_learning_curve.py \
  --summary-csv outputs/fixed_split_cv_crop/summary/cv_summary_metrics.csv \
  --output-dir outputs/fixed_split_cv_crop/summary
```

This writes files like:
- `learning_curve_cv_grid.png`
- `learning_curve_cv_grid.svg`
- `learning_curve_cv_f1.png`
- `learning_curve_cv_accuracy.png`

5. Plot the fixed-test learning curves

This plots the per-training-size results on the same untouched test set from:
- `summary/test_curve_summary_metrics.csv`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ./venv/bin/python scripts/plot_siglip_learning_curve.py \
  --summary-csv outputs/fixed_split_cv_crop/summary/test_curve_summary_metrics.csv \
  --output-dir outputs/fixed_split_cv_crop/summary
```

This writes files like:
- `learning_curve_test_grid.png`
- `learning_curve_test_grid.svg`
- `learning_curve_test_f1.png`
- `learning_curve_test_accuracy.png`

6. Plot the final untouched test comparison

This plots one bar chart per metric from:
- `summary/final_test_metrics.csv`

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ./venv/bin/python scripts/plot_siglip_learning_curve.py \
  --summary-csv outputs/fixed_split_cv_crop/summary/final_test_metrics.csv \
  --output-dir outputs/fixed_split_cv_crop/summary
```

This writes files like:
- `final_test_f1.png`
- `final_test_accuracy.png`
- `final_test_roc_auc.png`

7. Read the outputs

Main files:
- `outputs/fixed_split_cv_crop/manifest.json`
- `outputs/fixed_split_cv_crop/summary/cv_summary_metrics.csv`
- `outputs/fixed_split_cv_crop/summary/test_curve_summary_metrics.csv`
- `outputs/fixed_split_cv_crop/summary/final_test_metrics.csv`
- `outputs/fixed_split_cv_crop/summary/final_report.md`

Final models:
- `outputs/fixed_split_cv_crop/final_models/logreg.joblib`
- `outputs/fixed_split_cv_crop/final_models/lightgbm.joblib`
- `outputs/fixed_split_cv_crop/final_models/xgboost.joblib`

8. Rerun without deleting outputs

If you want to reuse cached embeddings, drop `--clear-output`.

If the dataset changed and you want to rebuild the caches, add `--force`:

```bash
./venv/bin/python scripts/run_siglip_fixed_split_cv.py \
  --train-root data/crop_split/train \
  --test-root data/crop_split/test \
  --train-sizes 8,16,24,32,40,48,56,64,72,76 \
  --folds 5 \
  --repeats 1 \
  --final-train-mode full_train \
  --output-root outputs/fixed_split_cv_crop \
  --model-name artifacts/siglip2-base-patch16-224 \
  --num-workers 0 \
  --force
```
