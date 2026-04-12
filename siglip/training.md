1. Clean up
```bash
rm -rf data/sglip outputs/embeddings outputs/models outputs/reports outputs/learning_curve_cv_chat_social
```

2. Split datasets
```bash
./venv/bin/python scripts/split_siglip_dataset.py --source-root data/chat --output-root data/sglip
```

Merge `chat` and `social` into one source-aware split:
```bash
./venv/bin/python scripts/split_siglip_dataset.py \
  --source-root data/chat \
  --source-root data/social \
  --output-root data/sglip \
  --clear-output
```

\[Optional\] Create fixed-holdout learning-curve splits for `chat + social`:
```bash
./venv/bin/python scripts/create_siglip_learning_curve_splits.py \
  --source-root data/chat \
  --source-root data/social \
  --train-sizes 8,16,24,32,40 \
  --val-total 40 \
  --test-total 40 \
  --repeats 3 \
  --output-root data/learning_curve \
  --clear-output
```

Run the stricter learning-curve workflow with one fixed final test set and
k-fold validation on the remaining development pool:
```bash
./venv/bin/python scripts/run_siglip_learning_curve_cv.py \                                              
  --source-root data/chat \
  --source-root data/social \
  --train-sizes 8,40,72,104,136,168,200,224 \
  --test-total 80 \
  --folds 5 \
  --repeats 1 \
  --final-train-mode full_dev \
  --output-root outputs/learning_curve_cv_chat_social \
  --model-name artifacts/siglip2-base-patch16-224 \
  --num-workers 0 \
  --clear-output
```

This CV workflow writes:
- `manifest.json`: exact fixed-test and development split
- `summary/cv_raw_metrics.csv`: one row per fold, repeat, model, and train size
- `summary/cv_summary_metrics.csv`: averaged validation learning curve
- `summary/final_test_metrics.csv`: untouched test benchmark after final training
- `summary/final_report.md`: readable report

Plot graph:
**validation:**
```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ./venv/bin/python scripts/plot_siglip_learning_curve.py \
  --summary-csv outputs/learning_curve_cv_chat_social/summary/cv_summary_metrics.csv \
  --output-dir outputs/learning_curve_cv_chat_social/summary
```

**test:**
```bash
MPLCONFIGDIR=/tmp/matplotlib-cache ./venv/bin/python scripts/plot_siglip_learning_curve.py \
  --summary-csv outputs/learning_curve_cv_chat_social/summary/test_curve_summary_metrics.csv \
  --output-dir outputs/learning_curve_cv_chat_social/summary
```

3. Extract embeddings
```bash
SIGLIP_MODEL_NAME=artifacts/siglip2-base-patch16-224 \
SIGLIP_NUM_WORKERS=0 \
./venv/bin/python siglip/extract_embeddings.py
```

4. Run all model (LR, LightGBM, XGBoost)
```bash
./venv/bin/python siglip/train_classifier.py --model all
```

5. Benchmark
```bash
./venv/bin/python siglip/evaluate.py --model all
```

6. Inference One
```bash
SIGLIP_MODEL_NAME=/Users/fuijingmin/Project/CS3264_PhishingDetection/artifacts/siglip2-base-patch16-224 \
./venv/bin/python siglip/inference.py --image path/to/your_image.png --model xgboost
```

---

### Install model
```bash
./venv/bin/python scripts/install_siglip_artifacts.py
```
