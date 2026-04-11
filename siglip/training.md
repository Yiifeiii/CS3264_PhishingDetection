1. Clean up
```bash
rm -rf data/sglip outputs/embeddings outputs/models outputs/reports
```

2. Split datasets
```bash
./venv/bin/python scripts/split_siglip_dataset.py --source-root data/chat --output-root data/sglip
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