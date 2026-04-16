# Proof of Concept

This directory is isolated from the main training pipeline.

It contains a small proof-of-concept that compares:

- full screenshot `shot.png` -> SigLIP -> Logistic Regression
- OCR crop proposals from `shot.png` -> pooled SigLIP -> Logistic Regression

The sampler builds a balanced split of:

- `500 real + 500 fake` for training
- `150 real + 150 fake` for testing

Sampling is random but weighted toward source names that look closer to Singapore-relevant brands or workflows such as `Outlook`, `Office365`, `WhatsApp`, `Telegram`, `DBS`, `UOB`, `OCBC`, `Grab`, and `.sg` domains.

Model inputs use image pixels only. The source name is used only to bias the sampling step.

## Run

```bash
./venv/bin/python proof-of-concept/run_poc.py
```

Outputs are written under:

```text
proof-of-concept/outputs/sample_1k_300_seed42
```
