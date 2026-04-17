"""BayesianPredictor — Steps 4 & 6 of the ensemble pipeline.

Wraps everything into a single ``predict(image_path, source_name='chat')``
function:

    extract features → fused classifier → calibrate → safety guardrail
                    → threshold to {low, medium, high}

Returned dict:

    {
        "verdict":            "low" | "medium" | "high",
        "calibrated_prob":    float,       # p̂_fused after guardrail
        "raw_fused_prob":     float,       # pre-guardrail calibrated prob
        "uncalibrated_prob":  float,
        "branch_scores": {
            "fuse_siglip_dino_prob":   float,
            "crop_siglip_dino_prob":   float,   # guardrail signal only
            "ocr_distilbert_combined": float,
            "ocr_distilbert_heuristic": float,
            "ocr_distilbert_model":    float,
        },
        "thresholds":         {"t_low": ..., "t_high": ...},
        "guardrail_fired":    bool,
        "reason":             str,
        "raw_text":           str,        # Ollama OCR transcription
    }
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import json
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGLIP_ROOT = PROJECT_ROOT / "siglip"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SIGLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(SIGLIP_ROOT))

from ocr.ocr_service import OCRService
from utils.config import Config
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer

from grounding_dino_fusion_utils import (  # noqa: E402
    CHAT_PROMPT_LABELS,
    DEFAULT_PROMPT_LABELS,
    SOCIAL_PROMPT_LABELS,
    build_grounding_dino_candidates,
    detect_device,
    detect_grounding_dino_regions,
    embed_pil_images,
    load_grounding_dino_processor_and_model,
    pool_crop_features,
    prompt_labels_for_source,
    rank_grounding_candidates,
)
from feature_utils import get_normalized_image_features  # noqa: E402
from hf_utils import load_siglip_processor_and_model  # noqa: E402

DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "ensemble"
FUSION_ROOT = PROJECT_ROOT / "outputs" / "chat_social_grounding_dino_fusion"


@dataclass
class PredictorConfig:
    """All tunable knobs for the live predictor."""

    siglip_model_name: str = DEFAULT_SIGLIP_MODEL
    gdino_model_name: str = DEFAULT_GDINO_MODEL

    fused_clf_path: Path = FUSION_ROOT / "fusion_concat_siglip" / "models" / "lightgbm.joblib"
    crop_clf_path: Path = FUSION_ROOT / "grounding_dino_crop_siglip" / "models" / "lightgbm.joblib"
    calibrated_path: Path = DEFAULT_ARTIFACT_ROOT / "calibrated_ensemble.joblib"
    thresholds_path: Path = DEFAULT_ARTIFACT_ROOT / "thresholds.json"

    box_threshold: float = 0.25
    text_threshold: float = 0.25
    padding_ratio: float = 0.05
    max_crops_per_image: int = 6
    batch_size: int = 8
    pooling: str = "avg"

    # Step 4 safety-guardrail knob
    crop_guardrail_threshold: float = 0.8


# ── Live image-branch embedding ─────────────────────────────────────

@torch.no_grad()
def embed_global(image: Image.Image, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feat = get_normalized_image_features(model, inputs)
    return feat.cpu().numpy()[0].astype(np.float32)


@torch.no_grad()
def embed_crop_pooled(
    image: Image.Image,
    source_name: str,
    cfg: PredictorConfig,
    siglip_processor,
    siglip_model,
    gdino_processor,
    gdino_model,
    device: str,
) -> np.ndarray:
    """Return pooled crop-SigLIP embedding (768-D, L2-normalized)."""
    width, height = image.size
    prompt_labels = prompt_labels_for_source(
        source_name=source_name,
        fallback_prompt_labels=DEFAULT_PROMPT_LABELS,
        chat_prompt_labels=CHAT_PROMPT_LABELS,
        social_prompt_labels=SOCIAL_PROMPT_LABELS,
    )

    try:
        detections = detect_grounding_dino_regions(
            image=image,
            processor=gdino_processor,
            model=gdino_model,
            device=device,
            prompt_labels=prompt_labels,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )
        candidates = build_grounding_dino_candidates(
            detections=detections,
            image_width=width,
            image_height=height,
            padding_ratio=cfg.padding_ratio,
        )
    except Exception:
        candidates = [{
            "candidate_type": "fallback_full_image",
            "bbox": [0, 0, width, height],
            "texts": [],
            "ocr_confidence_mean": 0.0,
            "ocr_confidence_max": 0.0,
            "ocr_region_count": 0,
            "score_hint": 0.0,
            "sources": [],
        }]

    candidates = rank_grounding_candidates(
        candidates, source_name=source_name,
        image_width=width, image_height=height,
    )
    if cfg.max_crops_per_image > 0:
        candidates = candidates[:cfg.max_crops_per_image]

    crop_images: list[Image.Image] = []
    crop_weights: list[float] = []
    for candidate in candidates:
        bbox = tuple(int(c) for c in candidate["bbox"])
        crop_images.append(image.crop(bbox).copy())
        # Match rank-based pooling weight used in training
        weight = max(float(candidate.get("score_hint", 0.0))
                     + float(candidate.get("ocr_confidence_mean", 0.0)), 0.05)
        crop_weights.append(weight)

    if not crop_images:
        crop_images = [image.copy()]
        crop_weights = [1.0]

    crop_features = embed_pil_images(
        images=crop_images,
        processor=siglip_processor,
        model=siglip_model,
        device=device,
        batch_size=cfg.batch_size,
    )
    pooled = pool_crop_features(
        crop_features, crop_weights=crop_weights, pooling=cfg.pooling,
    )
    return pooled.astype(np.float32)


# ── Text branch ─────────────────────────────────────────────────────

def score_text_branch(
    image_path: str,
    ocr: OCRService,
    processor: OCRTextProcessor,
    analyzer: TextRiskAnalyzer,
) -> tuple[dict, str]:
    """Return (scores_dict, raw_text)."""
    try:
        raw_text = ocr.extract_text(image_path) or ""
    except Exception:
        raw_text = ""

    processed = processor.process(raw_text)
    processed_text = str(processed.get("text") or "").strip()

    scores = {
        "ocr_distilbert_combined": 0.5,
        "ocr_distilbert_heuristic": 0.5,
        "ocr_distilbert_model": 0.5,
    }
    if processed_text:
        analysis = analyzer.analyze(processed_text)
        if analysis.get("score") is not None:
            scores["ocr_distilbert_combined"] = float(analysis["score"])
        if analysis.get("rule_score") is not None:
            scores["ocr_distilbert_heuristic"] = float(analysis["rule_score"])
        if analysis.get("model_score_raw") is not None:
            scores["ocr_distilbert_model"] = float(analysis["model_score_raw"])

    return scores, raw_text


# ── Main predictor ──────────────────────────────────────────────────

class BayesianPredictor:
    """One-shot live inference bundling every stage of the ensemble."""

    def __init__(self, cfg: PredictorConfig | None = None, app_config: Config | None = None):
        self.cfg = cfg or PredictorConfig()
        self.app_config = app_config or Config()
        self.device = detect_device()

        # Load thresholds
        thr = json.loads(Path(self.cfg.thresholds_path).read_text(encoding="utf-8"))
        self.t_low = float(thr["t_low"])
        self.t_high = float(thr["t_high"])

        # Load calibrated meta-classifier bundle
        bundle = joblib.load(self.cfg.calibrated_path)
        self.calibrated_model = bundle["model"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.calibration_method: str = bundle["method"]

        # Load branch classifiers
        self.fused_clf = joblib.load(self.cfg.fused_clf_path)
        self.crop_clf = joblib.load(self.cfg.crop_clf_path)

        # Load vision encoders
        self.siglip_processor, self.siglip_model = load_siglip_processor_and_model(
            self.cfg.siglip_model_name, self.device,
        )
        self.gdino_processor, self.gdino_model = load_grounding_dino_processor_and_model(
            self.cfg.gdino_model_name, self.device,
        )

        # Load text branch
        self.ocr = OCRService(
            languages=list(self.app_config.OCR_LANGUAGES),
            gpu=(self.device == "cuda"),
            backend="ollama",
            ollama_model=self.app_config.OCR_OLLAMA_MODEL,
            ollama_host=self.app_config.OCR_OLLAMA_HOST,
            ollama_timeout_seconds=self.app_config.OCR_OLLAMA_TIMEOUT_SECONDS,
            ollama_clean_output=self.app_config.OCR_OLLAMA_CLEAN_OUTPUT,
        )
        self.text_processor = OCRTextProcessor(self.app_config)
        self.text_analyzer = TextRiskAnalyzer(self.app_config)

    # ── Public API ──────────────────────────────────────────────────

    def predict(self, image_path: str, source_name: str = "chat") -> dict:
        # Image branch
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")

        global_feat = embed_global(
            image, self.siglip_processor, self.siglip_model, self.device,
        )
        crop_feat = embed_crop_pooled(
            image=image,
            source_name=source_name,
            cfg=self.cfg,
            siglip_processor=self.siglip_processor,
            siglip_model=self.siglip_model,
            gdino_processor=self.gdino_processor,
            gdino_model=self.gdino_model,
            device=self.device,
        )
        fused_vec = np.concatenate([global_feat, crop_feat], axis=0).reshape(1, -1)
        crop_vec = crop_feat.reshape(1, -1)

        fuse_siglip_dino_prob = float(self.fused_clf.predict_proba(fused_vec)[0, 1])
        crop_siglip_dino_prob = float(self.crop_clf.predict_proba(crop_vec)[0, 1])

        # Text branch
        text_scores, raw_text = score_text_branch(
            image_path, self.ocr, self.text_processor, self.text_analyzer,
        )

        branch_scores = {
            "fuse_siglip_dino_prob": fuse_siglip_dino_prob,
            "crop_siglip_dino_prob": crop_siglip_dino_prob,
            **text_scores,
        }

        # Meta-classifier feature vector in trained order
        x = np.array(
            [[branch_scores.get(name, 0.5) for name in self.feature_names]],
            dtype=np.float64,
        )
        uncal_prob = float(self.calibrated_model.predict_proba(x)[0, 1])
        calibrated_prob = uncal_prob  # CalibratedClassifierCV already produces calibrated output
        raw_fused_prob = calibrated_prob

        # Step 4: safety guardrail — crop-SigLIP branch veto
        guardrail_fired = False
        if crop_siglip_dino_prob >= self.cfg.crop_guardrail_threshold and calibrated_prob < self.t_low:
            calibrated_prob = max(calibrated_prob, self.t_low)
            guardrail_fired = True

        verdict = self._verdict_from_prob(calibrated_prob)
        reason = self._reason(
            verdict=verdict,
            fuse_prob=fuse_siglip_dino_prob,
            crop_prob=crop_siglip_dino_prob,
            text_score=text_scores["ocr_distilbert_combined"],
            guardrail_fired=guardrail_fired,
        )

        return {
            "verdict": verdict,
            "calibrated_prob": round(calibrated_prob, 4),
            "raw_fused_prob": round(raw_fused_prob, 4),
            "uncalibrated_prob": round(uncal_prob, 4),
            "branch_scores": {k: round(float(v), 4) for k, v in branch_scores.items()},
            "thresholds": {"t_low": self.t_low, "t_high": self.t_high},
            "guardrail_fired": guardrail_fired,
            "reason": reason,
            "raw_text": raw_text,
        }

    # ── Internals ───────────────────────────────────────────────────

    def _verdict_from_prob(self, prob: float) -> str:
        if prob >= self.t_high:
            return "high"
        if prob >= self.t_low:
            return "medium"
        return "low"

    def _reason(
        self,
        verdict: str,
        fuse_prob: float,
        crop_prob: float,
        text_score: float,
        guardrail_fired: bool,
    ) -> str:
        parts: list[str] = []
        if guardrail_fired:
            parts.append(
                f"crop-SigLIP branch flagged a highly suspicious image region "
                f"(p={crop_prob:.2f} ≥ {self.cfg.crop_guardrail_threshold}); "
                "elevated to medium by safety guardrail"
            )
        # Which branch drove the decision
        if fuse_prob - text_score > 0.15:
            parts.append(f"vision-dominant (fused SigLIP+DINO p={fuse_prob:.2f} "
                         f"> text score {text_score:.2f})")
        elif text_score - fuse_prob > 0.15:
            parts.append(f"text-dominant (OCR+DistilBERT score {text_score:.2f} "
                         f"> vision p={fuse_prob:.2f})")
        else:
            parts.append(f"vision+text agree (fused p={fuse_prob:.2f}, "
                         f"text={text_score:.2f})")

        if verdict == "low":
            parts.append("both branches look benign; no scam indicators dominated")
        elif verdict == "medium":
            parts.append("some scam indicators present; user should verify before acting")
        else:
            parts.append("strong scam indicators across branches; treat as likely phishing")
        return "; ".join(parts)


# ── CLI for quick smoke tests ───────────────────────────────────────

def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run BayesianPredictor on a single image.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--source", default="chat",
                        help="Source hint: 'chat', 'social', or '' for default.")
    parser.add_argument(
        "--text-model",
        default="",
        help="Optional text-model override for the OCR text branch.",
    )
    parser.add_argument(
        "--ollama-model",
        default="",
        help="Optional Ollama model override for the OCR text branch.",
    )
    parser.add_argument(
        "--ollama-host",
        default="",
        help="Optional Ollama host override for the OCR text branch.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help="Optional ensemble artifacts directory override containing calibrated_ensemble.joblib and thresholds.json.",
    )
    args = parser.parse_args()

    predictor_cfg = None
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
        predictor_cfg = PredictorConfig(
            calibrated_path=artifacts_dir / "calibrated_ensemble.joblib",
            thresholds_path=artifacts_dir / "thresholds.json",
        )

    app_config = Config()
    if args.text_model:
        app_config.TEXT_PHISHING_MODEL_NAME = args.text_model
    if args.ollama_model:
        app_config.OCR_OLLAMA_MODEL = args.ollama_model
    if args.ollama_host:
        app_config.OCR_OLLAMA_HOST = args.ollama_host

    predictor = BayesianPredictor(cfg=predictor_cfg, app_config=app_config)
    result = predictor.predict(args.image, source_name=args.source)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
