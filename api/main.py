"""
FastAPI backend for ScamCheck SG.

Exposes a single POST /api/analyse endpoint that accepts an image upload,
runs the full inference pipeline (deepfake image model + OCR + text risk
analysis + risk fusion), and returns a structured risk assessment.

Start with:
    uvicorn api.main:app --reload --port 8000
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.config import Config
from models.deepfake_model import DeepfakeModel
from preprocess.image_preprocessor import ImagePreprocessor
from utils.inference_service import InferenceService
from ocr.ocr_service import OCRService
from utils.ocr_text_processor import OCRTextProcessor
from utils.text_risk_analyzer import TextRiskAnalyzer
from utils.risk_fusion_service import RiskFusionService

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class AnalyseResponse(BaseModel):
    risk_level: str          # "low" | "medium" | "high"
    risk_score: float        # 0-1
    image_score: float
    text_score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Application & model singletons (loaded once at startup)
# ---------------------------------------------------------------------------

app = FastAPI(title="ScamCheck SG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Expo dev client needs this
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

cfg = Config()
deepfake_model = DeepfakeModel(cfg.DEEPFAKE_MODEL_NAME, cfg.DEVICE)
inference_svc = InferenceService(deepfake_model)
preprocessor = ImagePreprocessor()
ocr_svc = OCRService(list(cfg.OCR_LANGUAGES), gpu=(cfg.DEVICE == "cuda"))
ocr_text_processor = OCRTextProcessor(cfg)
text_analyzer = TextRiskAnalyzer(cfg)
risk_fusion = RiskFusionService(cfg)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyse", response_model=AnalyseResponse)
async def analyse(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return AnalyseResponse(
            risk_level="low",
            risk_score=0.0,
            image_score=0.0,
            text_score=0.0,
            reasons=[f"Unsupported file type: {suffix}"],
        )

    # Save upload to a temp file so OCR / PIL can read from disk.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(image.file, tmp)
        tmp.close()
        tmp_path = tmp.name

        # 1. Image analysis
        pil_image = preprocessor.load_image(tmp_path)
        image_result = inference_svc.predict(pil_image)

        # 2. OCR → text processing → text risk scoring
        raw_text = ocr_svc.extract_text(tmp_path)
        processed_text = ocr_text_processor.process(raw_text)
        text_result = text_analyzer.analyze(processed_text["text"])

        if processed_text.get("warning"):
            text_result.setdefault("warnings", []).append(processed_text["warning"])
        text_result["ocr_processing"] = processed_text

        # 3. Fuse image + text scores
        fused = risk_fusion.combine(image_result, text_result)

        return AnalyseResponse(
            risk_level=fused["risk_level"],
            risk_score=fused["risk_score"],
            image_score=fused["image_score"],
            text_score=fused["text_score"],
            reasons=fused["reasons"],
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
