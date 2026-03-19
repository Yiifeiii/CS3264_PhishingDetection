import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    DEEPFAKE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
    TEXT_PHISHING_MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    TEXT_MODEL_POSITIVE_CLASS_INDEX = 1

    # paths
    RAW_IMAGE_DIR = "data/raw"
    SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    REAL_IMAGE = "data/raw/image1.jpg"
    FAKE_IMAGE = "data/raw/image3.jpg"
    SAMPLE_IMAGE = "data/raw/image2.jpg"
    SAMPLE_IMAGES = [REAL_IMAGE, SAMPLE_IMAGE, FAKE_IMAGE]

    # phishing-text baseline
    PHISHING_KEYWORDS = {
        "urgency": [
            "urgent",
            "immediately",
            "asap",
            "suspended",
            "expired",
            "verify now",
        ],
        "credential_request": [
            "password",
            "otp",
            "pin",
            "login",
            "sign in",
            "verification code",
        ],
        "financial_request": [
            "bank",
            "payment",
            "invoice",
            "refund",
            "credit card",
            "transaction",
            "cash",
            "money",
        ],
        "call_to_action": [
            "click here",
            "open link",
            "update account",
            "confirm account",
            "reset account",
            "contact us",
            "call now",
            "message now",
        ],
        "prize_bait": [
            "win",
            "winner",
            "claim",
            "reward",
            "bonus",
            "free gift",
            "prize",
            "lucky draw",
        ],
    }

    KEYWORD_HIT_WEIGHT = 0.12
    URL_PRESENT_WEIGHT = 0.2
    EMAIL_PRESENT_WEIGHT = 0.1
    PHONE_PRESENT_WEIGHT = 0.15
    MONEY_PRESENT_WEIGHT = 0.15
    EMPTY_TEXT_PENALTY = 0.5
    MULTI_SIGNAL_BONUS = 0.08
    LINK_CREDENTIAL_BONUS = 0.1
    TEXT_RULE_WEIGHT = 0.6
    TEXT_MODEL_WEIGHT = 0.4
    MODEL_POSITIVE_THRESHOLD = 0.6

    IMAGE_SCORE_WEIGHT = 0.35
    TEXT_SCORE_WEIGHT = 0.65
    MEDIUM_RISK_THRESHOLD = 0.4
    HIGH_RISK_THRESHOLD = 0.7
