import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    DEEPFAKE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
    TEXT_PHISHING_MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    TEXT_MODEL_POSITIVE_CLASS_INDEX = 1
    CHINESE_TO_ENGLISH_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"

    # paths
    RAW_IMAGE_DIR = "data/raw"
    SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    OCR_LANGUAGES = ("en", "ch_sim")
    OCR_CHINESE_POLICY = "strip"  # valid options: "strip", "skip", "translate"

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
    TEXT_RELEVANCE_MIN_SCORE = 2.0
    TEXT_RELEVANCE_MAX_CHUNKS = 5
    TEXT_RELEVANCE_STRONG_SIGNALS = (
        "urgent", "verify", "verification", "identity", "account", "password", "otp",
        "pin", "login", "sign in", "click", "link", "reset", "update", "confirm",
        "whatsapp", "bank", "payment", "refund", "invoice", "claim", "reward",
        "bonus", "free", "gift", "prize", "frozen", "suspended", "expired",
        "singpass", "iras", "cpf", "limited", "action required", "official notice",
    )
    TEXT_RELEVANCE_WEAK_SIGNALS = (
        "call", "message", "now", "today", "immediately", "security", "alert",
        "notice", "government", "profile", "transaction", "money", "cash",
    )
    TEXT_RELEVANCE_NOISE_HINTS = (
        "channelnewsasia", "mothership", "straits times", "cna", "readmore",
        "prime minister", "singapore", "exclusive", "breaking", "news",
        "cabinet", "leadership", "diplomatic", "community chest",
    )

    IMAGE_SCORE_WEIGHT = 0.35
    TEXT_SCORE_WEIGHT = 0.65
    MEDIUM_RISK_THRESHOLD = 0.4
    HIGH_RISK_THRESHOLD = 0.7
