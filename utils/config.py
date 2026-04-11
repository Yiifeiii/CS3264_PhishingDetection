import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    DEEPFAKE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
    TEXT_PHISHING_MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    TEXT_MODEL_POSITIVE_CLASS_INDEX = 1
    # Experimental Chinese spam-email proxy model. This is the closest Chinese
    # sequence-classification checkpoint we found that exposes an explicit spam
    # class, but it is not a dedicated Chinese phishing model.
    CHINESE_TEXT_PHISHING_MODEL_NAME = "jason23322/email-classifier-optimized"
    CHINESE_TEXT_MODEL_POSITIVE_CLASS_INDEX = 3
    CHINESE_TO_ENGLISH_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"

    # paths
    RAW_IMAGE_DIR = "data/raw"
    SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    OCR_LANGUAGES = ("en", "ch_sim")
    OCR_CHINESE_POLICY = "strip"  # valid options: "strip", "skip", "translate", "route"

    # Allowlisted contacts suppress email/phone phishing heuristics but do not
    # hard-mark the sample as safe. Keep these lists tight and review them.
    TRUSTED_EMAIL_ADDRESSES = (
        "info@tech.gov.sg",
        "media@tech.gov.sg",
        "whistleblow@tech.gov.sg",
        "contact@ns.gov.sg",
    )
    TRUSTED_EMAIL_DOMAINS = (
        "gov.sg",
    )
    TRUSTED_PHONE_NUMBERS = (
        "1799",
        "62110888",
        "6562110888",
        "63353533",
        "6563353533",
        "18002550000",
        "18003568300",
        "6563568300",
        "18002271188",
        "6562271188",
        "18002223399",
        "18002263866",
        "18003676767",
        "65676767",
    )
    TRUSTED_URL_DOMAINS = (
        "gov.sg",
    )
    MASK_TRUSTED_CONTACTS_FOR_MODEL = True
    MASK_ALL_PHONE_NUMBERS_FOR_MODEL = True

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
            "account verification",
            "verify your identity",
            "identity",
            "account freeze",
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
            "initial deposit",
            "activation fee",
            "transfer funds",
            "safe account",
            "investment",
            "funds",
            "double your money",
        ],
        "call_to_action": [
            "click here",
            "click",
            "open link",
            "link",
            "update account",
            "confirm account",
            "reset account",
            "contact us",
            "call now",
            "message now",
            "whatsapp",
            "telegram",
            "action required",
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
            "giveaway",
            "voucher",
            "nric",
        ],
        "delivery_scam": [
            "parcel",
            "tracking",
            "customs",
            "clearance fee",
            "returned to sender",
            "held at singapore customs",
        ],
        "job_scam": [
            "work from home",
            "easy online jobs",
            "high pay",
            "activation fee",
        ],
    }

    KEYWORD_HIT_WEIGHT = 0.1
    URL_PRESENT_WEIGHT = 0.2
    EMAIL_PRESENT_WEIGHT = 0.1
    PHONE_PRESENT_WEIGHT = 0.05
    MONEY_PRESENT_WEIGHT = 0.15
    EMPTY_TEXT_PENALTY = 0.5
    MULTI_SIGNAL_BONUS = 0.08
    LINK_CREDENTIAL_BONUS = 0.1
    TEXT_RULE_WEIGHT = 0.2
    TEXT_MODEL_WEIGHT = 0.8
    MODEL_POSITIVE_THRESHOLD = 0.6
    BENIGN_CONTEXT_HIT_WEIGHT = 0.12
    BENIGN_CONTEXT_MAX_PENALTY = 0.45
    BENIGN_MODEL_PENALTY_MULTIPLIER = 0.9
    TRUSTED_SIGNAL_BENIGN_BONUS = 0.12
    TEXT_RELEVANCE_MIN_SCORE = 2.0
    TEXT_RELEVANCE_MAX_CHUNKS = 5
    TEXT_RELEVANCE_STRONG_SIGNALS = (
        "urgent", "verify", "verification", "identity", "account", "password", "otp",
        "pin", "login", "sign in", "click", "link", "reset", "update", "confirm",
        "whatsapp", "bank", "payment", "refund", "invoice", "claim", "reward",
        "bonus", "free", "gift", "prize", "frozen", "suspended", "expired",
        "singpass", "iras", "cpf", "limited", "action required", "official notice",
        "parcel", "tracking", "customs", "clearance fee", "returned to sender",
        "giveaway", "voucher", "nric", "initial deposit", "investment",
        "work from home", "activation fee", "safe account", "money laundering",
        "telegram", "t.me",
    )
    TEXT_RELEVANCE_WEAK_SIGNALS = (
        "call", "message", "now", "today", "immediately", "security", "alert",
        "notice", "government", "profile", "transaction", "money", "cash",
        "deposit", "sponsored", "salary", "recruitment", "register",
    )
    PHONE_RISK_CONTEXT_SIGNALS = (
        "call now", "contact us", "whatsapp", "telegram", "verify", "verification",
        "otp", "password", "login", "sign in", "bank", "payment", "transfer",
        "refund", "claim", "reward", "prize", "delivery", "parcel", "customs",
        "support", "hotline", "click", "link", "urgent", "immediately",
    )
    PHONE_BENIGN_CONTEXT_SIGNALS = (
        "if not authorised", "do not reply", "appointment", "healthhub",
        "your bill", "view your bill", "auto-payment", "auto payment",
        "card ending", "used at", "login was successful", "face verification login",
        "scam alert", "be wary", "if in doubt", "government officials will never",
        "verified channel", "reminder", "transaction alert", "never ask",
        "will never ask", "impersonating", "not done by you", "successful on",
        "automated notification", "current balance", "paynow transfer",
        "received a paynow transfer", "report at",
    )
    TEXT_RELEVANCE_STRONG_SIGNALS_ZH = (
        "验证", "验证码", "账户", "账号", "密码", "登录", "点击", "链接", "中奖",
        "奖励", "退款", "银行", "付款", "转账", "官方", "通知", "立即", "紧急", "客服",
    )
    TEXT_RELEVANCE_STRONG_SIGNALS_ZH = TEXT_RELEVANCE_STRONG_SIGNALS_ZH + (
        "\u5728\u5bb6\u5de5\u4f5c",
        "\u517c\u804c",
        "\u62db\u8058",
        "\u9ad8\u85aa",
    )
    BENIGN_CONTEXT_SIGNALS = (
        "if not authorised", "do not reply", "appointment", "reminder",
        "your bill", "view your bill", "auto-payment", "auto payment",
        "card ending", "used at", "login was successful", "face verification login",
        "scam alert", "be wary", "if in doubt", "government officials will never",
        "verified channel", "there are no fees", "official warning", "never ask",
        "will never ask", "impersonating", "not done by you", "successful on",
    )
    BENIGN_CONTEXT_SIGNALS = BENIGN_CONTEXT_SIGNALS + (
        "automated notification",
        "current balance",
        "paynow transfer",
        "received a paynow transfer",
        "your order",
        "estimated arrival",
        "enjoy your meal",
        "full story",
        "full story at",
        "report at",
    )
    BENIGN_PENALTY_BLOCKERS = (
        "whatsapp",
        "telegram",
        "click",
        "open link",
        "action required",
        "verify your identity",
        "account verification",
        "account freeze",
        "customs",
        "clearance fee",
        "returned to sender",
        "initial deposit",
        "double your money",
        "legal action",
        "suspended",
    )
    CHINESE_ROUTE_MIN_CHAR_COUNT = 4
    CHINESE_ROUTE_MIN_CHAR_RATIO = 0.2
    URL_TLD_ALLOWLIST = (
        "app", "biz", "co", "com", "edu", "gov", "info", "io", "me", "net", "org", "sg",
    )
    TEXT_RELEVANCE_NOISE_HINTS = (
        "channelnewsasia", "mothership", "straits times", "cna", "readmore",
        "prime minister", "singapore", "exclusive", "breaking", "news",
        "cabinet", "leadership", "diplomatic", "community chest",
    )

    IMAGE_SCORE_WEIGHT = 0.35
    TEXT_SCORE_WEIGHT = 0.65
    MEDIUM_RISK_THRESHOLD = 0.75
    HIGH_RISK_THRESHOLD = 0.7
