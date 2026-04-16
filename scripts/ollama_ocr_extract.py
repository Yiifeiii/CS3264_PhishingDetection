from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import re
import sys
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


DEFAULT_PROMPT = (
    "Transcribe the visible text from this image as faithfully as possible.\n"
    "Do not paraphrase, summarize, translate, correct spelling, or explain.\n"
    "Preserve original wording, suspicious spelling, URLs, phone numbers, codes, amounts, usernames, and punctuation exactly as shown whenever possible.\n"
    "If the image is a chat, messaging app, SMS, WhatsApp, Telegram, iMessage, or similar conversation screenshot, "
    "return only the actual sender message body text from the conversation content.\n"
    "Exclude app UI and chrome such as header phone numbers, profile names outside the message body, timestamps, online/typing status, "
    "signal bars, battery, keyboard letters, input box text, 'message', 'report junk', 'pinned message', contact banners, "
    "'the sender is not in your contact list', encryption banners, and footer buttons.\n"
    "If it is not a chat screenshot, return all visible text in reading order.\n"
    "Return a JSON object with exactly one key: text.\n"
    'Example schema: {"text": "..." }\n'
    "The text value must contain only the transcription itself. No markdown. No quotes around the whole answer. "
    "No prefaces like 'The extracted text is'."
)

SYSTEM_PROMPT = (
    "You are a strict OCR transcription engine. "
    "You must only return JSON matching the provided schema. "
    "Do not describe the image. Do not explain. Do not summarize."
)

OCR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}

LEADING_WRAPPER_PATTERNS = (
    re.compile(r"^\s*the extracted text is\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the text in the image reads\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*the image contains(?: the following)? text\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*verbatim transcription\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*transcription\s*:?\s*", re.IGNORECASE),
    re.compile(r"^\s*extracted text\s*:?\s*", re.IGNORECASE),
)

INLINE_UI_PHRASES = (
    "the sender is not in your contact list",
    "report junk",
)

UI_NOISE_PATTERNS = (
    re.compile(r"^\s*(report junk|message|messages|view profile|pinned message|learn more|block|delete|accept)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(online|typing|active now|last seen just now|business account|text message|sms|imessage)\s*$", re.IGNORECASE),
    re.compile(r"^\s*messages? and calls are secured with end-to-end encryption\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*only people in this chat can read,? listen,? or share them\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:\d{1,2}:\d{2}\s*(?:AM|PM)?\s+){2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:[A-Z]\s+){6,}[A-Z]?\s*$"),
    re.compile(r"^\s*(?:Q W E R T Y|A S D F G H J K L|Z X C V B N M).*$", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a local Ollama vision model as OCR / text extraction for one image or a directory of images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory of images to process.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file. Use .json or .csv for batch results, or .txt for a single image.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2-vision",
        help="Ollama vision model name. Example: llama3.2-vision, llava, gemma3.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent to the vision model.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders when --input is a directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to process.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="HTTP timeout for each Ollama request.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV output by skipping already processed image paths.",
    )
    parser.add_argument(
        "--disable-cleaning",
        action="store_true",
        help="Keep the raw model response instead of applying output cleanup.",
    )
    return parser.parse_args()


def collect_images(input_path: Path, recursive: bool, limit: int | None) -> list[Path]:
    cfg = Config()
    allowed_exts = {ext.lower() for ext in cfg.SUPPORTED_IMAGE_EXTENSIONS}

    if input_path.is_file():
        if input_path.suffix.lower() not in allowed_exts:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Missing input path: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in allowed_exts
    )
    if limit is not None:
        files = files[:limit]
    return files


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def clean_ollama_output(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    cleaned = re.sub(r"^\s*```(?:text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    for pattern in LEADING_WRAPPER_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()

    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        for phrase in INLINE_UI_PHRASES:
            line = re.sub(re.escape(phrase), "", line, flags=re.IGNORECASE).strip(" -|:")

        line = re.sub(r"(?:\s*\.\s*){2,}", ".", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if not line:
            continue

        if re.fullmatch(r"[.\-_|:;,\s]+", line):
            continue

        if any(pattern.match(line) for pattern in UI_NOISE_PATTERNS):
            continue

        if lines and line == lines[-1]:
            continue

        lines.append(line)

    result = "\n".join(lines).strip()

    for pattern in LEADING_WRAPPER_PATTERNS:
        result = pattern.sub("", result).strip()

    return result


def call_ollama_vision(
    host: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout_seconds: int,
) -> str:
    endpoint = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "format": OCR_RESPONSE_SCHEMA,
        "options": {
            "temperature": 0,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(http_request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    message = payload.get("message") or {}
    content = str(message.get("content") or payload.get("response") or "").strip()

    try:
        parsed = json.loads(content)
        text_value = parsed.get("text")
        if isinstance(text_value, str):
            return text_value.strip()
    except json.JSONDecodeError:
        pass

    return content


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "model", "text", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "model", "text", "error"]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_successful_paths_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()

    processed: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row.get("path") or "").strip()
            error_text = str(row.get("error") or "").strip()
            if image_path and not error_text:
                processed.add(image_path)
    return processed


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_single_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    image_paths = collect_images(input_path, args.recursive, args.limit)
    if not image_paths:
        raise ValueError(f"No supported images found under: {input_path}")

    output_path = Path(args.output) if args.output else None
    incremental_csv = bool(output_path and output_path.suffix.lower() == ".csv")
    if args.resume and not incremental_csv:
        raise ValueError("--resume currently requires --output with a .csv file.")

    if args.resume and output_path is not None:
        processed_paths = load_successful_paths_from_csv(output_path)
        if processed_paths:
            image_paths = [
                path for path in image_paths
                if str(path) not in processed_paths
            ]
            print(f"Resume mode: skipping {len(processed_paths)} image(s) with an existing successful OCR row.")
        if not image_paths:
            print("Nothing left to process.")
            print(f"Existing output: {output_path.resolve()}")
            return 0

    rows: list[dict] = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] Processing: {image_path}")
        try:
            image_b64 = encode_image(image_path)
            extracted_text = call_ollama_vision(
                host=args.host,
                model=args.model,
                prompt=args.prompt,
                image_b64=image_b64,
                timeout_seconds=args.timeout_seconds,
            )
            if not args.disable_cleaning:
                extracted_text = clean_ollama_output(extracted_text)
            rows.append(
                {
                    "path": str(image_path),
                    "model": args.model,
                    "text": extracted_text,
                    "error": "",
                }
            )
            preview = extracted_text[:160].replace("\n", " ")
            print(f"  extracted: {preview}")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            rows.append(
                {
                    "path": str(image_path),
                    "model": args.model,
                    "text": "",
                    "error": f"HTTP {exc.code}: {message}",
                }
            )
            print(f"  error: HTTP {exc.code}")
        except Exception as exc:
            rows.append(
                {
                    "path": str(image_path),
                    "model": args.model,
                    "text": "",
                    "error": str(exc),
                }
            )
            print(f"  error: {exc}")

        if incremental_csv and output_path is not None:
            append_csv_row(output_path, rows[-1])

    if output_path is not None and not incremental_csv:
        suffix = output_path.suffix.lower()
        if suffix == ".csv":
            write_csv(output_path, rows)
        elif suffix == ".json":
            write_json(output_path, rows)
        elif suffix == ".txt":
            if len(rows) != 1:
                raise ValueError(".txt output only supports a single input image.")
            write_single_text(output_path, rows[0]["text"])
        else:
            raise ValueError("Output must end with .csv, .json, or .txt")
        print(f"Saved output to: {output_path.resolve()}")
    elif output_path is not None and incremental_csv:
        print(f"Saved output to: {output_path.resolve()}")
    elif len(rows) == 1:
        print()
        print(rows[0]["text"])

    success_count = sum(1 for row in rows if not row["error"])
    print()
    print(f"Processed {len(rows)} image(s). Successful: {success_count}. Failed: {len(rows) - success_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
