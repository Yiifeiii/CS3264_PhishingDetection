from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import sys
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config


DEFAULT_PROMPT = (
    "Extract text from this image.\n"
    "If the image is a chat, messaging app, SMS, WhatsApp, Telegram, iMessage, or similar conversation screenshot, "
    "return only the actual sender/user message text body.\n"
    "Do not include UI chrome such as timestamps, phone numbers in headers, online/typing status, signal bars, "
    "battery, keyboard letters, input box text, 'message', 'report junk', 'pinned message', or other app interface labels.\n"
    "If it is not a chat screenshot, extract all visible text in reading order.\n"
    "Return only the extracted text. Do not explain anything. Do not add markdown."
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
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
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
    return content


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "model", "text", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    rows: list[dict] = []
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        try:
            image_b64 = encode_image(image_path)
            extracted_text = call_ollama_vision(
                host=args.host,
                model=args.model,
                prompt=args.prompt,
                image_b64=image_b64,
                timeout_seconds=args.timeout_seconds,
            )
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

    if args.output:
        output_path = Path(args.output)
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
    elif len(rows) == 1:
        print()
        print(rows[0]["text"])

    success_count = sum(1 for row in rows if not row["error"])
    print()
    print(f"Processed {len(rows)} image(s). Successful: {success_count}. Failed: {len(rows) - success_count}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
