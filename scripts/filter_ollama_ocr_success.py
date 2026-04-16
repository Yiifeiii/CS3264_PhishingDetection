from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only successful OCR rows from an Ollama OCR CSV."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Source OCR CSV with path, model, text, error columns.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV containing only rows with empty error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ["path", "model", "text", "error"])
        rows = list(reader)

    success_rows = [
        row for row in rows
        if not str(row.get("error") or "").strip()
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(success_rows)

    print(f"Source rows: {len(rows)}")
    print(f"Successful rows kept: {len(success_rows)}")
    print(f"Removed error rows: {len(rows) - len(success_rows)}")
    print(f"Saved filtered CSV to: {output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
