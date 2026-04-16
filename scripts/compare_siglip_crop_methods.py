import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two per-image crop-stream SigLIP result CSVs on the same image set."
    )
    parser.add_argument("--left-csv", required=True, help="First per_image_predictions.csv path.")
    parser.add_argument("--right-csv", required=True, help="Second per_image_predictions.csv path.")
    parser.add_argument("--left-name", default="ocr", help="Display name for the first method.")
    parser.add_argument("--right-name", default="grounding_dino", help="Display name for the second method.")
    parser.add_argument(
        "--assume-all-fake",
        action="store_true",
        help="Interpret fake-prediction rate as accuracy/recall because every image is phishing.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary output path.")
    return parser.parse_args()


def load_rows(path: Path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_path"]: row for row in rows}


def to_int(row, key: str):
    return int(row.get(key) or 0)


def to_float(row, key: str):
    return float(row.get(key) or 0.0)


def main():
    args = parse_args()
    left_path = Path(args.left_csv).resolve()
    right_path = Path(args.right_csv).resolve()

    left_rows = load_rows(left_path)
    right_rows = load_rows(right_path)
    common_paths = sorted(set(left_rows) & set(right_rows))
    if not common_paths:
        raise ValueError("The two CSVs do not share any image_path values.")

    left_pred_fake = 0
    right_pred_fake = 0
    both_fake = 0
    both_real = 0
    left_only_fake = 0
    right_only_fake = 0
    score_deltas = []
    disagreements = []

    for image_path in common_paths:
        left = left_rows[image_path]
        right = right_rows[image_path]
        left_pred = to_int(left, "pred_label_id")
        right_pred = to_int(right, "pred_label_id")
        left_score = to_float(left, "max_fake_probability")
        right_score = to_float(right, "max_fake_probability")
        score_delta = right_score - left_score

        left_pred_fake += int(left_pred == 1)
        right_pred_fake += int(right_pred == 1)
        both_fake += int(left_pred == 1 and right_pred == 1)
        both_real += int(left_pred == 0 and right_pred == 0)
        left_only_fake += int(left_pred == 1 and right_pred == 0)
        right_only_fake += int(left_pred == 0 and right_pred == 1)
        score_deltas.append(score_delta)

        if left_pred != right_pred:
            disagreements.append(
                {
                    "image_path": image_path,
                    "left_pred_label_id": left_pred,
                    "right_pred_label_id": right_pred,
                    "left_max_fake_probability": left_score,
                    "right_max_fake_probability": right_score,
                    "right_minus_left_probability": score_delta,
                    "left_crop_count": to_int(left, "crop_count"),
                    "right_crop_count": to_int(right, "crop_count"),
                }
            )

    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_csv": str(left_path),
        "right_csv": str(right_path),
        "common_image_count": len(common_paths),
        "left_pred_fake_count": left_pred_fake,
        "right_pred_fake_count": right_pred_fake,
        "both_fake_count": both_fake,
        "both_real_count": both_real,
        "left_only_fake_count": left_only_fake,
        "right_only_fake_count": right_only_fake,
        "mean_right_minus_left_probability": statistics.fmean(score_deltas),
        "median_right_minus_left_probability": statistics.median(score_deltas),
        "disagreement_count": len(disagreements),
        "top_disagreements": sorted(
            disagreements,
            key=lambda row: abs(row["right_minus_left_probability"]),
            reverse=True,
        )[:20],
    }

    if args.assume_all_fake:
        summary[f"{args.left_name}_accuracy_all_fake"] = left_pred_fake / len(common_paths)
        summary[f"{args.right_name}_accuracy_all_fake"] = right_pred_fake / len(common_paths)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as handle:
            json.dump(summary, handle, indent=2)

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# SigLIP Crop Method Comparison",
            "",
            f"- Left method: `{args.left_name}`",
            f"- Right method: `{args.right_name}`",
            f"- Common images: {len(common_paths)}",
            f"- `{args.left_name}` predicted fake: {left_pred_fake}",
            f"- `{args.right_name}` predicted fake: {right_pred_fake}",
            f"- Both fake: {both_fake}",
            f"- Both real: {both_real}",
            f"- `{args.left_name}` only fake: {left_only_fake}",
            f"- `{args.right_name}` only fake: {right_only_fake}",
            f"- Mean right-minus-left fake probability: {summary['mean_right_minus_left_probability']:.4f}",
            f"- Median right-minus-left fake probability: {summary['median_right_minus_left_probability']:.4f}",
        ]
        if args.assume_all_fake:
            lines.extend(
                [
                    f"- `{args.left_name}` all-fake accuracy: {summary[f'{args.left_name}_accuracy_all_fake']:.4f}",
                    f"- `{args.right_name}` all-fake accuracy: {summary[f'{args.right_name}_accuracy_all_fake']:.4f}",
                ]
            )
        lines.extend(
            [
                "",
                "## Top Disagreements",
                "| Image | Left Pred | Right Pred | Left Prob | Right Prob | Left Crops | Right Crops |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["top_disagreements"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["image_path"],
                        str(row["left_pred_label_id"]),
                        str(row["right_pred_label_id"]),
                        f"{row['left_max_fake_probability']:.4f}",
                        f"{row['right_max_fake_probability']:.4f}",
                        str(row["left_crop_count"]),
                        str(row["right_crop_count"]),
                    ]
                )
                + " |"
            )
        with open(output_md, "w") as handle:
            handle.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
