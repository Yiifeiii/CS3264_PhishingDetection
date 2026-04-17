from __future__ import annotations

from typing import Any


MODEL_REASON_MARKERS = (
    "distilbert model",
    "chinese text model",
    "text model",
    "model strongest text span",
    "trusted contacts masked before phishing-model scoring",
    "all detected phone numbers masked before phishing-model scoring",
    "warning:",
)
TEXT_DECISION_SOURCE_CHOICES = ("combined", "model", "model_raw")


def resolve_text_score_key(decision_source: str | None) -> str:
    normalized = str(decision_source or "combined").strip().lower()
    mapping = {
        "combined": "score",
        "model": "model_score",
        "model_raw": "model_score_raw",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported text decision source '{decision_source}'. "
            f"Expected one of: {', '.join(TEXT_DECISION_SOURCE_CHOICES)}"
        )
    return mapping[normalized]


def select_text_decision_score(
    text_result: dict[str, Any],
    decision_source: str | None,
) -> float:
    normalized = str(decision_source or "combined").strip().lower()
    fallback_keys = {
        "combined": ("score", "model_score", "model_score_raw", "rule_score"),
        "model": ("model_score", "score", "model_score_raw", "rule_score"),
        "model_raw": ("model_score_raw", "model_score", "score", "rule_score"),
    }
    if normalized not in fallback_keys:
        resolve_text_score_key(normalized)

    for key in fallback_keys[normalized]:
        value = text_result.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _region_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, int]:
    analysis = candidate["analysis"]
    processed_text = str(candidate["processed_text"] or "")
    model_score_raw = analysis.get("model_score_raw")
    text_score = analysis.get("score")
    return (
        float(model_score_raw if model_score_raw is not None else -1.0),
        float(text_score if text_score is not None else -1.0),
        len(processed_text),
    )


def _use_grounded_region_max(ocr) -> bool:
    return bool(
        getattr(ocr, "backend", "") == "easyocr"
        and getattr(ocr, "easyocr_use_grounding_dino", False)
        and getattr(ocr, "easyocr_grounding_text_aggregation", "concat") in {"max_model", "hybrid_max_model"}
    )


def _grounded_aggregation_mode(ocr) -> str:
    if not bool(
        getattr(ocr, "backend", "") == "easyocr"
        and getattr(ocr, "easyocr_use_grounding_dino", False)
    ):
        return "concat"
    return str(getattr(ocr, "easyocr_grounding_text_aggregation", "concat") or "concat").strip().lower()


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for reason in reasons:
        normalized = str(reason or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _is_model_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    return any(marker in normalized for marker in MODEL_REASON_MARKERS)


def _non_model_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if not _is_model_reason(reason)]


def _model_only_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if _is_model_reason(reason)]


def _combine_scores(analyzer, rule_score: float | None, model_score: float | None) -> float:
    normalized_rule = float(rule_score or 0.0)
    if model_score is None:
        return round(min(normalized_rule, 1.0), 4)
    combined = (
        float(getattr(analyzer.cfg, "TEXT_RULE_WEIGHT", 0.0)) * normalized_rule
        + float(getattr(analyzer.cfg, "TEXT_MODEL_WEIGHT", 0.0)) * float(model_score)
    )
    return round(min(combined, 1.0), 4)


def analyze_text_with_optional_model_text(
    analyzer,
    heuristic_text: str,
    model_text: str | None = None,
    *,
    aggregation_mode: str = "concat",
) -> dict[str, Any]:
    heuristic_text = str(heuristic_text or "")
    model_text = str(model_text or "")

    heuristic_analysis = analyzer.analyze(heuristic_text)
    if not model_text.strip() or model_text.strip() == heuristic_text.strip():
        heuristic_analysis["grounded_region_aggregation"] = aggregation_mode
        return heuristic_analysis

    model_analysis = analyzer.analyze(model_text)
    merged = dict(heuristic_analysis)
    merged["grounded_region_aggregation"] = aggregation_mode
    merged["score"] = _combine_scores(
        analyzer,
        heuristic_analysis.get("rule_score"),
        model_analysis.get("model_score"),
    )
    merged["model_score"] = model_analysis.get("model_score")
    merged["model_score_raw"] = model_analysis.get("model_score_raw")
    merged["model_loaded"] = model_analysis.get("model_loaded")
    merged["english_model_loaded"] = model_analysis.get("english_model_loaded")
    merged["chinese_model_loaded"] = model_analysis.get("chinese_model_loaded")
    merged["model_route"] = model_analysis.get("model_route")
    merged["model_scored_text"] = model_analysis.get("model_scored_text")
    merged["model_chunks_evaluated"] = model_analysis.get("model_chunks_evaluated")
    merged["model_best_chunk"] = model_analysis.get("model_best_chunk")
    merged["model_input_text"] = model_analysis.get("model_input_text")
    merged["masked_trusted_contacts"] = model_analysis.get("masked_trusted_contacts") or []
    merged["masked_phone_numbers_for_model"] = model_analysis.get("masked_phone_numbers_for_model") or []
    merged["model_filtered_text"] = model_analysis.get("filtered_text") or ""
    merged["model_relevant_chunks"] = model_analysis.get("relevant_chunks") or []
    merged["model_dropped_chunks"] = model_analysis.get("dropped_chunks") or []
    merged["warnings"] = list(model_analysis.get("warnings") or [])
    merged["reasons"] = _dedupe_reasons(
        _non_model_reasons(list(heuristic_analysis.get("reasons") or []))
        + _model_only_reasons(list(model_analysis.get("reasons") or []))
        + ["heuristics used full OCR text while model score used the strongest grounded crop"]
    )
    return merged


def _collect_region_candidates(ocr, processor, analyzer) -> list[dict[str, Any]]:
    region_candidates = []
    grounded_regions = list(getattr(ocr, "last_grounding_dino_regions", []) or [])
    for index, region in enumerate(grounded_regions, start=1):
        region_raw_text = str(region.get("ocr_text") or "").strip()
        if not region_raw_text:
            continue

        processed = processor.process(region_raw_text)
        processed_text = str(processed.get("text") or "").strip()
        if not processed_text:
            continue

        analysis = analyzer.analyze(processed_text)
        if processed.get("warning"):
            analysis.setdefault("warnings", []).append(processed["warning"])
        analysis["ocr_processing"] = processed

        region_candidates.append(
            {
                "index": index,
                "box": region.get("box"),
                "label": region.get("label"),
                "score": float(region.get("score") or 0.0),
                "raw_text": region_raw_text,
                "processed": processed,
                "processed_text": processed_text,
                "analysis": analysis,
            }
        )
    return region_candidates


def _region_candidate_summaries(region_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": candidate["index"],
            "box": candidate["box"],
            "label": candidate["label"],
            "detection_score": round(candidate["score"], 4),
            "model_score_raw": candidate["analysis"].get("model_score_raw"),
            "text_score": candidate["analysis"].get("score"),
            "processed_text_preview": (
                candidate["processed_text"][:160] + "..."
                if len(candidate["processed_text"]) > 160
                else candidate["processed_text"]
            ),
        }
        for candidate in region_candidates
    ]


def run_text_pipeline_on_image(ocr, processor, analyzer, image_path: str) -> dict[str, Any]:
    raw_text = ocr.extract_text(str(image_path))
    aggregation_mode = _grounded_aggregation_mode(ocr)

    if _use_grounded_region_max(ocr):
        region_candidates = _collect_region_candidates(ocr, processor, analyzer)
        if region_candidates:
            best_candidate = max(region_candidates, key=_region_candidate_sort_key)
            if aggregation_mode == "hybrid_max_model":
                processed = processor.process(raw_text)
                processed_text = str(processed.get("text") or "").strip()
                text_result = analyze_text_with_optional_model_text(
                    analyzer,
                    processed_text,
                    best_candidate["processed_text"],
                    aggregation_mode=aggregation_mode,
                )
                if processed.get("warning"):
                    text_result.setdefault("warnings", []).append(processed["warning"])
                text_result["ocr_processing"] = processed
                text_result["model_ocr_processing"] = best_candidate["processed"]
                text_result["model_raw_text"] = best_candidate["raw_text"]
                text_result["model_processed_text"] = best_candidate["processed_text"]
            else:
                processed = best_candidate["processed"]
                processed_text = best_candidate["processed_text"]
                text_result = dict(best_candidate["analysis"])
                text_result["grounded_region_aggregation"] = aggregation_mode
                text_result["ocr_processing"] = processed

            text_result["grounded_region_candidate_count"] = len(region_candidates)
            text_result["grounded_region_selected_index"] = best_candidate["index"]
            text_result["grounded_region_selected_box"] = best_candidate["box"]
            text_result["grounded_region_selected_label"] = best_candidate["label"]
            text_result["grounded_region_selected_detection_score"] = round(best_candidate["score"], 4)
            text_result["grounded_region_candidates"] = _region_candidate_summaries(region_candidates)
            return {
                "raw_text": raw_text if aggregation_mode == "hybrid_max_model" else best_candidate["raw_text"],
                "processed": processed,
                "processed_text": processed_text,
                "text_result": text_result,
                "combined_raw_text": raw_text,
                "used_grounded_region_max": True,
                "model_raw_text": best_candidate["raw_text"],
                "model_processed": best_candidate["processed"],
                "model_processed_text": best_candidate["processed_text"],
            }

    processed = processor.process(raw_text)
    processed_text = str(processed.get("text") or "")
    text_result = analyze_text_with_optional_model_text(
        analyzer,
        processed_text,
        None,
        aggregation_mode=aggregation_mode,
    )
    if processed.get("warning"):
        text_result.setdefault("warnings", []).append(processed["warning"])
    text_result["ocr_processing"] = processed
    return {
        "raw_text": raw_text,
        "processed": processed,
        "processed_text": processed_text,
        "text_result": text_result,
        "combined_raw_text": raw_text,
        "used_grounded_region_max": False,
        "model_raw_text": raw_text,
        "model_processed": processed,
        "model_processed_text": processed_text,
    }
