#!/usr/bin/env python3

"""Score Ego4D compositional start and end boundaries separately."""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .semantic_eval import DEFAULT_API_BASE, DEFAULT_MODEL
    from .unified import (
        BASE_SCORE_MODES,
        DIRECTION_MODES,
        MODES,
        PROJECT_ROOT,
        build_semantic_scorer,
        evaluate,
        load_annotations,
        load_predictions,
        resolve_annotation_path,
        resolve_pred_path,
    )
except ImportError:
    from semantic_eval import DEFAULT_API_BASE, DEFAULT_MODEL
    from unified import (
        BASE_SCORE_MODES,
        DIRECTION_MODES,
        MODES,
        PROJECT_ROOT,
        build_semantic_scorer,
        evaluate,
        load_annotations,
        load_predictions,
        resolve_annotation_path,
        resolve_pred_path,
    )


LABELS = ("start", "end")
_START_RE = re.compile(r"(?<!\w)start(?!\w)")
_END_RE = re.compile(r"(?<!\w)end(?!\w)")


def normalize_label_text(text: str) -> str:
    """Remove Unicode punctuation, lowercase, and normalize whitespace."""
    without_punctuation = "".join(
        character for character in str(text).casefold()
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def classify_label(text: str) -> Optional[str]:
    """Return start/end for an unambiguous normalized label, otherwise None."""
    normalized = normalize_label_text(text)
    matches = [label for label, pattern in (("start", _START_RE), ("end", _END_RE)) if pattern.search(normalized)]
    return matches[0] if len(matches) == 1 else None


def classify_prediction_label(text: str) -> Optional[str]:
    """Accept only a normalized standalone start/end prediction."""
    normalized = normalize_label_text(text)
    return normalized if normalized in LABELS else None


def split_outputs_by_label(outputs_by_key: dict) -> Tuple[Dict[str, dict], int]:
    """Route each valid prediction exclusively to its start/end label group."""
    groups: Dict[str, dict] = {label: {} for label in LABELS}
    unmatched = 0
    for key, outputs in outputs_by_key.items():
        for output in outputs:
            label = classify_prediction_label(output.get("content", ""))
            if label is None:
                unmatched += 1
                continue
            groups[label].setdefault(key, []).append(output)
    return groups, unmatched


def split_samples(samples: List[dict]) -> Tuple[Dict[str, List[dict]], int]:
    """Split GT samples by their response content and count unmatched samples."""
    groups: Dict[str, List[dict]] = {label: [] for label in LABELS}
    unmatched = 0
    for sample in samples:
        if sample.get("requires_no_output"):
            unmatched += 1
            continue
        label = classify_label(sample.get("reference", ""))
        if label is None:
            unmatched += 1
        else:
            groups[label].append({**sample, "label": label})
    return groups, unmatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Ego4D compositional start/end boundaries separately.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=Path("output/vllm/Perception_Event_compositional"))
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--pred", type=Path, default=None)
    parser.add_argument("--mode", choices=MODES, default="default")
    parser.add_argument("--direction", choices=DIRECTION_MODES, default="single")
    parser.add_argument("--base-score", choices=BASE_SCORE_MODES, default="constant")
    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--semantic-api-key", default="EMPTY")
    parser.add_argument("--semantic-timeout", type=float, default=600.0)
    parser.add_argument("--details", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    if args.semantic_timeout <= 0:
        raise ValueError("--semantic-timeout must be > 0")
    if args.details is None:
        stem = f"compositional_{args.mode}" if args.direction == "single" else f"compositional_bidirectional_{args.mode}"
        args.details = args.input_dir / "score" / f"{stem}.json"
    if args.summary is None:
        stem = f"compositional_{args.mode}" if args.direction == "single" else f"compositional_bidirectional_{args.mode}"
        args.summary = args.input_dir / "score" / f"{stem}_summary.json"
    return args


def evaluate_compositional(
    samples: List[dict], outputs_by_question: dict, outputs_by_video: dict,
    mode: str, base_score: str = "constant", semantic_scorer=None, direction: str = "single",
) -> Tuple[dict, dict]:
    groups, unmatched = split_samples(samples)
    question_outputs, unmatched_outputs = split_outputs_by_label(outputs_by_question)
    video_outputs, _ = split_outputs_by_label(outputs_by_video)
    group_summaries: Dict[str, dict] = {}
    group_details: Dict[str, List[dict]] = {}
    for label in LABELS:
        summary, details = evaluate(
            groups[label], question_outputs[label], video_outputs[label],
            mode, base_score, semantic_scorer, direction=direction,
        )
        group_summaries[label] = summary
        group_details[label] = [
            {**video, "label": label}
            for video in details
        ]

    start_score = group_summaries["start"]["overall_score"]
    end_score = group_summaries["end"]["overall_score"]
    summary = {
        "overall_score": (start_score + end_score) / 2.0,
        "start_score": start_score,
        "end_score": end_score,
        "num_start_inputs": len(groups["start"]),
        "num_end_inputs": len(groups["end"]),
        "num_unmatched_inputs": unmatched,
        "num_unmatched_outputs": unmatched_outputs,
        "num_start_videos": group_summaries["start"]["num_videos"],
        "num_end_videos": group_summaries["end"]["num_videos"],
        "groups": group_summaries,
    }
    return summary, group_details


def _output_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    annotation_path = resolve_annotation_path(root, args.input_dir, args.annotation)
    pred_path = resolve_pred_path(root, args.input_dir, args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    samples, question_to_video = load_annotations(annotation_path)
    outputs_by_question, outputs_by_video = load_predictions(pred_path, question_to_video)
    scorer = (
        build_semantic_scorer(args.semantic_model, args.semantic_api_base, args.semantic_api_key, args.semantic_timeout)
        if args.base_score == "semantic" else None
    )
    summary, details = evaluate_compositional(
        samples, outputs_by_question, outputs_by_video, args.mode, args.base_score, scorer, args.direction,
    )
    result = {
        "evaluator": "compositional_label",
        "mode": args.mode,
        "direction": args.direction,
        "base_score": args.base_score,
        "semantic_config": None if args.base_score != "semantic" else {
            "model": args.semantic_model,
            "api_base": args.semantic_api_base,
            "timeout": args.semantic_timeout,
        },
        "annotation_path": str(annotation_path),
        "pred_path": str(pred_path),
        **summary,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    for path_arg, payload in ((args.details, details), (args.summary, result)):
        path = _output_path(root, path_arg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
