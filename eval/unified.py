#!/usr/bin/env python3

"""Evaluate scores with constant, strict, or semantic reply base scores."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .semantic_eval import DEFAULT_API_BASE, DEFAULT_MODEL, evaluate_semantic_score
except ImportError:
    from semantic_eval import DEFAULT_API_BASE, DEFAULT_MODEL, evaluate_semantic_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
MODES = ("default", "first_only")
BASE_SCORE_MODES = ("constant", "strict", "semantic")
DIRECTION_MODES = ("single", "bidirectional")
Output = Dict[str, Any]
SemanticScorer = Callable[[str, str], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate interval scores with selectable numerator rules.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory.")
    parser.add_argument("--input-dir", type=Path, default=Path("output/vllm/Prediction_Crime/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--pred", type=Path, default=None)
    parser.add_argument("--mode", choices=MODES, default="default", help="Numerator rule.")
    parser.add_argument(
        "--direction",
        choices=DIRECTION_MODES,
        default="single",
        help="Assign outputs by single-direction GT intervals or nearest GT in both directions.",
    )
    parser.add_argument("--base-score", choices=BASE_SCORE_MODES, default="constant")
    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--semantic-api-key", default="EMPTY")
    parser.add_argument("--semantic-timeout", type=float, default=600.0)
    parser.add_argument("--semantic-scores", type=Path, default=None,
                        help="Persistent per-reply semantic score JSON path.")
    parser.add_argument("--details", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    if args.semantic_timeout <= 0:
        raise ValueError("--semantic-timeout must be > 0")
    if args.details is None:
        stem = args.mode if args.direction == "single" else f"bidirectional_{args.mode}"
        args.details = args.input_dir / "score" / f"{stem}.json"
    if args.summary is None:
        stem = args.mode if args.direction == "single" else f"bidirectional_{args.mode}"
        args.summary = args.input_dir / "score" / f"{stem}_summary.json"
    if args.semantic_scores is None:
        stem = "semantic" if args.direction == "single" else "bidirectional_semantic"
        args.semantic_scores = args.input_dir / "score" / f"{stem}.json"
    return args


def resolve_annotation_path(root: Path, input_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit if explicit.is_absolute() else root / explicit
    candidates = [root / input_dir / "inputs" / "annotation.json", root / input_dir / "inputs" / "annotations.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Annotation file not found. Tried: {[str(path) for path in candidates]}")


def resolve_pred_path(root: Path, input_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit if explicit.is_absolute() else root / explicit
    return root / input_dir / "pred.jsonl"


def normalize_video_id(video_name: Optional[str], question_id: str) -> str:
    return Path(video_name).stem if video_name else question_id


def load_annotations(annotation_path: Path) -> Tuple[List[dict], Dict[str, str]]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("videos", [])
    if not isinstance(data, list):
        raise ValueError("Annotation must be a list or an object containing a videos list")
    samples: List[dict] = []
    question_to_video: Dict[str, str] = {}
    for item in data:
        question_id = item["question_id"]
        video_id = normalize_video_id(item.get("video"), question_id)
        question_to_video[question_id] = video_id
        responses = sorted(
            ({"time": float(entry["time"]), "content": str(entry.get("content", ""))}
             for entry in item.get("response") or []
             if isinstance(entry, dict) and isinstance(entry.get("time"), (int, float))),
            key=lambda entry: entry["time"],
        )
        if not responses:
            samples.append({"question_id": question_id, "video_id": video_id, "gt_time": None, "requires_no_output": True})
            continue
        samples.extend({"question_id": question_id, "video_id": video_id, "gt_time": response["time"], "reference": response["content"]} for response in responses)
    return samples, question_to_video


def load_predictions(pred_path: Path, question_to_video: Dict[str, str]) -> Tuple[Dict[str, List[Output]], Dict[str, List[Output]]]:
    outputs_by_question: Dict[str, List[Output]] = defaultdict(list)
    outputs_by_video: Dict[str, List[Output]] = defaultdict(list)
    with pred_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            question_id = item["question_id"]
            video_id = question_to_video.get(question_id, question_id)
            outputs = []
            for message in item.get("model_response_list", []):
                if message.get("role") != "assistant" or not isinstance(message.get("time"), (int, float)):
                    continue
                content = normalize_evaluation_output(message.get("content", ""))
                if content is not None:
                    outputs.append({"time": float(message["time"]), "content": content})
            outputs_by_question[question_id].extend(outputs)
            outputs_by_video[video_id].extend(outputs)
    for outputs in outputs_by_question.values():
        outputs.sort(key=lambda output: output["time"])
    for outputs in outputs_by_video.values():
        outputs.sort(key=lambda output: output["time"])
    return outputs_by_question, outputs_by_video


def compute_time_decay(actual: float, gt_time: float) -> float:
    return math.exp(-ALPHA * (actual - gt_time))


def normalize_strict_text(text: str) -> str:
    """Normalize case, punctuation-like wrappers, and whitespace for fixed labels."""
    alphanumeric_and_space = "".join(
        character for character in str(text).casefold()
        if character.isalnum() or character.isspace()
    )
    return " ".join(alphanumeric_and_space.split())


def normalize_evaluation_output(text: str) -> Optional[str]:
    """Keep the final answer while removing exact silence signals from evaluation."""
    content = str(text or "").replace("<|im_end|>", "").strip()
    folded = content.casefold()
    for marker in ("</think>", "</response>"):
        position = folded.rfind(marker)
        if position >= 0:
            content = content[position + len(marker):].strip()
            folded = content.casefold()
    if not content:
        return None
    return None if normalize_strict_text(content) in {"silence", "no reply", "no response"} else content


def strict_match_score(reference: str, prediction: str) -> float:
    return 1.0 if normalize_strict_text(reference) == normalize_strict_text(prediction) else 0.0


def assign_outputs_to_nearest_gt(gt_times: List[float], outputs: List[Output]) -> List[List[Output]]:
    assignments: List[List[Output]] = [[] for _ in gt_times]
    for output in outputs:
        nearest_index = min(
            range(len(gt_times)),
            key=lambda index: (abs(output["time"] - gt_times[index]), index),
        )
        assignments[nearest_index].append(output)
    return assignments


def build_semantic_scorer(model: str, api_base: str, api_key: str, timeout: float) -> SemanticScorer:
    cache: Dict[Tuple[str, str], float] = {}

    def score(reference: str, prediction: str) -> float:
        key = (reference, prediction)
        if key not in cache:
            cache[key] = evaluate_semantic_score(reference, prediction, model, api_base, api_key, timeout)
        return cache[key]

    return score


def _semantic_key(question_id: str, gt_time: float, output_time: float, reference: str, prediction: str) -> tuple:
    return (question_id, float(gt_time), float(output_time), reference, prediction)


def load_or_generate_semantic_scores(
    path: Path, samples: List[dict], outputs_by_question: Dict[str, List[Output]],
    model: str, api_base: str, api_key: str, timeout: float,
    outputs_by_video: Optional[Dict[str, List[Output]]] = None,
    direction: str = "single",
) -> Dict[tuple, float]:
    """Load persistent per-reply scores, or generate them once when absent."""
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload["scores"] if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise ValueError("semantic score file must contain a scores list")
            index: Dict[tuple, float] = {}
            for record in records:
                required = {"question_id", "gt_time", "reference", "output_time", "prediction", "semantic_score"}
                if not isinstance(record, dict) or not required.issubset(record):
                    raise ValueError("semantic score record is missing required fields")
                score = float(record["semantic_score"])
                if score not in (0.0, 0.5, 1.0):
                    raise ValueError("semantic_score must be 0, 0.5, or 1")
                key = _semantic_key(record["question_id"], record["gt_time"], record["output_time"], record["reference"], record["prediction"])
                index[key] = score
            return index
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid semantic score file: {path}") from error

    scorer = build_semantic_scorer(model, api_base, api_key, timeout)
    records = []
    index: Dict[tuple, float] = {}
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        if not sample.get("requires_no_output") and sample.get("gt_time") is not None:
            grouped[sample["video_id"]].append(sample)
    for inputs in grouped.values():
        inputs.sort(key=lambda item: item["gt_time"])
    for sample in samples:
        if sample.get("requires_no_output") or sample.get("gt_time") is None:
            continue
        question_id = sample["question_id"]
        reference = str(sample.get("reference", ""))
        gt_time = float(sample["gt_time"])
        ordered = grouped[sample["video_id"]]
        sample_index = ordered.index(sample)
        upper_bound = ordered[sample_index + 1]["gt_time"] if sample_index + 1 < len(ordered) else math.inf
        if direction == "bidirectional":
            assignments = assign_outputs_to_nearest_gt(
                [item["gt_time"] for item in ordered],
                (outputs_by_video or {}).get(sample["video_id"], []),
            )
            outputs = assignments[sample_index]
        else:
            outputs = outputs_by_question.get(question_id)
            if outputs is None and outputs_by_video is not None:
                outputs = outputs_by_video.get(sample["video_id"], [])
        for output in outputs or []:
            if direction == "single" and (output["time"] < gt_time or output["time"] >= upper_bound):
                continue
            prediction = output["content"]
            score = scorer(reference, prediction)
            key = _semantic_key(question_id, gt_time, output["time"], reference, prediction)
            index[key] = score
            records.append({"question_id": question_id, "gt_time": gt_time, "reference": reference,
                            "output_time": output["time"], "prediction": prediction, "semantic_score": score})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scores": records, "semantic_config": {"model": model, "api_base": api_base, "timeout": timeout}}, ensure_ascii=True, indent=2), encoding="utf-8")
    return index


def compute_input_score(
    mode: str, gt_time: float, lower_bound: float, upper_bound: float, outputs: List[Output],
    base_score: str = "constant", reference: str = "", semantic_scorer: Optional[SemanticScorer] = None,
    question_id: str = "", semantic_scores: Optional[Dict[tuple, float]] = None,
) -> dict:
    interval_outputs = [output for output in outputs if lower_bound <= output["time"] < upper_bound]
    output_scores = []
    for output in interval_outputs:
        actual = output["time"]
        if actual < gt_time:
            output_scores.append({"time": actual, "content": output["content"], "base_score": None, "time_decay": None, "output_score": 0.0})
            continue
        if base_score == "semantic":
            if semantic_scores is None and semantic_scorer is None:
                raise ValueError("semantic_scorer is required when base_score is semantic")
            if semantic_scores is not None:
                key = _semantic_key(question_id, gt_time, output["time"], reference, output["content"])
                if key not in semantic_scores:
                    raise ValueError(f"Missing semantic score for reply in question {question_id}")
                reply_base_score = semantic_scores[key]
            else:
                reply_base_score = semantic_scorer(reference, output["content"])
        elif base_score == "strict":
            reply_base_score = strict_match_score(reference, output["content"])
        else:
            reply_base_score = 1.0
        decay = compute_time_decay(actual, gt_time)
        output_scores.append({"time": actual, "content": output["content"], "base_score": reply_base_score, "time_decay": decay, "output_score": reply_base_score * decay})

    post_gt_indices = [index for index, score in enumerate(output_scores) if score["time"] >= gt_time]
    first_valid_index = post_gt_indices[0] if post_gt_indices else None
    numerator = sum(score["output_score"] for score in output_scores) if mode == "default" else (output_scores[first_valid_index]["output_score"] if first_valid_index is not None else 0.0)
    zero_score_outputs = (
        [score["time"] for index, score in enumerate(output_scores) if index != first_valid_index]
        if mode == "first_only" else [score["time"] for score in output_scores if score["output_score"] == 0.0]
    )
    result = {
        "interval_outputs": [score["time"] for score in output_scores],
        "valid_output_times": [score["time"] for score in output_scores if score["time"] >= gt_time],
        "early_output_times": [score["time"] for score in output_scores if score["time"] < gt_time],
        "first_valid_output_time": output_scores[first_valid_index]["time"] if first_valid_index is not None else None,
        "zero_score_output_times": zero_score_outputs,
        "num_interval_outputs": len(output_scores), "num_zero_score_outputs": len(zero_score_outputs),
        "input_score": numerator / len(output_scores) if output_scores else 0.0,
    }
    if base_score == "semantic":
        result["output_score_details"] = output_scores
    return result


def compute_bidirectional_input_score(
    mode: str,
    gt_time: float,
    outputs: List[Output],
    base_score: str = "constant",
    reference: str = "",
    semantic_scorer: Optional[SemanticScorer] = None,
    question_id: str = "",
    semantic_scores: Optional[Dict[tuple, float]] = None,
) -> dict:
    output_scores = []
    for output in outputs:
        actual = output["time"]
        if base_score == "semantic":
            if semantic_scores is None and semantic_scorer is None:
                raise ValueError("semantic_scorer is required when base_score is semantic")
            if semantic_scores is not None:
                key = _semantic_key(question_id, gt_time, actual, reference, output["content"])
                if key not in semantic_scores:
                    raise ValueError(f"Missing semantic score for reply in question {question_id}")
                reply_base_score = semantic_scores[key]
            else:
                reply_base_score = semantic_scorer(reference, output["content"])
        elif base_score == "strict":
            reply_base_score = strict_match_score(reference, output["content"])
        else:
            reply_base_score = 1.0
        output_scores.append({
            "time": actual,
            "content": output["content"],
            "base_score": reply_base_score,
            "time_decay": math.exp(-ALPHA * abs(actual - gt_time)),
            "output_score": reply_base_score * math.exp(-ALPHA * abs(actual - gt_time)),
        })

    if not output_scores:
        return {
            "assigned_output_times": [],
            "nearest_assigned_output_time": None,
            "input_score": 0.0,
            "num_assigned_outputs": 0,
        }
    nearest_output_index = min(
        range(len(output_scores)),
        key=lambda index: (abs(output_scores[index]["time"] - gt_time), output_scores[index]["time"]),
    )
    numerator = (
        sum(score["output_score"] for score in output_scores)
        if mode == "default" else
        output_scores[nearest_output_index]["output_score"]
    )
    result = {
        "assigned_output_times": [score["time"] for score in output_scores],
        "nearest_assigned_output_time": output_scores[nearest_output_index]["time"],
        "input_score": numerator / len(output_scores),
        "num_assigned_outputs": len(output_scores),
    }
    if base_score == "semantic":
        result["output_score_details"] = output_scores
    return result


def evaluate(
    samples: List[dict], outputs_by_question: Dict[str, List[Output]], outputs_by_video: Dict[str, List[Output]],
    mode: str, base_score: str = "constant", semantic_scorer: Optional[SemanticScorer] = None,
    semantic_scores: Optional[Dict[tuple, float]] = None,
    direction: str = "single",
) -> Tuple[dict, List[dict]]:
    if base_score not in BASE_SCORE_MODES:
        raise ValueError(f"Unsupported base score mode: {base_score}")
    if direction not in DIRECTION_MODES:
        raise ValueError(f"Unsupported direction: {direction}")
    grouped_inputs: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        grouped_inputs[sample["video_id"]].append(sample)
    per_video_scores: Dict[str, float] = {}
    details: List[dict] = []
    for video_id, inputs in sorted(grouped_inputs.items()):
        no_output_inputs = [item for item in inputs if item.get("requires_no_output")]
        sorted_inputs = sorted((item for item in inputs if not item.get("requires_no_output")), key=lambda item: item["gt_time"])
        input_results: List[dict] = []
        if direction == "bidirectional" and sorted_inputs:
            assignments = assign_outputs_to_nearest_gt(
                [sample["gt_time"] for sample in sorted_inputs], outputs_by_video.get(video_id, [])
            )
            for sample, assigned_outputs in zip(sorted_inputs, assignments):
                score = compute_bidirectional_input_score(
                    mode, sample["gt_time"], assigned_outputs, base_score,
                    sample.get("reference", ""), semantic_scorer, sample["question_id"], semantic_scores,
                )
                input_results.append({"question_id": sample["question_id"], "gt_time": sample["gt_time"], "interval": None, **score})
        else:
            for index, sample in enumerate(sorted_inputs):
                gt_time = sample["gt_time"]
                interval_start = 0.0 if index == 0 else gt_time
                next_gt = sorted_inputs[index + 1]["gt_time"] if index + 1 < len(sorted_inputs) else math.inf
                outputs = outputs_by_question.get(sample["question_id"], outputs_by_video.get(video_id, []))
                score = compute_input_score(mode, gt_time, interval_start, next_gt, outputs, base_score, sample.get("reference", ""), semantic_scorer, sample["question_id"], semantic_scores)
                input_results.append({"question_id": sample["question_id"], "gt_time": gt_time, "interval": [interval_start, None if math.isinf(next_gt) else next_gt], **score})
        for sample in no_output_inputs:
            question_outputs = outputs_by_question.get(sample["question_id"], [])
            input_results.append({
                "question_id": sample["question_id"], "gt_time": None, "interval": None,
                "interval_outputs": [], "valid_output_times": [], "early_output_times": [], "first_valid_output_time": None,
                "zero_score_output_times": [], "num_interval_outputs": 0, "num_zero_score_outputs": 0,
                "input_score": 1.0 if not question_outputs else 0.0, "requires_no_output": True, "num_outputs": len(question_outputs),
            })
        if not input_results:
            continue
        video_score = sum(item["input_score"] for item in input_results) / len(input_results)
        per_video_scores[video_id] = video_score
        details.append({"video_id": video_id, "video_score": video_score, "num_inputs": len(input_results), "num_total_outputs": len(outputs_by_video.get(video_id, [])), "inputs": input_results})
    summary = {
        "direction": direction,
        "overall_score": sum(per_video_scores.values()) / len(per_video_scores) if per_video_scores else 0.0,
        "num_videos": len(per_video_scores), "num_gt_inputs": sum(len(item["inputs"]) for item in details),
        "num_pred_records_with_outputs": sum(1 for outputs in outputs_by_question.values() if outputs), "num_pred_videos": len(grouped_inputs),
    }
    return summary, details


def output_path(root: Path, path: Path) -> Path:
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
    scorer = None
    semantic_scores = None
    if args.base_score == "semantic":
        semantic_scores = load_or_generate_semantic_scores(output_path(root, args.semantic_scores), samples, outputs_by_question, args.semantic_model, args.semantic_api_base, args.semantic_api_key, args.semantic_timeout, outputs_by_video, args.direction)
    summary, details = evaluate(samples, outputs_by_question, outputs_by_video, args.mode, args.base_score, scorer, semantic_scores, args.direction)
    semantic_config = None if args.base_score != "semantic" else {"model": args.semantic_model, "api_base": args.semantic_api_base, "timeout": args.semantic_timeout}
    result = {"mode": args.mode, "direction": args.direction, "base_score": args.base_score, "semantic_config": semantic_config, "semantic_scores_path": str(output_path(root, args.semantic_scores)) if args.base_score == "semantic" else None, "annotation_path": str(annotation_path), "pred_path": str(pred_path), **summary}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    for path, payload in ((output_path(root, args.details), details), (output_path(root, args.summary), result)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
