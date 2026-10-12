#!/usr/bin/env python3

"""Read video metadata and build streaming video conversations."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image
from tqdm import tqdm

try:
    from .prompts import get_silence_signal, get_system_prompt, get_task_prompt
except ImportError:
    from prompts import get_silence_signal, get_system_prompt, get_task_prompt

LOGGER = logging.getLogger("vllm_infer")
EPS = 1e-6


def format_timestamp(timestamp: float) -> str:
    """Format a frame timestamp using the model's inline time-marker syntax."""
    seconds = max(0.0, float(timestamp))
    return f"<t = {seconds:.1f}s>"


def sample_time_range(start: float, end: float, fps: float) -> list[float]:
    if end < start:
        return []

    step = 1.0 / float(fps)
    timestamps: list[float] = []
    current = float(start)
    while current <= float(end) + EPS:
        timestamps.append(current)
        current += step
    if not timestamps or abs(timestamps[-1] - float(end)) > EPS:
        timestamps.append(float(end))
    return timestamps


def collect_sample_times(
    example: dict[str, Any],
    base_fps: float,
    boost_fps: float | None,
    boost_field: str,
) -> list[float]:
    """Build base-rate timestamps and optionally densify annotated time spans."""
    duration = max(float(example.get("duration", 0.0)), 0.0)
    timestamps = sample_time_range(0.0, duration, base_fps)

    if boost_fps is not None and boost_fps > 0:
        for answer in example.get("answer", []):
            span = answer.get(boost_field)
            if not isinstance(span, list) or len(span) != 2:
                continue
            start = max(0.0, min(float(span[0]), duration))
            end = max(0.0, min(float(span[1]), duration))
            timestamps.extend(sample_time_range(start, end, boost_fps))

    return sorted({round(float(timestamp), 6) for timestamp in timestamps if 0.0 <= float(timestamp) <= duration + EPS})


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def stage_annotation(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_url(source):
        import urllib.request

        with urllib.request.urlopen(source) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Local input file not found: {source}")
    shutil.copyfile(source_path, destination)


def validate_annotation(annotation_path: Path, video_dir: Path) -> list[dict[str, Any]]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("videos"), list) or not data["videos"]:
        raise ValueError("Annotation JSON must be an object with a non-empty videos array.")
    is_conversation_annotation = all(
        isinstance(example, dict)
        and isinstance(example.get("question_id"), str)
        and example["question_id"].startswith(("OVO_REC_", "Intervene_"))
        for example in data["videos"]
    )
    task_prompt = data.get("task_prompt")
    if not is_conversation_annotation and (
        not isinstance(task_prompt, dict)
        or task_prompt.get("role") != "user"
        or not isinstance(task_prompt.get("content"), str)
        or not task_prompt["content"].strip()
    ):
        raise ValueError("Non-REC annotation must contain a non-empty task_prompt with role=user.")

    video_dir = video_dir.resolve()
    for index, example in enumerate(data["videos"]):
        if not isinstance(example, dict):
            raise ValueError(f"Annotation item {index} must be an object.")
        missing = {"question_id", "video", "duration"} - set(example)
        if missing:
            raise ValueError(f"Annotation item {index} is missing keys: {sorted(missing)}")
        if not isinstance(example["video"], str) or not example["video"] or os.path.isabs(example["video"]):
            raise ValueError(f"Annotation item {index} has an invalid relative video path.")

        if example["question_id"].startswith("OVO_REC_"):
            conversation = example.get("conversation")
            if not isinstance(conversation, list) or not any(
                isinstance(turn, dict)
                and turn.get("role") == "user"
                and isinstance(turn.get("content"), str)
                and turn["content"].strip()
                for turn in conversation
            ):
                raise ValueError(
                    f"Conversation annotation item {index} must contain a non-empty user conversation prompt."
                )
        else:
            try:
                get_task_prompt(
                    example["question_id"],
                    example.get("object"),
                    conversation=example.get("conversation"),
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"Annotation item {index} has invalid task prompt metadata: {error.args[0]}") from error

        target = (video_dir / example["video"]).resolve()
        if os.path.commonpath((str(video_dir), str(target))) != str(video_dir):
            raise ValueError(f"Annotation item {index} escapes the video directory.")
        if not target.is_file():
            raise FileNotFoundError(f"Video for annotation item {index} not found: {target}")
    return data["videos"]


def discover_served_model_name(api_base: str, timeout: float) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=timeout) as response:
        payload = json.load(response)

    models = payload.get("data", [])
    if len(models) != 1 or not isinstance(models[0].get("id"), str) or not models[0]["id"]:
        raise ValueError("Expected exactly one served model. Set --model explicitly.")
    return models[0]["id"]

class VideoReader:
    def __init__(self, video_path: str, backend: str) -> None:
        self.video_path = video_path
        self.backend = backend
        if backend in {"auto", "decord"}:
            try:
                import decord

                self.reader = decord.VideoReader(video_path, num_threads=1)
                self.num_frames = len(self.reader)
                self.fps = float(self.reader.get_avg_fps())
                self.backend = "decord"
            except Exception as decord_error:
                if backend == "decord":
                    raise RuntimeError(f"Failed to read video with decord: {video_path}") from decord_error
                self._init_pyav(video_path)
        elif backend == "pyav":
            self._init_pyav(video_path)
        else:
            self._init_torchvision(video_path)

        if self.num_frames <= 0 or self.fps <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}: frames={self.num_frames}, fps={self.fps}")

    def _init_torchvision(self, video_path: str) -> None:
        try:
            from torchvision.io import read_video
        except ImportError as error:
            raise RuntimeError(
                "torchvision.io.read_video is unavailable in this torchvision build. "
                "Use --decode_backend pyav or install decord."
            ) from error

        video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
        self.reader = video
        self.num_frames = int(video.shape[0])
        self.fps = float(info.get("video_fps", 0.0))
        self.backend = "torchvision"

    def _init_pyav(self, video_path: str) -> None:
        try:
            import av
        except ImportError as error:
            raise RuntimeError(
                "PyAV is required for --decode_backend pyav or auto fallback. Install it with: pip install av"
            ) from error

        container = av.open(video_path)
        stream = container.streams.video[0]
        self.fps = float(stream.average_rate or stream.base_rate or 0.0)
        self.reader = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=stream.index)]
        container.close()
        self.num_frames = len(self.reader)
        self.backend = "pyav"

    def frame_index_at(self, timestamp: float) -> int:
        index = int(math.floor(max(0.0, float(timestamp)) * self.fps))
        return min(max(index, 0), self.num_frames - 1)

    def get_frame(self, frame_index: int) -> Image.Image:
        if self.backend == "decord":
            frame = self.reader[int(frame_index)].asnumpy()
            return Image.fromarray(frame).convert("RGB")
        if self.backend == "pyav":
            return Image.fromarray(self.reader[int(frame_index)]).convert("RGB")
        frame = self.reader[int(frame_index)].permute(1, 2, 0).numpy()
        return Image.fromarray(frame).convert("RGB")


def _is_image_turn(turn: dict[str, Any]) -> bool:
    content = turn.get("content")
    return (
        turn.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and isinstance(content[0], dict)
        and content[0].get("type") == "image"
        and "image" in content[0]
    )


def build_conversation(example: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    video_path = os.path.join(args.video_dir, example["video"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = VideoReader(video_path, args.decode_backend)
    sample_times = collect_sample_times(
        example=example,
        base_fps=args.base_fps,
        boost_fps=args.boost_fps,
        boost_field=args.boost_field,
    )

    sample_by_frame: dict[int, float] = {}
    for timestamp in sample_times:
        sample_by_frame.setdefault(reader.frame_index_at(timestamp), timestamp)

    sampled_frame_refs = sorted(sample_by_frame.items())
    num_unique_frames_before_limit = len(sampled_frame_refs)
    sampled_frames = [
        (frame_index, timestamp, reader.get_frame(frame_index))
        for frame_index, timestamp in sampled_frame_refs
    ]

    system_content = get_system_prompt(args.model)
    input_turns = [{"role": "system", "time": 0.0, "content": system_content}]
    if example["question_id"].startswith("OVO_REC_"):
        input_turns.extend(dict(turn) for turn in example["conversation"])
        task_prompt_source = "annotation.videos[].conversation"
    else:
        task_content = get_task_prompt(
            example["question_id"],
            example.get("object"),
            model_name=args.model,
            conversation=example.get("conversation"),
        )
        input_turns.append({"role": "user", "time": 0.0, "content": task_content})
        input_turns.extend(
            dict(turn)
            for turn in example["conversation"]
            if not (
                turn.get("role") == "user"
                and float(turn.get("time", 0.0)) == 0.0
            )
        )
        task_prompt_source = "infer/prompts.py"

    frame_turns = []
    for start in range(0, len(sampled_frames), args.frames_per_turn):
        chunk = sampled_frames[start : start + args.frames_per_turn]
        content = []
        for _, timestamp, image in chunk:
            content.extend(
                [
                    # {"type": "text", "text": format_timestamp(timestamp)},
                    {"type": "image", "image": image},
                ]
            )
        frame_turns.append(
            {
                "role": "user",
                "time": float(chunk[-1][1]),
                "content": content,
            }
        )

    conversation = [dict(turn) for turn in input_turns if turn.get("role") == "system"]
    user_questions = []
    assistant_turns = []
    for order, turn in enumerate(input_turns):
        if turn.get("role") == "system":
            continue
        turn["time"] = float(turn.get("time", 0.0))
        if turn.get("role") == "user":
            user_questions.append((turn["time"], order, turn))
        elif turn.get("role") == "assistant":
            assistant_turns.append((turn["time"], order, turn))

    for question_time, _, question in user_questions:
        if question_time <= 0.0:
            conversation.append(
                {
                    "role": "user",
                    "time": question_time,
                    "content": question.get("content", ""),
                    "skip_inference": True,
                }
            )

    conversation.extend(frame_turns)
    for question_time, _, question in user_questions:
        if question_time <= 0.0:
            continue
        target_index = next(
            (index for index, turn in enumerate(conversation) if turn.get("time", float("inf")) >= question_time),
            len(conversation),
        )
        if target_index == len(conversation):
            conversation.append(
                {"role": "user", "time": question_time, "content": question.get("content", "")}
            )
        else:
            target = conversation[target_index]
            target["content"].append({"type": "text", "text": str(question.get("content", ""))})

    # for assistant_time, _, assistant in sorted(assistant_turns, key=lambda item: (item[0], item[1])):
    #     insert_at = next(
    #         (index for index, turn in enumerate(conversation) if turn.get("time", float("inf")) > assistant_time),
    #         len(conversation),
    #     )
    #     conversation.insert(insert_at, assistant)

    if args.question_in_first_frame and conversation:
        first_question = next((turn for turn in conversation if turn.get("role") == "user" and not _is_image_turn(turn)), None)
        first_frame = next((turn for turn in conversation if _is_image_turn(turn)), None)
        if first_question is not None and first_frame is not None and first_question.get("time", 0.0) <= first_frame.get("time", 0.0):
            first_frame["content"] = list(first_frame["content"]) + [{"type": "text", "text": str(first_question["content"])}]
            conversation.remove(first_question)

    silence_signal = get_silence_signal(args.model)
    metadata = {
        "video_fps": reader.fps,
        "num_video_frames": reader.num_frames,
        "num_sample_times": len(sample_times),
        "num_unique_frames_before_limit": num_unique_frames_before_limit,
        "num_unique_frames": len(sampled_frames),
        "num_frame_turns": len(frame_turns),
        "frames_per_turn": args.frames_per_turn,
        "max_frames": args.max_frames,
        "frame_history": "full",
        "timestamp_in_frame_message": True,
        "base_fps": args.base_fps,
        "boost_fps": args.boost_fps,
        "boost_field": args.boost_field,
        "decode_backend": reader.backend,
        "question_in_first_frame": args.question_in_first_frame,
        "system_prompt_source": "infer/prompts.py",
        "task_prompt_source": task_prompt_source,
        "prompt_key": example["question_id"],
        "system_prompt": True,
        "silence_signal": silence_signal,
        "filtered_signals": [silence_signal],
    }
    return conversation, metadata

def post_process_conversation(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for turn in conversation:
        if turn.get("skip_evaluation", False):
            continue
        turn = dict(turn)
        if turn.get("role") == "assistant" and "evaluation_content" in turn:
            turn["content"] = turn["evaluation_content"]
        turn.pop("evaluation_content", None)
        turn.pop("skip_evaluation", None)
        content = turn.get("content")
        if isinstance(content, list):
            text_parts = [
                item["text"].strip()
                for item in content
                if isinstance(item, dict) and "text" in item
            ]
            turn["content"] = " ".join(text_parts)
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            result.append(turn)
    return result


def load_existing_ids(output_fname: Path) -> set[str]:
    if not output_fname.exists():
        return set()
    ids = set()
    with output_fname.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["question_id"])
    return ids
