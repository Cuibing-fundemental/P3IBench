#!/usr/bin/env python3

"""Run streaming video inference against a vLLM backend."""

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
    from .client import VllmClient
    from .data import (
        build_conversation,
        discover_served_model_name,
        load_existing_ids,
        post_process_conversation,
        stage_annotation,
        validate_annotation,
    )
except ImportError:
    from client import VllmClient
    from data import (
        build_conversation,
        discover_served_model_name,
        load_existing_ids,
        post_process_conversation,
        stage_annotation,
        validate_annotation,
    )

LOGGER = logging.getLogger("vllm_infer")
EPS = 1e-6

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video inference against an existing vLLM service.")
    parser.add_argument("--anno_source", required=True, help="Local annotation path or HTTP(S) URL.")
    parser.add_argument("--video_dir", required=True, help="Directory containing referenced video files.")
    parser.add_argument("--output_dir", required=True, help="Directory for staged inputs, logs, and predictions.")
    parser.add_argument("--output_fname", default=None, help="Optional explicit prediction jsonl path.")
    parser.add_argument("--api_base",default="http://127.0.0.1:8010/v1", help="OpenAI-compatible vLLM API base URL.")
    parser.add_argument("--api_key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="", help="Served model name. Auto-discovered if omitted.")
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--request_retries", type=int, default=0)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of videos to infer concurrently.",
    )
    parser.add_argument("--base_fps", type=float, default=1.0)
    parser.add_argument("--boost_fps", type=float, default=None)
    parser.add_argument("--boost_field", default="related_timespan")
    parser.add_argument(
        "--frames_per_turn",
        type=int,
        default=1,
        help="Number of newly sampled frames per inference turn; only 1 or 2 is allowed.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="Maximum number of most recent video frames included in each model request.",
    )
    parser.add_argument(
        "--frame_max_pixels",
        type=int,
        default=131072,
        help="Resize each frame proportionally before JPEG encoding so its total area does not exceed this value.",
    )
    parser.add_argument("--question_in_first_frame", action="store_true")
    parser.add_argument(
        "--decode_backend",
        choices=("auto", "decord", "pyav", "torchvision"),
        default="auto",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    if args.base_fps <= 0:
        raise ValueError("--base_fps must be > 0")
    if args.boost_fps is not None and args.boost_fps <= 0:
        raise ValueError("--boost_fps must be > 0 when provided")
    if args.frames_per_turn not in {1, 2}:
        raise ValueError("--frames_per_turn must be 1 or 2")
    if args.max_frames <= 0:
        raise ValueError("--max_frames must be > 0")
    if args.frame_max_pixels is not None and args.frame_max_pixels <= 0:
        raise ValueError("--frame_max_pixels must be > 0 when provided")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")
    if args.request_timeout <= 0:
        raise ValueError("--request_timeout must be > 0")
    if args.request_retries < 0:
        raise ValueError("--request_retries must be >= 0")
    if args.parallelism <= 0:
        raise ValueError("--parallelism must be > 0")
    return args


def infer_example(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run one video with isolated conversation/client state."""
    conversation, metadata = build_conversation(example, args)
    client = VllmClient(args)
    client.input_query_stream(conversation)
    model_output = client.inference()
    return {
        "question_id": example["question_id"],
        "model_response_list": post_process_conversation(model_output["conversation"]),
        "drop_ratio_list": [],
        "sampling_metadata": metadata,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Hide successful HTTP request logs while preserving warnings and errors.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parse_args()

    video_dir = Path(args.video_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_fname = Path(args.output_fname).resolve() if args.output_fname else output_dir / "pred.jsonl"
    anno_local = output_dir / "inputs" / "annotation.json"

    stage_annotation(args.anno_source, anno_local)
    data = validate_annotation(anno_local, video_dir)

    args.video_dir = str(video_dir)
    args.output_dir = str(output_dir)
    args.output_fname = str(output_fname)
    args.model = args.model or discover_served_model_name(args.api_base, args.request_timeout)

    end_idx = len(data) if args.end_idx is None else min(args.end_idx, len(data))
    existing_ids = load_existing_ids(output_fname)
    output_fname.parent.mkdir(parents=True, exist_ok=True)

    with output_fname.open("a", encoding="utf-8") as output:
        pending = [
            example
            for index, example in enumerate(data)
            if args.start_idx <= index < end_idx and example["question_id"] not in existing_ids
        ]
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {executor.submit(infer_example, example, args): example for example in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="vLLM inference"):
                example = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    LOGGER.error("Video %s failed: %s; skipping", example["question_id"], error)
                    continue
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
