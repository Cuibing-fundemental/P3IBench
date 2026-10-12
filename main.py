#!/usr/bin/env python3

"""Run inference and four evaluations on selected datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


DATASETS = (
    "Prediction_Accident",
    "Perception_Object",
    "Prediction_Crime",
    "Perception_Event_atomic",
    "Perception_Event_compositional",
    "Perception_Event_eventtype",
    "Intervention",
)

EVALUATIONS = (
    ("default", "default", "single", "Default"),
    ("first_only", "first_only", "single", "First-only"),
    ("bidirectional_default", "default", "bidirectional", "Bidirectional Default"),
    ("bidirectional_first_only", "first_only", "bidirectional", "Bidirectional First-only"),
)

STRICT_LABEL_DATASETS = {
    "Prediction_Accident", "Perception_Object", "Prediction_Crime",
    "Perception_Event_atomic", "Perception_Event_eventtype",
}
COMPOSITIONAL_DATASET = "Perception_Event_compositional"


def evaluation_script(project_root: Path, dataset: str) -> Path:
    """Return the evaluator dedicated to the dataset's scoring protocol."""
    evaluator = "compositional.py" if dataset == COMPOSITIONAL_DATASET else "unified.py"
    return project_root / "eval" / evaluator


def evaluation_base_score(dataset: str) -> str:
    """Return the base-score mode used by the selected evaluator."""
    return "strict" if dataset in STRICT_LABEL_DATASETS else "constant"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run inference and four unified evaluations for the selected datasets.")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS), metavar="DATASET",
        help=f"Datasets to run (default: all). Choices: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("VLLM_API_BASE"),
        help="Full OpenAI-compatible API base URL; overrides --port.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SERVED_MODEL_NAME"),
        help="Served model name; auto-discover from /models when omitted.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VLLM_PORT", "8010")),
        help="Local vLLM port (default: 8010).",
    )
    configured_data_root = os.environ.get("DATA_ROOT")
    parser.add_argument("--data-root", type=Path, default=Path(configured_data_root) if configured_data_root else None)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "output")))
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument(
        "--request-retries",
        type=int,
        default=int(os.environ.get("REQUEST_RETRIES", "3")),
        help="Retries per failed inference request (default: 3).",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=10,
        help="Videos inferred in parallel per dataset (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Most recent frames per inference request (default: 16).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video sampling frames per second (default: 1.0).",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def resolve_root(project_root: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured if configured.is_absolute() else project_root / configured
    return project_root / "data"


def find_annotation(dataset_dir: Path) -> Path:
    preferred = dataset_dir / f"{dataset_dir.name}.json"
    candidates = [preferred] if preferred.is_file() else sorted(dataset_dir.glob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("videos"), list):
            return candidate
    raise FileNotFoundError(f"No valid annotation found for {dataset_dir.name}")


def find_video_dir(dataset_dir: Path) -> Path:
    for name in ("videos", "video"):
        path = dataset_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No video directory found for {dataset_dir.name}")


def stop_process_group(process: subprocess.Popen, timeout: float = 5.0) -> None:
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    finally:
        process.wait()
        signal.signal(signal.SIGINT, previous_handler)


def run_command(command: list[str], log_path: Path, display_prefix: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"{display_prefix} {line}", end="", flush=True)
            result = process.wait()
        except BaseException:
            stop_process_group(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
    if result:
        raise RuntimeError(f"Command failed with exit code {result}: {log_path}")


def discover_model(api_base: str, timeout: float = 600.0, api_key: str | None = None) -> str:
    """Discover the served model, retrying while the backend starts."""
    url = f"{api_base.rstrip('/')}/models"
    last_error: Exception | None = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(10):
        try:
            request = urllib.request.Request(url)
            if api_key and api_key != "EMPTY":
                request.add_header("Authorization", f"Bearer {api_key}")
            with opener.open(request, timeout=10) as response:
                payload = json.load(response)
            models = payload.get("data", [])
            if len(models) == 1 and isinstance(models[0].get("id"), str) and models[0]["id"]:
                return models[0]["id"]
            raise RuntimeError("vLLM must expose exactly one model")
        except (OSError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Unable to discover model from {url}: {last_error}")


def build_infer_command(
    args: argparse.Namespace,
    infer_script: Path,
    annotation: Path,
    video_dir: Path,
    output_dir: Path,
    prediction_path: Path,
    model_name: str,
) -> list[str]:
    command = [
        args.python_bin, "-u", str(infer_script),
        "--anno_source", str(annotation), "--video_dir", str(video_dir),
        "--output_dir", str(output_dir), "--output_fname", str(prediction_path),
        "--api_base", args.api_base, "--model", model_name, "--base_fps", str(args.fps), "--boost_field", "related_timespan",
        "--frames_per_turn", "1", "--max_frames", str(args.max_frames), "--frame_max_pixels", "131072",
        "--parallelism", str(args.parallelism), "--request_timeout", "600", "--request_retries", str(args.request_retries),
        "--max_new_tokens", "256",
        "--decode_backend", "auto", "--start_idx", "0",
    ]
    api_key = os.environ.get("VLLM_API_KEY", "")
    if api_key:
        command.extend(["--api_key", api_key])
    return command

def main() -> int:
    args = parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be finite and > 0")
    if args.request_retries < 0:
        raise ValueError("--request-retries must be >= 0")
    if args.parallelism <= 0:
        raise ValueError("--parallelism must be > 0")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535")
    args.api_base = args.api_base or f"http://127.0.0.1:{args.port}/v1"
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    project_root = args.project_root.resolve()
    data_root = resolve_root(project_root, args.data_root).resolve()
    output_root = (args.output_root if args.output_root.is_absolute() else project_root / args.output_root).resolve()

    run_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    report_dir = run_root / "Docs"
    report_dir.mkdir()
    results: list[dict] = []
    model_name = args.model or discover_model(args.api_base, api_key=os.environ.get("VLLM_API_KEY"))

    infer_script = project_root / "infer" / "infer.py"
    datasets = tuple(dict.fromkeys(args.datasets))
    total = len(datasets)
    for index, name in enumerate(datasets, start=1):
        dataset_output = run_root / name
        log_path = report_dir / f"{name}.log"
        record = {
            "dataset": name,
            "model": model_name,
            "status": "failed",
            "output_dir": str(dataset_output),
        }
        print(f"\n[{index}/{total}] Starting dataset: {name}", flush=True)
        try:
            dataset_dir = data_root / name
            annotation = find_annotation(dataset_dir)
            video_dir = find_video_dir(dataset_dir)
            pred_path = dataset_output / "pred.jsonl"
            run_command(
                build_infer_command(
                    args, infer_script, annotation, video_dir, dataset_output, pred_path, model_name,
                ),
                log_path,
                f"[{index}/{total}][{name}][inference]",
            )
            score_dir = dataset_output / "score"
            for stem, mode, direction, label in EVALUATIONS:
                details = score_dir / f"{stem}.json"
                summary = score_dir / f"{stem}_summary.json"
                eval_script = evaluation_script(project_root, name)
                base_score = evaluation_base_score(name)
                run_command([
                    args.python_bin, str(eval_script), "--root", str(project_root),
                    "--input-dir", str(dataset_output), "--mode", mode,
                    "--direction", direction, "--base-score", base_score,
                    "--details", str(details), "--summary", str(summary),
                ], report_dir / f"{name}_{stem}_score.log", f"[{index}/{total}][{name}][{label} evaluation]")
                result = json.loads(summary.read_text(encoding="utf-8"))
                record[f"{stem}_score"] = result.get("overall_score")
                record[f"{stem}_summary"] = str(summary)
            record["status"] = "completed"
            print(
                f"[{index}/{total}] {name} completed: "
                + ", ".join(f"{label}: {record[f'{stem}_score']}" for stem, _, _, label in EVALUATIONS),
                flush=True,
            )
        except Exception as exc:
            record["error"] = str(exc)
            results.append(record)
            print(f"[{index}/{total}] {name} failed: {exc}; continuing", file=sys.stderr, flush=True)
            continue
        results.append(record)

    aggregate = {
        "run_id": run_id,
        "model": model_name,
        "api_base": args.api_base,
        "data_root": str(data_root),
        "results": results,
        "completed": len(results) == total and all(item["status"] == "completed" for item in results),
    }
    (report_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Dataset results ({run_id})", "", f"- Model: `{model_name}`",
        f"- vLLM API: `{args.api_base}`", f"- Data root: `{data_root}`", "",
        "| Dataset | Status | Default | First-only | Bidirectional Default | Bidirectional First-only |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        scores = [
            "" if item.get(f"{stem}_score") is None else f"{float(item[f'{stem}_score']):.6f}"
            for stem, _, _, _ in EVALUATIONS
        ]
        lines.append(f"| {item['dataset']} | {item['status']} | " + " | ".join(scores) + " |")
    lines.extend(["", "See `aggregate_summary.json` for complete results; individual logs and scores are saved in the run directory."])
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["completed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; child processes have been stopped.", file=sys.stderr, flush=True)
        raise SystemExit(130)
