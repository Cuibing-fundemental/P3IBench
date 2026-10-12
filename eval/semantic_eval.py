#!/usr/bin/env python3

"""Score a prediction against a reference with a local OpenAI-compatible judge."""

import argparse
from typing import Final


SYSTEM_PROMPT: Final[str] = """You are a strict semantic evaluator for a proactive video benchmark. The reference may contain one or more indispensable factual key points. Judge how completely and correctly the prediction covers those key points.

Return exactly one score:

- 1: All indispensable key points are correct, with no material contradiction.

- 0.5: At least one indispensable key point is correct, but the answer is incomplete or another key point is incorrect.

- 0: No indispensable key point is correct, the answer is irrelevant, or the core meaning contradicts the reference.

Rules:

- A concise semantic paraphrase counts as correct.

- A different object, action, state, location, direction, floor, brand, or list item does not match.

- Merely discussing the same task does not match. Do not infer missing information.

- Ignore extra information only when it does not contradict a reference key point.

- Return with only one number.

Examples:

- Reference: "The phone and charger are packed." Prediction: "The phone is packed." -> 0.5

- Reference: "The keys are packed." Prediction: "The keys are in the bag." -> 1

- Reference: "The chips section is ahead." Prediction: "Tomatoes are ahead." -> 0"""

DEFAULT_MODEL: Final[str] = "Qwen3-32B-FP8"
DEFAULT_API_BASE: Final[str] = "http://127.0.0.1:8010/v1"
VALID_SCORES: Final[dict[str, float]] = {"0": 0.0, "0.5": 0.5, "1": 1.0}


def _parse_score(content: str | None) -> float:
    if content is None:
        raise ValueError("LLM returned an empty completion")
    score = VALID_SCORES.get(content.strip())
    if score is None:
        raise ValueError(f"LLM returned an invalid semantic score: {content!r}")
    return score


def _create_client(api_key: str, api_base: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI Python package is required for semantic evaluation. Install it in the active environment."
        ) from error
    return OpenAI(api_key=api_key, base_url=api_base, timeout=timeout, max_retries=0)


def evaluate_semantic_score(
    reference: str,
    prediction: str,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = "EMPTY",
    timeout: float = 600.0,
) -> float:
    """Return the strict semantic score for one reference and prediction pair."""
    client = _create_client(api_key, api_base, timeout)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Reference: {reference}\nPrediction: {prediction}"},
    ]
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.casefold() else None
    last_error: Exception | None = None
    for _ in range(2):
        try:
            request = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 4,
            }
            if extra_body is not None:
                request["extra_body"] = extra_body
            response = client.chat.completions.create(**request)
            if not response.choices:
                raise ValueError("LLM returned no completion choices")
            return _parse_score(response.choices[0].message.content)
        except Exception as error:
            last_error = error
    raise RuntimeError("Semantic evaluation failed after 2 attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one reference/prediction pair semantically.")
    parser.add_argument("--reference", required=True, help="Ground-truth reference text.")
    parser.add_argument("--prediction", required=True, help="Model prediction text.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible served model name.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the local service.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    return args


def main() -> None:
    args = parse_args()
    score = evaluate_semantic_score(
        reference=args.reference,
        prediction=args.prediction,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"{score:g}")


if __name__ == "__main__":
    main()
