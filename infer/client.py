#!/usr/bin/env python3

"""Send streaming video messages to vLLM and filter silence replies."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
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
    from .prompts import get_silence_signal, normalize_model_reply
except ImportError:
    from prompts import get_silence_signal, normalize_model_reply

LOGGER = logging.getLogger("vllm_infer")
EPS = 1e-6

class VllmClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The OpenAI Python package is required for the vLLM client. Install it in the active environment."
            ) from error

        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_base,
            timeout=args.request_timeout,
            max_retries=self.args.request_retries,
        )
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.query_queue: list[dict[str, Any]] = []
        self.video_time = 0.0
        self.image_url_cache: dict[int, str] = {}

    def input_query_stream(self, conversation: list[dict[str, Any]]) -> None:
        for turn in conversation:
            if turn.get("role") in {"user", "assistant", "system"}:
                self.query_queue.append(turn)

    def _image_to_data_url(self, image: Image.Image) -> str:
        cache_key = id(image)
        cached = self.image_url_cache.get(cache_key)
        if cached is not None:
            return cached

        encoded_image = image.convert("RGB")
        width, height = encoded_image.size
        if self.args.frame_max_pixels is not None and width * height > self.args.frame_max_pixels:
            scale = math.sqrt(self.args.frame_max_pixels / float(width * height))
            resized_width = max(1, math.floor(width * scale))
            resized_height = max(1, math.floor(height * scale))
            encoded_image = encoded_image.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        encoded_image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{encoded}"
        self.image_url_cache[cache_key] = data_url
        return data_url

    def _to_api_messages(self) -> list[dict[str, Any]]:
        image_positions = [
            (turn_index, item_index)
            for turn_index, turn in enumerate(self.history)
            if isinstance(turn.get("content", ""), list)
            for item_index, item in enumerate(turn["content"])
            if "text" not in item and (item.get("type") == "image" or "image" in item)
        ]
        retained_images = set(image_positions[-self.args.max_frames:])
        messages: list[dict[str, Any]] = []

        for turn_index, turn in enumerate(self.history):
            content = turn.get("content", "")
            if not isinstance(content, list):
                messages.append({"role": turn["role"], "content": str(content)})
                continue

            api_content = []
            for item_index, item in enumerate(content):
                if "text" in item:
                    api_content.append({"type": "text", "text": str(item["text"])})
                elif item.get("type") == "image" or "image" in item:
                    if (turn_index, item_index) not in retained_images:
                        continue
                    image = item.get("image")
                    if isinstance(image, Image.Image):
                        image_url = self._image_to_data_url(image)
                    elif isinstance(image, str):
                        image_url = image
                    else:
                        raise TypeError(f"Unsupported image input type: {type(image)!r}")
                    api_content.append({"type": "image_url", "image_url": {"url": image_url}})
            if api_content:
                messages.append({"role": turn["role"], "content": api_content})
        return messages

    def _generate(self) -> str:
        extra_body: dict[str, Any] = {}
        if "qwen3" in self.args.model.casefold():
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.args.enable_thinking}
        if self.args.do_sample and self.args.top_k > 0:
            extra_body["top_k"] = self.args.top_k

        try:
            request_kwargs = {
                "model": self.args.model,
                "messages": self._to_api_messages(),
                "max_completion_tokens": self.args.max_new_tokens,
                "extra_body": extra_body,
            }
            # Some hosted models reject any temperature other than their default (1).
            # Omit the field for greedy mode so the provider applies its supported default.
            if self.args.do_sample:
                request_kwargs["temperature"] = self.args.temperature
            else:
                request_kwargs["temperature"] = 0.0
            response = self.client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise RuntimeError("LLM returned no completion choices")
            content = response.choices[0].message.content
            if not content:
                return get_silence_signal(self.args.model)
            return content.strip()
        except Exception as e:
            LOGGER.error("Error during LLM inference: %s", str(e))
            return get_silence_signal(self.args.model)

    def inference(self) -> dict[str, Any]:
        while self.query_queue:
            new_turns = []
            while self.query_queue:
                turn = self.query_queue.pop(0)
                self.history.append(turn)
                new_turns.append(turn)
                if turn.get("role") == "user" and not turn.get("skip_inference", False):
                    break

            last_turn = new_turns[-1]
            if "time" in last_turn:
                self.video_time = float(last_turn["time"])
            self.history[-1]["time"] = self.video_time

            if last_turn.get("skip_inference", False) or last_turn.get("role") in {"system", "assistant"}:
                continue

            raw_reply = self._generate()
            evaluation_reply = normalize_model_reply(raw_reply, self.args.model)
            assistant_turn = {
                "role": "assistant",
                "content": raw_reply,
                "time": self.video_time,
            }
            if evaluation_reply is None:
                assistant_turn["skip_evaluation"] = True
            else:
                assistant_turn["evaluation_content"] = evaluation_reply
            self.history.append(assistant_turn)
        return {"conversation": self.history, "drop_ratio": []}
