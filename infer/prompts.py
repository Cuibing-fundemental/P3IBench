#!/usr/bin/env python3

"""Provide task prompts and model-specific silence signals."""

import json
import re
from typing import Final


BASELINE_SYSTEM_PROMPT: Final[str] = """You are a real-time video assistant watching a video frame by frame.
At each turn, you are given newly observed visual frames from the video stream.

Rules:
1. Follow the instruction provided by the user and decide whether to reply.
2. If you do not need to reply now, output exactly "{silence_signal}".

Output only the reply text."""


SFT_SYSTEM_PROMPT: Final[str] = """You are a real-time video assistant watching a video stream frame by frame.
At each turn, you are given newly observed visual frames from the video stream.

Task:
- Follow the instruction provided by the user and decide whether to reply.

Reply Rules:
- Use </Silence> when the available visual information is insufficient or irrelevant to the instruction.
- Use </Response> when the available visual information is sufficient to answer the instruction.

Additional Constraints:
- Output exactly </Silence>, or begin with </Response> followed by the reply.
- Do not output any text before </Response> or after </Silence>.
"""


NEXAR_TASK_PROMPT: Final[str] = """Your task is to predict whether the vehicle is likely to be involved in an imminent collision. Focus on imminent risks such as obstacles, dangerous maneuvers, pedestrians, vehicles, and unsafe driving conditions.

Rules:
1. When an imminent collision risk is reliably identified, output exactly: Warning.
2. Respond at the earliest moment when the collision risk can be reliably predicted, before the collision occurs.
3. Output Warning only once for the whole video.
4. At all other times, set your reply to {silence_signal}.
"""

TAG_TASK_PROMPT: Final[str] = """Your task is to continuously monitor the specified object and detect each disappearance event.

The object to monitor: {object}

Rules:
1. When the object becomes no longer visible or identifiable, output exactly: Warning.
2. Respond at the earliest moment when the disappearance can be reliably confirmed.
3. Output Warning only once for each disappearance event; a later disappearance after the object becomes visible again should be treated as a new event.
4. At all other times, set your reply to {silence_signal}.
"""

UCF_TASK_PROMPT: Final[str] = """Your task is to monitor the video in real time and detect criminal or suspicious behavior.

Rules:
1. When criminal or suspicious behavior is reliably detected for the first time, output exactly: Warning.
2. Respond at the earliest moment when the behavior can be reliably confirmed.
3. Output Warning only once for the whole video.
4. At all other times, set your reply to {silence_signal}.
"""
EGO4D_ATOMIC_TASK_PROMPT: Final[str] = """Your task is to continuously monitor the specified atomic event and detect the target occurrence.

The atomic event to monitor: {query}
Target occurrence: {target}

Rules:
1. Count only events that fully match the atomic event and are visually supported.
2. Treat one temporally continuous action as one occurrence; count separate occurrences in chronological order.
3. When the target occurrence is reliably identified, output exactly: {target}.
4. Respond at the earliest moment when the target occurrence becomes clear; do not wait for it to finish.
5. Output the target occurrence only once.
6. At all other times, set your reply to {silence_signal}.
"""
EGO4D_EVENT_TYPE_TASK_PROMPT: Final[str] = """Your task is to continuously monitor the specified semantic event category and detect the target occurrence.

The event type to monitor: {query}
Target occurrence: {target}

Rules:
1. Count every visually supported atomic event belonging to the specified semantic category.
2. Do not count superficially similar events outside the category.
3. Treat one temporally continuous event as one occurrence; count separate occurrences in chronological order.
4. When the target occurrence is reliably identified, output exactly: {target}.
5. Respond at the earliest moment when the target occurrence becomes clear; do not wait for it to finish.
6. Output the target occurrence only once.
7. At all other times, set your reply to {silence_signal}.
"""
EGO4D_COMPOSITIONAL_TASK_PROMPT: Final[str] = """Your task is to continuously monitor the specified complete compositional process and detect its boundaries.

The process to monitor: {query}

Rules:
1. Monitor the complete process described by the query and its related atomic actions.
2. Output exactly: start when the process begins or its start is reliably clear.
3. Output exactly: end when the process is completed or its completion is reliably clear.
4. Ignore preparation outside the process, intermediate pauses, adjustments, and internal actions.
5. Output start and end at most once each, with start before end.
6. At all other times, set your reply to {silence_signal}.
"""
INTERVENE_TASK_PROMPT: Final[str] = """Your task is to proactively assist the user while continuously monitoring the video.

The user's task and intervention requirements are:
{instruction}

Rules:
1. Use the user's task, the visual evidence, and the conversation history together.
2. Respond proactively when an intervention requested by the user is visually supported; do not wait for another user question.
3. Provide a concise, useful natural-language reminder, correction, or guidance message.
4. Do not repeat an intervention after the corresponding task state has already been handled, unless new visual evidence makes it necessary.
5. At all other times, set your reply to {silence_signal}.
"""
TASK_PROMPTS: Final[tuple[tuple[str, str], ...]] = (
    ("Nexar_", NEXAR_TASK_PROMPT),
    ("TAG_", TAG_TASK_PROMPT),
    ("UCF_", UCF_TASK_PROMPT),
    ("Atomic_", EGO4D_ATOMIC_TASK_PROMPT),
    ("EventType_", EGO4D_EVENT_TYPE_TASK_PROMPT),
    ("Compositional_", EGO4D_COMPOSITIONAL_TASK_PROMPT),
    ("Intervene_", INTERVENE_TASK_PROMPT),
)


MODEL_SILENCE_SIGNALS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("sft",), "</Silence>"),
    (("joyvl", "joyai-vl", "joyai_vl"), "</silence>"),
    (("mmduet2",), "NO REPLY"),
)
DEFAULT_SILENCE_SIGNAL: Final[str] = "Silence"

JOY_MODEL_KEYWORDS: Final[tuple[str, ...]] = ("joyvl", "joyai-vl", "joyai_vl")
JOY_RESPONSE_MARKERS: Final[tuple[str, ...]] = ("</response>",)
JOY_HIDDEN_MARKERS: Final[tuple[str, ...]] = ("</delegation>", "<delegation>", "</delegate>")
SFT_MODEL_KEYWORDS: Final[tuple[str, ...]] = ("sft",)
SFT_RESPONSE_MARKER: Final[str] = "</Response>"


def get_silence_signal(model_name: str) -> str:
    """Return the silence signal used by the specified model."""
    normalized_name = (model_name or "").casefold()
    for keywords, signal in MODEL_SILENCE_SIGNALS:
        if any(keyword in normalized_name for keyword in keywords):
            return signal
    return DEFAULT_SILENCE_SIGNAL


def get_system_prompt(model_name: str) -> str:
    """Replace the silence placeholder with the model's native signal."""
    if _is_sft_model(model_name):
        return SFT_SYSTEM_PROMPT
    return BASELINE_SYSTEM_PROMPT.format(silence_signal=get_silence_signal(model_name))


def _is_sft_model(model_name: str) -> bool:
    normalized_name = (model_name or "").casefold()
    return any(keyword in normalized_name for keyword in SFT_MODEL_KEYWORDS)


def get_task_prompt(question_id: str, object_description=None, model_name: str = "", conversation=None) -> str:
    """Render a registered task prompt, extracting EGO4D fields from its conversation."""
    template = next((prompt for prefix, prompt in TASK_PROMPTS if question_id.startswith(prefix)), None)
    if template is None:
        raise KeyError(f"Task prompt is not registered: {question_id}")
    if question_id.startswith("TAG_"):
        if not isinstance(object_description, str) or not object_description.strip():
            raise KeyError(f"TAG sample {question_id} must contain a non-empty object field")
        return template.format(
            object=object_description.strip(),
            silence_signal=get_silence_signal(model_name),
        )
    if question_id.startswith(("Nexar_", "UCF_")):
        return template.format(silence_signal=get_silence_signal(model_name))
    if question_id.startswith("Intervene_"):
        if not isinstance(conversation, list):
            raise ValueError(f"Intervene sample {question_id} must contain a conversation")
        instruction = next(
            (
                turn.get("content", "").strip()
                for turn in conversation
                if isinstance(turn, dict)
                and turn.get("role") == "user"
                and isinstance(turn.get("content"), str)
                and turn.get("content", "").strip()
            ),
            None,
        )
        if instruction is None:
            raise ValueError(
                f"Intervene sample {question_id} conversation prompt must contain a non-empty user instruction"
            )
        return template.format(
            instruction=instruction,
            silence_signal=get_silence_signal(model_name),
        )
    if question_id.startswith(("Atomic_", "EventType_", "Compositional_")):
        if not isinstance(conversation, list):
            raise ValueError(f"EGO4D sample {question_id} must contain a conversation")
        text = "\n".join(
            turn.get("content", "")
            for turn in conversation
            if isinstance(turn, dict) and isinstance(turn.get("content"), str)
        )
        labels = {
            "Atomic_": "Atomic event",
            "EventType_": "Event type",
            "Compositional_": "Process",
        }
        label = next(prefix for prefix in labels if question_id.startswith(prefix))
        match = re.search(rf"(?m)^{re.escape(labels[label])}:\s*(.+?)\s*$", text)
        if not match:
            raise ValueError(f"EGO4D sample {question_id} is missing '{labels[label]}:'")
        query = match.group(1).strip()
        if label == "Compositional_":
            return template.format(query=query, silence_signal=get_silence_signal(model_name))
        target_match = re.search(r"(?m)^Target occurrence:\s*(\S.*?)\s*$", text)
        if not target_match:
            raise ValueError(f"EGO4D sample {question_id} is missing 'Target occurrence:'")
        return template.format(
            query=query,
            target=target_match.group(1).strip(),
            silence_signal=get_silence_signal(model_name),
        )
    return template


def _strip_json_markdown_fence(text: str) -> str:
    """Remove Markdown code fences surrounding JSON."""
    match = re.fullmatch(r"```\s*(?:json)?\s*\n?(.*?)\n?\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def normalize_model_reply(reply: str, model_name: str) -> str | None:
    text = (reply or "").replace("<|im_end|>", "").strip()
    if not text:
        return None
    text = _strip_json_markdown_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        text = payload["content"].strip()
    if not text:
        return None
    silence_signal = get_silence_signal(model_name)
    normalized_name = (model_name or "").casefold()
    folded_text = text.casefold()
    if _is_sft_model(model_name):
        if folded_text == silence_signal.casefold():
            return None
        marker = SFT_RESPONSE_MARKER.casefold()
        position = folded_text.find(marker)
        if position >= 0:
            text = text[position + len(SFT_RESPONSE_MARKER):].strip()
        return text or None
    if any(keyword in normalized_name for keyword in JOY_MODEL_KEYWORDS):
        actions = []
        if silence_signal.casefold() in folded_text:
            actions.append((folded_text.find(silence_signal.casefold()), "silent", silence_signal))
        actions.extend(
            (folded_text.find(marker.casefold()), "response", marker)
            for marker in JOY_RESPONSE_MARKERS
            if marker.casefold() in folded_text
        )
        if actions:
            position, action, marker = min(actions, key=lambda item: item[0])
            if action == "silent":
                return None
            text = text[position + len(marker):].strip()
        hidden = [
            text.casefold().find(marker.casefold())
            for marker in JOY_HIDDEN_MARKERS
            if marker.casefold() in text.casefold()
        ]
        if hidden:
            if not actions:
                return None
            text = text[:min(hidden)].strip()
        return text or None
    comparable = text.rstrip(".").strip().strip('"').strip().casefold()
    return None if comparable == silence_signal.casefold() else text
