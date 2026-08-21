from __future__ import annotations

import asyncio
import json
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ConnectionClosed = Exception


DEFAULT_ASR_URL = "ws://127.0.0.1:8080/v1/speech"
DEFAULT_ASR_MODEL = "zipformer_online"
DEFAULT_ASR_TASK = "asr"
DEFAULT_ASR_CHUNK_MS = 20


@dataclass
class AsrRunConfig:
    model: str = DEFAULT_ASR_MODEL
    task: str = DEFAULT_ASR_TASK
    language_hints: str = "zh"
    disfluency_removal_enabled: bool | None = None
    semantic_punctuation_enabled: bool | None = None
    punctuation_prediction_enabled: bool | None = None
    heartbeat: bool | None = None
    max_sentence_silence: int | None = None
    multi_threshold_mode_enabled: bool | None = None
    inverse_text_normalization_enabled: bool | None = None


class AsrTextTracker:
    """Tracks streaming ASR partial/final text and text already consumed by SoulX."""

    def __init__(self) -> None:
        self.final_sentences: list[str] = []
        self.last_partial = ""
        self.last_sentence_end = False
        self.response_count = 0
        self.consumed_norm_prefix = ""
        self._lock = asyncio.Lock()

    async def update(self, text: str, sentence_end: bool) -> str:
        async with self._lock:
            self.response_count += 1
            self.last_sentence_end = sentence_end
            if sentence_end:
                self.final_sentences.append(text)
                self.last_partial = ""
            else:
                self.last_partial = text
            return self._text_unlocked()

    async def snapshot(self) -> tuple[str, bool, int]:
        async with self._lock:
            return self._text_unlocked(), self.last_sentence_end, self.response_count

    async def pending_snapshot(self) -> tuple[str, bool, int]:
        async with self._lock:
            return (
                self._pending_text_unlocked(),
                self.last_sentence_end,
                self.response_count,
            )

    async def consume(self, text: str) -> None:
        consume_norm = normalize_asr_prefix(text)
        if not consume_norm:
            return

        async with self._lock:
            pending_text = self._pending_text_unlocked()
            pending_norm = normalize_asr_prefix(pending_text)
            if pending_norm.startswith(consume_norm):
                self.consumed_norm_prefix += consume_norm
                return

            if consume_norm.startswith(pending_norm) and pending_norm:
                self.consumed_norm_prefix += pending_norm
                return

            full_norm = normalize_asr_prefix(self._text_unlocked())
            candidate = self.consumed_norm_prefix + consume_norm
            if full_norm.startswith(candidate):
                self.consumed_norm_prefix = candidate
                return

            consume_at = pending_norm.find(consume_norm)
            if consume_at >= 0:
                self.consumed_norm_prefix += pending_norm[
                    : consume_at + len(consume_norm)
                ]

    def _text_unlocked(self) -> str:
        if self.last_partial:
            return "".join(self.final_sentences + [self.last_partial])
        return "".join(self.final_sentences)

    def _pending_text_unlocked(self) -> str:
        text = self._text_unlocked()
        full_norm = normalize_asr_prefix(text)
        if self.consumed_norm_prefix.startswith(full_norm):
            return ""

        consumed_index = raw_index_after_normalized_prefix(
            text,
            self.consumed_norm_prefix,
        )
        if consumed_index is None:
            return text
        return lstrip_asr_separators(text[consumed_index:])


def normalize_asr_prefix(text: str) -> str:
    pieces = []
    for char in unicodedata.normalize("NFKC", text):
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category[0] in {"P", "Z"}:
            continue
        pieces.append(char.lower())
    return "".join(pieces)


def lstrip_asr_separators(text: str) -> str:
    for index, char in enumerate(text):
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category[0] in {"P", "Z"}:
            continue
        return text[index:]
    return ""


def raw_index_after_normalized_prefix(text: str, norm_prefix: str) -> int | None:
    if not norm_prefix:
        return 0

    normalized = ""
    for index, char in enumerate(text):
        piece = normalize_asr_prefix(char)
        if not piece:
            continue
        normalized += piece
        if not norm_prefix.startswith(normalized):
            return None
        if len(normalized) >= len(norm_prefix):
            return index + 1
    return None


def build_asr_run_task(
    config: AsrRunConfig,
    task_id: str,
    sample_rate: int,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "format": "pcm",
        "sample_rate": sample_rate,
    }
    if config.language_hints:
        parameters["language_hints"] = [
            item.strip() for item in config.language_hints.split(",") if item.strip()
        ]

    optional_params = {
        "disfluency_removal_enabled": config.disfluency_removal_enabled,
        "semantic_punctuation_enabled": config.semantic_punctuation_enabled,
        "punctuation_prediction_enabled": config.punctuation_prediction_enabled,
        "heartbeat": config.heartbeat,
        "max_sentence_silence": config.max_sentence_silence,
        "multi_threshold_mode_enabled": config.multi_threshold_mode_enabled,
        "inverse_text_normalization_enabled": config.inverse_text_normalization_enabled,
    }
    for key, value in optional_params.items():
        if value is not None:
            parameters[key] = value

    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": config.task,
            "function": "recognition",
            "model": config.model,
            "parameters": parameters,
            "input": {},
        },
    }


def build_asr_finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def parse_gateway_result(message: str | bytes) -> tuple[str, bool, str]:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return "", False, ""

    try:
        response = json.loads(message)
    except json.JSONDecodeError:
        return "", False, ""

    if not isinstance(response, dict):
        return "", False, ""
    header = response.get("header")
    event = header.get("event") if isinstance(header, dict) else ""
    if event != "result-generated":
        return "", False, event

    payload = response.get("payload")
    output = payload.get("output") if isinstance(payload, dict) else None
    sentence = output.get("sentence") if isinstance(output, dict) else None
    if not isinstance(sentence, dict) or sentence.get("heartbeat") is True:
        return "", False, event

    text = sentence.get("text")
    if not isinstance(text, str):
        text = ""
    return text, sentence.get("sentence_end") is True, event


async def asr_receive_loop(
    ws: Any,
    tracker: AsrTextTracker,
    started: asyncio.Event,
    finished: asyncio.Event,
    failed: asyncio.Event,
    verbose: bool,
) -> None:
    try:
        async for message in ws:
            text, sentence_end, event = parse_gateway_result(message)
            if event == "task-started":
                started.set()
            elif event == "task-finished":
                finished.set()
                break
            elif event == "task-failed":
                failed.set()
                finished.set()
                break
            elif event == "result-generated" and text:
                merged_text = await tracker.update(text, sentence_end)
                if verbose:
                    prefix = time.strftime("%H:%M:%S")
                    print(
                        f"[{prefix}] asr text={merged_text!r} "
                        f"sentence_end={sentence_end}"
                    )
    except ConnectionClosed:
        if not finished.is_set():
            failed.set()
    except Exception as exc:
        failed.set()
        print(f"[asr recv error] {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        finished.set()


async def wait_for_started(
    started: asyncio.Event,
    failed: asyncio.Event,
    timeout: float,
) -> None:
    started_task = asyncio.create_task(started.wait())
    failed_task = asyncio.create_task(failed.wait())
    try:
        done, pending = await asyncio.wait(
            {started_task, failed_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if not done:
            raise TimeoutError(f"timed out waiting ASR task-started after {timeout:g}s")
        if failed.is_set():
            raise RuntimeError("ASR task failed before task-started")
    finally:
        for task in (started_task, failed_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(started_task, failed_task, return_exceptions=True)
