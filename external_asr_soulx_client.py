#!/usr/bin/env python3
"""
Single-stream bridge client:
1. stream an audio file to a local ASR Gateway WebSocket;
2. collect streaming ASR partial/final text;
3. stream the same audio to SoulX-Duplug with the latest cumulative ASR text.

Dependencies are already present in this repo requirements:
  websockets, soundfile, soxr, numpy
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import shutil
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import soxr

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ConnectionClosed = Exception


DEFAULT_ASR_URL = "ws://127.0.0.1:8080/v1/speech"
DEFAULT_SOULX_URL = "ws://127.0.0.1:8000/turn"
DEFAULT_MODEL = "zipformer_online"
DEFAULT_TASK = "asr"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 20
DEFAULT_SOULX_CHUNK_SIZE = 2560


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int


class AsrTextTracker:
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


def load_audio(path: str, target_sample_rate: int) -> AudioData:
    try:
        audio, sample_rate = sf.read(path, always_2d=False)
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"failed to read audio with soundfile and ffmpeg is unavailable: {exc}"
            ) from exc

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "-f",
            "f32le",
            "-",
        ]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        sample_rate = target_sample_rate

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if sample_rate != target_sample_rate:
        audio = soxr.resample(audio, sample_rate, target_sample_rate).astype(np.float32)
        sample_rate = target_sample_rate

    return AudioData(samples=np.clip(audio, -1.0, 1.0), sample_rate=sample_rate)


def add_tail_silence(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    tail_samples = max(0, int(seconds * sample_rate))
    if tail_samples == 0:
        return audio
    return np.concatenate([audio, np.zeros(tail_samples, dtype=np.float32)])


def float_chunk_to_s16_pcm(chunk: np.ndarray) -> bytes:
    clipped = np.clip(chunk, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def float_chunk_to_b64(chunk: np.ndarray) -> str:
    return base64.b64encode(np.asarray(chunk, dtype=np.float32).tobytes()).decode()


def iter_fixed_chunks(audio: np.ndarray, chunk_size: int):
    total_chunks = math.ceil(len(audio) / chunk_size)
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        yield chunk.astype(np.float32, copy=False)


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


def parse_soulx_state(message: str | bytes) -> dict[str, Any] | None:
    if isinstance(message, bytes):
        message = message.decode("utf-8", "replace")
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def format_soulx_state(data: dict[str, Any], show_raw_state: bool, show_json: bool) -> str:
    if show_json:
        return json.dumps(data, ensure_ascii=False)

    state = data.get("state", {})
    if not isinstance(state, dict):
        return json.dumps(data, ensure_ascii=False)

    state_name = state.get("state", "") or "unknown"
    raw_state = state.get("raw_state", "")
    pieces = [state_name]

    if show_raw_state and raw_state:
        pieces.append(f"raw={raw_state}")

    if state_name in {"nonidle", "backchannel", "incomplete"}:
        pieces.append(f"segment={state.get('asr_segment', '')!r}")
        pieces.append(f"buffer={state.get('asr_buffer', '')!r}")
    elif state_name == "speak":
        pieces.append(f"text={state.get('text', '')!r}")
        if state.get("asr_segment"):
            pieces.append(f"segment={state.get('asr_segment')!r}")
        if state.get("asr_buffer"):
            pieces.append(f"buffer={state.get('asr_buffer')!r}")

    return " ".join(pieces)


def build_asr_run_task(args: argparse.Namespace, task_id: str, sample_rate: int) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "format": "pcm",
        "sample_rate": sample_rate,
    }
    if args.language_hints:
        parameters["language_hints"] = [
            item.strip() for item in args.language_hints.split(",") if item.strip()
        ]

    optional_params = {
        "disfluency_removal_enabled": args.disfluency_removal_enabled,
        "semantic_punctuation_enabled": args.semantic_punctuation_enabled,
        "punctuation_prediction_enabled": args.punctuation_prediction_enabled,
        "heartbeat": args.heartbeat,
        "max_sentence_silence": args.max_sentence_silence,
        "multi_threshold_mode_enabled": args.multi_threshold_mode_enabled,
        "inverse_text_normalization_enabled": args.inverse_text_normalization_enabled,
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
            "task": args.asr_task,
            "function": "recognition",
            "model": args.asr_model,
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


async def soulx_receive_loop(
    ws: Any,
    stop_event: asyncio.Event,
    tracker: AsrTextTracker,
    show_raw_state: bool,
    show_json: bool,
) -> None:
    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except ConnectionClosed:
            break
        except Exception as exc:
            if not stop_event.is_set():
                print(f"[soulx recv error] {type(exc).__name__}: {exc}", file=sys.stderr)
            break

        data = parse_soulx_state(message)
        prefix = time.strftime("%H:%M:%S")
        if data is None:
            print(f"[{prefix}] soulx non-json response: {message!r}")
            continue
        state = data.get("state")
        if isinstance(state, dict) and state.get("state") == "speak":
            text = state.get("text")
            if isinstance(text, str):
                await tracker.consume(text)
        print(f"[{prefix}] soulx {format_soulx_state(data, show_raw_state, show_json)}")


async def wait_for_started(started: asyncio.Event, failed: asyncio.Event, timeout: float) -> None:
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


async def send_asr_audio_loop(
    ws: Any,
    audio: np.ndarray,
    chunk_size: int,
    chunk_ms: int,
    finished: asyncio.Event,
    no_realtime: bool,
) -> int:
    chunks_sent = 0
    loop = asyncio.get_running_loop()
    stream_start = loop.time()

    for chunk in iter_fixed_chunks(audio, chunk_size):
        if finished.is_set():
            break

        await ws.send(float_chunk_to_s16_pcm(chunk))
        chunks_sent += 1

        if not no_realtime:
            next_time = stream_start + chunks_sent * chunk_ms / 1000.0
            sleep_s = next_time - loop.time()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    return chunks_sent


async def send_soulx_audio_loop(
    ws: Any,
    audio: np.ndarray,
    sample_rate: int,
    chunk_size: int,
    send_interval: float,
    initial_delay_sec: float,
    session_id: str,
    tracker: AsrTextTracker,
    show_send: bool,
) -> int:
    chunks_sent = 0
    loop = asyncio.get_running_loop()

    if initial_delay_sec > 0:
        await asyncio.sleep(initial_delay_sec)

    stream_start = loop.time()

    for chunk in iter_fixed_chunks(audio, chunk_size):
        asr_text, asr_final, asr_responses = await tracker.pending_snapshot()
        payload = {
            "type": "audio",
            "session_id": session_id,
            "audio": float_chunk_to_b64(chunk),
            "asr_text": asr_text,
            "asr_final": asr_final,
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))
        chunks_sent += 1

        if show_send:
            prefix = time.strftime("%H:%M:%S")
            print(
                f"[{prefix}] send soulx_chunk={chunks_sent} "
                f"asr_responses={asr_responses} asr_text={asr_text!r}"
            )

        if send_interval > 0:
            next_time = stream_start + chunks_sent * send_interval
            sleep_s = next_time - loop.time()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    return chunks_sent


async def stream_to_asr_and_soulx(args: argparse.Namespace) -> None:
    audio_data = load_audio(args.audio, args.sample_rate)
    audio = add_tail_silence(audio_data.samples, audio_data.sample_rate, args.tail_silence_sec)
    soulx_session_id = args.session_id or uuid.uuid4().hex
    asr_task_id = args.asr_task_id or uuid.uuid4().hex
    tracker = AsrTextTracker()

    asr_chunk_size = max(1, audio_data.sample_rate * args.asr_chunk_ms // 1000)
    soulx_chunk_size = args.soulx_chunk_size
    send_interval = (
        args.send_interval_sec
        if args.send_interval_sec is not None
        else soulx_chunk_size / audio_data.sample_rate
    )
    soulx_audio_delay_sec = args.soulx_audio_delay_ms / 1000.0

    print(f"audio: {args.audio}")
    print(
        f"samples={len(audio_data.samples)} sample_rate={audio_data.sample_rate} "
        f"duration={len(audio_data.samples) / audio_data.sample_rate:.2f}s"
    )
    print(f"asr_url={args.asr_url} task_id={asr_task_id} chunk_ms={args.asr_chunk_ms}")
    print(
        f"soulx_url={args.soulx_url} session_id={soulx_session_id} "
        f"chunk_size={soulx_chunk_size}"
    )
    print(f"send_interval={send_interval:.3f}s realtime={args.send_interval_sec is None}")
    print(f"soulx_audio_delay={soulx_audio_delay_sec:.3f}s")

    asr_started = asyncio.Event()
    asr_finished = asyncio.Event()
    asr_failed = asyncio.Event()
    soulx_stop = asyncio.Event()

    async with connect(
        args.asr_url,
        open_timeout=args.connect_timeout,
        close_timeout=args.close_timeout,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        max_size=args.max_message_size,
        compression="deflate" if args.enable_compression else None,
    ) as asr_ws, connect(
        args.soulx_url,
        open_timeout=args.connect_timeout,
        close_timeout=args.close_timeout,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        max_size=args.max_message_size,
    ) as soulx_ws:
        asr_receiver = asyncio.create_task(
            asr_receive_loop(
                asr_ws,
                tracker,
                asr_started,
                asr_finished,
                asr_failed,
                args.show_asr,
            ),
            name="asr-receiver",
        )
        soulx_receiver = asyncio.create_task(
            soulx_receive_loop(
                soulx_ws,
                soulx_stop,
                tracker,
                args.show_raw_state,
                args.json,
            ),
            name="soulx-receiver",
        )

        try:
            await asr_ws.send(
                json.dumps(
                    build_asr_run_task(args, asr_task_id, audio_data.sample_rate),
                    ensure_ascii=False,
                )
            )
            await wait_for_started(asr_started, asr_failed, args.start_timeout)

            asr_sender = asyncio.create_task(
                send_asr_audio_loop(
                    asr_ws,
                    audio,
                    asr_chunk_size,
                    args.asr_chunk_ms,
                    asr_finished,
                    args.no_realtime,
                ),
                name="asr-sender",
            )
            soulx_sender = asyncio.create_task(
                send_soulx_audio_loop(
                    soulx_ws,
                    audio,
                    audio_data.sample_rate,
                    soulx_chunk_size,
                    0.0 if args.no_realtime else send_interval,
                    soulx_audio_delay_sec,
                    soulx_session_id,
                    tracker,
                    args.show_send,
                ),
                name="soulx-sender",
            )
            asr_sent, soulx_sent = await asyncio.gather(asr_sender, soulx_sender)

            if not asr_finished.is_set():
                await asr_ws.send(
                    json.dumps(build_asr_finish_task(asr_task_id), ensure_ascii=False)
                )
                try:
                    await asyncio.wait_for(asr_finished.wait(), timeout=args.finish_timeout)
                except asyncio.TimeoutError:
                    print(
                        f"[asr warning] timed out waiting task-finished after "
                        f"{args.finish_timeout:g}s",
                        file=sys.stderr,
                    )

            if args.final_asr_flush_sec > 0:
                loop = asyncio.get_running_loop()
                end_at = loop.time() + args.final_asr_flush_sec
                last_text = None
                while loop.time() < end_at:
                    asr_text, asr_final, _ = await tracker.pending_snapshot()
                    if asr_text and asr_text != last_text:
                        payload = {
                            "type": "audio",
                            "session_id": soulx_session_id,
                            "audio": float_chunk_to_b64(
                                np.zeros(soulx_chunk_size, dtype=np.float32)
                            ),
                            "asr_text": asr_text,
                            "asr_final": asr_final,
                        }
                        await soulx_ws.send(json.dumps(payload, ensure_ascii=False))
                        last_text = asr_text
                    await asyncio.sleep(min(0.2, args.final_asr_flush_sec))

            if args.drain_sec > 0:
                await asyncio.sleep(args.drain_sec)

            asr_text, _, asr_responses = await tracker.snapshot()
            print(
                f"stopped; soulx_chunks_sent={soulx_sent} asr_chunks_sent={asr_sent} "
                f"asr_responses={asr_responses} final_asr_text={asr_text!r}"
            )
        finally:
            soulx_stop.set()
            for ws in (asr_ws, soulx_ws):
                try:
                    await ws.close()
                except Exception:
                    pass
            await asyncio.gather(asr_receiver, soulx_receiver, return_exceptions=True)


def add_optional_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    dest: str,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one audio file to local ASR and SoulX-Duplug. The SoulX "
            "request carries the latest cumulative streaming ASR text."
        )
    )
    parser.add_argument("audio", help="Path to an audio file readable by soundfile or ffmpeg.")

    parser.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    parser.add_argument("--asr-model", default=DEFAULT_MODEL)
    parser.add_argument("--asr-task", default=DEFAULT_TASK)
    parser.add_argument("--asr-task-id", default="")
    parser.add_argument("--asr-chunk-ms", type=int, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--language-hints", default="zh")

    parser.add_argument("--soulx-url", default=DEFAULT_SOULX_URL)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--soulx-chunk-size", type=int, default=DEFAULT_SOULX_CHUNK_SIZE)

    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--tail-silence-sec", type=float, default=2.0)
    parser.add_argument("--drain-sec", type=float, default=2.0)
    parser.add_argument(
        "--send-interval-sec",
        type=float,
        default=None,
        help="Override send interval. Default sends SoulX chunks at audio realtime speed.",
    )
    parser.add_argument(
        "--soulx-audio-delay-ms",
        "--audio-delay-ms",
        dest="soulx_audio_delay_ms",
        type=float,
        default=0.0,
        help=(
            "Delay SoulX audio sending by N milliseconds so ASR has a head start. "
            "This approximates ASR latency alignment."
        ),
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send ASR and SoulX chunks as fast as possible.",
    )
    parser.add_argument(
        "--final-asr-flush-sec",
        type=float,
        default=1.0,
        help="After audio ends, keep sending silence with updated final ASR text for N seconds.",
    )

    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--start-timeout", type=float, default=30.0)
    parser.add_argument("--finish-timeout", type=float, default=60.0)
    parser.add_argument("--close-timeout", type=float, default=5.0)
    parser.add_argument("--ping-interval", type=float, default=20.0)
    parser.add_argument("--ping-timeout", type=float, default=20.0)
    parser.add_argument("--max-message-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--enable-compression", action="store_true")

    parser.add_argument("--show-asr", action="store_true")
    parser.add_argument("--show-send", action="store_true")
    parser.add_argument("--show-raw-state", action="store_true")
    parser.add_argument("--json", action="store_true")

    add_optional_bool_arg(
        parser,
        "disfluency-removal-enabled",
        "disfluency_removal_enabled",
        "disfluency_removal_enabled",
    )
    add_optional_bool_arg(
        parser,
        "semantic-punctuation-enabled",
        "semantic_punctuation_enabled",
        "semantic_punctuation_enabled",
    )
    add_optional_bool_arg(
        parser,
        "punctuation-prediction-enabled",
        "punctuation_prediction_enabled",
        "punctuation_prediction_enabled",
    )
    add_optional_bool_arg(parser, "heartbeat", "heartbeat", "heartbeat")
    add_optional_bool_arg(
        parser,
        "multi-threshold-mode-enabled",
        "multi_threshold_mode_enabled",
        "multi_threshold_mode_enabled",
    )
    add_optional_bool_arg(
        parser,
        "inverse-text-normalization-enabled",
        "inverse_text_normalization_enabled",
        "inverse_text_normalization_enabled",
    )
    parser.add_argument("--max-sentence-silence", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.asr_chunk_ms <= 0:
        raise SystemExit("--asr-chunk-ms must be > 0")
    if args.soulx_chunk_size <= 0:
        raise SystemExit("--soulx-chunk-size must be > 0")
    if args.soulx_audio_delay_ms < 0:
        raise SystemExit("--soulx-audio-delay-ms must be >= 0")

    try:
        asyncio.run(stream_to_asr_and_soulx(args))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
