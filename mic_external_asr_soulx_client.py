#!/usr/bin/env python3
"""
Single-microphone bridge client:
1. capture local microphone audio;
2. stream it to an external ASR Gateway WebSocket;
3. stream the same audio to SoulX-Duplug with the latest cumulative ASR text.

Run:
  python mic_external_asr_soulx_client.py --show-asr --show-raw-state
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
import uuid
from typing import Any

import numpy as np

from external_asr_soulx_client import (
    DEFAULT_ASR_URL,
    DEFAULT_CHUNK_MS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SOULX_CHUNK_SIZE,
    DEFAULT_SOULX_URL,
    DEFAULT_TASK,
    AsrTextTracker,
    add_optional_bool_arg,
    asr_receive_loop,
    build_asr_finish_task,
    build_asr_run_task,
    float_chunk_to_b64,
    float_chunk_to_s16_pcm,
    soulx_receive_loop,
    wait_for_started,
)

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


def import_sounddevice():
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "sounddevice/PortAudio is unavailable. On Ubuntu install "
            "`sudo apt-get install portaudio19-dev`, then reinstall sounddevice."
        ) from exc
    return sd


def list_devices() -> None:
    sd = import_sounddevice()
    print(sd.query_devices())


def install_stop_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
        except NotImplementedError:
            signal.signal(signum, lambda _sig, _frame: request_stop())


class SoulxChunkBuffer:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.parts: list[np.ndarray] = []
        self.sample_count = 0

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        if len(chunk) == 0:
            return []

        self.parts.append(chunk.astype(np.float32, copy=False))
        self.sample_count += len(chunk)

        ready: list[np.ndarray] = []
        while self.sample_count >= self.chunk_size:
            merged = np.concatenate(self.parts)
            ready.append(merged[: self.chunk_size].astype(np.float32, copy=False))
            remain = merged[self.chunk_size :]
            self.parts = [remain] if len(remain) else []
            self.sample_count = len(remain)
        return ready

    def flush_with_padding(self) -> np.ndarray | None:
        if self.sample_count == 0:
            return None
        merged = np.concatenate(self.parts)
        if len(merged) < self.chunk_size:
            merged = np.pad(merged, (0, self.chunk_size - len(merged)))
        self.parts = []
        self.sample_count = 0
        return merged.astype(np.float32, copy=False)


async def send_soulx_chunk(
    ws: Any,
    chunk: np.ndarray,
    session_id: str,
    tracker: AsrTextTracker,
    show_send: bool,
    chunk_index: int,
) -> None:
    asr_text, asr_final, asr_responses = await tracker.snapshot()
    payload = {
        "type": "audio",
        "session_id": session_id,
        "audio": float_chunk_to_b64(chunk),
        "asr_text": asr_text,
        "asr_final": asr_final,
    }
    await ws.send(json.dumps(payload, ensure_ascii=False))

    if show_send:
        prefix = time.strftime("%H:%M:%S")
        print(
            f"[{prefix}] send soulx_chunk={chunk_index} "
            f"asr_responses={asr_responses} asr_text={asr_text!r}"
        )


async def mic_send_loop(
    audio_queue: asyncio.Queue[np.ndarray],
    asr_ws: Any,
    soulx_ws: Any,
    tracker: AsrTextTracker,
    asr_finished: asyncio.Event,
    stop_event: asyncio.Event,
    args: argparse.Namespace,
) -> tuple[int, int]:
    soulx_buffer = SoulxChunkBuffer(args.soulx_chunk_size)
    asr_sent = 0
    soulx_sent = 0

    while not stop_event.is_set():
        try:
            mic_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        if not asr_finished.is_set():
            await asr_ws.send(float_chunk_to_s16_pcm(mic_chunk))
            asr_sent += 1

        for soulx_chunk in soulx_buffer.push(mic_chunk):
            soulx_sent += 1
            await send_soulx_chunk(
                soulx_ws,
                soulx_chunk,
                args.session_id,
                tracker,
                args.show_send,
                soulx_sent,
            )

    flush_chunk = soulx_buffer.flush_with_padding()
    if flush_chunk is not None:
        soulx_sent += 1
        await send_soulx_chunk(
            soulx_ws,
            flush_chunk,
            args.session_id,
            tracker,
            args.show_send,
            soulx_sent,
        )

    return asr_sent, soulx_sent


async def send_tail_silence(
    soulx_ws: Any,
    asr_ws: Any,
    tracker: AsrTextTracker,
    asr_finished: asyncio.Event,
    args: argparse.Namespace,
) -> tuple[int, int]:
    tail_samples = max(0, int(args.tail_silence_sec * args.sample_rate))
    if tail_samples == 0:
        return 0, 0

    asr_chunk_size = max(1, args.sample_rate * args.asr_chunk_ms // 1000)
    soulx_buffer = SoulxChunkBuffer(args.soulx_chunk_size)
    asr_sent = 0
    soulx_sent = 0

    for start in range(0, tail_samples, asr_chunk_size):
        chunk_size = min(asr_chunk_size, tail_samples - start)
        chunk = np.zeros(chunk_size, dtype=np.float32)

        if not asr_finished.is_set():
            await asr_ws.send(float_chunk_to_s16_pcm(chunk))
            asr_sent += 1

        for soulx_chunk in soulx_buffer.push(chunk):
            soulx_sent += 1
            await send_soulx_chunk(
                soulx_ws,
                soulx_chunk,
                args.session_id,
                tracker,
                args.show_send,
                soulx_sent,
            )

        await asyncio.sleep(args.asr_chunk_ms / 1000.0)

    flush_chunk = soulx_buffer.flush_with_padding()
    if flush_chunk is not None:
        soulx_sent += 1
        await send_soulx_chunk(
            soulx_ws,
            flush_chunk,
            args.session_id,
            tracker,
            args.show_send,
            soulx_sent,
        )

    return asr_sent, soulx_sent


async def final_asr_flush(
    soulx_ws: Any,
    tracker: AsrTextTracker,
    args: argparse.Namespace,
) -> None:
    if args.final_asr_flush_sec <= 0:
        return

    loop = asyncio.get_running_loop()
    end_at = loop.time() + args.final_asr_flush_sec
    last_text = None
    while loop.time() < end_at:
        asr_text, asr_final, _ = await tracker.snapshot()
        if asr_text and asr_text != last_text:
            payload = {
                "type": "audio",
                "session_id": args.session_id,
                "audio": float_chunk_to_b64(
                    np.zeros(args.soulx_chunk_size, dtype=np.float32)
                ),
                "asr_text": asr_text,
                "asr_final": asr_final,
            }
            await soulx_ws.send(json.dumps(payload, ensure_ascii=False))
            last_text = asr_text
        await asyncio.sleep(min(0.2, args.final_asr_flush_sec))


async def run_mic_bridge(args: argparse.Namespace) -> None:
    sd = import_sounddevice()

    args.session_id = args.session_id or uuid.uuid4().hex
    asr_task_id = args.asr_task_id or uuid.uuid4().hex
    asr_block_size = max(1, args.sample_rate * args.asr_chunk_ms // 1000)
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=args.queue_size)
    stop_event = asyncio.Event()
    tracker = AsrTextTracker()

    install_stop_handlers(stop_event)
    loop = asyncio.get_running_loop()

    def audio_callback(indata, frames, _time_info, status) -> None:
        if status:
            print(f"[audio warning] {status}", file=sys.stderr)
        if stop_event.is_set():
            return

        mono = indata[:, 0].astype(np.float32, copy=True)
        if frames < asr_block_size:
            mono = np.pad(mono, (0, asr_block_size - frames))

        def enqueue() -> None:
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                print("[audio warning] audio queue full; dropped oldest chunk", file=sys.stderr)
            audio_queue.put_nowait(mono)

        loop.call_soon_threadsafe(enqueue)

    asr_started = asyncio.Event()
    asr_finished = asyncio.Event()
    asr_failed = asyncio.Event()
    soulx_stop = asyncio.Event()

    print(f"asr_url={args.asr_url} task_id={asr_task_id} chunk_ms={args.asr_chunk_ms}")
    print(
        f"soulx_url={args.soulx_url} session_id={args.session_id} "
        f"chunk_size={args.soulx_chunk_size}"
    )
    print("recording; press Ctrl+C to stop")

    stream_kwargs = {
        "samplerate": args.sample_rate,
        "channels": 1,
        "dtype": "float32",
        "blocksize": asr_block_size,
        "callback": audio_callback,
    }
    if args.device is not None:
        stream_kwargs["device"] = args.device

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
                args.show_raw_state,
                args.json,
            ),
            name="soulx-receiver",
        )

        asr_sent = 0
        soulx_sent = 0
        try:
            await asr_ws.send(
                json.dumps(
                    build_asr_run_task(args, asr_task_id, args.sample_rate),
                    ensure_ascii=False,
                )
            )
            await wait_for_started(asr_started, asr_failed, args.start_timeout)

            with sd.InputStream(**stream_kwargs):
                asr_sent, soulx_sent = await mic_send_loop(
                    audio_queue,
                    asr_ws,
                    soulx_ws,
                    tracker,
                    asr_finished,
                    stop_event,
                    args,
                )

            tail_asr_sent, tail_soulx_sent = await send_tail_silence(
                soulx_ws,
                asr_ws,
                tracker,
                asr_finished,
                args,
            )
            asr_sent += tail_asr_sent
            soulx_sent += tail_soulx_sent

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

            await final_asr_flush(soulx_ws, tracker, args)

            if args.drain_sec > 0:
                await asyncio.sleep(args.drain_sec)

            asr_text, _, asr_responses = await tracker.snapshot()
            print(
                f"stopped; soulx_chunks_sent={soulx_sent} asr_chunks_sent={asr_sent} "
                f"asr_responses={asr_responses} final_asr_text={asr_text!r}"
            )
        finally:
            stop_event.set()
            soulx_stop.set()
            for ws in (asr_ws, soulx_ws):
                try:
                    await ws.close()
                except Exception:
                    pass
            await asyncio.gather(asr_receiver, soulx_receiver, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one local microphone stream, send it to external ASR, and "
            "send SoulX-Duplug audio chunks with latest cumulative ASR text."
        )
    )

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
    parser.add_argument("--queue-size", type=int, default=100)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--tail-silence-sec", type=float, default=1.0)
    parser.add_argument("--final-asr-flush-sec", type=float, default=1.0)
    parser.add_argument("--drain-sec", type=float, default=1.0)

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
    if args.list_devices:
        list_devices()
        return
    if args.asr_chunk_ms <= 0:
        raise SystemExit("--asr-chunk-ms must be > 0")
    if args.soulx_chunk_size <= 0:
        raise SystemExit("--soulx-chunk-size must be > 0")
    if args.queue_size <= 0:
        raise SystemExit("--queue-size must be > 0")

    try:
        asyncio.run(run_mic_bridge(args))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
