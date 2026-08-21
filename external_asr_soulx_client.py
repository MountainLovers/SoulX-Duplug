#!/usr/bin/env python3
"""
Stream an audio file to an external ASR Gateway and to SoulX-Duplug.

The external ASR adapter, text tracking, audio helpers, and SoulX WebSocket
payloads live in separate modules under clients/. This script only wires those
atomic capabilities together for a runnable example.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import numpy as np

from clients.audio import (
    add_tail_silence,
    float_chunk_to_s16_pcm,
    iter_fixed_chunks,
    load_audio,
)
from clients.external_asr_gateway import (
    DEFAULT_ASR_CHUNK_MS,
    DEFAULT_ASR_MODEL,
    DEFAULT_ASR_TASK,
    DEFAULT_ASR_URL,
    AsrRunConfig,
    AsrTextTracker,
    asr_receive_loop,
    build_asr_finish_task,
    build_asr_run_task,
    wait_for_started,
)
from clients.soulx_turn import (
    DEFAULT_SOULX_CHUNK_SIZE,
    DEFAULT_SOULX_URL,
    send_soulx_chunk,
    soulx_receive_loop,
)

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


DEFAULT_SAMPLE_RATE = 16000


def add_optional_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    dest: str,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}", dest=dest, action="store_true", default=None, help=help_text
    )
    group.add_argument(
        f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}"
    )


def build_asr_config(args: argparse.Namespace) -> AsrRunConfig:
    return AsrRunConfig(
        model=args.asr_model,
        task=args.asr_task,
        language_hints=args.language_hints,
        disfluency_removal_enabled=args.disfluency_removal_enabled,
        semantic_punctuation_enabled=args.semantic_punctuation_enabled,
        punctuation_prediction_enabled=args.punctuation_prediction_enabled,
        heartbeat=args.heartbeat,
        max_sentence_silence=args.max_sentence_silence,
        multi_threshold_mode_enabled=args.multi_threshold_mode_enabled,
        inverse_text_normalization_enabled=args.inverse_text_normalization_enabled,
    )


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
    send_pending_text: bool,
    show_send: bool,
) -> int:
    chunks_sent = 0
    loop = asyncio.get_running_loop()

    if initial_delay_sec > 0:
        await asyncio.sleep(initial_delay_sec)

    stream_start = loop.time()

    for chunk in iter_fixed_chunks(audio, chunk_size):
        if send_pending_text:
            asr_text, asr_final, asr_responses = await tracker.pending_snapshot()
        else:
            asr_text, asr_final, asr_responses = await tracker.snapshot()

        await send_soulx_chunk(
            ws,
            session_id=session_id,
            audio_chunk=chunk,
            asr_text=asr_text,
            asr_final=asr_final,
        )
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


async def flush_final_asr_to_soulx(
    ws: Any,
    session_id: str,
    chunk_size: int,
    tracker: AsrTextTracker,
    duration_sec: float,
) -> None:
    if duration_sec <= 0:
        return

    loop = asyncio.get_running_loop()
    end_at = loop.time() + duration_sec
    last_text = None
    silence = np.zeros(chunk_size, dtype=np.float32)

    while loop.time() < end_at:
        asr_text, asr_final, _ = await tracker.pending_snapshot()
        if asr_text and asr_text != last_text:
            await send_soulx_chunk(
                ws,
                session_id=session_id,
                audio_chunk=silence,
                asr_text=asr_text,
                asr_final=asr_final,
            )
            last_text = asr_text
        await asyncio.sleep(min(0.2, duration_sec))


async def stream_file_to_asr_and_soulx(args: argparse.Namespace) -> None:
    audio_data = load_audio(args.audio, args.sample_rate)
    audio = add_tail_silence(
        audio_data.samples, audio_data.sample_rate, args.tail_silence_sec
    )
    soulx_session_id = args.session_id or uuid.uuid4().hex
    asr_task_id = args.asr_task_id or uuid.uuid4().hex
    tracker = AsrTextTracker()

    asr_chunk_size = max(1, audio_data.sample_rate * args.asr_chunk_ms // 1000)
    send_interval = (
        args.send_interval_sec
        if args.send_interval_sec is not None
        else args.soulx_chunk_size / audio_data.sample_rate
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
        f"chunk_size={args.soulx_chunk_size}"
    )
    print(f"send_interval={send_interval:.3f}s realtime={not args.no_realtime}")
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
                    build_asr_run_task(
                        build_asr_config(args),
                        asr_task_id,
                        audio_data.sample_rate,
                    ),
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
                    args.soulx_chunk_size,
                    0.0 if args.no_realtime else send_interval,
                    soulx_audio_delay_sec,
                    soulx_session_id,
                    tracker,
                    args.send_pending_asr_text,
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
                    await asyncio.wait_for(
                        asr_finished.wait(), timeout=args.finish_timeout
                    )
                except asyncio.TimeoutError:
                    print(
                        f"[asr warning] timed out waiting task-finished after "
                        f"{args.finish_timeout:g}s"
                    )

            await flush_final_asr_to_soulx(
                soulx_ws,
                session_id=soulx_session_id,
                chunk_size=args.soulx_chunk_size,
                tracker=tracker,
                duration_sec=args.final_asr_flush_sec,
            )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one audio file to an external ASR Gateway and send SoulX-Duplug "
            "audio chunks with current-turn ASR text."
        )
    )
    parser.add_argument(
        "audio", help="Path to an audio file readable by soundfile or ffmpeg."
    )

    parser.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-task", default=DEFAULT_ASR_TASK)
    parser.add_argument("--asr-task-id", default="")
    parser.add_argument("--asr-chunk-ms", type=int, default=DEFAULT_ASR_CHUNK_MS)
    parser.add_argument("--language-hints", default="zh")

    parser.add_argument("--soulx-url", default=DEFAULT_SOULX_URL)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--soulx-chunk-size", type=int, default=DEFAULT_SOULX_CHUNK_SIZE
    )

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
        help="Delay SoulX audio sending so ASR has a head start.",
    )
    parser.add_argument(
        "--no-realtime", action="store_true", help="Send chunks as fast as possible."
    )
    parser.add_argument(
        "--final-asr-flush-sec",
        type=float,
        default=1.0,
        help="After audio ends, keep sending silence with updated final ASR text.",
    )
    text_scope_group = parser.add_mutually_exclusive_group()
    text_scope_group.add_argument(
        "--send-pending-asr-text",
        dest="send_pending_asr_text",
        action="store_true",
        help="Send only current-turn text not yet consumed by a SoulX speak state.",
    )
    text_scope_group.add_argument(
        "--send-full-asr-text",
        dest="send_pending_asr_text",
        action="store_false",
        help="Send the full ASR task transcript. This is mainly useful for debugging.",
    )
    parser.set_defaults(send_pending_asr_text=True)

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
        asyncio.run(stream_file_to_asr_and_soulx(args))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
