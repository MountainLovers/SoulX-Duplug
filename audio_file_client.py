import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import soxr
import websocket


def encode_audio(audio_chunk):
    return base64.b64encode(
        np.asarray(audio_chunk, dtype=np.float32).tobytes()
    ).decode()


def load_audio(path, target_sample_rate):
    try:
        audio, sample_rate = sf.read(path, always_2d=False)
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"Failed to read audio with soundfile and ffmpeg is not available: {exc}"
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

    return audio, sample_rate


def iter_chunks(audio, chunk_size, tail_silence_samples):
    if tail_silence_samples > 0:
        audio = np.concatenate(
            [audio, np.zeros(tail_silence_samples, dtype=np.float32)]
        )

    total_chunks = math.ceil(len(audio) / chunk_size)
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        yield chunk.astype(np.float32, copy=False)


def format_state(data, show_raw_state=False, show_json=False):
    if show_json:
        return json.dumps(data, ensure_ascii=False)

    state = data.get("state", {})
    state_name = state.get("state", "")
    raw_state = state.get("raw_state", "")
    pieces = [state_name or "unknown"]

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


def recv_loop(ws, stop_event, show_raw_state, show_json):
    while not stop_event.is_set():
        try:
            msg = ws.recv()
            if not msg:
                continue
            data = json.loads(msg)
            prefix = time.strftime("%H:%M:%S")
            print(f"[{prefix}] {format_state(data, show_raw_state, show_json)}")
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            if not stop_event.is_set():
                print(f"[recv error] {exc}", file=sys.stderr)
            break


def run(args):
    audio, sample_rate = load_audio(args.audio, args.sample_rate)
    tail_silence_samples = int(args.tail_silence_sec * sample_rate)
    session_id = args.session_id or uuid.uuid4().hex

    ws = websocket.create_connection(args.url, timeout=args.timeout)
    ws.settimeout(args.timeout)
    stop_event = threading.Event()
    recv_thread = threading.Thread(
        target=recv_loop,
        args=(ws, stop_event, args.show_raw_state, args.json),
        daemon=True,
    )
    recv_thread.start()

    print(f"connected: {args.url}")
    print(f"session_id: {session_id}")
    print(
        f"audio: {args.audio} samples={len(audio)} sample_rate={sample_rate} "
        f"duration={len(audio) / sample_rate:.2f}s"
    )
    send_interval = (
        args.send_interval_sec
        if args.send_interval_sec is not None
        else args.chunk_size / sample_rate
    )
    print(f"send_interval={send_interval:.3f}s realtime={args.send_interval_sec is None}")

    chunks_sent = 0

    try:
        for chunk in iter_chunks(audio, args.chunk_size, tail_silence_samples):
            payload = {
                "type": "audio",
                "session_id": session_id,
                "audio": encode_audio(chunk),
            }
            if args.asr_text:
                payload["asr_text"] = args.asr_text
            ws.send(json.dumps(payload))
            chunks_sent += 1

            if send_interval > 0:
                time.sleep(send_interval)

        time.sleep(args.drain_sec)
    finally:
        stop_event.set()
        try:
            ws.close()
        except Exception:
            pass
        recv_thread.join(timeout=1.0)
        print(f"stopped; chunks_sent={chunks_sent}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream an audio file to the SoulX-Duplug WebSocket server."
    )
    parser.add_argument(
        "audio", help="Path to an audio file readable by soundfile or ffmpeg."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/turn")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-size", type=int, default=2560)
    parser.add_argument("--tail-silence-sec", type=float, default=2.0)
    parser.add_argument("--drain-sec", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--send-interval-sec",
        type=float,
        default=None,
        help="Override send interval in seconds. Default sends at audio realtime speed.",
    )
    parser.add_argument("--show-raw-state", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--asr-text",
        default="",
        help="Optional cumulative ASR text to send with every audio chunk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
