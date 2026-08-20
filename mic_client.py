import argparse
import base64
import json
import queue
import signal
import sys
import threading
import time
import uuid

import numpy as np
import sounddevice as sd
import websocket


def list_devices():
    print(sd.query_devices())


def encode_audio(audio_chunk):
    return base64.b64encode(
        np.asarray(audio_chunk, dtype=np.float32).tobytes()
    ).decode()


def print_state(data):
    state = data.get("state", {})
    state_name = state.get("state", "")
    prefix = time.strftime("%H:%M:%S")

    if state_name == "idle":
        print(f"[{prefix}] idle")
    elif state_name == "blank":
        print(f"[{prefix}] blank")
    elif state_name == "nonidle":
        segment = state.get("asr_segment", "")
        buffer_text = state.get("asr_buffer", "")
        print(f"[{prefix}] nonidle segment={segment!r} buffer={buffer_text!r}")
    elif state_name == "backchannel":
        segment = state.get("asr_segment", "")
        buffer_text = state.get("asr_buffer", "")
        print(f"[{prefix}] backchannel segment={segment!r} buffer={buffer_text!r}")
    elif state_name == "incomplete":
        segment = state.get("asr_segment", "")
        buffer_text = state.get("asr_buffer", "")
        print(f"[{prefix}] incomplete segment={segment!r} buffer={buffer_text!r}")
    elif state_name == "speak":
        text = state.get("text", "")
        print(f"[{prefix}] speak text={text!r}")
    else:
        print(f"[{prefix}] {json.dumps(data, ensure_ascii=False)}")


def recv_loop(ws, stop_event):
    while not stop_event.is_set():
        try:
            msg = ws.recv()
            if not msg:
                continue
            print_state(json.loads(msg))
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as exc:
            if not stop_event.is_set():
                print(f"[recv error] {exc}", file=sys.stderr)
            stop_event.set()
            break


def run(args):
    audio_queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()

    def handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def audio_callback(indata, frames, _time_info, status):
        if status:
            print(f"[audio warning] {status}", file=sys.stderr)
        if stop_event.is_set():
            return
        mono = indata[:, 0].astype(np.float32, copy=True)
        if frames < args.chunk_size:
            mono = np.pad(mono, (0, args.chunk_size - frames))
        try:
            audio_queue.put_nowait(mono)
        except queue.Full:
            print("[audio warning] audio queue full; dropping chunk", file=sys.stderr)

    session_id = args.session_id or uuid.uuid4().hex
    ws = websocket.create_connection(args.url, timeout=args.timeout)
    ws.settimeout(args.timeout)

    recv_thread = threading.Thread(target=recv_loop, args=(ws, stop_event), daemon=True)
    recv_thread.start()

    print(f"connected: {args.url}")
    print(f"session_id: {session_id}")
    print("recording; press Ctrl+C to stop")

    stream_kwargs = {
        "samplerate": args.sample_rate,
        "channels": 1,
        "dtype": "float32",
        "blocksize": args.chunk_size,
        "callback": audio_callback,
    }
    if args.device is not None:
        stream_kwargs["device"] = args.device

    try:
        with sd.InputStream(**stream_kwargs):
            while not stop_event.is_set():
                try:
                    audio = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                payload = {
                    "type": "audio",
                    "session_id": session_id,
                    "audio": encode_audio(audio),
                }
                ws.send(json.dumps(payload))
    finally:
        stop_event.set()
        try:
            ws.close()
        except Exception:
            pass
        recv_thread.join(timeout=1.0)
        print("stopped")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream local microphone audio to the SoulX-Duplug WebSocket server."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/turn")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-size", type=int, default=2560)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.list_devices:
        list_devices()
    else:
        run(cli_args)
