from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import numpy as np

from clients.audio import float_chunk_to_b64
from clients.external_asr_gateway import AsrTextTracker

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ConnectionClosed = Exception


DEFAULT_SOULX_URL = "ws://127.0.0.1:8000/turn"
DEFAULT_SOULX_CHUNK_SIZE = 2560


def build_audio_payload(
    session_id: str,
    audio_chunk: np.ndarray,
    asr_text: str | None = None,
    asr_final: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "audio",
        "session_id": session_id,
        "audio": float_chunk_to_b64(audio_chunk),
    }
    if asr_text is not None:
        payload["asr_text"] = asr_text
        payload["asr_final"] = bool(asr_final)
    return payload


def parse_turn_state(message: str | bytes) -> dict[str, Any] | None:
    if isinstance(message, bytes):
        message = message.decode("utf-8", "replace")
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def format_turn_state(
    data: dict[str, Any],
    show_raw_state: bool = False,
    show_json: bool = False,
) -> str:
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


async def soulx_receive_loop(
    ws: Any,
    stop_event: asyncio.Event,
    tracker: AsrTextTracker | None = None,
    show_raw_state: bool = False,
    show_json: bool = False,
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
                print(
                    f"[soulx recv error] {type(exc).__name__}: {exc}", file=sys.stderr
                )
            break

        data = parse_turn_state(message)
        prefix = time.strftime("%H:%M:%S")
        if data is None:
            print(f"[{prefix}] soulx non-json response: {message!r}")
            continue

        state = data.get("state")
        if (
            tracker is not None
            and isinstance(state, dict)
            and state.get("state") == "speak"
        ):
            text = state.get("text")
            if isinstance(text, str):
                await tracker.consume(text)

        print(f"[{prefix}] soulx {format_turn_state(data, show_raw_state, show_json)}")


async def send_soulx_chunk(
    ws: Any,
    session_id: str,
    audio_chunk: np.ndarray,
    asr_text: str | None = None,
    asr_final: bool = False,
) -> None:
    payload = build_audio_payload(
        session_id=session_id,
        audio_chunk=audio_chunk,
        asr_text=asr_text,
        asr_final=asr_final,
    )
    await ws.send(json.dumps(payload, ensure_ascii=False))
