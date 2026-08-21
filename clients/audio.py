from __future__ import annotations

import base64
import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def load_audio(path: str, target_sample_rate: int) -> AudioData:
    """Load an audio file as mono float32 samples at target_sample_rate."""
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


def iter_fixed_chunks(audio: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    total_chunks = math.ceil(len(audio) / chunk_size)
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        yield chunk.astype(np.float32, copy=False)


def float_chunk_to_s16_pcm(chunk: np.ndarray) -> bytes:
    clipped = np.clip(chunk, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def float_chunk_to_b64(chunk: np.ndarray) -> str:
    return base64.b64encode(np.asarray(chunk, dtype=np.float32).tobytes()).decode()
