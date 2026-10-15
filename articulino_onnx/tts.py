"""Public API: TTS.from_pretrained(), set_voice(), say(), stream() - same as the PyTorch release."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import onnxruntime as ort
import soundfile as sf

from . import dsp
from . import text as T
from .engine import Engine, Voice

SAMPLE_RATE = 24000
PRECISIONS = ("fp32", "fp16", "int8")


@dataclass
class Speech:
    """Result of say(): float32 audio in [-1, 1], its sample rate and the seed."""
    audio: np.ndarray
    sample_rate: int
    seed: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str | os.PathLike) -> None:
        sf.write(str(path), self.audio, self.sample_rate, subtype="PCM_16")

    def __array__(self, dtype=None, copy=None):
        return self.audio if dtype is None else self.audio.astype(dtype)

    def __len__(self) -> int:
        return len(self.audio)


_frontend = None


def _german_frontend():
    """The German text frontend (numbers, dates, units, abbreviations ...), loaded once and silently."""
    global _frontend
    if _frontend is None:
        from . import frontend
        _frontend = frontend
    return _frontend


class _TagGuard:
    """Holds back a possibly incomplete [en] / [/en] tag at the end of a token stream."""

    def __init__(self):
        self.rest = ""

    def push(self, piece: str) -> str:
        s = T.protect_tags(self.rest + piece)
        i = s.rfind("[")
        if i >= 0 and len(s) - i <= 5 and "]" not in s[i:]:
            self.rest = s[i:]
            return s[:i]
        self.rest = ""
        return s

    def flush(self) -> str:
        s, self.rest = T.protect_tags(self.rest), ""
        return s


class TTS:
    """Articulino on ONNX Runtime.

    >>> tts = TTS.from_pretrained("path/to/Articulino-DE-EN-Realtime-Stream-TTS-ONNX")
    >>> tts.set_voice("voice.wav")
    >>> tts.say("Guten Morgen!", path="morgen.wav")
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, engine: Engine, uniform_rate: bool = True, retry_broken: bool = True, pause: float = 0.15):
        self.engine = engine
        self.uniform_rate = uniform_rate          # every sentence at the voice's own letter rate x speed (default of the original service)
        self.retry_broken = retry_broken          # redo a sentence once with seed+1 if it comes out broken
        self.pause = pause                        # seconds of silence between sentences
        self.voice: Voice | None = None
        c = engine.config["postprocess"]
        self._shelf = dsp.Shelf(c["low_shelf_db"], c["low_shelf_hz"])
        self._peak = float(c["peak"])

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str | os.PathLike = "eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-ONNX",
                        precision: str = "auto", providers: list | None = None, threads: int | None = None,
                        **kw) -> "TTS":
        """path: a local folder (config.json + fp32/ fp16/ int8/) or a Hugging Face repo id (needs huggingface_hub).
        precision: "fp32", "fp16" (GPU), "int8" (CPU) or "auto" (fp16 with CUDA/DirectML/CoreML, else fp32).
        providers: ONNX Runtime execution providers; default: CUDA, DirectML or CoreML if installed, else CPU.
        threads: CPU threads (default: ONNX Runtime's choice). All graphs share one process-wide thread pool, so the
        first TTS of a process fixes the number."""
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
                         if p in available] + ["CPUExecutionProvider"]
        if precision == "auto":
            precision = "fp16" if providers[0] != "CPUExecutionProvider" else "fp32"
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS} or 'auto'")
        folder = Path(path)
        if not folder.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise FileNotFoundError(f"{path} is not a folder; install huggingface_hub to download it") from exc
            folder = Path(snapshot_download(str(path), allow_patterns=["config.json", f"{precision}/*"]))
        return cls(Engine(folder, precision, providers, threads), **kw)

    # ------------------------------------------------------------------ voice
    def set_voice(self, voice: str | os.PathLike | np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        """A 3-8 s recording of clean speech (path or float array). Longer recordings are cut to 8 s."""
        if isinstance(voice, (str, os.PathLike)):
            wav, sample_rate = sf.read(str(voice), dtype="float32", always_2d=True)
            wav = wav.mean(1)
        else:
            wav = np.asarray(voice, np.float32)
            wav = wav.mean(1) if wav.ndim == 2 else wav
        wav = dsp.resample(wav, int(sample_rate), SAMPLE_RATE)
        self.voice = self.engine.voice(self.engine.encode(wav))

    # ------------------------------------------------------------------ speech
    def _prepare(self, text: str, language: str, normalize: bool) -> str:
        text = T.protect_tags(text)
        if normalize and language == "de":
            text = _german_frontend().normalize(text)
        return text

    def sentence(self, sentence: str, language: str = "de", speed: float = 1.0, seed: int = 0) -> np.ndarray:
        """One sentence (already through the frontend) -> float32 audio at 24 kHz."""
        if self.voice is None:
            raise RuntimeError("call set_voice() first")
        if language not in T.LANGUAGES:
            raise ValueError("language must be 'de' or 'en'")
        ids, lang = T.letters(sentence, language)
        if not ids:
            return np.zeros(0, np.float32)
        speed = min(1.6, max(0.7, float(speed)))
        e, v = self.engine, self.voice
        h, d = e.durations(ids, lang, v, speed, self.uniform_rate)
        for attempt in range(2 if self.retry_broken else 1):
            z = e.generate(h, d, v, np.random.default_rng(seed + attempt))
            full, lead = e.decode(z, v)
            if attempt == 0 and self.retry_broken and dsp.broken(full[lead:], z[0]):
                continue
            break
        wav = self._shelf(full)[lead:]                   # filter over the lead-in so the shelf has settled
        return dsp.limit_peak(wav, self._peak).astype(np.float32)

    def say(self, text: str, language: str = "de", speed: float = 1.0, seed: int = 0, normalize: bool = True,
            path: str | os.PathLike | None = None) -> Speech:
        """Speak a text of any length (split into sentences). Returns Speech(audio, sample_rate, seed)."""
        sentences, _ = T.split_sentences(self._prepare(text, language, normalize), final=True)
        gap = np.zeros(int(self.pause * SAMPLE_RATE), np.float32)
        parts = []
        for i, s in enumerate(sentences):
            if i:
                parts.append(gap)
            parts.append(self.sentence(s, language, speed, seed))
        speech = Speech(np.concatenate(parts) if parts else np.zeros(0, np.float32), SAMPLE_RATE, seed)
        if path is not None:
            speech.save(path)
        return speech

    def stream(self, pieces: Iterable[str] | str, language: str = "de", speed: float = 1.0, seed: int = 0,
               normalize: bool = True) -> Iterator[np.ndarray]:
        """Speak text that is still arriving (e.g. LLM tokens). Yields one float32 chunk per sentence, as soon as the
        sentence is complete; every chunk after the first starts with the pause between sentences."""
        if isinstance(pieces, str):
            pieces = [pieces]
        german = normalize and language == "de"
        abk = _german_frontend().StreamNormalizer() if german else None
        guard = _TagGuard()
        buffer = ""
        first = True
        gap = np.zeros(int(self.pause * SAMPLE_RATE), np.float32)

        def join(buf: str, piece: str) -> str:
            # raw tokens carry their own spaces; text released by the frontend is whole sentences
            if abk is None or not buf or not piece or buf[-1:].isspace() or piece[:1].isspace():
                return buf + piece
            return buf + " " + piece

        def speak(sentences):
            nonlocal first
            for s in sentences:
                wav = self.sentence(s, language, speed, seed)
                if len(wav):
                    yield wav if first else np.concatenate([gap, wav])
                    first = False

        for piece in pieces:
            piece = guard.push(piece)
            if abk is not None:
                piece = abk.push(piece)
            if piece:
                buffer = join(buffer, piece)
                done, buffer = T.split_sentences(buffer, final=False)
                yield from speak(done)
        tail = guard.flush()
        if abk is not None:
            tail = abk.flush(tail)
        done, _ = T.split_sentences(join(buffer, tail), final=True)
        yield from speak(done)
