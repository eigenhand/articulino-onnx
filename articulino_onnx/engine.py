"""The five ONNX graphs and the host-side math between them (NumPy only).

Per voice:     waveform -> encoder -> latents -> voice graph -> tokens, vector, cache (+ base speaking rate)
Per sentence:  letters -> text graph -> durations -> per-frame condition -> 4 x dit_step -> decoder -> waveform
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import text as T

GRAPHS = ("encoder", "voice", "text", "dit_step", "decoder")

# Neutral sentences the original service uses to measure a voice's own speaking rate (letters per second at speed 1).
CALIBRATION = ["Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
               "Ich habe die Unterlagen geprüft und fasse das Wichtigste kurz zusammen.",
               "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
               "Soll ich die Nachricht jetzt abschicken oder noch einmal vorlesen?",
               "Das Training läuft seit gestern Abend ohne Unterbrechung weiter.",
               "Draußen sind es zwölf Grad, und am Abend soll es regnen."]


_POOL: bool | None = None         # True once the process-wide ONNX Runtime thread pool exists


def _global_pool(threads: int | None) -> bool:
    """One intra-op thread pool for all five sessions. With a pool per session, the idle sessions' threads keep
    spinning while the next graph runs and the pipeline gets ~10x slower on CPU. The pool size is fixed for the
    process by the first Engine; later `threads` values are ignored."""
    global _POOL
    if _POOL is None:
        try:
            from onnxruntime.capi import _pybind_state as C
            C.set_global_thread_pool_sizes(int(threads or 0), 1)
            _POOL = True
        except Exception as exc:  # noqa: BLE001
            _POOL = "already been created" in str(exc)
    return _POOL


@dataclass
class Voice:
    latents: np.ndarray         # (1, R, 96) normalised voice latents
    tokens: np.ndarray          # (1, 32, 768)
    vector: np.ndarray          # (1, 768)
    cache: np.ndarray           # (layers, 2, 1, heads, P, head_dim) keys/values of the prefix
    rate: float = 0.0           # letters per second the duration model gives this voice at speed 1

    @property
    def prefix(self) -> np.ndarray:
        return self.latents[:, : self.cache.shape[4]]


def letter_rate(d: np.ndarray, ids: np.ndarray, start: int = 0, target: float | None = None) -> float:
    """abtasten.buchstaben_rate for one sequence: speaking rate in letters per second over letters/digits from `start`
    (pauses at spaces and punctuation do not count; single overlong letters are capped at 3x the median).
    target: rescale the letter durations in `d` (in place) to exactly that rate."""
    m = np.asarray(T.IS_LETTER)[ids].copy()
    m[:start] = False
    if not m.any():
        return 0.0
    x = np.maximum(d[m], 0.5)
    x = np.minimum(x, 3 * np.median(x))
    rate = float(m.sum() / (x.sum() / 100))
    if target:
        x = x * (rate / target)
        rate = target
        d[m] = x
    return rate


def upsample(h: np.ndarray, d: np.ndarray, first: int, frames: int, units: int = 4) -> np.ndarray:
    """modell.hochtasten for frames first .. frames-1: weight of letter l on frame j = overlap of
    [start_l, end_l) with [4j, 4j+4), divided by 4. h (N, C), d (N,) -> (frames - first, C).

    Each frame overlaps only a few letters, so this gathers those instead of a dense (frames x N) matrix product:
    a BLAS call here leaves NumPy's BLAS threads spinning on the cores ONNX Runtime needs next (first DiT step 4x
    slower)."""
    end = np.cumsum(d, dtype=np.float32)
    start = end - d
    lo = np.arange(first, frames, dtype=np.float32) * units
    hi = lo + units
    a = np.searchsorted(end, lo, side="right")          # first letter ending after the frame start
    b = np.searchsorted(start, hi, side="left")         # letters starting before the frame end
    k = max(1, int((b - a).max()))
    idx = a[:, None] + np.arange(k)[None]
    valid = idx < b[:, None]
    idx = np.minimum(idx, len(d) - 1)
    w = np.clip(np.minimum(end[idx], hi[:, None]) - np.maximum(start[idx], lo[:, None]), 0, None) * valid
    return (h[idx] * (w / units).astype(np.float32)[..., None]).sum(1, dtype=np.float32)


class Engine:
    def __init__(self, folder: str | Path, precision: str = "fp32", providers=None, threads: int | None = None,
                 session_options: ort.SessionOptions | None = None):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        gdir = folder / precision if (folder / precision).is_dir() else folder
        providers = providers or ["CPUExecutionProvider"]
        self.precision = precision
        self.sessions = {}
        for g in GRAPHS:
            path = str(gdir / self.config["graphs"][g])
            if session_options is not None:
                self.sessions[g] = ort.InferenceSession(path, session_options, providers=providers)
                continue
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if _global_pool(threads):
                so.use_per_session_threads = False
            else:                                     # own pool per session, but without spinning
                so.inter_op_num_threads = 1
                so.intra_op_num_threads = int(threads or 0)
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.sessions[g] = ort.InferenceSession(path, so, providers=providers)
        c = self.config
        self.prefix_frames = int(c["prefix_frames"])
        self.lead_in = int(c["decoder_lead_in_frames"])
        self.units = int(c["duration_units_per_frame"])
        self.hop = int(c["samples_per_frame"])
        self.times = [float(t) for t in c["times"]]
        self.max_units = float(c["max_duration_units"])
        self.max_samples = int(c["max_voice_seconds"] * c["sample_rate"]) // self.hop * self.hop

    def _run(self, graph: str, feeds: dict, outputs=None):
        return self.sessions[graph].run(outputs, feeds)

    # ------------------------------------------------------------------ voice
    def encode(self, wav: np.ndarray) -> np.ndarray:
        """24 kHz mono float32 -> normalised latents (1, R, 96); cut to whole frames and at most 8 s."""
        wav = np.asarray(wav, np.float32)
        wav = wav[: min(len(wav), self.max_samples) // self.hop * self.hop]
        if len(wav) < 4 * self.hop:
            raise ValueError("voice prompt too short (need at least 3 s of speech)")
        return self._run("encoder", {"waveform": wav[None]}, ["latents"])[0]

    def voice(self, latents: np.ndarray, rate: bool = True) -> Voice:
        tokens, vector, cache = self._run("voice", {"latents": latents})
        v = Voice(latents, tokens, vector, cache)
        if rate:
            v.rate = self.base_rate(v)
        return v

    def base_rate(self, v: Voice) -> float:
        """server.grundrate: median speaking rate of the calibration sentences (no prompt placeholder, speed 1)."""
        rates = []
        for s in CALIBRATION:
            ids = np.array([[T.ID[c] for c in T.normalize(s)]], np.int64)
            _, logd = self._run("text", {"text_ids": ids, "language": np.zeros_like(ids),
                                         "text_mask": np.ones(ids.shape, bool), "voice_vector": v.vector})
            rates.append(letter_rate(np.exp(logd[0]).astype(np.float32), ids[0]))
        return float(np.median(rates))

    # ------------------------------------------------------------------ sentence
    def durations(self, ids: list[int], lang: list[int], v: Voice, speed: float = 1.0, uniform_rate: bool = True):
        """-> (text features (N+1, 512), durations (N+1,) in 10 ms units incl. the prompt placeholder)."""
        n = len(ids)
        t_ids = np.array([[0] + ids], np.int64)
        t_lang = np.array([[lang[0]] + lang], np.int64)
        mask = np.array([[False] + [True] * n])
        h, logd = self._run("text", {"text_ids": t_ids, "language": t_lang, "text_mask": mask, "voice_vector": v.vector})
        d = np.minimum(np.exp(logd[0]) * mask[0], self.max_units).astype(np.float32) / np.float32(speed)
        if uniform_rate and v.rate:
            letter_rate(d, t_ids[0], start=1, target=v.rate * speed)
        d[0] = self.units * v.cache.shape[4]
        return h[0], d.astype(np.float32)

    def generate(self, h: np.ndarray, d: np.ndarray, v: Voice, rng: np.random.Generator, noise=None):
        """4 student steps on the new frames -> normalised latents of the new frames (1, L - P, 96).
        noise: optional list of 4 arrays (1, L - P, 96) instead of drawing from rng (for reference tests)."""
        p = v.cache.shape[4]
        frames = int(math.ceil(float(d.sum(dtype=np.float32)) / self.units))
        cond = upsample(h, d, p, frames, self.units)[None]
        shape = (1, frames - p, 96)
        draw = (lambda k: noise[k]) if noise is not None else (lambda k: rng.standard_normal(shape).astype(np.float32))
        x = draw(0)
        x1 = x
        for k, t in enumerate(self.times):
            vel = self._run("dit_step", {"x": x, "t": np.array([t], np.float32), "frame_cond": cond,
                                         "voice_tokens": v.tokens, "voice_vector": v.vector, "voice_cache": v.cache})[0]
            x1 = x + np.float32(1 - t) * vel
            if k + 1 < len(self.times):
                t2 = self.times[k + 1]
                x = np.float32(1 - t2) * draw(k + 1) + np.float32(t2) * x1
        return x1.astype(np.float32)

    def decode(self, z_new: np.ndarray, v: Voice) -> tuple[np.ndarray, int]:
        """Decoder with 12 prompt frames of lead-in -> (waveform incl. lead-in, lead-in samples)."""
        pre = v.prefix
        lead = min(pre.shape[1], self.lead_in)
        z = np.concatenate([pre[:, pre.shape[1] - lead:], z_new], 1).astype(np.float32)
        wav = self._run("decoder", {"latents": z})[0][0]
        return wav, lead * self.hop
