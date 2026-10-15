"""Real-time factor on the CPU, measured like the original evaluation script: the test voice with its voice cache computed
beforehand, then per sentence the whole chain (letters, text encoder, durations, 4 DiT steps, decoder, tone shelf),
seed 42+i, no rate equalisation, ten sentences after one warm-up sentence.

  python tools/rtf.py --precision fp32 --threads 12
  --engine torch   the same loop with the original PyTorch code (see tools/parts.py), float32
  --api            measure the public say() instead (German frontend, rate equalisation, broken-output check)

On a shared machine, each of --repeat passes waits until the cores it may use are nearly idle (< 20 % busy); per
sentence the fastest pass counts.
Prints one line: audio seconds per compute second, time to the first sentence, median per sentence.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOICE = str(ROOT / "tests" / "data" / "radiov.wav")
SENTENCES = str(ROOT / "tests" / "data" / "sentences10.txt")


def busy(cpus, dt=1.0):
    """Mean busy fraction of the given cores over dt seconds (from /proc/stat)."""
    def read():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                vals = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(vals) - vals[3] - vals[4], sum(vals))
        return out
    a = read()
    time.sleep(dt)
    b = read()
    return float(np.mean([(b[c][0] - a[c][0]) / max(b[c][1] - a[c][1], 1) for c in cpus]))


def wait_quiet(cpus, limit=0.20, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if busy(cpus) < limit:
            return True
        time.sleep(3)
    return False


def onnx_speaker(a):
    from articulino_onnx import TTS
    from articulino_onnx.engine import Engine
    tts = TTS(Engine(a.build, a.precision, threads=a.threads), uniform_rate=a.api, retry_broken=a.api)
    tts.set_voice(VOICE)
    if a.api:
        return lambda s, i: tts.say(s, seed=42 + i).audio
    return lambda s, i: tts.sentence(s, seed=42 + i)


def torch_speaker(a):
    """The original evaluation loop with the original modules."""
    import torch
    import soundfile as sf
    from scipy.signal import sosfilt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import parts as P
    import text as T
    from abtasten_schueler import erzeuge_schueler
    sys.path.insert(0, str(ROOT))
    from articulino_onnx import dsp
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    m, ae, vm, mean, std, _ = P.load_original()
    bass = np.array([dsp.shelf_coefficients(180, 2.0, False), dsp.shelf_coefficients(1000, 0.0, True)])
    y, _ = sf.read(VOICE, dtype="float32")
    y = y if y.ndim == 1 else y.mean(1)
    y = y[: min(len(y), 8 * 24000) // 960 * 960]
    ref = (ae.encoder(torch.from_numpy(y)[None])[0].transpose(1, 2) - mean) / std
    rm = torch.ones(ref.shape[:2], dtype=torch.bool)
    vor = ref[:, :125]
    tok, g = m.ref(ref, rm)
    cache = m.anfang_cache(vor, tok, g)

    def speak(s, i):
        with torch.inference_mode():
            t = torch.tensor([T.ids(s)])
            z = erzeuge_schueler(m, t, torch.zeros(1, dtype=torch.long), torch.ones_like(t, dtype=torch.bool), ref, rm,
                                 seed=42 + i, anfang=vor, cache=cache)[0]
            w = ae.decoder((z[:, 125 - 12:] * std + mean).transpose(1, 2).float())[0].numpy()[12 * 960:]
            return sosfilt(bass, w)
    return speak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnx", choices=["onnx", "torch"])
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--build", default=str(ROOT / "build"))
    ap.add_argument("--sentences", default=SENTENCES)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    sentences = [z.strip() for z in open(a.sentences, encoding="utf-8") if z.strip()]
    one = onnx_speaker(a) if a.engine == "onnx" else torch_speaker(a)
    cpus = sorted(os.sched_getaffinity(0))
    best = [float("inf")] * len(sentences)
    sec = 0.0
    quiet = True
    for _ in range(a.repeat):
        quiet &= wait_quiet(cpus)
        one("Ein kurzer Satz zum Aufwärmen.", 0)
        sec = 0.0
        for i, s in enumerate(sentences):
            t0 = time.perf_counter()
            w = one(s, i)
            best[i] = min(best[i], time.perf_counter() - t0)
            sec += len(w) / 24000
    name = f"ONNX {a.precision}{' say()' if a.api else ''}" if a.engine == "onnx" else "PyTorch fp32"
    print(f"{name} | CPUs {cpus[0]}-{cpus[-1]} ({len(cpus)}) | {a.threads} threads | {sec:.1f} s audio in {sum(best):.2f} s "
          f"= {sec / sum(best):.1f}x real time | first sentence {best[0]:.2f} s, median {np.median(best):.2f} s | "
          f"fastest of {a.repeat} per sentence{'' if quiet else ' (cores were not idle)'}", flush=True)


if __name__ == "__main__":
    main()
