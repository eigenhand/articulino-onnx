"""Equality of every ONNX graph (and of the whole chain) against tests/vectors, for one precision.

  python tools/check.py --precision fp32 [--json out.json]

Per graph the inputs come from the vectors (so errors do not add up); 'chain' runs the full engine from the voice
waveform and the letters with the vectors' noise and compares the final audio (before the tone shelf).
Metrics: rel = max|a - ref| / max|ref| (SPEC: < 1e-3 for float32), snr = 10 log10(sum ref^2 / sum (a - ref)^2) in dB.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from articulino_onnx.engine import Engine, Voice  # noqa: E402

VEC = ROOT / "tests" / "vectors"


def metrics(a, ref):
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    if a.shape != ref.shape:
        return {"rel": float("inf"), "snr": float("-inf"), "shape": [list(a.shape), list(ref.shape)]}
    err = a - ref
    rel = float(np.abs(err).max() / max(np.abs(ref).max(), 1e-12))
    snr = float(10 * np.log10((ref ** 2).sum() / max((err ** 2).sum(), 1e-30)))
    return {"rel": rel, "snr": snr}


_FB = None


def logmel_db(x):
    """80-band log-mel (n_fft 1024, hop 256, HTK) in dB, for spectral comparisons that ignore phase."""
    global _FB
    n, hop = 1024, 256
    if _FB is None:
        mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
        pts = 700 * (10 ** (np.linspace(mel(0), mel(12000), 82) / 2595) - 1)
        f = np.linspace(0, 12000, n // 2 + 1)
        _FB = np.maximum(0, np.minimum((f[None] - pts[:-2, None]) / (pts[1:-1, None] - pts[:-2, None]),
                                       (pts[2:, None] - f[None]) / (pts[2:, None] - pts[1:-1, None])))
    k = (len(x) - n) // hop + 1
    idx = np.arange(n)[None] + hop * np.arange(k)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1))
    return 20 * np.log10(np.maximum(spec @ _FB.T, 1e-5))


def mel_distance(a, ref):
    """Mean absolute log-mel difference in dB (0 = same spectrum)."""
    return float(np.abs(logmel_db(np.asarray(a, np.float64)) - logmel_db(np.asarray(ref, np.float64))).mean())


def worst(ms):
    return {"rel": max(m["rel"] for m in ms), "snr": min(m["snr"] for m in ms)}


def check(engine: Engine, name: str) -> dict:
    v = dict(np.load(VEC / f"{name}.npz"))
    info = json.loads((VEC / f"{name}.json").read_text())
    P = info["prompt_frames"]
    r = {}
    run = engine._run
    lat, logmel = run("encoder", {"waveform": v["voice_wav"][None]})
    r["encoder.log_mel"] = metrics(logmel[0], v["voice_logmel"][:, : logmel.shape[2]])
    r["encoder.latents"] = metrics(lat[0], v["voice_ref"])
    tok, vec, cache = run("voice", {"latents": v["voice_ref"][None]})
    r["voice.tokens"] = metrics(tok[0], v["voice_tokens"])
    r["voice.vector"] = metrics(vec[0], v["voice_vector"])
    r["voice.cache"] = worst([metrics(cache[i, j, 0], v[f"cache_{'kv'[j]}{i}"]) for i in range(8) for j in range(2)])
    n = len(v["text_ids"])
    lang = np.full((1, n), 0 if info["language"] == "de" else 1, np.int64)
    h, logd = run("text", {"text_ids": v["text_ids"][None], "language": lang, "text_mask": v["text_mask"][None],
                           "voice_vector": v["voice_vector"][None]})
    r["text.features"] = metrics(h[0], v["text_h"])
    r["text.log_durations"] = metrics(logd[0], v["duration_log"])
    cache_ref = np.stack([np.stack([v[f"cache_k{i}"], v[f"cache_v{i}"]]) for i in range(8)])[:, :, None]
    steps = []
    for k, t in enumerate(info["timesteps"]):
        vel = run("dit_step", {"x": v[f"step{k}_in"][None], "t": np.array([t], np.float32),
                               "frame_cond": v["frame_cond"][None, P:], "voice_tokens": v["voice_tokens"][None],
                               "voice_vector": v["voice_vector"][None], "voice_cache": cache_ref})[0]
        steps.append(metrics(vel[0], v[f"step{k}_velocity"]))
    r["dit_step.velocity"] = worst(steps)
    wav = run("decoder", {"latents": v["latents_norm"][None, P - info["context_frames"]:]})[0]
    r["decoder.waveform"] = metrics(wav[0], v["decoder_out"])
    r["decoder.waveform"]["mel_db"] = mel_distance(wav[0], v["decoder_out"])

    # whole chain from waveform + letters, vectors' noise, no rate equalisation (as in make_vectors.py)
    voice = engine.voice(engine.encode(v["voice_wav"]), rate=False)
    ids = [int(i) for i in v["text_ids_raw"]]
    hh, d = engine.durations(ids, [int(lang[0, 0])] * len(ids), voice, 1.0, uniform_rate=False)
    r["chain.durations"] = metrics(d, v["durations"])
    noise = [v[f"noise_{k}"][None] for k in range(4)]
    frames = int(np.ceil(d.sum(dtype=np.float32) / 4))
    if frames != info["frames_total"]:
        r["chain.audio"] = {"rel": float("inf"), "snr": float("-inf"), "frames": [frames, info["frames_total"]]}
        return r
    z = engine.generate(hh, d, voice, None, noise=noise)
    r["chain.latents"] = metrics(z[0], v["step3_x1"])
    full, lead = engine.decode(z, voice)
    r["chain.audio"] = metrics(full[lead:], v["audio"])
    r["chain.audio"]["mel_db"] = mel_distance(full[lead:], v["audio"])
    # baseline: same chain with other noise (what a different seed changes), median over three seeds
    other = []
    for seed in (7, 8, 9):
        full2, _ = engine.decode(engine.generate(hh, d, voice, np.random.default_rng(seed)), voice)
        other.append(mel_distance(full2[lead:], v["audio"]))
    r["chain.audio"]["mel_db_other_seed"] = float(np.median(other))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--build", default=str(ROOT / "build"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    eng = Engine(a.build, a.precision, threads=a.threads)
    out = {}
    for name in ("de_sentence", "en_sentence"):
        out[name] = check(eng, name)
    keys = list(out["de_sentence"])
    print(f"{a.precision}: {'graph.output':24s} {'rel de':>10s} {'rel en':>10s} {'SNR de':>8s} {'SNR en':>8s}"
          f"  log-mel dB de/en (other seed)")
    for k in keys:
        d, e = out["de_sentence"][k], out["en_sentence"][k]
        extra = ""
        if "mel_db" in d:
            extra = f"  {d['mel_db']:.2f} / {e['mel_db']:.2f}"
            if "mel_db_other_seed" in d:
                extra += f"  ({d['mel_db_other_seed']:.2f} / {e['mel_db_other_seed']:.2f})"
        print(f"   {k:24s} {d['rel']:10.2e} {e['rel']:10.2e} {d['snr']:8.1f} {e['snr']:8.1f}{extra}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
