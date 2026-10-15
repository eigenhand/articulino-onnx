"""Listening check: the four sample sentences of the model card, spoken with say() (seed 0, voice radiov), recognised by
a speech recognizer with an OpenAI-style /v1/audio/transcriptions endpoint (we used Parakeet-TDT 0.6B v3), with
0.4 s of silence in front. Writes the WAVs to build/listen/<precision>/.

  python tools/listen.py --precision fp32 --asr http://127.0.0.1:8084
Character error rate after running both texts through the German frontend and the letter normalisation.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from articulino_onnx import TTS  # noqa: E402
from articulino_onnx import text as T  # noqa: E402
from articulino_onnx.engine import Engine  # noqa: E402
from articulino_onnx.tts import _german_frontend  # noqa: E402

VOICE = str(ROOT / "tests" / "data" / "radiov.wav")
CARD = [("de-1", "de", "Guten Morgen! Dein Termin beim Zahnarzt ist heute um 15:30 Uhr in der Königstraße 12."),
        ("de-2", "de", "Die Stromrechnung über 87,40 € ist am 15.11. fällig; das PDF liegt in deinem Postfach."),
        ("de-3", "de", "Es gilt E = mc², und die Lichtgeschwindigkeit beträgt rund 300.000 km/s."),
        ("en-1", "en", "The delivery arrives tomorrow between ten and twelve.")]


def norm(s: str, lang: str) -> str:
    if lang == "de":
        s = _german_frontend().normalize(s)
    return "".join(c for c in T.normalize(s) if c.isalnum() or c == " ")


def cer(hyp: str, ref: str) -> float:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d.copy(), i
        for j, h in enumerate(hyp, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r != h))
    return float(d[-1] / max(len(ref), 1))


def transcribe(url: str, path: Path, lang: str) -> str:
    out = subprocess.run(["curl", "-s", f"{url.rstrip('/')}/v1/audio/transcriptions", "-F", f"file=@{path}",
                          "-F", f"language={lang}"], capture_output=True, text=True, timeout=120).stdout
    try:
        return json.loads(out).get("text", "")
    except json.JSONDecodeError:
        return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--build", default=str(ROOT / "build"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asr", required=True, help="base URL of the speech recognizer")
    a = ap.parse_args()
    out = Path(a.build) / "listen" / a.precision
    out.mkdir(parents=True, exist_ok=True)
    tts = TTS(Engine(a.build, a.precision, threads=8))
    tts.set_voice(VOICE)
    res = []
    for name, lang, txt in CARD:
        sp = tts.say(txt, language=lang, seed=a.seed, path=out / f"{name}.wav")
        probe = out / f"{name}.asr.wav"
        sf.write(probe, np.concatenate([np.zeros(int(0.4 * 24000), np.float32), sp.audio]), 24000)
        hyp = transcribe(a.asr, probe, lang)
        c = cer(norm(hyp, lang), norm(txt, lang))
        res.append({"name": name, "seconds": round(sp.duration, 2), "text": txt, "heard": hyp, "cer": round(c, 4)})
        print(f"{a.precision} {name} {sp.duration:5.2f} s  CER {c:.3f} | {hyp}", flush=True)
    (out / "asr.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{a.precision} mean CER {np.mean([r['cer'] for r in res]):.3f}")


if __name__ == "__main__":
    main()
