"""Check the torch export wrappers (parts.py) against tests/vectors before exporting anything.

  python tools/check_wrappers.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parts as P  # noqa: E402

VEC = Path(__file__).resolve().parents[1] / "tests" / "vectors"
torch.set_grad_enabled(False)


def rel(a, ref):
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-12))


def main():
    m, ae, vm, mean, std, times = P.load_original()
    enc, voice, text, step, dec = (P.EncoderGraph(ae, mean, std), P.VoiceGraph(m), P.TextGraph(m), P.DiTStepGraph(m),
                                   P.DecoderGraph(ae, mean, std))
    for name in ("de_sentence", "en_sentence"):
        v = dict(np.load(VEC / f"{name}.npz"))
        T = lambda k: torch.from_numpy(v[k])[None]  # noqa: E731
        lat, logmel = enc(T("voice_wav"))
        print(name, "logmel", rel(logmel[0], v["voice_logmel"][:, :logmel.shape[2]]), "ref", rel(lat[0], v["voice_ref"]))
        tok, vec, cache = voice(T("voice_ref"))
        print("  tokens", rel(tok[0], v["voice_tokens"]), "vector", rel(vec[0], v["voice_vector"]),
              "cache", max(max(rel(cache[i, 0, 0], v[f"cache_k{i}"]), rel(cache[i, 1, 0], v[f"cache_v{i}"])) for i in range(8)))
        n = len(v["text_ids"])
        lang = torch.full((1, n), 0 if name.startswith("de") else 1, dtype=torch.int64)
        h, logd = text(T("text_ids"), lang, T("text_mask"), T("voice_vector"))
        print("  text_h", rel(h[0], v["text_h"]), "log_dur", rel(logd[0][1:], v["duration_log"][1:]))
        cache_ref = torch.stack([torch.stack([torch.from_numpy(v[f"cache_k{i}"]), torch.from_numpy(v[f"cache_v{i}"])])
                                 for i in range(8)])[:, :, None]
        cond = T("frame_cond")[:, P.PREFIX:]
        worst = 0.0
        for k, t in enumerate(times):
            vel = step(T(f"step{k}_in"), torch.tensor([t]), cond, T("voice_tokens"), T("voice_vector"), cache_ref)
            worst = max(worst, rel(vel[0], v[f"step{k}_velocity"]))
        print("  velocity (worst step)", worst)
        wav = dec(T("latents_norm")[:, P.PREFIX - P.LEAD_IN:])
        print("  decoder", rel(wav[0], v["decoder_out"]))


if __name__ == "__main__":
    main()
