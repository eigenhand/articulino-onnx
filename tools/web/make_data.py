"""Write the de_sentence test vectors as raw little-endian files + manifest.json for the onnxruntime-web test (test.mjs).

  python tools/web/make_data.py build/webtest/data
"""
import json
import sys
from pathlib import Path

import numpy as np

VEC = Path(__file__).resolve().parents[2] / "tests" / "vectors" / "de_sentence.npz"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    v = dict(np.load(VEC))
    P = 125
    cache = np.stack([np.stack([v[f"cache_k{i}"], v[f"cache_v{i}"]]) for i in range(8)])[:, :, None]
    n = len(v["text_ids"])
    cases = {
        "encoder": ({"waveform": v["voice_wav"][None]}, {"latents": v["voice_ref"][None]}),
        "voice": ({"latents": v["voice_ref"][None]},
                  {"voice_tokens": v["voice_tokens"][None], "voice_vector": v["voice_vector"][None], "voice_cache": cache}),
        "text": ({"text_ids": v["text_ids"][None], "language": np.zeros((1, n), np.int64),
                  "text_mask": v["text_mask"][None], "voice_vector": v["voice_vector"][None]},
                 {"text_features": v["text_h"][None], "log_durations": v["duration_log"][None]}),
        "dit_step": ({"x": v["step1_in"][None], "t": np.array([0.25], np.float32), "frame_cond": v["frame_cond"][None, P:],
                      "voice_tokens": v["voice_tokens"][None], "voice_vector": v["voice_vector"][None],
                      "voice_cache": cache}, {"velocity": v["step1_velocity"][None]}),
        "decoder": ({"latents": v["latents_norm"][None, P - 12:]}, {"waveform": v["decoder_out"][None]}),
    }
    man = {}
    for g, (ins, outs) in cases.items():
        man[g] = {"inputs": {}, "outputs": {}}
        for kind, arrs in (("inputs", ins), ("outputs", outs)):
            for k, a in arrs.items():
                a = np.ascontiguousarray(a)
                if a.dtype == bool:
                    a = a.astype(np.uint8)
                    t = "bool"
                else:
                    t = {"float32": "float32", "int64": "int64"}[str(a.dtype)]
                f = f"{g}.{kind}.{k}.bin"
                a.tofile(out / f)
                man[g][kind][k] = {"file": f, "type": t, "dims": list(a.shape)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
