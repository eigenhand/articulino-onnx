"""Export the five Articulino graphs to ONNX (float32) into build/fp32/, plus build/config.json.

  python tools/export.py [--out build/fp32]

Graphs (opset 18, all sequence axes dynamic, batch dynamic):
  encoder.onnx   waveform -> voice latents (+ log-mel for tests)
  voice.onnx     voice latents -> voice tokens, voice vector, voice cache (keys/values of the 5 s prefix)
  text.onnx      letters + language + mask + voice vector -> text features, log durations
  dit_step.onnx  one of the 4 student steps on the new frames, with the voice cache as input
  decoder.onnx   latents -> 24 kHz waveform (iSTFT inside, as MatMul + overlap-add)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parts as P  # noqa: E402

OPSET = 18
IR_VERSION = 8          # the lowest IR version for opset 18 (ONNX 1.13): loads in onnxruntime >= 1.14


def pin_ir(path: Path):
    mdl = onnx.load(str(path))
    if mdl.ir_version != IR_VERSION:
        mdl.ir_version = IR_VERSION
        onnx.save(mdl, str(path))
torch.set_grad_enabled(False)


def rename_dims(path: Path, dims: dict[str, dict[int, str]]):
    """Give the dynamic axes of inputs/outputs readable names (the exporter calls them s0, s1, ...)."""
    mdl = onnx.load(str(path))
    for vi in list(mdl.graph.input) + list(mdl.graph.output):
        for ax, name in dims.get(vi.name, {}).items():
            d = vi.type.tensor_type.shape.dim[ax]
            if not d.HasField("dim_value"):
                d.dim_param = name
    onnx.save(mdl, str(path))


def export(module, args, path: Path, inputs, outputs, dyn, names):
    module.eval()
    prog = torch.onnx.export(module, args, dynamo=True, opset_version=OPSET, input_names=inputs,
                             output_names=outputs, dynamic_shapes=dyn, external_data=False, optimize=True,
                             verbose=False)
    prog.save(str(path), external_data=False)
    rename_dims(path, names)
    pin_ir(path)
    onnx.checker.check_model(str(path), full_check=True)
    mdl = onnx.load(str(path))
    ops = sorted({n.op_type for n in mdl.graph.node})
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, {len(mdl.graph.node)} nodes, opset "
          f"{[o.version for o in mdl.opset_import if o.domain in ('', 'ai.onnx')]}, ops {ops}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "build" / "fp32"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m, ae, vm, mean, std, times = P.load_original()
    Dim = torch.export.Dim
    B = Dim("batch", min=1, max=64)

    # encoder: T = 960 * R
    r = Dim("frames", min=4, max=2000)
    wav = torch.randn(2, 960 * 126) * 0.1           # batch 2: with batch 1 the exporter fixes the batch axis to 1
    export(P.EncoderGraph(ae, mean, std), (wav,), out / "encoder.onnx", ["waveform"], ["latents", "log_mel"],
           {"wav": {0: B, 1: 960 * r}},
           {"waveform": {0: "batch", 1: "samples"}, "latents": {0: "batch", 1: "frames"},
            "log_mel": {0: "batch", 2: "mel_frames"}})

    # voice
    lat = torch.randn(2, 200, 96)
    rv = Dim("voice_frames", min=8, max=400)
    export(P.VoiceGraph(m), (lat,), out / "voice.onnx", ["latents"], ["voice_tokens", "voice_vector", "voice_cache"],
           {"lat": {0: B, 1: rv}},
           {"latents": {0: "batch", 1: "voice_frames"}, "voice_tokens": {0: "batch"}, "voice_vector": {0: "batch"},
            "voice_cache": {2: "batch", 4: "prefix_frames"}})

    # text
    n = Dim("letters", min=2, max=1024)
    ids = torch.randint(1, 53, (1, 60))
    ids[:, 0] = 0
    lang = torch.zeros(1, 60, dtype=torch.int64)
    mask = torch.ones(1, 60, dtype=torch.bool)
    mask[:, 0] = False
    vec = torch.randn(1, 768)
    export(P.TextGraph(m), (ids, lang, mask, vec), out / "text.onnx",
           ["text_ids", "language", "text_mask", "voice_vector"], ["text_features", "log_durations"],
           {"ids": {0: B, 1: n}, "language": {0: B, 1: n}, "mask": {0: B, 1: n}, "vec": {0: B}},
           {"text_ids": {0: "batch", 1: "letters"}, "language": {0: "batch", 1: "letters"},
            "text_mask": {0: "batch", 1: "letters"}, "voice_vector": {0: "batch"},
            "text_features": {0: "batch", 1: "letters"}, "log_durations": {0: "batch", 1: "letters"}})

    # DiT step
    L = Dim("new_frames", min=1, max=2000)
    Pd = Dim("prefix_frames", min=1, max=400)
    x = torch.randn(1, 117, 96)
    t = torch.tensor([0.25])
    cond = torch.randn(1, 117, 512)
    tok = torch.randn(1, 32, 768)
    cache = torch.randn(8, 2, 1, 6, 125, 64)
    export(P.DiTStepGraph(m), (x, t, cond, tok, vec, cache), out / "dit_step.onnx",
           ["x", "t", "frame_cond", "voice_tokens", "voice_vector", "voice_cache"], ["velocity"],
           {"x": {0: B, 1: L}, "t": {0: B}, "cond": {0: B, 1: L}, "tokens": {0: B}, "vec": {0: B},
            "cache": {2: B, 4: Pd}},
           {"x": {0: "batch", 1: "new_frames"}, "t": {0: "batch"}, "frame_cond": {0: "batch", 1: "new_frames"},
            "voice_tokens": {0: "batch"}, "voice_vector": {0: "batch"}, "voice_cache": {2: "batch", 4: "prefix_frames"},
            "velocity": {0: "batch", 1: "new_frames"}})

    # decoder
    Ld = Dim("frames", min=2, max=4000)
    z = torch.randn(1, 129, 96)
    export(P.DecoderGraph(ae, mean, std), (z,), out / "decoder.onnx", ["latents"], ["waveform"],
           {"lat": {0: B, 1: Ld}},
           {"latents": {0: "batch", 1: "frames"}, "waveform": {0: "batch", 1: "samples"}})

    cfg = {
        "model": "Articulino-DE-EN-Realtime-Stream-TTS",
        "format": "onnx", "opset": OPSET,
        "sample_rate": 24000, "latent_dim": 96, "latent_rate": 25, "samples_per_frame": 960,
        "duration_units_per_frame": 4, "prefix_frames": P.PREFIX, "decoder_lead_in_frames": P.LEAD_IN,
        "max_voice_seconds": 8.0, "max_duration_units": 400.0, "times": list(times),
        "charset": " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789", "languages": {"de": 0, "en": 1},
        "layers": len(m.bloecke), "heads": m.koepfe, "head_dim": m.dim // m.koepfe,
        "postprocess": {"low_shelf_db": 2.0, "low_shelf_hz": 180.0, "shelf_slope": 0.7, "peak": 0.95},
        "graphs": {k: f"{k}.onnx" for k in ("encoder", "voice", "text", "dit_step", "decoder")},
    }
    (out.parent / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print("config.json written", flush=True)


if __name__ == "__main__":
    main()
