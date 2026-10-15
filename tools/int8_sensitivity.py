"""Which MatMul/Gemm nodes can be quantised to int8 without hurting? Quantises one node at a time and measures the
noise it adds to the graph outputs (de vector inputs), relative to the float32 graph.

  python tools/int8_sensitivity.py <graph> <scratch dir>
Writes <scratch>/<graph>.json: per node the activation rows (dynamic or fixed), weight shape and output SNR in dB."""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize import fixed_matrices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VEC = ROOT / "tests" / "vectors"


def feeds_for(graph, v):
    P = 125
    cache = np.stack([np.stack([v[f"cache_k{i}"], v[f"cache_v{i}"]]) for i in range(8)])[:, :, None]
    return {
        "encoder": {"waveform": v["voice_wav"][None]},
        "voice": {"latents": v["voice_ref"][None]},
        "text": {"text_ids": v["text_ids"][None], "language": np.zeros((1, len(v["text_ids"])), np.int64),
                 "text_mask": v["text_mask"][None], "voice_vector": v["voice_vector"][None]},
        "dit_step": {"x": v["step1_in"][None], "t": np.array([0.25], np.float32), "frame_cond": v["frame_cond"][None, P:],
                     "voice_tokens": v["voice_tokens"][None], "voice_vector": v["voice_vector"][None], "voice_cache": cache},
        "decoder": {"latents": v["latents_norm"][None, P - 12:]},
    }[graph]


def run(path, feeds):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]).run(None, feeds)


def snr(a, ref):
    return float(10 * np.log10((ref.astype(np.float64) ** 2).sum() / max(((a - ref).astype(np.float64) ** 2).sum(), 1e-30)))


def main():
    graph, scratch = sys.argv[1], Path(sys.argv[2])
    src = ROOT / "build" / "fp32" / f"{graph}.onnx"
    mdl = onnx.load(str(src))
    del mdl.graph.value_info[:]
    clean = scratch / f"{graph}.clean.onnx"
    onnx.save(mdl, str(clean))
    inf = onnx.shape_inference.infer_shapes(mdl).graph
    shp = {v.name: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in list(inf.value_info) + list(inf.input)}
    inits = {i.name: list(i.dims) for i in mdl.graph.initializer}
    fixed = fixed_matrices(mdl.graph)
    cand = [n for n in mdl.graph.node if n.op_type in ("MatMul", "Gemm") and n.name not in fixed
            and len(n.input) > 1 and n.input[1] in inits]
    feeds = feeds_for(graph, dict(np.load(VEC / "de_sentence.npz")))
    ref = run(clean, feeds)
    out = []
    for i, n in enumerate(cand):
        dst = scratch / f"{graph}.q.onnx"
        quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                         nodes_to_quantize=[n.name], op_types_to_quantize=["MatMul", "Gemm"],
                         extra_options={"MatMulConstBOnly": True})
        res = run(dst, feeds)
        rows = shp.get(n.input[0], [])
        out.append({"node": n.name, "rows": rows[-2] if len(rows) >= 2 else None, "weight": inits[n.input[1]],
                    "snr": [snr(a, r) for a, r in zip(res, ref) if a.dtype == np.float32]})
        print(i, n.name, out[-1]["rows"], out[-1]["weight"], [round(x, 1) for x in out[-1]["snr"]], flush=True)
    (scratch / f"{graph}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
