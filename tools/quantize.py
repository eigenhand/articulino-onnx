"""Derive the float16 (GPU) and dynamic int8 (CPU) graphs from the float32 export.

  python tools/quantize.py [--build build] [--only fp16|int8]

float16: weights and activations in half precision, inputs/outputs stay float32. Kept in float32 because half
precision is too coarse there:
  * sin/cos tables (RoPE positions up to ~600, time features t*1000*f) and everything that feeds them,
  * the log-mel front end of the encoder (DFT, magnitude, mel, log),
  * the decoder head and iSTFT (phases reach +-74 rad; fp16 spacing there is 0.06 rad).
int8: dynamic quantisation (weights int8 per output channel, activations quantised per call) of the MatMul/Gemm
layers of text, dit_step and decoder, except output heads, input projections and the fixed DFT / iSTFT matrices
(see int8_nodes); convolutions, attention products and norms stay float32. encoder and voice run once per voice and
are sensitive (voice tokens), so the int8 folder carries their float16 versions (computed in float32 on CPUs).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

GRAPHS = ("encoder", "voice", "text", "dit_step", "decoder")
IR_VERSION = 8          # the lowest IR version for opset 18; loads in onnxruntime >= 1.14


def pin_ir(path: Path):
    mdl = onnx.load(str(path))
    if mdl.ir_version != IR_VERSION:
        mdl.ir_version = IR_VERSION
        onnx.save(mdl, str(path))


def _graph_maps(g):
    producer = {o: n for n in g.node for o in n.output}
    consumers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)
    return producer, consumers


def ancestors(g, nodes):
    producer, _ = _graph_maps(g)
    seen, stack = set(), list(nodes)
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        stack += [producer[i] for i in n.input if i in producer]
    return seen


def descendants(g, nodes):
    _, consumers = _graph_maps(g)
    seen, stack = set(), list(nodes)
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        for o in n.output:
            stack += consumers.get(o, [])
    return seen


def const_shapes(g):
    return {i.name: tuple(i.dims) for i in g.initializer}


def fixed_matrices(g):
    """Names of MatMul nodes with the constant DFT / mel / iSTFT matrices (not model weights)."""
    shapes = const_shapes(g)
    return {n.name for n in g.node if n.op_type == "MatMul" and any(shapes.get(i) in ((513, 100), (962, 960))
                                                                    for i in n.input)}


def fp32_islands(name, g):
    keep = set()
    trig = [n for n in g.node if n.op_type in ("Sin", "Cos")]
    if name == "decoder":
        shapes = const_shapes(g)
        head = [n for n in g.node if n.op_type in ("MatMul", "Gemm") and any(shapes.get(i) in ((512, 962), (962, 512))
                                                                             for i in n.input)]
        assert len(head) == 1, head
        keep |= descendants(g, head)
    else:
        keep |= ancestors(g, trig)
    if name == "encoder":
        logs = [n for n in g.node if n.op_type == "Log"]
        assert len(logs) == 1
        keep |= ancestors(g, logs)
    return keep


FLOAT, FLOAT16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16


def convert_fp16(mdl: onnx.ModelProto, keep: set[str]) -> onnx.ModelProto:
    """float32 -> float16 for every node not in `keep`; graph inputs/outputs stay float32.

    Small own converter (the onnxconverter-common / onnxruntime.transformers ones mistype graph outputs that are also
    used inside the graph, and Cast nodes): each float tensor gets a float16 and/or float32 version on demand, with a
    Cast inserted where a consumer needs the other one."""
    inferred = onnx.shape_inference.infer_shapes(mdl, strict_mode=True)
    g = inferred.graph
    etype = {v.name: v.type.tensor_type.elem_type for v in list(g.value_info) + list(g.input) + list(g.output)}
    inits = {i.name: i for i in g.initializer}
    for name, i in inits.items():
        etype[name] = i.data_type
    outputs = {o.name for o in g.output}
    have = {FLOAT: {}, FLOAT16: {}}                  # original tensor name -> name of its version in that type
    nodes, new_inits = [], []
    for i in g.input:
        if etype.get(i.name) == FLOAT:
            have[FLOAT][i.name] = i.name

    def get(name: str, want: int) -> str:
        if name in have[want]:
            return have[want][name]
        if name in inits:                             # float initializer: convert once
            arr = numpy_helper.to_array(inits[name])
            new = name + ("__fp16" if want == FLOAT16 else "__fp32")
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16 if want == FLOAT16 else np.float32), new))
        else:
            other = FLOAT if want == FLOAT16 else FLOAT16
            src = have[other][name]
            new = f"{name}__to{'16' if want == FLOAT16 else '32'}"
            nodes.append(onnx.helper.make_node("Cast", [src], [new], name=new + "_cast", to=want))
        have[want][name] = new
        return new

    for n in g.node:
        want = FLOAT if n.name in keep else FLOAT16
        ins = [get(i, want) if i and etype.get(i) == FLOAT else i for i in n.input]
        outs = []
        for o in n.output:
            if etype.get(o) == FLOAT:
                name = o + "__fp16" if (want == FLOAT16 and o in outputs) else o
                have[want][o] = name
                outs.append(name)
            else:
                outs.append(o)
        nn = onnx.helper.make_node(n.op_type, ins, outs, name=n.name, domain=n.domain)
        nn.attribute.extend(n.attribute)
        if want == FLOAT16:
            for a in nn.attribute:
                if n.op_type == "Cast" and a.name == "to" and a.i == FLOAT:
                    a.i = FLOAT16
                if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == FLOAT:
                    a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float16), a.t.name))
        nodes.append(nn)
    for o in g.output:
        if etype.get(o.name) == FLOAT and o.name not in (have[FLOAT].get(o.name),):
            nodes.append(onnx.helper.make_node("Cast", [have[FLOAT16][o.name]], [o.name], name=o.name + "_out_cast",
                                               to=FLOAT))
    keep_inits = [i for i in g.initializer if i.data_type != FLOAT]
    used = {i for n in nodes for i in n.input}
    keep_inits += [i for i in g.initializer if i.data_type == FLOAT and i.name in used]
    graph = onnx.helper.make_graph(nodes, g.name, list(mdl.graph.input), list(mdl.graph.output),
                                   keep_inits + [i for i in new_inits if i.name in used])
    out = onnx.helper.make_model(graph, opset_imports=list(mdl.opset_import), ir_version=mdl.ir_version,
                                 producer_name="articulino-onnx fp16")
    return onnx.shape_inference.infer_shapes(out)


def to_fp16(src: Path, dst: Path, name: str):
    mdl = onnx.load(str(src))
    for i, n in enumerate(mdl.graph.node):           # node names identify the float32 islands
        if not n.name:
            n.name = f"node_{i}"
    keep = fp32_islands(name, mdl.graph)
    out = convert_fp16(mdl, keep)
    onnx.save(out, str(dst))
    onnx.checker.check_model(str(dst), full_check=True)
    return len(keep)


def int8_nodes(mdl: onnx.ModelProto, ff_down: bool = True) -> list[str]:
    """MatMul/Gemm nodes to quantise (from a per-node sensitivity scan, tools/int8_sensitivity.py): all constant-weight
    products except output heads (-> 96 latents, -> 1 duration, decoder magnitude/phase head), projections of raw graph
    inputs and latents (cheap, and the most sensitive), and the fixed DFT/mel/iSTFT matrices. ff_down=False also keeps
    the feed-forward down projections (1536/2048 -> d) in float32, the most sensitive of the remaining layers."""
    g = mdl.graph
    inits = {i.name: list(i.dims) for i in g.initializer}
    inputs = {i.name for i in g.input}
    fixed = fixed_matrices(g)
    out = []
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm") or n.name in fixed or len(n.input) < 2 or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        k, m = (w[1], w[0]) if n.op_type == "Gemm" and any(a.name == "transB" and a.i for a in n.attribute) else (w[0], w[1])
        head = m <= 96 or (k, m) == (512, 962)
        down = k >= 1536
        if not head and n.input[0] not in inputs and k != 96 and (ff_down or not down):
            out.append(n.name)
    return out


def to_int8(src: Path, dst: Path, name: str, ff_down: bool = True):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    mdl = onnx.load(str(src))
    del mdl.graph.value_info[:]                      # stale shape annotations trip the quantiser's shape inference
    nodes = int8_nodes(mdl, ff_down)
    tmp = dst.with_suffix(".tmp.onnx")
    onnx.save(mdl, str(tmp))
    try:
        quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                         nodes_to_quantize=nodes, op_types_to_quantize=["MatMul", "Gemm"],
                         extra_options={"MatMulConstBOnly": True})
    finally:
        tmp.unlink(missing_ok=True)
    onnx.checker.check_model(str(dst))
    return len(nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(Path(__file__).resolve().parents[1] / "build"))
    ap.add_argument("--only")
    ap.add_argument("--stage-dir", help="output folder name for the int8 stage (default int8)")
    ap.add_argument("--keep-ff-down", action="store_true", help="int8: keep feed-forward down projections float32")
    a = ap.parse_args()
    b = Path(a.build)
    for stage in ("fp16", "int8"):
        if a.only and a.only != stage:
            continue
        sd = b / (a.stage_dir or stage if stage == "int8" else stage)
        sd.mkdir(exist_ok=True)
        for g in GRAPHS:
            src, dst = b / "fp32" / f"{g}.onnx", sd / f"{g}.onnx"
            if stage == "fp16" or g in ("encoder", "voice"):
                k, what = to_fp16(src, dst, g), "nodes kept float32 (float16 graph)"
            else:
                k, what = to_int8(src, dst, g, ff_down=not a.keep_ff_down), "MatMul/Gemm quantised"
            pin_ir(dst)
            onnx.checker.check_model(str(dst), full_check=True)
            print(f"{sd.name}/{g}.onnx: {dst.stat().st_size / 1e6:.1f} MB, {k} {what}", flush=True)


if __name__ == "__main__":
    main()
