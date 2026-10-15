// Runs every Articulino ONNX graph in onnxruntime-web (WebAssembly backend, the browser's CPU path) under Node and
// compares the outputs with the reference vectors.
//   cd build/webtest && node ../../tools/web/test.mjs <precision> [threads]
import * as ort from "onnxruntime-web";
import fs from "node:fs";
import path from "node:path";

const precision = process.argv[2] || "fp32";
ort.env.wasm.numThreads = Number(process.argv[3] || 1);
const data = path.resolve("data");
const man = JSON.parse(fs.readFileSync(path.join(data, "manifest.json")));
const typed = { float32: Float32Array, int64: BigInt64Array, bool: Uint8Array };

function load(spec) {
  const buf = fs.readFileSync(path.join(data, spec.file));
  const T = typed[spec.type];
  const arr = new T(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return new ort.Tensor(spec.type, spec.type === "bool" ? Uint8Array.from(arr) : arr, spec.dims);
}

for (const [graph, spec] of Object.entries(man)) {
  const file = path.resolve("..", precision, `${graph}.onnx`);
  const t0 = performance.now();
  let session;
  try {
    session = await ort.InferenceSession.create(fs.readFileSync(file), { executionProviders: ["wasm"] });
  } catch (e) {
    console.log(`${precision}/${graph}: LOAD FAILED ${e.message.slice(0, 300)}`);
    continue;
  }
  const feeds = {};
  for (const [k, s] of Object.entries(spec.inputs)) feeds[k] = load(s);
  const t1 = performance.now();
  let out;
  try {
    out = await session.run(feeds);
  } catch (e) {
    console.log(`${precision}/${graph}: RUN FAILED ${e.message.slice(0, 300)}`);
    continue;
  }
  const t2 = performance.now();
  const res = [];
  for (const [k, s] of Object.entries(spec.outputs)) {
    const ref = load(s).data, got = out[k].data;
    let maxErr = 0, maxRef = 0, e2 = 0, r2 = 0;
    for (let i = 0; i < ref.length; i++) {
      const d = got[i] - ref[i];
      maxErr = Math.max(maxErr, Math.abs(d)); maxRef = Math.max(maxRef, Math.abs(ref[i]));
      e2 += d * d; r2 += ref[i] * ref[i];
    }
    res.push(`${k} rel ${(maxErr / maxRef).toExponential(2)} SNR ${(10 * Math.log10(r2 / Math.max(e2, 1e-30))).toFixed(1)} dB`);
  }
  console.log(`${precision}/${graph}: load ${(t1 - t0).toFixed(0)} ms, run ${(t2 - t1).toFixed(0)} ms | ${res.join(" | ")}`);
}
