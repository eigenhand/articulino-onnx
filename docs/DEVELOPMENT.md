# Development notes: ONNX

State of 15 October 2026. Runtime package in `articulino_onnx/`, build and test scripts in `tools/`, reference vectors
in `tests/vectors/`, test voice and the ten speed sentences in `tests/data/`. Build output goes to `build/` (not in
git): `fp32/ fp16/ int8/`, `config.json`, `repo/` (the assembled Hugging Face repository), `listen/`, `rtf.txt`.

The computation follows the inference specification in the PyTorch repository
([docs/SPEC.md](https://github.com/eigenhand/articulino/blob/main/docs/SPEC.md)).

## Rebuilding

The export starts from the training checkpoints and the original inference modules, which are not public (see
`tools/parts.py`: `ARTICULINO_ORIGINAL_CODE`, `ARTICULINO_CHECKPOINTS`). Tested with torch 2.14.1 (CPU), onnx 1.23.2,
onnxruntime 1.31.0, onnxscript 0.7.2; SciPy only for checks.

```bash
python tools/check_wrappers.py                  # the export wrappers in torch against the reference vectors
python tools/export.py                          # fp32 graphs into build/fp32
python tools/quantize.py                        # fp16 and int8
python tools/check.py --precision fp32          # (fp16, int8) every graph against the reference vectors
python tools/listen.py --precision fp32 --asr http://127.0.0.1:8084   # listening check through a speech recognizer
tools/bench.sh > build/rtf.txt                  # real-time factors
python tools/assemble.py                        # build/repo
```

## Design

- `tools/parts.py` wraps the original modules with purely algebraic rewrites: the AdaNorm linear layer once per
  sentence instead of per frame, the zero halves of the DiT input layer cut away, mask fill −1e4 instead of −1e9
  (float16-safe), log-mel as a convolution with a windowed DFT basis, inverse STFT as a matrix product plus four
  shifted additions divided by the window envelope.
- `tools/export.py`: five graphs, opset 18, IR 8 (runs on onnxruntime 1.17 and newer), dynamo exporter, every axis
  dynamic including batch, standard operators only. Cut by how often a part runs: once per voice (`encoder`,
  `voice`), once per sentence (`text`, `decoder`), four times per sentence (`dit_step`, with the voice cache as
  input). Host side in NumPy: durations, rate equalization, frame expansion, noise, the sampling loop, shelf filter,
  peak limit, sentence splitting, German frontend.
- `tools/quantize.py`:
  - fp16: an own converter (onnxconverter-common and onnxruntime's float16 tool produced invalid graphs here). Kept in
    float32: the sin/cos tables with their inputs (RoPE, time embedding), the log-mel front, the decoder head and the
    inverse STFT (phase up to ±74 rad).
  - int8: dynamic quantization (standard `DynamicQuantizeLinear` / `MatMulInteger`), chosen per node by a sensitivity
    scan (`tools/int8_sensitivity.py`): every MatMul/Gemm with constant weights except the output heads (→ 96, → 1,
    decoder head), projections of raw inputs and latents, and the DFT/iSTFT bases. The `encoder` and `voice` graphs
    are the fp16 ones (the voice tokens react strongly to int8: 23.6 dB for the FF down-projection in the pooling,
    25.9 dB for the token head; both run once per voice). Keeping the FF down-projections in float32 gave the same
    transcripts but was clearly slower, so it was dropped.
- Runtime package `articulino_onnx` (onnxruntime, numpy, soundfile, num2words): `TTS.from_pretrained`, `set_voice`,
  `say`, `stream`, `Speech`; the German frontend is the same module as in the PyTorch package. Rate equalization and
  one retry with seed + 1 when the output looks broken, as in the original service; low-shelf as a 1024-tap FIR
  (equal to `sosfilt` within 7e-9), peak limit 0.95, 0.15 s between sentences. Noise from `np.random.default_rng(seed)`
  in the order of the reference vectors (seed 1234 reproduces them).

## Findings

- **Thread pools:** one session per graph, each with its own spinning pool, made the pipeline 10× slower (1219 instead
  of 134 ms per sentence). Fix: one process-wide pool (`set_global_thread_pool_sizes`, `use_per_session_threads=False`).
- **BLAS threads:** a NumPy matmul in the frame expansion let OpenBLAS threads spin on the same cores, and the first
  DiT step took 80 instead of 18 ms. The frame expansion is now BLAS-free (band structure, gather).
- **Measuring on shared cores:** single runs scattered between 5× and 36×. `tools/rtf.py` waits for < 20 % load before
  every pass and takes the best of five passes per sentence; PyTorch is measured by the same script.
- **„Königstraße zwölf.“** With rate equalization (the default), Parakeet hears the sample sentence de-1 as
  „Königstraße 12 Uhr.“ (seeds 0 and 5, all precisions); without it, correctly. Not an export error; the PyTorch
  package and the original service do the same.

## Equality per graph (`tools/check.py`, reference inputs per graph; German / English; SNR in dB)

| Graph | fp32 rel | fp32 SNR | fp16 SNR | int8 SNR |
|---|---|---|---|---|
| encoder.log_mel | 1.4e-5 | 119.3 | 119.3 | 119.3 |
| encoder.latents | 3.7e-6 | 114.5 | 64.6 | 64.6 (fp16 graph) |
| voice.tokens / vector / cache | ≤ 4.7e-6 | 105.9 / 107.1 / 114.6 | 56.9 / 57.9 / 59.7 | as fp16 |
| text.features | 5.2e-7 / 4.7e-7 | 124.1 / 120.8 | 68.1 / 63.4 | 38.9 / 33.7 |
| text.log_durations | 7.3e-7 / 4.4e-7 | 127.3 / 128.1 | 72.1 / 71.3 | 45.2 / 42.8 |
| dit_step.velocity (worst step) | 1.6e-6 / 9.8e-7 | 121.8 / 123.2 | 64.9 / 66.1 | 28.3 / 31.1 |
| decoder.waveform | 6.3e-6 / 9.2e-6 | 112.0 / 107.5 | 59.9 / 57.7 | 23.3 / 16.4 (log-mel 0.30 dB) |
| whole chain, audio | 3.2e-3 / 5.3e-4 | 65.3 / 74.9 | log-mel 0.67 / 0.28 dB | log-mel 3.2 / 4.3 dB |

The fp32 tolerances of the specification hold (every block < 1e-3, audio > 40 dB). For scale, another seed changes
the chain by 5.1 (de) / 11.4 (en) dB log-mel. Batch 2 equals batch 1 in every graph. onnxruntime 1.17.3 (Python 3.11,
NumPy 1.26) gives the same values (fp32 chain 69.1 / 71.4 dB).

## Speed

See the README. Sizes: fp32 537 MB, fp16 273 MB, int8 223 MB (encoder and voice in fp16).

Browser: onnxruntime-web 1.30 (WebAssembly, under Node.js 24) loads and runs all 15 graphs with the same deviations as
native. Roughly 3× real time in fp32 and 4.5× in int8 with one thread, 8× and 12× with four threads.

## Open points

- GPU: fp16 was only checked on a CPU; CUDA, ROCm/MIGraphX, DirectML, CoreML and WebGPU are untested.
- In the browser, the host code in JavaScript and a German frontend are missing.
- Re-basing the export on the released safetensors (instead of the training checkpoints) would make it reproducible
  from public files.
