#!/bin/bash
# Real-time factor series: ONNX fp32/fp16/int8 at 12, 8, 4 and 2 threads (see rtf.py: idle wait before each pass,
# fastest of 5 passes per sentence), then the public say() at 12 threads. Optional: PYTHON=..., CPUS=16-27 to pin
# the runs to a set of cores, TORCH=1 to add the original PyTorch code (needs tools/parts.py set up).
#   tools/bench.sh > build/rtf.txt
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
PIN=()
[ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
for th in 12 8 4 2; do
  for p in fp32 fp16 int8; do
    "${PIN[@]}" "$PY" tools/rtf.py --precision $p --threads $th 2>/dev/null | grep ONNX
  done
  [ -n "$TORCH" ] && "${PIN[@]}" "$PY" tools/rtf.py --engine torch --threads $th 2>/dev/null | grep PyTorch
done
for p in fp32 int8; do
  "${PIN[@]}" "$PY" tools/rtf.py --precision $p --threads 12 --api 2>/dev/null | grep ONNX
done
