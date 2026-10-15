Reference values computed with the original inference code (CPU, float32, given noise), one file set per
sentence: `<name>.npz` (every intermediate step), `<name>.json` (sentence, language, timesteps, frame counts and the
shape of every array) and `<name>.wav` (the resulting audio). Every engine recomputes each step from the reference
inputs and must stay within the tolerances of docs/SPEC.md (in the PyTorch repository, eigenhand/articulino).
