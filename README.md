---
language:
- de
- en
license: apache-2.0
pipeline_tag: text-to-speech
library_name: onnx
base_model: eigenhand/Articulino-DE-EN-Realtime-Stream-TTS
tags:
- text-to-speech
- tts
- onnx
- onnxruntime
- german
- english
- voice-cloning
- zero-shot
- flow-matching
- dmd2
- cpu
---

# Articulino-DE-EN-Realtime-Stream-TTS-ONNX

**The ONNX version of [Articulino-DE-EN-Realtime-Stream-TTS](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS):
small, fast German and English text to speech with zero-shot voice cloning, without PyTorch.**
It runs on [ONNX Runtime](https://onnxruntime.ai) (CPU, CUDA and other execution providers; the graphs also run in
onnxruntime-web). The weights are the released ones, exported graph by graph and checked against reference values
computed with the original PyTorch code. Code and weights are Apache 2.0, and commercial use is allowed.

| | |
|---|---|
| Languages | German (main), English (limited) |
| Voice | any voice from a 3 to 8 second reference recording (zero-shot cloning) |
| Output | 24 kHz mono |
| Runtime | `onnxruntime`, `numpy`, `soundfile`, `num2words` (German frontend); no PyTorch, no SciPy |
| Precisions | `fp32` (reference), `fp16` (GPU), `int8` (CPU, dynamic quantisation) |
| Speed (whole pipeline) | 33× real time in fp32 and 48× in int8 on 12 CPU efficiency cores (PyTorch on the same cores: 27×) |
| Streaming | sentence by sentence while the text is still arriving |
| License | Apache 2.0 for code and weights |

Everything about the model itself (training data, training, evaluation, limitations, responsible use) is in the
[main model card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS). This card covers the ONNX
graphs, the runtime package and their measurements.

## Quickstart

```bash
pip install huggingface_hub
hf download eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-ONNX --local-dir articulino-onnx
pip install ./articulino-onnx            # installs the small runtime package articulino_onnx
```

```python
from articulino_onnx import TTS

tts = TTS.from_pretrained("articulino-onnx")              # fp16 with a GPU provider, else fp32; or precision="int8"
tts.set_voice("voice.wav")                                # 3–8 s of clean speech from someone who agreed to this
speech = tts.say("Guten Morgen! Der Termin ist um 15:30 Uhr.", path="morgen.wav")
print(speech.duration, speech.sample_rate)                # seconds, 24000
```

`say()` returns a `Speech` with the audio as a float32 NumPy array between -1 and 1, its sample rate and the seed;
with `path` it also writes a WAV file. German text goes through the German frontend first, which spells out numbers,
dates, times, money, units, phone numbers, formulas, e-mail and web addresses, and abbreviations letter by letter
(PDF → „peh deh eff“). Long text is split into sentences, and each sentence is generated on its own.

| Option | Values | Default |
|---|---|---|
| `language` | `"de"`, `"en"` | `"de"` |
| `speed` | 0.7 to 1.6 | 1.0 |
| `seed` | non-negative integer; the same seed gives the same audio | 0 |
| `normalize` | run the German frontend (only for `language="de"`) | `True` |
| `path` | a `.wav` file | none |

Words in the other language can be marked inside a sentence: `"Das [en]Meeting[/en] ist verschoben."`.

`TTS.from_pretrained(path, precision="auto", providers=None, threads=None, **options)`: `path` is a local folder or
the repo id (then only the chosen precision is downloaded, with `huggingface_hub`); `providers` are ONNX Runtime
execution providers (default: CUDA, DirectML or CoreML when installed, else CPU); `threads` is the number of CPU
threads, shared by all graphs of the process. Options, as in the service the model was built for:
`uniform_rate=True` (every sentence at the voice's own letter rate × `speed`), `retry_broken=True` (a sentence with
outlier latents or a whistle tone is made once more with seed + 1) and `pause=0.15` (seconds between sentences).

### `stream()`: speak while the text is still coming

```python
import sounddevice as sd                          # pip install sounddevice

def llm_tokens():                                 # any iterator of text pieces, e.g. tokens from a language model
    yield from ["Guten Mor", "gen! Heute ", "regnet es ab 18 Uhr. ", "Nimm einen Schirm mit."]

with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as speaker:
    for chunk in tts.stream(llm_tokens()):
        speaker.write(chunk)                      # one chunk per sentence
```

`stream()` collects incoming text until a sentence is complete, then speaks it at once; the frontend works on the
stream without cutting through abbreviations. Sentences of fewer than 20 letters wait for the next one, and very long
ones are split at commas. Every chunk after the first starts with 0.15 s of silence (the pause between sentences).

## Files

| File | Size | What it does | Runs |
|---|---|---|---|
| `config.json` | 1 kB | constants of the pipeline (prefix and lead-in frames, step times, post-processing) | |
| `{fp32,fp16,int8}/encoder.onnx` | 91 / 48 / 48 MB | waveform → normalised voice latents (log-mel inside) | once per voice |
| `{fp32,fp16,int8}/voice.onnx` | 171 / 85 / 85 MB | voice latents → 32 voice tokens, voice vector, voice cache | once per voice |
| `{fp32,fp16,int8}/text.onnx` | 84 / 42 / 27 MB | letters → text features and log durations | once per sentence |
| `{fp32,fp16,int8}/dit_step.onnx` | 101 / 50 / 29 MB | one of the four student steps | 4× per sentence |
| `{fp32,fp16,int8}/decoder.onnx` | 91 / 48 / 34 MB | latents → 24 kHz waveform (iSTFT inside) | once per sentence |
| `articulino_onnx/` | | the runtime package (NumPy host code, German frontend) | |

Export, quantisation and test scripts are in [`tools/`](https://github.com/eigenhand/articulino-onnx/tree/main/tools)
of the code repository [eigenhand/articulino-onnx](https://github.com/eigenhand/articulino-onnx) (the export needs
PyTorch and the original checkpoints).

| Folder | Total | Use |
|---|---|---|
| `fp32/` | 537 MB | reference; CPU and GPU |
| `fp16/` | 273 MB | GPU (CUDA, DirectML, CoreML, WebGPU); on a CPU it runs at about float32 speed |
| `int8/` | 224 MB | CPU: text, DiT and decoder with dynamically quantised int8 matrix products; encoder and voice graphs (run once per voice) are the float16 ones |

## Graphs

Opset 18 (IR version 8), standard ONNX operators only, no contrib ops; the int8 graphs use the standard
`DynamicQuantizeLinear` and `MatMulInteger`. Every sequence axis is dynamic, and so is the batch axis. Tested with
onnxruntime 1.31.0 (Python 3.12, NumPy 2.5) and 1.17.3 (Python 3.11, NumPy 1.26).

| Graph | Inputs | Outputs |
|---|---|---|
| `encoder` | `waveform` f32 [B, S] (24 kHz mono, S a multiple of 960) | `latents` f32 [B, S/960, 96]; `log_mel` f32 [B, 100, S/240] (for tests) |
| `voice` | `latents` f32 [B, R, 96] | `voice_tokens` f32 [B, 32, 768]; `voice_vector` f32 [B, 768]; `voice_cache` f32 [8, 2, B, 6, P, 64] with P = min(R, 125) |
| `text` | `text_ids` i64 [B, N]; `language` i64 [B, N] (0 de, 1 en); `text_mask` bool [B, N]; `voice_vector` f32 [B, 768] | `text_features` f32 [B, N, 512]; `log_durations` f32 [B, N] |
| `dit_step` | `x` f32 [B, L, 96]; `t` f32 [B]; `frame_cond` f32 [B, L, 512]; `voice_tokens`; `voice_vector`; `voice_cache` | `velocity` f32 [B, L, 96] |
| `decoder` | `latents` f32 [B, F, 96] | `waveform` f32 [B, F·960] |

Inputs and outputs are float32 in every precision. Inside the fp16 graphs, the sine/cosine tables (rotary positions
up to ~600, time features t·1000·f), the log-mel front end and the decoder head with its iSTFT (phases reach ±74 rad)
stay float32. The STFT of the log-mel and the inverse STFT of the decoder are written as matrix products (windowed DFT
bases), the overlap-add as four shifted additions divided by the overlap-added squared window, exactly like
`torch.istft`; so no STFT operator is needed.

### The pipeline, for other runtimes

`articulino_onnx/engine.py` is the whole host side in about 200 lines of NumPy. In short:

```text
once per voice
  wav      = mono 24 kHz, cut to at most 8 s and to a multiple of 960 samples
  latents  = encoder(wav)                                   # [1, R, 96]
  tokens, vector, cache = voice(latents)                    # cache holds the first P = min(R, 125) frames (5 s)

per sentence (letters: ids 1..52 of " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789")
  ids  = [0] + letters ; mask = [false] + [true]·n ; lang = [lang_1] + lang per letter   # 0 = prompt placeholder
  h, logd = text(ids, lang, mask, vector)
  d    = min(exp(logd)·mask, 400) / speed                   # 10 ms units
  (optional, on by default: scale the letter durations to the voice's own letter rate × speed, see below)
  d[0] = 4·P                                                # the placeholder carries the voice prompt
  L    = ceil(sum(d) / 4)                                   # 25 Hz latent frames, prompt included
  cond[j] = Σ_l overlap([start_l, end_l), [4j, 4j+4)) / 4 · h[l]     for j = P .. L-1
  x = noise_0                                               # [1, L-P, 96] standard normal
  for k, t in enumerate([0, 0.25, 0.5, 0.75]):
      v  = dit_step(x, t, cond, tokens, vector, cache)
      x1 = x + (1 - t)·v
      if k < 3:  x = (1 - t_next)·noise_{k+1} + t_next·x1
  wav = decoder(concat(latents[:, P-12:P], x1))             # 12 prompt frames as lead-in
  wav = low_shelf(+2 dB at 180 Hz, RBJ, S = 0.7)(wav)[12·960:]   # filter over the lead-in, then cut it
  wav = wav · 0.95 / max|wav|  if max|wav| > 0.95
```

Rate equalisation (as in the live service): once per voice, the median letter rate (letters and digits per second,
single letters capped at 3× the median) that the duration model gives six neutral sentences; per sentence, the letter
durations are scaled to that rate × `speed`, pauses at spaces and punctuation are left alone. A sentence whose latents
have outliers (std > 1.6 or |z| > 9) or whose audio has a narrow-band whistle above 1.8 kHz is generated once more with
seed + 1.

## Accuracy

Every graph was compared with reference values computed by the original PyTorch code on a CPU in float32 (one German
and one English sentence, voice prompt and noise fixed; every intermediate result). Per graph, the inputs are the
reference inputs, so errors do not add up; `rel` is max|Δ| / max|reference|, SNR is in dB.

| Graph output | fp32 rel | fp32 SNR | fp16 SNR | int8 SNR |
|---|---|---|---|---|
| encoder: log-mel | 1.4·10⁻⁵ | 119.3 | 119.3 | 119.3 |
| encoder: latents | 3.7·10⁻⁶ | 114.5 | 64.6 | 64.6 (fp16 graph) |
| voice: tokens / vector / cache | ≤ 4.7·10⁻⁶ | 105.9 / 107.1 / 114.6 | 56.9 / 57.9 / 59.7 | as fp16 |
| text: features (de / en) | 5.2·10⁻⁷ / 4.7·10⁻⁷ | 124.1 / 120.8 | 68.1 / 63.4 | 38.9 / 33.7 |
| text: log durations (de / en) | 7.3·10⁻⁷ / 4.4·10⁻⁷ | 127.3 / 128.1 | 72.1 / 71.3 | 45.2 / 42.8 |
| dit_step: velocity, worst of 4 steps (de / en) | 1.6·10⁻⁶ / 9.8·10⁻⁷ | 121.8 / 123.2 | 64.9 / 66.1 | 28.3 / 31.1 |
| decoder: waveform (de / en) | 6.3·10⁻⁶ / 9.2·10⁻⁶ | 112.0 / 107.5 | 59.9 / 57.7 | 23.3 / 16.4 |

The waveform SNR of the int8 decoder is low because it counts phase errors; its log-mel distance to the reference is
0.3 dB. The int8 recipe comes from a per-layer sensitivity scan: output heads, input projections and the fixed
DFT/iSTFT matrices stay float32, and so do the encoder and voice graphs (the voice tokens react strongly to int8, and
both run only once per voice).

The whole chain (waveform → … → audio, with the same noise) gives an audio SNR of 65.3 dB (German) and 74.9 dB (English) against PyTorch in
float32. In float16 and int8 the four steps let small differences grow, as any change of the input noise would, so
the waveform is no longer comparable sample by sample. Spectrally, the change is small: the mean log-mel distance to
the float32 reference is 0.7 / 0.3 dB (fp16) and 3.2 / 4.3 dB (int8), while drawing the noise with another
seed changes it by 5.1 / 11.4 dB.

## Quality on unseen speakers

Each precision spoke the test sets of the main card's evaluation with **the same noise** as the PyTorch reference
(float32, CPU): 320 German utterances from 160 speakers and 107 English utterances from 4 speakers that the model
never saw, three seeds each, plus the 129-sentence German pronunciation suite. Whisper large-v3 (beam 5) for WER and
CER, ECAPA cosine to the voice prompt for similarity, UTMOS for quality; 95 % bootstrap intervals over utterances.

| German (320) | WER | CER | Similarity | UTMOS |
|---|---|---|---|---|
| PyTorch reference | 4.72 % [3.90, 5.59] | 1.84 % [1.46, 2.24] | 0.659 [0.649, 0.669] | 3.30 [3.27, 3.34] |
| ONNX fp32 | 4.73 % [3.91, 5.59] | 1.84 % [1.46, 2.24] | 0.659 [0.649, 0.669] | 3.30 [3.27, 3.34] |
| ONNX fp16 | 4.76 % [3.92, 5.63] | 1.83 % [1.45, 2.23] | 0.659 [0.649, 0.670] | 3.30 [3.27, 3.34] |
| ONNX int8 | 4.92 % [4.09, 5.79] | 1.84 % [1.46, 2.26] | 0.659 [0.648, 0.669] | 3.29 [3.26, 3.33] |

| English (107) | WER | CER | Similarity | UTMOS |
|---|---|---|---|---|
| PyTorch reference | 2.12 % [1.40, 2.92] | 0.85 % [0.55, 1.19] | 0.590 [0.577, 0.603] | 4.28 [4.26, 4.30] |
| ONNX fp32 | 2.12 % [1.40, 2.92] | 0.85 % [0.55, 1.19] | 0.590 [0.577, 0.603] | 4.28 [4.26, 4.30] |
| ONNX fp16 | 2.12 % [1.39, 2.93] | 0.86 % [0.54, 1.21] | 0.590 [0.578, 0.603] | 4.28 [4.27, 4.30] |
| ONNX int8 | 2.25 % [1.52, 3.09] | 0.86 % [0.55, 1.21] | 0.589 [0.576, 0.601] | 4.27 [4.25, 4.29] |

| Pronunciation suite (129) | CER, Whisper, 3 seeds | Sentences correct in all seeds | Parakeet, seed 0: correct |
|---|---|---|---|
| PyTorch reference | 1.41 % [0.71, 2.32] | 116 | 122 |
| ONNX fp32 | 1.41 % [0.71, 2.32] | 116 | 122 |
| ONNX fp16 | 1.42 % [0.70, 2.32] | 116 | 122 |
| ONNX int8 | 1.16 % [0.50, 2.02] | 120 | 122 |

**All three precisions are equivalent to the reference** on all three test sets: the 90 % interval of every paired
difference lies inside the margins fixed before the measurement (±1.0 point WER, ±0.5 point CER, ±0.01 similarity,
±0.05 UTMOS). The largest difference, int8 in German, is +0.19 points WER [−0.07, 0.48]. Median log-mel distance to
the reference output: 0.001 dB (fp32), 0.20 dB (fp16), 2.7 dB (int8); another seed of the reference itself differs
by 5.9 dB.

## Speed

Whole pipeline per sentence (letters, text graph, durations, four DiT steps, decoder, shelf filter), ten German
sentences (40.7 s of audio) after one warm-up sentence, voice cache computed beforehand, seeds 42 + i. Measured the same
way as the PyTorch numbers of the main card, on the same machine and cores: Intel Core i9-14900K efficiency cores in a
virtual machine, ONNX Runtime 1.31, PyTorch 2.14 in float32.

| Threads | ONNX fp32 | ONNX fp16 | ONNX int8 | PyTorch fp32, same method | PyTorch, main card | First sentence (fp32 / int8) |
|---|---|---|---|---|---|---|
| 12 | 33.0× | 30.6× | 47.8× | 27.1× | 24.9× | 0.13 s / 0.09 s |
| 8 | 28.2× | 26.7× | 42.1× | 23.0× | 20.6× | 0.16 s / 0.10 s |
| 4 | 17.3× | 16.6× | 28.2× | 14.7× | 12.9× | 0.26 s / 0.16 s |
| 2 | 9.7× | 9.7× | 16.9× | 8.5× | 8.3× | 0.46 s / 0.26 s |
| 1 | 5.2× | – | 9.4× | – | – | – |

The cores were shared with other jobs, so every pass waited for idle cores and each sentence counts with its fastest
of five passes (`tools/rtf.py`); PyTorch was measured by the same script, which is why it comes out a little faster
than in the main card. The public `say()` with the German frontend, rate equalisation and the broken-output check runs
at 32.2× (fp32) and 46.6× (int8) on 12 threads. fp16 on a CPU is computed in float32 by ONNX Runtime.

ONNX Runtime keeps one thread pool per session by default. With five sessions, the idle pools keep spinning while the
next graph runs, and the pipeline was ten times slower; `articulino_onnx` therefore creates one process-wide pool
(`threads` of the first `TTS` in a process is used for all of them).

## Browser and other platforms

The graphs use only standard operators at opset 18. All five run in **onnxruntime-web 1.30** (WebAssembly backend,
tested under Node.js) with the same results as onnxruntime: all five graphs in all three precisions load and run, with the same deviations as in the table above (float32: ≤ 7·10⁻⁶). Rough speed of the WebAssembly backend under Node.js on the same cores: 3× real time in fp32 and 4.5× in int8 on one thread, 8× and 12× on four threads. A browser needs the host code above in
JavaScript (about 200 lines), the letter table and a German frontend of its own (the Python one is not ported).
WebGPU, mobile (onnxruntime-mobile, CoreML, NNAPI) and Windows (DirectML) were not tested.

## Limitations

- **Same seed, other noise than PyTorch.** The noise comes from NumPy's default generator, so a seed gives different
  (equally good) audio than the PyTorch package with the same seed. With the same noise passed in, the fp32 audio
  matches PyTorch at 65–75 dB SNR.
- **GPU numbers are missing.** The fp16 graphs were checked on a CPU (ONNX Runtime upcasts them there); CUDA was not
  available for this export.
- **int8 changes the sound a little** (2.7 dB log-mel, less than half of what another seed changes); intelligibility,
  similarity and quality stay equivalent to float32 (see Quality).
- **One thread pool per process** (see Speed): a second `TTS` in the same process shares the first one's thread count.
- **Sentence-level streaming only,** like the PyTorch release; a sentence's audio is ready all at once.
- Everything in the [limitations of the main card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS#limitations)
  applies: German first, letters instead of phonemes, about one in fifteen renderings has a small slip, 24 kHz output.

## Responsible use

Articulino can make a convincing copy of a person's voice from a few seconds of audio. Only clone voices with consent,
tell people when they hear synthetic speech, and never use it to impersonate someone or to get around voice-based
authentication. Since 2 August 2026, Article 50 of the EU AI Act requires providers of systems that generate synthetic
audio to mark it in a machine-readable way; these weights contain no watermark. See the
[main card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS#responsible-use) for details.

## License and attribution

Code and weights: [Apache License 2.0](LICENSE). Three training corpora are CC BY 4.0; their attribution is in
[NOTICE](NOTICE), which must stay with the model when you redistribute it. The German frontend is part of the code.
The overall recipe follows [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) by Canberk Aslan;
see the main card.

## Citation

```bibtex
@misc{articulino2026,
  title        = {Articulino-DE-EN-Realtime-Stream-TTS: Small German and English Text to Speech with Zero-Shot Voice Cloning},
  author       = {eigenhand},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS}}
}
```
