"""Export-friendly wrappers around the original inference code of the model.

The export starts from the training checkpoints and the original inference modules, which are not public: set
ARTICULINO_ORIGINAL_CODE to the folder with those modules and ARTICULINO_CHECKPOINTS to the folder with the
checkpoints. The released ONNX files are the output of this export.

Every wrapper takes the original modules (with their trained weights) and recomputes exactly what they compute,
written so that torch.onnx.export produces plain, portable ONNX ops:

  EncoderGraph   waveform (B, T) -> normalised latents (B, T/960, 96)          log-mel as a DFT convolution
  VoiceGraph     latents (B, R, 96) -> voice tokens, voice vector, voice cache  RefEncoder + DiT prefix pass
  TextGraph      ids, language, mask, voice vector -> text features, log durations
  DiTStepGraph   new frames, t, frame condition, tokens, vector, cache -> velocity
  DecoderGraph   normalised latents (B, L, 96) -> waveform (B, L*960)           iSTFT as MatMul + overlap-add

Deviations from the original code are purely algebraic (identical in exact arithmetic):
  * AdaNorm: the condition is the same for every frame, so its linear layer runs once on (B, 1, d) and broadcasts.
  * DiT input layer: the zero halves of concat(x, 0, 0) / concat(0, prefix, 1) are dropped by slicing the weight.
  * Boolean masks fill with -1e4 instead of -1e9 (exp underflows to exactly 0 either way; -1e4 survives float16).
  * STFT / iSTFT / mel are written as matrix products with the same window, padding and normalisation as torch.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ORIGINAL_CODE = Path(os.environ.get("ARTICULINO_ORIGINAL_CODE", "original_code"))
CHECKPOINTS = Path(os.environ.get("ARTICULINO_CHECKPOINTS", "checkpoints"))
STUDENT = CHECKPOINTS / "schueler-dmd1-020000.pt"
CODEC = CHECKPOINTS / "vocoder-d1-020000.pt"

if str(ORIGINAL_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE))

PREFIX = 125          # latent frames of the voice prompt used as cached prefix (5 s)
LEAD_IN = 12          # latent frames of the prefix given to the decoder before the new frames
NEG = -1.0e4          # mask fill (float16-safe; exp() underflows to 0 just like -1e9)


def load_original():
    """-> (student, codec, mean, std, times) from the original code and the training checkpoints."""
    from schueler_modell import Schueler
    ck = torch.load(STUDENT, map_location="cpu", weights_only=False)
    m = Schueler(**ck["cfg"])
    m.load_state_dict(ck["ema"])
    m.eval()
    spec = importlib.util.spec_from_file_location("vocoder_modell", ORIGINAL_CODE / "vocoder_modell.py")
    vm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    vck = torch.load(CODEC, map_location="cpu", weights_only=False)
    ae = vm.Autoencoder(**vck["cfg"])
    ae.load_state_dict(vck["ae"])
    ae.eval()
    mean = torch.as_tensor(np.asarray(ck["mittel"])).float()
    std = torch.as_tensor(np.asarray(ck["streuung"])).float()
    return m, ae, vm, mean, std, tuple(ck.get("zeiten", (0.0, 0.25, 0.5, 0.75)))


# ----------------------------------------------------------------------------------------------- shared pieces
def rope(x, cos, sin):
    """Interleaved RoPE exactly as modell.rope: pairs (x[0::2], x[1::2])."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


def rope_table(start, length, freq):
    """cos/sin for positions start .. start+length-1; freq as in modell.rope_tabelle."""
    pos = torch.arange(length, dtype=torch.int64, device=freq.device) + start
    w = pos.to(torch.float32)[:, None] * freq[None, :]
    return w.cos(), w.sin()


def rope_freq(dh):
    return 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))


def attention(att, x, ctx, heads, mask=None, rot=None):
    """modell.Attention.forward. mask: (B, S) bool of valid keys or None."""
    b, l, d = x.shape
    s = ctx.shape[1]
    q = att.q(x).view(b, l, heads, d // heads).transpose(1, 2)
    kv = att.kv(ctx).view(b, s, 2, heads, d // heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    if rot is not None:
        q, k = rope(q, *rot), rope(k, *rot)
    a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // heads)
    if mask is not None:
        a = torch.where(mask[:, None, None, :], a, torch.full_like(a, NEG))
    y = torch.matmul(a.softmax(-1), v).transpose(1, 2).reshape(b, l, d)
    return att.aus(y)


def ff(f, x):
    return f.b(F.gelu(f.a(x)))


def block(blk, x, heads, mask, rot):
    """modell.Block (pre-norm transformer layer of text and voice encoders)."""
    xn = blk.n1(x)
    x = x + attention(blk.att, xn, xn, heads, mask, rot)
    return x + ff(blk.ff, blk.n2(x))


def ada(norm, x, c):
    """modell.AdaNorm with c of shape (B, 1, d) (broadcast over frames)."""
    g, b = norm.lin(c).chunk(2, -1)
    return norm.norm(x) * (1 + g) + b


def time_features(t, f):
    """Lehrer.zeit_merkmal: t (B,) -> (B, 256)."""
    w = t[:, None] * 1000 * f[None]
    return torch.cat([w.sin(), w.cos()], -1)


def dit_block(m, blk, x, c, tokens, rot, k_prev=None, v_prev=None, kv_only=False):
    """Lehrer._block. Returns (x, k, v); k/v are the keys/values of x itself (after RoPE on k)."""
    b, l, d = x.shape
    h = m.koepfe
    xn = ada(blk.n1, x, c)
    kv = blk.att.kv(xn).view(b, l, 2, h, d // h).permute(2, 0, 3, 1, 4)
    k_new, v_new = rope(kv[0], *rot), kv[1]
    if kv_only:
        return x, k_new, v_new
    q = rope(blk.att.q(xn).view(b, l, h, d // h).transpose(1, 2), *rot)
    k, v = k_new, v_new
    if k_prev is not None:
        k, v = torch.cat([k_prev, k_new], 2), torch.cat([v_prev, v_new], 2)
    a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // h)
    x = x + blk.att.aus(torch.matmul(a.softmax(-1), v).transpose(1, 2).reshape(b, l, d))
    x = x + attention(blk.kreuz, blk.n2(x), tokens, h)
    return x + ff(blk.ff, ada(blk.n3, x, c)), k_new, v_new


# ----------------------------------------------------------------------------------------------- graphs
class EncoderGraph(nn.Module):
    """Waveform (B, T), T a multiple of 960 -> normalised latents (B, T/960, 96) and log-mel (B, 100, T/240).

    Log-mel = torchaudio MelSpectrogram(24000, n_fft 1024, win 960, hop 240, 100 mels, HTK, power 1, center,
    reflect) as a strided convolution with a windowed DFT basis. Only the T/240 frames the encoder keeps are computed
    (right reflect pad 272 instead of 512 drops exactly the extra last frame)."""

    def __init__(self, ae, mean, std):
        super().__init__()
        import torchaudio
        self.enc = ae.encoder
        n_fft, win = 1024, 960
        w = torch.zeros(n_fft, dtype=torch.float64)
        w[(n_fft - win) // 2:(n_fft - win) // 2 + win] = torch.hann_window(win, periodic=True, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * ((k * n) % n_fft) / n_fft
        basis = torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], 0)            # (1026, 1024)
        self.register_buffer("dft", basis.float()[:, None, :])
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0.0, 12000.0, 100, 24000, None, "htk")
        self.register_buffer("fb", fb.float())                                      # (513, 100)
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, wav):
        x = F.pad(wav[:, None, :], (512, 272), mode="reflect")
        spec = F.conv1d(x, self.dft, stride=240)                                    # (B, 1026, T/240)
        re, im = spec[:, :513], spec[:, 513:]
        mag = torch.sqrt(re * re + im * im)
        mel = torch.matmul(mag.transpose(1, 2), self.fb).transpose(1, 2)           # (B, 100, T/240)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        h = self.enc.stufen(self.enc.ein(logmel)).transpose(1, 2)
        mu = F.linear(self.enc.norm(h), self.enc.aus.weight[:96], self.enc.aus.bias[:96])   # (B, T/960, 96); logvar unused
        return (mu - self.mean) / self.std, logmel


class VoiceGraph(nn.Module):
    """Normalised voice latents (B, R, 96) -> voice tokens (B, 32, 768), voice vector (B, 768),
    voice cache (layers 8, k/v 2, B, heads 6, P, 64) for the prefix P = min(R, 125) (Lehrer.anfang_cache)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        r = m.ref
        self.r = r
        w = m.ein.weight                                                             # (384, 193): [x_t | anfang | flag]
        self.w_pre = nn.Parameter(w[:, 96:192].detach().clone())
        self.b_pre = nn.Parameter((m.ein.bias + w[:, 192]).detach().clone())
        self.register_buffer("tf", torch.exp(-math.log(10000) * torch.arange(128) / 128))
        self.register_buffer("freq_ref", rope_freq(r.ein.out_features // r.koepfe))
        self.register_buffer("freq_dit", rope_freq(m.dim // m.koepfe))

    def forward(self, lat):
        m, r = self.m, self.r
        b, n, _ = lat.shape
        # RefEncoder (all frames valid)
        x = r.ein(lat)
        rot = rope_table(0, n, self.freq_ref)
        for blk in r.bloecke:
            x = block(blk, x, r.koepfe, None, rot)
        q = r.fragen[None].expand(b, -1, -1)
        for n1, att, n2, f in r.pool:
            q = q + attention(att, n1(q), x, r.koepfe)
            q = q + ff(f, n2(q))
        q = r.norm(q)
        tokens, vec = r.aus(q), r.vektor(q.mean(1))
        # prefix pass through the DiT (t = 1, no text), keep keys/values of every block
        pre = lat[:, :PREFIX]
        p = pre.shape[1]
        tok, g = m.tok_zu(tokens), m.g_zu(vec)
        x = torch.matmul(pre, self.w_pre.t()) + self.b_pre
        one = torch.ones(b, dtype=lat.dtype)
        c = (m.zeit(time_features(one, self.tf)) + g)[:, None]
        rot = rope_table(0, p, self.freq_dit)
        kvs = []
        last = len(m.bloecke) - 1
        for i, blk in enumerate(m.bloecke):
            x, k, v = dit_block(m, blk, x, c, tok, rot, kv_only=(i == last))
            kvs.append(torch.stack([k, v], 0))
        return tokens, vec, torch.stack(kvs, 0)


class TextGraph(nn.Module):
    """ids (B, N) int64, language (B, N) int64 (0 de, 1 en), mask (B, N) bool, voice vector (B, 768)
    -> text features (B, N, 512) (zero where mask is False), log durations (B, N) in log 10-ms units."""

    def __init__(self, m):
        super().__init__()
        self.t, self.d = m.text, m.dauer
        self.register_buffer("freq", rope_freq(m.text.emb.embedding_dim // m.text.koepfe))

    def forward(self, ids, language, mask, vec):
        te, dp = self.t, self.d
        x = te.emb(ids) + te.sprache(language)
        rot = rope_table(0, ids.shape[1], self.freq)
        for blk in te.bloecke:
            x = block(blk, x, te.koepfe, mask, rot)
        mf = mask.to(x.dtype)[..., None]
        h = te.norm(x) * mf
        y = (h + dp.stimme(vec)[:, None]) * mf
        for c, n in ((dp.c1, dp.n1), (dp.c2, dp.n2), (dp.c3, dp.n3)):
            y = n(F.relu(c(y.transpose(1, 2))).transpose(1, 2)) * mf
        return h, dp.aus(y)[..., 0]


class DiTStepGraph(nn.Module):
    """One student step on the new frames only (Lehrer.geschwindigkeit_cache):
    x (B, L, 96), t (B,), frame condition (B, L, 512), tokens (B, 32, 768), vector (B, 768),
    voice cache (8, 2, B, 6, P, 64) -> velocity (B, L, 96). New frames sit at RoPE positions P .. P+L-1."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.w_x = nn.Parameter(m.ein.weight[:, :96].detach().clone())
        self.b_x = nn.Parameter(m.ein.bias.detach().clone())
        self.register_buffer("tf", torch.exp(-math.log(10000) * torch.arange(128) / 128))
        self.register_buffer("freq", rope_freq(m.dim // m.koepfe))

    def forward(self, x, t, cond, tokens, vec, cache):
        m = self.m
        l, p = x.shape[1], cache.shape[4]
        tok, g = m.tok_zu(tokens), m.g_zu(vec)
        h = torch.matmul(x, self.w_x.t()) + self.b_x + m.text_zu(cond)
        c = (m.zeit(time_features(t, self.tf)) + g)[:, None]
        rot = rope_table(p, l, self.freq)
        for i, blk in enumerate(m.bloecke):
            h, _, _ = dit_block(m, blk, h, c, tok, rot, cache[i, 0], cache[i, 1])
        return m.aus(ada(m.norm, h, c))


def overlap_add(frames):
    """(B, F, 960) frames at hop 240 -> (B, (F+3)*240): sum of the four quarter-shifted copies (no ConvTranspose)."""
    b, f, _ = frames.shape
    q = frames.reshape(b, f, 4, 240)
    y = (F.pad(q[:, :, 0], (0, 0, 0, 3)) + F.pad(q[:, :, 1], (0, 0, 1, 2))
         + F.pad(q[:, :, 2], (0, 0, 2, 1)) + F.pad(q[:, :, 3], (0, 0, 3, 0)))
    return y.reshape(b, -1)


class DecoderGraph(nn.Module):
    """Normalised latents (B, L, 96) -> waveform (B, L*960).

    Same as vocoder_modell.Decoder: logmag/phase per bin -> torch.istft(n_fft 960, hop 240, periodic Hann, center,
    length L*960). The inverse real DFT and the synthesis window are one (962, 960) matrix; overlap-add is four
    padded adds; the result is divided by the overlap-added squared window like torch.istft."""

    def __init__(self, ae, mean, std):
        super().__init__()
        self.dec = ae.decoder
        n = 960
        k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * ((k * t) % n) / n
        c = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64)
        c[0] = 1.0
        c[-1] = 1.0
        win = torch.hann_window(n, periodic=True, dtype=torch.float64)
        basis = torch.cat([c * torch.cos(ang) / n, -c * torch.sin(ang) / n], 0) * win[None]   # (962, 960)
        self.register_buffer("basis", basis.float())
        self.register_buffer("win2", (win * win).float())
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, lat):
        d = self.dec
        z = (lat * self.std + self.mean).transpose(1, 2)
        h = d.bloecke(d.hoch(d.ein(z)))
        h = d.kopf(d.norm(h.transpose(1, 2)))                                      # (B, F, 962)
        logmag, phase = h[..., :481], h[..., 481:]
        mag = torch.clamp(torch.exp(logmag), max=100.0)
        spec = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], -1)
        y = overlap_add(torch.matmul(spec, self.basis))
        env = overlap_add(torch.ones_like(h[:1, :, :1]) * self.win2)
        return (y / env)[:, 480:-240]
