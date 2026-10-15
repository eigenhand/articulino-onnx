"""Signal helpers in plain NumPy: tone shelf (as in the original service), peak limit, resampling, broken-output check."""
from __future__ import annotations

import numpy as np

SR = 24000


def shelf_coefficients(f0: float, db: float, high: bool, sr: int = SR, s: float = 0.7) -> np.ndarray:
    """RBJ cookbook shelving biquad -> (b0, b1, b2, a0, a1, a2) normalised by a0 (server._shelf)."""
    A = 10 ** (db / 40)
    w = 2 * np.pi * f0 / sr
    c = np.cos(w)
    q = 2 * np.sqrt(A) * np.sin(w) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    z = 1 if high else -1
    b = [A * ((A + 1) + z * (A - 1) * c + q), -2 * z * A * ((A - 1) + z * (A + 1) * c), A * ((A + 1) + z * (A - 1) * c - q)]
    a = [(A + 1) - z * (A - 1) * c + q, 2 * z * ((A - 1) - z * (A + 1) * c), (A + 1) - z * (A - 1) * c - q]
    return np.array([x / a[0] for x in b + a])


def _biquad(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    b0, b1, b2, _, a1, a2 = sos
    y = np.zeros_like(x)
    z1 = z2 = 0.0
    for i, v in enumerate(x):                       # direct form II transposed (only used for short impulse responses)
        o = b0 * v + z1
        z1 = b1 * v - a1 * o + z2
        z2 = b2 * v - a2 * o
        y[i] = o
    return y


class Shelf:
    """Low shelf (+ optional high shelf) of the original service, applied as an FIR of its impulse response.

    The shelving poles decay within a few hundred samples (|h| < 1e-19 after 1024), so 1024 taps reproduce
    scipy.signal.sosfilt to float precision (7e-9) while staying vectorised (FFT convolution) without SciPy."""

    def __init__(self, bass_db=2.0, bass_hz=180.0, treble_db=0.0, treble_hz=1000.0, taps=1024):
        imp = np.zeros(taps)
        imp[0] = 1.0
        for sos in (shelf_coefficients(bass_hz, bass_db, False), shelf_coefficients(treble_hz, treble_db, True)):
            imp = _biquad(sos, imp)
        self.h = imp
        self.identity = bass_db == 0 and treble_db == 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.identity or len(x) == 0:
            return x.astype(np.float32)
        n = len(x) + len(self.h) - 1
        nfft = 1 << (n - 1).bit_length()
        y = np.fft.irfft(np.fft.rfft(x.astype(np.float64), nfft) * np.fft.rfft(self.h, nfft), nfft)[: len(x)]
        return y.astype(np.float32)


def limit_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return x * (peak / m) if m > peak else x


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited FFT resampling (for voice prompts that are not 24 kHz)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x.astype(np.float64))
    keep = min(len(spec), n_out // 2 + 1)
    out = np.zeros(n_out // 2 + 1, complex)
    out[:keep] = spec[:keep]
    return (np.fft.irfft(out, n_out) * (n_out / len(x))).astype(np.float32)


def broken(wav: np.ndarray, z: np.ndarray, tone_share=0.35, tone_ms=60, std_max=1.6, abs_max=9.0) -> str:
    """server.auffaellig: '' if fine, else a reason (latent outliers or a narrow-band whistle above 1.8 kHz)."""
    std, top = float(z.std()), float(np.abs(z).max())
    if std > std_max or top > abs_max or not np.isfinite(std):
        return f"latents std {std:.2f} max {top:.1f}"
    n, h = 1024, 256
    if len(wav) < 2 * n:
        return ""
    k = (len(wav) - n) // h + 1
    idx = np.arange(n)[None, :] + h * np.arange(k)[:, None]
    sp = np.abs(np.fft.rfft(wav[idx] * np.hanning(n), axis=1)) ** 2
    lo = int(1800 / (SR / n))
    hi = sp[:, lo:]
    peak = hi.argmax(1)
    pad = np.pad(hi, ((0, 0), (2, 2)))
    narrow = sum(pad[np.arange(k), peak + 2 + o] for o in range(-2, 3))
    total = sp.sum(1) + 1e-12
    hit = (narrow / total > tone_share) & (total > 1e-2 * n)
    run = best = 0
    for t in hit:
        run = run + 1 if t else 0
        best = max(best, run)
    if best * h / SR * 1000 >= tone_ms:
        return f"tone {best * h / SR * 1000:.0f} ms at {(np.median(peak[hit]) + lo) * SR / n:.0f} Hz"
    return ""
