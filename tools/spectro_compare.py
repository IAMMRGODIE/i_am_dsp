"""Compare a render with a recording over *time*, not only over the harmonic
series.

A harmonic list says how a steady note is balanced; it says nothing about the
attack, the way the spectrum settles, or the beat between partials. This builds
a 1/12-octave spectrogram of both, matches each frame's level so loudness is not
part of the number, and reports the mean absolute difference per band over time.

    cargo run -r -p i_am_string --example sweep -- target/bow cello 28:68 4.5
    python tools/spectro_compare.py target/bow "C:/Samples/Orchestral/Strings/Celli Legato"

The renders must be named preset_note_hertz.wav (what the sweep example writes).
"""

import os, sys, glob, warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
warnings.filterwarnings("ignore")

PER_OCT = 12
LO, HI = 80.0, 8000.0
CENTRES = np.array([2 ** (k / PER_OCT) for k in range(int(np.ceil(np.log2(LO) * PER_OCT)), int(np.floor(np.log2(HI) * PER_OCT)) + 1)])
EDGES = np.array([c * 2 ** (-0.5 / PER_OCT) for c in CENTRES] + [CENTRES[-1] * 2 ** (0.5 / PER_OCT)])

def load(p):
    sr, d = wavfile.read(p)
    if d.ndim > 1:
        d = d.mean(axis=1)
    x = d.astype(np.float64)
    return sr, x / (32768.0 if np.max(np.abs(x)) > 1.5 else 1.0)

def f0_of(x, sr):
    """Coarse pitch from the strongest partial and its spacing."""
    seg = x[int(0.10 * sr):int(0.60 * sr)] if len(x) > int(0.60 * sr) else x
    if len(seg) < 2048:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
    freqs = np.arange(len(spec)) * sr / (1 << 16)
    peaks = np.where((spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:]))[0] + 1
    peaks = peaks[(freqs[peaks] > 40) & (freqs[peaks] < 2000)]
    if len(peaks) < 4:
        return 0.0
    peaks = peaks[np.argsort(spec[peaks])[::-1][:12]]
    best, best_score = 0.0, -1.0
    for p in peaks:
        for k in (1, 2, 3):
            cand = freqs[p] / k
            if cand < 45:
                continue
            score = 0.0
            for m in range(1, 12):
                target = cand * m
                if target > 6000:
                    break
                close = np.abs(freqs[peaks] - target) < max(2.0, target * 0.015)
                if close.any():
                    score += 1.0 / m
            if score > best_score:
                best, best_score = cand, score
    return best

def bands_of(x, sr, n=2048, hop=512):
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    frames = []
    spec_all = None
    for s in range(0, max(1, len(x) - n), hop):
        mag = np.abs(np.fft.rfft(x[s:s + n] * win)) ** 2
        band = np.zeros(len(CENTRES))
        for i in range(len(CENTRES)):
            m = (freqs >= EDGES[i]) & (freqs < EDGES[i + 1])
            band[i] = mag[m].mean() if m.any() else 0.0
        frames.append(10 * np.log10(band + 1e-20))
    return np.array(frames)

def align(x, sr):
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak <= 0:
        return x
    start = int(np.argmax(np.abs(x) > 0.02 * peak))
    return x[max(0, start - int(0.01 * sr)):]

def retune(x, sr, ratio):
    """Shift a signal's pitch by ratio (f0_new = f0_old * ratio), resampling."""
    n = max(2, int(len(x) / ratio))
    source = np.linspace(0, len(x) - 1, n)
    return np.interp(source, np.arange(len(x)), x)


def compare(model_path, sample_path):
    sr, m = load(model_path)
    sr2, s = load(sample_path)
    if sr != sr2:
        return None
    m, s = align(m, sr), align(s, sr)
    # The model is rendered on the equal-tempered grid, so its partials sit
    # within a few percent of the recording's; without removing that offset the
    # comparison would be measuring tuning, and a one percent detune moves a
    # 2 kHz partial by a whole band.
    m_f0 = note_of(model_path)
    s_f0 = f0_of(s, sr)
    if m_f0 > 0 and s_f0 > 0:
        m = retune(m, sr, s_f0 / m_f0)
    fm = bands_of(m, sr)
    fs = bands_of(s, sr)
    n = min(len(fm), len(fs), int(1.6 * sr / 512))
    if n < 10:
        return None
    fm, fs = fm[:n], fs[:n]
    # compare against the reference's own level: every frame is matched in level,
    # so what is measured is the shape over frequency and time, not the loudness.
    for f in range(n):
        if fs[f].max() > -80:
            fm[f] += fs[f].max() - fm[f].max()
    # Only bands the recording actually has energy in: a band that is digital
    # silence in one and -90 dB in the other would otherwise count as a 110 dB
    # error and drown the bands a person can hear.
    loud = fs > (fs.max(axis=1, keepdims=True) - 40.0)
    valid_frames = fs.max(axis=1) > -60
    rows = np.where(valid_frames)[0]
    diff_per_frame = []
    for fi in rows:
        m = loud[fi]
        diff_per_frame.append(float(np.abs(fm[fi][m] - fs[fi][m]).mean()) if m.any() else 0.0)
    diff = np.array(diff_per_frame)
    per_frame = diff
    frame_t = rows * 512 / sr
    return {
        "mean": float(diff.mean()) if diff.size else float("nan"),
        "per_frame": (frame_t, per_frame),
        "frames": len(rows),
    }

def note_of(path):
    """The hertz in a render's name: preset_note_hertz.wav."""
    base = os.path.basename(path).rsplit(".", 1)[0]
    parts = base.split("_")
    for p in reversed(parts):
        try:
            return float(p)
        except ValueError:
            continue
    return 0.0

def pair_up(model_dir, sample_dir):
    models = sorted(glob.glob(os.path.join(model_dir, "*.wav")))
    samples = sorted(glob.glob(os.path.join(sample_dir, "*.wav")))
    mf = [(price, os.path.basename(p)) for p in models for price in [note_of(p)]]
    pairs = []
    for sp in samples:
        sr, s = load(sp)
        f = f0_of(s, sr)
        if f <= 0:
            continue
        best = min(models, key=lambda mp: abs(np.log2(max(note_of(mp), 20.0) / f)))
        if abs(np.log2(max(note_of(best), 20.0) / f)) < 0.03:
            pairs.append((best, sp, f))
    return pairs

if __name__ == "__main__":
    model_dir, sample_dir = sys.argv[1], sys.argv[2]
    pairs = pair_up(model_dir, sample_dir)
    print(f"{len(pairs)} pairs")
    all_frames = {}
    means = []
    for mp, sp, f in pairs:
        r = compare(mp, sp)
        if not r:
            continue
        means.append(r["mean"])
        for t, d in zip(*r["per_frame"]):
            all_frames.setdefault(round(t, 2), []).append(d)
        print(f"  {os.path.basename(sp)[:34]:36s} f0 {f:7.1f}  mean {r['mean']:5.2f} dB over {r['frames']} frames")
    if means:
        print(f"average over pairs: {np.mean(means):.2f} dB")
    # where in frequency the difference lives, at the attack and in the sustain
    print("band profile (Hz: |diff| dB at 0.1 s / 1.0 s)")
    acc = {"attack": [], "sustain": []}
    for mp, sp, f in pairs:
        sr, m = load(mp); sr2, s = load(sp)
        m, s = align(m, sr), align(s, sr)
        fm = bands_of(m, sr); fs = bands_of(s, sr)
        n = min(len(fm), len(fs), int(1.6 * sr / 512))
        fm, fs = fm[:n], fs[:n]
        for fi in range(n):
            if fs[fi].max() > -80:
                fm[fi] += fs[fi].max() - fm[fi].max()
        if n > 3:
            acc["attack"].append(np.abs(fm[3] - fs[3]))
        if n > 60:
            acc["sustain"].append(np.abs(fm[60] - fs[60]))
    if acc["attack"]:
        a = np.mean(acc["attack"], axis=0)
        b = np.mean(acc["sustain"], axis=0) if acc["sustain"] else np.full(len(CENTRES), np.nan)
        for i, c in enumerate(CENTRES):
            if int(np.log2(c) * 12) % 4 == 0:
                print(f"  {c:7.0f} {a[i]:5.1f} {b[i]:5.1f}")

    print("time profile (s: mean |diff| dB)")
    keys = sorted(all_frames)
    for target in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        near = min(keys, key=lambda k: abs(k - target))
        print(f"  {near:4.2f}  {np.mean(all_frames[near]):5.2f}  n={len(all_frames[near])}")
