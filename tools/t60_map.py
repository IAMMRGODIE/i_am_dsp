"""Per-partial T60 of the piano model and of the reference samples.

    python tools/t60_map.py measure-model target/wav/piano-t60 model.json
    python tools/t60_map.py measure-sample target/wav/ref-grand sample.json
    python tools/t60_map.py plot model.json sample.json t60.png
"""

import json
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt

PICK = [36, 48, 60, 72, 84]
NAMES = ['E2', 'C3', 'C4', 'C5', 'C6']
T_MAX = 2.0        # seconds of decay used for the rate estimate
SPAN = 12.0        # dB down from the partial's own peak: the prompt decay
FLOOR = 52.0       # dB below the note's peak: never fit into the noise
WEAK = 40.0        # a partial more than this below the note's peak is noise


def load(path):
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(abs(np.iinfo(data.dtype).min))
    else:
        data = data.astype(np.float64)
    data = data - data.mean()
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    return float(rate), data


def bandpass(x, rate, freq, band):
    nyq = rate * 0.5
    low = max(25.0, freq - band)
    high = min(nyq * 0.95, freq + band)
    if high - low < 8.0:
        return None
    sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def frames(y, rate, width=0.04, hop=0.01):
    size = int(width * rate)
    step = int(hop * rate)
    times = []
    db = []
    for i in range(0, len(y) - size, step):
        seg = y[i:i + size]
        rms = np.sqrt(np.mean(seg * seg))
        times.append((i + size * 0.5) / rate)
        db.append(20.0 * np.log10(rms + 1e-12))
    return np.array(times), np.array(db)


def robust_fit(t, db):
    """Least squares on the decay, ignoring the dips from beating.

    Two detuned unison strings add and subtract, so the envelope swings
    around the mean decay; the upper side of the swing is the decay, the
    lower side is the cancellation. Fit, drop what sits far below the fit,
    fit again.
    """
    keep = np.ones(len(t), dtype=bool)
    slope, offset = np.polyfit(t, db, 1)
    for _ in range(4):
        resid = db - (slope * t + offset)
        keep = resid > -1.0
        if keep.sum() < 8:
            keep = np.ones(len(t), dtype=bool)
            break
        slope, offset = np.polyfit(t[keep], db[keep], 1)
    resid = db - (slope * t + offset)
    return slope, offset, keep, float(np.sqrt(np.mean(resid[keep] ** 2)))


def crossing(t, db, level, start):
    for index in range(start, len(db) - 1):
        if db[index] > level >= db[index + 1]:
            span = db[index] - db[index + 1]
            frac = (db[index] - level) / span if span > 0 else 0.0
            return t[index] + frac * (t[index + 1] - t[index])
    return None


def partial_t60(x, rate, freq, band, reference):
    y = bandpass(x, rate, freq, band)
    if y is None:
        return None
    t, db = frames(y, rate)
    peak = float(db.max())
    if peak < reference - WEAK:
        return None
    start = int(np.argmax(db > peak - 3.0))
    end = min(T_MAX, t[-1] * 0.92)
    t1 = t[start]
    t2 = crossing(t, db, peak - SPAN, start)
    if t2 is None or t2 > end:
        t2 = end
    mask = (t >= t1) & (t <= t2)
    if mask.sum() < 6 or t2 - t1 < 0.08:
        return None
    slope, offset, keep, rms = robust_fit(t[mask], db[mask])
    if slope >= 0.0:
        return None
    drop = float(-slope * (t2 - t1))
    if drop < 6.0:
        return None
    return {'f': float(freq), 't60': float(60.0 / -slope),
            'peak': peak, 'span': float(t2 - t1), 'drop': drop, 'rms': rms,
            'bounded': bool(t2 >= end - 1e-9)}

def measure_file(path, f0, count=24):
    rate, x = load(path)
    reference = float(20.0 * np.log10(np.abs(x).max() + 1e-12))
    partials = []
    for k in range(1, count + 1):
        freq = f0 * k
        if freq > 0.42 * rate:
            break
        got = partial_t60(x, rate, freq, f0 * 0.32, reference)
        if got is None:
            continue
        got['k'] = k
        partials.append(got)
    return partials


def measure_model(directory, out):
    results = []
    for note, name in zip(PICK, NAMES):
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        partials = measure_file('%s/piano-%03d.wav' % (directory, note), f0)
        print(name, note, len(partials))
        results.append({'note': note, 'name': name, 'f0': f0,
                        'partials': partials})
    json.dump(results, open(out, 'w'), indent=1)


def measure_sample(directory, out):
    notes = json.load(open(directory + '/notes.json'))
    by_note = {}
    for name, f0, dur, midi in notes:
        if midi is None or (isinstance(midi, float) and np.isnan(midi)):
            continue
        key = int(round(midi))
        if key in PICK and key not in by_note:
            by_note[key] = (name, f0, dur)
    results = []
    for note, label in zip(PICK, NAMES):
        name, f0, dur = by_note[note]
        partials = measure_file('%s/%s' % (directory, name), f0)
        print(label, note, name, round(f0, 1), round(dur, 2), len(partials))
        results.append({'note': note, 'name': label, 'f0': f0,
                        'file': name, 'dur': dur, 'partials': partials})
    json.dump(results, open(out, 'w'), indent=1)


def plot(model_path, sample_path, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model = json.load(open(model_path))
    sample = json.load(open(sample_path))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for index, (mn, sn) in enumerate(zip(model, sample)):
        ax = axes.flat[index]
        for entry, color, label in ((mn, '#1f77b4', 'model'),
                                    (sn, '#d62728', 'sample')):
            ks = [p['k'] for p in entry['partials']]
            ts = [p['t60'] for p in entry['partials']]
            solid = [not p['bounded'] for p in entry['partials']]
            hollow = [not s for s in solid]
            ax.semilogy([k for k, m in zip(ks, solid) if m],
                        [t for t, m in zip(ts, solid) if m], 'o',
                        color=color, markersize=5, label=label)
            ax.semilogy([k for k, m in zip(ks, hollow) if m],
                        [t for t, m in zip(ts, hollow) if m], 'o',
                        color=color, markersize=5, markerfacecolor='none')
        duration = sn.get('dur')
        ax.set_title('%s  %.0f Hz   (sample %.1f s)' % (mn['name'], mn['f0'], duration))
        ax.grid(True, which='both', alpha=0.25)
        ax.set_xlabel('partial')
        ax.set_ylim(0.4, 30)
    axes.flat[0].set_ylabel('T60 of the prompt decay (s)')
    axes.flat[3].set_ylabel('T60 of the prompt decay (s)')
    summary = axes.flat[5]
    names = []
    ratios = []
    for mn, sn in zip(model, sample):
        ms = {p['k']: p['t60'] for p in mn['partials']}
        ss = {p['k']: p['t60'] for p in sn['partials'] if not p['bounded']}
        common = [k for k in sorted(set(ms) & set(ss))]
        if not common:
            continue
        ratio = np.exp(np.mean([np.log(ms[k] / ss[k]) for k in common]))
        names.append('%s\n%d' % (mn['name'], len(common)))
        ratios.append(ratio)
    summary.bar(names, ratios, color='#4c78a8')
    summary.axhline(1.0, color='#888', linewidth=1)
    summary.set_yscale('log')
    summary.set_ylabel('model / sample (T60)')
    summary.set_title('mean ratio over measured partials')
    summary.grid(True, axis='y', alpha=0.25)
    axes.flat[0].legend(loc='lower left')
    fig.suptitle('per-partial T60, fitted from -3 to -15 dB (hollow: clip ended first)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print('wrote', out_path)

def main():
    if sys.argv[1] == 'measure-model':
        measure_model(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'measure-sample':
        measure_sample(sys.argv[2], sys.argv[3])
    else:
        plot(sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == '__main__':
    main()