"""Per-partial T60 of the piano model and of the reference samples.

    python tools/t60_map.py measure-model target/wav/piano-t60 model.json
    python tools/t60_map.py measure-grand   target/wav/ref-grand grand.json
    python tools/t60_map.py measure-library ../../refer/sample/'Keyzone Classic' "Keyzone Classic" keyzone.json
    python tools/t60_map.py plot model.json grand.json keyzone.json out.png

Renders the model with `cargo run -p i_am_string --example probe_piano -- DIR 30 42 54 66 78 90`.
"""

import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

PICK = [42, 54, 66, 78, 90]
NAMES = ['F#2', 'F#3', 'F#4', 'F#5', 'F#6']

T_MAX = 2.0        # seconds of decay used for the rate estimate
SPAN = 12.0        # dB down from the partial's own peak: the prompt decay
FLOOR = 52.0       # dB below the note's peak: never fit into the noise
WEAK = 40.0        # a partial more than this below the note's peak is noise


def load(path):
    rate, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
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
    for _ in range(3):
        resid = db - (slope * t + offset)
        keep = resid > -1.0
        if keep.sum() < 8:
            return slope, offset, np.ones(len(t), dtype=bool), 0.0
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
    bounded = t2 is None or t2 > end
    if bounded:
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
            'bounded': bool(bounded)}


def estimate_f0(x, rate, expected):
    """The sample's own fundamental, within a quarter tone of the nominal."""
    n = min(len(x), int(0.36 * rate))
    spectrum = np.abs(np.fft.rfft(x[:n] * np.hanning(n), 8 * n))
    size = 8 * n
    best = (0.0, expected)
    for cents in np.arange(-250.0, 250.0, 5.0):
        f = expected * 2.0 ** (cents / 1200.0)
        score = 0.0
        for k in range(1, 7):
            position = k * f * size / rate
            index = int(round(position))
            if 1 <= index < len(spectrum) - 1:
                score += float(spectrum[index - 1:index + 2].max())
        if score > best[0]:
            best = (score, f)
    return float(best[1])


def partials_for(x, rate, f0):
    reference = float(20.0 * np.log10(np.abs(x).max() + 1e-12))
    partials = []
    for k in range(1, 25):
        freq = f0 * k
        if freq > 0.42 * rate:
            break
        got = partial_t60(x, rate, freq, f0 * 0.32, reference)
        if got is None:
            continue
        got['k'] = k
        partials.append(got)
    return partials


def note_name(midi):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return '%s%d' % (names[midi % 12], midi // 12 - 1)


def measure_model(directory, out):
    results = []
    for note, name in zip(PICK, NAMES):
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        rate, x = load('%s/piano-%03d.wav' % (directory, note))
        partials = partials_for(x, rate, f0)
        print(name, note, len(partials))
        results.append({'note': note, 'name': name, 'f0': f0,
                        'partials': partials})
    json.dump(results, open(out, 'w'), indent=1)


def measure_grand(directory, out):
    notes = json.load(open(directory + '/notes.json'))
    by_note = {}
    for name, f0, dur, midi in notes:
        if midi is None or (isinstance(midi, float) and np.isnan(midi)):
            continue
        key = int(round(midi))
        if key not in by_note:
            by_note[key] = (name, f0, dur)
    results = []
    for note, label in zip(PICK, NAMES):
        name, f0, dur = by_note[note]
        rate, x = load('%s/%s' % (directory, name))
        partials = partials_for(x, rate, f0)
        print(label, note, name, round(f0, 1), round(dur, 2), len(partials))
        results.append({'note': note, 'name': label, 'f0': f0,
                        'file': name, 'dur': dur, 'partials': partials})
    json.dump(results, open(out, 'w'), indent=1)


def measure_library(directory, prefix, out):
    """Samples named `<prefix>_<note>_127.wav`, one octave above the pitch."""
    results = []
    for note, label in zip(PICK, NAMES):
        name = '%s_%s_127.wav' % (prefix, note_name(note + 12))
        rate, x = load(os.path.join(directory, name))
        expected = 440.0 * 2.0 ** ((note - 69) / 12.0)
        f0 = estimate_f0(x, rate, expected)
        dur = len(x) / rate
        partials = partials_for(x, rate, f0)
        print(label, note, name, round(f0, 1), round(dur, 2), len(partials))
        results.append({'note': note, 'name': label, 'f0': f0,
                        'file': name, 'dur': dur, 'partials': partials})
    json.dump(results, open(out, 'w'), indent=1)


SOURCES = [('model', '#1f77b4'), ('grand', '#d62728'),
           ('keyzone', '#2ca02c'), ('gentleman', '#ff7f0e')]


def plot(paths, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {label: json.load(open(path)) for label, path in paths.items()}
    available = [(k, c) for k, c in SOURCES if k in data]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for index in range(5):
        ax = axes.flat[index]
        for key, color in available:
            entry = data[key][index]
            solid = [p for p in entry['partials'] if not p['bounded']]
            hollow = [p for p in entry['partials'] if p['bounded']]
            ax.semilogy([p['k'] for p in solid], [p['t60'] for p in solid],
                        'o', color=color, markersize=5, label=key)
            ax.semilogy([p['k'] for p in hollow], [p['t60'] for p in hollow],
                        'o', color=color, markersize=5, markerfacecolor='none')
        durations = ', '.join('%s %.1fs' % (k, data[k][index]['dur'])
                              for k, _ in available if 'dur' in data[k][index])
        ax.set_title('%s  %.0f Hz\n%s' % (data['model'][index]['name'],
                                           data['model'][index]['f0'],
                                           durations), fontsize=9)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_xlabel('partial')
        ax.set_ylim(0.4, 30)
    axes.flat[0].set_ylabel('T60 of the prompt decay (s)')
    axes.flat[3].set_ylabel('T60 of the prompt decay (s)')
    axes.flat[0].legend(loc='lower left', fontsize=8)

    summary = axes.flat[5]
    keys = [k for k, _ in available if k != 'model']
    width = 0.8 / max(len(keys), 1)
    for offset, key in enumerate(keys):
        ratios = []
        for index in range(5):
            model = {p['k']: p['t60'] for p in data['model'][index]['partials']}
            other = {p['k']: p['t60'] for p in data[key][index]['partials']
                     if not p['bounded']}
            common = sorted(set(model) & set(other))
            ratios.append(np.exp(np.mean([np.log(model[k] / other[k])
                                          for k in common])) if common else np.nan)
        x = np.arange(5) + (offset - (len(keys) - 1) / 2.0) * width
        summary.bar(x, ratios, width, label=key,
                    color=dict(SOURCES).get(key))
    summary.axhline(1.0, color='#666', linewidth=1)
    summary.set_yscale('log')
    summary.set_xticks(np.arange(5))
    summary.set_xticklabels(NAMES)
    summary.set_ylabel('model / sample (T60)')
    summary.set_title('mean ratio over measured partials')
    summary.legend(fontsize=7)
    summary.grid(True, axis='y', alpha=0.25)

    fig.suptitle('per-partial T60, fitted from -3 to -15 dB (hollow: clip ended first)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print('wrote', out_path)


def main():
    command = sys.argv[1]
    if command == 'measure-model':
        measure_model(sys.argv[2], sys.argv[3])
    elif command == 'measure-grand':
        measure_grand(sys.argv[2], sys.argv[3])
    elif command == 'measure-library':
        measure_library(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        pairs = sys.argv[2:-1]
        paths = {pair.split('=')[0]: pair.split('=')[1] for pair in pairs}
        plot(paths, sys.argv[-1])


if __name__ == '__main__':
    main()