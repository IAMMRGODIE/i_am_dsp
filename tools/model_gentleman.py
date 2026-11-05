"""A modal model of The Gentleman: analysis, resynthesis, and interpolation.

Every sample is decomposed into modes, each a decaying sinusoid whose decay is
fitted where the note itself lives - roughly the first half second. The clip's
second half is room tone, and fitting a mode to it was what made the first
attempts ring for two seconds in a way the samples do not.

    python tools/model_gentleman.py analyse
    python tools/model_gentleman.py self
    python tools/model_gentleman.py interp
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

import t60_map as tm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
LIBRARY = os.path.join(ROOT, '..', 'refer', 'sample', 'The Gentleman')
OUT = os.path.join(ROOT, 'target', 'wav', 'gentleman-model')
MODES = os.path.join(OUT, 'modes.json')

F_MAX = 9000.0
FLOOR = 55.0
END = 0.55
LN1000 = 6.907755


def note_name(midi):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return '%s%d' % (names[midi % 12], midi // 12 - 1)


def parse_note(name):
    body = name.split('_')[-2]
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if body[1] == '#':
        pitch, octave = body[:2], int(body[2:])
    else:
        pitch, octave = body[0], int(body[1:])
    named = (octave + 1) * 12 + names.index(pitch)
    return named - 12


def detect_modes(x, rate, f0):
    start = int(0.06 * rate)
    size = int(0.9 * rate)
    seg = x[start:start + size] * np.hanning(size)
    spectrum = np.abs(np.fft.rfft(seg, 8 * size)) / size
    freqs = np.fft.rfftfreq(8 * size, 1.0 / rate)
    db = 20.0 * np.log10(spectrum + 1e-12)
    peak_db = float(db.max())
    distance = max(3, int(0.30 * f0 / (freqs[1] - freqs[0])))
    index, _ = find_peaks(db, height=peak_db - FLOOR, distance=distance)

    times = np.arange(len(x)) / rate
    span = max(3, int(rate * 0.7 / f0))
    kernel = np.hanning(span)
    kernel /= kernel.sum()

    raw = {}
    for i in index:
        if not (25.0 < freqs[i] < F_MAX):
            continue
        left, right = db[i - 1], db[i + 1]
        offset = 0.5 * (left - right) / (left - 2.0 * db[i] + right)
        f = float(freqs[i] + offset * (freqs[1] - freqs[0]))
        k = int(round(f / f0))
        if k < 1 or abs(f / f0 - k) > 0.2 or k in raw:
            continue
        envelope = np.abs(np.convolve(x * np.exp(-2j * np.pi * f * times),
                                      kernel, mode='same'))
        early = (times > 0.05) & (times < END)
        if early.sum() < 30:
            continue
        slope, intercept = np.polyfit(times[early],
                                      20.0 * np.log10(envelope[early] + 1e-12), 1)
        if slope >= -0.2:
            continue
        t60 = 60.0 / -slope
        if not (0.08 < t60 < 20.0):
            continue
        raw[k] = {'k': k, 'f': f, 't60': float(t60)}

    if not raw:
        return []
    ordered = [raw[key] for key in sorted(raw)]
    columns = []
    for mode in ordered:
        decay = LN1000 / mode['t60']
        omega = 2.0 * np.pi * mode['f']
        envelope = np.exp(-decay * times)
        columns.append(np.cos(omega * times) * envelope)
        columns.append(np.sin(omega * times) * envelope)
    design = np.vstack(columns).T
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    for index, mode in enumerate(ordered):
        a = coef[2 * index]
        b = coef[2 * index + 1]
        mode['amp'] = float(np.hypot(a, b))
        mode['phase'] = float(-np.arctan2(b, a))
    return ordered


def analyse():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for path in sorted(glob.glob(os.path.join(LIBRARY, '*.wav'))):
        name = os.path.basename(path)
        note = parse_note(name)
        rate, x = tm.load(path)
        if np.abs(x).max() < 1e-3:
            continue
        expected = 440.0 * 2.0 ** ((note - 69) / 12.0)
        f0 = tm.estimate_f0(x, rate, expected)
        modes = detect_modes(x, rate, f0)
        print('%-42s note %3d f0 %6.1f  modes %3d' % (name, note, f0, len(modes)))
        results.append({'file': name, 'note': note, 'f0': f0, 'rate': rate,
                        'duration': len(x) / rate, 'modes': modes})
    json.dump(results, open(MODES, 'w'), indent=1)
    print('wrote', MODES)


def render(modes, rate, seconds):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    signal = np.zeros(n)
    for mode in modes:
        decay = LN1000 / max(mode['t60'], 0.05)
        signal += mode['amp'] * np.cos(2.0 * np.pi * mode['f'] * t + mode['phase']) \
            * np.exp(-decay * t)
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * 0.89
    return signal


def write_wav(path, rate, signal):
    data = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, int(rate), data)


def run_self():
    results = json.load(open(MODES))
    directory = os.path.join(OUT, 'self')
    os.makedirs(directory, exist_ok=True)
    for entry in results:
        signal = render(entry['modes'], entry['rate'], entry['duration'])
        name = os.path.basename(entry['file']).replace('The Gentleman_', 'self_')
        write_wav(os.path.join(directory, name), entry['rate'], signal)
    print('wrote', directory)


def table(results):
    keys = {}
    for entry in results:
        for mode in entry['modes']:
            keys.setdefault(mode['k'], []).append((entry['f0'], mode))
    for key in keys:
        keys[key].sort(key=lambda row: row[0])
    return keys


def interpolate(keys, f0):
    x = np.log(f0)
    modes = []
    for k, rows in keys.items():
        xs = np.log([row[0] for row in rows])
        if x < xs[0] or x > xs[-1]:
            continue
        nearest = rows[int(np.argmin(np.abs(xs - x)))][1]
        f = float(np.exp(np.interp(x, xs, [np.log(row[1]['f']) for row in rows])))
        t60 = float(np.exp(np.interp(x, xs, [np.log(row[1]['t60']) for row in rows])))
        amp = float(np.exp(np.interp(x, xs, [np.log(row[1]['amp'] + 1e-9)
                                              for row in rows])))
        modes.append({'f': f, 't60': t60, 'amp': amp,
                      'phase': nearest['phase']})
    return modes


def run_interp():
    results = json.load(open(MODES))
    keys = table(results)
    rate = results[0]['rate']
    directory = os.path.join(OUT, 'interpolated')
    os.makedirs(directory, exist_ok=True)
    for note in range(38, 86):
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        modes = interpolate(keys, f0)
        if not modes:
            continue
        signal = render(modes, rate, 1.9)
        write_wav(os.path.join(directory, 'interp_%03d_%s.wav' % (note, note_name(note))),
                  rate, signal)
    print('wrote', directory)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else 'analyse'
    {'analyse': analyse, 'self': run_self, 'interp': run_interp}[command]()


if __name__ == '__main__':
    main()