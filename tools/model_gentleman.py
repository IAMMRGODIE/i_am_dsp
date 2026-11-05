"""A modal model of The Gentleman: analysis, resynthesis, and interpolation.

Every sample is decomposed into modes, each with two decays - a prompt one and
a tail - because a piano partial does not fall at a single rate. Modes are
picked from the spectrum without asking them to be harmonic: a sample's fine
partials and its room are not, and throwing them away was what made the first
version sound empty.

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
FLOOR = 70.0
MAX_MODES = 90
PROMPT_END = 0.30
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


def decay_fit(times, envelope, low, high):
    keep = (times > low) & (times < high)
    if keep.sum() < 12:
        return None
    slope, intercept = np.polyfit(times[keep],
                                  20.0 * np.log10(envelope[keep] + 1e-12), 1)
    if slope >= -0.5:
        return None
    t60 = 60.0 / -slope
    if not (0.05 < t60 < 30.0):
        return None
    return t60, float(10.0 ** (intercept / 20.0)), float(slope)


def detect_modes(x, rate, f0):
    start = int(0.06 * rate)
    size = int(0.9 * rate)
    seg = x[start:start + size] * np.hanning(size)
    spectrum = np.abs(np.fft.rfft(seg, 8 * size)) / size
    freqs = np.fft.rfftfreq(8 * size, 1.0 / rate)
    db = 20.0 * np.log10(spectrum + 1e-12)
    peak_db = float(db.max())
    step = freqs[1] - freqs[0]
    distance = max(4, int(max(8.0, 0.10 * f0) / step))
    index, _ = find_peaks(db, height=peak_db - FLOOR, prominence=6.0,
                          distance=distance)
    order = np.argsort(db[index])[::-1][:MAX_MODES]
    index = np.sort(index[order])

    times = np.arange(len(x)) / rate
    span = max(3, int(rate * 0.5 / f0))
    kernel = np.hanning(span)
    kernel /= kernel.sum()

    raw = []
    for i in index:
        if not (25.0 < freqs[i] < F_MAX):
            continue
        left, right = db[i - 1], db[i + 1]
        offset = 0.5 * (left - right) / (left - 2.0 * db[i] + right)
        f = float(freqs[i] + offset * step)
        envelope = np.abs(np.convolve(x * np.exp(-2j * np.pi * f * times),
                                      kernel, mode='same'))
        floor = float(np.percentile(envelope[times > 1.2], 25)) \
            if (times > 1.2).any() else 0.0
        peak_time = float(times[int(np.argmax(envelope))])
        tail = decay_fit(times, envelope, 0.35, min(1.85, times[-1] - 0.05))
        if tail is None:
            continue
        tail_t60, tail_amp, slope = tail
        if envelope[times > 1.2].size and tail_amp < floor * 2.0:
            tail_amp = 0.0
        prompt = decay_fit(times, envelope, 0.03, PROMPT_END)
        prompt_t60 = prompt[0] if prompt is not None else 0.10
        raw.append({'f': f, 'k': int(round(f / f0)),
                    'prompt_t60': float(min(prompt_t60, 0.25)),
                    'tail_t60': float(tail_t60),
                    'tail_amp': float(tail_amp),
                    'prompt_amp': 0.0, 'prompt_phase': 0.0,
                    'tail_phase': 0.0, 'peak': float(db[i])})

    if not raw:
        return []
    columns = []
    for mode in raw:
        for t60 in (mode['prompt_t60'], mode['tail_t60']):
            omega = 2.0 * np.pi * mode['f']
            envelope = np.exp(-LN1000 / t60 * times)
            columns.append(np.cos(omega * times) * envelope)
            columns.append(np.sin(omega * times) * envelope)
    design = np.vstack(columns).T
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    for index, mode in enumerate(raw):
        a, b, c, d = coef[4 * index:4 * index + 4]
        mode['prompt_amp'] = float(np.hypot(a, b))
        mode['prompt_phase'] = float(-np.arctan2(b, a))
        mode['tail_amp'] = float(np.hypot(c, d))
        mode['tail_phase'] = float(-np.arctan2(d, c))
    return raw


def render(modes, rate, seconds):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    signal = np.zeros(n)
    for mode in modes:
        omega = 2.0 * np.pi * mode['f']
        if mode['prompt_amp'] > 0.0:
            signal += mode['prompt_amp'] * np.cos(omega * t + mode['prompt_phase']) \
                * np.exp(-LN1000 / mode['prompt_t60'] * t)
        if mode['tail_amp'] > 0.0:
            signal += mode['tail_amp'] * np.cos(omega * t + mode['tail_phase']) \
                * np.exp(-LN1000 / mode['tail_t60'] * t)
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * 0.89
    return signal


def write_wav(path, rate, signal):
    data = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, int(rate), data)


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
            if mode['k'] < 1 or abs(mode['f'] - mode['k'] * entry['f0']) > 0.15 * entry['f0']:
                continue
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
        def lerp(name):
            return float(np.exp(np.interp(x, xs, [np.log(max(row[1][name], 1e-9))
                                                    for row in rows])))
        if min(row[1]['tail_amp'] for row in rows) <= 0.0:
            continue
        modes.append({'f': lerp('f'), 'prompt_t60': lerp('prompt_t60'),
                      'prompt_amp': lerp('prompt_amp'),
                      'prompt_phase': nearest['prompt_phase'],
                      'tail_t60': lerp('tail_t60'),
                      'tail_amp': lerp('tail_amp'),
                      'tail_phase': nearest['tail_phase']})
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