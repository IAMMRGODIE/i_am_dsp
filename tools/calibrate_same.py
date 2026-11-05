"""Compare the engine and The Gentleman at the same frequencies, the same way.

Both signals are projected onto the sample's own mode frequencies with the
same kernel, so leakage and windowing cancel in the ratio. What is left is how
much the engine has of that partial, and how fast it decays.
"""

import json
import os
import sys

import numpy as np

import t60_map as tm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
ENGINE = os.path.join(ROOT, 'target', 'wav', 'engine-cal')
LIBRARY = os.path.abspath(os.path.join(ROOT, '..', 'refer', 'sample', 'The Gentleman'))
MODES = os.path.join(ROOT, 'target', 'wav', 'gentleman-model', 'modes.json')


def measure(x, rate, f, f0):
    times = np.arange(len(x)) / rate
    span = max(3, int(rate * 0.5 / f0))
    kernel = np.hanning(span)
    kernel /= kernel.sum()
    envelope = np.abs(np.convolve(x * np.exp(-2j * np.pi * f * times),
                                  kernel, mode='same'))
    peak = float(envelope.max())
    if peak < 1e-7:
        return None
    floor = float(np.percentile(envelope[times > 1.2], 25)) \
        if (times > 1.2).any() else 0.0
    prompt = (times > 0.03) & (times < 0.25)
    tail = (times > 0.35) & (envelope > floor * 2.0)
    prompt_slope = np.polyfit(times[prompt],
                              20 * np.log10(envelope[prompt] + 1e-12), 1)[0]
    tail_slope = np.polyfit(times[tail],
                            20 * np.log10(envelope[tail] + 1e-12), 1)[0] \
        if tail.sum() > 10 else prompt_slope
    return {'amp': peak, 'prompt': 60.0 / max(-prompt_slope, 0.5),
            'tail': 60.0 / max(-tail_slope, 0.5)}


def main():
    library = json.load(open(MODES))
    ratios = []
    for entry in library:
        note = entry['note']
        path = os.path.join(ENGINE, 'piano-%03d.wav' % note)
        if not os.path.exists(path):
            continue
        rate, engine = tm.load(path)
        _, target = tm.load(os.path.join(LIBRARY, entry['file']))
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        rows = []
        for mode in entry['modes']:
            a = measure(target, rate, mode['f'], f0)
            b = measure(engine, rate, mode['f'], f0)
            if a is None or b is None:
                continue
            rows.append((mode['f'], a, b))
        if len(rows) < 3:
            continue
        target_peak = max(row[1]['amp'] for row in rows)
        engine_peak = max(row[2]['amp'] for row in rows)
        gains = np.array([(row[1]['amp'] / target_peak) / (row[2]['amp'] / engine_peak)
                          for row in rows])
        t60 = np.array([row[1]['tail'] / row[2]['tail'] for row in rows])
        ratios.extend(np.log(gains))
        print('note %3d  modes %2d  gain %5.2f..%5.2f med %4.2f   tail ratio %4.2f..%4.2f med %4.2f' % (
              note, len(rows), gains.min(), gains.max(), np.median(gains),
              t60.min(), t60.max(), np.median(t60)))
    print('overall gain median %.2f, 10-90%% %.2f..%.2f' % (
          float(np.exp(np.median(ratios))),
          float(np.exp(np.percentile(ratios, 10))),
          float(np.exp(np.percentile(ratios, 90)))))


if __name__ == '__main__':
    main()