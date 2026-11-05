"""Fit the piano's decay anchors to the reference libraries.

The measurement the plot shows is the target: per-partial T60 of the prompt
decay. This script renders the preset with candidate anchors (through the
IAM_T60_* environment hook) and minimises the log distance to the libraries.
"""

import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

import t60_map as tm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
EXE = os.path.join(ROOT, 'target', 'release', 'examples', 'probe_piano.exe')
WORK = os.path.join(ROOT, 'target', 'wav', 'piano-fit')

NOTES = tm.PICK
NAMES = tm.NAMES

LIBRARIES = {
    'grand': 'target/wav/piano-t60/grand.json',
    'keyzone': 'target/wav/piano-t60/keyzone.json',
    'gentleman': 'target/wav/piano-t60/gentleman.json',
}
WEIGHT = {'grand': 1.0, 'keyzone': 1.0, 'gentleman': 1.0}

PRIOR = np.array([4.4, 3.2, 14.0, 0.45])
LOWER = np.array([1.2, 0.8, 2.0, 0.05])
UPPER = np.array([20.0, 12.0, 40.0, 3.0])
REGULARISATION = 0.03


def load_targets():
    sources = {key: json.load(open(os.path.join(ROOT, path)))
               for key, path in LIBRARIES.items()}
    targets = []
    for index, note in enumerate(NOTES):
        keys = {}
        for source in sources:
            for partial in sources[source][index]['partials']:
                if partial['bounded'] or partial['t60'] > 40.0:
                    continue
                keys.setdefault(partial['k'], []).append(
                    (WEIGHT[source], np.log(partial['t60'])))
        for k, values in keys.items():
            total = sum(w for w, _ in values)
            target = sum(w * value for w, value in values) / total
            targets.append((index, k, target))
    return targets


def render_and_measure(params, seconds=6):
    environment = os.environ.copy()
    environment['IAM_T60_A0'] = '%.4f' % params[0]
    environment['IAM_T60_C8'] = '%.4f' % params[1]
    environment['IAM_T60_HIGH_A0'] = '%.4f' % params[2]
    environment['IAM_T60_HIGH_C8'] = '%.4f' % params[3]
    arguments = [EXE, WORK, str(seconds)] + [str(note) for note in NOTES]
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True,
                   stdout=subprocess.DEVNULL)
    model = []
    for note in NOTES:
        path = os.path.join(WORK, 'piano-%03d.wav' % note)
        rate, x = tm.load(path)
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        model.append({p['k']: p['t60'] for p in tm.partials_for(x, rate, f0)})
    return model


EVALUATIONS = [0]


def loss(log_params, targets, verbose=False):
    params = np.exp(log_params)
    outside = np.maximum(LOWER - params, 0.0) + np.maximum(params - UPPER, 0.0)
    if np.any(outside > 0.0):
        return 1e3 + float(np.sum(outside))
    model = render_and_measure(params)
    total = 0.0
    used = 0
    per_note = [[] for _ in NOTES]
    for index, k, target in targets:
        value = model[index].get(k)
        if value is None or value > 40.0:
            # The model could not resolve it; count it as a miss but mildly.
            total += 0.25 / np.sqrt(k)
            used += 1
            continue
        error = np.log(value) - target
        weight = 1.0 / np.sqrt(k)
        total += weight * error * error
        per_note[index].append(np.exp(-error))
        used += 1
    prior = np.sum(((params - PRIOR) / PRIOR) ** 2) * REGULARISATION
    score = total / max(used, 1) + prior
    EVALUATIONS[0] += 1
    print('eval %3d  loss %.4f  %s' % (EVALUATIONS[0], score,
        ' '.join('%.3f' % v for v in params)), flush=True)
    if verbose:
        print('  loss %.4f  phase %.4f  partials %d' % (score, prior, used))
    return score


def report(params, targets, out_path):
    model = render_and_measure(params)
    print('params', ['%.3f' % value for value in params])
    for index, name in enumerate(NAMES):
        line = '%s %6.1f Hz  model %s' % (name, NOTES[index],
                                          ' '.join('%d:%.1f' % (k, v)
                                                   for k, v in sorted(model[index].items())))
        print(line)
    sources = {key: json.load(open(os.path.join(ROOT, path)))
               for key, path in LIBRARIES.items()}
    for key in LIBRARIES:
        ratios = []
        for index in range(len(NOTES)):
            other = {p['k']: p['t60'] for p in sources[key][index]['partials']
                     if not p['bounded']}
            common = sorted(set(model[index]) & set(other))
            ratios.extend(np.log(model[index][k] / other[k]) for k in common)
        if ratios:
            print('  model/%s geo %.2f' % (key, np.exp(np.mean(ratios))))


def main():
    targets = load_targets()
    print('targets', len(targets))
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report(PRIOR, targets, None)
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        report(np.array([float(v) for v in sys.argv[2:6]]), targets, None)
        return
    start = np.log(PRIOR if len(sys.argv) <= 1 else np.array([float(v) for v in sys.argv[1:5]]))
    print('start', np.exp(start))
    loss(start, targets, verbose=True)
    result = minimize(loss, start, args=(targets,), method='Nelder-Mead',
                      options={'maxiter': 90, 'xatol': 5e-3, 'fatol': 5e-4,
                               'adaptive': True})
    print(result.message)
    print('fitted', ' '.join('%.4f' % v for v in np.exp(result.x)), 'loss %.4f' % result.fun)
    report(np.exp(result.x), targets, None)


if __name__ == '__main__':
    main()