import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t60_map as tm
from calibrate_same import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.abspath(os.path.join(ROOT, '..', 'refer', 'sample', 'The Gentleman'))
MODES = os.path.join(ROOT, 'target', 'wav', 'gentleman-model', 'modes.json')

library = json.load(open(MODES))
for tag, directory in (('bank=table', 'cal-test'),):
    gains, t60s = [], []
    for entry in library:
        note = entry['note']
        rate, engine = tm.load(os.path.join(ROOT, 'target', 'wav', directory, 'piano-%03d.wav' % note))
        _, target = tm.load(os.path.join(LIBRARY, entry['file']))
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        rows = []
        for mode in entry['modes']:
            a = measure(target, rate, mode['f'], f0)
            b = measure(engine, rate, mode['f'], f0)
            if a and b:
                rows.append((mode['f'], a, b))
        if len(rows) < 3:
            continue
        pk_a = max(r[1]['amp'] for r in rows)
        pk_b = max(r[2]['amp'] for r in rows)
        gains.extend(np.log([(r[1]['amp'] / pk_a) / (r[2]['amp'] / pk_b) for r in rows]))
        t60s.extend(np.log([r[1]['tail'] / r[2]['tail'] for r in rows]))
    print('%-7s gain 10-90%% %4.2f..%4.2f  tail 10-90%% %4.2f..%4.2f' % (
          tag, float(np.exp(np.percentile(gains, 10))), float(np.exp(np.percentile(gains, 90))),
          float(np.exp(np.percentile(t60s, 10))), float(np.exp(np.percentile(t60s, 90)))))