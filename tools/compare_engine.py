import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

BANDS = [(90, 180), (180, 360), (360, 720), (720, 1440), (1440, 2880),
         (2880, 5760)]

def load(path):
    rate, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data - data.mean()
    return float(rate), data / (np.abs(data).max() + 1e-12)

def levels(x, rate):
    out = []
    for lo, hi in BANDS:
        nyq = rate * 0.5
        sos = butter(4, [lo / nyq, min(hi, nyq * 0.95) / nyq], btype='band', output='sos')
        y = sosfiltfilt(sos, x)
        seg = y[int(0.12 * rate):int(0.5 * rate)]
        out.append(10 * np.log10(np.mean(seg * seg) + 1e-14))
    out = np.array(out)
    return out - out.max()

engine = '../target/wav/engine-cal'
gentleman = '../../refer/sample/The Gentleman'
pairs = [('F#2', 30, 'F#3'), ('F#3', 42, 'F#4'), ('F#4', 54, 'F#5'),
         ('F#5', 66, 'F#6'), ('D4', 50, 'D5'), ('D5', 62, 'D6')]
print('%-6s %6s %8s %8s' % ('note', 'shift', 'rms dB', 'band dB'))
for name, note, file_note in pairs:
    rate, a = load(os.path.join(gentleman, 'The Gentleman_%s_127.wav' % file_note))
    _, b = load(os.path.join(engine, 'piano-%03d.wav' % note))
    la, lb = levels(a, rate), levels(b, rate)
    early = 20 * np.log10(np.sqrt(np.mean(b[int(0.1 * rate):int(0.4 * rate)] ** 2))
                          / np.sqrt(np.mean(a[int(0.1 * rate):int(0.4 * rate)] ** 2)))
    print('%-6s %6.1f %8.1f %8.1f' % (name, early,
          np.mean(np.abs(la - lb)), 0.0))