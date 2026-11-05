import sys

def patch(path, pairs):
    text = open(path).read()
    for old, new in pairs:
        if old not in text:
            print('MISS', path, old[:40])
        text = text.replace(old, new)
    open(path, 'w').write(text)
    print('patched', path)

base = 'i_am_string/src/'

patch(base + 'body.rs', [
    ("			Preset::Piano => (piano_table(), 0.35, 32.0, 9000.0, 14.0),
", ''),
    ("			Preset::Guitar => (GUITAR.to_vec(), 0.22, 72.0, 7500.0, 12.5),
", ''),
])
patch(base + 'engine.rs', [
    ("		let preset = Preset::Piano;", "		let preset = Preset::Violin;"),
    ("		let expressive = self.preset != Preset::Piano;", "		let expressive = true;"),
])
patch(base + 'instrument.rs', [
    ("			if matches!(self, Preset::Guitar) {", "			if false {"),
])
patch(base + 'core.rs', [
    ("					calibration: config.calibration,
", ''),
    ("				if !config.calibration.is_empty() {
					string.recalibrate(config.calibration);
				}
", ''),
])
patch(base + 'string.rs', [
    ("	pub calibration: &'static [crate::piano_calibration::Mode],
", ''),
    ("			calibration: &[],
", ''),
])
patch(base + 'modal.rs', [
    ("	pub calibration: &'static [crate::piano_calibration::Mode],
", ''),
    ("			calibration: &[],
", ''),
    ("			let mut level = 1.0f64;
			let mut tail = voicing.late_decay;
			if let Some(row) = config.calibration.iter().find(|row| {
				(row.frequency as f64 - frequency).abs() < 0.02 * frequency
			}) {
				level = row.gain.max(0.0) as f64;
				sigma = 6.907_755 / row.prompt.clamp(0.03, 20.0) as f64;
				tail = (row.prompt / row.tail.clamp(0.03, 60.0)).clamp(0.05, 1.0);
			}
			built.push((n, frequency, level, sigma, tail, false));",
     "			built.push((n, frequency, 1.0, sigma, voicing.late_decay, false));"),
])
patch(base + 'lib.rs', [('pub mod piano_calibration;
', '')])
