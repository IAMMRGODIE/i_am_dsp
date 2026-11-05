import pathlib

def drop_func(path, marker):
    lines = pathlib.Path(path).read_text().split(chr(10))
    i = next(k for k, l in enumerate(lines) if marker in l)
    j = i
    while j > 0:
        s = lines[j - 1].strip()
        if s.startswith('///') or s.startswith('#[') or s.startswith('//'):
            j -= 1
        else:
            break
    depth = 0
    k = i
    started = False
    while k < len(lines):
        depth += lines[k].count('{') - lines[k].count('}')
        if '{' in lines[k]:
            started = True
        if started and depth <= 0:
            break
        k += 1
    del lines[j:k + 1]
    pathlib.Path(path).write_text(chr(10).join(lines))
    print('dropped', marker, 'in', path)

def drop_lines(path, needles):
    lines = pathlib.Path(path).read_text().split(chr(10))
    keep = [l for l in lines if not any(n in l for n in needles)]
    pathlib.Path(path).write_text(chr(10).join(keep))
    print('lines', path, len(lines) - len(keep))

base = 'i_am_string/src/'
drop_func(base + 'core.rs', 'pub fn recalibrate')
drop_func(base + 'modal.rs', 'pub fn recalibrate')
drop_lines(base + 'core.rs', ['calibration: config.calibration,', 'if !config.calibration.is_empty() {', 'string.recalibrate(config.calibration);'])
drop_lines(base + 'engine.rs', ['pub calibrated: bool,', 'calibrated: false,'])
drop_lines(base + 'instrument.rs', ['calibration: &[],'])
drop_lines(base + 'lib.rs', ['pub mod piano_calibration;'])
drop_lines(base + 'modal.rs', ['pub calibration:', 'calibration: &[],'])
drop_lines(base + 'string.rs', ['pub calibration:', 'calibration: &[],'])
drop_lines('i_am_dsp/src/generators/string_engine.rs', ['calibrated: false,'])
