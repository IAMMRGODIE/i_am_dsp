import pathlib
p = pathlib.Path('i_am_string/src/body.rs')
lines = p.read_text().split(chr(10))
start = next(i for i, l in enumerate(lines) if 'fn piano_table()' in l)
while start > 0 and lines[start - 1].strip().startswith('///'):
    start -= 1
end = next(i for i, l in enumerate(lines) if i > start and l.strip() == '];' and 'GUITAR' in chr(10).join(lines[:i])[-2000:])
del lines[start:end + 1]
p.write_text(chr(10).join(lines))
print('removed', start, end)
