import sys

path = sys.argv[1]
lines = open(path).read().split('\n')
out = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'Preset::Piano =>' in line or 'Preset::Guitar =>' in line:
        while i < len(lines) and 'Preset::Violin =>' not in lines[i]:
            i += 1
        continue
    out.append(line)
    i += 1
open(path, 'w').write('\n'.join(out))
print('stripped', path)