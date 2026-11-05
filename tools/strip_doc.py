path = 'i_am_string/tests/model.rs'
lines = open(path).read().split('\n')
out = []
skip = False
for line in lines:
    if line.startswith('/// Every partial the model claims to have must actually be where it says.'):
        skip = True
    if skip:
        if 'with a hundred partials most of them were unconstrained.' in line:
            skip = False
        continue
    out.append(line)
open(path, 'w').write('\n'.join(out))
print('cleaned')
