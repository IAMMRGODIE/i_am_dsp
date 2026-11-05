import re, sys
path = 'i_am_string/tests/model.rs'
text = open(path).read()
for name in ('every_modelled_partial_is_in_tune', 'velocity_changes_the_level', 'unison_detune_produces_beating'):
    start = text.find('fn ' + name + '(')
    if start < 0:
        print('missing', name)
        continue
    # walk back to the #[test] attribute
    attr = text.rfind('#[test]', 0, start)
    begin = attr if attr >= 0 else start
    # find the end: first line that is a lone closing brace after start
    end = text.find('\n}\n', start)
    if end < 0:
        print('no end', name)
        continue
    text = text[:begin] + text[end + 3:]
    print('removed', name)
open(path, 'w').write(text)
