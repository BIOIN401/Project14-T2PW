p = 'C:/t/corr-assess/scripts/a1_check.py'
lines = open(p, encoding='utf-8', newline='').read().split('\n')
for i, l in enumerate(lines):
    if l.lstrip().startswith('na = lambda s:'):
        lines[i] = '    na = lambda s: "".join(ch if ord(ch) < 128 else "?" for ch in n(s))  # mysql client returned non-ascii as ?'
open(p, 'w', encoding='utf-8', newline='').write('\n'.join(lines))
print('ok')
