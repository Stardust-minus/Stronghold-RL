"""FIN (11-13): compare two runs' train_ppo argument lists (meta.json 'args') after argparse (train_ppo.py's own parser).
usage: argcmp.py <tree>/tools/rl/py/train_ppo.py <ref meta.json> <new meta.json>"""
import json, sys, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
src = open(sys.argv[1]).read(); cut = src.index('a = ap.parse_args()')
g = {'__name__': 'argcmp', '__file__': sys.argv[1]}; sys.argv_ = sys.argv; sys.argv = ['train_ppo.py']
exec(compile(src[:cut], 'train_ppo_head', 'exec'), g)
ap = g['ap']; ref, new = (json.load(open(p))['args'] for p in sys.argv_[2:4])
X = ['--init', '-', '--out', '-']; A, Bn = vars(ap.parse_args(X + ref)), vars(ap.parse_args(X + new))
for k in sorted(set(A) | set(Bn)):
    if A.get(k) != Bn.get(k): print(f'{k}: {A.get(k)!r} -> {Bn.get(k)!r}')
print('compared', len(A), 'options')
