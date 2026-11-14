"""FIN finsum (11-13): milestone table of the FINAL runs (evals/out/<run>-<M>M-<a|b>-<solo|coop>.json): win, paired delta vs NMA 30M
(same setting, same seeds: a = the NMA ev11 NMA-30000000, b = the picker c9 + xboss-v3 NMA 30M run, reproduced code-A7 by gate.sh),
picker lift b - a for the same checkpoint, and the stacking block (stackrep.summarize: natMain, natMainT2, main tier, layers, elites)."""
import json, sys, os, glob, re
import numpy as np
B = os.environ['B']  # the work dir; E = B + '/FINAL/evals'
sys.path.insert(0, B + '/code-A6/tools/rl/py')
from stackrep import summarize
REF = {('a', 'solo'): B + '/N/ev11/NMA-30000000-solo480.json', ('a', 'coop'): B + '/N/ev11/NMA-30000000-coop4h1-480.json',
       ('b', 'solo'): E + '/ref/nma-P9pool-solo.json', ('b', 'coop'): E + '/ref/nma-P9pool-coop4h1.json'}
load = lambda p: list(json.load(open(p))['configs'].values())[0]
wins = lambda c: {m['seed']: m['win'] for m in c['perMatch']}
def delta(c, r):
    w, w0 = wins(c), wins(r); s = sorted(set(w) & set(w0)); d = np.array([w[x] - w0[x] for x in s], float)
    return f'{d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}' if len(d) > 1 else 'n/a'
def row(label, ck, st, cfg, c, extra):
    r = summarize(c['perMatch'], c['teamWin'])
    top = ', '.join(f'{k} {v:.2f}' for k, v in r['topMain'][:2]); sec = ', '.join(f'{k} {v:.2f}' for k, v in r['topSec'][:2])
    return (f"| {label} | {ck} | {st} | {cfg} | {c['teamWin']:.3f} | {extra} | {r['natMain']:.2f} | {r['natMainT2']:.2f} | {r['mainTier']:.2f} | "
            f"{r['mainLayers']:.0f} / {r['secLayers']:.0f} / {r['sumLayers']:.0f} | {r['elites']:.1f} | {top} | {sec} |")
print('| run | ckpt | set | config | win | Δ vs NMA30 (same set) · picker lift b−a | natMain | natMainT2 | main tier | main / sec / Σ layers | elites | top main | top sec |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (st, cfg), p in sorted(REF.items()):
    if os.path.exists(p): c = load(p); print(row('NMA 30M (ref)', '30000000', st, cfg, c, f"n {len(c['perMatch'])}"))
key = lambda p: re.match(r'(FIN-\w+-s\d+)-(\d+)M-([ab])-(solo|coop)\.json$', os.path.basename(p))
files = sorted((f for f in glob.glob(E + '/out/*.json') if key(f)), key=lambda f: (key(f).group(1), int(key(f).group(2)), key(f).group(3), key(f).group(4)))
for f in files:
    run, M, st, cfg = key(f).groups(); c = load(f)
    ck = open(f[:-5] + '.ckpt').read().split()[0] if os.path.exists(f[:-5] + '.ckpt') else '?'
    ex = delta(c, load(REF[(st, cfg)])) if os.path.exists(REF[(st, cfg)]) else 'n/a'
    fa = f.replace(f'-b-{cfg}.json', f'-a-{cfg}.json')
    if st == 'b' and os.path.exists(fa): ex += f' · lift {delta(c, load(fa))}'
    f60 = f.replace(f'-{M}M-', '-60M-')
    if int(M) > 60 and os.path.exists(f60): ex += f' · vs own 60M {delta(c, load(f60))}'
    print(row(f'{run} {M}M', ck, st, cfg, c, f"{ex} (n {len(c['perMatch'])})"))
