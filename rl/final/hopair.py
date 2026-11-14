"""hopair.py [ref] (11-14): the held-out table (evals/heldout/<cand>-b-{solo,coop}.json, setting b, --seed 200000000) with full
stacking and paired deltas vs a reference candidate (default FIN-MLP-s2-70M, the pick at the time the held-out round ran).
Rule of that round: the reference stays unless a candidate beats it on held-out, paired: paired mean of the solo and coop
deltas > 0, both held-out >= .50. SE of the mean delta = 0.5 * sqrt(se_solo^2 + se_coop^2). The last table pools the
selection seeds (--seed 7000000) with the held-out seeds, 952 seeds per cell.
  B=<work dir> python3 hopair.py [ref] [--py-dir <tools/rl/py>]   (B/FINAL/evals is read; output is markdown on stdout)"""
import json, os, sys, glob
import numpy as np
B = os.environ['B']; H = B + '/FINAL/evals/heldout'
_a = sys.argv[1:]; PYD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'py')
if '--py-dir' in _a: i = _a.index('--py-dir'); PYD = _a[i + 1]; del _a[i:i + 2]
sys.path.insert(0, os.path.abspath(PYD))
from stackrep import summarize  # noqa: E402
REF = _a[0] if _a else 'FIN-MLP-s2-70M'
load = lambda p: list(json.load(open(p))['configs'].values())[0] if os.path.exists(p) else None
def delta(c, r):
    w = {m['seed']: m['win'] for m in c['perMatch']}; w0 = {m['seed']: m['win'] for m in r['perMatch']}
    s = sorted(set(w) & set(w0)); d = np.array([w[x] - w0[x] for x in s], float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)
ck = {}
for l in open(H + '/ckpts.txt'):
    p = l.split()
    if len(p) >= 2: ck.setdefault(p[0], p[1])
cands = sorted({os.path.basename(f)[:-len('-b-solo.json')] for f in glob.glob(H + '/*-b-solo.json')})
D = {c: (load(f'{H}/{c}-b-solo.json'), load(f'{H}/{c}-b-coop.json')) for c in cands}
D = {c: v for c, v in D.items() if v[0] and v[1]}
r0 = D.get(REF)
rows = []
for c, (s, co) in D.items():
    m = (s['teamWin'] + co['teamWin']) / 2
    if r0 and c != REF:
        ds, dc = delta(s, r0[0]), delta(co, r0[1]); dm = ((ds[0] + dc[0]) / 2, 0.5 * np.hypot(ds[1], dc[1]))
    else:
        ds = dc = dm = None
    rows.append((c, s, co, m, ds, dc, dm))
rows.sort(key=lambda x: -x[3])
f = lambda t: f'{t[0]:+.3f}±{t[1]:.3f}' if t else '-'
print(f'## Held-out (b = c9 pool + xboss-v3, code-A7; --seed 200000000: solo:ABYSS seeds 202,000,000-202,000,475, coop4h1:ABYSS 780,000,000-780,000,475; 476 each)\n')
print(f'Paired deltas vs {REF} (the reference). Rule: {REF} stays unless a candidate has paired mean delta > 0 and both held-out >= .50.\n')
print('| candidate | md5 | solo | coop4h1 | mean | Δ solo | Δ coop | Δ mean | beats ref |\n|---|---|---|---|---|---|---|---|---|')
for c, s, co, m, ds, dc, dm in rows:
    b = '(ref)' if c == REF else ('yes' if dm and dm[0] > 0 and s['teamWin'] >= .5 and co['teamWin'] >= .5 else 'no')
    print(f"| {c} | {ck.get(c, '?')} | {s['teamWin']:.3f} | {co['teamWin']:.3f} | {m:.3f} | {f(ds)} | {f(dc)} | {f(dm)} | {b} |")
print('\n### Stacking on the held-out seeds (solo / coop4h1; coop = the model seats)\n')
print('| candidate | main bond (solo) | main tier | natMain | natMain T2 | main layers | sec layers | Σ layers | elites | top secondary (solo) |\n|---|---|---|---|---|---|---|---|---|---|')
for c, s, co, *_ in rows:
    S, C = summarize(s['perMatch']), summarize(co['perMatch'])
    p = lambda k, d=2: f'{S[k]:.{d}f} / {C[k]:.{d}f}'
    print(f"| {c} | {', '.join(f'{k} {v:.2f}' for k, v in S['topMain'][:2])} | {p('mainTier')} | {p('natMain')} | {p('natMainT2')} | {p('mainLayers', 0)} | "
          f"{p('secLayers', 0)} | {p('sumLayers', 0)} | {p('elites', 1)} | {', '.join(f'{k} {v:.2f}' for k, v in S['topSec'][:3])} |")
# Pooled view: selection seeds (--seed 7000000, evals/out or evals/cand) + held-out seeds, 952 seeds per cell, paired vs REF.
E = B + '/FINAL/evals'
sel = lambda c, g: load(f'{E}/cand/{c}-b-{g}.json') if c == 'CB10' else load(f'{E}/out/{c}-b-{g}.json')
def pdelta(cs, rs):
    d = []
    for c, r in zip(cs, rs):
        w = {m['seed']: m['win'] for m in c['perMatch']}; w0 = {m['seed']: m['win'] for m in r['perMatch']}
        d += [w[x] - w0[x] for x in sorted(set(w) & set(w0))]
    d = np.array(d, float); return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)
if r0 and sel(REF, 'solo') and sel(REF, 'coop'):
    print(f'\n### Pooled selection + held-out (952 seeds per cell), paired vs {REF}\n')
    print('| candidate | selection solo / coop (mean) | held-out solo / coop (mean) | pooled solo | pooled coop | pooled mean | pooled Δ solo | pooled Δ coop | pooled Δ mean |\n|---|---|---|---|---|---|---|---|---|')
    R = {g: sel(REF, g) for g in ('solo', 'coop')}
    for c, s, co, m, *_ in rows:
        S = {g: sel(c, g) for g in ('solo', 'coop')}
        if not (S['solo'] and S['coop']): continue
        HOd = {'solo': s, 'coop': co}
        pw = {g: (S[g]['teamWin'] * len(S[g]['perMatch']) + HOd[g]['teamWin'] * len(HOd[g]['perMatch'])) / (len(S[g]['perMatch']) + len(HOd[g]['perMatch'])) for g in HOd}
        if c == REF:
            ds = dc = dm = None
        else:
            ds, dc = pdelta([S['solo'], s], [R['solo'], r0[0]]), pdelta([S['coop'], co], [R['coop'], r0[1]]); dm = ((ds[0] + dc[0]) / 2, 0.5 * np.hypot(ds[1], dc[1]))
        sm = (S['solo']['teamWin'] + S['coop']['teamWin']) / 2
        print(f"| {c} | {S['solo']['teamWin']:.3f} / {S['coop']['teamWin']:.3f} ({sm:.3f}) | {s['teamWin']:.3f} / {co['teamWin']:.3f} ({m:.3f}) | {pw['solo']:.3f} | {pw['coop']:.3f} | {(pw['solo'] + pw['coop']) / 2:.3f} | {f(ds)} | {f(dc)} | {f(dm)} |")
