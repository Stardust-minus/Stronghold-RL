"""FIN finalpick (11-14): the final candidate table and pick.
Candidates: FIN-{MLP,V5}-s{1,2} at 60M and 70M (co-op stage), NMA 30M (seed 0) and CB10 (seed 0 + co-op stage).
Rule (11-14): setting b (c9 pool + xboss-v3) solo:ABYSS and coop4h1 both >= .50, rank by their mean.
Gate of the co-op stage: 70M vs its own 60M; adopt if co-op rises and solo drops by no more than .02 (shown for b and a).
Held-out (evals/heldout/<cand>-b-{solo,coop}.json, --seed 200000000) is shown when present; that is the reported number.
Extension (11-14): also discovers the extension candidates in evals/out (FIN-MLPX-s<k>-<M>M / FIN-V5X-s<k>-<M>M = stage 2 continued
60M -> 90M, ext/launch.sh; FIN-MLPXC / FIN-V5XC = their co-op stage; FIN-V5L = the v5 co-op stage at lr 3e-4), M > 60; gate rows:
each ext milestone vs its own 60M, an XC stage vs its start (the ext milestone it started from), V5L 70M vs the run's 60M."""
import json, os, sys, glob, re
import numpy as np
B = os.environ['B']  # the work dir; E = B + '/FINAL/evals'
sys.path.insert(0, B + '/code-A6/tools/rl/py')
from stackrep import summarize
load = lambda p: list(json.load(open(p))['configs'].values())[0] if os.path.exists(p) else None
def files(cand):
    if cand == 'NMA-30M':
        return {('b', 'solo'): E + '/ref/nma-P9pool-solo.json', ('b', 'coop'): E + '/ref/nma-P9pool-coop4h1.json',
                ('a', 'solo'): B + '/N/ev11/NMA-30000000-solo480.json', ('a', 'coop'): B + '/N/ev11/NMA-30000000-coop4h1-480.json'}
    if cand == 'CB10':
        return {(s, c): f'{E}/cand/CB10-{s}-{c}.json' for s in 'ab' for c in ('solo', 'coop')}
    return {(s, c): f'{E}/out/{cand}-{s}-{c}.json' for s in 'ab' for c in ('solo', 'coop')}
def delta(c, r):
    w = {m['seed']: m['win'] for m in c['perMatch']}; w0 = {m['seed']: m['win'] for m in r['perMatch']}
    s = sorted(set(w) & set(w0)); d = np.array([w[x] - w0[x] for x in s], float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)
fmt = lambda t: f'{t[0]:+.3f}±{t[1]:.3f}'
cands = [f'FIN-{a}-s{k}-{m}M' for a in ('MLP', 'V5') for k in (1, 2) for m in (60, 70)] + ['NMA-30M', 'CB10']
XK = re.compile(r'(FIN-(MLP|V5)(X|XC|L)-s(\d))-(\d+)M-b-solo\.json$')
xc = sorted({m.group(0)[:-len('-b-solo.json')] for m in (XK.match(os.path.basename(f)) for f in glob.glob(E + '/out/*-b-solo.json')) if m and int(m.group(5)) > 60},
            key=lambda c: (c.split('-')[1], c.split('-')[2], int(c.split('-')[3][:-1])))
cands += xc
XSTART = {}  # XC candidate -> the ext milestone it started from (evals/out/<XC cand>.start, written by the extension launcher)
for c in xc:
    if 'XC-' in c and os.path.exists(f'{E}/out/{c}.start'): XSTART[c] = open(f'{E}/out/{c}.start').read().split()[0]
rows = []
for cand in cands:
    F = {k: load(v) for k, v in files(cand).items()}
    if F[('b', 'solo')] is None or F[('b', 'coop')] is None:
        rows.append((cand, None, F)); continue
    bs, bc = F[('b', 'solo')]['teamWin'], F[('b', 'coop')]['teamWin']
    rows.append((cand, (bs, bc, (bs + bc) / 2, bs >= .5 and bc >= .5), F))
print('## Candidates (setting b = c9 pool + xboss-v3; a = no pickers; 476 seeds each, seed set 7000000)\n')
print('| candidate | b solo | b coop | mean | both >= .50 | a solo | a coop | natMain / T2 solo | natMain / T2 coop | Σ layers solo / coop | elites solo / coop | top secondary (solo) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for cand, r, F in rows:
    if r is None: print(f'| {cand} | pending | | | | | | | | | | |'); continue
    S, C = summarize(F[('b', 'solo')]['perMatch']), summarize(F[('b', 'coop')]['perMatch'])
    a = lambda k: f"{F[k]['teamWin']:.3f}" if F[k] else '-'
    print(f"| {cand} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {'yes' if r[3] else 'no'} | {a(('a', 'solo'))} | {a(('a', 'coop'))} | {S['natMain']:.2f} / {S['natMainT2']:.2f} | "
          f"{C['natMain']:.2f} / {C['natMainT2']:.2f} | {S['sumLayers']:.0f} / {C['sumLayers']:.0f} | {S['elites']:.1f} / {C['elites']:.1f} | {', '.join(f'{k} {v:.2f}' for k, v in S['topSec'][:2])} |")
print('\n## Co-op stage gate: 70M vs own 60M (paired; adopt if co-op rises and solo >= -.02)\n')
print('| run | b solo Δ | b coop Δ | a solo Δ | a coop Δ | adopt (b) | adopt (a) |\n|---|---|---|---|---|---|---|')
def own60(c):  # FIN-MLPX-s2-80M -> FIN-MLP-s2-60M
    m = re.match(r'FIN-(MLP|V5)(X|XC|L)-s(\d)-', c); return f'FIN-{m.group(1)}-s{m.group(3)}-60M'
G = [(f'FIN-{a}-s{k}', f'FIN-{a}-s{k}-70M', f'FIN-{a}-s{k}-60M') for a in ('MLP', 'V5') for k in (1, 2)] + [('NMA/CB10', 'CB10', 'NMA-30M')]
G += [(f'{c} vs ' + (XSTART.get(c) or own60(c)), c, XSTART.get(c) or own60(c)) for c in xc]
for run, c70, c60 in G:
    f70, f60 = {k: load(v) for k, v in files(c70).items()}, {k: load(v) for k, v in files(c60).items()}
    out, ad = [], {}
    for s in 'ba':
        ds, dc = [(delta(f70[(s, c)], f60[(s, c)]) if f70[(s, c)] and f60[(s, c)] else None) for c in ('solo', 'coop')]
        out += [fmt(ds) if ds else '-', fmt(dc) if dc else '-']
        ad[s] = '-' if not (ds and dc) else ('yes' if dc[0] > 0 and ds[0] >= -.02 else 'no')
    print(f"| {run} | {out[0]} | {out[1]} | {out[2]} | {out[3]} | {ad['b']} | {ad['a']} |")
ok = sorted([x for x in rows if x[1] and x[1][3]], key=lambda x: -x[1][2])
rest = sorted([x for x in rows if x[1] and not x[1][3]], key=lambda x: -x[1][2])
print('\n## Ranking (rule: both b >= .50, by mean)\n')
for i, (cand, r, F) in enumerate(ok + rest):
    top = (ok + rest)[0]
    d = '' if i == 0 else f" | vs #1 solo {fmt(delta(F[('b', 'solo')], top[2][('b', 'solo')]))}, coop {fmt(delta(F[('b', 'coop')], top[2][('b', 'coop')]))}"
    print(f"{i + 1}. {cand}: mean {r[2]:.3f} (solo {r[0]:.3f}, coop {r[1]:.3f}){'' if r[3] else ' [fails >= .50]'}{d}")
H = E + '/heldout'
if os.path.isdir(H):
    print('\n## Held-out (b protocol, --seed 200000000: solo seeds 202,000,000+, coop4h1 780,000,000+; never used for selection)\n')
    print('| candidate | solo | coop4h1 | mean | natMain / T2 solo | natMain / T2 coop | Σ layers solo / coop | elites solo / coop |\n|---|---|---|---|---|---|---|---|')
    for cand in cands:
        s, c = load(f'{H}/{cand}-b-solo.json'), load(f'{H}/{cand}-b-coop.json')
        if not (s and c): continue
        S, C = summarize(s['perMatch']), summarize(c['perMatch'])
        print(f"| {cand} | {s['teamWin']:.3f} | {c['teamWin']:.3f} | {(s['teamWin'] + c['teamWin']) / 2:.3f} | {S['natMain']:.2f} / {S['natMainT2']:.2f} | {C['natMain']:.2f} / {C['natMainT2']:.2f} | "
              f"{S['sumLayers']:.0f} / {C['sumLayers']:.0f} | {S['elites']:.1f} / {C['elites']:.1f} |")
