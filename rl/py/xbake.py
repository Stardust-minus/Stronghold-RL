"""xplace (X7): teacher bake-off report from xteach records. python3 xbake.py 'dir/*.jsonl' [...]
Per teacher x budget: honest utility (fresh seeds, vs cur) mean +- SE, paired vs hc at the same budget, non-legacy pick share,
sims; pool diversity: where the search best / the scorer's pick comes from."""
import sys, glob, json, collections
import numpy as np

recs = []
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get('teach'): recs.append(r)
print(f'decisions {len(recs)}; pool size mean {np.mean([r["poolN"] for r in recs]):.1f}; sims/decision {np.mean([r["nsim"] for r in recs]):.0f}; ms {np.mean([r["ms"] for r in recs]):.0f}')
LEG = ('cur', 'leg')
se = lambda v: np.std(v) / max(1, len(v)) ** 0.5
names = sorted({k for r in recs for k in r['tch']}, key=lambda k: (k.rstrip('0123456789'), int(''.join(c for c in k if c.isdigit()) or 0)))
by = collections.defaultdict(dict)
for i, r in enumerate(recs):
    for k, x in r['tch'].items():
        if x.get('h') is not None: by[k][i] = x
print('| teacher | n | honest util | vs hc same budget | search value | non-legacy pick | sims |')
print('|---|---|---|---|---|---|---|')
for k in names:
    d = by[k]; idx = sorted(d)
    h = np.array([d[i]['h'] for i in idx]); nl = np.mean([d[i].get('src') not in LEG for i in idx])
    b = ''.join(c for c in k if c.isdigit()); hc = by.get('hc' + b, {}) if b else {}
    com = [i for i in idx if i in hc]; dv = np.array([d[i]['h'] - hc[i]['h'] for i in com]) if com and not k.startswith('hc') else None
    sv = [d[i].get('v') for i in idx if d[i].get('v') is not None]
    sims = [d[i].get('sims') for i in idx if d[i].get('sims') is not None]
    print(f"| {k} | {len(idx)} | {h.mean():+.2f}±{se(h):.2f} | {'' if dv is None else f'{dv.mean():+.2f}±{se(dv):.2f}'} | {np.mean(sv) if sv else float('nan'):+.2f} | {nl:.2f} | {np.mean(sims) if sims else float('nan'):.0f} |")
src = collections.Counter(); qsrc = collections.Counter(); tpl_better = collections.Counter(); tpl_n = collections.Counter()
for r in recs:
    def srcof(p): return r['cands'][p]['src'] if p is not None and p >= 0 else (r['lcands'][-1 - p]['src'] if p is not None else None)
    src[srcof(r['tch']['searchBest']['pick'])] += 1
    if 'qpool' in r['tch']: qsrc[srcof(r['tch']['qpool']['pick'])] += 1
    c0 = r['cands'][0]['s']
    for c in r['cands'] + r['lcands']:
        if c['src'] in LEG or not c['s'] or not all(c['s']) or not all(c0): continue
        tpl_n[c['src']] += 1
        if np.mean([x[0] for x in c['s']]) <= np.mean([x[0] for x in c0]) and np.mean([x[1] for x in c['s']]) > np.mean([x[1] for x in c0]): tpl_better[c['src']] += 1
n = len(recs)
print('search best by source:', {k: round(v / n, 3) for k, v in src.most_common()})
print('scorer (qpool) pick by source:', {k: round(v / n, 3) for k, v in qsrc.most_common()})
print('share of decisions: search best non-legacy %.3f, scorer pick non-legacy %.3f' % (sum(v for k, v in src.items() if k not in LEG) / n, sum(v for k, v in qsrc.items() if k not in LEG) / max(1, sum(qsrc.values()))))
print('candidates beating cur on the leaf seeds (no more leaks, more leader damage):', {k: f'{tpl_better[k]}/{tpl_n[k]}' for k in sorted(tpl_n)})
