#!/usr/bin/env python3
"""Regenerate records/final/ from the final pipeline's eval outputs (summaries only, no per-match rows).

  python3 records_final.py --final-dir <FINAL dir> --out <repo>/records/final \
      [--py-dir <tools/rl/py>] [--ref-a-solo <json>] [--ref-a-coop <json>] [--cand-dir <dir>] [--heldout-dir <dir>]

<FINAL dir>/evals is read: out/<run>-<M>M-<a|b>-<solo|coop>.json (+ .ckpt), ref/nma-P9pool-{solo,coop4h1}.json (setting b
reference), cand/CB10-*.json, heldout/<cand>-b-{solo,coop}.json, gate/result.txt and summary.md. Setting a = no pickers,
setting b = c9 normal-round pool picker + xboss-v3 boss placement. The a-reference (NMA 30M without pickers) lives outside the
FINAL dir: pass it with --ref-a-solo / --ref-a-coop. --py-dir is the tools/rl/py dir that holds stackrep.py (default: ../py next
to this script, i.e. the repo's rl/py once copied to tools/rl). Output is deterministic for the same inputs.
Files: milestones.csv (every milestone eval, as finsum.py), candidates.csv / coopgate.csv / ranking.csv / heldout.csv (as
finalpick.py), repro-gate.csv (gate.sh), summary.md (finsum.py's table, copied), index.csv."""
import argparse, csv, glob, json, math, os, re, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument('--final-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--py-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'py'))
ap.add_argument('--ref-a-solo', default=None)
ap.add_argument('--ref-a-coop', default=None)
ap.add_argument('--cand-dir', default=None, help='default <final-dir>/evals/cand')
ap.add_argument('--heldout-dir', default=None, help='default <final-dir>/evals/heldout')
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.py_dir))
from stackrep import summarize  # noqa: E402

E = os.path.join(a.final_dir, 'evals'); OUT = a.out; os.makedirs(OUT, exist_ok=True)
CAND = a.cand_dir or os.path.join(E, 'cand'); H = a.heldout_dir or os.path.join(E, 'heldout')
load = lambda p: list(json.load(open(p))['configs'].values())[0] if p and os.path.exists(p) else None
REF = {('a', 'solo'): a.ref_a_solo, ('a', 'coop'): a.ref_a_coop,
       ('b', 'solo'): os.path.join(E, 'ref', 'nma-P9pool-solo.json'), ('b', 'coop'): os.path.join(E, 'ref', 'nma-P9pool-coop4h1.json')}

def delta(c, r):
    w = {m['seed']: m['win'] for m in c['perMatch']}; w0 = {m['seed']: m['win'] for m in r['perMatch']}
    s = sorted(set(w) & set(w0)); d = [float(w[x]) - float(w0[x]) for x in s]
    if len(d) < 2: return None
    mu = sum(d) / len(d); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
    return (round(mu, 4), round(sd / math.sqrt(len(d)), 4), len(d))
f3 = lambda x: '' if x is None else f'{x:.3f}'
def dcols(t): return ['', ''] if t is None else [f'{t[0]:+.3f}', f'{t[1]:.3f}']
def stack(c):
    r = summarize(c['perMatch'], c.get('teamWin'))
    top = lambda lst: '; '.join(f'{k} {v:.2f}' for k, v in lst[:2])
    return [f"{r['natMain']:.3f}", f"{r['natMainT2']:.3f}", f"{r['mainTier']:.3f}", f"{r['mainLayers']:.1f}", f"{r['secLayers']:.1f}",
            f"{r['sumLayers']:.1f}", f"{r['elites']:.2f}", top(r['topMain']), top(r['topSec'])]
STK = ['natMain', 'natMainT2', 'mainTier', 'mainLayers', 'secLayers', 'sumLayers', 'elites', 'topMain', 'topSec']
def write(name, head, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(head); w.writerows(rows)

# milestones (finsum.py)
rows = []
for (st, cfg), p in sorted(REF.items()):
    c = load(p)
    if c: rows.append(['NMA-30M-ref', 30, '30000000', '', st, cfg, len(c['perMatch']), f3(c['teamWin'])] + [''] * 6 + stack(c))
key = lambda p: re.match(r'(FIN-\w+-s\d+)-(\d+)M-([ab])-(solo|coop)\.json$', os.path.basename(p))
files = sorted((f for f in glob.glob(os.path.join(E, 'out', '*.json')) if key(f)),
               key=lambda f: (key(f).group(1), int(key(f).group(2)), key(f).group(3), key(f).group(4)))
ckmd5 = {}
for f in files:
    run, M, st, cfg = key(f).groups(); c = load(f)
    ck = open(f[:-5] + '.ckpt').read().split() if os.path.exists(f[:-5] + '.ckpt') else []
    step = ck[0] if ck else ''; md5 = ck[1] if len(ck) > 1 and re.fullmatch(r'[0-9a-f]{8,32}', ck[1]) else ''
    if md5: ckmd5[f'{run}-{M}M'] = (step, md5[:8])
    r = load(REF[(st, cfg)]); fa = f.replace(f'-b-{cfg}.json', f'-a-{cfg}.json'); f60 = f.replace(f'-{M}M-', '-60M-')
    dl = delta(c, r) if r else None
    lift = delta(c, load(fa)) if st == 'b' and os.path.exists(fa) else None
    own = delta(c, load(f60)) if int(M) > 60 and os.path.exists(f60) else None
    rows.append([run, int(M), step, md5[:8], st, cfg, len(c['perMatch']), f3(c['teamWin'])] + dcols(dl) + dcols(lift) + dcols(own) + stack(c))
write('milestones.csv', ['run', 'M', 'ckpt_step', 'ckpt_md5', 'setting', 'config', 'n', 'win', 'd_vs_nma30', 'se', 'lift_b_minus_a', 'se_lift',
                         'd_vs_own60', 'se_own60'] + STK, rows)

# candidates / co-op stage gate / ranking / held-out (finalpick.py)
def cfiles(cand):
    if cand == 'NMA-30M': return dict(REF)
    if cand == 'CB10': return {(s, c): os.path.join(CAND, f'CB10-{s}-{c}.json') for s in 'ab' for c in ('solo', 'coop')}
    return {(s, c): os.path.join(E, 'out', f'{cand}-{s}-{c}.json') for s in 'ab' for c in ('solo', 'coop')}
cands = [f'FIN-{m}-s{k}-{x}M' for m in ('MLP', 'V5') for k in (1, 2) for x in (60, 70)] + ['NMA-30M', 'CB10']
XK = re.compile(r'(FIN-(MLP|V5)(X|XC|L)-s(\d))-(\d+)M-b-solo\.json$')
xc = sorted({mm.group(0)[:-len('-b-solo.json')] for mm in (XK.match(os.path.basename(f)) for f in glob.glob(os.path.join(E, 'out', '*-b-solo.json')))
             if mm and int(mm.group(5)) > 60}, key=lambda c: (c.split('-')[1], c.split('-')[2], int(c.split('-')[3][:-1])))
cands += xc
XSTART = {c: open(os.path.join(E, 'out', f'{c}.start')).read().split()[0] for c in xc
          if 'XC-' in c and os.path.exists(os.path.join(E, 'out', f'{c}.start'))}
C = {}; crow = []
for cand in cands:
    F = {k: load(v) for k, v in cfiles(cand).items()}; C[cand] = F
    if F[('b', 'solo')] is None or F[('b', 'coop')] is None:
        crow.append([cand, 'pending'] + [''] * 15); continue
    bs, bc = F[('b', 'solo')]['teamWin'], F[('b', 'coop')]['teamWin']
    S, Cc = stack(F[('b', 'solo')]), stack(F[('b', 'coop')])
    aw = lambda k: f3(F[k]['teamWin']) if F[k] else ''
    crow.append([cand, f3(bs), f3(bc), f3((bs + bc) / 2), 'yes' if bs >= .5 and bc >= .5 else 'no', aw(('a', 'solo')), aw(('a', 'coop')),
                 S[0], S[1], Cc[0], Cc[1], S[5], Cc[5], S[6], Cc[6], S[8], Cc[8]])
write('candidates.csv', ['candidate', 'b_solo', 'b_coop', 'b_mean', 'both_ge_0.50', 'a_solo', 'a_coop', 'natMain_solo', 'natMainT2_solo',
                         'natMain_coop', 'natMainT2_coop', 'sumLayers_solo', 'sumLayers_coop', 'elites_solo', 'elites_coop', 'topSec_solo', 'topSec_coop'], crow)
def own60(c):
    mm = re.match(r'FIN-(MLP|V5)(X|XC|L)-s(\d)-', c); return f'FIN-{mm.group(1)}-s{mm.group(3)}-60M'
G = [(f'FIN-{m}-s{k}', f'FIN-{m}-s{k}-70M', f'FIN-{m}-s{k}-60M') for m in ('MLP', 'V5') for k in (1, 2)] + [('NMA/CB10', 'CB10', 'NMA-30M')]
G += [(c, c, XSTART.get(c) or own60(c)) for c in xc]
grow = []
for run, c70, c60 in G:
    f70 = C.get(c70) or {k: load(v) for k, v in cfiles(c70).items()}; f60 = C.get(c60) or {k: load(v) for k, v in cfiles(c60).items()}
    out, ad = [], {}
    for s in 'ba':
        ds, dc = [(delta(f70[(s, c)], f60[(s, c)]) if f70[(s, c)] and f60[(s, c)] else None) for c in ('solo', 'coop')]
        out += dcols(ds) + dcols(dc)
        ad[s] = '' if not (ds and dc) else ('yes' if dc[0] > 0 and ds[0] >= -.02 else 'no')
    grow.append([run, c70, c60] + out + [ad['b'], ad['a']])
write('coopgate.csv', ['run', 'candidate', 'vs', 'b_solo_d', 'se', 'b_coop_d', 'se', 'a_solo_d', 'se', 'a_coop_d', 'se', 'adopt_b', 'adopt_a'], grow)
done = [(cand, C[cand]) for cand in cands if C[cand][('b', 'solo')] and C[cand][('b', 'coop')]]
mean = lambda fs: (fs[('b', 'solo')]['teamWin'] + fs[('b', 'coop')]['teamWin']) / 2
ok = lambda fs: fs[('b', 'solo')]['teamWin'] >= .5 and fs[('b', 'coop')]['teamWin'] >= .5
order = sorted([x for x in done if ok(x[1])], key=lambda x: -mean(x[1])) + sorted([x for x in done if not ok(x[1])], key=lambda x: -mean(x[1]))
rrow = []
for i, (cand, F) in enumerate(order):
    top = order[0][1]
    vs = [] if i == 0 else dcols(delta(F[('b', 'solo')], top[('b', 'solo')])) + dcols(delta(F[('b', 'coop')], top[('b', 'coop')]))
    rrow.append([i + 1, cand, f3(mean(F)), f3(F[('b', 'solo')]['teamWin']), f3(F[('b', 'coop')]['teamWin']), 'yes' if ok(F) else 'no'] + (vs or [''] * 4))
write('ranking.csv', ['rank', 'candidate', 'b_mean', 'b_solo', 'b_coop', 'both_ge_0.50', 'solo_d_vs_1', 'se', 'coop_d_vs_1', 'se'], rrow)
hrow = []
if os.path.isdir(H):
    for cand in cands:
        s, c = load(os.path.join(H, f'{cand}-b-solo.json')), load(os.path.join(H, f'{cand}-b-coop.json'))
        if not (s or c): continue
        S = stack(s) if s else [''] * 9; Cc = stack(c) if c else [''] * 9
        hrow.append([cand, f3(s['teamWin']) if s else 'pending', len(s['perMatch']) if s else '', f3(c['teamWin']) if c else 'pending',
                     len(c['perMatch']) if c else '', f3((s['teamWin'] + c['teamWin']) / 2) if s and c else '',
                     S[0], S[1], Cc[0], Cc[1], S[5], Cc[5], S[6], Cc[6]])
write('heldout.csv', ['candidate', 'b_solo', 'n_solo', 'b_coop', 'n_coop', 'b_mean', 'natMain_solo', 'natMainT2_solo', 'natMain_coop',
                      'natMainT2_coop', 'sumLayers_solo', 'sumLayers_coop', 'elites_solo', 'elites_coop'], hrow)
gr = []
gp = os.path.join(E, 'gate', 'result.txt')
if os.path.exists(gp):
    what = {'g1': 'training tree, no pickers, solo:ABYSS vs NMA 30M reference', 'g3': 'eval tree, c9 pool + xboss-v3, solo:ABYSS vs reference',
            'g4': 'eval tree, c9 pool + xboss-v3, coop4h1:ABYSS vs reference'}
    for l in open(gp):
        t = l.split()
        if len(t) >= 6 and t[2] == 'n': gr.append([t[0], what.get(t[0], ''), t[1], t[3], t[5]])
write('repro-gate.csv', ['gate', 'what', 'result', 'n', 'differing'], gr)
if os.path.exists(os.path.join(E, 'summary.md')): shutil.copyfile(os.path.join(E, 'summary.md'), os.path.join(OUT, 'summary.md'))
idx = [['milestones.csv', 'every milestone eval of the final runs (476 paired seeds, seed set 7000000), paired deltas, stacking'],
       ['candidates.csv', 'final candidates, settings a and b'], ['coopgate.csv', 'co-op stage gate: stage output vs its start'],
       ['ranking.csv', 'pick rule: both b >= .50, ranked by the b mean'], ['heldout.csv', 'held-out confirmation (--seed 200000000), setting b'],
       ['repro-gate.csv', 'eval trees reproduce the NMA 30M reference evals per match'], ['summary.md', 'the milestone table as markdown']]
write('index.csv', ['file', 'what'], [r for r in idx if os.path.exists(os.path.join(OUT, r[0]))])
print('records_final:', len(rows), 'milestone rows,', len(crow), 'candidates,', len(hrow), 'held-out rows ->', OUT)
