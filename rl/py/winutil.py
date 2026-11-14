"""Win-probability boss utility (b1): fit P(win | rehearsal share of the picked layout) on solo round-14 decisions (the in-game ensemble
pick, real match outcome), then score every candidate by P(win | its rehearsal share) and compare scorers on held-out decisions (seed % 5 == 0)."""
import json, glob, numpy as np, sys, os
B = os.environ.get('SPRL_WORK', '.')  # the work dir holding the label / head inputs (was a host path)
heads = {n: json.load(open(p)) for n, p in [('hj', f'{B}/P/inputs/head-joint.json'), ('hjE', f'{B}/P/b1/heads/hjE.json'), ('hjW', f'{B}/P/b1/heads/hjW.json')]}
def logits(h, c, rows):
    W1 = np.array(h['W1']); C = h['C']; z = np.maximum(0, (W1[:, :C] @ c + np.array(h['b1']))[None, :] + rows @ W1[:, C:].T)
    return z @ np.array(h['w2']) + h['b2'] + h['prior'] * rows[:, 1]
z = lambda v: (v - v.mean()) / (v.std() or 1)
def ep(e, ok): b = int(np.argmax(np.where(ok, e, -1e9))); return 0 if e[b] - e[0] <= 0 else b
M = {}
for f in glob.glob(f'{B}/P/b1/out/s*/res.json'):
    for m in list(json.load(open(f))['configs'].values())[0]['perMatch']: M[m['seed']] = m
R = []
for f in glob.glob(f'{B}/P/b1/out/s*/lab/boss-*.jsonl'):
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        if d['round'] != 14 or 'q' not in d or d['seed'] not in M or not (M[d['seed']].get('fa') or {}).get('maxHp'): continue
        R.append(d)
# fit: win ~ logistic(a + b * share_picked + c * lpc_picked) on the in-game picks
F = []
for d in R:
    sc = d['scores']; ok = np.array([s is not None for s in sc]); q = np.array(d['q'], float); c = np.array(d['c']); rows = np.array(d['rows'])
    b = ep(z(q) + z(logits(heads['hj'], c, rows)), ok); hp = M[d['seed']]['fa']['maxHp']
    F.append((sc[b]['boss'] / hp, sc[b]['leaks'], M[d['seed']]['win']))
F = np.array(F); x = np.c_[np.ones(len(F)), F[:, 0], F[:, 1]]; y = F[:, 2]; w = np.zeros(3)
for _ in range(3000):
    p = 1 / (1 + np.exp(-x @ w)); w -= 0.5 * x.T @ (p - y) / len(y) + 1e-4 * w * np.r_[0, 1, 1]
print('fit win ~ share, leaks (solo r14, in-game picks): w', np.round(w, 2), 'n', len(y))
P = lambda share, lk: 1 / (1 + np.exp(-(w[0] + w[1] * share + w[2] * lk)))
cols = ['q5', 'q5+hj', 'q5+hjE', 'q5+hjW', 'oracleWin', 'oracleLeak']; U = {k: [] for k in cols}; VAL = []
for d in R:
    sc = d['scores']; k = len(sc); ok = np.array([s is not None for s in sc]); hp = M[d['seed']]['fa']['maxHp']
    pw = np.array([P(s['boss'] / hp, s['leaks']) if s else -1 for s in sc]); q = np.array(d['q'], float); c = np.array(d['c']); rows = np.array(d['rows'])
    le = np.array([s['leaks'] if s else 99 for s in sc]); bo = np.array([s['boss'] if s else 0 for s in sc])
    picks = {'q5': ep(q, ok), 'oracleWin': int(np.argmax(pw)), 'oracleLeak': sorted([j for j in range(k) if ok[j]], key=lambda j: (le[j], -bo[j]))[0]}
    for n in ('hj', 'hjE', 'hjW'): picks['q5+' + n] = ep(z(q) + z(logits(heads[n], c, rows)), ok)
    for n, j in picks.items(): U[n].append(pw[j] - pw[0])
    VAL.append(((d['seed'] * 2654435761 >> 7) % 5) == 0)
VAL = np.array(VAL)
for sel_name, sel in (('held-out (seed % 5 == 0)', VAL), ('all solo r14', np.ones(len(VAL), bool))):
    print(f'-- {sel_name}: n {int(sel.sum())}; mean P(win) gain vs legacy default (in-game: q5 .565, q5+hj .590, q5+hjE .546, q5+hjW .590, none .553)')
    ref = np.array(U['q5+hj'])[sel]
    for n in cols:
        u = np.array(U[n])[sel]; dd = u - ref
        print(f'   {n:10s} {u.mean():+.4f}±{u.std() / np.sqrt(len(u)):.4f}   vs q5+hj {dd.mean():+.4f}±{dd.std() / np.sqrt(len(dd)):.4f}')
