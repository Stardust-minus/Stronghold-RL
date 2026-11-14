"""confirm.py [out dir] (11-14): the confirmation round tables. Two finalists, b protocol, --seed 300000000 (fresh set, never used before).
1. Fresh set: solo:ABYSS + coop4h1:ABYSS (legacy human proxy, the headline protocol), paired V5X - MLP.
2. Human-robustness cells (coop4h1, same seeds): nat:<P>Ship human per nation (the teammate proxies of rl/proxies.mjs; live seeds = seeds
   where the forced nation is live) and class; real-player-weighted coop4h1 = the nation mix of 126 real matches (aggregates only; 78% national: siracusa 13, yan 6, kjerag 5,
   egir 4, kazimierz 4, victoria 4, laterano 3, sargon 2; 22% class), per-cell binomial variance; paired Δ per cell on the
   live seeds of both, weighted Δ SE = sqrt(sum w^2 se^2). Hole = cell win - the model's own legacy-proxy coop win on the same seeds.
  B=<work dir> python3 confirm.py [out dir] [--py-dir <tools/rl/py>]   (default out dir: B/FINAL/evals/confirm)"""
import json, os, sys
import numpy as np
B = os.environ['B']; _a = sys.argv[1:]; PYD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'py')
if '--py-dir' in _a: i = _a.index('--py-dir'); PYD = _a[i + 1]; del _a[i:i + 2]
D = _a[0] if _a else B + '/FINAL/evals/confirm'
sys.path.insert(0, os.path.abspath(PYD))
from stackrep import summarize
C = ['FIN-MLP-s2-70M', 'FIN-V5X-s1-90M']; A, V = C
W = {'siracusa': 13, 'yan': 6, 'kjerag': 5, 'egir': 4, 'kazimierz': 4, 'victoria': 4, 'laterano': 3, 'sargon': 2}; NAT = 0.78
NAME = {'siracusa': '叙拉古', 'yan': '炎', 'kjerag': '谢拉格', 'egir': '阿戈尔', 'kazimierz': '卡西米尔', 'victoria': '维多利亚', 'laterano': '拉特兰', 'sargon': '萨尔贡', 'class': 'class'}
load = lambda p: list(json.load(open(p))['configs'].values())[0] if os.path.exists(p) else None
def live(c):
    out = set()
    for pm in c['perMatch']:
        hs = pm.get('human') or []
        if hs and all((h['proxy'].get('forcedLive', True) if 'forcedLive' in h['proxy'] else h['proxy'].get('live', True)) for h in hs): out.add(pm['seed'])
    return out
def wins(c, S=None): return {m['seed']: m['win'] for m in c['perMatch'] if S is None or m['seed'] in S}
def pd(a, b, S=None):
    x, y = wins(a, S), wins(b, S); s = sorted(set(x) & set(y)); d = np.array([x[k] - y[k] for k in s], float)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(s)
f = lambda t: f'{t[0]:+.3f} ± {t[1]:.3f}'
st = lambda c, S=None: summarize([m for m in c['perMatch'] if S is None or m['seed'] in S])
def strow(r): return f"{r['natMain']:.2f} / {r['natMainT2']:.2f} | {r['mainTier']:.2f} | {r['mainLayers']:.0f} / {r['secLayers']:.0f} / {r['sumLayers']:.0f} | {r['elites']:.1f} | {', '.join(f'{k} {v:.2f}' for k, v in r['topMain'][:2])}"
FR = {(c, g): load(f'{D}/FR-{c}-{g}.json') for c in C for g in ('solo', 'coop')}
print('## Confirmation round (11-14): fresh seed set --seed 300000000, b protocol (code-A7, c9 pool + xboss-v3), 476 per cell\n')
print('Fresh set: solo:ABYSS seeds 302,000,000-302,000,475, coop4h1:ABYSS 880,000,000-880,000,475. Untouched by any selection. The only other use: the JS check matches of the release exports (`rl/release/play.mjs`, parity only, chose nothing) on solo:ABYSS 302,000,000-302,000,007 and coop4h1:ABYSS 880,000,000-880,000,007; the last column drops them.\n')
print('### 1. Fresh-seed headline (paired V5X-s1-90M - MLP-s2-70M)\n')
print('| candidate | solo:ABYSS | coop4h1:ABYSS | mean | solo / coop w/o the check-match seeds |\n|---|---|---|---|---|')
SMOKE = {302000000 + i for i in range(8)} | {880000000 + i for i in range(8)}  # JS check matches of the release exports
for c in C:
    s, co = FR[(c, 'solo')], FR[(c, 'coop')]
    if not (s and co): print(f'| {c} | pending | | | |'); continue
    s2 = np.mean([m['win'] for m in s['perMatch'] if m['seed'] not in SMOKE]); c2 = np.mean([m['win'] for m in co['perMatch'] if m['seed'] not in SMOKE])
    print(f"| {c} | {s['teamWin']:.3f} | {co['teamWin']:.3f} | {(s['teamWin'] + co['teamWin']) / 2:.3f} | {s2:.3f} / {c2:.3f} |")
if all(FR.values()):
    ds, dc = pd(FR[(V, 'solo')], FR[(A, 'solo')]), pd(FR[(V, 'coop')], FR[(A, 'coop')])
    allS = {m['seed'] for m in FR[(A, 'solo')]['perMatch']} - SMOKE; ds2 = pd(FR[(V, 'solo')], FR[(A, 'solo')], allS)
    allC = {m['seed'] for m in FR[(A, 'coop')]['perMatch']} - SMOKE; dc2 = pd(FR[(V, 'coop')], FR[(A, 'coop')], allC)
    print(f"\nPaired V5X - MLP: solo {f(ds)} (n {ds[2]}; w/o check-match seeds {f(ds2)}), coop {f(dc)} (n {dc[2]}; w/o {f(dc2)}), mean {(ds[0] + dc[0]) / 2:+.3f} ± {0.5 * np.hypot(ds[1], dc[1]):.3f}\n")
    print('Stacking (fresh set; solo / coop rows):\n\n| candidate | cfg | natMain / T2 | main tier | main / sec / Σ layers | elites | main bond |\n|---|---|---|---|---|---|---|')
    for c in C:
        for g in ('solo', 'coop'): print(f'| {c} | {g} | {strow(st(FR[(c, g)]))} |')
print('\n### 2. Human-robustness cells (coop4h1:ABYSS, same fresh seeds; nation cells on live seeds)\n')
print('| cell | live seeds | MLP-s2-70M | V5X-s1-90M | Δ V5X - MLP (paired) | MLP hole vs own legacy | V5X hole vs own legacy | MLP natMain / T2, Σ, elites | V5X natMain / T2, Σ, elites |\n|---|---|---|---|---|---|---|---|---|')
est = {c: [0.0, 0.0] for c in C}; dw = [0.0, 0.0]; ok = True
for cell in list(W) + ['class']:
    H = {c: load(f'{D}/HC-{c}-{cell}.json') for c in C}
    if not all(H.values()): print(f'| {NAME[cell]} | pending | | | | | | | |'); ok = False; continue
    S = (live(H[A]) & live(H[V])) if cell != 'class' else None
    p = {c: np.mean(list(wins(H[c], S).values())) for c in C}; n = len(wins(H[A], S))
    d = pd(H[V], H[A], S); hole = {c: pd(H[c], FR[(c, 'coop')], S) if FR[(c, 'coop')] else None for c in C}
    sm = {c: st(H[c], S) for c in C}
    sx = lambda r: f"{r['natMain']:.2f} / {r['natMainT2']:.2f}, {r['sumLayers']:.0f}, {r['elites']:.1f}"
    print(f"| {NAME[cell]} | {n if S is not None else 'all ' + str(n)} | {p[A]:.3f} | {p[V]:.3f} | {f(d)} | {f(hole[A]) if hole[A] else '-'} | {f(hole[V]) if hole[V] else '-'} | {sx(sm[A])} | {sx(sm[V])} |")
    wt = NAT * W[cell] / sum(W.values()) if cell != 'class' else 1 - NAT
    for c in C: est[c][0] += wt * p[c]; est[c][1] += wt ** 2 * p[c] * (1 - p[c]) / n
    dw[0] += wt * d[0]; dw[1] += wt ** 2 * d[1] ** 2
if ok:
    print('\n### 3. Real-player-weighted coop4h1 (real-player nation mix)\n')
    print('| candidate | real-player-weighted coop4h1 | legacy-proxy coop (fresh) |\n|---|---|---|')
    for c in C: print(f"| {c} | {est[c][0]:.3f} ± {np.sqrt(est[c][1]):.3f} | {FR[(c, 'coop')]['teamWin']:.3f} |" if FR[(c, 'coop')] else f'| {c} | {est[c][0]:.3f} | - |')
    print(f'\nPaired weighted Δ V5X - MLP = {dw[0]:+.3f} ± {np.sqrt(dw[1]):.3f}')
