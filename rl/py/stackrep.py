"""Stacking metrics per logged AI seat (eval --bond-log; the last battle's board): main bond (most units, then tier; noStack
bonds such as 绝技 excluded), whether it is national (bondType SEASON) and its tier, main / secondary layers (secondary = the
best other stacking bond by tier, then layers), Σ stacking layers, elite (golden) count, the share with a national main at
tier >= 1 / >= 2, and the same at R9 / R12. python stackrep.py <eval.json> [<eval.json> ...] [--label name ...]"""
import json, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BONDS = json.load(open(os.path.join(HERE, '../../../data/bonds.json')))
STACK = {k for k, v in BONDS.items() if v.get('perStackParams') and not v.get('noStack')}
NAT = {k for k, v in BONDS.items() if v.get('bondType') == 'SEASON'}
NAME = {k: v['name'] for k, v in BONDS.items()}

def seat_metrics(row):
    act = [a for a in (row or {}).get('active', []) if a[0] in BONDS and not BONDS[a[0]].get('noStack')]
    if not act: return None
    main = max(act, key=lambda a: (a[3], a[1], a[2]))
    others = [a for a in act if a[0] != main[0] and a[0] in STACK]
    sec = max(others, key=lambda a: (a[1], a[2])) if others else None
    return {'main': main[0], 'nat': main[0] in NAT, 'tier': main[1], 'mainLayers': main[2] if main[0] in STACK else 0,
            'sec': sec[0] if sec else None, 'secLayers': sec[2] if sec else 0, 'sumLayers': sum(a[2] for a in act if a[0] in STACK),
            'elites': (row or {}).get('elites', 0), 'natTier': max([a[1] for a in act if a[0] in NAT] or [0])}

def summarize(per, win=None):
    """Stacking metrics of one config's perMatch rows (None without bond logs)."""
    seats = [(s, m['win']) for m in per for s in m.get('bondLog') or []]
    if not seats: return None
    E = [(seat_metrics(s.get('end')), w) for s, w in seats]; E = [(e, w) for e, w in E if e]
    if not E: return None
    mean = lambda f: float(np.mean([f(e) for e, _ in E]))
    t2 = [w for e, w in E if e['nat'] and e['tier'] >= 2]; ot = [w for e, w in E if not (e['nat'] and e['tier'] >= 2)]
    r = {'matches': len(per), 'win': win, 'seats': len(E),
         'natMain': mean(lambda e: e['nat']), 'natMainT2': mean(lambda e: e['nat'] and e['tier'] >= 2), 'anyNatT2': mean(lambda e: e['natTier'] >= 2),
         'mainTier': mean(lambda e: e['tier']), 'mainLayers': mean(lambda e: e['mainLayers']), 'secLayers': mean(lambda e: e['secLayers']),
         'sumLayers': mean(lambda e: e['sumLayers']), 'elites': mean(lambda e: e['elites']),
         'winNatT2': float(np.mean(t2)) if t2 else None, 'winOther': float(np.mean(ot)) if ot else None,
         'topMain': [(NAME.get(k, k), round(n / len(E), 2)) for k, n in Counter(e['main'] for e, _ in E).most_common(4)],
         'topSec': [(NAME.get(k, k), round(n / len(E), 2)) for k, n in Counter(e['sec'] for e, _ in E if e['sec']).most_common(3)]}
    for rr in ('r9', 'r12'):
        X = [seat_metrics(s.get(rr)) for s, _ in seats if s.get(rr)]; X = [x for x in X if x]
        if X: r[rr] = {'natMain': float(np.mean([x['nat'] for x in X])), 'sumLayers': float(np.mean([x['sumLayers'] for x in X])), 'elites': float(np.mean([x['elites'] for x in X]))}
    return r

def line(r, head):
    f = lambda x: 'n/a' if x is None else f'{x:.3f}'
    return (f"{head}: win {f(r['win'])} | seats {r['seats']} | national main {r['natMain']:.2f} (tier>=2 {r['natMainT2']:.2f}; any national tier>=2 {r['anyNatT2']:.2f})"
            f" | main tier {r['mainTier']:.2f} layers {r['mainLayers']:.0f} | secondary layers {r['secLayers']:.0f} | Σlayers {r['sumLayers']:.0f} | elites {r['elites']:.1f}"
            f" | win: national-main t2 {f(r['winNatT2'])} vs other {f(r['winOther'])} | main {r['topMain']} | secondary {r['topSec']}"
            + (f" | R9 national {r['r9']['natMain']:.2f} Σ {r['r9']['sumLayers']:.0f}" if 'r9' in r else '') + (f" | R12 national {r['r12']['natMain']:.2f} Σ {r['r12']['sumLayers']:.0f}" if 'r12' in r else ''))

def report(path, label=None):
    d = json.load(open(path))['configs']; out = {}
    for cfg, c in d.items():
        r = summarize(c['perMatch'], c['teamWin'])
        if r: out[cfg] = r; print(line(r, f"{label or os.path.basename(path)} {cfg}"))
    return out

if __name__ == '__main__':
    args = sys.argv[1:]; labels = []
    if '--label' in args:
        i = args.index('--label'); labels = args[i + 1:]; args = args[:i]
    for k, p in enumerate(args): report(p, labels[k] if k < len(labels) else None)
