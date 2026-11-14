"""(11-14) Co-op plan diagnostics of an eval JSON (eval.py --bond-log --nat-plan ... --human-proxy '{...,"log":true}'), per config:
collision = the share of model seats whose plan nation P is the human seat's nation (the proxy's own plan P when it has one, else
the human's main bond at its last battle when that bond is national); replans per match = the model seats' plan changes
(natPlan.coopReplan; 0 without it). Also the team win of matches with and without a collision.
python coopstats.py <eval.json> [<eval.json> ...]"""
import json, os, sys
from stackrep import seat_metrics

def human_nations(m):
    out = set()
    for h in m.get('human') or []:
        plan = (h.get('proxy') or {}).get('plan')
        if isinstance(plan, dict) and plan.get('P'): out.add(plan['P']); continue
        e = seat_metrics((h.get('bondLog') or {}).get('end'))
        if e and e['nat']: out.add(e['main'])
    return out

def summarize(per):
    seats = col = rep = 0; wc, wn = [], []
    for m in per:
        if not m.get('human') or not m.get('nat'): continue
        hn = human_nations(m); c = sum(1 for s in m['nat'] if s and s.get('P') in hn)
        seats += sum(1 for s in m['nat'] if s); col += c; rep += sum((s or {}).get('replans', 0) for s in m['nat'])
        (wc if c else wn).append(m['win'])
    n = len(wc) + len(wn)
    if not n: return None
    mean = lambda x: sum(x) / len(x) if x else None
    return {'matches': n, 'seats': seats, 'collision': col / max(1, seats), 'replansPerMatch': rep / n,
            'winCollided': mean(wc), 'winOther': mean(wn), 'matchesCollided': len(wc)}

if __name__ == '__main__':
    for p in sys.argv[1:]:
        for cfg, c in json.load(open(p))['configs'].items():
            r = summarize(c['perMatch'])
            if r: print(f"{os.path.basename(p)} {cfg}: " + ' | '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()))
