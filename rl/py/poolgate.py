"""Pool pilot offline numbers : per normal-round decision over the candidate pool (mean of 2 per-match seeds):
the D-rule best (no extra leaks vs default, most layers, ties kills) over the whole pool vs over legacy K8 only; the share of decisions whose best is
non-legacy, by source; and what spatial heads pick on the pool (reg margin 0.5 / score) vs the pool best. python poolstat.py <lab glob> [head.pt ...]
(l8 poolgate: held-out decisions only, seed % VALMOD == 0 as picker_train5; writes a JSON summary to GATE_OUT)"""
import glob, json, sys, os, collections
import numpy as np
B = os.environ.get('SPRL_WORK', '.')  # the work dir holding the label / head inputs (was a host path)
recs = [json.loads(l) for f in sorted(glob.glob(sys.argv[1])) for l in open(f)]
recs = [d for d in recs if d.get('fmt') == 'p4' and d['scores'] and d['scores'][0] and d.get('src')]
VM = int(os.environ.get('VALMOD', '20')); recs = [d for d in recs if d['seed'] % VM == 0]; SUMMARY = {}
def lg(s): return sum((s.get('lg') or {}).values()) if s else 0.0
def dbest(sc, idx):
    b = idx[0]
    for j in idx[1:]:
        x, y = sc[j], sc[b]
        if not x or x['leaks'] > sc[0]['leaks']: continue
        if lg(x) > lg(y) or (lg(x) == lg(y) and x['kills'] > y['kills']): b = j
    return b
cat = lambda t: 'legacy' if t in ('default', 'legacy') else t.split(':')[0] if t.startswith('edit') else t
S = collections.Counter(); G = collections.defaultdict(list); n = 0; P = []
for d in recs:
    sc, src = d['scores'], d['src']; K = len(sc)
    leg = [j for j in range(K) if src[j] in ('default', 'legacy')]
    bp, bl = dbest(sc, list(range(K))), dbest(sc, leg)
    n += 1; S[src[bp]] += 1
    dl = lambda j: (sc[j]['leaks'] - sc[0]['leaks'], lg(sc[j]) - lg(sc[0]), sc[j]['kills'] - sc[0]['kills'])
    G['pool'].append(dl(bp)); G['k8'].append(dl(bl)); G['cands'].append(K)
print(f'decisions {n}, pool size mean {np.mean(G["cands"]):.1f}')
nl = sum(v for k, v in S.items() if k not in ('default', 'legacy'))
print(f'best is non-legacy in {nl / n:.2%} of decisions; by source: ' + ', '.join(f'{k} {v / n:.1%}' for k, v in S.most_common()))
for k in ('k8', 'pool'):
    a = np.array(G[k]); print(f'  D-best over {k:4s}: Δleaks {a[:,0].mean():+.3f}  Δlayers {a[:,1].mean():+.2f}  Δkills {a[:,2].mean():+.2f}  (vs default)')
# spatial heads on the pool
if len(sys.argv) > 2:
    import torch, torch.nn as nn
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from model5 import Tower, pool
    member = json.load(open(f'{B}/A/lookahead.json'))['pieceBonds']
    for hp in sys.argv[2:]:
        ck = torch.load(hp, map_location='cpu'); a = ck['args']; spec = json.load(open(ck['spec'])); D = spec['dim']
        class Net(nn.Module):
            def __init__(s):
                super().__init__(); s.tower = Tower(spec, member, a['d'], a['layers'], a['heads'], 2, True, True)
                s.head = nn.Sequential(nn.Linear(3 * a['d'], a['hidden']), nn.ReLU(), nn.Linear(a['hidden'], a['hidden']), nn.ReLU(), nn.Linear(a['hidden'], 4))
        net = Net(); net.load_state_dict(ck['state']); net.eval(); torch.set_num_threads(8)
        R = {'reg': [], 'score': [], 'regK8': []}
        for d in recs:
            sc = d['scores']; base = np.zeros(D, np.float32); base[d['obs']['i']] = d['obs']['v']; off, L = d['board']['off'], d['board']['len']
            X = np.repeat(base[None], len(d['cands']), 0); X[:, off:off + L] = 0
            for j, c in enumerate(d['cands']): X[j, [off + i for i in c['b']['i']]] = c['b']['v']
            with torch.no_grad(): x, kp, _ = net.tower(torch.from_numpy(X)); out = net.head(pool(x, kp)).numpy()
            u = -2 * (out[:, 1] - out[0, 1]) + (out[:, 2] - out[0, 2]) / 5 + 0.05 * (out[:, 3] - out[0, 3]); u[0] = 0
            ok = [j for j in range(len(sc)) if sc[j]]; leg = [j for j in ok if d['src'][j] in ('default', 'legacy')]
            pr = max(ok, key=lambda j: u[j]) if max(u[j] for j in ok) > 0.5 else 0
            pk = max(leg, key=lambda j: u[j]) if max(u[j] for j in leg) > 0.5 else 0
            ps_ = max(ok, key=lambda j: out[j, 0])
            for k, j in (('reg', pr), ('score', ps_), ('regK8', pk)): R[k].append((sc[j]['leaks'] - sc[0]['leaks'], lg(sc[j]) - lg(sc[0]), j != 0, d['src'][j] not in ('default', 'legacy')))
        for k, v in R.items():
            v = np.array(v, float); SUMMARY[f'{os.path.basename(os.path.dirname(hp))}/{os.path.basename(hp)}:{k}'] = {'dLeaks': float(v[:,0].mean()), 'dLayers': float(v[:,1].mean()), 'switch': float(v[:,2].mean()), 'n': len(v)}; print(f'  {os.path.basename(os.path.dirname(hp))}/{os.path.basename(hp)} {k:6s}: Δleaks {v[:,0].mean():+.3f} Δlayers {v[:,1].mean():+.2f} switch {v[:,2].mean():.2f} non-legacy pick {v[:,3].mean():.2f}')
# teacher-search bake-off (records with 'search')
Ss = [d['search'] for d in recs if 'search' in d]
if Ss:
    for meth in ('hill', 'cem'):
        g = np.array([s[meth]['u'] - s[meth]['u0'] for s in Ss if meth in s])
        if len(g): print(f'search {meth}: n {len(g)}, mean utility gain over the pool best {g.mean():+.3f} (improved {np.mean(g > 0):.0%})')
    print('decisions whose final best came from search:', sum(1 for d in recs if 'search' in d and d['src'][d['best']].startswith('search')), 'of', len(Ss))

if os.environ.get('GATE_OUT'): json.dump(SUMMARY, open(os.environ['GATE_OUT'], 'w'), indent=1)
