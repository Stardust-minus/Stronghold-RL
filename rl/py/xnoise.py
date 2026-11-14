"""xplace: seed noise of boss utilities on validated candidates (seed 0 vs seeds 1-2)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtrain as X
X.UCLIP = 30.0
recs = []
for p in sys.argv[1].split(','): recs += X.load_records(p, 'boss')
recs = [r for r in recs if r.get('salt')]
a0, a12, best_h, best_0, legb_h, legb_0, nleg = [], [], [], [], [], [], 0
for r in recs:
    cs = r['cands']; c0 = cs[0]['s']
    if not c0 or len(c0) < 3 or not all(c0[:3]): continue
    for j in r.get('val', []):
        if j == 0: continue
        s = cs[j]['s']
        if len(s) < 3 or not all(s[:3]): continue
        u = [X.util(True, s[k], c0[k]) for k in range(3)]
        a0.append(u[0]); a12.append((u[1] + u[2]) / 2)
        if j == r.get('best0'): best_0.append(u[0]); best_h.append((u[1] + u[2]) / 2)
    # best legacy (K8) by seed 0, if validated
    legs = [j for j, c in enumerate(cs) if c['src'] == 'leg' and c['s'] and c['s'][0]][:7]
    if legs:
        u0 = [X.util(True, cs[j]['s'][0], c0[0]) for j in legs]; jb = legs[int(np.argmax(u0))]
        if max(u0) > 0 and jb in r.get('val', []) and len(cs[jb]['s']) >= 3 and all(cs[jb]['s'][:3]):
            s = cs[jb]['s']; legb_0.append(X.util(True, s[0], c0[0])); legb_h.append((X.util(True, s[1], c0[1]) + X.util(True, s[2], c0[2])) / 2)
        nleg += 1
a0, a12 = np.array(a0), np.array(a12)
print(f'val cands n {len(a0)}: seed0 mean {a0.mean():+.2f} sd {a0.std():.2f}; seeds1-2 mean {a12.mean():+.2f} sd {a12.std():.2f}; corr {np.corrcoef(a0, a12)[0,1]:.3f}')
print(f'search best0 n {len(best_0)}: seed0 {np.mean(best_0):+.2f} -> honest (seeds 1-2) {np.mean(best_h):+.2f}')
print(f'K8 legacy best-by-seed0 (validated, >0) n {len(legb_0)} of {nleg}: seed0 {np.mean(legb_0):+.2f} -> honest {np.mean(legb_h):+.2f}')
