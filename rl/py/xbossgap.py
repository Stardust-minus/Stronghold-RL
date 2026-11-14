"""xplace: where is the boss-round gap on a trunk's boards? Per salted boss record: utility (seed 0, vs the current legacy layout)
of the K8 / K12 legacy-candidate oracle, of the full search pick (3-seed mean), and of a QNet's pick among the K8 candidates.
python xbossgap.py <qnet.pt> 'glob1' ['glob2' ...]"""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
import xtrain as X
ck = torch.load(sys.argv[1], map_location='cpu'); c = ck['cfg']
net = X.QNet(c['FP'], c['FT'], c['FE'], c['FG'], c['V'], d=c['d'], de=c['de'], attn=c.get('attn', 0), heads=c.get('heads', 4), qattn=c.get('qattn', 1), cf=c.get('cf', 0)); net.load_state_dict(ck['state']); net.eval()
X.UCLIP = 30.0
for pat in sys.argv[2:]:
    recs = [r for r in X.load_records(pat, 'boss') if r.get('salt')]
    if not recs: print(pat, 'no salted boss records'); continue
    D = X.Data(recs, recs[0]['dims'], seed0=True)
    st = dict(n=0, k8=0.0, k12=0.0, search=0.0, q=0.0, qbetter=0, qworse=0, k8pos=0)
    for i, r in enumerate(recs):
        cs = r['cands']; s0 = cs[0]['s'][0] if cs[0]['s'] else None
        if not s0: continue
        legs = [j for j, q in enumerate(cs) if q['src'] == 'leg' and q['s'] and q['s'][0]]
        k8 = [0] + legs[:7]; k12 = [0] + legs[:11]
        u = lambda j: X.util(True, cs[j]['s'][0], s0)
        u8 = [u(j) for j in k8]; u12 = [u(j) for j in k12]
        sp = X.util(True, X.mean_s(cs[r['pick']]['s']), X.mean_s(cs[0]['s']))
        n = min(r['n'], X.NMAX); tg = np.full((len(k8), X.NMAX, 2), -1, np.int64)
        for q, j in enumerate(k8): tg[q, :n] = np.array(cs[j]['a'][:n])
        cfa = np.array([cs[j].get('f') or [0.0] * X.XCF for j in k8], np.float16).astype(np.float32)
        b, cur, boss, w = D.batch(np.array([i]), 'cpu'); M = len(k8); rep = lambda x: x.expand(M, *x.shape[1:]).contiguous()
        with torch.no_grad():
            Q = net.score(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), torch.from_numpy(tg),
                          torch.from_numpy(cfa), torch.from_numpy(cfa[:1]).expand(M, X.XCF)).numpy()
        uq = u8[int(np.argmax(Q))]
        st['n'] += 1; st['k8'] += max(u8); st['k12'] += max(u12); st['search'] += sp; st['q'] += uq
        st['qbetter'] += uq > 1e-9; st['qworse'] += uq < -1e-9; st['k8pos'] += max(u8) > 1e-9
    n = max(1, st['n'])
    print(pat.split('/')[-2] if '/' in pat else pat, 'n', st['n'], '| oracle K8 %+.2f  K12 %+.2f  search pick %+.2f | QNet pick (K8) %+.2f (better %.2f, worse %.2f) | K8 oracle > legacy in %.2f of decisions'
          % (st['k8'] / n, st['k12'] / n, st['search'] / n, st['q'] / n, st['qbetter'] / n, st['qworse'] / n, st['k8pos'] / n))
