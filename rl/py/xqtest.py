"""xplace: PyTorch QNet scores of stored candidates (parity with xqtest.mjs). python xqtest.py model.pt file.jsonl N"""
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from xtrain import QNet, load_records, Data, NMAX, XCF
ck = torch.load(sys.argv[1], map_location='cpu'); c = ck['cfg']
net = QNet(c['FP'], c['FT'], c['FE'], c['FG'], c['V'], d=c['d'], de=c['de'], attn=c.get('attn', 0), heads=c.get('heads', 4), qattn=c.get('qattn', 1), cf=c.get('cf', 0)); net.load_state_dict(ck['state']); net.eval()
recs = load_records(sys.argv[2])[:int(sys.argv[3])]
D = Data(recs, recs[0]['dims'], seed0=True)
for i, r in enumerate(recs):
    b, cur, boss, w = D.batch(np.array([i]), 'cpu'); cs = r['cands'][:8]; M = len(cs); n = min(r['n'], NMAX)
    tg = np.full((M, NMAX, 2), -1, np.int64)
    for q, cc in enumerate(cs): tg[q, :n] = np.array(cc['a'][:n])
    rep = lambda x: x.expand(M, *x.shape[1:]).contiguous()
    cfa = torch.from_numpy(np.array([cc.get('f') or [0.0] * XCF for cc in cs], np.float16).astype(np.float32))
    with torch.no_grad(): Q = net.score(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), torch.from_numpy(tg), cfa, cfa[:1].expand(M, XCF))
    print(json.dumps({'q': [round(float(x), 4) for x in Q]}))
