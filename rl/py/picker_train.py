"""Boss-round candidate picker head (track 2): fit on fixed-seed rehearsal labels (soft listwise targets), report held-out
top-1 agreement and regret, export the head for tools/rl/picker.mjs.
python picker_train.py --train 'a/*.jsonl' --val 'b/*.jsonl' --out head.json [--H 64 --tau 0.1 --epochs 40]"""
import argparse, glob, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True); ap.add_argument('--val', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--H', type=int, default=64); ap.add_argument('--tau', type=float, default=0.1); ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--K', type=int, default=8)
ap.add_argument('--leak', type=float, default=1.0, help='weight of a counted leak in the soft target (vs 0.5 x leader-damage share)')
a = ap.parse_args()

def load(pattern):
    C, R, M, U, L, B = [], [], [], [], [], []
    for f in sorted(sum((glob.glob(p_) for p_ in pattern.split(',')), [])):
        for line in open(f):
            d = json.loads(line); sc = d['scores']; rows = d['rows'][:a.K]; k = len(rows)
            if k < 2 or not sc or all(s is None for s in sc[:k]): continue
            r = np.zeros((a.K, len(rows[0])), np.float32); r[:k] = rows; m = np.zeros(a.K, bool)
            le = np.full(a.K, 99.0); bo = np.zeros(a.K)
            ki = np.zeros(a.K)
            for j in range(k):
                s = sc[j]
                if s is None: continue
                m[j] = True; le[j] = s['leaks']; bo[j] = s['boss']; ki[j] = s['kills']
            bmax = max(bo.max(), 1.0); kmax = max(ki.max(), 1.0)
            u = -a.leak * le + 0.5 * bo / bmax + 0.01 * ki / kmax
            C.append(d['c']); R.append(r); M.append(m); U.append(u); L.append(le); B.append(bo / bmax)
    return [np.array(x) for x in (C, R, M, U, L, B)]
tr, va = load(a.train), load(a.val)
print(json.dumps({'train': len(tr[0]), 'val': len(va[0])}))
Cdim, Fdim = tr[0].shape[1], tr[1].shape[2]

class Head(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(Cdim + Fdim, a.H); s.l2 = nn.Linear(a.H, 1); s.prior = nn.Parameter(torch.tensor(1.0))
    def forward(s, c, r):
        x = torch.cat([c[:, None, :].expand(-1, r.shape[1], -1), r], -1)
        return s.l2(F.relu(s.l1(x))).squeeze(-1) + s.prior * r[..., 1]
net = Head(); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
T = [torch.from_numpy(x) for x in tr]; V = [torch.from_numpy(x) for x in va]
def target(u, m):
    z = (u / a.tau).masked_fill(~m, -1e9); return F.softmax(z, -1)
def evaluate(D):
    c, r, m, u, le, bo = D
    with torch.no_grad(): lg = net(c.float(), r.float()).masked_fill(~m, -1e9)
    p = lg.argmax(1); best = u.masked_fill(~m, -1e9).argmax(1); i = torch.arange(len(p))
    reg = lambda q: {'dLeaks': float((le[i, q] - le[i, best]).mean()), 'dBossShare': float((bo[i, best] - bo[i, q]).mean())}
    zero = torch.zeros_like(p)
    return {'top1': float((p == best).float().mean()), 'nonDefault': float((p != 0).float().mean()), 'labelNonDefault': float((best != 0).float().mean()),
            'regret': reg(p), 'regretDefault': reg(zero), 'ce': float(-(target(u.float(), m) * F.log_softmax(lg, -1)).sum(-1).mean())}
best_ce, best_state = 1e9, None
for ep in range(a.epochs):
    perm = torch.randperm(len(T[0]))
    for b in range(0, len(perm), 256):
        idx = perm[b:b + 256]; c, r, m, u = T[0][idx].float(), T[1][idx].float(), T[2][idx], T[3][idx].float()
        lg = net(c, r).masked_fill(~m, -1e9); loss = -(target(u, m) * F.log_softmax(lg, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ev = evaluate(V)
    if ev['ce'] < best_ce: best_ce, best_state = ev['ce'], {k: v.clone() for k, v in net.state_dict().items()}
    if ep % 5 == 4 or ep == a.epochs - 1: print(json.dumps({'epoch': ep, 'val': ev}))
net.load_state_dict(best_state); ev = evaluate(V); print(json.dumps({'final_val': ev, 'train': evaluate(T)}))
W1 = net.l1.weight.detach().numpy().astype(float)
json.dump({'C': Cdim, 'F': Fdim, 'H': a.H, 'W1': W1.round(6).tolist(), 'b1': net.l1.bias.detach().numpy().round(6).tolist(),
           'w2': net.l2.weight.detach().numpy()[0].round(6).tolist(), 'b2': float(net.l2.bias.item()), 'prior': float(net.prior.item()), 'val': ev}, open(a.out, 'w'))
print('saved', a.out)
