"""NAT arm surgery (11-13): a checkpoint without the 'plan' block -> the same network on the env's natPlan spec (the plan block
appended: its encoder freshly initialised, its pooled columns in trunk.0 / vtrunk.0 zero), so every output is unchanged for any
plan features. Equivalence on recorded obs (old obs + random plan features). python natsurg.py --ckpt in.pt --spec spec.json --obs obs.npz --out out.pt"""
import argparse, json, numpy as np, torch
from model import load, save, SPNet
from model2 import SPNet2
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--spec', required=True); ap.add_argument('--obs', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); torch.set_num_threads(4); torch.manual_seed(0)
src, ck = load(a.ckpt); src.eval(); spec = json.load(open(a.spec))
assert 'plan' in spec['blocks'] and 'plan' not in src.spec['blocks']
for k, b in src.spec['blocks'].items(): assert spec['blocks'][k] == b, k
cfg = dict(getattr(src, 'cfg', {}) or {})
dst = SPNet2(spec, **cfg) if cfg else SPNet(spec)
sd, ss = dst.state_dict(), src.state_dict()
for k, v in ss.items():
    t = sd[k]
    if t.shape == v.shape: sd[k] = v.clone(); continue
    assert k in ('trunk.0.weight', 'vtrunk.0.weight') and t.shape[0] == v.shape[0] and t.shape[1] > v.shape[1], (k, t.shape, v.shape)
    w = torch.zeros_like(t); w[:, :v.shape[1]] = v; sd[k] = w  # the plan's pooled features are the trunk's last input columns
dst.load_state_dict(sd); dst.eval()
d = np.load(a.obs); O, M = torch.from_numpy(d['obs']), torch.from_numpy(d['mask'])
P = spec['blocks']['plan']; X = torch.zeros(len(O), spec['dim']); X[:, :O.shape[1]] = O
X[:, P['offset']:P['offset'] + P['width']] = (torch.rand(len(O), P['width']) < 0.2).float(); X[:, P['offset']] = 1.0
with torch.no_grad(): l0, v0 = src(O, M); l1, v1 = dst(X, M)
r = {'params_src': int(sum(p.numel() for p in src.parameters())), 'params': int(sum(p.numel() for p in dst.parameters())), 'rows': len(O), 'D': [src.D, dst.D],
     'max_dlogit': float((l1 - l0).abs().masked_fill(~M, 0).max()), 'max_dvalue': float((v1 - v0).abs().max()), 'steps': ck.get('steps')}
print(json.dumps(r))
save(a.out, dst, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'natsurgery': a.ckpt})
