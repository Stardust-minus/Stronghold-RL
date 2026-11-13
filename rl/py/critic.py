"""LIN-C init (11-13): an SPNet checkpoint -> SPNet2 (width 1) with a separate critic trunk (vtrunk) copied from the shared
trunk, so policy and value are function-identical to the source; then the 6k-decision equivalence check.
python critic.py --ckpt in.pt --obs obs.npz --out out.pt"""
import argparse, json, numpy as np, torch
from model import load, save, SPNet
from model2 import SPNet2
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--obs', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); torch.set_num_threads(4)
src, ck = load(a.ckpt); src.eval(); assert type(src) is SPNet
dst = SPNet2(src.spec, width=1.0, critic=1)
sd = dst.state_dict(); ss = src.state_dict()
for k, v in ss.items(): assert sd[k].shape == v.shape, k; sd[k] = v.clone()
for k in [k for k in sd if k.startswith('vtrunk.')]: sd[k] = ss['trunk.' + k[len('vtrunk.'):]].clone()
assert not [k for k in sd if k not in ss and not k.startswith('vtrunk.')]
dst.load_state_dict(sd); dst.eval()
d = np.load(a.obs); O, M = torch.from_numpy(d['obs']), torch.from_numpy(d['mask'])
with torch.no_grad(): l0, v0 = src(O, M); l1, v1 = dst(O, M)
r = {'params': int(sum(p.numel() for p in dst.parameters())), 'params_src': int(sum(p.numel() for p in src.parameters())), 'rows': int(len(O)),
     'max_dlogit': float((l1 - l0).abs().masked_fill(~M, 0).max()), 'max_dvalue': float((v1 - v0).abs().max()), 'steps': ck.get('steps')}
print(json.dumps(r))
save(a.out, dst, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'arm': 'LIN-C', 'from': a.ckpt})
net2, _ = load(a.out); net2.eval()
with torch.no_grad(): l2, v2 = net2(O, M)
print('reload identical:', bool(torch.equal(l2, l1) and torch.equal(v2, v1)))
