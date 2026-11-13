"""Band-head swap (11-13): copy enc.bands.* / ptr.bands.* of --head into --dst (any SPNet / SPNet2 with the same band shapes),
check that nothing else changed, write --out. python swaphead.py --dst ck.pt --head fitted.pt --out out.pt"""
import argparse, torch
from model import load, save
ap = argparse.ArgumentParser(); ap.add_argument('--dst', required=True); ap.add_argument('--head', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
net, ck = load(a.dst); hn, _ = load(a.head)
sd, hs = net.state_dict(), hn.state_dict()
keys = [k for k in sd if k.startswith('enc.bands.') or k.startswith('ptr.bands.')]
assert keys and all(sd[k].shape == hs[k].shape for k in keys)
before = {k: v.clone() for k, v in sd.items()}
net.load_state_dict({**sd, **{k: hs[k] for k in keys}})
after = net.state_dict(); changed = [k for k in after if not torch.equal(after[k], before[k])]
assert set(changed) <= set(keys), set(changed) - set(keys)
save(a.out, net, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'band_head_from': a.head})
print({'dst': a.dst, 'steps': ck.get('steps'), 'band_tensors': len(keys), 'changed': len(changed), 'out': a.out})
