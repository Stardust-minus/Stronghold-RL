"""Sharpen a policy checkpoint: every final logit layer (scalar heads, pointer heads, version 4 move / equip / tile / target)
scaled by `factor` (logit temperature 1/factor); the value head and the rest untouched.
python sharpen.py --ckpt in.pt --out out.pt [--factor 2]"""
import argparse, torch
from model import load, save
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--factor', type=float, default=2.0)
a = ap.parse_args()
net, ck = load(a.ckpt, 'cpu')
layers = [net.scalar] + [h[-1] for h in net.ptr.values()]
for name in ('ptr4',):
    if hasattr(net, name): layers += [h[-1] for h in getattr(net, name).values()]
for name in ('tile', 'target'):
    if hasattr(net, name): layers.append(getattr(net, name)[-1])
if hasattr(net, 'scalar2'): layers.append(net.scalar2)
with torch.no_grad():
    for l in layers: l.weight.mul_(a.factor); l.bias.mul_(a.factor)
extra = {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')}; extra['sharpened'] = a.factor
save(a.out, net, extra)
print('sharpened', len(layers), 'logit layers x', a.factor, '->', a.out)
