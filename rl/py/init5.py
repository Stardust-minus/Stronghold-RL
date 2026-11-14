"""(A1, 11-13) A random-init model5.SPNet5 checkpoint for a spec taken from a record.mjs shard directory.
python init5.py --data <data-id dir> --member lookahead.json --cfg '{...}' --seed 0 --out init.pt"""
import argparse, glob, json
import torch
from model import save, param_count
from model5 import SPNet5
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--member', required=True); ap.add_argument('--cfg', default='{}')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)
spec = json.load(open(sorted(glob.glob(a.data + '/*.meta.json'))[0]))['spec']
cfg = json.loads(a.cfg); cfg.setdefault('critic', {'d': 192, 'layers': 3, 'heads': 4, 'ffn': 4})
net = SPNet5(spec, member=json.load(open(a.member))['pieceBonds'], **cfg)
save(a.out, net, {'init': 'random', 'seed': a.seed})
print(json.dumps({'out': a.out, 'params': param_count(net), 'critic': param_count(net.value), 'D': spec['dim'], 'ext': spec.get('ext')}))
