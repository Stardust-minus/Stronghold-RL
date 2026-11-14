"""(A3, 11-13) A recurrent model5.SPNet5 init from a feed-forward one: the same weights plus a zero-init residual recurrent
branch (cfg rnn {cell: lstm|gru, hidden}); the policy's outputs are unchanged at init (rnn_out = 0).
python addrnn.py --init bc.pt --cell lstm --hidden 128 --seed 0 --out init.pt"""
import argparse, json
import torch
from model import load, save, param_count
from model5 import SPNet5
ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True); ap.add_argument('--cell', choices=['lstm', 'gru'], required=True)
ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)
src, ck = load(a.init, 'cpu')
assert src.cfg.get('arch') == 'v5' and not src.cfg.get('rnn')
cfg = dict(src.cfg); cfg['rnn'] = {'cell': a.cell, 'hidden': a.hidden}
net = SPNet5(src.spec, **cfg)
missing, unexpected = net.load_state_dict(src.state_dict(), strict=False)
assert not unexpected and all(k.startswith('rnn_') for k in missing), (missing, unexpected)
save(a.out, net, {'from': a.init, 'rnn': cfg['rnn'], 'seed': a.seed})
print(json.dumps({'out': a.out, 'params': param_count(net), 'new': sorted(missing), 'state_size': net.state_size}))
