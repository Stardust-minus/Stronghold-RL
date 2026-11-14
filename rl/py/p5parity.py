# torch Net (picker_train5) vs infer5 JS (export_p5) on recorded candidates: dump obs + torch outputs; the JS side reads the dump
import json, sys, os, numpy as np, torch
B = os.environ.get('SPRL_WORK', '.')  # the work dir holding the label / head inputs (was a host path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this directory (model5.py)
from model5 import Tower, pool
ckp, recf, out = sys.argv[1:4]
ck = torch.load(ckp, map_location='cpu'); a = ck['args']; spec = json.load(open(ck['spec'])); member = json.load(open(f'{B}/A/lookahead.json'))['pieceBonds']
import torch.nn as nn
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.tower = Tower(spec, member, a['d'], a['layers'], a['heads'], 2, True, True)
        s.head = nn.Sequential(nn.Linear(3 * a['d'], a['hidden']), nn.ReLU(), nn.Linear(a['hidden'], a['hidden']), nn.ReLU(), nn.Linear(a['hidden'], 4))
net = Net(); net.load_state_dict(ck['state']); net.eval()
D = spec['dim']; rows = []
for line in open(recf):
    d = json.loads(line)
    if d.get('fmt') != 'p4': continue
    base = np.zeros(D, np.float32); base[d['obs']['i']] = d['obs']['v']; off, L = d['board']['off'], d['board']['len']
    for c in d['cands'][:4]:
        o = base.copy(); o[off:off + L] = 0; o[[off + i for i in c['b']['i']]] = c['b']['v']; rows.append(o)
    if len(rows) >= 120: break
X = torch.from_numpy(np.stack(rows))
with torch.no_grad(): x, kp, _ = net.tower(X); P = net.head(pool(x, kp)).numpy()
json.dump({'obs': [[(i, float(v)) for i, v in enumerate(r) if v != 0] for r in rows], 'out': P.tolist(), 'D': D}, open(out, 'w'))
print('dumped', len(rows))
