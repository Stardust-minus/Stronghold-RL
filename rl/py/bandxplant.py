"""Band-head transplant + check. Copies only the band encoder (enc.bands) and band pointer head (ptr.bands) weights of
--head into --dst (everything else, the value head included, untouched) and writes --out; then reports, on the band
decisions of --obs, the per-config band-pick distribution (mean probability, top-5) against the Step 0 table targets.
Without --head: check --dst only.
python bandxplant.py --dst ck.pt [--head fitted.pt --out out.pt] --obs <bandobs prefix> --table band_table.json [--tau-fa 0.03 --tau-win 0.05]"""
import argparse, json, sys, numpy as np, torch, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this directory (model.py)
from model import load, save
import os; torch.set_num_threads(int(os.environ.get("THREADS", "4")))
ap = argparse.ArgumentParser()
ap.add_argument('--dst', required=True); ap.add_argument('--head', default=''); ap.add_argument('--out', default='')
ap.add_argument('--obs', required=True); ap.add_argument('--table', required=True)
ap.add_argument('--tau-fa', type=float, default=0.03); ap.add_argument('--tau-win', type=float, default=0.05)
a = ap.parse_args()
dev = os.environ.get('DEV', 'cpu')
net, ck = load(a.dst, dev); net.eval()
if a.head:
    hn, _ = load(a.head, dev)
    sd, hs = net.state_dict(), hn.state_dict()
    keys = [k for k in sd if k.startswith('enc.bands.') or k.startswith('ptr.bands.')]
    assert keys and all(sd[k].shape == hs[k].shape for k in keys)
    before = {k: v.clone() for k, v in sd.items()}
    net.load_state_dict({**sd, **{k: hs[k] for k in keys}})
    after = net.state_dict()
    changed = [k for k in after if not torch.equal(after[k], before[k])]
    assert set(changed) <= set(keys), set(changed) - set(keys)
    print(json.dumps({'transplanted': keys, 'changed': len(changed), 'steps': ck.get('steps')}))
    if a.out:
        save(a.out, net, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'band_head_from': a.head}); print('saved', a.out)
meta = json.load(open(a.obs + '.json')); n, D, A, b0, bl = meta['n'], meta['D'], meta['A'], meta['bandAction0'], meta['bandList']
O = torch.from_numpy(np.fromfile(a.obs + '.f32', np.float32).reshape(n, D)); M = torch.from_numpy(np.fromfile(a.obs + '.u8', np.uint8).reshape(n, A).astype(bool))
T = json.load(open(a.table))
def row_of(cfg):
    mode, diff = cfg.split(':'); mode = 'coop4' if mode.startswith('coop') else 'solo'; key = f'{mode}:{"ABYSS" if diff == "ABYSS" else "HARD"}'
    t = T[key]; rnd = t.get('random') or {}; by = {r['band']: r for r in t['bands']}
    if diff == 'ABYSS': return key, np.array([by[b]['fa'] - rnd.get('fa', 0) if b in by else 0.0 for b in bl]), a.tau_fa
    return key, np.array([by[b].get('win', 0) - rnd.get('win', 0) if b in by else 0.0 for b in bl]), a.tau_win
TG = torch.zeros(n, A)
for i, cfg in enumerate(meta['configs']):
    _, s, tau = row_of(cfg); legal = M[i, b0:b0 + len(bl)].numpy()
    z = np.where(legal, s / tau, -np.inf); z = z - z[legal].max(); p = np.where(legal, np.exp(z), 0.0); p /= p.sum()
    TG[i, b0:b0 + len(bl)] = torch.from_numpy(p).float()
with torch.no_grad():
    lg, _ = net(O.to(dev), M.to(dev)); lp = F.log_softmax(lg, 1).cpu()
ce = float(-(TG * lp.clamp_min(-1e4)).sum(1).mean()); P = lp.exp()
ent = float(-(P[:, b0:b0 + len(bl)] * lp[:, b0:b0 + len(bl)].clamp_min(-1e4)).sum(1).mean())
print(json.dumps({'ckpt': a.out or a.dst, 'ce_vs_table': round(ce, 3), 'band_entropy': round(ent, 3), 'ln38': round(float(np.log(38)), 3)}))
for key in ('solo:ABYSS', 'coop4:ABYSS', 'solo:HARD', 'coop4:HARD'):
    idx = [i for i, c in enumerate(meta['configs']) if c == key]
    pm = P[idx, b0:b0 + len(bl)].mean(0).numpy(); tm = TG[idx, b0:b0 + len(bl)].mean(0).numpy()
    top = lambda v: ' '.join(f'{bl[j].replace("band_", "")}:{v[j]:.2f}' for j in np.argsort(-v)[:5])
    print(f'{key:12s} n={len(idx):3d} head [{top(pm)}] | table [{top(tm)}]')
