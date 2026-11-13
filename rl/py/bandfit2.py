"""Band-head refit v2 (11-13): fit only the band encoder + band pointer head (enc.bands / ptr.bands) of a v3 checkpoint.
Targets per band-decision row (bandobs2): softmax(d fa / tau) over the row's legal bands from band_table2.json for the
measured settings — solo:ABYSS, coop4 ABYSS with 1-3 human proxies (coop4h:ABYSS), coop2 ABYSS with 1 human proxy
(coop2h1:ABYSS); a band below the table's minimum n scores 0 (neutral). Every other row (HARD / NORMAL, coop3, coop*l,
model-only rooms) keeps the checkpoint's own band distribution (distillation target). Everything else frozen: non-band
decisions keep their outputs exactly.
python bandfit2.py --ckpt in.pt --obs <bandobs2 prefix> --table band_table2.json --out out.pt [--tau-solo 0.03 --tau-coop 0.05 --steps 400]
  [--keep-from ck.pt: the distillation targets from this checkpoint's head instead of --ckpt's (e.g. a KL ref refit)]"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
from model import load, save
torch.set_num_threads(int(os.environ.get('THREADS', '4')))
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--obs', required=True); ap.add_argument('--table', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--tau-solo', type=float, default=0.03); ap.add_argument('--tau-coop', type=float, default=0.05); ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--keep-from', default='')
a = ap.parse_args()
dev = os.environ.get('DEV', 'cpu')
meta = json.load(open(a.obs + '.json')); n, D, A, b0, bl = meta['n'], meta['D'], meta['A'], meta['bandAction0'], meta['bandList']
O = torch.from_numpy(np.fromfile(a.obs + '.f32', np.float32).reshape(n, D)); M = torch.from_numpy(np.fromfile(a.obs + '.u8', np.uint8).reshape(n, A).astype(bool))
T = json.load(open(a.table))
def key_of(cfg, humans):
    mode, diff = cfg.split(':')
    if diff != 'ABYSS' or mode.endswith('l') or mode.endswith('ai'): return None
    if mode == 'solo': return 'solo:ABYSS'
    if mode.startswith('coop4') and 1 <= humans <= 3: return 'coop4h:ABYSS'
    if mode.startswith('coop2') and humans == 1: return 'coop2h1:ABYSS'
    return None
scores = {}
for key, t in T.items():
    by = {r['band']: r for r in t['bands'] if r.get('used')}
    scores[key] = np.array([by[b]['dfa'] if b in by else 0.0 for b in bl])
net, ck = load(a.ckpt, dev); O, M = O.to(dev), M.to(dev)
src = load(a.keep_from, dev)[0] if a.keep_from else net
with torch.no_grad(): lk, _ = src(O, M); keepP = F.softmax(lk[:, b0:b0 + len(bl)], 1).cpu().numpy()
keys = [key_of(c, h) for c, h in zip(meta['configs'], meta['humans'])]
TG = torch.zeros(n, A)
for i, key in enumerate(keys):
    legal = M[i, b0:b0 + len(bl)].cpu().numpy()
    if key is None or key not in scores: p = keepP[i] * legal; p = p / p.sum()
    else:
        tau = a.tau_solo if key.startswith('solo') else a.tau_coop
        z = np.where(legal, scores[key] / tau, -np.inf); z = z - z[legal].max(); p = np.where(legal, np.exp(z), 0.0); p /= p.sum()
    TG[i, b0:b0 + len(bl)] = torch.from_numpy(p).float()
TG = TG.to(dev)
for p in net.parameters(): p.requires_grad_(False)
train = list(net.ptr['bands'].parameters()) + list(net.enc['bands'].parameters())
for p in train: p.requires_grad_(True)
opt = torch.optim.Adam(train, lr=1e-3)
def ce():
    lg, _ = net(O, M); lp = F.log_softmax(lg, 1); return -(TG * lp.clamp_min(-1e4)).sum(1).mean(), lp
with torch.no_grad(): ce0, lp0 = ce()
probe = O[:64].clone(); bb = net.spec['blocks']['bands']; probe[:, bb['offset']:bb['offset'] + bb['count'] * bb['width']] = 0
with torch.no_grad(): ref, _ = net(probe, torch.ones(64, A, dtype=torch.bool, device=dev))
for k in range(a.steps):
    opt.zero_grad(); l, _ = ce(); l.backward(); opt.step()
with torch.no_grad():
    ce1, lp1 = ce(); now, _ = net(probe, torch.ones(64, A, dtype=torch.bool, device=dev))
print(json.dumps({'n': n, 'ce_before': round(float(ce0), 3), 'ce_after': round(float(ce1), 3), 'non_band_logit_change': float((now[:, :b0] - ref[:, :b0]).abs().max())}))
P0 = lp0.exp().cpu().numpy()[:, b0:b0 + len(bl)]; P1 = lp1.exp().cpu().numpy()[:, b0:b0 + len(bl)]; TGn = TG.cpu().numpy()[:, b0:b0 + len(bl)]
top = lambda v: ' '.join(f'{bl[j].replace("band_", "")}:{v[j]:.2f}' for j in np.argsort(-v)[:5])
for key in sorted(set(k for k in keys if k)) + [None]:
    idx = [i for i, k in enumerate(keys) if k == key]
    if not idx: continue
    tv = lambda P: float(0.5 * np.abs(P[idx] - TGn[idx]).sum(1).mean())
    print(f'{str(key):14s} rows {len(idx):4d} | old head TV to target {tv(P0):.3f} -> new {tv(P1):.3f} | new [{top(P1[idx].mean(0))}] | target [{top(TGn[idx].mean(0))}]')
save(a.out, net, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'bandfit2': {'table': a.table, 'tau_solo': a.tau_solo, 'tau_coop': a.tau_coop, 'ce_after': float(ce1)}})
print('saved', a.out)
