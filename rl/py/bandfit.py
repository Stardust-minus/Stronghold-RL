"""Offline band-head warm start: fit only the band encoder + band pointer head of a v3 checkpoint to the Step 0 table
(softmax(score / tau) over the legal bands of each band decision; ABYSS: fa gain over the random pick, tau_fa; HARD /
NORMAL: win gain, tau_win; coop2 uses coop4's rows, NORMAL uses HARD's). Everything else frozen: non-band decisions keep
their outputs exactly (band rows are absent there, the trunk's band columns stay zero).
python bandfit.py --ckpt in.pt --obs <bandobs prefix> --table band_table.json --out out.pt [--tau-fa 0.03 --tau-win 0.05 --steps 400]"""
import argparse, json, sys, numpy as np, torch, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this directory (model.py)
from model import load, save
import os; torch.set_num_threads(int(os.environ.get("THREADS", "4")))
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--obs', required=True); ap.add_argument('--table', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--tau-fa', type=float, default=0.03); ap.add_argument('--tau-win', type=float, default=0.05); ap.add_argument('--steps', type=int, default=400)
a = ap.parse_args()
dev = os.environ.get('DEV', 'cpu')
meta = json.load(open(a.obs + '.json')); n, D, A, b0, bl = meta['n'], meta['D'], meta['A'], meta['bandAction0'], meta['bandList']
O = torch.from_numpy(np.fromfile(a.obs + '.f32', np.float32).reshape(n, D)); M = torch.from_numpy(np.fromfile(a.obs + '.u8', np.uint8).reshape(n, A).astype(bool))
T = json.load(open(a.table))
def row_of(cfg):
    mode, diff = cfg.split(':'); mode = 'coop4' if mode.startswith('coop') else 'solo'; key = f'{mode}:{"ABYSS" if diff == "ABYSS" else "HARD"}'
    t = T[key]; rnd = t.get('random') or {}
    by = {r['band']: r for r in t['bands']}
    if diff == 'ABYSS': s = np.array([by[b]['fa'] - rnd.get('fa', 0) if b in by else 0.0 for b in bl]); tau = a.tau_fa
    else: s = np.array([by[b].get('win', 0) - rnd.get('win', 0) if b in by else 0.0 for b in bl]); tau = a.tau_win
    return key, s, tau
TG = torch.zeros(n, A)
for i, cfg in enumerate(meta['configs']):
    _, s, tau = row_of(cfg); legal = M[i, b0:b0 + len(bl)].cpu().numpy()
    z = np.where(legal, s / tau, -np.inf); z = z - z[legal].max(); p = np.where(legal, np.exp(z), 0.0); p /= p.sum()
    TG[i, b0:b0 + len(bl)] = torch.from_numpy(p).float()
net, ck = load(a.ckpt, dev); O, M, TG = O.to(dev), M.to(dev), TG.to(dev)
for p in net.parameters(): p.requires_grad_(False)
train = list(net.ptr['bands'].parameters()) + list(net.enc['bands'].parameters())
for p in train: p.requires_grad_(True)
opt = torch.optim.Adam(train, lr=1e-3)
def ce():
    lg, _ = net(O, M); lp = F.log_softmax(lg, 1); return -(TG * lp.clamp_min(-1e4)).sum(1).mean(), lp
with torch.no_grad(): ce0, lp0 = ce()
# function preservation: logits on a non-band input must not change (the trunk, all other heads frozen)
probe = O[:64].clone(); probe[:, :] = O[:64]; bb = net.spec['blocks']['bands']; probe[:, bb['offset']:bb['offset'] + bb['count'] * bb['width']] = 0
with torch.no_grad(): ref, _ = net(probe, torch.ones(64, A, dtype=torch.bool, device=dev))
for k in range(a.steps):
    opt.zero_grad(); l, _ = ce(); l.backward(); opt.step()
with torch.no_grad():
    ce1, lp1 = ce(); now, _ = net(probe, torch.ones(64, A, dtype=torch.bool, device=dev))
print(json.dumps({'n': n, 'ce_before': round(float(ce0), 3), 'ce_after': round(float(ce1), 3), 'ln38': round(float(np.log(38)), 3),
                  'non_band_logit_change': float((now[:, :b0] - ref[:, :b0]).abs().max())}))
P1 = lp1.exp().cpu(); TG = TG.cpu()
for key in ('solo:ABYSS', 'coop4:ABYSS', 'solo:HARD', 'coop4:HARD'):
    idx = [i for i, c in enumerate(meta['configs']) if row_of(c)[0] == key and c == key]
    if not idx: continue
    pm = P1[idx, b0:b0 + len(bl)].mean(0).numpy(); tm = TG[idx, b0:b0 + len(bl)].mean(0).numpy()
    top = lambda v: [bl[j].replace('band_', '') for j in np.argsort(-v)[:5]]
    print(key, 'n', len(idx), '| head top5', top(pm), round(float(np.sort(pm)[-5:].sum()), 2), '| table top5', top(tm), round(float(np.sort(tm)[-5:].sum()), 2))
save(a.out, net, {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')} | {'bandfit': {'tau_fa': a.tau_fa, 'tau_win': a.tau_win, 'ce_after': float(ce1)}})
print('saved', a.out)
