"""(l8, 11-13: + --init <head.pt> warm start (same d / layers))
Spatial normal-round picker (11-13): the v5 tile / unit encoder (model5.Tower) over each candidate layout's
observation (the decision's version 3 + ext.board observation with the ext.board blocks of that candidate: unit tokens with
bond-membership bias, 28 tile tokens with terrain, routes, occupant, own range coverage and block), pooled -> a score and
three regression heads (leaks, layers, kills of the candidate; the label is candidate − default). Labels: env pLabel (fmt p4).
Reports val R² of Δleaks / Δlayers / Δkills, and the realised gain of the picks vs the D-rule label.
  python picker_train5.py --data '<l5>/out/*/lab/*.jsonl' --spec <l5>/out/<job>/lab/spec.json --member $B/A/lookahead.json --out head5.pt
      [--max-dec 50000 --epochs 12 --d 128 --layers 2]"""
import argparse, glob, json, os, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this directory (model5.py)
from model5 import Tower, pool
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--spec', required=True); ap.add_argument('--member', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--max-dec', type=int, default=0); ap.add_argument('--K', type=int, default=12); ap.add_argument('--val-mod', type=int, default=20)
ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--bs', type=int, default=48); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=1e-4)
ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--hidden', type=int, default=256)
ap.add_argument('--tau', type=float, default=0.3); ap.add_argument('--default-bonus', type=float, default=0.05); ap.add_argument('--aux-w', type=float, default=1.0)
ap.add_argument('--layer-w', type=float, default=1.0); ap.add_argument('--leak-w', type=float, default=2.0); ap.add_argument('--kill-w', type=float, default=0.05)
ap.add_argument('--init', default=None); ap.add_argument('--eval-bs', type=int, default=128)  # (P2 11-13: picker_train5j = 5i + --eval-bs; K 36 OOMs at 128 under 2.8 GB)
ap.add_argument('--cache-dir', default='cache'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--device', default='cuda'); ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.cpu_threads)
spec = json.load(open(a.spec)); member = json.load(open(a.member))['pieceBonds']; D = spec['dim']
def emit(**k): print(json.dumps(k), flush=True)
t0 = time.time()
files = sorted(sum((glob.glob(p) for p in a.data.split(',')), []))
K = a.K
# packed-data cache: keyed by the label files (name + size), K and D; holds every record packed, split / subsampled after
import hashlib
ckey = hashlib.md5(json.dumps([[f, os.path.getsize(f)] for f in files] + [K, D]).encode()).hexdigest()[:12]
cpath = os.path.join(a.cache_dir, f'p5pack-{ckey}.pt') if a.cache_dir else None
def lg(s): return sum((s.get('lg') or {}).values()) if s else 0.0
def pack(R):
    N = len(R); off, L = R[0]['board']['off'], R[0]['board']['len']
    base = np.zeros((N, D), np.float16); sl = np.zeros((N, K, L), np.float16); m = np.zeros((N, K), bool)
    le = np.zeros((N, K), np.float32); ly = np.zeros((N, K), np.float32); ki = np.zeros((N, K), np.float32)
    for n, d in enumerate(R):
        base[n, d['obs']['i']] = d['obs']['v']
        for j, c in enumerate(d['cands'][:K]):
            s = d['scores'][j] if j < len(d['scores']) else None
            if s is None: continue
            sl[n, j, c['b']['i']] = c['b']['v']; m[n, j] = True; le[n, j] = s['leaks']; ly[n, j] = lg(s); ki[n, j] = s['kills']
    return dict(base=torch.from_numpy(base), sl=torch.from_numpy(sl), m=torch.from_numpy(m), le=torch.from_numpy(le), ly=torch.from_numpy(ly), ki=torch.from_numpy(ki), off=off, L=L)
if cpath and os.path.exists(cpath):
    ALL = torch.load(cpath); emit(event='cache', path=cpath, n=len(ALL['seed']))
else:
    recs = []
    for f in files:
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if d.get('fmt') != 'p4' or not d['scores'] or not d['scores'][0]: continue
            recs.append(d)
    ALL = pack(recs); ALL['seed'] = torch.tensor([d['seed'] for d in recs]); del recs
    if cpath: os.makedirs(a.cache_dir, exist_ok=True); torch.save(ALL, cpath + '.tmp'); os.replace(cpath + '.tmp', cpath)
g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(len(ALL['seed']), generator=g)
isval = (ALL['seed'][perm] % a.val_mod) == 0; vi, ti = perm[isval], perm[~isval]
if a.max_dec: ti = ti[:a.max_dec]
sub = lambda idx: {k: (v[idx] if torch.is_tensor(v) else v) for k, v in ALL.items()}
TR, VA = sub(ti), sub(vi)
emit(event='data', files=len(files), train=len(ti), val=len(vi), load_s=round(time.time() - t0, 1), cands_mean=float(TR['m'].float().sum(1).mean()))
dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.tower = Tower(spec, member, a.d, a.layers, a.heads, 2, True, True)
        s.head = nn.Sequential(nn.Linear(3 * a.d, a.hidden), nn.ReLU(), nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 4))
    def forward(s, base, sl, off, L):  # base (B, D), sl (B, K, L) -> (B, K, 4)
        B, Kk, _ = sl.shape; obs = base.unsqueeze(1).expand(B, Kk, D).clone(); obs[:, :, off:off + L] = sl
        x, kp, _ = s.tower(obs.reshape(B * Kk, D)); return s.head(pool(x, kp)).view(B, Kk, 4)
net = Net()
if a.init: net.load_state_dict(torch.load(a.init, map_location='cpu')['state']); emit(event='init', path=a.init)
net = net.to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
emit(event='model', params=sum(p.numel() for p in net.parameters()))
def deltas(X): return torch.stack([X['le'] - X['le'][:, :1], X['ly'] - X['ly'][:, :1], X['ki'] - X['ki'][:, :1]], -1)  # (N, K, 3)
def util(X):
    dl = X['ly'] - X['ly'][:, :1]; scale = dl.masked_fill(~X['m'], 0).abs().amax(1, keepdim=True).clamp_min(1.0)
    u = a.layer_w * dl / scale - a.leak_w * (X['le'] - X['le'][:, :1]).clamp_min(0) + a.kill_w * X['ki'] / X['ki'].amax(1, keepdim=True).clamp_min(1.0)
    u[:, 0] += a.default_bonus; return u.masked_fill(~X['m'], -1e9)
TR['u'], VA['u'] = util(TR), util(VA); TR['d'], VA['d'] = deltas(TR), deltas(VA)
_sel = TR['m'].clone(); _sel[:, 0] = False
dsd = torch.tensor([float(TR['d'][..., i][_sel].std()) + 1e-6 for i in range(3)])  # Δ scales of the regression heads
emit(event='dscale', leaks=float(dsd[0]), layers=float(dsd[1]), kills=float(dsd[2]))
def batch(X, idx):
    return (X['base'][idx].to(dev).float(), X['sl'][idx].to(dev).float(), X['m'][idx].to(dev), X['u'][idx].to(dev), X['d'][idx].to(dev))
def evaluate(X):
    net.eval(); P = []
    with torch.no_grad():
        for b in range(0, len(X['m']), a.eval_bs):
            idx = torch.arange(b, min(b + a.eval_bs, len(X['m']))); base, sl, m, u, dd = batch(X, idx); P.append(net(base, sl, X['off'], X['L']).cpu())
    net.train(); P = torch.cat(P); m = X['m']; dd = X['d']
    pd = P[..., 1:] - P[:, :1, 1:]; sel = m.clone(); sel[:, 0] = False
    r2 = [float(1 - ((pd[..., i] - dd[..., i])[sel] ** 2).mean() / dd[..., i][sel].var()) for i in range(3)]
    i = torch.arange(len(m)); pick = P[..., 0].masked_fill(~m, -1e9).argmax(1); lab = X['u'].argmax(1)
    g = lambda q: {'dLeaks': float(dd[i, q, 0].mean()), 'dLayers': float(dd[i, q, 1].mean()), 'dKills': float(dd[i, q, 2].mean())}
    # regression pick: argmax of −2 Δleaks + Δlayers / 5 + 0.05 Δkills (predicted), switch only above a margin
    ur = (-2 * pd[..., 0] + pd[..., 1] / 5 + 0.05 * pd[..., 2]).masked_fill(~sel, -1e9); rb = ur.argmax(1); rv = ur[i, rb]
    regp = {f'm{mg}': {**g(torch.where(rv > mg, rb, torch.zeros_like(rb))), 'switch': float((rv > mg).float().mean())} for mg in (0.0, 0.5, 1.0)}
    rnd = {'dLeaks': float(dd[..., 0][sel].mean()), 'dLayers': float(dd[..., 1][sel].mean()), 'dKills': float(dd[..., 2][sel].mean())}  # a random non-default candidate
    return {'n': len(m), 'randomNonDefault': rnd, 'r2': {'leaks': r2[0], 'layers': r2[1], 'kills': r2[2]}, 'pick': {**g(pick), 'nonDefault': float((pick != 0).float().mean())}, 'label': g(lab), 'regPick': regp}
best, best_state = -1e9, None
for ep in range(a.epochs):
    perm = torch.randperm(len(TR['m'])); t1 = time.time(); tl = 0
    for b in range(0, len(perm), a.bs):
        idx = perm[b:b + a.bs]; base, sl, m, u, dd = batch(TR, idx)
        P = net(base, sl, TR['off'], TR['L']); lg_ = P[..., 0].masked_fill(~m, -1e9)
        ce = -(F.softmax(u / a.tau, -1) * F.log_softmax(lg_, -1)).sum(-1).mean()
        pd = P[..., 1:] - P[:, :1, 1:]; sel = m.clone(); sel[:, 0] = False
        aux = (((pd - dd) / dsd.to(dev)) ** 2)[sel].mean() if sel.any() else torch.zeros((), device=dev)
        loss = ce + a.aux_w * aux; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item()
    ev = evaluate(VA); score = ev['r2']['leaks'] + ev['r2']['layers']
    emit(event='epoch', ep=ep, loss=round(tl / max(1, len(perm) // a.bs), 4), s=round(time.time() - t1, 1), val=ev)
    if score > best: best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state); ev = evaluate(VA)
torch.save({'state': best_state, 'args': vars(a), 'spec': a.spec, 'val': ev}, a.out); emit(event='final', val=ev, out=a.out)
