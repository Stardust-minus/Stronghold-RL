"""(A1, 11-13, arch) Behaviour cloning of the legacy teacher into model5.SPNet5 (tokenized entity transformer + its
separate critic): the same shards, split and targets as train_bc.py (policy cross-entropy + vf x return MSE), AdamW with
a linear warm-up and a cosine decay, bf16 autocast, best-val checkpoint.
python train_bc5.py --data <data-id dir> --out <dir> --member lookahead.json --cfg '{"d":128,"layers":2,...}' [--epochs 6]"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from data import load_shards, unpack_mask
from model import param_count, save
from model5 import SPNet5
torch.set_num_threads(4)

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--member', required=True, help='lookahead_table.mjs output (pieceBonds)')
ap.add_argument('--cfg', default='{}', help='SPNet5 keyword overrides, e.g. {"d":128,"layers":2,"critic":{"d":192,"layers":3}}')
ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=0.01)
ap.add_argument('--warmup', type=int, default=300); ap.add_argument('--vf', type=float, default=0.5)
ap.add_argument('--gamma', type=float, default=0.997); ap.add_argument('--val-every', type=int, default=20)
ap.add_argument('--max-train', type=int, default=0); ap.add_argument('--data-device', default='cuda')
ap.add_argument('--amp', type=int, default=1); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--mem-frac', type=float, default=0.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda')
if a.mem_frac > 0:
    torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
log = open(os.path.join(a.out, 'bc-log.jsonl'), 'a')
def emit(**kw):
    kw['t'] = round(time.time(), 1); print(json.dumps(kw), flush=True); log.write(json.dumps(kw) + '\n'); log.flush()

t0 = time.time()
shards = load_shards(a.data, a.gamma)
spec = shards[0][1]['spec']; A = spec['actions']
def stack(sel):
    obs = np.concatenate([np.asarray(s[2]['obs']) for s in sel])
    mask = np.concatenate([unpack_mask(np.asarray(s[2]['mask']), A) for s in sel])
    act = np.concatenate([np.asarray(s[2]['action']).astype(np.int64) for s in sel])
    ret = np.concatenate([s[3] for s in sel])
    return obs, mask, act, ret
val_sel = [s for i, s in enumerate(shards) if i % a.val_every == a.val_every - 1] or shards[-1:]
train_sel = [s for s in shards if s not in val_sel] or shards
tr, va = stack(train_sel), stack(val_sel)
if a.max_train and len(tr[0]) > a.max_train:
    idx = np.random.default_rng(0).choice(len(tr[0]), a.max_train, replace=False); idx.sort()
    tr = tuple(x[idx] for x in tr)
emit(event='data', train=len(tr[0]), val=len(va[0]), shards=len(shards), D=spec['dim'], ext=spec.get('ext'), load_s=round(time.time() - t0, 1),
     val_tags=[s[0] for s in val_sel])
ddev = torch.device(a.data_device)
def put(x):
    obs, mask, act, ret = x
    t = (torch.from_numpy(obs), torch.from_numpy(mask), torch.from_numpy(act), torch.from_numpy(ret))
    return tuple(y.to(ddev) for y in t) if ddev.type == 'cuda' else tuple(y.pin_memory() for y in t)
TR, VA = put(tr), put(va)
del tr, va
member = json.load(open(a.member))['pieceBonds']
cfg = json.loads(a.cfg)
cfg.setdefault('critic', {'d': 192, 'layers': 3, 'heads': 4, 'ffn': 4})
net = SPNet5(spec, member=member, **cfg).to(dev)
n_pol = param_count(net) - (param_count(net.value) if hasattr(net, 'value') else 0)
emit(event='model', params=param_count(net), policy_params=n_pol, critic_params=param_count(net.value), cfg={k: v for k, v in net.cfg.items() if k != 'member'})
decay = [p for n, p in net.named_parameters() if p.dim() >= 2 and 'emb' not in n and 'bond_id' not in n and 'typ' not in n]
nodecay = [p for n, p in net.named_parameters() if not (p.dim() >= 2 and 'emb' not in n and 'bond_id' not in n and 'typ' not in n)]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': nodecay, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, 0.98))
N = TR[0].shape[0]
steps_total = a.epochs * math.ceil(N / a.batch)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
act_groups = {'scalar': (0, 4), 'buy': (4, 14), 'sell': (14, 41), 'reward': (41, 47), 'personal': (47, 52), 'card': (52, 60), 'band': (60, 100)}
amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(a.amp))

def batch(D, idx):
    if ddev.type == 'cuda':
        return (D[0][idx].float(), D[1][idx], D[2][idx], D[3][idx])
    idc = idx.cpu()
    return (D[0][idc].to(dev, non_blocking=True).float(), D[1][idc].to(dev, non_blocking=True), D[2][idc].to(dev, non_blocking=True), D[3][idc].to(dev, non_blocking=True))

@torch.no_grad()
def evaluate(D):
    net.eval()
    ce = vl = 0.0; correct = 0; n = 0
    by = {k: [0, 0] for k in act_groups}
    M = D[0].shape[0]
    for i in range(0, M, 4096):
        idx = torch.arange(i, min(M, i + 4096), device=ddev)
        obs, mask, act, ret = batch(D, idx)
        with amp:
            logits, v = net(obs, mask)
        logits = logits.float(); v = v.float()
        ce += F.cross_entropy(logits, act, reduction='sum').item(); vl += F.mse_loss(v, ret, reduction='sum').item()
        ok = logits.argmax(1) == act; correct += ok.sum().item(); n += len(act)
        for k, (lo, hi) in act_groups.items():
            sel = (act >= lo) & (act < hi); by[k][0] += ok[sel].sum().item(); by[k][1] += sel.sum().item()
    net.train()
    return {'ce': ce / n, 'acc': correct / n, 'vmse': vl / n, 'acc_by': {k: round(c / t, 4) if t else None for k, (c, t) in by.items()}}

best = math.inf; step = 0; tl = time.time()
for ep in range(a.epochs):
    perm = torch.randperm(N, device=ddev)
    for i in range(0, N, a.batch):
        idx = perm[i:i + a.batch]
        obs, mask, act, ret = batch(TR, idx)
        with amp:
            logits, v = net(obs, mask)
        pl = F.cross_entropy(logits.float(), act); vl = F.mse_loss(v.float(), ret)
        loss = pl + a.vf * vl
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            emit(event='train', epoch=ep, step=step, ce=round(pl.item(), 4), vmse=round(vl.item(), 4), gn=round(float(gn), 3), lr=sched.get_last_lr()[0],
                 sps=round(100 * a.batch / (time.time() - tl))); tl = time.time()
    m = evaluate(VA)
    emit(event='val', epoch=ep, step=step, **m)
    if m['ce'] < best:
        best = m['ce']; save(os.path.join(a.out, 'bc.pt'), net, {'val': m, 'epoch': ep, 'args': vars(a)})
emit(event='done', best_ce=best, seconds=round(time.time() - t0, 1))
