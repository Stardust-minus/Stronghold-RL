"""Lock-step architecture / capacity probe on the BC shards (ppo3). One process: the shards resident on the GPU (obs f16,
mask bits, actions, kinds, returns), every run trained in lock-step on the same batch stream (each run its own
network, optimiser and cosine schedule), TF32 matmuls and bf16 autocast (losses, log-softmax and master weights fp32).
Same protocol as bc_probe.py: shard split (every val_every-th shard is validation), epochs, batch, lr, cosine, vf.
python bc_probe2.py --data <dataId dir> --out <dir> --runs 'a;w05;b;e8;aL;c;a,loss=class;a,seed=2;a,frac=0.25' [--epochs 40]
A run is `variant[,seed=S][,frac=F][,loss=class][,amp=fp32|bf16][,name=N]`. Runs with the same (seed, frac) share one
batch stream. Writes <out>/<name>.json (bc_probe.py's fields + harness, amp, samplesPerSec) and <name>.pt."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
import os, concurrent.futures as _cf
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
from data import load_shards
from model2 import SPNet2, flops_per_forward, deployed_macs
from bc_probe_variants import VARIANTS, GROUPS, GROUPS4, SPNET_JS_MS

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--runs', required=True)
ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--vf', type=float, default=0.0); ap.add_argument('--gamma', type=float, default=0.997)
ap.add_argument('--val-every', type=int, default=20); ap.add_argument('--amp', default='bf16', choices=['bf16', 'fp32'])
ap.add_argument('--tag', default='ls'); ap.add_argument('--lookahead', default=None)
ap.add_argument('--max-train', type=int, default=0, help='smoke test: first n training samples only')
ap.add_argument('--eval-sub', type=int, default=200_000); ap.add_argument('--compile', type=int, default=0)
ap.add_argument('--mem-frac', type=float, default=0.9)
ap.add_argument('--graphs', type=int, default=1, help='capture each run\'s whole train step in a CUDA graph (shared pool)')
ap.add_argument('--eval-chunk', type=int, default=16384, help='rows per evaluation forward (memory)')
ap.add_argument('--eval-init', type=int, default=0, help='evaluate every run before training (logged as epoch -1: a fine-tune\'s starting point)')
ap.add_argument('--set-loss', type=int, default=0, help='MOVE samples with a placement set (record.mjs --labels 1): -log sum of pi over the set')
ap.add_argument('--wd', type=float, default=0.01, help='AdamW weight decay on the Linear matrices')
ap.add_argument('--warmup', type=float, default=0.02, help='linear lr warm-up over this share of the steps')
ap.add_argument('--zloss', type=float, default=1e-4, help='z-loss weight on logsumexp^2 of the legal logits (keeps the logit scale from running away)')
ap.add_argument('--ckpt-every', type=int, default=0, help='save <name>-last.pt every n epochs (PPO can start from it)')
ap.add_argument('--host', type=int, default=0, help='keep the observations in host memory (version 4 data is larger than the GPU)')
ap.add_argument('--micro', type=int, default=0, help='micro-batch (gradient accumulation, eager mode): bounds activation memory; 0 = whole batch')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision('high')
torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)

# ---- data, resident on the GPU ----
t0 = time.time()
shards = load_shards(a.data, a.gamma)
spec = shards[0][1]['spec']; A = spec['actions']; D = spec['dim']; MB = shards[0][1]['maskBytes']
if A > 60: GROUPS = GROUPS4 if A >= 295 else {**GROUPS, 'band': (60, 100)}
val_ix = [i for i in range(len(shards)) if i % a.val_every == a.val_every - 1] or [len(shards) - 1]
def to_gpu(sel, limit=0):
    n = sum(len(shards[i][2]) for i in sel)
    if limit: n = min(n, limit)
    O = torch.empty((n, D), dtype=torch.float16, device='cpu' if a.host else dev); M = torch.empty((n, MB), dtype=torch.uint8, device=dev)
    AC = torch.empty(n, dtype=torch.long, device=dev); K = torch.empty(n, dtype=torch.uint8, device=dev); R = torch.empty(n, dtype=torch.float32, device=dev)
    AX = torch.zeros(n, dtype=torch.long, device=dev)
    o = 0
    for i in sel:
        rec, ret = shards[i][2], shards[i][3]; k = min(len(rec), n - o)
        if k <= 0: break
        for j in range(0, k, 500_000):  # host staging in pieces (a shard is a few hundred MB)
            e = min(k, j + 500_000); part = rec[j:e]
            O[o + j:o + e] = torch.from_numpy(np.ascontiguousarray(part['obs'])).to(O.device)
            M[o + j:o + e] = torch.from_numpy(np.ascontiguousarray(part['mask'])).to(dev)
            AC[o + j:o + e] = torch.from_numpy((part['action'].astype(np.int64) + (part['pad'].astype(np.int64) << 8))).to(dev)
            K[o + j:o + e] = torch.from_numpy(np.ascontiguousarray(part['kind'])).to(dev)
            R[o + j:o + e] = torch.from_numpy(ret[j:e]).to(dev)
            if 'aux' in part.dtype.names: AX[o + j:o + e] = torch.from_numpy(part['aux'].astype(np.int64)).to(dev)
        o += k
    return O, M, AC, K, R, AX
TR = to_gpu([i for i in range(len(shards)) if i not in val_ix], a.max_train)
VA = to_gpu(val_ix, a.max_train // 10 if a.max_train else 0)
N = TR[0].shape[0]; NV = VA[0].shape[0]
load_s = time.time() - t0
print(json.dumps({'loaded': N, 'val': NV, 'load_s': round(load_s, 1), 'gpu_gb': round(torch.cuda.memory_allocated() / 2**30, 2)}), flush=True)
SHIFT = torch.arange(8, device=dev, dtype=torch.uint8)
def unpack(bits):  # (B, MB) little-endian bit order -> (B, A) bool, on the device
    return ((bits.unsqueeze(-1) >> SHIFT) & 1).reshape(bits.shape[0], -1)[:, :A].bool()
def batch(Dt, idx):
    O, M, AC, K, R, AX = Dt
    o = O.index_select(0, idx.cpu()).to(dev, non_blocking=True) if O.device.type == 'cpu' else O.index_select(0, idx)
    return o.float(), unpack(M.index_select(0, idx)), AC.index_select(0, idx), K.index_select(0, idx), R.index_select(0, idx), AX.index_select(0, idx)
g_sub = torch.Generator(device='cpu').manual_seed(0)
SUB = torch.randperm(N, generator=g_sub)[:min(N, a.eval_sub)].to(dev)

# equivalence classes of slot actions (twin-aware loss / class accuracy): exact random projections per block
_g = torch.Generator().manual_seed(0); SPANS = []
for _name, _start in (('shop', 4), ('owned', 14), ('rewards', 41), ('personal', 47), ('cards', 52)):
    _b = spec['blocks'][_name]
    SPANS.append((_b['offset'], _b['count'], _b['width'], _start, torch.randn(_b['width'], generator=_g, dtype=torch.float64).to(dev)))
if A >= 295:  # version 4: moves, equips and targets pick owned rows too (tile / scalar actions stay distinct)
    _ob = spec['blocks']['owned']
    for _start in (100, 127, 268):
        SPANS.append((_ob['offset'], _ob['count'], _ob['width'], _start, torch.randn(_ob['width'], generator=_g, dtype=torch.float64).to(dev)))
def same_class(o, m, ac):
    B = o.shape[0]; key = torch.zeros((B, A), dtype=torch.float64, device=dev)
    key[:, :4] = -torch.arange(1, 5, device=dev, dtype=torch.float64)
    if A > 60: key[:, 60:] = -1e7 - torch.arange(A - 60, device=dev, dtype=torch.float64)  # distinct unless a span below sets them
    for bi, (off, cnt, w, start, proj) in enumerate(SPANS):
        rows = o[:, off:off + cnt * w].double().reshape(B, cnt, w)
        key[:, start:start + cnt] = rows @ proj + 1e6 * (bi + 1)
    return (key == key.gather(1, ac[:, None])) & m

# ---- runs ----
def parse_run(s):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    r = {'variant': parts[0], 'seed': 1, 'frac': 1.0, 'loss': 'ce', 'amp': a.amp, 'name': None, 'lr': a.lr, 'init': None, 'start': 0}
    for p in parts[1:]:
        k, v = p.split('='); r[k] = (int(v) if k in ('seed', 'start') else float(v) if k in ('frac', 'lr') else v)
    if r['variant'] not in VARIANTS: raise SystemExit(f'unknown variant {r["variant"]}')
    tag = a.tag + (f'f{int(round(r["frac"] * 100))}' if r['frac'] < 1 else '') + ('twin' if r['loss'] == 'class' else '') + ('fp32' if r['amp'] == 'fp32' and a.amp != 'fp32' else '')
    r['name'] = r['name'] or f"{r['variant']}-s{r['seed']}-{tag}"
    return r
runs = [parse_run(s) for s in a.runs.split(';') if s.strip()]
streams = {}
for r in runs:
    key = (r['seed'], r['frac'])
    if key not in streams:
        gen = torch.Generator(device='cpu').manual_seed(1000 + r['seed'])
        pool = torch.arange(N) if r['frac'] >= 1 else torch.sort(torch.randperm(N, generator=torch.Generator().manual_seed(7))[:int(N * r['frac'])]).values
        streams[key] = {'pool': pool.to(dev), 'pool_cpu': pool, 'gen': gen, 'perm': None, 'pos': 0, 'epoch': 0, 'steps_per_epoch': max(1, len(pool) // a.batch), 'cur': None, 'step': -1}
    r['stream'] = streams[key]
for r in runs:
    cfg = {k: (a.lookahead if v == '@lookahead' else v) for k, v in VARIANTS[r['variant']].items()}
    if 'lookahead' in VARIANTS[r['variant']] and not cfg.get('lookahead'): raise SystemExit('this variant needs --lookahead')
    torch.manual_seed(r['seed'])
    net = SPNet2(spec, **cfg).to(dev)
    if r['init']: net.load_state_dict(torch.load(r['init'], map_location=dev, weights_only=False)['state'])  # fine-tune from a checkpoint
    r['cfg'] = cfg; r['net'] = net; r['params'] = sum(p.numel() for p in net.parameters())
    r['base_lr'] = r['lr']; r['lr'] = torch.tensor(r['base_lr'], device=dev) if a.graphs else r['base_lr']
    # weight decay 0.01 on the Linear matrices only (no biases, LayerNorms, embeddings, type vectors)
    _mat = {id(m_.weight) for m_ in net.modules() if isinstance(m_, torch.nn.Linear)}
    _groups = [{'params': [p_ for p_ in net.parameters() if id(p_) in _mat], 'weight_decay': a.wd},
               {'params': [p_ for p_ in net.parameters() if id(p_) not in _mat], 'weight_decay': 0.0}]
    r['opt'] = torch.optim.AdamW(_groups, lr=r['lr'], fused=True, capturable=bool(a.graphs))
    r['attn_mods'] = [m_ for m_ in net.modules() if hasattr(m_, 'track')]
    r['skipped'] = 0; r['best_val'] = float('inf'); r['diverged_at'] = None
    total = a.epochs * r['stream']['steps_per_epoch']; r['total'] = total
    r['fwd'] = torch.compile(net, mode='reduce-overhead') if a.compile else net
    r['graph'] = None
    r['step'] = r['start'] * r['stream']['steps_per_epoch']; r['step0'] = r['step']; r['log'] = [];  # start: epochs already done (resume with init=)
    r['_unused'] = 0; r['log'] = []; r['gpu_ms'] = 0.0; r['ev'] = []; r['t_train'] = 0.0; r['done'] = False
print(json.dumps({'runs': [{k: r[k] for k in ('name', 'variant', 'seed', 'frac', 'loss', 'amp', 'params', 'total')} for r in runs]}), flush=True)

class Feeder:
    """Host-resident observations (--host): a worker thread gathers the next batch's rows into one of two pinned
    buffers (torch's intra-op threads) while the GPU trains on the current one; the H2D copy is asynchronous."""
    def __init__(self):
        self.ex = _cf.ThreadPoolExecutor(max_workers=1)
        self.pin = [torch.empty((a.batch, D), dtype=torch.float16).pin_memory() for _ in range(2)]
        self.ev = [None, None]; self.k = 0; self.fut = None
        self.gpu = torch.empty((a.batch, D), dtype=torch.float16, device=dev)
    def _gather(self, k, idx):
        if self.ev[k] is not None: self.ev[k].synchronize()  # the copy that last read this buffer has finished
        torch.index_select(TR[0], 0, idx, out=self.pin[k][:len(idx)])
        return k, len(idx)
    def prefetch(self, idx):
        self.fut = self.ex.submit(self._gather, self.k, idx); self.k ^= 1
    def take(self):
        k, n = self.fut.result()
        self.gpu[:n].copy_(self.pin[k][:n], non_blocking=True)
        ev = torch.cuda.Event(); ev.record(); self.ev[k] = ev
        return self.gpu[:n]

def _next_idx(st):
    if st['perm'] is None or st['pos'] >= st['steps_per_epoch']:
        st['perm'] = st['pool_cpu'][torch.randperm(len(st['pool_cpu']), generator=st['gen'])]; st['pos'] = 0
    idx = st['perm'][st['pos'] * a.batch:(st['pos'] + 1) * a.batch]; st['pos'] += 1
    return idx

def stream_batch(st, step):
    """The stream's batch for global step `step` (computed once per step, shared by its runs)."""
    if st['step'] == step: return st['cur']
    if a.host:
        if 'feeder' not in st:
            st['feeder'] = Feeder(); st['pending'] = _next_idx(st); st['feeder'].prefetch(st['pending'])
        idx = st['pending']; og = st['feeder'].take()
        st['pending'] = _next_idx(st); st['feeder'].prefetch(st['pending'])
        idg = idx.to(dev, non_blocking=True)
        O_, M_, AC_, K_, R_, AX_ = TR
        st['cur'] = (og.float(), unpack(M_.index_select(0, idg)), AC_.index_select(0, idg), K_.index_select(0, idg), R_.index_select(0, idg), AX_.index_select(0, idg))
        st['step'] = step
        return st['cur']
    if st['perm'] is None or st['pos'] >= st['steps_per_epoch']:
        st['perm'] = st['pool'][torch.randperm(len(st['pool']), generator=st['gen']).to(dev)]; st['pos'] = 0
    idx = st['perm'][st['pos'] * a.batch:(st['pos'] + 1) * a.batch]; st['pos'] += 1
    o, m, ac, k, rr, ax = batch(TR, idx)
    if a.graphs:  # static buffers the captured steps read
        if 'buf' not in st: st['buf'] = (o.clone(), m.clone(), ac.clone(), k.clone(), rr.clone(), ax.clone())
        else:
            for dst, src in zip(st['buf'], (o, m, ac, k, rr, ax)): dst.copy_(src)
        st['cur'] = st['buf']
    else:
        st['cur'] = (o, m, ac, k, rr, ax)
    st['step'] = step
    return st['cur']

def amp_ctx(r, graphs=False):
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=(r['amp'] == 'bf16'), cache_enabled=not graphs)

@torch.no_grad()
def evaluate(r, Dt, idx_all=None):
    net = r['net']; net.eval(); n = Dt[0].shape[0] if idx_all is None else len(idx_all)
    ce = vl = 0.0; ok_n = 0; cls_ok = 0; maxlog = 0.0; ents = None; set_ok = set_n = 0
    by = {g: [0, 0] for g in GROUPS}; byk = {'prep': [0, 0, 0.0], 'card': [0, 0, 0.0]}
    for i in range(0, n, a.eval_chunk):
        idx = torch.arange(i, min(n, i + a.eval_chunk), device=dev) if idx_all is None else idx_all[i:i + a.eval_chunk]
        o, m, ac, k, rr, ax = batch(Dt, idx)
        first = i == 0
        for mod in r['attn_mods']: mod.track = first
        with amp_ctx(r):
            lg, v = net(o, m)
        if first:
            ents = [mod.entropy for mod in r['attn_mods']]
            for mod in r['attn_mods']: mod.track = False
        lg = lg.float(); v = v.float()
        maxlog = max(maxlog, float(lg.masked_fill(~m, 0).abs().max()))
        ce_i = F.cross_entropy(lg, ac, reduction='none'); ce += ce_i.sum().item(); vl += F.mse_loss(v, rr, reduction='sum').item()
        pred = lg.argmax(1); ok = pred == ac; ok_n += ok.sum().item()
        cls_ok += same_class(o, m, ac).gather(1, pred[:, None]).sum().item()
        if A >= 295:
            sel = (ax != 0) & (ac >= 100) & (ac < 127); set_n += sel.sum().item()
            if sel.any(): set_ok += (((ax >> (pred - 100).clamp(0, 26)) & 1).bool() & (pred >= 100) & (pred < 127) & sel).sum().item()
        for g, (lo, hi) in GROUPS.items():
            sel = (ac >= lo) & (ac < hi); by[g][0] += ok[sel].sum().item(); by[g][1] += sel.sum().item()
        for kn, kv in (('prep', 0), ('card', 1)):
            sel = k == kv; byk[kn][0] += ok[sel].sum().item(); byk[kn][1] += sel.sum().item(); byk[kn][2] += ce_i[sel].sum().item()
    net.train()
    return {'ce': ce / n, 'acc': ok_n / n, 'classAcc': cls_ok / n, 'vmse': vl / n, 'maxLogit': maxlog, 'attnEntropy': ents, 'moveSetAcc': (set_ok / set_n) if set_n else None, 'acc_by': {g: (c / t if t else None) for g, (c, t) in by.items()},
            'by_kind': {kn: {'acc': c / t if t else None, 'ce': s / t if t else None, 'n': t} for kn, (c, t, s) in byk.items()}}

BIT27 = torch.arange(27, device=dev)
def pol_loss(lg, ac, ax):
    """Cross-entropy; with --set-loss, a MOVE sample with a placement set scores -log sum of pi over the set."""
    if not (a.set_loss and A >= 295): return F.cross_entropy(lg, ac)
    lp = F.log_softmax(lg, 1); nll = -lp.gather(1, ac[:, None]).squeeze(1)
    use = (ax != 0) & (ac >= 100) & (ac < 127)
    bits = ((ax[:, None] >> BIT27) & 1).bool()
    setl = -torch.logsumexp(lp[:, 100:127].masked_fill(~bits, -1e9), 1)
    return torch.where(use, setl, nll).mean()

def lr_at(r):
    w = min(1.0, (r['step'] - r['step0'] + 1) / max(1.0, a.warmup * r['total']))  # (re-)warm-up after a start or a resume
    return r['base_lr'] * w * 0.5 * (1 + math.cos(math.pi * min(1.0, r['step'] / r['total'])))

def set_lr(r):
    if a.graphs: r['lr'].fill_(lr_at(r))
    else:
        for g_ in r['opt'].param_groups: g_['lr'] = lr_at(r)

def _step_body(r, o, m, ac, rr, ax, graphs=False):
    net, opt = r['net'], r['opt']
    if a.micro and not graphs and o.shape[0] > a.micro:  # gradient accumulation: the same update as the whole batch
        opt.zero_grad(set_to_none=True); B = o.shape[0]
        for j in range(0, B, a.micro):
            sl = slice(j, j + a.micro); f = (min(B, j + a.micro) - j) / B
            with amp_ctx(r):
                lg, v = r['fwd'](o[sl], m[sl])
            lg = lg.float()
            if r['loss'] == 'class':
                lp = F.log_softmax(lg, 1); sc = same_class(o[sl], m[sl], ac[sl])
                pol = -torch.logsumexp(lp.masked_fill(~sc, -1e9), 1).mean()
            else:
                pol = pol_loss(lg, ac[sl], ax[sl])
            zl = a.zloss * (torch.logsumexp(lg, 1) ** 2).mean() if a.zloss > 0 else 0
            ((pol + zl + (a.vf * F.mse_loss(v.float(), rr[sl]) if a.vf > 0 else 0)) * f).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, foreach=True)
        if torch.isfinite(gn): opt.step()
        else: r['skipped'] += 1; opt.zero_grad(set_to_none=True)
        return
    with amp_ctx(r, graphs):
        lg, v = r['fwd'](o, m)
    lg = lg.float()
    if r['loss'] == 'class':
        lp = F.log_softmax(lg, 1); sc = same_class(o, m, ac)
        pol = -torch.logsumexp(lp.masked_fill(~sc, -1e9), 1).mean()
    else:
        pol = pol_loss(lg, ac, ax)
    loss = pol + (a.zloss * (torch.logsumexp(lg, 1) ** 2).mean() if a.zloss > 0 else 0) + (a.vf * F.mse_loss(v.float(), rr) if a.vf > 0 else 0)
    if not graphs: opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, foreach=True); opt.step()

POOL = [None]
def train_step(r, o, m, ac, rr, ax):
    set_lr(r)
    if not a.graphs or r.get('eager'):
        _step_body(r, o, m, ac, rr, ax); return
    if r['graph'] is not None:
        r['graph'].replay(); return
    if r.get('warm', 0) < 3:  # eager warm-up steps (real steps) on a side stream, then capture
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            _step_body(r, o, m, ac, rr, ax)
        torch.cuda.current_stream().wait_stream(s_)
        r['warm'] = r.get('warm', 0) + 1; return
    try:
        r['opt'].zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=POOL[0]):
            _step_body(r, o, m, ac, rr, ax, graphs=True)
        POOL[0] = POOL[0] or g.pool(); r['graph'] = g
        g.replay()  # the capture itself did not run the step
    except Exception as e:  # capture refused: this run stays eager
        print(json.dumps({'run': r['name'], 'graph_capture_failed': str(e)[:300]}), flush=True)
        torch.cuda.synchronize(); r['eager'] = True; r['graph'] = None
        _step_body(r, o, m, ac, rr, ax)

t1 = time.time(); step = 0; last_print = t1
if a.eval_init:
    for r in runs:
        e = {'epoch': -1, 'train': evaluate(r, TR, SUB), 'val': evaluate(r, VA), 'seconds': 0.0, 'gpu_s': 0.0}
        r['log'].append(e)
        print(json.dumps({'run': r['name'], 'epoch': -1, 'val_ce': round(e['val']['ce'], 4), 'val_acc': round(e['val']['acc'], 4),
                          'moveSetAcc': e['val'].get('moveSetAcc'), 'acc_by': {k: (round(v, 3) if v is not None else None) for k, v in e['val']['acc_by'].items()}}), flush=True)
while not all(r['done'] for r in runs):
    for r in runs:
        if r['done']: continue
        o, m, ac, k, rr, ax = stream_batch(r['stream'], step)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); train_step(r, o, m, ac, rr, ax); e1.record(); r['ev'].append((e0, e1))
        r['step'] += 1
        spe = r['stream']['steps_per_epoch']
        if r['step'] % spe == 0:
            torch.cuda.synchronize()
            r['gpu_ms'] += sum(x.elapsed_time(y) for x, y in r['ev']); r['ev'] = []
            ep = r['step'] // spe - 1
            e = {'epoch': ep, 'train': evaluate(r, TR, SUB), 'val': evaluate(r, VA), 'seconds': round(time.time() - t1, 1),
                 'gpu_s': round(r['gpu_ms'] / 1000, 1)}
            if e['val']['ce'] < r['best_val']: r['best_val'] = e['val']['ce']
            elif r['diverged_at'] is None and (not math.isfinite(e['val']['ce']) or e['val']['ce'] > 3 * r['best_val']): r['diverged_at'] = ep
            e['skipped'] = r['skipped']; e['diverged_at'] = r['diverged_at']
            r['log'].append(e)
            print(json.dumps({'run': r['name'], 'epoch': ep, 'moveSetAcc': e['val'].get('moveSetAcc'), 'acc_by': {k_: (round(v_, 3) if v_ is not None else None) for k_, v_ in e['val']['acc_by'].items()}, 'div': r['diverged_at'], 'maxLogit': round(e['val']['maxLogit'], 1), 'attnH': [round(x, 3) for x in (e['val']['attnEntropy'] or [])], 'val_ce': round(e['val']['ce'], 4), 'val_acc': round(e['val']['acc'], 4),
                              'train_ce': round(e['train']['ce'], 4), 'sec': e['seconds'],
                              'gpu_samples_per_s': round((r['step'] - r['step0']) * a.batch / max(1e-3, r['gpu_ms'] / 1000))}), flush=True)
            if a.ckpt_every and (ep + 1) % a.ckpt_every == 0:
                from model import save as _save
                _c = SPNet2(spec, **r['cfg']).eval(); _c.load_state_dict({k_: v_.float().cpu() for k_, v_ in r['net'].state_dict().items()})
                _save(os.path.join(a.out, r['name'] + '-last.pt'), _c, {'probe': e['val'], 'epoch': ep})
            if r['step'] >= r['total']:
                r['done'] = True; r['t_train'] = time.time() - t1
    step += 1
    if time.time() - last_print > 120:
        last_print = time.time()
        live = [r for r in runs if not r['done']]
        print(json.dumps({'step': step, 'wall_s': round(time.time() - t1), 'wall_samples_per_s_per_run': round(step * a.batch / (time.time() - t1)),
                          'live': len(live)}), flush=True)

# ---- results (bc_probe.py fields) ----
from model import save
VAo = VA[0][:2000].float().cpu(); VAm = unpack(VA[1][:2000]).cpu()
ref = SPNet2(spec, **VARIANTS['a']).eval(); ref_fl = flops_per_forward(ref, VAo[:64], VAm[:64])
for r in runs:
    net_cpu = SPNet2(spec, **r['cfg']).eval(); net_cpu.load_state_dict({k_: v_.float().cpu() for k_, v_ in r['net'].state_dict().items()})
    fl = flops_per_forward(net_cpu, VAo[:64], VAm[:64]); dm = deployed_macs(net_cpu, VAo[:500], VAm[:500])
    cost = {'deployedMacs': round(dm), 'msChrome4x': round(2.381 * dm / 1215748, 2), 'msChrome6x': round(3.570 * dm / 1215748, 2), 'fits10ms4x': bool(2.381 * dm / 1215748 <= 10)}
    gpu_s = r['gpu_ms'] / 1000
    res = {'variant': r['variant'], 'cfg': r['cfg'], 'seed': r['seed'], 'dataFrac': r['frac'], 'loss': r['loss'], 'amp': r['amp'], 'harness': 'lockstep',
           **cost, 'params': r['params'], 'train_samples': int(len(r['stream']['pool'])), 'val_samples': NV, 'load_s': round(load_s, 1),
           'train_s': round(r['t_train'], 1), 'gpu_s': round(gpu_s, 1), 'samplesPerSec': round((r['step'] - r['step0']) * a.batch / max(1e-3, gpu_s)),
           'flops': fl, 'flops_vs_a': fl / ref_fl, 'js_ms_est': round(SPNET_JS_MS * fl / ref_fl, 2), 'log': r['log'],
           'gpu_mem_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2), 'epochs': a.epochs, 'batch': a.batch, 'lr': r['base_lr'], 'vf': a.vf,
           'zloss': a.zloss, 'skipped': r['skipped'], 'diverged_at': r['diverged_at'], 'best_val_ce': r['best_val']}
    if not a.max_train:
        save(os.path.join(a.out, r['name'] + '.pt'), net_cpu, {'probe': r['log'][-1]['val'] if r['log'] else None})
    json.dump(res, open(os.path.join(a.out, r['name'] + '.json'), 'w'), indent=1)
    print('done', r['name'], 'params', r['params'], 'val_ce', round(r['log'][-1]['val']['ce'], 4) if r['log'] else None,
          'deployedMacs', round(dm), 'ms4x', cost['msChrome4x'], flush=True)
