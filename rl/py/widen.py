"""Width / structure A/B inits (11-13) from one SPNet checkpoint, function-preserving, with an equivalence check.
  python widen.py --ckpt W1.pt --obs obs.npz --out-dir dir [--width 2] [--entattn 1 --attn-dim 128 --heads 4 --qkln]
Writes dir/W1.pt (an exact copy), dir/W2.pt (SPNet2 width x: net2net widening of every hidden layer, embeddings kept) and
dir/AT.pt (SPNet2 width 1 + residual entity self-attention, zero-initialised output). Net2net here: each new unit copies an
old unit's incoming weights and bias; the outgoing weights of an old unit are split over its copies with random positive
fractions summing to 1 (per receiving unit), so the function is preserved exactly (up to float rounding) and the copies'
gradients differ from the first step (no symmetric twins). Reports max |d logit| (legal actions) and max |d value| vs W1."""
import argparse, json, os, numpy as np, torch
from model import load, save, SPNet
from model2 import SPNet2

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--obs', required=True); ap.add_argument('--out-dir', required=True)
ap.add_argument('--width', type=float, default=2.0); ap.add_argument('--entattn', type=int, default=1); ap.add_argument('--attn-dim', type=int, default=128)
ap.add_argument('--heads', type=int, default=4); ap.add_argument('--qkln', action='store_true'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
torch.set_num_threads(4)
src, ck = load(a.ckpt); src.eval()
assert type(src) is SPNet, 'widen.py starts from a plain SPNet checkpoint'
rng = np.random.default_rng(a.seed)
extra = {k: v for k, v in ck.items() if k not in ('state', 'spec', 'cfg')}

def umap(n_old, n_new):
    """new unit -> old unit: the first n_old are the old units, the rest copies (round-robin when n_new is a multiple)."""
    if n_new == n_old: return np.arange(n_old)
    assert n_new > n_old
    rest = np.arange(n_new - n_old) % n_old if (n_new % n_old == 0) else rng.choice(n_old, n_new - n_old, replace=n_new - n_old > n_old)
    return np.concatenate([np.arange(n_old), rest])

def split_fracs(n_out, gin):
    """(n_out, n_in_new) fractions: for each old input unit, positive random shares over its copies summing to 1 per row."""
    F = rng.uniform(0.5, 1.5, size=(n_out, len(gin)))
    for i in np.unique(gin):
        cols = np.where(gin == i)[0]
        F[:, cols] = F[:, cols] / F[:, cols].sum(1, keepdims=True) if len(cols) > 1 else 1.0
    return F

def widen_linear(old, new, gout, gin):
    W = old.weight.detach().double().numpy(); b = old.bias.detach().double().numpy()
    F = split_fracs(len(gout), gin)
    Wn = W[np.ix_(gout, gin)] * F
    with torch.no_grad():
        new.weight.copy_(torch.from_numpy(Wn).float()); new.bias.copy_(torch.from_numpy(b[gout]).float())

def build_w2():
    dst = SPNet2(src.spec, width=a.width)
    with torch.no_grad():
        for k in src.emb: dst.emb[k].weight.copy_(src.emb[k].weight)
    G = {}  # space -> unit map
    for n in src.names:
        e0, e2 = src.enc[n][0], src.enc[n][2]; d0, d2 = dst.enc[n][0], dst.enc[n][2]
        G[f'{n}.h'] = umap(e0.out_features, d0.out_features); G[f'{n}.o'] = umap(e2.out_features, d2.out_features)
        widen_linear(e0, d0, G[f'{n}.h'], np.arange(e0.in_features))
        widen_linear(e2, d2, G[f'{n}.o'], G[f'{n}.h'])
    G['bonds.h'] = umap(src.bonds[0].out_features, dst.bonds[0].out_features); G['bonds.o'] = umap(src.bonds[2].out_features, dst.bonds[2].out_features)
    widen_linear(src.bonds[0], dst.bonds[0], G['bonds.h'], np.arange(src.bonds[0].in_features))
    widen_linear(src.bonds[2], dst.bonds[2], G['bonds.o'], G['bonds.h'])
    # trunk input: [global, bonds, (mean, max) per entity block] (old column offsets)
    segs, off = [], 0
    def seg(space, n_old):
        nonlocal off; segs.append(G[space] + off); off += n_old
    seg('global.o', src.enc['global'][2].out_features); seg('bonds.o', src.bonds[2].out_features)
    for n in src.names:
        if n == 'global': continue
        no = src.enc[n][2].out_features; seg(f'{n}.o', no); seg(f'{n}.o', no)
    gin = np.concatenate(segs); assert off == src.trunk[0].in_features and len(gin) == dst.trunk[0].in_features
    G['H'] = umap(src.trunk[0].out_features, dst.trunk[0].out_features); G['C'] = umap(src.trunk[2].out_features, dst.trunk[2].out_features)
    widen_linear(src.trunk[0], dst.trunk[0], G['H'], gin)
    widen_linear(src.trunk[2], dst.trunk[2], G['C'], G['H'])
    widen_linear(src.scalar, dst.scalar, np.arange(src.scalar.out_features), G['C'])
    for n in src.pointer:
        p0, p2 = src.ptr[n][0], src.ptr[n][2]; q0, q2 = dst.ptr[n][0], dst.ptr[n][2]
        no = src.enc[n][2].out_features
        gin = np.concatenate([G[f'{n}.o'], G['C'] + no]); G[f'ptr.{n}'] = umap(p0.out_features, q0.out_features)
        widen_linear(p0, q0, G[f'ptr.{n}'], gin); widen_linear(p2, q2, np.arange(1), G[f'ptr.{n}'])
    G['v'] = umap(src.value[0].out_features, dst.value[0].out_features)
    widen_linear(src.value[0], dst.value[0], G['v'], G['C']); widen_linear(src.value[2], dst.value[2], np.arange(1), G['v'])
    return dst

def build_at():
    dst = SPNet2(src.spec, width=1.0, entattn=a.entattn, attn_dim=a.attn_dim, heads=a.heads, qkln=a.qkln)
    sd = dst.state_dict(); ss = src.state_dict()
    missing = [k for k in ss if k not in sd]; assert not missing, missing
    for k, v in ss.items():
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape); sd[k] = v.clone()
    dst.load_state_dict(sd)
    new = sorted({k.split('.')[0] for k in sd if k not in ss}); return dst, new

d = np.load(a.obs); O = torch.from_numpy(d['obs']); M = torch.from_numpy(d['mask']); K = d['kinds'] % 16
@torch.no_grad()
def out(net):
    net.eval(); lg, v = [], []
    for i in range(0, len(O), 1024):
        l, vv = net(O[i:i + 1024], M[i:i + 1024]); lg.append(l); v.append(vv)
    return torch.cat(lg).double(), torch.cat(v).double()
L1, V1 = out(src)
def compare(net, name):
    L, V = out(net); dl = (L - L1).abs().masked_fill(~M, 0)
    p1, p = torch.softmax(L1, 1), torch.softmax(L, 1)
    r = {'arm': name, 'params': int(sum(x.numel() for x in net.parameters())), 'rows': int(len(O)),
         'max_dlogit': float(dl.max()), 'max_dvalue': float((V - V1).abs().max()), 'max_dprob': float((p - p1).abs().max()),
         'argmax_same': float((L.argmax(1) == L1.argmax(1)).double().mean()), 'logit_scale': float(L1.masked_fill(~M, 0).abs().max()),
         'per_kind_max_dlogit': {int(k): float(dl[torch.from_numpy(K == k)].max()) for k in np.unique(K)}}
    print(json.dumps(r)); return r
res = [compare(src, 'W1')]
w1 = os.path.join(a.out_dir, 'W1.pt'); save(w1, src, extra | {'arm': 'W1'})
w2 = build_w2(); res.append(compare(w2, 'W2')); save(os.path.join(a.out_dir, 'W2.pt'), w2, extra | {'arm': 'W2', 'widened_from': a.ckpt})
at, new = build_at(); r = compare(at, 'AT'); r['new_modules'] = new; res.append(r)
save(os.path.join(a.out_dir, 'AT.pt'), at, extra | {'arm': 'AT', 'from': a.ckpt})
# reload check: the saved files build the same networks
for nm in ('W2', 'AT'):
    net2, _ = load(os.path.join(a.out_dir, f'{nm}.pt')); L, V = out(net2); ref = out(w2 if nm == 'W2' else at)
    print(nm, 'reload identical:', bool(torch.equal(L, ref[0]) and torch.equal(V, ref[1])))
json.dump(res, open(os.path.join(a.out_dir, 'equivalence.json'), 'w'), indent=1)
