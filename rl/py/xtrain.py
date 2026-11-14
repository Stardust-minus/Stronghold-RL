"""xplace (11-13): train the learned layout policy (XNet) on search labels from tools/rl/xlabel.mjs.

Autoregressive pointer over (tile, facing) per fielded piece, pieces in the featurizer's canonical order (xfeat.mjs xOrder).
Loss: cross-entropy of the label layout (the search pick) under teacher forcing; optional listwise ranking over the
evaluated candidates (--rank-w). Exports <out>.json for the JS forward pass (tools/rl/xplace.mjs).
  python xtrain.py --data 'dir/*.jsonl' --out x1 --epochs 20
"""
import argparse, glob, json, math, os, random, time, hashlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

TILES, NMAX = 28, 12
R0, C0, NR, NC = 9, 3, 4, 7
DV = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # UP RIGHT DOWN LEFT (sim/dir.js: UP = +row)


def tables():
    def idx(r, c):
        return (r - R0) * NC + (c - C0) if R0 <= r < R0 + NR and C0 <= c < C0 + NC else TILES
    Fr = np.full((TILES, 4), TILES, np.int64); Bk = np.full((TILES, 4), TILES, np.int64)
    N4 = np.full((TILES, 4), TILES, np.int64); RW = np.full((TILES, NC - 1), TILES, np.int64)
    for t in range(TILES):
        r, c = R0 + t // NC, C0 + t % NC
        for d, (dr, dc) in DV.items():
            Fr[t, d] = idx(r + dr, c + dc); Bk[t, d] = idx(r - dr, c - dc); N4[t, d] = idx(r + dr, c + dc)
        RW[t] = [idx(r, cc) for cc in range(C0, C0 + NC) if cc != c]
    return Fr, Bk, N4, RW


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class XNet(nn.Module):
    def __init__(s, FP, FT, FE, FG, V, d=96, de=16, attn=0, heads=4):
        super().__init__()
        s.cfg = dict(FP=FP, FT=FT, FE=FE, FG=FG, V=V, d=d, de=de, attn=attn, heads=heads)
        s.att = nn.ModuleList([nn.ModuleDict(dict(ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d), ln2=nn.LayerNorm(d), ff=mlp(d, 2 * d, d)))
                               for _ in range(attn)])
        s.emb = nn.Embedding(V, de)
        s.pe = mlp(FP + de, d, d); s.te = mlp(FT, d, d); s.ge = mlp(FG + 2 * d, d, d)
        s.A = nn.Linear(d, d); s.Bt = nn.Linear(d, d, bias=False); s.Cd = nn.Parameter(torch.zeros(4, d))
        s.We = nn.Linear(FE, d, bias=False); s.G = nn.Linear(d, d, bias=False)
        s.occ = nn.Linear(d + 4, d)
        s.R = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(6)])  # front, back, nbr, row, facedBy, backBy
        s.h1 = nn.Linear(d, d); s.out = nn.Linear(d, 1)
        Fr, Bk, N4, RW = tables()
        for k, v in dict(Fr=Fr, Bk=Bk, N4=N4, RW=RW).items(): s.register_buffer(k, torch.from_numpy(v), persistent=False)

    def encode(s, fp, ids, ft, fg, pm):
        hp = s.pe(torch.cat([fp, s.emb(ids)], -1))  # [B,n,d]
        B, n, d = hp.shape; H = s.cfg['heads']; dh = d // H
        for L in s.att:  # pre-LN self-attention over the fielded pieces
            q, k, v = L['qkv'](L['ln1'](hp)).split(d, -1)
            q, k, v = (x.reshape(B, n, H, dh).transpose(1, 2) for x in (q, k, v))
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            sc = sc.masked_fill(~pm[:, None, None, :], -1e9)
            o = (torch.softmax(sc, -1) @ v).transpose(1, 2).reshape(B, n, d)
            hp = hp + L['proj'](o); hp = hp + L['ff'](L['ln2'](hp))
        ht = s.te(ft)  # [B,28,d]
        pmf = pm.unsqueeze(-1).float()
        mean = (hp * pmf).sum(1) / pmf.sum(1).clamp(min=1)
        mx = hp.masked_fill(~pm.unsqueeze(-1), -1e4).max(1).values
        g = s.ge(torch.cat([fg, mean, mx], -1))
        return hp, ht, g

    def forward(s, fp, ids, ft, fe, fg, legal, pm, tgt):
        """Teacher-forced logits [B,n,28,4] (masked) for target assignment tgt [B,n,2] (tile, dir; -1 pad)."""
        B, n = fp.shape[:2]
        hp, ht, g = s.encode(fp, ids, ft, fg, pm)
        tt, td = tgt[..., 0].clamp(min=0), tgt[..., 1].clamp(min=0)
        valid = pm & (tgt[..., 0] >= 0)
        occ = F.relu(s.occ(torch.cat([hp, F.one_hot(td, 4).float()], -1))) * valid.unsqueeze(-1).float()  # [B,n,d]
        T1 = TILES + 1
        def excl_cum(idx):  # per step i: sum over j < i of onehot(idx_j) x occ_j -> [B,n,29,d]
            S = F.one_hot(torch.where(valid, idx, torch.full_like(idx, TILES)), T1).float()  # [B,n,29]
            X = S.unsqueeze(-1) * occ.unsqueeze(2)  # [B,n,29,d]
            c = X.cumsum(1)
            return torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], 1), S
        OccT, S = excl_cum(tt)
        Face, _ = excl_cum(s.Fr[tt, td]); Back, _ = excl_cum(s.Bk[tt, td])
        occupied = torch.cat([torch.zeros_like(S[:, :1]), S.cumsum(1)[:, :-1]], 1)[..., :TILES] > 0  # [B,n,28]
        P = [s.R[k](OccT) for k in range(4)]  # [B,n,29,d]
        front = P[0][:, :, s.Fr.reshape(-1)].reshape(B, n, TILES, 4, -1)
        back = P[1][:, :, s.Bk.reshape(-1)].reshape(B, n, TILES, 4, -1)
        nbr = P[2][:, :, s.N4.reshape(-1)].reshape(B, n, TILES, 4, -1).sum(3)
        row = P[3][:, :, s.RW.reshape(-1)].reshape(B, n, TILES, NC - 1, -1).sum(3)
        fb = s.R[4](Face[:, :, :TILES]) + s.R[5](Back[:, :, :TILES])
        z = (s.A(hp)[:, :, None, None] + s.Bt(ht)[:, None, :, None] + s.Cd[None, None, None] + s.We(fe) + s.G(g)[:, None, None, None]
             + front + back + (nbr + row + fb)[:, :, :, None])
        lg = s.out(F.relu(s.h1(F.relu(z)))).squeeze(-1)  # [B,n,28,4]
        ok = legal & ~occupied[..., None]
        return lg.masked_fill(~ok, -1e9), ok


class QNet(XNet):
    """Layout scorer Q(s, L): every piece's embedding in the full layout L (its tile / facing, pair features, the occupants
    in front / behind / around / in its row and the pieces facing or backing onto it), self-attention over the placed
    pieces, then sum / max pooling with the state context -> a scalar (trained listwise within a decision)."""
    def __init__(s, FP, FT, FE, FG, V, d=96, de=16, attn=0, heads=4, qattn=1, cf=0):
        super().__init__(FP, FT, FE, FG, V, d=d, de=de, attn=attn, heads=heads)
        s.cfg.update(kind='q', qattn=qattn, cf=cf)
        if cf: s.cfe = mlp(2 * cf, d, d)
        s.qatt = nn.ModuleList([nn.ModuleDict(dict(ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d), ln2=nn.LayerNorm(d), ff=mlp(d, 2 * d, d)))
                                for _ in range(qattn)])
        s.qh = mlp((4 if cf else 3) * d, d, 1); s.aux = mlp((4 if cf else 3) * d, d, 4)  # aux: leaks, layers / 10, kills / 10, boss share
        del s.out

    def score(s, fp, ids, ft, fe, fg, legal, pm, tgt, cf=None, cf0=None):
        """Q of layouts tgt [B,n,2] (tile, dir; -1 = unplaced piece) -> [B]."""
        B, n = fp.shape[:2]
        hp, ht, g = s.encode(fp, ids, ft, fg, pm)
        valid = pm & (tgt[..., 0] >= 0)
        tt, td = tgt[..., 0].clamp(min=0), tgt[..., 1].clamp(min=0)
        occ = F.relu(s.occ(torch.cat([hp, F.one_hot(td, 4).float()], -1))) * valid.unsqueeze(-1).float()
        T1 = TILES + 1
        def tot(idx):
            S = F.one_hot(torch.where(valid, idx, torch.full_like(idx, TILES)), T1).float()
            return torch.einsum('bnt,bnd->btd', S, occ)  # [B,29,d]
        OccT = tot(tt); Face = tot(s.Fr[tt, td]); Back = tot(s.Bk[tt, td])
        bi = torch.arange(B, device=fp.device)[:, None]
        P = [s.R[k](OccT) for k in range(4)]
        front = P[0][bi, s.Fr[tt, td]]; back = P[1][bi, s.Bk[tt, td]]
        nbr = P[2][bi[..., None], s.N4[tt]].sum(2); row = P[3][bi[..., None], s.RW[tt]].sum(2)
        fb = s.R[4](Face)[bi, tt] + s.R[5](Back)[bi, tt]
        we = s.We(fe[bi, torch.arange(n, device=fp.device)[None], tt, td])
        z = s.A(hp) + s.Bt(ht)[bi, tt] + s.Cd[td] + we + s.G(g)[:, None] + front + back + nbr + row + fb
        u = F.relu(s.h1(F.relu(z))) * valid.unsqueeze(-1).float()
        d = u.shape[-1]; H = s.cfg['heads']; dh = d // H
        for L in s.qatt:
            q, k, v = L['qkv'](L['ln1'](u)).split(d, -1)
            q, k, v = (x.reshape(B, n, H, dh).transpose(1, 2) for x in (q, k, v))
            sc = ((q @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~valid[:, None, None, :], -1e9)
            o = (torch.softmax(sc, -1) @ v).transpose(1, 2).reshape(B, n, d)
            u = u + L['proj'](o); u = u + L['ff'](L['ln2'](u)); u = u * valid.unsqueeze(-1).float()
        vf = valid.unsqueeze(-1).float()
        pooled = torch.cat([(u * vf).sum(1) / 4, u.masked_fill(~valid.unsqueeze(-1), -1e4).max(1).values.clamp(min=-1e3), g], -1)
        if s.cfg.get('cf'): pooled = torch.cat([pooled, s.cfe(torch.cat([cf, cf - cf0], -1))], -1)
        s._aux = s.aux(pooled)
        return s.qh(pooled).squeeze(-1)


AUX_W = 0.0; EU_W = 0.0; NLL_W = 1.0
def q_epoch(net, D, idx, dev, opt=None, bs=48, M=12, tau=2.0):
    """Listwise loss over M sampled candidates per decision: CE(softmax(Q), softmax(utility / tau))."""
    net.train(opt is not None); tot = dict(loss=0.0, n=0, top1=0.0, ndec=0)
    for k in range(0, len(idx), bs):
        bi = idx[k:k + bs]; B = len(bi)
        sel = np.zeros((B, M), np.int64); msk = np.zeros((B, M), bool)
        for q, i in enumerate(bi):
            ok = np.nonzero(D.cm[i])[0]; c0 = D.ccur[i]; rest = ok[ok != c0]
            ch = np.concatenate([[c0], rest if len(rest) <= M - 1 else np.random.choice(rest, M - 1, replace=False)])
            sel[q, :len(ch)] = ch; msk[q, :len(ch)] = D.cm[i][ch]
        b, cur, boss, w = D.batch(bi, dev)
        tgt = torch.from_numpy(D.ca[bi][np.arange(B)[:, None], sel].astype(np.int64)).to(dev).reshape(B * M, NMAX, 2)
        u = torch.from_numpy(D.cu[bi][np.arange(B)[:, None], sel]).to(dev); mk = torch.from_numpy(msk).to(dev)
        rep = lambda x: x.repeat_interleave(M, 0)
        with torch.set_grad_enabled(opt is not None):
            cfs = torch.from_numpy(D.cf[bi][np.arange(B)[:, None], sel].astype(np.float32)).to(dev)
            cf0 = cfs[:, :1].expand(B, M, XCF).reshape(B * M, XCF)
            Q = net.score(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), tgt, cfs.reshape(B * M, XCF), cf0).reshape(B, M)
            Q = Q.masked_fill(~mk, -1e9); tq = torch.softmax((u / tau).masked_fill(~mk, -1e9), -1)
            use = mk.sum(1) >= 2
            l = -(tq * torch.log_softmax(Q, -1)).sum(1)
            loss = (l * w)[use].mean() if use.any() else Q.sum() * 0
            if EU_W > 0:  # expected utility of a softmax(Q) pick (cost-sensitive: mass on a worse layout costs its loss)
                pq = torch.softmax(Q, -1); loss = loss - EU_W * ((pq * u.masked_fill(~mk, 0)).sum(1) * w)[use].mean() / 10
            if AUX_W > 0:  # deltas vs the current layout (slot 0 of the sample), seed 0
                ax = net._aux.reshape(B, M, 4); cs_ = torch.from_numpy(D.cs[bi][np.arange(B)[:, None], sel]).to(dev)
                da = ax - ax[:, :1]; dt = cs_ - cs_[:, :1]; am = mk & mk[:, :1]
                al = F.smooth_l1_loss(da, dt, reduction='none').sum(-1)
                loss = loss + AUX_W * (al * am.float()).sum() / am.float().sum().clamp(min=1)
            if opt is not None:
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            top = Q.argmax(1); ub = u.masked_fill(~mk, -1e9).max(1).values
            tot['top1'] += float(((u.gather(1, top[:, None]).squeeze(1) >= ub - 1e-6) & use).sum()); tot['ndec'] += int(use.sum())
            tot['loss'] += float(loss) * B; tot['n'] += B
    return dict(loss=tot['loss'] / max(1, tot['n']), top1=tot['top1'] / max(1, tot['ndec']))


@torch.no_grad()
def q_picker_report(net, D, recs, idx, dev, srcs=('cur', 'leg', 'pert'), margin=0.0, rule=None):
    """Offline: argmax Q over the deploy-available candidates (no sims needed to generate them), seed-0 utilities."""
    net.eval(); st = dict(n=0, uModel=0.0, uOracle=0.0, cur=0, better=0, worse=0, ncand=0)
    global WCTX
    for i in idx:
        r = recs[i]; cs = r['cands']; n = min(r['n'], NMAX); WCTX = wctx(r)
        js = [j for j, c in enumerate(cs) if c['src'] in srcs and c['s'] and c['s'][0]]
        if len(js) < 2 or not cs[0]['s'][0]: continue
        u = [util(r['boss'], cs[j]['s'][0], cs[0]['s'][0]) for j in js]
        tg = np.full((len(js), NMAX, 2), -1, np.int64)
        for q, j in enumerate(js): tg[q, :n] = np.array(cs[j]['a'][:n])
        b, cur, boss, w = D.batch(np.array([i]), dev); M = len(js)
        rep = lambda x: x.expand(M, *x.shape[1:]).contiguous()
        cfa = np.zeros((M, XCF), np.float32)
        for q, j in enumerate(js):
            if cs[j].get('f') and len(cs[j]['f']) == XCF: cfa[q] = cs[j]['f']
        cft = torch.from_numpy(cfa).to(dev)
        Q = net.score(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), torch.from_numpy(tg).to(dev), cft, cft[:1].expand(M, XCF)).cpu().numpy()
        k = int(np.argmax(Q))
        if rule is not None:  # aux-head rule D: predicted extra leaks below rule[0], then the most predicted layers (> margin)
            ax = net._aux.cpu().numpy(); dl = ax[:, 0] - ax[0, 0]; dy = (ax[:, 1] - ax[0, 1]) * 10; db = ax[:, 3] - ax[0, 3]
            v = np.where(dl <= rule[0], (db * 10 if r['boss'] else dy), -1e9); v[0] = 0.0; k = int(np.argmax(v))
            if v[k] <= margin: k = 0
        elif Q[k] - Q[0] <= margin: k = 0  # js[0] is the current layout
        st['n'] += 1; st['uModel'] += u[k]; st['uOracle'] += max(u); st['cur'] += js[k] == 0; st['better'] += u[k] > 1e-9; st['worse'] += u[k] < -1e-9; st['ncand'] += len(js)
    n = max(1, st['n']); return {k: (v if k == 'n' else v / n) for k, v in st.items()}


def load_records(pattern, rounds='all'):
    recs = []
    for f in sorted(glob.glob(pattern)):
        binf = f[:-6] + '.bin'
        if not os.path.exists(binf): continue
        raw = open(binf, 'rb').read()
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if rounds != 'all' and (rounds == 'boss') != r['boss']: continue
            off, ln = r['bin']
            if off + ln > len(raw): continue
            r['_raw'] = raw[off:off + ln]; r['_file'] = f
            recs.append(r)
    return recs


def unpack(r):
    d = r['dims']; n = r['n']; FP, FT, FE, FG = d['FP'], d['FT'], d['FE'], d['FG']
    b = r['_raw']; o = 0
    def take(cnt, dt, sz):
        nonlocal o
        a = np.frombuffer(b[o:o + cnt * sz], dtype=dt).copy(); o += cnt * sz; return a
    fp = take(n * FP, np.float16, 2).reshape(n, FP); ft = take(TILES * FT, np.float16, 2).reshape(TILES, FT)
    fe = take(n * TILES * 4 * FE, np.float16, 2).reshape(n, TILES, 4, FE); fg = take(FG, np.float16, 2)
    ids = take(n, np.int32, 4); legal = take(n * TILES * 4, np.uint8, 1).reshape(n, TILES, 4); cur = take(n * 2, np.int16, 2).reshape(n, 2)
    return fp, ft, fe, fg, ids, legal, cur


W_LEAK = 8.0
def key(boss, s):
    if s is None: return None
    l, b, k, y = s
    return -l * 1e7 + b + 1e-3 * k if boss else -W_LEAK * l + y + 0.01 * k


def mean_s(ss):
    ss = [x for x in ss if x]
    return [sum(x[i] for x in ss) / len(ss) for i in range(4)] if ss else None


NKEY = 'd'; BOSS_LEAK_W = 10.0; UCLIP = 1e9; MSEED = 0
# X10 --wutil fa.json --winfit winfit.json (default off): the boss utility becomes the win utility of picker (P/b1/winfit.json):
# u = 100 x [P(win | share) - P(win | share of the current layout)], P = sigmoid(a_cfg + b_cfg x share), share = (rehearsal leader
# damage + the other seats' real FA damage) / FA pool max HP (solo: own damage only); round-14 boss decisions with a known FA only.
WFA = None; WFIT = None; WCTX = None
def wctx(r):
    """(maxHp, other seats' real FA damage, a, b) of a boss record, or None when the win utility does not apply."""
    if WFA is None or not r.get('boss') or r.get('round') != 14: return None
    fa = WFA.get(str(r['seed']))
    if not fa or not fa.get('maxHp'): return None
    kind = 'solo' if str(r.get('cfg', '')).startswith('solo') else 'coop4h1'
    by = fa.get('by') or {}; other = 0.0 if kind == 'solo' else sum(v for k, v in by.items() if k != r.get('pid'))
    a, b = WFIT[kind]; return (float(fa['maxHp']), float(other), a, b)
def winp(s, c):
    return 1.0 / (1.0 + math.exp(-(c[2] + c[3] * (s[1] + c[1]) / c[0])))  # X7 --mseed: mean over the seeds a candidate shares with the current layout
def util(boss, s, s0):
    """Candidate utility relative to the current layout (mean scores [leaks, boss, kills, layers]). Normal rounds, NKEY d
    (the picker's rule D): layers + 0.01 kills, an extra leak over the current layout costs 20; soft: -8 x leaks + layers."""
    if s is None or s0 is None: return None
    if boss and WCTX is not None: return 100.0 * (winp(s, WCTX) - winp(s0, WCTX))
    if boss: return max(-UCLIP, min(UCLIP, -BOSS_LEAK_W * (s[0] - s0[0]) + 10 * (s[1] - s0[1]) / max(s0[1], 1e3) + 1e-3 * (s[2] - s0[2])))
    if NKEY == 'd': return (s[3] - s0[3]) + 0.01 * (s[2] - s0[2]) - 20 * max(0.0, s[0] - s0[0] - 1e-9)
    return -W_LEAK * (s[0] - s0[0]) + (s[3] - s0[3]) + 0.01 * (s[2] - s0[2])


def relabel(r, nseeds=3):
    """The pick under the current NKEY among the candidates scored on every validation seed (always incl. the current)."""
    cs = r['cands']; s0 = mean_s(cs[0]['s']); best, bu = 0, 0.0
    for j, c in enumerate(cs):
        if sum(1 for x in c['s'] if x) < nseeds: continue
        u = util(r['boss'], mean_s(c['s']), s0)
        if u is not None and u > bu + 1e-9: best, bu = j, u
    return best


MC = 32; XCF = 26
class Data:
    def __init__(s, recs, dims, seed0=False, srcs=None):
        N = len(recs); FP, FT, FE, FG = dims['FP'], dims['FT'], dims['FE'], dims['FG']
        s.N = N
        s.fp = np.zeros((N, NMAX, FP), np.float16); s.ft = np.zeros((N, TILES, FT), np.float16)
        s.fe = np.zeros((N, NMAX, TILES, 4, FE), np.float16); s.fg = np.zeros((N, FG), np.float16)
        s.ids = np.zeros((N, NMAX), np.int64); s.legal = np.zeros((N, NMAX, TILES, 4), bool); s.pm = np.zeros((N, NMAX), bool)
        s.tgt = np.full((N, NMAX, 2), -1, np.int64); s.cur = np.full((N, NMAX, 2), -1, np.int64)
        s.boss = np.zeros(N, bool); s.copy = np.zeros(N, bool); s.seed = np.zeros(N, np.int64); s.w = np.ones(N, np.float32)
        s.ca = np.full((N, MC, NMAX, 2), -1, np.int16); s.cu = np.zeros((N, MC), np.float32); s.cm = np.zeros((N, MC), bool)
        s.cs = np.zeros((N, MC, 4), np.float32); s.ccur = np.zeros(N, np.int64); s.cf = np.zeros((N, MC, XCF), np.float16)  # seed-0 [leaks, layers/10, kills/10, boss/boss0]
        rng = random.Random(7)
        global WCTX
        for i, r in enumerate(recs):
            WCTX = wctx(r)
            fp, ft, fe, fg, ids, legal, cur = unpack(r); n = min(r['n'], NMAX)
            s.fp[i, :n] = fp[:n]; s.ft[i] = ft; s.fe[i, :n] = fe[:n]; s.fg[i] = fg; s.ids[i, :n] = ids[:n]; s.legal[i, :n] = legal[:n] > 0
            s.pm[i, :n] = True; s.cur[i, :n] = cur[:n]
            a = np.array(r['cands'][r['pick']]['a'][:n], np.int64); s.tgt[i, :n] = a
            s.boss[i] = r['boss']; s.copy[i] = r['pick'] == 0; s.seed[i] = r['seed']
            cs = r['cands']; s0 = (cs[0]['s'][0] if cs[0]['s'] else None) if seed0 else mean_s(cs[0]['s'])
            rest = [j for j in range(len(cs)) if j not in (r['pick'], 0) and (srcs is None or cs[j]['src'] in srcs)]
            vals = [j for j in r.get('val', []) if j in rest]
            others = vals + rng.sample([j for j in rest if j not in vals], min(len(rest) - len(vals), MC))
            sel = ([r['pick']] + ([0] if r['pick'] != 0 else []) + others)[:MC]
            s.ccur[i] = sel.index(0)
            b0 = max(1e3, (cs[0]['s'][0][1] if cs[0]['s'] and cs[0]['s'][0] else 0) or 0)
            for q, j in enumerate(sel):
                u = util(r['boss'], (cs[j]['s'][0] if cs[j]['s'] else None) if seed0 else mean_s(cs[j]['s']), s0)
                if MSEED and seed0 and u is not None:  # X7: both scored on >= 2 common seeds -> the paired mean
                    ks = [k for k in range(min(len(cs[j]['s']), len(cs[0]['s']))) if cs[j]['s'][k] and cs[0]['s'][k]]
                    if len(ks) >= 2: u = util(r['boss'], mean_s([cs[j]['s'][k] for k in ks]), mean_s([cs[0]['s'][k] for k in ks]))
                if u is None: continue
                s.ca[i, q, :n] = np.array(cs[j]['a'][:n], np.int16); s.cu[i, q] = u; s.cm[i, q] = True
                x = cs[j]['s'][0] if cs[j]['s'] and cs[j]['s'][0] else None
                if x: s.cs[i, q] = [x[0], x[3] / 10, x[2] / 10, x[1] / b0]
                if cs[j].get('f') and len(cs[j]['f']) == XCF: s.cf[i, q] = cs[j]['f']
        # a label tile must be legal; unplaced (-1) pieces are dropped from the loss
        bad = 0
        for i in range(N):
            for j in range(NMAX):
                t, d = s.tgt[i, j]
                if t >= 0 and not s.legal[i, j, t, d]: s.tgt[i, j] = -1; bad += 1
        s.bad = bad

    def batch(s, idx, dev):
        T = lambda a, dt=None: torch.from_numpy(a[idx]).to(dev) if dt is None else torch.from_numpy(a[idx].astype(dt)).to(dev)
        return dict(fp=T(s.fp, np.float32), ids=T(s.ids), ft=T(s.ft, np.float32), fe=T(s.fe, np.float32), fg=T(s.fg, np.float32),
                    legal=T(s.legal), pm=T(s.pm), tgt=T(s.tgt)), T(s.cur), T(s.boss), T(s.w)


def rank_loss(net, b, D, bi, dev, M, tau):
    """Listwise loss: softmax over M sampled candidates of the sequence log-prob vs softmax(utility / tau)."""
    B = len(bi); sel = np.zeros((B, M), np.int64); msk = np.zeros((B, M), bool)
    for q, i in enumerate(bi):
        ok = np.nonzero(D.cm[i])[0]
        if len(ok) < 2: continue
        pick = ok[:1].tolist() + (np.random.choice(ok[1:], min(M - 1, len(ok) - 1), replace=False).tolist() if len(ok) > 1 else [])
        sel[q, :len(pick)] = pick; msk[q, :len(pick)] = True
    if msk.sum(1).max() < 2: return None
    rep = lambda x: x.repeat_interleave(M, 0)
    tgt = torch.from_numpy(D.ca[bi][np.arange(B)[:, None], sel].astype(np.int64)).to(dev).reshape(B * M, NMAX, 2)
    u = torch.from_numpy(D.cu[bi][np.arange(B)[:, None], sel]).to(dev); mk = torch.from_numpy(msk).to(dev)
    # a candidate whose label tile is illegal for its piece (should not happen) is dropped
    lg, ok = net(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), tgt)
    Bm, n = lg.shape[:2]
    valid = rep(b['pm']) & (tgt[..., 0] >= 0)
    lab = tgt[..., 0].clamp(min=0) * 4 + tgt[..., 1].clamp(min=0)
    lp = torch.log_softmax(lg.reshape(Bm, n, -1), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    lp = (lp.clamp(min=-30) * valid.float()).sum(1).reshape(B, M)
    lp = lp.masked_fill(~mk, -1e9); tq = torch.softmax((u / tau).masked_fill(~mk, -1e9), -1)
    use = mk.sum(1) >= 2
    return -(tq * torch.log_softmax(lp, -1)).sum(1)[use].mean()


def run_epoch(net, D, idx, dev, opt=None, bs=128, rank_w=0.0, rank_m=6, tau=2.0):
    tot = dict(loss=0.0, steps=0, acc=0.0, exact=0.0, n=0, copyacc=0.0, ncopy=0, curacc=0.0)
    net.train(opt is not None)
    for k in range(0, len(idx), bs):
        bi = idx[k:k + bs]
        b, cur, boss, w = D.batch(bi, dev)
        with torch.set_grad_enabled(opt is not None):
            lg, ok = net(**b)
            B, n = lg.shape[:2]
            tgt = b['tgt']; valid = b['pm'] & (tgt[..., 0] >= 0)
            lab = (tgt[..., 0].clamp(min=0) * 4 + tgt[..., 1].clamp(min=0))
            flat = lg.reshape(B, n, -1)
            ce = F.cross_entropy(flat.reshape(B * n, -1), lab.reshape(-1), reduction='none').reshape(B, n)
            ce = (ce * valid.float()).sum(1) / valid.float().sum(1).clamp(min=1)
            loss = NLL_W * (ce * w).mean()
            if rank_w > 0 and opt is not None:
                rl = rank_loss(net, b, D, bi, dev, rank_m, tau)
                if rl is not None: loss = loss + rank_w * rl
            if opt is not None:
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            pred = flat.argmax(-1); hit = (pred == lab) & valid
            tot['loss'] += float(loss) * len(bi); tot['n'] += len(bi)
            tot['acc'] += float(hit.sum()); tot['steps'] += float(valid.sum())
            tot['exact'] += float((hit | ~valid).all(1).float().sum())
            curlab = cur[..., 0].clamp(min=0) * 4 + cur[..., 1].clamp(min=0)
            tot['curacc'] += float(((curlab == lab) & valid).sum())
    return {k: (v / tot['n'] if k in ('loss', 'exact') else v / max(1, tot['steps']) if k in ('acc', 'curacc') else v) for k, v in tot.items() if k not in ('n', 'steps', 'ncopy', 'copyacc')}


@torch.no_grad()
def decode(net, D, idx, dev, margin=0.0, bs=256):
    """Greedy AR decode; a piece leaves its current (legacy) tile / facing only when the best logit beats the current
    choice's by more than `margin` (the current choice must be legal and free). Returns [len(idx), NMAX, 2]."""
    net.eval(); outs = []
    for k in range(0, len(idx), bs):
        bi = idx[k:k + bs]; b, cur, boss, w = D.batch(bi, dev)
        B = len(bi); tgt = torch.full((B, NMAX, 2), -1, dtype=torch.long, device=dev)
        for j in range(NMAX):
            if not bool(b['pm'][:, j].any()): break
            b['tgt'] = tgt; lg, ok = net(**b)
            fl = lg[:, j].reshape(B, -1); am = fl.argmax(-1); bv = fl.max(-1).values
            cl = cur[:, j, 0].clamp(min=0) * 4 + cur[:, j, 1].clamp(min=0); cv = fl.gather(1, cl[:, None]).squeeze(1)
            keep = (cur[:, j, 0] >= 0) & (cv > -1e8) & (bv - cv <= margin)
            ch = torch.where(keep, cl, am)
            live = b['pm'][:, j] & (bv > -1e8)
            tgt[:, j, 0] = torch.where(live, ch // 4, torch.full_like(ch, -1)); tgt[:, j, 1] = torch.where(live, ch % 4, torch.full_like(ch, -1))
        outs.append(tgt.cpu().numpy())
    return np.concatenate(outs)


def decode_report(net, D, recs, idx, dev, margins=(0.0, 1.0, 2.0, 4.0)):
    """Offline proxy on held-out decisions: how often the decoded layout is the pick / the current one / another evaluated
    candidate, and the mean utility (vs the current layout) of the decodes whose candidate score is known."""
    out = {}
    for mg in margins:
        A = decode(net, D, idx, dev, mg)
        st = dict(n=len(idx), pick=0, cur=0, known=0, unknown=0, u=0.0, uPick=0.0, better=0, worse=0)
        for q, i in enumerate(idx):
            r = recs[i]; n = min(r['n'], NMAX); a = A[q, :n].tolist(); cs = r['cands']; s0 = mean_s(cs[0]['s'])
            sig = {json.dumps(c['a'][:n]): j for j, c in enumerate(cs)}
            j = sig.get(json.dumps(a)); up = util(r['boss'], mean_s(cs[r['pick']]['s']), s0) or 0.0
            st['uPick'] += up
            if j is None: st['unknown'] += 1; continue
            u = util(r['boss'], mean_s(cs[j]['s']), s0)
            if u is None: st['unknown'] += 1; continue
            st['known'] += 1; st['u'] += u; st['pick'] += j == r['pick']; st['cur'] += j == 0; st['better'] += u > 1e-9; st['worse'] += u < -1e-9
        k = max(1, st['known']); out[mg] = dict(n=st['n'], pick=st['pick'] / st['n'], cur=st['cur'] / st['n'], known=st['known'] / st['n'],
                                                 uKnown=st['u'] / k, better=st['better'] / k, worse=st['worse'] / k, uPick=st['uPick'] / st['n'])
    return out


@torch.no_grad()
def seq_logp(net, b, tgts, dev):
    """Sequence log-prob of each assignment in tgts [M, NMAX, 2] for the one decision in batch b (B = 1)."""
    M = len(tgts); rep = lambda x: x.expand(M, *x.shape[1:]).contiguous()
    t = torch.from_numpy(np.asarray(tgts, np.int64)).to(dev)
    lg, ok = net(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), t)
    n = lg.shape[1]; valid = rep(b['pm']) & (t[..., 0] >= 0)
    lab = t[..., 0].clamp(min=0) * 4 + t[..., 1].clamp(min=0)
    lp = torch.log_softmax(lg.reshape(M, n, -1), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    return (lp.clamp(min=-30) * valid.float()).sum(1).cpu().numpy()


def picker_report(net, D, recs, idx, dev, srcs=('cur', 'leg', 'pert')):
    """Offline: the model as a picker (argmax sequence log-prob) over the deploy-available candidates of each held-out
    decision (generated without sims: the current layout, legacy variants, random perturbations); seed-0 utilities."""
    net.eval(); st = dict(n=0, uModel=0.0, uOracle=0.0, cur=0, better=0, worse=0, ncand=0)
    global WCTX
    for i in idx:
        r = recs[i]; cs = r['cands']; n = min(r['n'], NMAX); WCTX = wctx(r)
        js = [j for j, c in enumerate(cs) if c['src'] in srcs and c['s'] and c['s'][0]]
        if len(js) < 2 or not cs[0]['s'][0]: continue
        u = [util(r['boss'], cs[j]['s'][0], cs[0]['s'][0]) for j in js]
        tg = np.full((len(js), NMAX, 2), -1, np.int64)
        for q, j in enumerate(js): tg[q, :n] = np.array(cs[j]['a'][:n])
        b, cur, boss, w = D.batch(np.array([i]), dev)
        lp = seq_logp(net, b, tg, dev); k = int(np.argmax(lp))
        st['n'] += 1; st['uModel'] += u[k]; st['uOracle'] += max(u); st['cur'] += js[k] == 0; st['better'] += u[k] > 1e-9; st['worse'] += u[k] < -1e-9; st['ncand'] += len(js)
    n = max(1, st['n']); return {k: (v if k == 'n' else v / n) for k, v in st.items()}


def export(net, path, meta):
    sd = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    tensors, chunks, off = {}, [], 0
    for k, a in sd.items():
        tensors[k] = {'offset': off, 'shape': list(a.shape)}; chunks.append(a.astype('<f4').tobytes()); off += a.size
    import base64
    blob = b''.join(chunks)
    man = {'format': 'xplace/1', 'cfg': net.cfg, 'tensors': tensors, 'meta': meta, 'data': base64.b64encode(blob).decode(), 'md5': hashlib.md5(blob).hexdigest()}
    json.dump(man, open(path, 'w'))
    return man['md5']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=96); ap.add_argument('--rounds', default='all'); ap.add_argument('--val-mod', type=int, default=10)
    ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--device', default='cuda')
    ap.add_argument('--gain-w', type=float, default=0.0, help='weight a decision by 1 + gain-w x (the pick key - the current key, normal rounds, clipped to [0, 10])')
    ap.add_argument('--max-records', type=int, default=0); ap.add_argument('--report-only', default='')
    ap.add_argument('--boss-leak-w', type=float, default=10.0); ap.add_argument('--uclip', type=float, default=1e9)
    ap.add_argument('--salted-only', action='store_true'); ap.add_argument('--salt-w', type=float, default=1.0)
    ap.add_argument('--save-last', action='store_true'); ap.add_argument('--rank-src', default=''); ap.add_argument('--init', default=''); ap.add_argument('--nll-w', type=float, default=1.0)
    ap.add_argument('--cf', type=int, default=0); ap.add_argument('--aux-w', type=float, default=0.0); ap.add_argument('--eu-w', type=float, default=0.0); ap.add_argument('--frac', type=float, default=1.0); ap.add_argument('--arch', default='ar'); ap.add_argument('--qattn', type=int, default=1); ap.add_argument('--qm', type=int, default=16)
    ap.add_argument('--nkey', default='d'); ap.add_argument('--boss-w', type=float, default=1.0); ap.add_argument('--attn', type=int, default=0)
    ap.add_argument('--rank-w', type=float, default=0.0); ap.add_argument('--rank-m', type=int, default=6); ap.add_argument('--tau', type=float, default=2.0)
    # X7 (default off): --data entries 'pattern@w' weight a source; --val-fold k = the k-th split of hash(seed) % val-mod;
    # --val-src s: only records whose file path contains s can be held out (others always train); --mseed 1: paired multi-seed utilities
    ap.add_argument('--wutil', default=''); ap.add_argument('--winfit', default='')  # X10: the win utility (see WFA)
    ap.add_argument('--val-fold', type=int, default=0); ap.add_argument('--val-src', default=''); ap.add_argument('--mseed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.device == 'cuda' and os.environ.get('SPRL_CUDA_MEM_FRAC'): torch.cuda.set_per_process_memory_fraction(float(os.environ['SPRL_CUDA_MEM_FRAC']))
    t0 = time.time()
    recs = []
    for p in a.data.split(','):
        if '@' in p:  # X7: per-source weight
            p, sw = p.rsplit('@', 1); rs = load_records(p, a.rounds)
            for r in rs: r['_w'] = float(sw)
            recs += rs
        else: recs += load_records(p, a.rounds)
    if a.max_records: recs = recs[:a.max_records]
    dims = recs[0]['dims']; vh = recs[0]['vh']
    recs = [r for r in recs if r['dims'] == dims and r['vh'] == vh]
    if a.salted_only: recs = [r for r in recs if r.get('salt')]
    if a.wutil:  # X10: boss records only where the win utility applies (round 14, FA known); normal-round records unchanged
        global WFA, WFIT; WFA = json.load(open(a.wutil)); WFIT = json.load(open(a.winfit)); n0 = len(recs)
        recs = [r for r in recs if not r['boss'] or wctx(r) is not None]
        print(f'win utility: kept {len(recs)} of {n0} records (boss: round 14 with a known FA)', flush=True)
        global WCTX; npk = 0
        for r in recs:  # the label (pick) of a boss record: the best candidate by the win utility over its scored seeds
            if not r['boss']: continue
            WCTX = wctx(r); cs = r['cands']; k0 = [x for x in cs[0]['s'] if x]
            if not k0: continue
            best, bu = 0, 0.0
            for j, c in enumerate(cs):
                if len([x for x in c['s'] if x]) < len(k0): continue
                u = util(True, mean_s(c['s'][:len(k0)]), mean_s(cs[0]['s'][:len(k0)]))
                if u is not None and u > bu + 1e-9: best, bu = j, u
            npk += best != r['pick']; r['pick'] = best
        WCTX = None; print(f'win utility: {npk} boss picks relabelled', flush=True)
    global NKEY, AUX_W, EU_W, NLL_W, BOSS_LEAK_W, UCLIP, MSEED; MSEED = a.mseed; BOSS_LEAK_W = a.boss_leak_w; UCLIP = a.uclip; NKEY = a.nkey; AUX_W = a.aux_w; EU_W = a.eu_w; NLL_W = a.nll_w
    nre = 0
    for r in recs:
        if r.get('nkey', 'soft') != a.nkey and not r['boss']:
            p = relabel(r); nre += p != r['pick']; r['pick'] = p
    print(f'relabelled {nre} soft-key records to nkey {a.nkey}', flush=True)
    D = Data(recs, dims, seed0=a.arch == 'q', srcs=set(a.rank_src.split(',')) if a.rank_src else None)
    if a.gain_w > 0:
        for i, r in enumerate(recs):
            if r['boss']: continue
            k0 = key(False, r['cands'][0]['s'][0]); kp = key(False, r['cands'][r['pick']]['s'][0])
            if k0 is not None and kp is not None: D.w[i] = 1 + a.gain_w * min(10, max(0, kp - k0)) / 10
    D.w[D.boss] *= a.boss_w
    if a.salt_w != 1: D.w[np.array([bool(r.get('salt')) for r in recs])] *= a.salt_w
    if any('_w' in r for r in recs): D.w *= np.array([r.get('_w', 1.0) for r in recs], np.float32)
    V = int(max(D.ids.max() + 1, 2048))
    isva = (D.seed * 2654435761) % (2 ** 32) % a.val_mod == a.val_fold
    if a.val_src: isva &= np.array([a.val_src in r['_file'] for r in recs])
    va = np.nonzero(isva)[0]; tr = np.nonzero(~isva)[0]
    if a.frac < 1: tr = tr[((D.seed[tr] * 2654435761) // 7919) % 1000 < a.frac * 1000]
    print(f'records {D.N} (train {len(tr)} val {len(va)}) bad-labels {D.bad} copy-share {D.copy.mean():.3f} boss {D.boss.mean():.3f} load {time.time() - t0:.0f}s', flush=True)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == 'cpu' else 'cpu')
    net = XNet(dims['FP'], dims['FT'], dims['FE'], dims['FG'], V, d=a.d, attn=a.attn).to(dev)
    if a.init and a.arch != 'q':
        ck0 = torch.load(a.init, map_location=dev); c0 = ck0['cfg']
        net = XNet(c0['FP'], c0['FT'], c0['FE'], c0['FG'], c0['V'], d=c0['d'], de=c0['de'], attn=c0.get('attn', 0), heads=c0.get('heads', 4)).to(dev); net.load_state_dict(ck0['state'])
        print('init from', a.init, flush=True)
    if a.report_only and a.arch == 'q':
        ck = torch.load(a.report_only, map_location=dev); c = ck['cfg']
        net = QNet(c['FP'], c['FT'], c['FE'], c['FG'], c['V'], d=c['d'], de=c['de'], attn=c.get('attn', 0), heads=c.get('heads', 4), qattn=c.get('qattn', 1), cf=c.get('cf', 0)).to(dev); net.load_state_dict(ck['state'])
        if a.aux_w > 0 or 'aux.0.weight' in ck['state']:
            for th in (0.0, 0.1, 0.3):
                for mg in (0.0, 1.0, 2.0):
                    for split, ii in (('normal', va[~D.boss[va]]), ('boss', va[D.boss[va]])):
                        if len(ii): print(f'qrule {split} leakth {th} margin {mg}:', {k: round(v, 3) for k, v in q_picker_report(net, D, recs, ii[:1500], dev, ('cur', 'leg'), mg, (th,)).items() if k in ('n', 'uModel', 'uOracle', 'cur', 'better', 'worse')}, flush=True)
        for srcs in (('cur', 'leg'), ('cur', 'leg', 'pert')):
            for mg in (0.0, 0.5, 1.0, 2.0):
                for split, ii in (('normal', va[~D.boss[va]]), ('boss', va[D.boss[va]])):
                    if len(ii): print(f'qpicker {split} {srcs} margin {mg}:', {k: round(v, 3) for k, v in q_picker_report(net, D, recs, ii[:1500], dev, srcs, mg).items() if k in ('n', 'uModel', 'uOracle', 'cur', 'better', 'worse')}, flush=True)
        return
    if a.arch == 'q':
        net = QNet(dims['FP'], dims['FT'], dims['FE'], dims['FG'], V, d=a.d, attn=a.attn, qattn=a.qattn, cf=XCF if a.cf else 0).to(dev)
        if a.init:
            ck0 = torch.load(a.init, map_location=dev); c0 = ck0['cfg']
            net = QNet(c0['FP'], c0['FT'], c0['FE'], c0['FG'], c0['V'], d=c0['d'], de=c0['de'], attn=c0.get('attn', 0), heads=c0.get('heads', 4), qattn=c0.get('qattn', 1), cf=c0.get('cf', 0)).to(dev)
            net.load_state_dict(ck0['state']); print('init from', a.init, flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        nb = math.ceil(len(tr) / 48)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb + 1, pct_start=0.1)
        class O:
            def zero_grad(s): opt.zero_grad()
            def step(s): opt.step(); sched.step()
        best = None; hist = []
        for ep in range(a.epochs):
            trm = q_epoch(net, D, tr[np.random.permutation(len(tr))], dev, O(), 48, a.qm, a.tau)
            vam = q_epoch(net, D, va, dev, None, 96, a.qm, a.tau)
            hist.append(dict(ep=ep, tr=trm, va=vam))
            print(f"ep {ep} tr loss {trm['loss']:.4f} top1 {trm['top1']:.3f} | va loss {vam['loss']:.4f} top1 {vam['top1']:.3f} | {time.time() - t0:.0f}s", flush=True)
            if best is None or vam['loss'] < best:
                best = vam['loss']; torch.save({'state': net.state_dict(), 'cfg': net.cfg, 'dims': dims, 'vh': vh}, a.out + '.pt')
        ck = torch.load(a.out + '.pt', map_location=dev); net.load_state_dict(ck['state'])
        for split, ii in (('normal', va[~D.boss[va]]), ('boss', va[D.boss[va]])):
            if len(ii):
                pr = q_picker_report(net, D, recs, ii[:1500], dev); hist.append({'qpicker_' + split: pr})
                print(f'qpicker {split}: ' + ' '.join(f'{k} {x:.3f}' if isinstance(x, float) else f'{k} {x}' for k, x in pr.items()), flush=True)
        md5 = export(net, a.out + '.json', {'dims': dims, 'vh': vh, 'records': D.N, 'args': vars(a), 'hist': hist, 'bestVal': best})
        print('exported', a.out + '.json', md5, flush=True)
        return
    if a.report_only:  # offline reports of a trained checkpoint on this data's held-out split
        ck = torch.load(a.report_only, map_location=dev); c = ck['cfg']
        net = XNet(c['FP'], c['FT'], c['FE'], c['FG'], c['V'], d=c['d'], de=c['de'], attn=c.get('attn', 0), heads=c.get('heads', 4)).to(dev); net.load_state_dict(ck['state'])
        for split, ii in (('normal', va[~D.boss[va]]), ('boss', va[D.boss[va]])):
            if len(ii): print(f'picker {split}:', picker_report(net, D, recs, ii[:1500], dev), flush=True)
        return
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * math.ceil(len(tr) / a.bs) + 1, pct_start=0.1)
    best = None; hist = []
    for ep in range(a.epochs):
        perm = tr[np.random.permutation(len(tr))]
        class O:  # step the schedule with the optimizer
            def zero_grad(s): opt.zero_grad()
            def step(s): opt.step(); sched.step()
        trm = run_epoch(net, D, perm, dev, O(), a.bs, a.rank_w, a.rank_m, a.tau)
        vam = run_epoch(net, D, va, dev, None, 512)
        vb = run_epoch(net, D, va[D.boss[va]], dev, None, 512) if D.boss[va].any() else {}
        hist.append(dict(ep=ep, tr=trm, va=vam, vaBoss=vb))
        print(f"ep {ep} tr loss {trm['loss']:.3f} acc {trm['acc']:.3f} | va loss {vam['loss']:.3f} acc {vam['acc']:.3f} exact {vam['exact']:.3f} (copy-legacy acc {vam['curacc']:.3f})"
              + (f" | boss va acc {vb['acc']:.3f}" if vb else '') + f" | {time.time() - t0:.0f}s", flush=True)
        if a.save_last or best is None or vam['loss'] < best:
            best = vam['loss']; torch.save({'state': net.state_dict(), 'cfg': net.cfg, 'dims': dims, 'vh': vh}, a.out + '.pt')
    ck = torch.load(a.out + '.pt', map_location=dev); net.load_state_dict(ck['state'])
    for split, ii in (('normal', va[~D.boss[va]]), ('boss', va[D.boss[va]])):
        if len(ii) == 0: continue
        pr = picker_report(net, D, recs, ii[:1500], dev)
        print(f'picker {split}: ' + ' '.join(f'{k} {x:.3f}' if isinstance(x, float) else f'{k} {x}' for k, x in pr.items()), flush=True)
        hist.append({'picker_' + split: pr})
        rep_ = decode_report(net, D, recs, ii[:2000], dev)
        for mg, v in rep_.items(): print(f'decode {split} margin {mg}: ' + ' '.join(f'{k} {x:.3f}' if isinstance(x, float) else f'{k} {x}' for k, x in v.items()), flush=True)
        hist.append({'decode_' + split: {str(k): v for k, v in rep_.items()}})
    md5 = export(net, a.out + '.json', {'dims': dims, 'vh': vh, 'records': D.N, 'args': vars(a), 'hist': hist[-1], 'bestVal': best})
    print('exported', a.out + '.json', md5, flush=True)


if __name__ == '__main__':
    main()
