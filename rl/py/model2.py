"""Architecture probe (ppo3): SPNet generalised. width scales every hidden size; trunk 'mlp' (SPNet) or 'resln' (residual
blocks with LayerNorm, SimBa / BRO style); attn > 0 adds pre-LN self-attention layers over all entity rows plus a global
token before pooling, and the pointer heads read the attended rows. Linear, ReLU, LayerNorm and softmax attention only:
all mirrorable in plain JS (infer.mjs would need LayerNorm and attention added)."""
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EMB, ENTITY_OUT, POINTER, SCALARS


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o), nn.ReLU())


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__(); self.ln = nn.LayerNorm(d); self.f1 = nn.Linear(d, 4 * d // 2); self.f2 = nn.Linear(4 * d // 2, d)
    def forward(self, x):
        return x + self.f2(F.relu(self.f1(self.ln(x))))


def _entropy(q, k, keep):
    """Mean attention entropy (nats) over the present query rows (diagnostics: a collapse to ~0 precedes a logit runaway)."""
    q, k, keep = q[:1024], k[:1024], keep[:1024]  # a sub-batch: the full attention map of a big eval batch is GBs
    att = ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).float().masked_fill(~keep[:, None, None, :], -1e9).softmax(-1)
    ent = -(att * att.clamp_min(1e-12).log()).sum(-1)          # (B, h, Tq)
    return float(ent.mean())


class AttnLayer(nn.Module):
    def __init__(self, d, heads, qkln=False):
        super().__init__(); self.h = heads; self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.qkln = qkln; self.track = False; self.entropy = None
        if qkln: self.qn = nn.LayerNorm(d // heads); self.kn = nn.LayerNorm(d // heads)  # QK-LayerNorm (ViT-22B): bounds the attention logits
        self.ln2 = nn.LayerNorm(d); self.f1 = nn.Linear(d, 2 * d); self.f2 = nn.Linear(2 * d, d)
        self.fused = True
    def forward(self, x, keep):  # x (B, T, d), keep (B, T) bool
        B, T, d = x.shape; h = self.h
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
        if self.qkln: q = self.qn(q); k = self.kn(k)
        if self.track: self.entropy = _entropy(q, k, keep)
        if self.fused:  # the same softmax attention through the fused kernel (every row keeps its global token)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        else:
            att = (q @ k.transpose(-1, -2)) / (d // h) ** 0.5
            y = att.masked_fill(~keep[:, None, None, :], -1e9).softmax(-1) @ v
        x = x + self.o(y.transpose(1, 2).reshape(B, T, d))
        return x + self.f2(F.relu(self.f1(self.ln2(x))))


class XAttn(nn.Module):
    """Cross-attention, one head, pre-LN, zero-initialised output (starts as the identity): queries `q` (B, Tq, dq) read
    keys / values `k` (B, Tk, dk) of present rows `keep` (B, Tk). For version 4: tiles <- owned pieces, owned <- tiles."""
    def __init__(self, dq, dk, d, qkln=False):
        super().__init__(); self.lq = nn.LayerNorm(dq); self.lk = nn.LayerNorm(dk)
        self.q = nn.Linear(dq, d); self.k = nn.Linear(dk, d); self.v = nn.Linear(dk, d); self.o = nn.Linear(d, dq)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias); self.d = d
        self.qkln = qkln; self.track = False; self.entropy = None
        if qkln: self.qn = nn.LayerNorm(d); self.kn = nn.LayerNorm(d)  # QK-LayerNorm
    def forward(self, q, k, keep):
        kk = self.lk(k)
        any_ = keep.any(1, keepdim=True)  # a row with no present key attends to nothing (adds 0)
        qq, kq = self.q(self.lq(q)), self.k(kk)
        if self.qkln: qq = self.qn(qq); kq = self.kn(kq)
        if self.track: self.entropy = _entropy(qq.unsqueeze(1), kq.unsqueeze(1), keep | ~any_)
        y = F.scaled_dot_product_attention(qq.unsqueeze(1), kq.unsqueeze(1), self.v(kk).unsqueeze(1),
                                           attn_mask=(keep | ~any_)[:, None, None, :]).squeeze(1)
        return q + self.o(y) * any_.unsqueeze(-1).to(q.dtype)


import os as _os, torch.utils.checkpoint  # noqa: E402
EA_CKPT = _os.environ.get('SPRL_EA_CKPT') == '1'
EA_SKIP = ('global', 'extra', 'bands')  # entattn: rows kept out (the single global / extra rows; the band rows: the frozen band head)
V4_OUT = {'globalX': 32, 'tiles': 64}  # version 4 entity blocks (ownedX joins the owned rows)
LA_FIELD = {'shop': 3, 'owned': 4, 'rewards': 2}  # the piece-code field of each candidate block (spec.mjs ids)
LA_PER_BOND, LA_BONDS = 8, 3


class SPNet2(nn.Module):
    def __init__(self, spec, width=1.0, trunk='mlp', depth=2, attn=0, attn_dim=128, heads=4, ctx=256, lookahead=None, emb_mult=1, xattn=0, attn_skip=(), qkln=False,
                 entattn=0, critic=0):
        super().__init__()
        self.spec = spec; self.cfg = dict(width=width, trunk=trunk, depth=depth, attn=attn, attn_dim=attn_dim, heads=heads, ctx=ctx, lookahead=lookahead,
                                          emb_mult=emb_mult, xattn=xattn, attn_skip=tuple(attn_skip), qkln=qkln)
        if entattn: self.cfg['entattn'] = entattn  # (11-13) absent from older configs: they build as before
        if critic: self.cfg['critic'] = critic  # (11-13, LIN-C) a separate critic trunk: the value head reads its own copy
        EMBX = {k: d * emb_mult for k, d in EMB.items()}  # emb_mult: parameters without deployed compute (folded into tables)
        self.embx = EMBX
        self.la = None
        if lookahead:  # bond look-ahead per candidate slot (static tables from lookahead_table.mjs)
            import json as _json
            t = _json.load(open(lookahead)); nb = len(t['bonds']); maxT = max(1, max(len(b['thresholds']) for b in t['bonds']))
            pb = torch.zeros(t['pieceVocab'] + 1, LA_BONDS, dtype=torch.long)
            for code, lst in t['pieceBonds'].items():
                for j, k in enumerate(lst[:LA_BONDS]): pb[int(code), j] = k + 1
            th = torch.full((nb + 1, maxT), 1e9)
            for k, b in enumerate(t['bonds']):
                for j, x in enumerate(b['thresholds']): th[k + 1, j] = float(x)
            self.register_buffer('la_pb', pb, persistent=False); self.register_buffer('la_th', th, persistent=False)
            self.register_buffer('la_core', torch.tensor([0.0] + [1.0 if b['core'] else 0.0 for b in t['bonds']]), persistent=False)
            self.la = nb
        blocks, vocab = spec['blocks'], spec['vocab']
        self.D, self.A = spec['dim'], spec['actions']
        W = lambda n: max(8, int(round(n * width)))
        self.emb = nn.ModuleDict({k: nn.Embedding(vocab[k], d) for k, d in EMBX.items() if k in vocab})
        self.v4 = 'tiles' in blocks
        EO = {**ENTITY_OUT, **V4_OUT}
        self.names = [n for n in EO if n in blocks]
        ids = defaultdict(list)
        for block, field, voc in spec['ids']:
            ids[block].append((field, voc))
        self.layout, self.enc, self.out = {}, nn.ModuleDict(), {}
        if self.v4:  # ownedX: more fields of the owned rows (same row index), concatenated before the owned encoder
            b = blocks['ownedX']; idf = {f for f, _ in ids['ownedX']}
            self.register_buffer('num_ownedX', torch.tensor([i for i in range(b['width']) if i not in idf], dtype=torch.long), persistent=False)
            self.layoutX = (b['offset'], b['count'], b['width'], ids['ownedX'])
            ownedX_in = len(self.num_ownedX) + sum(EMBX[v] for _, v in ids['ownedX'])
        for name in self.names:
            b = blocks[name]; idf = {f for f, _ in ids[name]}
            numeric = [i for i in range(b['width']) if i not in idf]
            self.register_buffer(f'num_{name}', torch.tensor(numeric, dtype=torch.long), persistent=False)
            self.layout[name] = (b['offset'], b['count'], b['width'], ids[name])
            o = W(EO[name]); self.out[name] = o
            extra = LA_BONDS * LA_PER_BOND if self.la and name in LA_FIELD else 0
            if self.v4 and name == 'owned': extra += ownedX_in
            self.enc[name] = mlp(len(numeric) + sum(EMBX[v] for _, v in ids[name]) + extra, max(o, 64), o)
        nb = blocks['bonds']; self.bonds_at = (nb['offset'], nb['count'] * nb['width'])
        self.bonds = mlp(self.bonds_at[1], W(128), W(128))
        ents = [n for n in self.names if n != 'global']
        C = W(ctx)
        self.att_ents = [n for n in ents if n not in set(attn_skip)]  # attn_skip: blocks kept out of the self-attention
        if attn:
            da = attn_dim
            self.proj = nn.ModuleDict({n: nn.Linear(self.out[n], da) for n in self.att_ents})
            self.gproj = nn.Linear(self.out['global'] + W(128), da)
            self.typ = nn.Parameter(torch.zeros(len(self.att_ents) + 1, da))
            self.layers = nn.ModuleList([AttnLayer(da, heads, qkln) for _ in range(attn)])
            self.lnf = nn.LayerNorm(da)
            trunk_in = self.out['global'] + W(128) + da + sum(2 * (da if n in self.att_ents else self.out[n]) for n in ents)
            slot_dim = {n: (da if n in self.att_ents else self.out[n]) for n in ents}
        else:
            trunk_in = self.out['global'] + W(128) + sum(2 * self.out[n] for n in ents)
            slot_dim = dict(self.out)
        H = W(512)
        if trunk == 'resln':
            self.trunk = nn.Sequential(nn.Linear(trunk_in, H), *[ResBlock(H) for _ in range(depth)], nn.LayerNorm(H), nn.Linear(H, C), nn.ReLU())
        else:
            self.trunk = nn.Sequential(nn.Linear(trunk_in, H), nn.ReLU(), nn.Linear(H, C), nn.ReLU())
        if critic:  # same layers as the trunk; widen / critic init copies the trunk's weights (function-identical start)
            self.vtrunk = (nn.Sequential(nn.Linear(trunk_in, H), *[ResBlock(H) for _ in range(depth)], nn.LayerNorm(H), nn.Linear(H, C), nn.ReLU())
                           if trunk == 'resln' else nn.Sequential(nn.Linear(trunk_in, H), nn.ReLU(), nn.Linear(H, C), nn.ReLU()))
        self.scalar = nn.Linear(C, SCALARS)
        self.pointer = [n for n in POINTER if n in blocks]
        self.ptr = nn.ModuleDict({n: nn.Sequential(nn.Linear(slot_dim[n] + C, W(128)), nn.ReLU(), nn.Linear(W(128), 1)) for n in self.pointer})
        self.value = nn.Sequential(nn.Linear(C, W(128)), nn.ReLU(), nn.Linear(W(128), 1))
        self.ents = ents
        if entattn:
            # (11-13, arm AT) residual entity self-attention: every entity row (projected to attn_dim, plus a type embedding)
            # and two read-only token sets — a global token (global + bonds encodings) and one token per bond row (raw bond
            # features) — go through `entattn` pre-LN AttnLayers; each row gets ea_out(x_L - x_0) added, ea_out zero-initialised,
            # so the network starts as the plain SPNet2 (function-identical); absent rows are never changed
            assert not attn, 'entattn and attn are alternatives'
            da = attn_dim; self.ea_ents = [n for n in ents if n not in EA_SKIP]
            self.ea_in = nn.ModuleDict({n: nn.Linear(self.out[n], da) for n in self.ea_ents})
            self.ea_out = nn.ModuleDict({n: nn.Linear(da, self.out[n]) for n in self.ea_ents})
            for m_ in self.ea_out.values(): nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)
            nb_ = blocks['bonds']; self.ea_bonds_rows = (nb_['offset'], nb_['count'], nb_['width'])
            self.ea_bond = nn.Linear(nb_['width'], da)
            self.ea_g = nn.Linear(self.out['global'] + W(128), da)
            self.ea_typ = nn.Parameter(torch.zeros(len(self.ea_ents) + 2, da))
            self.ea_layers = nn.ModuleList([AttnLayer(da, heads, qkln) for _ in range(entattn)])
        if self.v4:
            # version 4 heads, in the action order after the 100 of version 3: MOVE / EQUIP (owned rows), TILE (tile x 4
            # facings, given the selected owned row), BENCH + CANCEL, TARGET (owned rows, given the selected item)
            Ho, Ht = slot_dim['owned'], slot_dim['tiles']
            self.sel_at = blocks['globalX']['offset'] + 7
            self.ptr4 = nn.ModuleDict({k: nn.Sequential(nn.Linear(Ho + C, W(128)), nn.ReLU(), nn.Linear(W(128), 1)) for k in ('move', 'equip')})
            self.target = nn.Sequential(nn.Linear(Ho + C + Ho, W(128)), nn.ReLU(), nn.Linear(W(128), 1))
            self.tile = nn.Sequential(nn.Linear(Ht + C + Ho, W(128)), nn.ReLU(), nn.Linear(W(128), 4))
            self.scalar2 = nn.Linear(C, 2)
            if xattn:  # piece <-> tile interaction before pooling: tiles read the owned pieces, then the pieces read the tiles
                self.xa = XAttn(self.out['tiles'], self.out['owned'], xattn, qkln); self.xb = XAttn(self.out['owned'], self.out['tiles'], xattn, qkln)  # before any self-attention: encoder widths

    def _bond_state(self, obs):
        # per bond (index 1..NB, 0 = none): count, next threshold above it, the threshold reached, layers, main-core flag
        b = self.spec['blocks']['bonds']; B = obs.shape[0]
        x = obs[:, b['offset']:b['offset'] + b['count'] * b['width']].reshape(B, b['count'], b['width'])
        cnt = torch.cat([torch.zeros(B, 1, device=obs.device), (x[..., 0] * 6).round()], 1)
        lay = torch.cat([torch.zeros(B, 1, device=obs.device), x[..., 3]], 1)
        th = self.la_th.unsqueeze(0)
        nxt = torch.where(th > cnt.unsqueeze(-1), th, torch.full_like(th, 1e9)).min(-1).values
        cur = torch.where(th <= cnt.unsqueeze(-1), th, torch.zeros_like(th)).max(-1).values
        core_cnt = cnt * self.la_core
        main = torch.zeros_like(cnt); top = core_cnt.argmax(1); ok = core_cnt.gather(1, top[:, None]).squeeze(1) >= 2
        main[torch.arange(B, device=obs.device)[ok], top[ok]] = 1.0
        return cnt, nxt, cur, lay, main

    def _lookahead(self, x, name, st):
        cnt, nxt, cur, lay, main = st
        code = x[..., LA_FIELD[name]].round().long().clamp_(0, self.la_pb.shape[0] - 1)
        bi = self.la_pb[code]                                   # (B, n, 3) bond indices (+1)
        g = lambda t: t.gather(1, bi.reshape(bi.shape[0], -1)).reshape(bi.shape)
        c, nx, cu, ly, mn = g(cnt), g(nxt), g(cur), g(lay), g(main)
        has = (bi > 0).float(); core = self.la_core[bi]
        f = torch.stack([c / 6, ((c + 1) >= nx).float(), ((nx - c).clamp(0, 6)) / 6, ((c >= cu) & (c - 1 < cu) & (cu > 0)).float(),
                         ly, mn, core, has], -1) * has.unsqueeze(-1)
        return f.reshape(x.shape[0], x.shape[1], LA_BONDS * LA_PER_BOND)

    def _block(self, obs, name, st=None):
        off, n, w, ids = self.layout[name]
        x = obs[:, off:off + n * w].reshape(obs.shape[0], n, w)
        parts = [x.index_select(2, getattr(self, f'num_{name}'))]
        for field, voc in ids:
            code = x[..., field].round().long().clamp_(0, self.emb[voc].num_embeddings - 1)
            parts.append(self.emb[voc](code))
        if self.la and name in LA_FIELD:
            parts.append(self._lookahead(x, name, st))
        if self.v4 and name == 'owned':
            offX, nX, wX, idsX = self.layoutX
            xx = obs[:, offX:offX + nX * wX].reshape(obs.shape[0], nX, wX)
            parts.append(xx.index_select(2, self.num_ownedX))
            for field, voc in idsX:
                parts.append(self.emb[voc](xx[..., field].round().long().clamp_(0, self.emb[voc].num_embeddings - 1)))
        return self.enc[name](torch.cat(parts, dim=-1)), x[..., 0] > 0.5

    @staticmethod
    def _pool(h, present):
        m = present.unsqueeze(-1).to(h.dtype); cnt = m.sum(1).clamp_min(1.0)
        mx = (h * m - (1 - m) * 1e4).max(1).values
        mx = torch.where(present.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([(h * m).sum(1) / cnt, mx], dim=-1)

    def forward(self, obs, mask=None):
        g, _ = self._block(obs, 'global'); g = g[:, 0]
        bo = self.bonds(obs[:, self.bonds_at[0]:self.bonds_at[0] + self.bonds_at[1]])
        hs, pres = {}, {}
        st = self._bond_state(obs) if self.la else None
        for name in self.ents:
            hs[name], pres[name] = self._block(obs, name, st)
        feats = [g, bo]
        if self.v4 and self.cfg.get('xattn'):
            hs['tiles'] = self.xa(hs['tiles'], hs['owned'], pres['owned'])
            hs['owned'] = self.xb(hs['owned'], hs['tiles'], pres['tiles'])
        if self.cfg.get('entattn'):
            B_ = obs.shape[0]; off_, n_, w_ = self.ea_bonds_rows
            xb = obs[:, off_:off_ + n_ * w_].reshape(B_, n_, w_)
            toks = [self.ea_g(torch.cat([g, bo], -1))[:, None] + self.ea_typ[0]]; keep = [torch.ones(B_, 1, dtype=torch.bool, device=obs.device)]
            spans, t0 = {}, 1
            for j, name in enumerate(self.ea_ents):
                toks.append(self.ea_in[name](hs[name]) + self.ea_typ[j + 1]); keep.append(pres[name])
                spans[name] = (t0, t0 + hs[name].shape[1]); t0 += hs[name].shape[1]
            toks.append(self.ea_bond(xb) + self.ea_typ[-1]); keep.append(xb[..., 0] > 0)
            x0 = torch.cat(toks, 1); kp = torch.cat(keep, 1); x = x0
            for layer in self.ea_layers:  # SPRL_EA_CKPT=1: activation checkpointing in training (same values, less memory)
                x = torch.utils.checkpoint.checkpoint(layer, x, kp, use_reentrant=False) if self.training and EA_CKPT else layer(x, kp)
            dx = x - x0
            for name in self.ea_ents:
                a_, b_ = spans[name]
                hs[name] = hs[name] + self.ea_out[name](dx[:, a_:b_]) * pres[name].unsqueeze(-1).to(dx.dtype)
        if self.cfg['attn']:
            toks, keep, spans, t0 = [self.gproj(torch.cat([g, bo], -1))[:, None] + self.typ[0]], [torch.ones_like(pres[self.ents[0]][:, :1])], {}, 1
            for j, name in enumerate(self.att_ents):
                toks.append(self.proj[name](hs[name]) + self.typ[j + 1]); keep.append(pres[name])
                spans[name] = (t0, t0 + hs[name].shape[1]); t0 += hs[name].shape[1]
            x = torch.cat(toks, 1); kp = torch.cat(keep, 1)
            for layer in self.layers:
                x = layer(x, kp)
            x = self.lnf(x)
            feats.append(x[:, 0])
            for name in self.ents:
                if name in spans: a_, b_ = spans[name]; hs[name] = x[:, a_:b_]
                feats.append(self._pool(hs[name], pres[name]))
        else:
            for name in self.ents:
                feats.append(self._pool(hs[name], pres[name]))
        fc = torch.cat(feats, -1); c = self.trunk(fc)
        cv = self.vtrunk(fc) if self.cfg.get('critic') else c  # LIN-C: the value from the critic trunk; the policy never reads it
        logits = [self.scalar(c)]
        for name in self.pointer:
            h = hs[name]; cc = c.unsqueeze(1).expand(-1, h.shape[1], -1)
            logits.append(self.ptr[name](torch.cat([h, cc], -1)).squeeze(-1))
        if self.v4:
            ho, ht = hs['owned'], hs['tiles']; B = ho.shape[0]
            co = c.unsqueeze(1).expand(-1, ho.shape[1], -1)
            sel = obs[:, self.sel_at].round().long() - 1
            has = (sel >= 0).to(ho.dtype).unsqueeze(-1)
            sh = ho[torch.arange(B, device=ho.device), sel.clamp(0, ho.shape[1] - 1)] * has  # the selected owned row (0 = none)
            logits.append(self.ptr4['move'](torch.cat([ho, co], -1)).squeeze(-1))
            logits.append(self.ptr4['equip'](torch.cat([ho, co], -1)).squeeze(-1))
            ct = c.unsqueeze(1).expand(-1, ht.shape[1], -1)
            logits.append(self.tile(torch.cat([ht, ct, sh.unsqueeze(1).expand(-1, ht.shape[1], -1)], -1)).reshape(B, -1))
            logits.append(self.scalar2(c))
            logits.append(self.target(torch.cat([ho, co, sh.unsqueeze(1).expand(-1, ho.shape[1], -1)], -1)).squeeze(-1))
        logits = torch.cat(logits, -1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits, self.value(cv).squeeze(-1)


def flops_per_forward(net, obs, mask):
    """Multiply-adds x 2 of Linear layers and attention matmuls for one observation (batch 1)."""
    total = [0]
    hooks = []
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m_, i, o: total.__setitem__(0, total[0] + 2 * i[0].numel() // i[0].shape[-1] * m_.in_features * m_.out_features)))
    with torch.no_grad():
        net(obs[:1], mask[:1])
    for h in hooks:
        h.remove()
    if net.cfg['attn']:  # q·k and att·v per layer
        T = 1 + sum(net.layout[n][1] for n in getattr(net, 'att_ents', net.ents)); da = net.cfg['attn_dim']
        total[0] += net.cfg['attn'] * 2 * 2 * T * T * da
    return total[0]


def deployed_macs(net, obs, mask):
    """Mean multiply-adds per decision of the deployed forward (infer2.mjs rules: embeddings folded into first-layer
    tables, absent rows skipped, pointer heads split, legal actions only, no value head) for an mlp-trunk SPNet2;
    a residual trunk adds its blocks; attention adds its token layers. obs (N, D) / mask (N, A) tensors (CPU)."""
    lin = lambda m_: m_.in_features * m_.out_features
    N = obs.shape[0]; tot = 0.0
    trunk = sum(lin(m_) for m_ in net.trunk.modules() if isinstance(m_, nn.Linear))
    bonds = sum(lin(m_) for m_ in net.bonds.modules() if isinstance(m_, nn.Linear))
    C = net.scalar.in_features
    for k in range(N):
        o, mk = obs[k], mask[k]; t = trunk + bonds
        t += int(mk[:4].sum()) * C
        for name in net.names:
            off, n, w, ids = net.layout[name]; enc0, enc2 = net.enc[name][0], net.enc[name][2]
            h0 = enc0.out_features; numeric = len(getattr(net, f'num_{name}'))
            extra = (enc0.in_features - numeric - sum(net.embx[v] for _, v in ids))  # look-ahead inputs (computed, not folded)
            per = h0 * (numeric + len(ids) + extra) + lin(enc2)
            rows = n if name == 'global' else int((o[off:off + n * w].reshape(n, w)[:, 0] > 0.5).sum())
            t += rows * per
        start = 4
        for name in net.pointer:
            cnt = net.layout[name][1]; p0, p2 = net.ptr[name][0], net.ptr[name][2]
            legal = int(mk[start:start + cnt].sum()); start += cnt
            if legal:
                t += p0.out_features * C + legal * (p0.out_features * (p0.in_features - C) + p2.in_features)
        if net.cfg['attn']:
            T = 1 + sum(net.layout[x][1] for x in getattr(net, 'att_ents', net.ents)); da = net.cfg['attn_dim']
            t += net.cfg['attn'] * T * (lin(net.layers[0].qkv) + lin(net.layers[0].o) + lin(net.layers[0].f1) + lin(net.layers[0].f2) + 2 * T * da)
        tot += t
    return tot / N
