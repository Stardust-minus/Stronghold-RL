"""(A1, 11-13, arch) v5-lite: a tokenized entity transformer policy with a separate, larger, training-only critic.

Every entity row of the version 3 observation becomes a token (shop slots, owned pieces, bonds, enemy groups, reward
slots, personal cards, draft cards, teammates; with the A1 ext.board blocks also the 28 deployable tiles), plus one
global token (global + extra + the ext.hist battle history). Rows of companion blocks with the same row index are merged
into their token (owned <- posX3, shop <- shopH, mates <- mateBonds). Each token: its numeric fields and embedded id
fields through a two-layer row encoder, plus a type embedding; a unit token (a piece code: shop, owned, rewards, tiles3)
also adds a projection of its catalogue bond membership (multi-hot over the bonds, a static table from
lookahead_table.mjs), and with `member_bias` every head's attention logit between a unit token and the token of a bond it
belongs to gets a learned bias (unit <-> bond membership, the distinct counts behind national tiers).
`layers` pre-LN transformer layers (QK-LayerNorm, GELU-tanh FFN), a final LayerNorm, then a context vector from the global
token and the masked mean / max of the present tokens. Actions: END / REFRESH / LEVEL / FREEZE from the context; buy /
sell / reward / personal / card pointers from each slot's token and the context; band rows (only filled at band
decisions) through their own row encoder (no attention), as in SPNet.
The critic (`critic` cfg, default a d192 x 3 tower) has its own token encoder and transformer and reads the observation
only: it is never deployed. `critic=None`: the value from the policy context (a shared trunk, for ablations).
Only Embedding, Linear, LayerNorm, GELU-tanh, softmax attention, masked mean / max: a plain-JS forward can mirror it.
"""
import math
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EMB, POINTER, SCALARS

TOK = ['shop', 'owned', 'bonds', 'enemies', 'rewards', 'personal', 'cards', 'mates', 'tiles3', 'plan', 'planNat']
# (A4) the natPlan blocks: 'plan' (the seat's plan P / S and its progress) is a token of its own, linked to the bond tokens
# of P and S like a unit to its bonds; 'planNat' (one row per nation at plan decisions) are tokens read by the plan head; the
# per-slot plan rows (in P, in S, new / sole P member) join their unit tokens
MERGE = {'owned': ['posX3', 'planOwned'], 'shop': ['shopH', 'planShop'], 'rewards': ['planRewards'], 'mates': ['mateBonds']}
GMERGE = ['extra', 'hist']
PIECE_FIELD = {'shop': 3, 'owned': 4, 'rewards': 2, 'tiles3': 11}
BOND_EMB = 16


def gelu(x):
    return F.gelu(x, approximate='tanh')


class Layer(nn.Module):
    def __init__(self, d, heads, ffn, qkn):
        super().__init__()
        self.h = heads; self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.f1 = nn.Linear(d, ffn * d); self.f2 = nn.Linear(ffn * d, d)
        self.qkn = qkn
        if qkn:
            self.qn = nn.LayerNorm(d // heads); self.kn = nn.LayerNorm(d // heads)

    def forward(self, x, bias):  # x (B, T, d); bias (B, 1|h, T|1, T) additive
        B, T, d = x.shape; h = self.h
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)
        if self.qkn:
            q = self.qn(q); k = self.kn(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, d))
        return x + self.f2(gelu(self.f1(self.ln2(x))))


class Tower(nn.Module):
    """Tokens of one observation batch -> attended tokens (B, T, d), present mask (B, T), spans {block: (a, b)}."""

    def __init__(self, spec, member, d=128, layers=2, heads=4, ffn=2, qkn=True, member_bias=True, use=None):
        super().__init__()
        blocks, vocab = spec['blocks'], spec['vocab']
        use = set(use) if use is not None else None  # optional subset of the extension blocks (ablations)
        ok = lambda n: n in blocks and (use is None or n not in ('posX3', 'tiles3', 'hist', 'shopH') or n in use)
        self.emb = nn.ModuleDict({k: nn.Embedding(vocab[k], e) for k, e in EMB.items() if k in vocab})
        ids = defaultdict(list)
        for b, f, v in spec['ids']:
            ids[b].append((f, v))
        self.lay = {}
        def reg(name):
            b = blocks[name]; idf = {f for f, _ in ids[name]}
            num = [i for i in range(b['width']) if i not in idf]
            self.register_buffer(f'num_{name}', torch.tensor(num, dtype=torch.long), persistent=False)
            self.lay[name] = (b['offset'], b['count'], b['width'], ids[name])
            return len(num) + sum(EMB[v] for _, v in ids[name])
        self.tok = [n for n in TOK if ok(n)]
        self.merge = {n: [m for m in MERGE.get(n, []) if ok(m)] for n in self.tok}
        self.gmerge = [m for m in GMERGE if ok(m)]
        self.rowenc = nn.ModuleDict()
        NB = blocks['bonds']['count']; self.NB = NB
        for n in self.tok:
            i = reg(n) + sum(reg(m) for m in self.merge[n]) + (BOND_EMB if n == 'bonds' else 0)
            self.rowenc[n] = nn.Sequential(nn.Linear(i, d), nn.GELU(approximate='tanh'), nn.Linear(d, d))
        gi = reg('global') + sum(reg(m) for m in self.gmerge)
        self.genc = nn.Sequential(nn.Linear(gi, d), nn.GELU(approximate='tanh'), nn.Linear(d, d))
        self.bond_id = nn.Parameter(torch.randn(NB, BOND_EMB) * 0.1)
        self.typ = nn.Parameter(torch.randn(len(self.tok) + 1, d) * 0.02)
        # catalogue bond membership: piece code -> multi-hot over the bonds (lookahead_table.mjs pieceBonds)
        V = vocab['piece']; M = torch.zeros(V, NB)
        for code, lst in (member or {}).items():
            for k in lst:
                if int(code) < V and 0 <= int(k) < NB:
                    M[int(code), int(k)] = 1.0
        self.register_buffer('member', M, persistent=False)
        self.memb = nn.ModuleDict({n: nn.Linear(NB, d, bias=False) for n in list(PIECE_FIELD) + ['plan'] if n in self.tok})
        self.layers = nn.ModuleList([Layer(d, heads, ffn, qkn) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.heads = heads; self.d = d
        self.beta = nn.Parameter(torch.zeros(heads)) if member_bias and 'bonds' in self.tok else None

    def _rows(self, obs, name):
        off, n, w, ids = self.lay[name]
        x = obs[:, off:off + n * w].reshape(obs.shape[0], n, w)
        parts = [x.index_select(2, getattr(self, f'num_{name}'))]
        for f, v in ids:
            parts.append(self.emb[v](x[..., f].round().long().clamp_(0, self.emb[v].num_embeddings - 1)))
        return x, parts

    def forward(self, obs):
        B = obs.shape[0]
        _, gp = self._rows(obs, 'global')
        for m in self.gmerge:
            gp += self._rows(obs, m)[1]
        toks = [self.genc(torch.cat(gp, -1)) + self.typ[0]]
        keep = [torch.ones(B, 1, dtype=torch.bool, device=obs.device)]
        spans, t0, mem = {}, 1, {}
        for j, n in enumerate(self.tok):
            x, parts = self._rows(obs, n)
            for m in self.merge[n]:
                parts += self._rows(obs, m)[1]
            if n == 'bonds':
                parts.append(self.bond_id.unsqueeze(0).expand(B, -1, -1))
                pres = torch.ones(B, x.shape[1], dtype=torch.bool, device=obs.device)
            else:
                pres = x[..., 0] > 0.5
            h = self.rowenc[n](torch.cat(parts, -1)) + self.typ[j + 1]
            if n in self.memb:
                if n == 'plan':  # (A4) the plan's P and S one-hots (fields 1..NB, 1+NB..2NB) as its "bonds"
                    mm = (x[..., 1:1 + self.NB] + x[..., 1 + self.NB:1 + 2 * self.NB]).clamp(0, 1)
                else:
                    mm = self.member[x[..., PIECE_FIELD[n]].round().long().clamp_(0, self.member.shape[0] - 1)]  # (B, n, NB)
                mm = mm * pres.unsqueeze(-1).to(mm.dtype)
                h = h + self.memb[n](mm); mem[n] = mm
            toks.append(h); keep.append(pres); spans[n] = (t0, t0 + x.shape[1]); t0 += x.shape[1]
        xt = torch.cat(toks, 1); kp = torch.cat(keep, 1); T = xt.shape[1]
        neg = torch.zeros(B, 1, 1, T, device=obs.device, dtype=xt.dtype).masked_fill(~kp[:, None, None, :], -1e4)
        if self.beta is not None and mem:
            Mf = torch.zeros(B, T, T, device=obs.device, dtype=xt.dtype)
            a0, a1 = spans['bonds']
            for n, mm in mem.items():
                s0, s1 = spans[n]
                Mf[:, s0:s1, a0:a1] = mm; Mf[:, a0:a1, s0:s1] = mm.transpose(1, 2)
            bias = neg + self.beta.view(1, -1, 1, 1) * Mf.unsqueeze(1)
        else:
            bias = neg
        for L in self.layers:
            xt = L(xt, bias)
        return self.lnf(xt), kp, spans


def pool(x, kp):
    """[global token, masked mean, masked max] of the present tokens (the global token excluded from mean / max)."""
    m = kp[:, 1:].unsqueeze(-1).to(x.dtype); xs = x[:, 1:]
    mean = (xs * m).sum(1) / m.sum(1).clamp_min(1.0)
    mx = (xs * m - (1 - m) * 1e4).max(1).values
    mx = torch.where(kp[:, 1:].any(1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([x[:, 0], mean, mx], -1)


class Critic(nn.Module):
    def __init__(self, spec, member, d=192, layers=3, heads=4, ffn=4, qkn=True, member_bias=True, hidden=512, use=None):
        super().__init__()
        self.tower = Tower(spec, member, d, layers, heads, ffn, qkn, member_bias, use)
        self.head = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs):
        x, kp, _ = self.tower(obs)
        return self.head(pool(x, kp)).squeeze(-1)


class SPNet5(nn.Module):
    def __init__(self, spec, arch='v5', d=128, layers=2, heads=4, ffn=2, qkn=True, member_bias=True, ctx=256, critic=None,
                 member=None, use=None, rnn=None):
        super().__init__()
        self.spec = spec
        self.cfg = dict(arch=arch, d=d, layers=layers, heads=heads, ffn=ffn, qkn=qkn, member_bias=member_bias, ctx=ctx,
                        critic=critic, member=member, use=use)
        if rnn:  # (A3) absent from older configs: they build as before
            self.cfg['rnn'] = rnn
        blocks = spec['blocks']; self.D, self.A = spec['dim'], spec['actions']
        self.tower = Tower(spec, member, d, layers, heads, ffn, qkn, member_bias, use)
        self.trunk = nn.Sequential(nn.Linear(3 * d, 512), nn.ReLU(), nn.Linear(512, ctx), nn.ReLU())
        self.scalar = nn.Linear(ctx, SCALARS)
        self.pointer = [n for n in POINTER if n in blocks]
        self.ptr = nn.ModuleDict({n: nn.Sequential(nn.Linear((64 if n == 'bands' else d) + ctx, 128), nn.ReLU(), nn.Linear(128, 1))
                                  for n in self.pointer})
        if 'bands' in self.pointer:  # band rows: their own row encoder, outside the attention (filled at band decisions only)
            b = blocks['bands']; ids = [(f, v) for bb, f, v in spec['ids'] if bb == 'bands']; idf = {f for f, _ in ids}
            self.register_buffer('num_bands', torch.tensor([i for i in range(b['width']) if i not in idf], dtype=torch.long), persistent=False)
            self.band_lay = (b['offset'], b['count'], b['width'], ids)
            self.band_emb = nn.ModuleDict({v: nn.Embedding(spec['vocab'][v], EMB[v]) for _, v in ids})
            self.band_enc = nn.Sequential(nn.Linear(len(self.num_bands) + sum(EMB[v] for _, v in ids), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.value = Critic(spec, member, use=use, **critic) if critic is not None else nn.Sequential(nn.Linear(ctx, 128), nn.ReLU(), nn.Linear(128, 1))
        # (A3) zero-init residual recurrent branch: c' = c + W_out h_t, h_t = cell(c_t, h_{t-1}); W_out = 0 at init, so the policy
        # starts as the feed-forward one (function-preserving). The critic stays feed-forward (it reads the observation only).
        self.rnn = None
        if rnn:
            H = int(rnn.get('hidden', 128)); kind = rnn.get('cell', 'lstm')
            assert kind in ('lstm', 'gru'), kind
            self.rnn = kind; self.rnn_h = H; self.state_size = 2 * H if kind == 'lstm' else H
            self.rnn_cell = nn.LSTMCell(ctx, H) if kind == 'lstm' else nn.GRUCell(ctx, H)
            self.rnn_out = nn.Linear(H, ctx)
            nn.init.zeros_(self.rnn_out.weight); nn.init.zeros_(self.rnn_out.bias)
        counts = [blocks[n]['count'] for n in self.pointer]
        assert SCALARS + sum(counts) == self.A, (counts, self.A)
        self.with_value = True  # eval.py may switch it off (the critic is never needed to act)

    def _cell(self, c, state):
        """(A3) one recurrent step: (B, ctx), state (B, S) -> new state (B, S), the residual (B, ctx)."""
        H = self.rnn_h
        if self.rnn == 'lstm':
            h, cs = self.rnn_cell(c, (state[:, :H], state[:, H:]))
            return torch.cat([h, cs], -1), self.rnn_out(h)
        h = self.rnn_cell(c, state)
        return h, self.rnn_out(h)

    def forward(self, obs, mask=None, state=None):
        """obs (B, D), mask (B, A) -> logits, value; a recurrent net (cfg rnn) also takes the state before this decision
        (B, S; None = zeros, a trajectory's first decision) and then returns (logits, value, new state)."""
        x, kp, spans = self.tower(obs)
        c = self.trunk(pool(x, kp))
        if self.rnn is not None:
            st = state if state is not None else torch.zeros(obs.shape[0], self.state_size, device=obs.device, dtype=c.dtype)
            new, dc = self._cell(c, st.to(c.dtype))
            logits, v = self._heads(obs, x, spans, c + dc, mask)
            return logits, v, new
        return self._heads(obs, x, spans, c, mask)

    def forward_seq(self, obs, mask, h0, valid, burn):
        """(A3) recurrent PPO training: obs (N, T, D), mask (N, T, A), h0 (N, S) = the stored state before step 0, valid (N, T)
        bool (False = padding: the state is carried unchanged), the first `burn` steps a burn-in (no gradient, no outputs: they
        only rebuild the state from the stored one under the current weights). -> logits (N, T - burn, A), value (N, T - burn)."""
        N, T, D = obs.shape; st = h0.to(obs.dtype)
        if burn:
            with torch.no_grad():
                ob = obs[:, :burn].reshape(N * burn, D)
                x, kp, _ = self.tower(ob)
                c = self.trunk(pool(x, kp)).reshape(N, burn, -1)
                for t in range(burn):
                    new, _ = self._cell(c[:, t], st.to(c.dtype))
                    st = torch.where(valid[:, t:t + 1], new.to(st.dtype), st)
            st = st.detach()
        L = T - burn
        ol = obs[:, burn:].reshape(N * L, D); ml = mask[:, burn:].reshape(N * L, -1)
        x, kp, spans = self.tower(ol)
        c = self.trunk(pool(x, kp)).reshape(N, L, -1)
        outs = []
        for t in range(L):
            new, dc = self._cell(c[:, t], st.to(c.dtype))
            outs.append(c[:, t] + dc)
            st = torch.where(valid[:, burn + t:burn + t + 1], new.to(st.dtype), st)
        cc = torch.stack(outs, 1).reshape(N * L, -1)
        logits, v = self._heads(ol, x, spans, cc, ml)
        return logits.reshape(N, L, -1), v.reshape(N, L)

    def _heads(self, obs, x, spans, c, mask):
        logits = [self.scalar(c)]
        for n in self.pointer:
            if n == 'bands':
                off, k, w, ids = self.band_lay
                xb = obs[:, off:off + k * w].reshape(obs.shape[0], k, w)
                parts = [xb.index_select(2, self.num_bands)] + [self.band_emb[v](xb[..., f].round().long().clamp_(0, self.band_emb[v].num_embeddings - 1)) for f, v in ids]
                h = self.band_enc(torch.cat(parts, -1))
            else:
                a_, b_ = spans[n]; h = x[:, a_:b_]
            cc = c.unsqueeze(1).expand(-1, h.shape[1], -1)
            logits.append(self.ptr[n](torch.cat([h, cc], -1)).squeeze(-1))
        logits = torch.cat(logits, -1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        if not self.with_value:
            v = torch.zeros(obs.shape[0], device=obs.device)
        elif isinstance(self.value, Critic):
            v = self.value(obs)
        else:
            v = self.value(c).squeeze(-1)
        return logits, v


def deployed_macs5(net, obs, mask):
    """Mean multiply-adds per decision of the deployed policy forward (present tokens only, embeddings folded into the
    row encoders' first layers, legal pointer rows only, no critic) for SPNet5. obs (N, D) / mask (N, A) CPU tensors."""
    tw = net.tower; d = tw.d; L = len(tw.layers); ffn = tw.layers[0].f1.out_features // d if L else 2
    lin = lambda m_: m_.in_features * m_.out_features
    tot = 0.0; N = obs.shape[0]
    for k in range(N):
        o, mk = obs[k:k + 1], mask[k]
        T = 1; t = lin(tw.genc[0]) + lin(tw.genc[2])
        for n in tw.tok:
            off, cnt, w, ids = tw.lay[n]
            x = o[0, off:off + cnt * w].reshape(cnt, w)
            rows = cnt if n == 'bonds' else int((x[:, 0] > 0.5).sum())
            enc0 = tw.rowenc[n][0]
            t += rows * (enc0.out_features * (enc0.in_features) * 0 + enc0.out_features * (len(getattr(tw, f'num_{n}')) + len(ids)) + lin(tw.rowenc[n][2]))
            if n in tw.memb:
                t += rows * d * 3  # ≤ 3 bonds per piece: a sum of 3 columns
            T += rows
        t += L * (T * (4 * d * d + 2 * ffn * d * d) + 2 * T * T * d)
        t += sum(lin(m_) for m_ in net.trunk if isinstance(m_, nn.Linear))
        C = net.scalar.in_features; t += int(mk[:4].sum()) * C
        start = 4
        for n in net.pointer:
            cnt = net.spec['blocks'][n]['count']; p0, p2 = net.ptr[n][0], net.ptr[n][2]
            legal = int(mk[start:start + cnt].sum()); start += cnt
            if legal:
                t += p0.out_features * C + legal * (p0.out_features * (p0.in_features - C) + p2.in_features)
        tot += t
    return tot / N
