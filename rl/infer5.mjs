// tools/rl/infer5.mjs — (A1, 11-13, arch) plain-JS forward of the model5.SPNet5 policy (export5.py manifest + bin).
// Written fresh (not from infer2at.mjs). Exact rewrite of the torch forward up to float rounding:
//  - every id field's embedding is folded into a per-code table of its row encoder's first layer (the bond identity
//    embedding into a per-row bias), as infer2.mjs does;
//  - absent tokens are skipped: in torch their keys get -1e4 (softmax weight exactly 0 in f32), they never enter the
//    pooling, and their actions are masked, so the present tokens' values do not depend on them;
//  - the unit <-> bond attention bias (beta per head) is added for each (unit token, bond token of one of its bonds) pair
//    in both directions; GELU is the tanh form; LayerNorm eps 1e-5 (biased variance); QK-LayerNorm per head.
//   const net = SPPolicy5.load('policy5');  const { logits } = net.forward(obs, mask);
import { readFileSync } from 'node:fs';

const C0 = Math.sqrt(2 / Math.PI);
const gelu = (x) => 0.5 * x * (1 + Math.tanh(C0 * (x + 0.044715 * x * x * x)));

export class SPPolicy5 {
  constructor(m, f32) {
    this.m = m; this.D = m.D; this.A = m.A;
    const cfg = m.cfg; this.d = cfg.d; this.H = cfg.heads; this.dh = cfg.d / cfg.heads; this.ffn = cfg.ffn;
    const W = (name) => { const t = m.tensors[name]; if (!t) throw new Error(`missing tensor ${name}`); return { data: f32.subarray(t.offset, t.offset + t.shape.reduce((x, y) => x * y, 1)), shape: t.shape }; };
    const lin = (p, bias = true) => { const w = W(`${p}.weight`); return { w: w.data, b: bias ? W(`${p}.bias`).data : null, out: w.shape[0], inn: w.shape[1] }; };
    const ln = (p) => ({ w: W(`${p}.weight`).data, b: W(`${p}.bias`).data, n: W(`${p}.weight`).shape[0] });
    this.W = W; this.lin = lin;
    const d = this.d;
    // folded first layers: columns [own numeric, own id embeddings..., merged blocks' (numeric, ids)..., bond id (bonds)]
    const fold = (l0, parts, embPrefix) => {
      let col = 0; const segs = [];
      for (const L of parts) {
        segs.push({ L, kind: 'num', col }); col += L.numeric.length;
        for (const [field, voc] of L.ids) {
          const E = W(`${embPrefix}${voc}.weight`), dim = m.emb[voc], n = E.shape[0], T = new Float32Array(n * l0.out);
          for (let code = 0; code < n; code++) for (let o = 0; o < l0.out; o++) {
            let s = 0; const row = o * l0.inn + col;
            for (let k = 0; k < dim; k++) s += l0.w[row + k] * E.data[code * dim + k];
            T[code * l0.out + o] = s;
          }
          segs.push({ L, kind: 'id', field, n, T, col }); col += dim;
        }
      }
      return { segs, col };
    };
    this.typ = W('tower.typ').data;
    this.blocks = [];
    const bondId = W('tower.bond_id');
    m.tok.forEach((name, j) => {
      const L = m.layout[name], parts = [L, ...(m.merge[name] || []).map((x) => m.layout[x])];
      const l0 = lin(`tower.rowenc.${name}.0`), l2 = lin(`tower.rowenc.${name}.2`);
      const { segs, col } = fold(l0, parts, 'tower.emb.');
      let rowBias = null;
      if (name === 'bonds') { // the bond identity embedding: a constant per row
        const be = bondId.shape[1]; rowBias = new Float32Array(L.count * l0.out);
        for (let r = 0; r < L.count; r++) for (let o = 0; o < l0.out; o++) {
          let s = 0; for (let k = 0; k < be; k++) s += l0.w[o * l0.inn + col + k] * bondId.data[r * be + k];
          rowBias[r * l0.out + o] = s;
        }
      }
      const pf = m.pieceField[name], planM = name === 'plan' && !!m.tensors['tower.memb.plan.weight']; // (A4) the plan token's P / S links
      this.blocks.push({ name, j, L, l0, l2, segs, rowBias, pf: pf === undefined ? -1 : pf, planM,
        memb: pf === undefined && !planM ? null : lin(`tower.memb.${name}`, false) });
    });
    const gL = [m.layout.global, ...(m.gmerge || []).map((x) => m.layout[x])];
    this.g0 = lin('tower.genc.0'); this.g2 = lin('tower.genc.2'); this.gfold = fold(this.g0, gL, 'tower.emb.');
    this.layers = [];
    for (let i = 0; i < cfg.layers; i++) {
      const p = `tower.layers.${i}`;
      this.layers.push({ ln1: ln(`${p}.ln1`), qkv: lin(`${p}.qkv`), o: lin(`${p}.o`), ln2: ln(`${p}.ln2`), f1: lin(`${p}.f1`), f2: lin(`${p}.f2`),
        qn: cfg.qkn ? ln(`${p}.qn`) : null, kn: cfg.qkn ? ln(`${p}.kn`) : null });
    }
    this.lnf = ln('tower.lnf');
    this.beta = m.memberBias ? W('tower.beta').data : null;
    this.member = new Map(Object.entries(m.member).map(([k, v]) => [Number(k), v]));
    this.t0 = lin('trunk.0'); this.t2 = lin('trunk.2'); this.sc = lin('scalar');
    this.ptr = {};
    for (const n of m.pointer) this.ptr[n] = { p0: lin(`ptr.${n}.0`), p2: lin(`ptr.${n}.2`) };
    if (m.band) {
      this.b0 = lin('band_enc.0'); this.b2 = lin('band_enc.2'); this.bfold = fold(this.b0, [m.band], 'band_emb.');
    }
    this.logits = new Float32Array(this.A);
  }

  static load(base) {
    const m = JSON.parse(readFileSync(`${base}.json`, 'utf8')), buf = readFileSync(`${base}.bin`);
    return new SPPolicy5(m, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)));
  }

  static dense(l, x, xo, y, yo, act) { // y[yo..] = act(W x[xo..] + b); act: 0 none, 1 relu, 2 gelu
    const { w, b, out, inn } = l;
    for (let o = 0; o < out; o++) {
      let s = b ? b[o] : 0; const row = o * inn;
      for (let i = 0; i < inn; i++) s += w[row + i] * x[xo + i];
      y[yo + o] = act === 1 ? (s < 0 ? 0 : s) : act === 2 ? gelu(s) : s;
    }
  }

  static layerNorm(p, x, xo, y, yo, n) {
    let mu = 0; for (let i = 0; i < n; i++) mu += x[xo + i]; mu /= n;
    let v = 0; for (let i = 0; i < n; i++) { const t = x[xo + i] - mu; v += t * t; } v /= n;
    const r = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < n; i++) y[yo + i] = (x[xo + i] - mu) * r * p.w[i] + p.b[i];
  }

  // first layer of a folded row encoder for row r of its parts, then the second layer: out (d)
  _row(l0, l2, segs, obs, r, h0, out, rowBias) {
    for (let o = 0; o < l0.out; o++) h0[o] = l0.b[o] + (rowBias ? rowBias[r * l0.out + o] : 0);
    for (const s of segs) {
      const base = s.L.offset + r * s.L.width;
      if (s.kind === 'num') {
        const num = s.L.numeric, nn = num.length;
        for (let o = 0; o < l0.out; o++) { let t = 0; const row = o * l0.inn + s.col; for (let i = 0; i < nn; i++) t += l0.w[row + i] * obs[base + num[i]]; h0[o] += t; }
      } else {
        let code = Math.round(obs[base + s.field]); if (!(code >= 0)) code = 0; if (code > s.n - 1) code = s.n - 1;
        const T = s.T, at = code * l0.out; for (let o = 0; o < l0.out; o++) h0[o] += T[at + o];
      }
    }
    for (let o = 0; o < l0.out; o++) h0[o] = gelu(h0[o]);
    SPPolicy5.dense(l2, h0, 0, out, 0, 0);
  }

  forward(obs, mask) {
    const d = this.d, m = this.m, H = this.H, dh = this.dh;
    // ---- tokens (present only): global first
    const toks = [], meta = [];
    const h0 = new Float32Array(Math.max(this.g0.out, ...this.blocks.map((b) => b.l0.out)));
    { const x = new Float32Array(d); this._row(this.g0, this.g2, this.gfold.segs, obs, 0, h0, x, null); for (let k = 0; k < d; k++) x[k] += this.typ[k]; toks.push(x); meta.push({ block: null, r: 0, bonds: null, bond: -1 }); }
    const span = {};
    for (const B of this.blocks) {
      const { L, j } = B; span[B.name] = new Int32Array(L.count).fill(-1);
      for (let r = 0; r < L.count; r++) {
        const base = L.offset + r * L.width;
        if (B.name !== 'bonds' && !(obs[base] > 0.5)) continue;
        const x = new Float32Array(d); this._row(B.l0, B.l2, B.segs, obs, r, h0, x, B.rowBias);
        const ty = (j + 1) * d; for (let k = 0; k < d; k++) x[k] += this.typ[ty + k];
        let bonds = null;
        if (B.memb) {
          if (B.planM) { // the plan row's P one-hot (fields 1..NB) and S one-hot (1+NB..2NB), clamped to 0 / 1
            bonds = []; for (let k = 0; k < m.NB; k++) if (obs[base + 1 + k] + obs[base + 1 + m.NB + k] > 0) bonds.push(k);
          } else {
            let code = Math.round(obs[base + B.pf]); if (!(code >= 0)) code = 0; if (code > m.pieceVocab - 1) code = m.pieceVocab - 1;
            bonds = (this.member.get(code) || []).filter((k) => k >= 0 && k < m.NB);
          }
          const W = B.memb.w, NB = B.memb.inn;
          for (const k of bonds) for (let o = 0; o < d; o++) x[o] += W[o * NB + k];
        }
        span[B.name][r] = toks.length; toks.push(x); meta.push({ block: B.name, r, bonds, bond: B.name === 'bonds' ? r : -1 });
      }
    }
    const T = toks.length;
    // membership pairs: bias[i][j] != 0 only for (unit i, bond token of one of i's bonds) and the reverse
    let pair = null;
    if (this.beta) {
      pair = new Uint8Array(T * T);
      const bondTok = span.bonds;
      for (let i = 0; i < T; i++) if (meta[i].bonds) for (const k of meta[i].bonds) { const t = bondTok ? bondTok[k] : -1; if (t >= 0) { pair[i * T + t] = 1; pair[t * T + i] = 1; } }
    }
    // ---- transformer layers
    const a = new Float32Array(d), qkv = new Float32Array(T * 3 * d), y = new Float32Array(d), o1 = new Float32Array(d);
    const f = new Float32Array(this.ffn * d), sc = new Float32Array(T), sq = 1 / Math.sqrt(dh);
    for (const Ly of this.layers) {
      for (let i = 0; i < T; i++) {
        SPPolicy5.layerNorm(Ly.ln1, toks[i], 0, a, 0, d);
        SPPolicy5.dense(Ly.qkv, a, 0, qkv, i * 3 * d, 0);
        if (Ly.qn) for (let h = 0; h < H; h++) {
          SPPolicy5.layerNorm(Ly.qn, qkv, i * 3 * d + h * dh, qkv, i * 3 * d + h * dh, dh);
          SPPolicy5.layerNorm(Ly.kn, qkv, i * 3 * d + d + h * dh, qkv, i * 3 * d + d + h * dh, dh);
        }
      }
      const nx = [];
      for (let i = 0; i < T; i++) {
        for (let h = 0; h < H; h++) {
          const qo = i * 3 * d + h * dh, bh = this.beta ? this.beta[h] : 0;
          let mx = -Infinity;
          for (let j = 0; j < T; j++) {
            let s = 0; const ko = j * 3 * d + d + h * dh;
            for (let k = 0; k < dh; k++) s += qkv[qo + k] * qkv[ko + k];
            s *= sq; if (pair && pair[i * T + j]) s += bh;
            sc[j] = s; if (s > mx) mx = s;
          }
          let z = 0; for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
          for (let k = 0; k < dh; k++) y[h * dh + k] = 0;
          for (let j = 0; j < T; j++) { const p = sc[j] / z, vo = j * 3 * d + 2 * d + h * dh; for (let k = 0; k < dh; k++) y[h * dh + k] += p * qkv[vo + k]; }
        }
        SPPolicy5.dense(Ly.o, y, 0, o1, 0, 0);
        const x = new Float32Array(d); for (let k = 0; k < d; k++) x[k] = toks[i][k] + o1[k];
        nx.push(x);
      }
      for (let i = 0; i < T; i++) {
        const x = nx[i];
        SPPolicy5.layerNorm(Ly.ln2, x, 0, a, 0, d);
        SPPolicy5.dense(Ly.f1, a, 0, f, 0, 2);
        SPPolicy5.dense(Ly.f2, f, 0, o1, 0, 0);
        for (let k = 0; k < d; k++) x[k] += o1[k];
        toks[i] = x;
      }
    }
    for (let i = 0; i < T; i++) { const x = new Float32Array(d); SPPolicy5.layerNorm(this.lnf, toks[i], 0, x, 0, d); toks[i] = x; }
    // ---- pooling: [global token, mean, max] over the present tokens except the global one
    const cat = new Float32Array(3 * d); cat.set(toks[0], 0);
    if (T > 1) {
      for (let k = 0; k < d; k++) { cat[d + k] = 0; cat[2 * d + k] = -Infinity; }
      for (let i = 1; i < T; i++) { const x = toks[i]; for (let k = 0; k < d; k++) { cat[d + k] += x[k]; if (x[k] > cat[2 * d + k]) cat[2 * d + k] = x[k]; } }
      for (let k = 0; k < d; k++) cat[d + k] /= T - 1;
    }
    const t0 = new Float32Array(this.t0.out), c = new Float32Array(this.t2.out);
    SPPolicy5.dense(this.t0, cat, 0, t0, 0, 1); SPPolicy5.dense(this.t2, t0, 0, c, 0, 1);
    // ---- heads
    const logits = this.logits; logits.fill(-1e9);
    for (let q = 0; q < this.sc.out; q++) if (!mask || mask[q]) { let s = this.sc.b[q]; for (let i = 0; i < this.sc.inn; i++) s += this.sc.w[q * this.sc.inn + i] * c[i]; logits[q] = s; }
    let k = this.sc.out;
    for (const n of m.pointer) {
      const { p0, p2 } = this.ptr[n], cnt = n === 'bands' ? m.band.count : m.layout[n].count, hd = p0.inn - c.length;
      let need = false; for (let r = 0; r < cnt; r++) if (!mask || mask[k + r]) { need = true; break; }
      if (!need) { k += cnt; continue; }
      const cc = new Float32Array(p0.out), hid = new Float32Array(p0.out);
      for (let jj = 0; jj < p0.out; jj++) { let s = p0.b[jj]; const row = jj * p0.inn + hd; for (let i = 0; i < c.length; i++) s += p0.w[row + i] * c[i]; cc[jj] = s; }
      const bh0 = n === 'bands' ? new Float32Array(this.b0.out) : null, bh = n === 'bands' ? new Float32Array(this.b2.out) : null;
      for (let r = 0; r < cnt; r++, k++) {
        if (mask && !mask[k]) continue;
        let h;
        if (n === 'bands') { // band rows: their own encoder (ReLU), no attention
          for (let o = 0; o < this.b0.out; o++) bh0[o] = this.b0.b[o];
          for (const s of this.bfold.segs) {
            const base = s.L.offset + r * s.L.width;
            if (s.kind === 'num') { const num = s.L.numeric; for (let o = 0; o < this.b0.out; o++) { let t = 0; const row = o * this.b0.inn + s.col; for (let i = 0; i < num.length; i++) t += this.b0.w[row + i] * obs[base + num[i]]; bh0[o] += t; } }
            else { let code = Math.round(obs[base + s.field]); if (!(code >= 0)) code = 0; if (code > s.n - 1) code = s.n - 1; for (let o = 0; o < this.b0.out; o++) bh0[o] += s.T[code * this.b0.out + o]; }
          }
          for (let o = 0; o < this.b0.out; o++) if (bh0[o] < 0) bh0[o] = 0;
          SPPolicy5.dense(this.b2, bh0, 0, bh, 0, 1); h = bh;
        } else {
          const t = span[n][r]; if (t < 0) continue; // an absent row (its action is masked in a valid observation)
          h = toks[t];
        }
        for (let jj = 0; jj < p0.out; jj++) { let s = cc[jj]; const row = jj * p0.inn; for (let i = 0; i < hd; i++) s += p0.w[row + i] * h[i]; hid[jj] = s < 0 ? 0 : s; }
        let s = p2.b[0]; for (let jj = 0; jj < p2.inn; jj++) s += p2.w[jj] * hid[jj];
        logits[k] = s;
      }
    }
    return { logits, tokens: T };
  }
}
