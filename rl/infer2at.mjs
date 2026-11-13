// tools/rl/infer2at.mjs — plain-JS forward of arm AT (11-13): infer2.mjs (SPPolicy2) plus SPNet2's residual entity
// self-attention (model2.py, cfg.entattn): a global token (ea_g on the global + bonds encodings) and one token per present
// row of the ea blocks (ea_in) and per present bond row (ea_bond on the raw bond fields), each plus its type embedding,
// go through cfg.entattn pre-LN layers (QK-LayerNorm per head when cfg.qkln, softmax over the present tokens only, FFN);
// every present ea row then gets ea_out(x_L - x_0) added before pooling. Absent rows never take part (exactly as in torch,
// where they are masked keys and their own update is multiplied by 0).
//   const net = SPPolicyAT.load('policy');  const { logits } = net.forward(obs, mask, { value: false });
import { readFileSync } from 'node:fs';
import { SPPolicy2 } from './infer2.mjs';

export class SPPolicyAT extends SPPolicy2 {
  constructor(m, f32) {
    super(m, f32);
    const cfg = m.cfg || {}; if (!cfg.entattn) throw new Error('not an entattn checkpoint');
    const T = (name) => { const t = m.tensors[name]; if (!t) throw new Error(`missing tensor ${name}`); return { data: f32.subarray(t.offset, t.offset + t.shape.reduce((x, y) => x * y, 1)), shape: t.shape }; };
    const lin = (p) => { const w = T(`${p}.weight`), b = T(`${p}.bias`); return { w: w.data, b: b.data, out: w.shape[0], inn: w.shape[1] }; };
    const ln = (p) => ({ g: T(`${p}.weight`).data, b: T(`${p}.bias`).data });
    this.da = cfg.attn_dim; this.H = cfg.heads; this.qkln = !!cfg.qkln;
    this.eaEnts = Object.keys(m.tensors).filter((k) => /^ea_in\.[^.]+\.weight$/.test(k)).map((k) => k.split('.')[1]);
    this.eaIn = Object.fromEntries(this.eaEnts.map((n) => [n, lin(`ea_in.${n}`)])); this.eaOut = Object.fromEntries(this.eaEnts.map((n) => [n, lin(`ea_out.${n}`)]));
    this.eaG = lin('ea_g'); this.eaBond = lin('ea_bond'); this.typ = T('ea_typ').data;
    const [bo, bsz] = m.bondsAt; this.bw = this.eaBond.inn; this.bn = bsz / this.bw; this.bo0 = bo;
    this.layers = [];
    for (let l = 0; l < cfg.entattn; l++) {
      const p = `ea_layers.${l}`;
      this.layers.push({ ln1: ln(`${p}.ln1`), qkv: lin(`${p}.qkv`), o: lin(`${p}.o`), ln2: ln(`${p}.ln2`), f1: lin(`${p}.f1`), f2: lin(`${p}.f2`),
        ...(this.qkln ? { qn: ln(`${p}.qn`), kn: ln(`${p}.kn`) } : {}) });
    }
    const maxT = 1 + this.eaEnts.reduce((s, n) => s + this.blocks[n].L.count, 0) + this.bn, d = this.da;
    this.x = new Float32Array(maxT * d); this.x0 = new Float32Array(maxT * d); this.a = new Float32Array(maxT * d);
    this.qkvb = new Float32Array(maxT * 3 * d); this.y = new Float32Array(maxT * d); this.f = new Float32Array(2 * d);
    this.att = new Float64Array(maxT); this.gcat = new Float32Array(this.eaG.inn); this.tmp = new Float32Array(Math.max(...this.eaEnts.map((n) => this.blocks[n].l2.out)));
    this.ea = this._entAttn; // infer2.mjs calls it after the encoders and the bonds MLP, before pooling
  }

  static load(base) {
    const m = JSON.parse(readFileSync(`${base}.json`, 'utf8')), buf = readFileSync(`${base}.bin`);
    return new SPPolicyAT(m, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)));
  }

  static layerNorm(x, xo, n, p, y, yo) { // torch LayerNorm: biased variance, eps 1e-5
    let mu = 0; for (let i = 0; i < n; i++) mu += x[xo + i]; mu /= n;
    let v = 0; for (let i = 0; i < n; i++) { const t = x[xo + i] - mu; v += t * t; } v /= n;
    const r = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < n; i++) y[yo + i] = (x[xo + i] - mu) * r * p.g[i] + p.b[i];
  }

  static affine(l, x, xo, y, yo, add) { // y = W x + b (+ y when add)
    const { w, b, out, inn } = l;
    for (let o = 0; o < out; o++) { let s = b[o]; const row = o * inn; for (let i = 0; i < inn; i++) s += w[row + i] * x[xo + i]; y[yo + o] = add ? y[yo + o] + s : s; }
  }

  _entAttn(obs) {
    const d = this.da, H = this.H, dh = d / H, x = this.x, x0 = this.x0, typ = this.typ, A = SPPolicyAT.affine, LN = SPPolicyAT.layerNorm;
    // tokens: global, present rows of each ea block, present bond rows
    const g = this.blocks.global.rows[0]; this.gcat.set(g, 0); this.gcat.set(this.bo, g.length);
    let T = 0; const own = [];
    A(this.eaG, this.gcat, 0, x, 0, false); for (let k = 0; k < d; k++) x[k] += typ[k]; T = 1;
    this.eaEnts.forEach((n, j) => {
      const B = this.blocks[n];
      for (let r = 0; r < B.L.count; r++) if (B.present[r]) {
        A(this.eaIn[n], B.rows[r], 0, x, T * d, false); for (let k = 0; k < d; k++) x[T * d + k] += typ[(j + 1) * d + k];
        own.push([n, r, T]); T++;
      }
    });
    const tb = (this.eaEnts.length + 1) * d;
    for (let r = 0; r < this.bn; r++) {
      const base = this.bo0 + r * this.bw; if (!(obs[base] > 0)) continue;
      A(this.eaBond, obs, base, x, T * d, false); for (let k = 0; k < d; k++) x[T * d + k] += typ[tb + k]; T++;
    }
    x0.set(x.subarray(0, T * d));
    const a = this.a, qkv = this.qkvb, y = this.y, att = this.att, sc = 1 / Math.sqrt(dh);
    for (const L of this.layers) {
      for (let t = 0; t < T; t++) { LN(x, t * d, d, L.ln1, a, t * d); A(L.qkv, a, t * d, qkv, t * 3 * d, false); }
      if (this.qkln) for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
        LN(qkv, t * 3 * d + h * dh, dh, L.qn, qkv, t * 3 * d + h * dh); LN(qkv, t * 3 * d + d + h * dh, dh, L.kn, qkv, t * 3 * d + d + h * dh);
      }
      for (let t = 0; t < T; t++) for (let h = 0; h < H; h++) {
        const qo = t * 3 * d + h * dh; let mx = -Infinity;
        for (let u = 0; u < T; u++) { const ko = u * 3 * d + d + h * dh; let s = 0; for (let i = 0; i < dh; i++) s += qkv[qo + i] * qkv[ko + i]; s *= sc; att[u] = s; if (s > mx) mx = s; }
        let z = 0; for (let u = 0; u < T; u++) { att[u] = Math.exp(att[u] - mx); z += att[u]; }
        for (let i = 0; i < dh; i++) { let s = 0; for (let u = 0; u < T; u++) s += att[u] * qkv[u * 3 * d + 2 * d + h * dh + i]; y[t * d + h * dh + i] = s / z; }
      }
      for (let t = 0; t < T; t++) A(L.o, y, t * d, x, t * d, true);
      for (let t = 0; t < T; t++) {
        LN(x, t * d, d, L.ln2, a, t * d); SPPolicy2.dense(L.f1, a, t * d, this.f, true); A(L.f2, this.f, 0, x, t * d, true);
      }
    }
    for (const [n, r, t] of own) {
      const rows = this.blocks[n].rows[r], o = this.eaOut[n], dx = this.tmp;
      for (let k = 0; k < d; k++) a[k] = x[t * d + k] - x0[t * d + k];
      A(o, a, 0, dx, 0, false); for (let k = 0; k < o.out; k++) rows[k] += dx[k];
    }
  }
}
