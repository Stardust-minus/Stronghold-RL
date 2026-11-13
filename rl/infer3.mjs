// tools/rl/infer3.mjs — deployment forward pass (ppo3 work): infer2.mjs plus
//  (1) incremental: per-seat cache; an entity row whose inputs did not change since that seat's previous decision keeps
//      its encoding and, for pointer rows, its W_h·h part. The pointer hidden is defined as cc + P with P = Σ W_h·h summed
//      from 0 in a fixed order, so a cached value and a recomputed one are bit-identical whatever the cache history:
//      client and server fallback agree exactly;
//  (2) batch: several AI seats in one call; the large layers stream each weight row once for all seats;
//  (3) a decision with exactly one legal action needs no forward at all (decide()).
// Fixed operation order, IEEE float only, no transcendental functions.
import { readFileSync } from 'node:fs';

export class SPPolicy3 {
  constructor(m, f32) {
    this.m = m; this.D = m.D; this.A = m.A;
    const W = (n) => { const t = m.tensors[n]; return { data: f32.subarray(t.offset, t.offset + t.shape.reduce((x, y) => x * y, 1)), shape: t.shape }; };
    const lin = (p) => { const w = W(`${p}.weight`), b = W(`${p}.bias`); return { w: w.data, b: b.data, out: w.shape[0], inn: w.shape[1] }; };
    this.names = Object.keys(m.layout); this.blocks = {};
    for (const name of this.names) {
      const L = m.layout[name], l0 = lin(`enc.${name}.0`), l2 = lin(`enc.${name}.2`), tables = []; let col = L.numeric.length;
      for (const [field, voc] of L.ids) {
        const E = W(`emb.${voc}.weight`), dim = m.emb[voc], n = E.shape[0], T = new Float64Array(n * l0.out);
        for (let code = 0; code < n; code++) for (let o = 0; o < l0.out; o++) {
          let s = 0; const row = o * l0.inn + col; for (let k = 0; k < dim; k++) s += l0.w[row + k] * E.data[code * dim + k]; T[code * l0.out + o] = s;
        }
        tables.push({ field, n, T }); col += dim;
      }
      this.blocks[name] = { L, l0, l2, tables };
    }
    this.bonds0 = lin('bonds.0'); this.bonds2 = lin('bonds.2'); this.trunk0 = lin('trunk.0'); this.trunk2 = lin('trunk.2'); this.scalarL = lin('scalar');
    this.ptr = {}; this.C = this.trunk2.out;
    for (const name of m.pointer) { const p0 = lin(`ptr.${name}.0`), p2 = lin(`ptr.${name}.2`); this.ptr[name] = { p0, p2, o: p0.inn - this.C }; }
    this.value0 = lin('value.0'); this.value2 = lin('value.2');
    this.h0 = new Float32Array(512); this.stats = { rows: 0, rowsReused: 0 };
  }
  static load(base) { const m = JSON.parse(readFileSync(`${base}.json`, 'utf8')), buf = readFileSync(`${base}.bin`); return new SPPolicy3(m, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length))); }
  newCache() { return { rows: {} }; } // one per AI seat; reset at a new prep or match (optional: equal rows are always reused safely)

  // encoded row r of block `name` (cached when its inputs equal the cached copy); returns { y, P } (P: pointer partial or null)
  row(obs, name, r, cache) {
    const B = this.blocks[name], { L, l0, l2, tables } = B, base = L.offset + r * L.width;
    const key = `${name}:${r}`, c = cache?.rows[key];
    this.stats.rows++;
    if (c) { let same = true; for (let i = 0; i < L.width; i++) if (c.x[i] !== obs[base + i]) { same = false; break; } if (same) { this.stats.rowsReused++; return c; } }
    const h0 = this.h0, num = L.numeric, nn = num.length;
    for (let o = 0; o < l0.out; o++) {
      let s = l0.b[o]; const rw = o * l0.inn;
      for (let i = 0; i < nn; i++) s += l0.w[rw + i] * obs[base + num[i]];
      for (const t of tables) { let code = Math.round(obs[base + t.field]); if (!(code >= 0)) code = 0; if (code > t.n - 1) code = t.n - 1; s += t.T[code * l0.out + o]; }
      h0[o] = s < 0 ? 0 : s;
    }
    const y = new Float32Array(l2.out);
    for (let o = 0; o < l2.out; o++) { let s = l2.b[o]; const rw = o * l2.inn; for (let i = 0; i < l2.inn; i++) s += l2.w[rw + i] * h0[i]; y[o] = s < 0 ? 0 : s; }
    let P = null;
    const pt = this.ptr[name];
    if (pt) { // the row half of the pointer head's first layer, summed from 0 (canonical order)
      P = new Float64Array(pt.p0.out);
      for (let j = 0; j < pt.p0.out; j++) { let s = 0; const rw = j * pt.p0.inn; for (let i = 0; i < pt.o; i++) s += pt.p0.w[rw + i] * y[i]; P[j] = s; }
    }
    const out = { x: obs.slice(base, base + L.width), y, P };
    if (cache) cache.rows[key] = out;
    return out;
  }

  // features of one seat: global row, bonds MLP, pooled entities; rows kept for the pointer heads
  features(obs, cache) {
    const g = this.row(obs, 'global', 0, cache).y, [bo, bn] = this.m.bondsAt;
    const bh = new Float32Array(this.bonds0.out), bv = new Float32Array(this.bonds2.out);
    for (let o = 0; o < this.bonds0.out; o++) { let s = this.bonds0.b[o]; const rw = o * this.bonds0.inn; for (let i = 0; i < bn; i++) s += this.bonds0.w[rw + i] * obs[bo + i]; bh[o] = s < 0 ? 0 : s; }
    for (let o = 0; o < this.bonds2.out; o++) { let s = this.bonds2.b[o]; const rw = o * this.bonds2.inn; for (let i = 0; i < this.bonds2.inn; i++) s += this.bonds2.w[rw + i] * bh[i]; bv[o] = s < 0 ? 0 : s; }
    const cat = new Float32Array(this.trunk0.inn); cat.set(g, 0); let off = g.length; cat.set(bv, off); off += bv.length;
    const rows = {};
    for (const name of this.names) {
      if (name === 'global') continue;
      const B = this.blocks[name], d = B.l2.out, list = []; let n = 0;
      for (let k = 0; k < d; k++) { cat[off + k] = 0; cat[off + d + k] = -Infinity; }
      for (let r = 0; r < B.L.count; r++) {
        if (!(obs[B.L.offset + r * B.L.width] > 0.5)) { list.push(null); continue; }
        const e = this.row(obs, name, r, cache); list.push(e); n++;
        for (let k = 0; k < d; k++) { cat[off + k] += e.y[k]; if (e.y[k] > cat[off + d + k]) cat[off + d + k] = e.y[k]; }
      }
      if (!n) for (let k = 0; k < 2 * d; k++) cat[off + k] = 0; else for (let k = 0; k < d; k++) cat[off + k] /= n;
      rows[name] = list; off += 2 * d;
    }
    return { cat, rows };
  }

  // batch forward: seats = [{ obs, mask, cache }]; the trunk streams each weight row once for every seat
  forwardBatch(seats, { value = false } = {}) {
    const F = seats.map((s) => this.features(s.obs, s.cache)), S = seats.length;
    const t0 = this.trunk0, t2 = this.trunk2, h = F.map(() => new Float32Array(t0.out)), c = F.map(() => new Float32Array(t2.out));
    for (let o = 0; o < t0.out; o++) { const rw = o * t0.inn; for (let k = 0; k < S; k++) { let s = t0.b[o]; const x = F[k].cat; for (let i = 0; i < t0.inn; i++) s += t0.w[rw + i] * x[i]; h[k][o] = s < 0 ? 0 : s; } }
    for (let o = 0; o < t2.out; o++) { const rw = o * t2.inn; for (let k = 0; k < S; k++) { let s = t2.b[o]; const x = h[k]; for (let i = 0; i < t2.inn; i++) s += t2.w[rw + i] * x[i]; c[k][o] = s < 0 ? 0 : s; } }
    return seats.map((seat, k) => this.heads(c[k], F[k].rows, seat.mask, value));
  }
  forward(obs, mask, { value = false, cache = null } = {}) { return this.forwardBatch([{ obs, mask, cache }], { value })[0]; }

  heads(c, rows, mask, value) {
    const logits = new Float32Array(this.A).fill(-1e9), sc = this.scalarL;
    for (let a = 0; a < sc.out; a++) if (!mask || mask[a]) { let s = sc.b[a]; for (let i = 0; i < sc.inn; i++) s += sc.w[a * sc.inn + i] * c[i]; logits[a] = s; }
    let k = sc.out;
    for (const name of this.m.pointer) {
      const { p0, p2, o } = this.ptr[name], list = rows[name], cnt = this.blocks[name].L.count;
      let need = false; for (let r = 0; r < cnt; r++) if (!mask || mask[k + r]) { need = true; break; }
      let cc = null;
      if (need) { cc = new Float64Array(p0.out); for (let j = 0; j < p0.out; j++) { let s = p0.b[j]; const rw = j * p0.inn + o; for (let i = 0; i < c.length; i++) s += p0.w[rw + i] * c[i]; cc[j] = s; } }
      for (let r = 0; r < cnt; r++, k++) {
        if ((mask && !mask[k]) || !list[r]) continue;
        const P = list[r].P; let s = p2.b[0];
        for (let j = 0; j < p0.out; j++) { const v = cc[j] + P[j]; s += p2.w[j] * (v < 0 ? 0 : v); }
        logits[k] = s;
      }
    }
    let v = null;
    if (value) { const hv = new Float32Array(this.value0.out); for (let o2 = 0; o2 < this.value0.out; o2++) { let s = this.value0.b[o2]; const rw = o2 * this.value0.inn; for (let i = 0; i < c.length; i++) s += this.value0.w[rw + i] * c[i]; hv[o2] = s < 0 ? 0 : s; } let s = this.value2.b[0]; for (let j = 0; j < hv.length; j++) s += this.value2.w[j] * hv[j]; v = s; }
    return { logits, value: v };
  }

  /** The greedy action, without a forward when only one action is legal. */
  decide(obs, mask, cache = null) {
    let only = -1, n = 0; for (let a = 0; a < this.A; a++) if (mask[a]) { n++; only = a; }
    if (n === 1) return { action: only, skipped: true };
    const { logits } = this.forward(obs, mask, { cache }); let best = -1;
    for (let a = 0; a < this.A; a++) if (mask[a] && (best < 0 || logits[a] > logits[best])) best = a;
    return { action: best, skipped: false };
  }
}
