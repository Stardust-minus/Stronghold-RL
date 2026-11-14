// tools/rl/xplace.mjs — xplace (11-13): the learned layout policy (XNet, tools/rl/py/xtrain.py) in plain JS. Deploy
// path: features of the seat's placement problem (xfeat.mjs; no battle is built), an autoregressive greedy decode of a
// (tile, facing) per fielded piece in the featurizer's canonical order, then legacy's own plan application (tokens too).
import { readFileSync } from 'node:fs';
import { TILES } from './spec.mjs';
import { xPlaceProblemForTraining, applyPlanForPicker, LayoutPlan, layoutCandidatesForPicker, xPerturbForTraining } from '../../server/match/bot.js';
import { xFeaturize, xAssignOf, xCandFeatures, DIR_VEC } from './xfeat.mjs';
import { pickerRow } from './picker.mjs';
import { createPool } from './xpool.mjs'; // X9: o.pool (off by default)

const DIRS = ['UP', 'RIGHT', 'DOWN', 'LEFT'], R0 = 9, C0 = 3, NR = 4, NC = 7;
const idx = (r, c) => (r >= R0 && r < R0 + NR && c >= C0 && c < C0 + NC ? (r - R0) * NC + (c - C0) : TILES);
const Fr = [], Bk = [], N4 = [], RW = [];
for (let t = 0; t < TILES; t++) {
  const r = R0 + Math.floor(t / NC), c = C0 + (t % NC);
  Fr.push(DIRS.map((d) => idx(r + DIR_VEC[d][0], c + DIR_VEC[d][1]))); Bk.push(DIRS.map((d) => idx(r - DIR_VEC[d][0], c - DIR_VEC[d][1])));
  N4.push(DIRS.map((d) => idx(r + DIR_VEC[d][0], c + DIR_VEC[d][1])));
  const row = []; for (let cc = C0; cc < C0 + NC; cc++) if (cc !== c) row.push(idx(r, cc)); RW.push(row);
}

function linear(W, b, x, nIn, nOut, out = new Float32Array(nOut), add = false) {
  for (let o = 0; o < nOut; o++) { let s = b ? b[o] : 0; const w = o * nIn; for (let i = 0; i < nIn; i++) s += W[w + i] * x[i]; out[o] = add ? out[o] + s : s; }
  return out;
}
const relu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; };
function layerNorm(x, w, b) {
  let m = 0; for (let i = 0; i < x.length; i++) m += x[i]; m /= x.length;
  let v = 0; for (let i = 0; i < x.length; i++) v += (x[i] - m) ** 2; v /= x.length;
  const s = 1 / Math.sqrt(v + 1e-5), o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = (x[i] - m) * s * w[i] + b[i];
  return o;
}

/** Pre-LN self-attention block (prefix p) over the vectors xs, in place (xtrain.py: ln1 / qkv / proj / ln2 / ff). */
function attnBlock(T, p, xs, d, H, valid = null) {
  const n = xs.length, dh = d / H, act = valid || xs.map(() => true);
  const qkv = xs.map((x) => linear(T[p + 'qkv.weight'], T[p + 'qkv.bias'], layerNorm(x, T[p + 'ln1.weight'], T[p + 'ln1.bias']), d, 3 * d));
  const o = xs.map(() => new Float32Array(d));
  for (let h = 0; h < H; h++) for (let i = 0; i < n; i++) {
    const sc = new Float64Array(n); let mx = -Infinity;
    for (let j = 0; j < n; j++) { if (!act[j]) { sc[j] = -Infinity; continue; } let s = 0; for (let k = 0; k < dh; k++) s += qkv[i][h * dh + k] * qkv[j][d + h * dh + k]; sc[j] = s / Math.sqrt(dh); if (sc[j] > mx) mx = sc[j]; }
    let z = 0; for (let j = 0; j < n; j++) { sc[j] = act[j] ? Math.exp(sc[j] - mx) : 0; z += sc[j]; }
    for (let j = 0; j < n; j++) { if (!sc[j]) continue; const a = sc[j] / z; for (let k = 0; k < dh; k++) o[i][h * dh + k] += a * qkv[j][2 * d + h * dh + k]; }
  }
  for (let i = 0; i < n; i++) {
    const pr = linear(T[p + 'proj.weight'], T[p + 'proj.bias'], o[i], d, d); for (let k = 0; k < d; k++) xs[i][k] += pr[k];
    const ln = layerNorm(xs[i], T[p + 'ln2.weight'], T[p + 'ln2.bias']);
    const f = linear(T[p + 'ff.2.weight'], T[p + 'ff.2.bias'], relu(linear(T[p + 'ff.0.weight'], T[p + 'ff.0.bias'], ln, d, 2 * d)), 2 * d, d);
    for (let k = 0; k < d; k++) xs[i][k] += f[k];
  }
  return xs;
}

export class XNetJS {
  constructor(man) {
    this.cfg = man.cfg; this.meta = man.meta || {};
    const buf = Buffer.from(man.data, 'base64'), all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    this.T = {}; for (const [k, v] of Object.entries(man.tensors)) this.T[k] = all.subarray(v.offset, v.offset + v.shape.reduce((a, b) => a * b, 1));
    this.md5 = man.md5;
  }
  static load(path) { return new XNetJS(JSON.parse(readFileSync(path, 'utf8'))); }
  mlp(name, x, nIn, nH, nOut) {
    const T = this.T, h = relu(linear(T[`${name}.0.weight`], T[`${name}.0.bias`], x, nIn, nH));
    return linear(T[`${name}.2.weight`], T[`${name}.2.bias`], h, nH, nOut);
  }
  /** Greedy decode of features X: [[tile, dir] per ordered piece] (-1 when nothing is legal). */
  /** Encoder: piece embeddings (with self-attention), tile embeddings, the global context (as xtrain.py XNet.encode). */
  encode(X) {
    const { FP, FT, FG, V, d, de } = this.cfg, T = this.T, n = X.n;
    const hp = [], tmp = new Float32Array(FP + de);
    for (let i = 0; i < n; i++) {
      tmp.set(X.fp.subarray(i * FP, (i + 1) * FP)); const id = Math.min(V - 1, Math.max(0, X.ids[i]));
      tmp.set(T['emb.weight'].subarray(id * de, (id + 1) * de), FP);
      hp.push(this.mlp('pe', tmp, FP + de, d, d));
    }
    for (let L = 0; L < (this.cfg.attn || 0); L++) attnBlock(T, `att.${L}.`, hp, d, this.cfg.heads || 4);
    const ht = []; for (let t = 0; t < TILES; t++) ht.push(this.mlp('te', X.ft.subarray(t * FT, (t + 1) * FT), FT, d, d));
    const gin = new Float32Array(FG + 2 * d); gin.set(X.fg);
    for (let k = 0; k < d; k++) { let s = 0, m = -1e4; for (let i = 0; i < n; i++) { s += hp[i][k]; if (hp[i][k] > m) m = hp[i][k]; } gin[FG + k] = n ? s / n : 0; gin[FG + d + k] = n ? m : -1e4; }
    const g = this.mlp('ge', gin, FG + 2 * d, d, d);
    return { hp, ht, g, Gg: linear(T['G.weight'], null, g, d, d), Bt: ht.map((h) => linear(T['Bt.weight'], null, h, d, d)),
      Ah: hp.map((h) => linear(T['A.weight'], T['A.bias'], h, d, d)) };
  }
  /** Greedy decode of features X: [[tile, dir] per ordered piece] (-1 when nothing is legal); margin: keep the current
   * tile / facing unless the best logit beats it by more than margin. */
  /** Sampled decode (training-time proposals, expert iteration): softmax(logit / temp) per piece; rng() in [0, 1). */
  sample(X, temp, rng) { return this.decode(X, 0, { temp, rng }); }
  decode(X, margin = 0, smp = null) {
    const { FE, d } = this.cfg, T = this.T, n = X.n;
    const { hp, Gg, Bt, Ah: AhAll } = this.encode(X);
    const acc = [0, 1, 2, 3].map(() => Array.from({ length: TILES + 1 }, () => new Float32Array(d))); // R0..R3 (occupant at tile)
    const face = Array.from({ length: TILES + 1 }, () => new Float32Array(d)), back = Array.from({ length: TILES + 1 }, () => new Float32Array(d));
    const occupied = new Uint8Array(TILES), out = [], z = new Float32Array(d), h1 = new Float32Array(d), we = new Float32Array(d);
    const Cd = T.Cd, W1 = T['h1.weight'], b1 = T['h1.bias'], Wo = T['out.weight'], bo = T['out.bias'][0], WE = T['We.weight'];
    for (let i = 0; i < n; i++) {
      const Ah = AhAll[i];
      let best = -Infinity, bt = -1, bd = -1, curV = -Infinity; const opts = smp ? [] : null;
      const ct = X.cur ? X.cur[i * 2] : -1, cdir = X.cur ? X.cur[i * 2 + 1] : -1;
      for (let t = 0; t < TILES; t++) {
        if (occupied[t]) continue;
        // tile-level terms shared by the 4 facings
        const base = new Float32Array(d);
        for (let k = 0; k < d; k++) base[k] = Ah[k] + Bt[t][k] + Gg[k] + face[t][k] + back[t][k];
        for (const q of N4[t]) { const a = acc[2][q]; for (let k = 0; k < d; k++) base[k] += a[k]; }
        for (const q of RW[t]) { const a = acc[3][q]; for (let k = 0; k < d; k++) base[k] += a[k]; }
        for (let dd = 0; dd < 4; dd++) {
          if (!X.legal[(i * TILES + t) * 4 + dd]) continue;
          const fo = ((i * TILES + t) * 4 + dd) * FE;
          linear(WE, null, X.fe.subarray(fo, fo + FE), FE, d, we);
          const af = acc[0][Fr[t][dd]], ab = acc[1][Bk[t][dd]];
          for (let k = 0; k < d; k++) { const v = base[k] + Cd[dd * d + k] + we[k] + af[k] + ab[k]; z[k] = v > 0 ? v : 0; }
          linear(W1, b1, z, d, d, h1); let s = bo; for (let k = 0; k < d; k++) if (h1[k] > 0) s += Wo[k] * h1[k];
          if (s > best) { best = s; bt = t; bd = dd; }
          if (opts) opts.push([s, t, dd]);
          if (t === ct && dd === cdir) curV = s;
        }
      }
      if (margin > 0 && curV > -Infinity && best - curV <= margin) { bt = ct; bd = cdir; }
      if (opts && opts.length) { // sample instead of argmax
        let z = 0; const w = opts.map(([s]) => { const e = Math.exp((s - best) / smp.temp); z += e; return e; });
        let u = smp.rng() * z, k = 0; while (k < w.length - 1 && u > w[k]) { u -= w[k]; k++; }
        bt = opts[k][1]; bd = opts[k][2];
      }
      out.push([bt, bd]);
      if (bt < 0) continue;
      occupied[bt] = 1;
      const oin = new Float32Array(d + 4); oin.set(hp[i]); oin[d + bd] = 1;
      const occ = relu(linear(T['occ.weight'], T['occ.bias'], oin, d + 4, d));
      for (let r = 0; r < 4; r++) linear(T[`R.${r}.weight`], null, occ, d, d, acc[r][bt], true);
      linear(T['R.4.weight'], null, occ, d, d, face[Fr[bt][bd]], true);
      linear(T['R.5.weight'], null, occ, d, d, back[Bk[bt][bd]], true);
    }
    return out;
  }
}

/** Layout scorer Q(s, L) (xtrain.py QNet): every placed piece's embedding in the full layout, self-attention over them, pooling. */
export class QNetJS extends XNetJS {
  static load(path) { return new QNetJS(JSON.parse(readFileSync(path, 'utf8'))); }
  /** Q of each assignment ([[tile, dir] per ordered piece], -1 = unplaced) of features X, enc = this.encode(X). */
  score(X, enc, assigns, cfs = null) {
    const { FE, d } = this.cfg, T = this.T, n = X.n, H = this.cfg.heads || 4;
    const { hp, g, Gg, Bt, Ah } = enc, Cd = T.Cd, WE = T['We.weight'];
    const occC = new Map();
    const occOf = (i, dd) => {
      const k = i * 4 + dd; let v = occC.get(k); if (v) return v;
      const oin = new Float32Array(d + 4); oin.set(hp[i]); oin[d + dd] = 1;
      const occ = relu(linear(T['occ.weight'], T['occ.bias'], oin, d + 4, d));
      v = [0, 1, 2, 3, 4, 5].map((r) => linear(T[`R.${r}.weight`], null, occ, d, d)); occC.set(k, v); return v;
    };
    const out = [];
    for (const a of assigns) {
      const at = new Int16Array(TILES + 1).fill(-1), face = new Map(), back = new Map(), vi = [];
      for (let i = 0; i < n; i++) { const [t, dd] = a[i]; if (t < 0) continue; at[t] = i; vi.push(i); }
      const addTo = (mp, t, v) => { if (t >= TILES) return; let x = mp.get(t); if (!x) { x = new Float32Array(d); mp.set(t, x); } for (let k = 0; k < d; k++) x[k] += v[k]; };
      for (const i of vi) { const [t, dd] = a[i], r = occOf(i, dd); addTo(face, Fr[t][dd], r[4]); addTo(back, Bk[t][dd], r[5]); }
      const z = new Float32Array(d), we = new Float32Array(d), us = [];
      for (const i of vi) {
        const [t, dd] = a[i], fo = ((i * TILES + t) * 4 + dd) * FE;
        linear(WE, null, X.fe.subarray(fo, fo + FE), FE, d, we);
        for (let k = 0; k < d; k++) z[k] = Ah[i][k] + Bt[t][k] + Cd[dd * d + k] + we[k] + Gg[k];
        const add = (v) => { for (let k = 0; k < d; k++) z[k] += v[k]; };
        const pf = at[Fr[t][dd]]; if (Fr[t][dd] < TILES && pf >= 0) add(occOf(pf, a[pf][1])[0]);
        const pb = at[Bk[t][dd]]; if (Bk[t][dd] < TILES && pb >= 0) add(occOf(pb, a[pb][1])[1]);
        for (const q of N4[t]) { const j = q < TILES ? at[q] : -1; if (j >= 0) add(occOf(j, a[j][1])[2]); }
        for (const q of RW[t]) { const j = q < TILES ? at[q] : -1; if (j >= 0) add(occOf(j, a[j][1])[3]); }
        const f = face.get(t); if (f) add(f); const b = back.get(t); if (b) add(b);
        us.push(relu(linear(T['h1.weight'], T['h1.bias'], relu(Float32Array.from(z)), d, d)));
      }
      for (let L = 0; L < (this.cfg.qattn || 0); L++) attnBlock(T, `qatt.${L}.`, us, d, H);
      const CF = this.cfg.cf || 0, pin = new Float32Array((CF ? 4 : 3) * d);
      for (let k = 0; k < d; k++) { let sm = 0, mx = -1e4; for (const u of us) { sm += u[k]; if (u[k] > mx) mx = u[k]; } pin[k] = sm / 4; pin[d + k] = Math.max(-1e3, mx); pin[2 * d + k] = g[k]; }
      if (CF) {
        const cf = cfs[out.length], cf0 = cfs[0], cin = new Float32Array(2 * CF);
        for (let k = 0; k < CF; k++) { cin[k] = cf[k]; cin[CF + k] = cf[k] - cf0[k]; }
        pin.set(this.mlp('cfe', cin, 2 * CF, d, d), 3 * d);
      }
      out.push(this.mlp('qh', pin, pin.length, d, 1)[0]);
    }
    return out;
  }
}

function rngOf(str) { // mulberry32 seeded by an FNV hash (deploy: visible state only)
  let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  let a = h || 1;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** env xPlace: { model: path, rounds: 'all' | 'normal' | 'boss', minPieces } — applies the policy's layout to an RL seat. */
export class XPlacer {
  /** o: { model, rounds 'all' | 'normal' | 'boss', margin, minPieces, K (q: legacy candidates), pert (q: extra random legal
   * perturbations of the legacy candidates, seeded by the visible board), climb (q: Q hill-climb proposals; off) } */
  static fromOptions(o) {
    const p = new XPlacer(), man = JSON.parse(readFileSync(o.model, 'utf8'));
    p.q = man.cfg && man.cfg.kind === 'q'; p.net = p.q ? new QNetJS(man) : new XNetJS(man);
    p.rounds = o.rounds || 'all'; p.minPieces = o.minPieces || 1; p.margin = Number(o.margin) || 0;
    p.K = Number(o.K) || 12; p.pert = Number(o.pert) || 0; p.climb = Number(o.climb) || 0;
    p.gen = o.gen ? XNetJS.load(o.gen) : null; p.genN = Number(o.genN) || 0; p.genT = Number(o.genT) || 1; // generator proposals (q mode)
    p.ens = Number(o.ens) || 0; // > 0: Q z-score + ens x the env's boss picker head z-score over the picker's own K candidates
    // X9 pool { ...xpool options }: argmax Q over the deploy-legal pool (xpool.mjs: legacy K, generator, stacking templates, last round's
    // board refitted, one-step edits of Q's pick; lineup variants are not featurizable and stay off); no sims. Replaces K / pert / gen / ens.
    p.pool = o.pool && p.q ? createPool({ ...o.pool, lineup: 0, qnet: p.net }) : null;
    return p;
  }
  apply(m, ps, slot, env, S) {
    const boss = !m.wave;
    if (this.pool) { this._pkey = `${slot.seed}:${ps.playerId}`; if (this.rounds !== 'all' && (this.rounds === 'boss') !== boss) { this.pool.observe(m, ps, this._pkey); return; } }
    if (this.rounds !== 'all' && (this.rounds === 'boss') !== boss) return;
    const P = xPlaceProblemForTraining(m, ps); if (!P || !P.fight || P.pieces.length < this.minPieces) return;
    const seat = env._seat(slot, ps), X0 = xFeaturize(m, ps, P, { lp0: seat?.lp0 });
    const h = (a) => Float32Array.from(Float16Array.from(a)); // the training features are float16
    const X = { ...X0, fp: h(X0.fp), ft: h(X0.ft), fe: h(X0.fe), fg: h(X0.fg) };
    let plan;
    if (this.q) plan = this._pick(m, ps, P, X, S, env, slot);
    else {
      const assign = this.net.decode(X, this.margin); plan = new LayoutPlan();
      X.uids.forEach((u, i) => { const [t, dd] = assign[i]; if (t < 0) return; plan.set(u, `${R0 + Math.floor(t / NC)},${C0 + (t % NC)}`); plan.dirs.set(u, DIRS[dd]); });
    }
    S.decisions++; S.pieces += X.n;
    if (!plan) return;
    const a = xAssignOf(X, plan); let moved = 0;
    a.forEach(([t, dd], i) => { if (X.cur[i * 2] !== t || X.cur[i * 2 + 1] !== dd) moved++; });
    if (moved) { applyPlanForPicker(m, ps, P.chosen, plan); S.changed++; S.moved += moved; }
  }
  /** QNet: argmax Q over legacy's candidate layouts (+ optional perturbations / Q hill-climb); null keeps the current. */
  _pick(m, ps, P, X, S, env = null, slot = null) {
    if (this.pool) { // X9: the deploy-legal pool (observe after the build: 'prev' = the previous round's board)
      const R = this.pool.build(m, ps, P, X, this._pkey); this.pool.observe(m, ps, this._pkey);
      if (!R || R.cands.length < 2) return null;
      const plans = R.cands.filter((c) => c.li == null).map((c) => c.plan), Q = this.pool.score(m, ps, P, X, plans, this.net);
      let best = 0; for (let j = 1; j < Q.length; j++) if (Q[j] > Q[best]) best = j;
      S.cands = (S.cands || 0) + plans.length; S.poolSrc = S.poolSrc || {}; const sr = R.cands.filter((c) => c.li == null)[best].src; S.poolSrc[sr] = (S.poolSrc[sr] || 0) + 1;
      if (best === 0 || Q[best] - Q[0] <= this.margin) return null;
      return plans[best];
    }
    const PK = this.ens > 0 && env && env.picker ? env.picker : null;
    const SP = env && env.bspatial && (env.bspatial.rounds === 'boss' || m.round === m.gd.bossRound) ? env.bspatial : null; // (code-P9) the spatial boss picker
    const C = layoutCandidatesForPicker(m, ps, PK ? PK.K : SP ? SP.K : this.K); if (!C || !C.cands.length) return null;
    const sig = (pl) => [...pl.entries()].sort((x, y) => x[0] - y[0]).map(([u, k]) => `${u}@${k}:${pl.dirs?.get(u) ?? 'RIGHT'}`).join(' ');
    const plans = C.cands.slice(), seen = new Set(plans.map(sig));
    const rng = rngOf(`xq:${m.round}:${ps.seat}:${sig(plans[0])}`);
    for (let g = 0; this.pert && plans.length < C.cands.length + this.pert && g < this.pert * 20; g++) {
      const q = xPerturbForTraining(m, ps, P.chosen, plans[Math.floor(rng() * plans.length)], rng), sg = sig(q);
      if (!seen.has(sg)) { seen.add(sg); plans.push(q); }
    }
    if (this.gen && this.genN > 0) for (let g = 0; g < this.genN; g++) { // the generator's sampled layouts (no sims)
      const asg = this.gen.sample(X, this.genT, rng), q = new LayoutPlan();
      X.uids.forEach((u, i) => { const [t, dd] = asg[i]; if (t < 0) return; q.set(u, `${R0 + Math.floor(t / NC)},${C0 + (t % NC)}`); q.dirs.set(u, DIRS[dd]); });
      const sg = sig(q); if (!seen.has(sg)) { seen.add(sg); plans.push(q); }
    }
    const h16 = (a) => Float32Array.from(Float16Array.from(a)); // the training candidate features are float16
    const cfOf = (pl) => (this.net.cfg.cf ? h16(xCandFeatures(m, ps, P.chosen, pl, plans[0])) : null);
    const cfs = plans.map(cfOf);
    const enc = this.net.encode(X), Q = this.net.score(X, enc, plans.map((pl) => xAssignOf(X, pl)), cfs);
    if (PK && C.cands.length > 1) { // ensemble with the boss picker head (its rows / trunk c, as env._bossPick)
      const s = env._seat(slot, ps), ob = new Float32Array(env.encoder.dim), mk = new Uint8Array(env.encoder.A);
      env.encoder.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: env.rules, cache: {} }, ob, mk);
      PK.trunk.forward(ob, mk, { value: false });
      const c = Float32Array.from(PK.trunk.c), rows = C.cands.map((_, j) => pickerRow(C.feats[j], C.variants[j], j === 0, C.feats[0]));
      const lg = PK.head.score(c, rows), z = (v) => { const n = v.length, mu = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / n) || 1; return v.map((x) => (x - mu) / sd); };
      const zq = z(Q.slice(0, C.cands.length)), zp = z(Array.from(lg));
      for (let j = 0; j < C.cands.length; j++) Q[j] = zq[j] + this.ens * zp[j];
      for (let j = C.cands.length; j < Q.length; j++) Q[j] = -Infinity;
    }
    if (SP && C.cands.length > 1) { // (code-P9) + w x z(the spatial boss picker) over the same K candidates (z(Q) first if there is no head ensemble)
      const zf = (v) => { const n = v.length, mu = v.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / n) || 1; return v.map((x) => (x - mu) / sd); };
      const n = C.cands.length, zs = zf(env._bossSpatialScores(m, ps, slot, C)), b = PK ? Array.from(Q.slice(0, n)) : zf(Array.from(Q.slice(0, n)));
      for (let j = 0; j < n; j++) Q[j] = b[j] + SP.w * zs[j];
      for (let j = n; j < Q.length; j++) Q[j] = -Infinity;
      S.sp = (S.sp || 0) + 1;
    }
    let best = 0; for (let j = 1; j < Q.length; j++) if (Q[j] > Q[best]) best = j;
    for (let it = 0; it < this.climb; it++) { // Q hill-climb (model-only refinement; off by default)
      const q = xPerturbForTraining(m, ps, P.chosen, plans[best], rng), sg = sig(q); if (seen.has(sg)) continue; seen.add(sg);
      const v = this.net.score(X, enc, [xAssignOf(X, plans[0]), xAssignOf(X, q)], [cfs[0], cfOf(q)])[1]; plans.push(q); Q.push(v); if (v > Q[best]) best = plans.length - 1;
    }
    S.cands = (S.cands || 0) + plans.length;
    if (best === 0 || Q[best] - Q[0] <= this.margin) return null;
    return plans[best];
  }
}
