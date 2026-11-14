// tools/rl/xteach.mjs — xplace (code-X7, 11-13): training-time teacher search over the deploy-legal pool (xpool.mjs).
// Reached through env xLabel { teach: 1, ... } (xsearch.mjs dispatches; without `teach` nothing here runs).
// Per decision (boss rounds by default) of an RL seat: build the pool, run each teacher at each sim budget, evaluate every
// teacher's pick (plus cur, the scorer's picks and the K8 search best) on fresh honest seeds, record, play `play`.
// Leaf value = boss utility vs the current layout over `seeds` salted rehearsal seeds (per match, never the live seed;
// the learned Q is a prior only, never a value). Teachers (equal sim budget; a sim = one rehearsal):
//   hc   — the top half of the leaf budget on the pool by prior, then first-improvement one-step edits of the incumbent
//   cem  — population = the pool's top by prior, then per-piece (tile, dir) categoricals refit to the elite (25 %),
//          smoothed 0.7 / 0.3 with the previous one; samples are legal and collision-free
//   puct — pool members as root children (prior softmax(z(Q))), one-step edits as children with progressive widening
//          (ceil(1.5 sqrt N)), PUCT c 1.5 on values / 10; the pick is the best evaluated leaf
// o: { out, pool {..createPool opts}, q (prior QNet json), budgets [64, 256], teachers ['hc', 'cem', 'puct'], seeds 2,
//      hseeds 3, rounds 'boss', play ('cur' | '<teacher><budget>' | 'best'), storeObs, cfg, maxMs }
import { appendFileSync, writeFileSync, openSync, writeSync, closeSync, existsSync, statSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { xPlaceProblemForTraining, xRehearseForTraining, applyPlanForPicker, xApplyLineupForTraining } from '../../server/match/bot.js';
import { xFeaturize, xAssignOf, xCandFeatures } from './xfeat.mjs';
import { QNetJS } from './xplace.mjs';
import { TILES, DIRS, tileRC } from './spec.mjs';
import { createPool, layoutOps, sigOf, rngOf, planOfAssign } from './xpool.mjs';
import { bossPoolHp } from '../../server/match/finalAssault.js'; // X10: the FA pool max HP for the win objective

const UCLIP = 30;
const mean = (xs) => { const ok = xs.filter(Boolean); if (!ok.length) return null; const o = { leaks: 0, boss: 0, kills: 0, layers: 0 }; for (const x of ok) for (const k in o) o[k] += x[k] / ok.length; return o; };
const pack = (s) => (s ? [s.leaks, +s.boss.toFixed(1), s.kills, s.layers] : null);
function util(boss, s, s0) { // = xtrain.util (boss: leaks x10, leader damage share x10, kills x1e-3; normal: rule D)
  if (!s || !s0) return null;
  const v = boss ? -10 * (s.leaks - s0.leaks) + 10 * (s.boss - s0.boss) / Math.max(s0.boss, 1e3) + 1e-3 * (s.kills - s0.kills)
    : (s.layers - s0.layers) + 0.01 * (s.kills - s0.kills) - 20 * Math.max(0, s.leaks - s0.leaks - 1e-9);
  return Math.max(-UCLIP, Math.min(UCLIP, v));
}

export function createTeachHook(env, o = {}) {
  const OUT = o.out; if (!OUT) throw new Error('xteach: out required');
  const NS = Number(o.seeds ?? 2), HS = Number(o.hseeds ?? 3), BUD = (o.budgets || [64, 256]).map(Number), TEACH = o.teachers || ['hc', 'cem', 'puct'];
  const ROUNDS = o.rounds || 'boss', PLAY = o.play || 'cur', STORE_OBS = !!o.storeObs, cfg = o.cfg || null;
  const pool = createPool(o.pool || {}), PRIOR = o.q ? QNetJS.load(o.q) : null;
  // X10 o.win { solo: [a, b], coop4h1: [a, b], other }: the teachers' objective and every reported utility become the picker's win utility,
  // 100 x mean over seeds of sigmoid(a + b x share) minus the same for the current layout; share = rehearsal leader damage / the FA pool
  // max HP (+ `other`, the teammates' expected share in co-op; labels are re-scored later with their real damage); round 14 only.
  const WIN = o.win || null;
  if (existsSync(OUT + '.jsonl') && statSync(OUT + '.jsonl').size > 0 && !o.overwrite) throw new Error(`${OUT}.jsonl exists (overwrite)`);
  writeFileSync(OUT + '.jsonl', ''); const binFd = openSync(OUT + '.bin', 'w'); let binOff = 0;
  const S = { decisions: 0, sims: 0, ms: 0, skipped: 0, errors: 0 };
  function writeBin(arrs) {
    const off = binOff, parts = [];
    for (const a of arrs) { const f = a instanceof Float32Array ? Float16Array.from(a) : a; parts.push(Buffer.from(f.buffer, f.byteOffset, f.byteLength)); }
    const buf = Buffer.concat(parts); writeSync(binFd, buf); binOff += buf.length; return [off, buf.length];
  }

  const hook = (m, ps, slot, envIn = null) => {
    const E = env || envIn;
    if (!slot.rl.has(ps.playerId) || slot.humans.has(ps.playerId)) return;
    const key = `${slot.seed}:${ps.playerId}`;
    const boss = !m.wave;
    if (ROUNDS !== 'all' && (ROUNDS === 'boss') !== boss) { pool.observe(m, ps, key); return; }
    if (WIN && m.round !== m.gd.bossRound) { pool.observe(m, ps, key); return; } // X10: round 14 (the FA) only
    const t0 = performance.now();
    const P = xPlaceProblemForTraining(m, ps); if (!P || !P.fight) { S.skipped++; pool.observe(m, ps, key); return; }
    const seat = E._seat(slot, ps), X0 = xFeaturize(m, ps, P, { lp0: seat?.lp0 });
    const h = (a) => Float32Array.from(Float16Array.from(a)), X = { ...X0, fp: h(X0.fp), ft: h(X0.ft), fe: h(X0.fe), fg: h(X0.fg) };
    const nb = m.newBattle; let poolSims = 0; m.newBattle = function (...x) { poolSims++; return nb.apply(this, x); }; // deploy-legal check: the pool builds no battle
    let R; try { R = pool.build(m, ps, P, X, key); } finally { m.newBattle = nb; }
    pool.observe(m, ps, key);
    if (!R || R.cands.length < 2) { S.skipped++; return; }
    const cands = R.cands.map((c) => ({ ...c, sig: sigOf(c.plan), s: [], hs: [] })), bySig = new Map(cands.map((c, j) => [c.sig, j]));
    const ops = layoutOps(P, X), salt = `${slot.seed}`;
    const addCand = (plan, src) => { const sg = sigOf(plan); if (bySig.has(sg)) return bySig.get(sg); cands.push({ plan, src, sig: sg, s: [], hs: [] }); bySig.set(sg, cands.length - 1); return cands.length - 1; };
    const chosenOf = (c) => (c.li != null ? R.lineups[c.li] : R.chosen);
    let nsim = 0;
    const sim = (j, seedIx, field = 's') => { // cached rehearsal of candidate j on seed index seedIx
      const c = cands[j]; if (c[field][seedIx] !== undefined) return c[field][seedIx];
      const sx = field === 's' ? seedIx : 100 + seedIx;
      const r = xRehearseForTraining(m, ps, chosenOf(c), [c.plan], [sx], salt); nsim++;
      c[field][seedIx] = r && r[0] ? r[0][0] : null; return c[field][seedIx];
    };
    for (let k = 0; k < NS; k++) sim(0, k);
    if (!cands[0].s.slice(0, NS).every(Boolean)) { S.skipped++; return; }
    const m0 = mean(cands[0].s.slice(0, NS));
    let wv = null, wmeta = null;
    if (WIN) {
      const alive = typeof m.alivePlayers === 'function' ? m.alivePlayers().length : 1, kind = String(cfg || slot.cfg?.name || '').startsWith('solo') ? 'solo' : 'coop4h1';
      const hp = bossPoolHp(m.gd, m.bossId, alive, { hpMul: typeof m.enemyHpMulAt === 'function' ? m.enemyHpMulAt() : (m.enemyHpMul ?? 1) });
      const [wa, wb] = WIN[kind], other = kind === 'solo' ? 0 : Number(WIN.other ?? 0.635);
      wmeta = { maxHp: hp, kind, other, alive };
      wv = (ss) => { const ok = ss.filter(Boolean); if (!ok.length) return null; return ok.reduce((t, x) => t + 1 / (1 + Math.exp(-(wa + wb * (x.boss / hp + other)))), 0) / ok.length; };
    }
    const w0 = WIN ? wv(cands[0].s.slice(0, NS)) : 0;
    const leafV = (j) => { for (let k = 0; k < NS; k++) sim(j, k); if (WIN) { const v = wv(cands[j].s.slice(0, NS)); return v == null ? null : 100 * (v - w0); } return util(boss, mean(cands[j].s.slice(0, NS)), m0); };
    // priors: the scorer's Q over the featurizable candidates (lineup variants last)
    const qOf = (idx) => {
      if (!PRIOR) return idx.map(() => 0);
      const ok = idx.filter((j) => cands[j].li == null), out = new Map();
      if (ok.length) { const sc = pool.score(m, ps, P, X, [cands[0].plan, ...ok.map((j) => cands[j].plan)], PRIOR); ok.forEach((j, q) => out.set(j, sc[q + 1])); }
      const lo = Math.min(...[...out.values(), 0]) - 1; return idx.map((j) => (out.has(j) ? out.get(j) : lo));
    };
    const pool0 = cands.map((_, j) => j), q0 = qOf(pool0);
    const order = pool0.slice().sort((a, b) => q0[b] - q0[a]);
    const rng = rngOf(`xteach:${salt}:${m.round}:${ps.seat}`);
    const editOf = (j) => { if (cands[j].li != null) return null; const all = ops.edits(cands[j].plan); return all.length ? all[Math.floor(rng() * all.length)] : null; };

    const runTeacher = (name, budget) => {
      const T = { seen: new Map(), sims: NS }; // cur's own seeds are paid by every teacher
      const L = Math.floor(budget / NS) - 1; let best = 0, bv = 0; T.seen.set(0, 0);
      const ev = (j) => { if (T.seen.has(j)) return T.seen.get(j); if (T.seen.size - 1 >= L) return null; const v = leafV(j); T.seen.set(j, v); T.sims += NS; if (v != null && v > bv + 1e-9) { bv = v; best = j; } return v; };
      const left = () => L - (T.seen.size - 1);
      if (name === 'hc') {
        for (const j of order.slice(0, Math.max(1, Math.floor(L / 2)) + 1)) { if (j === 0) continue; if (ev(j) === null) break; }
        for (let g = 0; left() > 0 && g < L * 20; g++) { const q = editOf(best); if (!q) break; const j = addCand(q, 'hc'); if (!T.seen.has(j)) ev(j); }
        for (const j of order) { if (left() <= 0) break; if (!T.seen.has(j)) ev(j); } // a lineup best has no edits: the rest of the pool
      } else if (name === 'cem') {
        const n0 = Math.max(4, Math.floor(L / 4)), uids = X.uids, nP = uids.length;
        for (const j of order.slice(0, n0 + 1)) { if (j !== 0) ev(j); }
        let dist = null; // per piece: Float64Array(TILES * 4)
        const fit = () => {
          const evd = [...T.seen.entries()].filter(([j, v]) => v != null && cands[j].li == null).sort((a, b) => b[1] - a[1]);
          const el = evd.slice(0, Math.max(2, Math.ceil(evd.length / 4)));
          const nd = uids.map(() => new Float64Array(TILES * 4));
          for (const [j] of el) xAssignOf(X, cands[j].plan).forEach(([t, d], i) => { if (t >= 0) nd[i][t * 4 + d] += 1 / el.length; });
          dist = dist ? nd.map((a, i) => a.map((x, k) => 0.7 * x + 0.3 * dist[i][k])) : nd;
        };
        for (let it = 0; left() > 0 && it < 50; it++) {
          fit(); let made = 0;
          for (let g = 0; made < n0 && left() > 0 && g < n0 * 10; g++) {
            const occ = new Uint8Array(TILES), asg = uids.map(() => [-1, -1]), perm = uids.map((_, i) => i).sort(() => rng() - 0.5); let ok = true;
            for (const i of perm) {
              let z = 0; const w = new Float64Array(TILES * 4);
              for (let t = 0; t < TILES; t++) { if (occ[t]) continue; for (let d = 0; d < 4; d++) if (X.legal[(i * TILES + t) * 4 + d]) { w[t * 4 + d] = dist[i][t * 4 + d] + 0.02 / (TILES * 4); z += w[t * 4 + d]; } }
              if (z <= 0) { ok = false; break; }
              let u = rng() * z, k = 0; while (k < w.length - 1 && (w[k] === 0 || u > w[k])) { u -= w[k]; k++; }
              asg[i] = [Math.floor(k / 4), k % 4]; occ[Math.floor(k / 4)] = 1;
            }
            if (!ok) continue;
            const j = addCand(planOfAssign(X, asg), 'cem'); if (T.seen.has(j)) continue; ev(j); made++;
          }
          if (!made) break;
        }
      } else if (name === 'puct') {
        const N = new Map(), Wv = new Map(), kids = new Map(), Pr = new Map(); const C = 1.5;
        const z = (v) => { const mu = v.reduce((a, b) => a + b, 0) / v.length, sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) || 1; return v.map((x) => (x - mu) / sd); };
        const zq = z(q0), ex = zq.map((x) => Math.exp(x)), Z = ex.reduce((a, b) => a + b, 0);
        kids.set('root', pool0.filter((j) => j !== 0)); pool0.forEach((j) => Pr.set(j, ex[j] / Z));
        const val = (j) => (N.get(j) ? Wv.get(j) / N.get(j) : 0);
        for (let it = 0; left() > 0 && it < L * 10; it++) {
          let node = 'root', path = [], Np = it + 1;
          for (;;) {
            const ch = kids.get(node) || [];
            const allow = node === 'root' ? ch.length : Math.ceil(1.5 * Math.sqrt(N.get(node) || 0));
            if (node !== 'root' && ch.length < allow) { const q = editOf(node); if (q) { const j = addCand(q, 'puct'); if (!Pr.has(j)) Pr.set(j, 1 / allow); if (!ch.includes(j) && j !== node) ch.push(j); kids.set(node, ch); } }
            if (!ch.length) break;
            let bj = -1, bs = -Infinity;
            for (const j of ch) { const s = val(j) + C * (Pr.get(j) || 0.05) * Math.sqrt(Np) / (1 + (N.get(j) || 0)); if (s > bs) { bs = s; bj = j; } }
            if (path.includes(bj)) break; // edits can lead back to an ancestor
            path.push(bj); Np = (N.get(bj) || 0) + 1; node = bj;
            if (!T.seen.has(bj)) break;
          }
          if (!path.length) break;
          const leaf = path[path.length - 1]; let v = T.seen.has(leaf) ? T.seen.get(leaf) : ev(leaf);
          if (v === null && !T.seen.has(leaf)) break;
          v = (v ?? -UCLIP) / 10;
          for (const j of path) { N.set(j, (N.get(j) || 0) + 1); Wv.set(j, (Wv.get(j) || 0) + v); }
        }
      }
      return { pick: best, v: bv, sims: T.sims, nEval: T.seen.size };
    };

    const teach = {};
    for (const b of BUD) for (const t of TEACH) { try { teach[`${t}${b}`] = runTeacher(t, b); } catch (e) { S.errors++; teach[`${t}${b}`] = { err: String(e && e.message).slice(0, 120) }; } }
    // reference picks: the scorer's argmax over the pool / over K8 (deploy-legal), the K8 best by the leaf seeds (ceiling)
    const K8 = pool0.filter((j) => cands[j].src === 'cur' || cands[j].src === 'leg').slice(0, 8);
    if (PRIOR) {
      const feat = pool0.filter((j) => cands[j].li == null); let b = feat[0]; for (const j of feat) if (q0[j] > q0[b]) b = j;
      let b8 = 0; for (const j of K8) if (q0[j] > q0[b8]) b8 = j;
      teach.qpool = { pick: b }; teach.qK8 = { pick: b8 };
    }
    for (const src of o.always || []) { const j = pool0.find((x) => cands[x].src === src); if (j != null) teach[src] = { pick: j, v: leafV(j) }; } // X9: always scored (e.g. 'model')
    let o8 = 0, o8v = 0; for (const j of K8) { const v = leafV(j); if (v != null && v > o8v + 1e-9) { o8v = v; o8 = j; } } teach.oracleK8 = { pick: o8, v: o8v };
    let ob = 0, obv = 0; cands.forEach((c, j) => { if (c.s.slice(0, NS).every((x) => x !== undefined && x)) { const v = WIN ? 100 * (wv(c.s.slice(0, NS)) - w0) : util(boss, mean(c.s.slice(0, NS)), m0); if (v > obv + 1e-9) { obv = v; ob = j; } } }); teach.searchBest = { pick: ob, v: obv };
    // honest seeds for every distinct pick (and cur)
    const picks = [...new Set([0, ...Object.values(teach).filter((x) => x && x.pick != null).map((x) => x.pick)])];
    for (const j of picks) for (let k = 0; k < HS; k++) sim(j, k, 'hs');
    const h0 = mean(cands[0].hs.slice(0, HS));
    const hw0 = WIN && HS ? wv(cands[0].hs.slice(0, HS)) : null;
    for (const x of Object.values(teach)) if (x && x.pick != null) { x.h = WIN ? (HS ? 100 * (wv(cands[x.pick].hs.slice(0, HS)) - hw0) : null) : util(boss, mean(cands[x.pick].hs.slice(0, HS)), h0); x.src = cands[x.pick].src; }
    // record (xtrain-compatible: featurizable candidates in `cands`, lineup variants in `lcands`)
    const fi = cands.map((c, j) => j).filter((j) => cands[j].li == null), lj = cands.map((c, j) => j).filter((j) => cands[j].li != null);
    const remap = new Map(fi.map((j, q) => [j, q]));
    const tch = Object.fromEntries(Object.entries(teach).map(([k, x]) => [k, { ...x, pick: x.pick == null ? null : (remap.has(x.pick) ? remap.get(x.pick) : -1 - lj.indexOf(x.pick)) }]));
    const play = PLAY === 'cur' ? 0 : PLAY === 'best' ? ob : (teach[PLAY]?.pick ?? 0);
    const [off, len] = writeBin([X0.fp, X0.ft, X0.fe, X0.fg, X0.ids, X0.legal, X0.cur]);
    const pk = remap.has(ob) ? remap.get(ob) : 0;
    const rec = { salt: 1, nkey: 'd', teach: 1, seed: slot.seed, cfg: cfg || slot.cfg?.name, round: m.round, pid: ps.playerId, boss, n: X0.n, v: X0.version, vh: X0.vocabHash,
      dims: X0.dims, bin: [off, len], pick: pk, val: [0, pk], NS, HS, poolN: R.cands.length, tch,
      cands: fi.map((j) => ({ a: xAssignOf(X0, cands[j].plan), src: cands[j].src, s: cands[j].s.slice(0, NS).map(pack), hs: cands[j].hs.length ? cands[j].hs.map(pack) : undefined,
        f: Array.from(xCandFeatures(m, ps, R.chosen, cands[j].plan, cands[0].plan), (x) => +x.toFixed(4)) })),
      lcands: lj.map((j) => ({ li: cands[j].li, src: cands[j].src, s: cands[j].s.slice(0, NS).map(pack), hs: cands[j].hs.length ? cands[j].hs.map(pack) : undefined })),
      lineups: R.lineups.map((arr) => arr.map((p) => p.id)), poolSims, ...(wmeta ? { win: wmeta } : {}), nsim, ms: Math.round(performance.now() - t0) };
    appendFileSync(OUT + '.jsonl', JSON.stringify(rec) + '\n');
    if (play !== 0) {
      const c = cands[play];
      if (c.li != null) xApplyLineupForTraining(m, ps, R.lineups[c.li], c.plan); else applyPlanForPicker(m, ps, R.chosen, c.plan);
    }
    S.decisions++; S.sims += nsim; S.ms += performance.now() - t0;
  };
  return { hook, S, close() { closeSync(binFd); } };
}
