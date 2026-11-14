// tools/rl/xpool.mjs — xplace (code-X7, 11-13): deploy-legal candidate pool for one seat's placement decision.
// No sims and no battle: only what the seat sees (its board, bench, the round's enemy preview routes, its own previous
// board). Training-time teachers (tools/rl/xteach.mjs) search over the pool; a scorer trained on it stays deploy-legal.
//
//   const pool = createPool({ gen, genN, tpl: 1, prev: 1, edits: 6, q, lineup: 4 });
//   pool.observe(m, ps, key)              // every round (any prep): remembers the seat's board for 'prev'
//   const R = pool.build(m, ps, P, X, key) // -> { cands: [{ plan, src, li }], lineups, stats }
//
// Sources (src): cur (the seat's current layout = legacy's), leg (legacy's other variants, K - 1), model (the generator's
// greedy decode), gen (its samples), tpl:<name> (stacking templates per positional bond), prev (last round's board refitted
// to this round's pieces), edit (one-step edits — move / swap / turn — of the base: the QNet's pick when `q` is set, else
// cur), lineup (bench <-> board swaps placed by legacy's variants; li = index into R.lineups, not featurizable by X).
// Options (all off unless given): K 8, gen (XNet json path), genN 0, genT 1, tpl 0, prev 0, edits 0, q (QNet json path),
// lineup 0 (lineups), lineupV 2 (layouts each), max 48. Dedupe by layout signature; the order above is the priority.
import { readFileSync } from 'node:fs';
import { layoutCandidatesForPicker, xCurrentPlanForTraining, LayoutPlan, rangeTiles, xLineupProblemForTraining,
  xPlansForLineupForTraining } from '../../server/match/bot.js';
import { TILES, DIRS, tileIndex, tileRC } from './spec.mjs';
import { xAssignOf, xCandFeatures } from './xfeat.mjs';
import { XNetJS, QNetJS } from './xplace.mjs';

export function rngOf(str) { // mulberry32 seeded by an FNV hash (deploy: visible state only)
  let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  let a = h || 1;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const sigOf = (plan) => [...plan.entries()].sort((a, b) => a[0] - b[0]).map(([u, k]) => `${u}@${k}:${plan.dirs?.get(u) ?? 'RIGHT'}`).join(' ');
export const clonePlan = (plan) => { const q = new LayoutPlan([...plan.entries()]); for (const [u, d] of plan.dirs) q.dirs.set(u, d); return q; };
const keyOfT = (t) => { const [r, c] = tileRC(t); return `${r},${c}`; };
const tOfKey = (k) => { const [r, c] = k.split(',').map(Number); return tileIndex(r, c); };
export const planOfAssign = (X, asg) => { const q = new LayoutPlan(); X.uids.forEach((u, i) => { const [t, d] = asg[i]; if (t < 0) return; q.set(u, keyOfT(t)); q.dirs.set(u, DIRS[d]); }); return q; };

/** Layout helpers over the featurized problem (X.legal: piece x tile x dir). */
export function layoutOps(P, X) {
  const ix = new Map(X.uids.map((u, i) => [u, i]));
  const legal = (u, k, d) => { const i = ix.get(u), t = k == null ? -1 : tOfKey(k); return i != null && t >= 0 && t < TILES && !!X.legal[(i * TILES + t) * 4 + DIRS.indexOf(d)]; };
  const firstDir = (u, k, pref) => (legal(u, k, pref) ? pref : DIRS.find((d) => legal(u, k, d)) || null);
  const occ = (plan) => { const o = new Map(); for (const [u, k] of plan) o.set(k, u); return o; };
  /** u to key (dir), the occupant (if any) to u's old tile; null when illegal. */
  const place = (plan, u, k, d) => {
    if (!legal(u, k, d)) return null;
    const q = clonePlan(plan), k0 = q.get(u), d0 = q.dirs.get(u) ?? 'RIGHT', v = occ(q).get(k);
    if (v != null && v !== u) {
      if (k0 == null) return null;
      const dv = firstDir(v, k0, q.dirs.get(v) ?? 'RIGHT'); if (!dv) return null;
      q.set(v, k0); q.dirs.set(v, dv);
    }
    q.set(u, k); q.dirs.set(u, d); void d0; return q;
  };
  /** Every one-step edit of plan: turns, moves to free legal tiles, swaps. */
  const edits = (plan) => {
    const out = [], o = occ(plan), us = [...plan.keys()];
    for (const u of us) {
      const k = plan.get(u), d0 = plan.dirs.get(u) ?? 'RIGHT';
      for (const d of DIRS) if (d !== d0 && legal(u, k, d)) { const q = clonePlan(plan); q.dirs.set(u, d); out.push(q); }
      for (let t = 0; t < TILES; t++) {
        const kk = keyOfT(t); if (o.has(kk)) continue;
        const d = firstDir(u, kk, d0); if (d) { const q = clonePlan(plan); q.set(u, kk); q.dirs.set(u, d); out.push(q); }
      }
    }
    for (let a = 0; a < us.length; a++) for (let b = a + 1; b < us.length; b++) {
      const q = place(plan, us[a], plan.get(us[b]), firstDir(us[a], plan.get(us[b]), plan.dirs.get(us[a]) ?? 'RIGHT') || 'RIGHT');
      if (q) out.push(q);
    }
    return out;
  };
  return { ix, legal, firstDir, occ, place, edits };
}

/** Route weight per own-board tile (ground and flying routes; weight = the route's spawn count n). */
function routeWeights(P) {
  const g = new Map(), f = new Map();
  for (const rt of (P.model && P.model.routes) || []) for (const k of rt.tiles || []) { const mp = rt.fly ? f : g; mp.set(k, (mp.get(k) || 0) + (rt.n || 1)); }
  return { g, f };
}

/** Stacking templates per positional bond (the stacking catalogue): each returns a plan or null. */
function templates(P, X, base, ops) {
  const out = [], W = routeWeights(P), byUid = new Map(P.pieces.map((p) => [p.uid, p]));
  const has = (p, b) => (p.rec0?.bonds || []).includes(b), nm = (p) => p.rec0?.name || '';
  const cov = (p, k, d) => { const [r, c] = k.split(',').map(Number); let s = 0; for (const x of rangeTiles(p.rec, r, c, d)) s += (W.g.get(x) || 0) + (p.air ? (W.f.get(x) || 0) : 0); return s; };
  const front = (k, d) => { const [r, c] = k.split(',').map(Number); return rangeTiles({ rangeGrid: [[0, 1]] }, r, c, d)[0]; };
  const best = (p, plan, tiles, score) => { let bk = null, bd = null, bv = -Infinity; for (const k of tiles) for (const d of DIRS) { if (!ops.legal(p.uid, k, d)) continue; const v = score(k, d); if (v > bv) { bv = v; bk = k; bd = d; } } return bk ? [bk, bd] : null; };
  const allTiles = Array.from({ length: TILES }, (_, t) => keyOfT(t));
  const push = (name, q) => { if (q) out.push({ plan: q, src: `tpl:${name}` }); };
  const ps_ = P.pieces;
  // 阿戈尔: devour order = members left to right, each devours the operator in front of it -> feed every member a non-member
  const eg = ps_.filter((p) => has(p, 'egirShip'));
  if (eg.length) {
    let q = base;
    for (const p of eg.slice().sort((a, b) => (q.get(a.uid) || '').localeCompare(q.get(b.uid) || ''))) {
      const k = q.get(p.uid); if (!k) continue;
      const o = ops.occ(q), food = (kk) => { const v = o.get(kk); return v != null && !has(byUid.get(v) || {}, 'egirShip'); };
      if (food(front(k, q.dirs.get(p.uid) ?? 'RIGHT'))) continue;
      const d = DIRS.find((dd) => ops.legal(p.uid, k, dd) && food(front(k, dd)));
      if (d) { q = clonePlan(q); q.dirs.set(p.uid, d); continue; }
      // move the weakest non-member onto the tile in front (legacy facing kept)
      const fk = front(k, q.dirs.get(p.uid) ?? 'RIGHT');
      const nonm = ps_.filter((x) => !has(x, 'egirShip') && q.get(x.uid)).sort((a, b) => (a.dps || 0) - (b.dps || 0));
      for (const v of nonm) { const dv = ops.firstDir(v.uid, fk, q.dirs.get(v.uid) ?? 'RIGHT'); const q2 = dv && ops.place(q, v.uid, fk, dv); if (q2) { q = q2; break; } }
    }
    if (q !== base) push('egir', q);
  }
  // 突袭 (and 卡西米尔's redeploy engine): members idle with no enemy in range jump -> start where their range sees the least route
  const rd = ps_.filter((p) => has(p, 'raidShip'));
  if (rd.length) {
    let q = base;
    for (const p of rd) { const b = best(p, q, allTiles, (k, d) => -cov(p, k, d) - 0.01 * Math.abs(Number(k.split(',')[1]) - 3)); const q2 = b && ops.place(q, p.uid, b[0], b[1]); if (q2) q = q2; }
    if (q !== base) push('raid', q);
  }
  // 谢拉格 cold / freeze (初雪, 银灰): the most route coverage; then every 谢拉格 member
  const kj = ps_.filter((p) => nm(p) === '初雪' || nm(p) === '银灰'), kja = ps_.filter((p) => has(p, 'kjeragShip'));
  for (const [name, set] of [['kjerag-freeze', kj], ['kjerag-all', kja.length > kj.length ? kja : []]]) {
    if (!set.length) continue;
    let q = base;
    for (const p of set) { const b = best(p, q, allTiles, (k, d) => cov(p, k, d)); const q2 = b && ops.place(q, p.uid, b[0], b[1]); if (q2) q = q2; }
    if (q !== base) push(name, q);
  }
  // 莫斯提马: casters on its 4 neighbours (its aura)
  for (const p of ps_.filter((x) => nm(x) === '莫斯提马')) {
    let q = base; const k = q.get(p.uid); if (!k) continue;
    const [r, c] = k.split(',').map(Number), nb = [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]].map(([a, b]) => `${a},${b}`).filter((x) => tOfKey(x) >= 0 && tOfKey(x) < TILES);
    const cas = ps_.filter((x) => x !== p && x.rec0?.profession === 'CASTER');
    for (const x of cas) {
      if (nb.includes(q.get(x.uid))) continue;
      for (const kk of nb) { const o = ops.occ(q).get(kk); if (o != null && cas.some((y) => y.uid === o)) continue; const d = ops.firstDir(x.uid, kk, q.dirs.get(x.uid) ?? 'RIGHT'); const q2 = d && ops.place(q, x.uid, kk, d); if (q2) { q = q2; break; } }
    }
    if (q !== base) push('mostima', q);
  }
  // 远牙: its skill hits the whole row ahead -> the row's end tiles (rightmost, leftmost), the facing with the most route coverage
  for (const p of ps_.filter((x) => nm(x) === '远牙')) {
    const k = base.get(p.uid); if (!k) continue; const r = Number(k.split(',')[0]);
    const row = allTiles.filter((x) => Number(x.split(',')[0]) === r && DIRS.some((d) => ops.legal(p.uid, x, d))).sort((a, b) => Number(a.split(',')[1]) - Number(b.split(',')[1]));
    for (const [name, kk] of [['fartooth-right', row[row.length - 1]], ['fartooth-left', row[0]]]) {
      if (!kk || kk === k) continue; const b = best(p, base, [kk], (x, d) => cov(p, x, d)); push(name, b && ops.place(base, p.uid, b[0], b[1]));
    }
  }
  // generic: every piece turned to its best-coverage legal facing (turns only)
  { const q = clonePlan(base); let ch = 0;
    for (const p of ps_) { const k = q.get(p.uid); if (!k || p.heal) continue; const b = best(p, q, [k], (x, d) => cov(p, x, d) + (d === (q.dirs.get(p.uid) ?? 'RIGHT') ? 1e-6 : 0)); if (b && b[1] !== (q.dirs.get(p.uid) ?? 'RIGHT')) { q.dirs.set(p.uid, b[1]); ch++; } }
    if (ch) push('cover', q); }
  return out;
}

/** Last round's board refitted: kept pieces stay where they were (if still legal), the others keep the current tile when free, else the first free legal tile. */
function refit(prev, base, ops, P) {
  const q = new LayoutPlan(), used = new Set();
  for (const p of P.pieces) { const k = prev.get(p.uid); if (!k) continue; const d = ops.firstDir(p.uid, k, prev.dirs.get(p.uid) ?? 'RIGHT'); if (d && !used.has(k)) { q.set(p.uid, k); q.dirs.set(p.uid, d); used.add(k); } }
  for (const p of P.pieces) {
    if (q.has(p.uid)) continue;
    const k0 = base.get(p.uid), d0 = base.dirs.get(p.uid) ?? 'RIGHT';
    const tiles = [k0, ...Array.from({ length: TILES }, (_, t) => keyOfT(t))].filter((k) => k && !used.has(k));
    for (const k of tiles) { const d = ops.firstDir(p.uid, k, d0); if (d) { q.set(p.uid, k); q.dirs.set(p.uid, d); used.add(k); break; } }
  }
  return q.size === base.size ? q : null;
}

export function createPool(o = {}) {
  const K = Number(o.K ?? 8), GEN = o.gen ? XNetJS.load(o.gen) : null, GENN = Number(o.genN || 0), GENT = Number(o.genT || 1);
  const TPL = !!Number(o.tpl || 0), PREV = !!Number(o.prev || 0), EDITS = Number(o.edits || 0), Q = o.qnet || (o.q ? QNetJS.load(o.q) : null); // X9: qnet = a loaded QNetJS
  const LINEUP = Number(o.lineup || 0), LV = Number(o.lineupV || 2), MAX = Number(o.max || 48);
  const last = new Map(), stats = { builds: 0, cands: 0, bySrc: {} };
  const h16 = (a) => Float32Array.from(Float16Array.from(a));
  return {
    stats,
    observe(m, ps, key) { if (PREV) last.set(key, { round: m.round, plan: xCurrentPlanForTraining(ps) }); },
    /** Q of plans (featurizable ones) with the pool's scorer (null when none). */
    score(m, ps, P, X, plans, net = Q) {
      if (!net) return null;
      const cfs = plans.map((pl) => (net.cfg.cf ? h16(xCandFeatures(m, ps, P.chosen, pl, plans[0])) : null));
      return net.score(X, net.encode(X), plans.map((pl) => xAssignOf(X, pl)), cfs);
    },
    build(m, ps, P, X, key) {
      const C = layoutCandidatesForPicker(m, ps, K); if (!C) return null;
      const cur = C.cands[0], cands = [], seen = new Set(), ops = layoutOps(P, X);
      const rng = rngOf(`xpool:${m.round}:${ps.seat}:${sigOf(cur)}`);
      const add = (plan, src, extra = {}) => { if (!plan || cands.length >= MAX) return false; const s = sigOf(plan); if (seen.has(s)) return false; seen.add(s); cands.push({ plan, src, ...extra }); return true; };
      C.cands.forEach((pl, j) => add(pl, j === 0 ? 'cur' : 'leg'));
      if (GEN) {
        add(planOfAssign(X, GEN.decode(X, 0)), 'model');
        for (let g = 0; g < GENN; g++) add(planOfAssign(X, GEN.sample(X, GENT, rng)), 'gen');
      }
      if (TPL) for (const t of templates(P, X, cur, ops)) add(t.plan, t.src);
      if (PREV) { const L = last.get(key); if (L && L.round < m.round) add(refit(L.plan, cur, ops, P), 'prev'); }
      if (EDITS > 0) {
        let base = cur;
        if (Q) { const sc = this.score(m, ps, P, X, cands.map((c) => c.plan)); let b = 0; for (let j = 1; j < sc.length; j++) if (sc[j] > sc[b]) b = j; base = cands[b].plan; }
        const all = ops.edits(base);
        for (let g = 0, n = 0; n < EDITS && all.length && g < EDITS * 10; g++) { const j = Math.floor(rng() * all.length); if (add(all[j], 'edit')) n++; all.splice(j, 1); }
      }
      const lineups = [];
      if (LINEUP > 0) { // bench <-> board single swaps (adds when a slot is free), legacy's first LV layouts each
        const LP = xLineupProblemForTraining(m, ps), gd = ps.gd || m.gd;
        if (LP.bench.length) {
          const alts = [], seenL = new Set([LP.board.map((p) => p.uid).sort().join(',')]);
          const pushL = (arr) => { const k = arr.map((p) => p.uid).sort().join(','); if (!seenL.has(k)) { seenL.add(k); alts.push(arr); } };
          for (const b of LP.bench) { if (LP.count < LP.cap) pushL([...LP.board, b]); for (const u of LP.board) pushL(LP.board.filter((x) => x !== u).concat([b])); }
          const nat = (arr) => { const per = new Map(); for (const p of arr) { const r = gd.chess(p.id); for (const b of (r && r.bonds) || []) { if (!per.has(b)) per.set(b, new Set()); per.get(b).add(gd.baseIdOf ? gd.baseIdOf(p.id) : p.id); } }
            let s = 0; for (const [b, st] of per) { const th = gd.bond(b)?.thresholds || []; s += th.filter((x) => st.size >= x).length * (gd.bond(b)?.isCore ? 10 : 4); } return s; };
          alts.sort((a, b) => nat(b) - nat(a));
          for (const arr of alts.slice(0, LINEUP)) { const li = lineups.length; lineups.push(arr); for (const pl of xPlansForLineupForTraining(m, ps, arr, LV)) add(pl, 'lineup', { li }); }
        }
      }
      stats.builds++; stats.cands += cands.length; for (const c of cands) stats.bySrc[c.src] = (stats.bySrc[c.src] || 0) + 1;
      return { cands, lineups, chosen: C.chosen, legacy: C };
    },
  };
}
