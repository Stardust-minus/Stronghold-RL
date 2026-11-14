// tools/rl/xfeat.mjs — xplace (11-13): features of a seat's placement problem for the learned layout policy. The same
// code serves the label generator (xlabel.mjs) and deploy (xplace.mjs); nothing here builds a battle.
// Board: rows 9-12 x cols 3-9 (spec.mjs TILES = 28), facings UP / RIGHT / DOWN / LEFT (sim/dir.js: UP = +row).
// Per piece FP numbers + an id code (embedded), per tile FT, per (piece, tile, facing) FE, one global row FG.
import { TILES, DIRS, tileIndex, tileRC, buildVocab } from './spec.mjs';
import { rangeTiles, xAirflowForTraining, layoutFeatures, positionalGainForTraining } from '../../server/match/bot.js';

export const XFEAT_VERSION = 1;
const PROF = ['VANGUARD', 'GUARD', 'TANK', 'SNIPER', 'CASTER', 'MEDIC', 'SUPPORT', 'SPECIAL'];
// positional wording in a trait (garrison) description: the relations a layer engine / gift can depend on
const POS_RX = [/身前/, /身后/, /同一?行/, /同一?列/, /周围|四周|相邻|旁边|两侧/, /最右|最左|最前|最后/, /前方/, /层数/];
const NAMES = ['凛御银灰', '魔王', '华法琳', '寒芒克洛丝', '耀骑士临光', '远牙', '莫斯提马', '砾', '瑕光', '初雪', '银灰', '阿戈尔'];
const KEYS = ['SERVER_ADD_BOND_POSITION', 'SERVER_ADD_BOND_METHOD', 'SERVER_FRONT_SAME_EFFECT_PREP_FIN', 'SERVER_FRONT_SAME_EFFECT_PREP_START',
  'SERVER_MOST_BOND', 'SERVER_POOL_CHAR'];
const EV = ['SERVER_PREP_FIN', 'SERVER_PREP_START', 'IN_BATTLE'];
const symlog = (x) => Math.sign(x) * Math.log1p(Math.abs(x));
export const DIR_VEC = { UP: [1, 0], RIGHT: [0, 1], DOWN: [-1, 0], LEFT: [0, -1] };

const vocabCache = new WeakMap();
export function xVocab(gd) {
  let v = vocabCache.get(gd);
  if (!v) { const b = buildVocab(gd); v = { piece: b.piece, bondIds: b.bondIds, bond: b.bond, NB: b.bondIds.length, hash: b.hash }; vocabCache.set(gd, v); }
  return v;
}

export function xDims(NB) {
  const FP = 40 + POS_RX.length + NAMES.length + KEYS.length + EV.length + 2 * NB;
  return { FP, FT: 20, FE: 14, FG: 16 + NB };
}

const traitCache = new Map();
function traitFlags(gd, rec) {
  const id = rec && rec.chessId; if (!id) return null;
  let f = traitCache.get(id); if (f) return f;
  f = new Float32Array(POS_RX.length + NAMES.length + KEYS.length + EV.length);
  const gars = (rec.garrisonIds || []).map((g) => gd.garrison(g)).filter(Boolean);
  for (const g of gars) {
    const d = String(g.desc || '');
    POS_RX.forEach((rx, i) => { if (rx.test(d)) f[i] = 1; });
    const k = KEYS.indexOf(g.effectKey); if (k >= 0) f[POS_RX.length + NAMES.length + k] = 1;
    const e = EV.indexOf(g.eventType); if (e >= 0 && /层/.test(d)) f[POS_RX.length + NAMES.length + KEYS.length + e] = 1;
  }
  const n = NAMES.indexOf(rec.name || ''); if (n >= 0) f[POS_RX.length + n] = 1;
  traitCache.set(id, f);
  return f;
}

/** Canonical decode order of the problem's pieces: blockers (block count desc), then DPS desc, then uid. */
export function xOrder(P) {
  return P.pieces.map((p, i) => i).sort((a, b) => (P.pieces[b].block - P.pieces[a].block) || (P.pieces[b].dps - P.pieces[a].dps) || (P.pieces[a].uid - P.pieces[b].uid));
}

/**
 * Features of problem P (bot.js xPlaceProblemForTraining) of seat ps: { n, order, ids (Int32 piece codes), fp, ft, fe, fg,
 * legal (Uint8 n x TILES x 4: legal tile and facing), cur (Int16 n x 2: the current tile / facing index or -1) }.
 * fe is n x TILES x 4 x FE (zero where illegal). `ctx` { lp0, solo }.
 */
export function xFeaturize(m, ps, P, ctx = {}) {
  const gd = ps.gd || m.gd, V = xVocab(gd), NB = V.NB, D = xDims(NB), model = P.model;
  const order = xOrder(P), n = order.length;
  const pcs = order.map((i) => P.pieces[i]);
  // ---- field: ground flow / prog / fly per own-board tile (route tiles may lie outside the deploy board) ----
  const ground = model.ground || new Map(), fly = new Map(), dwell = new Map();
  let gTot = 0, fTot = 0;
  for (const rt of model.routes || []) {
    for (const k of rt.tiles) {
      if (rt.fly) { fly.set(k, (fly.get(k) || 0) + rt.n * rt.tileTime); fTot += rt.n * rt.tileTime; }
      else gTot += rt.n * rt.tileTime;
      if (rt.dwell) dwell.set(k, (dwell.get(k) || 0) + rt.dwell);
    }
  }
  const gTime = new Map(); // ground exposure per tile: Σ n x tileTime
  for (const rt of model.routes || []) if (!rt.fly) for (const k of rt.tiles) gTime.set(k, (gTime.get(k) || 0) + rt.n * rt.tileTime);
  const gEarly = new Map();
  for (const rt of model.routes || []) if (!rt.fly) rt.tiles.forEach((k, i) => { const pr = rt.tiles.length > 1 ? i / (rt.tiles.length - 1) : 1; gEarly.set(k, (gEarly.get(k) || 0) + rt.n * rt.tileTime * (1 - pr)); });
  const gT = Math.max(1e-6, gTot), fT = Math.max(1e-6, fTot);
  const routeKeys = [...ground.keys()];
  // ---- tiles ----
  const ft = new Float32Array(TILES * D.FT);
  const curOcc = new Set(pcs.map((p) => p.key).filter(Boolean));
  for (let t = 0; t < TILES; t++) {
    const [r, c] = tileRC(t), k = `${r},${c}`, cls = P.map.get(k), o = t * D.FT, gr = ground.get(k);
    ft[o] = 1; ft[o + 1] = cls === 'melee' ? 1 : 0; ft[o + 2] = cls === 'ranged' ? 1 : 0; ft[o + 3] = cls ? 0 : 1;
    ft[o + 4] = (r - 9) / 3; ft[o + 5] = (c - 3) / 6;
    if (gr) { ft[o + 6] = symlog(gr.flow) / 4; ft[o + 7] = gr.prog; ft[o + 8] = 1; }
    ft[o + 9] = symlog(fly.get(k) || 0) / 4;
    let dmin = 6; for (const q of routeKeys) { const [qr, qc] = q.split(',').map(Number); dmin = Math.min(dmin, Math.abs(qr - r) + Math.abs(qc - c)); }
    ft[o + 10] = dmin / 6;
    let f1 = 0, f2 = 0, adj = 0;
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) {
      const g = ground.get(`${r + dr},${c + dc}`); if (!g) continue;
      if (Math.max(Math.abs(dr), Math.abs(dc)) <= 1) f1 += g.flow; f2 += g.flow;
      if (Math.abs(dr) + Math.abs(dc) === 1) adj++;
    }
    ft[o + 11] = symlog(f1) / 5; ft[o + 12] = symlog(f2) / 6;
    let air = 0; for (const d of DIRS) if (Math.abs(xAirflowForTraining(model, k, d) - 1) > 1e-6) air = 1;
    ft[o + 13] = air; ft[o + 14] = P.boss ? 1 : 0; ft[o + 15] = P.boss && P.boss.side === 'R' ? 1 : 0;
    ft[o + 16] = symlog(dwell.get(k) || 0) / 4; ft[o + 17] = curOcc.has(k) ? 1 : 0; ft[o + 18] = adj / 4;
    ft[o + 19] = (gTime.get(k) || 0) / gT;
  }
  // ---- pieces ----
  const fp = new Float32Array(n * D.FP), ids = new Int32Array(n), cur = new Int16Array(n * 2).fill(-1);
  const bondsView = ps.bonds || {};
  pcs.forEach((p, i) => {
    const rec = p.rec0 || {}, st = rec.stats || {}, o = i * D.FP, grid = (p.rec && p.rec.rangeGrid) || rec.rangeGrid || [];
    ids[i] = V.piece.get(gd.baseIdOf ? gd.baseIdOf(p.id) : p.id) ?? V.piece.get(p.id) ?? 0;
    fp[o] = 1; const ci = ['melee', 'ranged', 'all', 'high'].indexOf(p.cls); if (ci >= 0) fp[o + 1 + ci] = 1;
    fp[o + 5] = p.block / 3; fp[o + 6] = p.heal ? 1 : 0; fp[o + 7] = p.air ? 1 : 0; fp[o + 8] = p.ground ? 1 : 0;
    fp[o + 9] = symlog(p.dps) / 8; fp[o + 10] = symlog(st.maxHp || 0) / 10; fp[o + 11] = (st.def || 0) / 1000; fp[o + 12] = (st.res || 0) / 100;
    fp[o + 13] = ((st.bat || 1) * 100 / Math.max(10, st.aspd || 100)) / 3; fp[o + 14] = grid.length / 20;
    let fw = 0, bk = 0, sd = 0; for (const [dr, dc] of grid) { fw = Math.max(fw, dc); bk = Math.max(bk, -dc); sd = Math.max(sd, Math.abs(dr)); }
    fp[o + 15] = fw / 5; fp[o + 16] = bk / 3; fp[o + 17] = sd / 3;
    fp[o + 18] = p.golden ? 1 : 0; fp[o + 19] = (rec.tier || 0) / 6; fp[o + 20] = (rec.rarity || 0) / 6;
    const pi = PROF.indexOf(String(rec.profession || '').toUpperCase()); if (pi >= 0) fp[o + 21 + pi] = 1;
    fp[o + 29] = rec.dmgType === 'arts' ? 1 : 0; fp[o + 30] = rec.attackKind === 'none' ? 1 : 0; fp[o + 31] = (p.items || []).length / 3;
    if (p.key) {
      const [r, c] = p.key.split(',').map(Number), t = tileIndex(r, c), d = DIRS.indexOf(p.dir);
      fp[o + 32] = 1; fp[o + 33] = (r - 9) / 3; fp[o + 34] = (c - 3) / 6; if (d >= 0) fp[o + 35 + d] = 1;
      cur[i * 2] = t; cur[i * 2 + 1] = d;
    }
    fp[o + 39] = (p.items || []).some((x) => /chess_item_3_06_e|chess_item_3_01_e/.test(x)) ? 1 : 0;
    const tf = traitFlags(gd, rec); if (tf) fp.set(tf, o + 40);
    const ob = o + 40 + (tf ? tf.length : POS_RX.length + NAMES.length + KEYS.length + EV.length);
    for (const b of rec.bonds || []) { const j = V.bond.get(b); if (j === undefined) continue; fp[ob + j] = 1; if (bondsView[b]?.active) fp[ob + NB + j] = 1; }
  });
  // ---- pairs (piece, tile, facing) ----
  const fe = new Float32Array(n * TILES * 4 * D.FE), legal = new Uint8Array(n * TILES * 4);
  const deployKeys = new Set(P.map.keys());
  pcs.forEach((p, i) => {
    const lg = new Set(p.legal), [lr, lc] = p.key ? p.key.split(',').map(Number) : [null, null];
    for (let t = 0; t < TILES; t++) {
      const [r, c] = tileRC(t), k = `${r},${c}`; if (!lg.has(k)) continue;
      const gr = ground.get(k);
      for (let d = 0; d < 4; d++) {
        const dir = DIRS[d], o = ((i * TILES + t) * 4 + d) * D.FE;
        legal[(i * TILES + t) * 4 + d] = 1;
        const cov = rangeTiles(p.rec || p.rec0, r, c, dir);
        let cg = 0, cn = 0, cf = 0, ce = 0, cd = 0, co = 0;
        for (const q of cov) {
          const gt = gTime.get(q); if (gt) { cg += gt; cn++; ce += gEarly.get(q) || 0; }
          const fq = fly.get(q); if (fq) cf += fq;
          if (deployKeys.has(q) && q !== k) cd++;
          if (curOcc.has(q) && q !== k) co++;
        }
        const mul = xAirflowForTraining(model, k, dir);
        fe[o] = cg / gT; fe[o + 1] = cn / 8; fe[o + 2] = p.air ? cf / fT : 0; fe[o + 3] = ce / gT;
        if (p.block && gr) { fe[o + 4] = (gTime.get(k) || 0) / gT; fe[o + 5] = gr.prog; }
        fe[o + 6] = mul - 1;
        if (lr !== null) { fe[o + 7] = lr === r && lc === c ? 1 : 0; fe[o + 8] = fe[o + 7] && dir === p.dir ? 1 : 0; fe[o + 9] = (Math.abs(lr - r) + Math.abs(lc - c)) / 6; }
        fe[o + 10] = symlog(p.dps * mul * (cg / gT + (p.air ? cf / fT : 0))) / 8;
        fe[o + 11] = cd / 8; fe[o + 12] = co / 6; fe[o + 13] = p.ground && !p.heal ? 0 : 1;
      }
    }
  });
  // ---- global ----
  const fg = new Float32Array(D.FG);
  fg[0] = 1; fg[1] = (m.round || 0) / 15; fg[2] = P.boss ? 1 : 0; fg[3] = m.round === m.gd.hiddenRound && m.round !== m.gd.bossRound ? 1 : 0;
  fg[4] = m.isSolo ? 1 : 0; fg[5] = n / 12; fg[6] = symlog(gTot) / 6; fg[7] = symlog(fTot) / 6;
  fg[8] = Math.max(0, ps.lp || 0) / Math.max(1, ctx.lp0 || ps.lp || 1); fg[9] = (ps.level || 0) / 10;
  fg[10] = String(m.difficulty || '') === 'ABYSS' ? 1 : 0; fg[11] = (m.players?.size || 1) / 4;
  let mainT = 0; for (const [b, v] of Object.entries(bondsView)) if (v && v.active && gd.bond(b)?.isCore) mainT = Math.max(mainT, v.tier || 0);
  fg[12] = mainT / 2; fg[13] = Object.values(bondsView).reduce((a, v) => a + (v && v.active ? 1 : 0), 0) / 8;
  fg[14] = symlog(Object.values(ps.layers || {}).reduce((a, b) => a + (Number(b) || 0), 0)) / 7; fg[15] = deployKeys.size / 28;
  V.bondIds.forEach((b, j) => { const v = bondsView[b]; if (v && v.active) fg[16 + j] = 0.5 + 0.5 * Math.min(1, symlog(v.layers || 0) / 6); });
  return { version: XFEAT_VERSION, n, order, uids: pcs.map((p) => p.uid), ids, fp, ft, fe, fg, legal, cur, dims: D, vocabHash: V.hash };
}

/** A LayoutPlan-like assignment [[tile, dir] per ordered piece] of features X → { uid: [key, dir] } entries. */
export function xPlanEntries(X, assign) {
  return X.uids.map((u, i) => { const [t, d] = assign[i]; const [r, c] = tileRC(t); return [u, `${r},${c}`, DIRS[d]]; });
}

/** A LayoutPlan (uid → key, .dirs) of the ordered pieces of X as [[tile, dir] ...] (-1 when unplaced). */
export function xAssignOf(X, plan) {
  return X.uids.map((u) => { const k = plan.get(u); if (!k) return [-1, -1]; const [r, c] = k.split(',').map(Number); return [tileIndex(r, c), DIRS.indexOf(plan.dirs?.get(u) ?? 'RIGHT')]; });
}

// ---- candidate-level layout features (no sims): legacy's layout-model terms (bot.js layoutFeatures) and the prep-time
// positional layer gain + battle-time positional conditions (positionalGainForTraining), each vs the current layout too ----
export const XCF = 26;
const LF_KEYS = ['value', 'cov', 'covMin', 'flyCov', 'onRoad', 'leader', 'diff', 'units', 'leaderCov', 'leaderBlock', 'leaderExp', 'healCov'];
const BC_KEYS = ['kjeragFront', 'moFront', 'skillFront', 'kazFront', 'farRight', 'mostima', 'sides', 'helmet', 'total'];
export function xCandFeatures(m, ps, chosen, plan, plan0) {
  const f = new Float32Array(XCF);
  const lf = layoutFeatures(m, ps, chosen, plan, plan0) || {};
  LF_KEYS.forEach((k, i) => { const v = Number(lf[k]) || 0; f[i] = k === 'units' ? v / 12 : k === 'value' ? Math.tanh(v) : v; });
  const pg = positionalGainForTraining(m, ps, plan) || {};
  f[12] = (pg.layers || 0) / 20; f[13] = (pg.gifts || 0) / 3; f[14] = (pg.fin || 0) / 20; f[15] = (pg.start || 0) / 20;
  BC_KEYS.forEach((k, i) => { f[16 + i] = ((pg.battle && pg.battle[k]) || 0) / 3; });
  f[25] = 1; // present
  return f;
}
