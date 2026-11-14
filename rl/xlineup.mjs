// tools/rl/xlineup.mjs — xplace (11-13): lineup (上场名单) ceiling oracle, training / diagnostics only (sims are the teacher).
// At every RL-seat prep with a fight (after prepBegin): candidate lineups = legacy's fielded chess, plus (mode 'lineup') single
// swaps bench <-> board (and adds when a board slot is free), ranked by a bond-progress heuristic (national tiers first); each
// lineup is placed by legacy's layout variants (default + REHEARSAL_VARIANTS + PICK_EXTRA_VARIANTS); every (lineup, plan) is
// rehearsed on per-match fixed seeds (never the live one) and the best by the search key (normal: rule D vs the current layout;
// boss: leaks, leader damage, kills) is played. mode 'place' = the same oracle restricted to legacy's lineup (the control arm).
// env option xLineup { mode, L, V, seeds, top }; per match stats slot.xl -> results 'xlineup'.
import { performance } from 'node:perf_hooks';
import { xPlaceProblemForTraining, xRehearseForTraining, xCurrentPlanForTraining, applyPlanForPicker, xLineupProblemForTraining,
  xPlansForLineupForTraining, xApplyLineupForTraining } from '../../server/match/bot.js';

const mean = (xs) => { const ok = xs.filter(Boolean); if (!ok.length) return null; const o = { leaks: 0, boss: 0, kills: 0, layers: 0 }; for (const x of ok) for (const k in o) o[k] += x[k] / ok.length; return o; };
const keyOf = (boss, s, s0) => (!s ? -Infinity : boss ? -s.leaks * 1e7 + s.boss + 1e-3 * s.kills
  : s.layers + 0.01 * s.kills - 1000 * Math.max(0, s.leaks - (s0 ? s0.leaks : s.leaks) - 1e-9));

function bondCounts(gd, pieces) { // distinct base ids per bond among the pieces
  const per = new Map();
  for (const p of pieces) {
    const rec = gd.chess(p.id); if (!rec) continue; const base = gd.baseIdOf ? gd.baseIdOf(p.id) : p.id;
    for (const b of rec.bonds || []) { let s = per.get(b); if (!s) { s = new Set(); per.set(b, s); } s.add(base); }
  }
  return per;
}
const tierOf = (gd, b, n) => { const t = gd.bond(b)?.thresholds || []; let k = 0; for (const x of t) if (n >= x) k++; return k; };

/** Heuristic score of a lineup's bond state: core (national) tiers weigh most, then class tiers, then main-nation distinct count. */
function bondScore(gd, pieces, main) {
  const per = bondCounts(gd, pieces); let s = 0;
  for (const [b, set] of per) { const core = !!gd.bond(b)?.isCore; s += tierOf(gd, b, set.size) * (core ? 10 : 4); }
  if (main) s += 1.5 * (per.get(main)?.size || 0);
  return s;
}

export function createLineupHook(o = {}) {
  const MODE = o.mode || 'lineup', L = Number(o.L || 8), V = Number(o.V || 5), SEEDS = Number(o.seeds || 3), TOP = Number(o.top || 3);
  const hook = (m, ps, slot, env) => {
    if (!slot.rl.has(ps.playerId) || slot.humans.has(ps.playerId)) return;
    const t0 = performance.now(), S = (slot.xl ||= { decisions: 0, eligible: 0, lineupChanged: 0, planChanged: 0, applyFail: 0, rehearsals: 0, ms: 0, cands: 0 });
    const P = xPlaceProblemForTraining(m, ps); if (!P || !P.fight) return;
    const boss = !m.wave, gd = ps.gd || m.gd, LP = xLineupProblemForTraining(m, ps);
    const lineups = [LP.board];
    if (MODE === 'lineup' && LP.bench.length) {
      S.eligible++;
      // main nation: the core bond with the most distinct fielded members (ties: layers)
      let main = null, best = -1;
      for (const [b, v] of Object.entries(ps.bonds || {})) if (v && gd.bond(b)?.isCore && (v.count || 0) > best) { best = v.count || 0; main = b; }
      const alts = [], s0 = bondScore(gd, LP.board, main);
      const seen = new Set([LP.board.map((p) => p.uid).sort().join(',')]);
      const push = (arr) => { const k = arr.map((p) => p.uid).sort().join(','); if (seen.has(k)) return; seen.add(k); alts.push({ arr, sc: bondScore(gd, arr, main) - s0 }); };
      for (const b of LP.bench) {
        if (LP.count < LP.cap) push([...LP.board, b]);
        for (const u of LP.board) push(LP.board.filter((x) => x !== u).concat([b]));
      }
      alts.sort((a, b) => b.sc - a.sc);
      for (const a of alts.slice(0, L - 1)) lineups.push(a.arr);
    }
    const cur = xCurrentPlanForTraining(ps);
    const cands = [{ li: 0, plan: cur, s: [] }];
    lineups.forEach((arr, li) => { for (const plan of xPlansForLineupForTraining(m, ps, arr, V)) cands.push({ li, plan, s: [] }); });
    const salt = `${slot.seed}`;
    // seed 0 for all candidates (grouped by lineup: xRehearseForTraining needs the lineup's pieces as `chosen`)
    const byLi = new Map(); cands.forEach((c, j) => { if (!byLi.has(c.li)) byLi.set(c.li, []); byLi.get(c.li).push(j); });
    const score = (idx, sx) => {
      const g = new Map(); for (const j of idx) { const li = cands[j].li; if (!g.has(li)) g.set(li, []); g.get(li).push(j); }
      for (const [li, js] of g) {
        const r = xRehearseForTraining(m, ps, lineups[li], js.map((j) => cands[j].plan), [sx], salt) || [];
        js.forEach((j, q) => { cands[j].s[sx] = r[q] ? r[q][0] : null; }); S.rehearsals += js.length;
      }
    };
    score(cands.map((_, j) => j), 0);
    if (!cands[0].s[0]) return;
    const c00 = cands[0].s[0];
    const rank = cands.map((_, j) => j).filter((j) => cands[j].s[0]).sort((a, b) => keyOf(boss, cands[b].s[0], c00) - keyOf(boss, cands[a].s[0], c00));
    const val = [...new Set([0, ...rank.slice(0, TOP)])];
    for (let sx = 1; sx < SEEDS; sx++) score(val, sx);
    const m0 = mean(cands[0].s.slice(0, SEEDS));
    let pick = 0; for (const j of val) if (keyOf(boss, mean(cands[j].s.slice(0, SEEDS)), m0) > keyOf(boss, mean(cands[pick].s.slice(0, SEEDS)), m0)) pick = j;
    S.decisions++; S.cands += cands.length;
    const c = cands[pick];
    if (pick !== 0) {
      if (c.li !== 0) { const ok = xApplyLineupForTraining(m, ps, lineups[c.li], c.plan); S.lineupChanged++; if (!ok) S.applyFail++; }
      else { applyPlanForPicker(m, ps, LP.board, c.plan); S.planChanged++; }
    }
    S.ms += performance.now() - t0;
  };
  return { hook };
}
