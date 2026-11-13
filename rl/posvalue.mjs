// Value of positional traits (track 2 diagnostics), same seeds per arm:
//   legacy  legacy's placement (the potential of a trait-aware improver is still measured, dry: nothing moves)
//   trait   legacy's placement, then traitImproveForTraining on the exact prep-time layer gain (catalogue A)
//   trait+  the same plus the battle-time recipe conditions (catalogue B / C, battleW per satisfied condition)
//   oracle  trait+, then a real-wave rehearsal hill-climb from --oracle-from (live boss seed: an upper bound)
// The economy: --econ legacy (every seat legacy: the teacher boards) or --policy <export base> (version 3, greedy).
// --plan kjeragShip [--force-band band_kirara]: ai_0's legacy focus bond (a 谢拉格 board).
// Per changed prep the engine's own onPrepEnd forecast (EffectDispatcher.forecastLayers) checks the predicted layer gain.
//   node tools/rl/posvalue.mjs --arm trait --econ legacy --config solo:ABYSS --seed0 9000000 --matches 24
import { SPEnv } from './env.mjs';
import { SPPolicy2 } from './infer2.mjs';
import { readFileSync } from 'node:fs';
import { appendFileSync } from 'node:fs';
import { PickerHead, pickerRow, bestOf } from './picker.mjs';
import { positionalGainForTraining, traitImproveForTraining, oracleImproveForTraining, searchImproveForTraining, pickCandidatesForTraining, equipItemsPicked,
  layoutCandidatesForPicker, rehearseCandidatesForTraining, applyPlanForPicker, bountySpawnsForTraining, stackOracleForTraining } from '../../server/match/bot.js';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const T0 = performance.now();
const arm = arg('arm', 'legacy'), cfg = arg('config', 'solo:ABYSS'), econLegacy = arg('econ', '') === 'legacy';
const net = econLegacy ? null : SPPolicy2.load(arg('policy'));
const coopAi = cfg.split(':')[0].endsWith('ai');
const BATTLE_W = Number(arg('battle-w', 3)), ORACLE_FROM = Number(arg('oracle-from', 12)), ORACLE_K = Number(arg('oracle-k', 32));
const prep = { preps: 0, changed: 0, predFin: 0, exactFin: 0, predStart: 0, gifts: 0, battleBefore: 0, battleAfter: 0, potential: 0, byRound: {},
  oracle: { runs: 0, improved: 0, candidates: 0 }, search: { runs: 0, improved: 0, candidates: 0, rehearsals: 0, ms: 0, msMax: 0 }, forecastErrors: 0 };
const RECIPES = JSON.parse(readFileSync(arg('recipes', new URL('./recipes.json', import.meta.url)), 'utf8'));
const REC = arg('rec', ''), HEAD = arg('head') ? PickerHead.load(arg('head')) : null;
const SK = Number(arg('K', 16)), SS = Number(arg('S', 2)), SEARCH_FROM = Number(arg('from', 12));
const perSeed = new Map(); // seed -> { potential (dry improver gain, R>=9), changed preps }
const forecast = (m, ps) => { try { const r = m.dispatcher.forecastLayers(ps, 'onPrepEnd'); return Object.values(r.bonds || {}).reduce((s, b) => s + (b.n || 0), 0); } catch { prep.forecastErrors++; return NaN; } };
const readyHook = (m, ps, slot) => {
  if (!econLegacy && !slot.rl.has(ps.playerId) && !slot.recipeSeat?.has(ps.playerId)) return;
  if (slot.humans.has(ps.playerId)) return;
  const S = perSeed.get(slot.seed) || { potential: 0, changed: 0, kjerag: 0, nation: 0 }; perSeed.set(slot.seed, S);
  if (ps.playerId === 'ai_0') { // the tracked seat: Σ layers, its main (recipe bond / most-member core) and secondary (recipe sub / top non-core) bond
    S.layers = Object.values(ps.layers || {}).reduce((a, b) => a + (Number(b) || 0), 0);
    const act = (ps.privateView().bonds || []).filter((b) => b.active && !b.off), rk = slot.recipeSeat?.get(ps.playerId), rc = rk ? RECIPES[rk - 1] : null;
    const core = (b) => !!m.gd.bond(b.bondId)?.isCore, byL = (a, b) => (b.layers || 0) - (a.layers || 0);
    const main = rc ? act.find((b) => b.bondId === rc.bond) : act.filter(core).sort((a, b) => (b.count || 0) - (a.count || 0) || byL(a, b))[0];
    const sec = (rc ? act.filter((b) => (rc.subs || []).includes(b.bondId)) : act.filter((b) => !core(b))).sort(byL)[0];
    const pack = (b) => (b ? { id: b.bondId, tier: b.tier || 0, layers: b.layers || 0 } : null);
    S.main = pack(main); S.sec = pack(sec); S.mainWanted = rc ? rc.bond : null;
    if (main) { (S.mainTiers ||= {})[m.round] = main.tier || 0; if ((main.tier || 0) >= 2 && S.tier2Round == null) { S.tier2Round = m.round; S.tier2Layers = S.layers; } }
  }
  const bonds = ps.bonds || {};
  if (m.round >= 9) { if (bonds.kjeragShip?.active) S.kjerag++; if (['kjeragShip', 'kazimierzShip', 'lateranoShip', 'sargonShip', 'siracusaShip', 'victoriaShip', 'egirShip'].some((b) => bonds[b]?.active)) S.nation++; }
  const g0 = positionalGainForTraining(m, ps);
  const R = (prep.byRound[m.round] ||= { preps: 0, fin0: 0, fin1: 0, start0: 0, start1: 0, battle0: 0, battle1: 0, changed: 0 });
  R.preps++; prep.preps++; R.fin0 += g0.fin; R.start0 += g0.start; R.battle0 += g0.battle.total; prep.battleBefore += g0.battle.total;
  let g1 = g0;
  { // bounty rounds (every arm): which rounds qualify, by the bounty's first-seen round and card
    const bs = bountySpawnsForTraining(m, ps), Bq = (prep.bounty ||= { rounds: 0, byFirst: {}, byCard: {}, picks: 0, nonDefault: 0, leaks0: 0, leaks1: 0, ms: 0 });
    const seen = (slot.bountySeen ||= new Map());
    for (const b of ps.bounties || []) if (!seen.has(b.id)) seen.set(b.id, m.round);
    if (bs.length) {
      Bq.rounds++;
      for (const b of ps.bounties || []) { const f = seen.get(b.id); Bq.byFirst[f] = (Bq.byFirst[f] || 0) + 1; const c = b.card?.effectId || b.card?.name || '?'; Bq.byCard[c] = (Bq.byCard[c] || 0) + 1; }
      if (arm === 'bountyPick') {
        const t0 = performance.now(), C = layoutCandidatesForPicker(m, ps, SK, { mixBounty: true });
        if (C && C.cands.length > 1) {
          const sc = rehearseCandidatesForTraining(m, ps, C.chosen, C.cands, { seed: arg('seed', 'fixed') }) || [], idx = Math.max(0, bestOf(sc));
          Bq.picks++; if (sc[0]) Bq.leaks0 += sc[0].leaks; if (sc[idx]) Bq.leaks1 += sc[idx].leaks;
          if (idx > 0) { applyPlanForPicker(m, ps, C.chosen, C.cands[idx]); Bq.nonDefault++; S.changed++; prep.changed++; }
        }
        Bq.ms += performance.now() - t0;
      }
    }
    if (arm === 'bountyPick' || arm === 'bountyA') return;
  }
  if (arm === 'stackoracle' && arg('oracle-seats', 'all') === 'recipe' && !slot.recipeSeat?.has(ps.playerId)) return;
  if (arm === 'stackoracle' && arg('oracle-rounds', 'all') !== 'all' && (arg('oracle-rounds') === 'boss') !== !m.wave) return;
  if (arm === 'stackoracle') { // teacher-only oracle placement every round with a fight (normal: layers under no extra leaks; boss: outcome)
    const t0 = performance.now(), o = stackOracleForTraining(m, ps, { K: SK, perturb: arg('perturb', '1') !== '0', seed: arg('seed', 'live'), score: arg('score', 'layers') });
    if (o) { const X = (prep.stack ||= { normal: 0, boss: 0, changed: 0, dLayers: 0, ms: 0 }); X[o.boss ? 'boss' : 'normal']++; if (o.idx > 0) { X.changed++; S.changed++; prep.changed++; }
      if (!o.boss) X.dLayers += o.layers - o.layers0; X.ms += performance.now() - t0; }
    return;
  }
  if (arm === 'layers') { // normal rounds: the spread of this battle's layer gains across K legacy candidates (fixed seed), or pick by them
    const lo = Number(arg('from', 4)), hi = Number(arg('to', 12));
    if (!m.wave || m.round < lo || m.round > hi) return;
    const C = layoutCandidatesForPicker(m, ps, SK); if (!C || C.cands.length < 2) return;
    const sc = rehearseCandidatesForTraining(m, ps, C.chosen, C.cands) || [];
    const sum = (x) => (x ? Object.values(x.lg || {}).reduce((a, b) => a + b, 0) : -1), L = (prep.layers ||= { n: 0, d: [], dOk: [], base: 0, byBond: {}, byBondOk: {}, leaks0: 0, leaksPick: 0, changed: 0, byRound: {} });
    const s0 = sum(sc[0]); if (s0 < 0) return;
    let jm = 0, jok = 0;
    sc.forEach((x, j) => { if (!x) return; if (sum(x) > sum(sc[jm])) jm = j; if (x.leaks <= sc[0].leaks && sum(x) > sum(sc[jok])) jok = j; });
    L.bondBase ||= {}; for (const [b, v] of Object.entries(sc[0].lg || {})) L.bondBase[b] = (L.bondBase[b] || 0) + v;
    L.n++; L.base += s0; L.d.push(sum(sc[jm]) - s0); L.dOk.push(sum(sc[jok]) - s0); L.leaks0 += sc[0].leaks;
    const R = (L.byRound[m.round] ||= { n: 0, d: 0, dOk: 0 }); R.n++; R.d += sum(sc[jm]) - s0; R.dOk += sum(sc[jok]) - s0;
    for (const [B, j] of [[L.byBond, jm], [L.byBondOk, jok]]) for (const b of new Set([...Object.keys(sc[j].lg || {}), ...Object.keys(sc[0].lg || {})])) {
      const v = (sc[j].lg?.[b] || 0) - (sc[0].lg?.[b] || 0); if (v) B[b] = (B[b] || 0) + v; }
    if (arg('mode', 'spread') === 'pick') { L.leaksPick += sc[jok].leaks; if (jok > 0) { applyPlanForPicker(m, ps, C.chosen, C.cands[jok]); L.changed++; S.changed++; prep.changed++; } }
    return;
  }
  if (arm === 'equip') return; // the equipment pick runs in equipHook (legacy's equipItems step), not at Ready
  if (arm === 'pickrec' || arm === 'picker') { // boss rounds: label (fixed-seed rehearsal of every candidate, play the best) or learned pick
    if (m.wave || (m.round !== m.gd.bossRound && m.round !== m.gd.hiddenRound)) return;
    const X = (prep.picker ||= { decisions: 0, nonDefault: 0, cands: 0, ms: 0, msMax: 0, sims: 0, labels: 0, single: 0 });
    const t0 = performance.now(), nb = m.newBattle; let sims = 0;
    if (arm === 'picker') m.newBattle = function (...a) { sims++; return nb.apply(this, a); }; // deploy-rule counter: no battle may be built
    try {
      const C = layoutCandidatesForPicker(m, ps, SK); if (!C || C.cands.length < 2) { X.single++; return; }
      const s = env._seat(slot, ps), ob = new Float32Array(env.encoder.dim), mk = new Uint8Array(env.encoder.A);
      env.encoder.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: env.rules, cache: {} }, ob, mk);
      net.forward(ob, mk, { value: false }); const c = Float32Array.from(net.c);
      const rows = C.cands.map((_, j) => pickerRow(C.feats[j], C.variants[j], j === 0, C.feats[0]));
      let idx;
      if (arm === 'picker') { const lg = HEAD.score(c, rows); idx = 0; for (let j = 1; j < lg.length; j++) if (lg[j] > lg[idx]) idx = j; }
      else {
        const sc = rehearseCandidatesForTraining(m, ps, C.chosen, C.cands); idx = Math.max(0, bestOf(sc || []));
        if (REC) { const oi = [], ov = []; for (let q = 0; q < ob.length; q++) if (ob[q] !== 0) { oi.push(q); ov.push(+ob[q].toFixed(5)); }
          appendFileSync(REC, JSON.stringify({ seed: slot.seed, round: m.round, pid: ps.playerId, D: ob.length, obs: { i: oi, v: ov }, c: Array.from(c, (x) => +x.toFixed(4)), rows: rows.map((r) => Array.from(r, (x) => +x.toFixed(4))), scores: sc }) + '\n'); X.labels++; }
      }
      if (idx > 0) { applyPlanForPicker(m, ps, C.chosen, C.cands[idx]); X.nonDefault++; S.changed++; prep.changed++; }
      X.decisions++; X.cands += C.cands.length;
    } finally {
      if (arm === 'picker') { m.newBattle = nb; X.sims += sims; if (sims) { console.error(`deploy rule broken: ${sims} battles built at a picker decision`); process.exitCode = 3; } }
      const dt = performance.now() - t0; X.ms += dt; X.msMax = Math.max(X.msMax, dt);
    }
    return;
  }
  if (arm === 'pick') { // phase 1: pick among K legacy candidates by a fixed-seed or live-seed rehearsal, in boss or normal rounds
    const isBoss = !m.wave, want = arg('pick-round', 'boss');
    if ((want === 'boss') === isBoss) {
      const o = pickCandidatesForTraining(m, ps, { K: SK, seed: arg('seed', 'fixed') });
      if (o) { const X = (prep.pick ||= { runs: 0, nonDefault: 0, cands: 0, ms: 0, msMax: 0, none: 0 }); X.runs++; X.cands += o.n; X.ms += o.ms; X.msMax = Math.max(X.msMax, o.ms);
        if (o.idx > 0) { X.nonDefault++; S.changed++; prep.changed++; } else if (o.idx < 0) X.none++; }
    }
    return;
  }
  if (arm === 'peek') { // the peek hill-climb alone (live seeds), from --from with --K candidates
    if (m.round >= SEARCH_FROM) {
      m._oracle ||= { liveSeed: true };
      const t0 = performance.now();
      const o = oracleImproveForTraining(m, ps, { K: SK, budgetMs: Number(arg('budget', Infinity)) });
      const dt = performance.now() - t0;
      if (o) { prep.oracle.runs++; prep.oracle.candidates += o.candidates; prep.oracle.ms = (prep.oracle.ms || 0) + dt; prep.oracle.msMax = Math.max(prep.oracle.msMax || 0, dt);
        if (o.improved) { prep.oracle.improved++; S.changed++; prep.changed++; } }
    }
    return;
  }
  if (arm === 'search') { // the deployable no-peek search on legacy's layout (sampled seeds, the visible wave)
    if (m.round >= SEARCH_FROM) {
      const o = searchImproveForTraining(m, ps, { K: SK, S: SS });
      if (o) { const X = prep.search; X.runs++; X.candidates += o.candidates; X.rehearsals += o.rehearsals; X.ms += o.ms; X.msMax = Math.max(X.msMax, o.ms); if (o.improved) { X.improved++; S.changed++; prep.changed++; } }
    }
    return;
  }
  if (arm === 'legacy') {
    const dry = traitImproveForTraining(m, ps, { battleW: BATTLE_W, apply: false });
    if (dry && m.round >= 9) { S.potential += dry.after - dry.before; prep.potential += dry.after - dry.before; }
  } else {
    const f0 = forecast(m, ps);
    const r = traitImproveForTraining(m, ps, { battleW: arm === 'trait' ? 0 : BATTLE_W });
    if (arm === 'oracle' && m.round >= ORACLE_FROM) {
      m._oracle ||= { liveSeed: true };
      const o = oracleImproveForTraining(m, ps, { K: ORACLE_K });
      if (o) { prep.oracle.runs++; prep.oracle.candidates += o.candidates; if (o.improved) prep.oracle.improved++; }
    }
    g1 = positionalGainForTraining(m, ps);
    if (r && r.moves) {
      prep.changed++; R.changed++; S.changed++;
      const f1 = forecast(m, ps);
      prep.predFin += g1.fin - g0.fin; prep.exactFin += f1 - f0;
    }
  }
  R.fin1 += g1.fin; R.start1 += g1.start; R.battle1 += g1.battle.total; prep.battleAfter += g1.battle.total; prep.predStart += g1.start - g0.start;
};
const env = new SPEnv({ envs: Number(arg('envs', 6)), configs: [cfg], seed: Number(arg('seed0', 9_000_000)), envSeed: Number(arg('seed0', 9_000_000)),
  maxMatches: Number(arg('matches', 24)), humans: arg('humans') !== undefined ? (arg('humans') === 'random' ? 'random' : Number(arg('humans'))) : coopAi ? 0 : 'random', legacyMates: Number(arg('mates', 0)), rehearsal: Number(arg('rehearsal', 0)), bossRehearsal: Number(arg('boss-rehearsal', 3)), obsVersion: 3,
  ...(arg('timing') ? { timing: true } : {}), ...(arg('band-policy') ? { bandPolicy: { share: Number(arg('band-policy')), uniform: false } } : {}),
  ...(arg('recipe') ? { recipeSeats: { recipes: RECIPES, seats: { ai_0: Number(arg('recipe')) } } } : {}),
  mode: econLegacy ? 'legacy' : 'policy', check: true, metrics: true, readyHook, ...(arm === 'bountyA' ? { bountyAware: true } : {}),
  ...(arm === 'equip' ? { equipHook: (m, ps) => equipItemsPicked(m, ps, { K: SK, seed: arg('seed', 'fixed'), rounds: arg('equip-rounds', 'all'), stats: (prep.equip ||= { decisions: 0, changed: 0, cands: 0, ms: 0, byRound: {} }) }) } : {}),
  ...(arg('force-band') ? { forceBand: arg('force-band') } : {}), ...(arg('plan') ? { seatPlanBonds: { ai_0: arg('plan') } } : {}) });
let b = env.reset();
while (b.n) {
  const a = new Int32Array(b.n);
  if (net) for (let k = 0; k < b.n; k++) {
    const { logits } = net.forward(b.obs.subarray(k * b.D, (k + 1) * b.D), b.mask.subarray(k * b.A, (k + 1) * b.A), { value: false });
    let best = 0, bv = -Infinity; for (let j = 0; j < b.A; j++) if (b.mask[k * b.A + j] && logits[j] > bv) { bv = logits[j]; best = j; }
    a[k] = best;
  }
  b = env.step(a);
}
env.close();
const seeds = env.results.map((r) => ({ seed: r.seed, win: r.seats.some((x) => x.win) ? 1 : 0, fa: +(r.faShare || 0).toFixed(4), lp: r.seats.filter((x) => x.rl).reduce((a, x) => a + (x.lp || 0), 0),
  ...(perSeed.get(r.seed) || {}) }));
const T = env.timing ? Object.values(env.timing).reduce((a, t) => ({ rehN: a.rehN + (t.rehN || 0), rehMs: a.rehMs + (t.rehSim || 0) }), { rehN: 0, rehMs: 0 }) : null;
console.log(JSON.stringify({ arm, config: cfg, rehearsal: Number(arg('rehearsal', 0)), timing: T, seconds: +((performance.now() - T0) / 1000).toFixed(1), matches: env.results.length, econ: econLegacy ? 'legacy' : arg('policy'), plan: arg('plan') || null, band: arg('force-band') || null, prep, seeds,
  errors: env.stats.errors, rejected: env.stats.rejected, violations: env.stats.violations, samples: env.errorSamples.slice(0, 3) }));
