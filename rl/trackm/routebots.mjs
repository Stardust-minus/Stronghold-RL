// Track M step 1: scripted route bots on ONE economy executor (legacy bot.js buy loop + a procurement policy object,
// the interface botIoleta uses). Every route bot gets the same layer trade (tradeOnce) and expiring-fund refreshes
// (spendDown) through the gated policy.trade / policy.spendDown flags of the local tree copy; placement is legacy's.
//   nat   national main P (all live members toward T2, merges after) + secondary class bond S, focus = P
//   class Ioleta's 助力 / 坚守 / 精准 elite plan (botIoletaPolicy.js), forced on for any band
//   legacy production legacy bot
import * as bot from '../../../server/match/bot.js';
import { LEGACY_BOT } from '../../../server/match/botStrategyContract.js';
import * as IP from '../../../server/match/botIoletaPolicy.js';

// auto-pick order = recipes-M.json value rank (2026-11-13: full matches + controlled boards)
export const NATIONS = ['lateranoShip', 'victoriaShip', 'kjeragShip', 'kazimierzShip', 'yanShip', 'sargonShip', 'siracusaShip', 'egirShip'];
// community secondary pairs (AI-STRATEGY §2), first live one is used
export const PAIRS = {
  lateranoShip: ['preciShip', 'skillfulShip'], sargonShip: ['preciShip', 'swiftShip', 'skillfulShip'], victoriaShip: ['arcaneShip', 'preciShip'],
  siracusaShip: ['arcaneShip', 'swiftShip'], yanShip: ['arcaneShip', 'preciShip'], kjeragShip: ['arcaneShip', 'preciShip', 'skillfulShip'],
  kazimierzShip: ['raidShip', 'indomShip', 'steadShip'], egirShip: ['raidShip', 'indomShip'],
};
const off = (m) => new Set([...(m.disabledBonds || []), ...(m.staticInactiveBonds || []), ...m.gd.modeInactiveBonds]);
const T2of = (gd, P) => gd.bond(P).thresholds[1];
const thOf = (gd, S) => { const t = gd.bond(S).thresholds; return t[Math.min(1, t.length - 1)]; }; // 精准 / 奥术 3, 突袭 2

/** Pick P (forced or the first live nation in NATIONS order) and S for a seat; stored on ps._nat. */
export function natSetup(m, ps, forced = null) {
  const o = off(m);
  const used = new Set([...m.players.values()].filter((q) => q !== ps && q._nat).map((q) => q._nat.P));
  const P = forced && !o.has(forced) ? forced : (NATIONS.find((b) => !o.has(b) && !used.has(b)) || NATIONS.find((b) => !o.has(b)));
  const S = (PAIRS[P] || []).find((b) => !o.has(b)) || null;
  ps._nat = { P, S, forced, forcedLive: !forced || !o.has(forced), T2: T2of(m.gd, P), thS: S ? thOf(m.gd, S) : 0,
    lowTier: m.gd.bond(P).visibleMembers.filter((id) => m.gd.tierOf(id) <= 4 && m.pool.has(id)).length >= T2of(m.gd, P) };
  ps._envPlanBonds = [{ id: P, priority: 1 }]; // bot.js bondPlan: legacy's focus bond = P (lineup, keepers, trade)
  return ps._nat;
}
const bonds = (gd, id) => gd.chess(id)?.bonds || [];
function held(gd, ps) {
  const normal = new Map(), elites = new Set();
  for (const p of ps.allChess()) { const b = gd.baseIdOf(p.id); if (gd.isGolden(p.id)) elites.add(b); else normal.set(b, (normal.get(b) || 0) + 1); }
  return { normal, elites, has: (b) => normal.has(b) || elites.has(b) };
}
export function natDistinct(gd, ps) {
  const P = ps._nat?.P; if (!P) return 0;
  const h = held(gd, ps); let n = 0;
  for (const b of new Set([...h.normal.keys(), ...h.elites])) if (bonds(gd, b).includes(P)) n++;
  return n;
}
function sOwned(gd, ps) {
  const S = ps._nat?.S; if (!S) return 0;
  const h = held(gd, ps); let n = 0;
  for (const b of new Set([...h.normal.keys(), ...h.elites])) if (bonds(gd, b).includes(S)) n++;
  return n;
}
const natTarget = (m, ps, id) => {
  const gd = ps.gd || m.gd, N = ps._nat; if (!N) return null;
  const base = gd.baseIdOf(id), bs = bonds(gd, base);
  if (bs.includes(N.P)) return { base, bond: N.P, elite: true, stage: gd.tierOf(base), copies: gd.mergeCount(base) || 3 };
  if (N.S && bs.includes(N.S) && (sOwned(gd, ps) < N.thS || held(gd, ps).has(base))) return { base, bond: N.S, elite: true, stage: gd.tierOf(base), copies: gd.mergeCount(base) || 3 };
  return null;
};
export const NAT_POLICY = Object.freeze({
  trade: true, spendDown: true,
  target: natTarget,
  core(m, ps, id) { const t = natTarget(m, ps, id); return !!t && t.stage <= ps.shop.level; },
  buyBonus(m, ps, id) {
    const gd = ps.gd || m.gd, N = ps._nat; if (!N) return 0;
    const rec = gd.chess(id); if (!rec) return 0;
    const base = gd.baseIdOf(id), h = held(gd, ps), n = h.normal.get(base) || 0;
    if (!rec.isGolden && h.elites.has(base)) return -160; // a copy of an owned elite is dead weight
    const bs = bonds(gd, base);
    if (bs.includes(N.P)) {
      if (!h.has(base)) { const d = natDistinct(gd, ps); return d + 1 >= N.T2 ? 95 : d < N.T2 ? 70 : 10; }
      return n >= 2 ? 25 : 8; // merge progress on a member (legacy adds its own merge value)
    }
    if (N.S && bs.includes(N.S) && !h.has(base) && sOwned(gd, ps) < N.thS) return 30;
    return 0;
  },
  canLevel(m, ps) {
    const gd = ps.gd || m.gd, N = ps._nat, lv = ps.shop.level, r = m.round;
    if (lv >= gd.maxShopLevel || ps.funds < Math.max(0, ps.shop.upgradePrice)) return false;
    if (ps.shop.upgradePrice <= 0) return true;
    const t2 = N && natDistinct(gd, ps) >= N.T2;
    const want = lv < 2 ? r >= 3 : lv < 3 ? r >= 5 : lv < 4 ? r >= 7
      : lv < 5 ? (N?.lowTier ? (t2 ? r >= 9 : r >= 11) : r >= 9) : (t2 ? r >= 11 : r >= 12);
    return want;
  },
  searchTargets(m, ps) {
    const gd = ps.gd || m.gd, N = ps._nat; if (!N) return [];
    const h = held(gd, ps), out = [];
    if (natDistinct(gd, ps) < N.T2) for (const b of gd.bond(N.P).visibleMembers) if (!h.has(b) && gd.tierOf(b) <= ps.shop.level && (ps.poolOf?.(b) || m.pool).left(b) > 0) out.push({ base: b });
    return out;
  },
});
export const CLASS_POLICY = Object.freeze({
  trade: true, spendDown: true,
  buyBonus: IP.ioletaBuyBonus, core: IP.ioletaCore, canLevel: IP.ioletaCanLevel, target: IP.ioletaTarget, searchTargets: IP.ioletaSearchTargets,
});
const NAT_PREP0 = bot.createBotPrepSteps({}, NAT_POLICY);
const CLASS_PREP0 = bot.createBotPrepSteps({}, CLASS_POLICY);
/** Prep step sets; `oracle` = { K, seed, rounds } runs stackOracleForTraining at Ready (teacher-only placement, the
 * same for every route bot; normal rounds: most layers with no extra leaks, boss rounds: outcome). */
function prepSets(oracle) {
  if (!oracle) return { nat: NAT_PREP0, class: CLASS_PREP0, legacy: LEGACY_BOT };
  const ready = (m) => (ps) => {
    const boss = m.round === m.gd.bossRound || m.round === m.gd.hiddenRound;
    if (oracle.rounds === 'all' || (oracle.rounds === 'boss') === boss) {
      try { if (typeof bot.stackOracleForTraining !== 'function') throw new Error('no stackOracleForTraining in this tree'); bot.stackOracleForTraining(m, ps, { K: oracle.K, perturb: true, seed: oracle.seed || 'fixed', score: 'layers' }); } catch { /* keep legacy's layout */ }
    }
    try { ps.setReady(true); } catch { /* refused */ }
  };
  // createBotPrepSteps' ready action gets (ps) only: bind the match through a per-call closure
  const mk = (policy) => ({
    *prepBegin(m, ps, opts) { return yield* bot.createBotPrepSteps({ ready: ready(m) }, policy).prepBegin(m, ps, opts); },
    *prepEnd(m, ps, job) { return yield* bot.createBotPrepSteps({ ready: ready(m) }, policy).prepEnd(m, ps, job); },
  });
  return { nat: mk(NAT_POLICY), class: mk(CLASS_POLICY), legacy: mk(null) };
}

/** Per-seat strategy: seats = { playerId: 'legacy' | 'class' | 'nat' | 'nat:<bondId>' }; anything unlisted is legacy. */
export function seatStrategy(seats, { onReady = null, setRehearsal = null, oracle = null, oracleSeats = null } = {}) {
  const kind = (ps) => seats[ps.playerId] || 'legacy';
  const P0 = prepSets(null), PO = prepSets(oracle);
  const sets = (ps) => (oracle && (!oracleSeats || oracleSeats.includes(ps.playerId)) ? PO : P0);
  return {
    id: 'track-m',
    pickBand(m, ps) {
      const k = kind(ps);
      if (k === 'classb' || k.startsWith('natb')) {
        if (k.startsWith('natb') && !ps._nat) natSetup(m, ps, k.includes(':') ? k.split(':')[1] : null);
        return LEGACY_BOT.pickBand(m, ps); // route bot with the legacy band picker (band LP / buffs as legacy)
      }
      if (k === 'class') {
        const id = 'band_ioleta';
        if (m.gd.bandIds().includes(id) && !m.bandTaken(id, ps.playerId)) return id;
      } else if (k.startsWith('nat')) {
        const N = ps._nat || natSetup(m, ps, k.includes(':') ? k.split(':')[1] : null);
        const ok = m.gd.bandIds().filter((b) => m.gd.bandAllowed(b) && !m.bandTaken(b, ps.playerId) && m.gd.bandBondIds(b).includes(N.P));
        if (ok.length) return ok[0];
      }
      return LEGACY_BOT.pickBand(m, ps);
    },
    pickCard: LEGACY_BOT.pickCard,
    bountyKillChance: LEGACY_BOT.bountyKillChance,
    *prepBegin(m, ps, opts) {
      if (setRehearsal) setRehearsal(m);
      const k = kind(ps);
      if (k.startsWith('class')) { ps._ioletaForce = true; if (!ps.alive || ps.ready) return null; IP.refreshIoletaPlan(m, ps); return yield* sets(ps).class.prepBegin(m, ps, opts); }
      if (k.startsWith('nat')) { if (!ps._nat) natSetup(m, ps, k.includes(':') ? k.split(':')[1] : null); return yield* sets(ps).nat.prepBegin(m, ps, opts); }
      return yield* sets(ps).legacy.prepBegin(m, ps, opts);
    },
    *prepEnd(m, ps, job) {
      if (setRehearsal) setRehearsal(m);
      const k = kind(ps);
      const S = sets(ps), r = yield* (k.startsWith('class') ? S.class : k.startsWith('nat') ? S.nat : S.legacy).prepEnd(m, ps, job);
      if (onReady) onReady(m, ps);
      return r;
    },
  };
}

/** Teacher procurement for env.mjs teacher mode (the PROCURE_ONLY slot): the nat hunt, no arrange / equip. Call
 * natSetup(m, ps, P) once (P = the seat's plan nation, e.g. from the NAT plan block) before the first prep. */
const noopT = () => {};
// eslint-disable-next-line require-yield -- an empty step generator
const noneT = function* () { return null; };
export const NAT_PROCURE_ONLY = bot.createBotPrepSteps({ arrangeSteps: noneT, equipItems: noopT }, NAT_POLICY);
