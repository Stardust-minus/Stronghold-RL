// tools/rl/env.mjs — batched whole-match RL environment (route B, plan §8). One Node process runs `envs` matches in
// virtual time (Match + VirtualScheduler, as tools/botbench.mjs). The policy owns, per RL seat, the prep economy
// (buy / sell / refresh / level / freeze / reward / personal choice) and the 机变 draft card; the unchanged legacy
// planner places the lineup (rehearsal 0 or 1 in training), takes leftover rewards, resolves temp and readies.
// A match with a decision pending runs no scheduler callback, so its virtual clock stands still until the policy acts.
import { performance } from 'node:perf_hooks';
import { Names, summarize, actionLabel } from './describe.mjs';
import { Match } from '../../server/match/Match.js';
import { VirtualScheduler } from '../../server/match/scheduler.js';
import { getData } from '../../server/data.js';
import { defineBotStrategy, LEGACY_BOT } from '../../server/match/botStrategyContract.js';
import { createBotPrepSteps, runSteps } from '../../server/match/bot.js';
import { collectViolations } from '../../server/match/invariants.js';
import { snapshotMatch, restoreMatch } from '../../server/match/snapshot.js';
import { createRngFromState } from '../../server/sim/rng.js';
import { Encoder, ACT, KIND, SHOP, OWNED, REWARDS, PERSONAL, CARDS, BANDS, DIRS, tileRC, tileIndex, humanBonds, humanPairs, chessInBonds, mainCoreBondOf, rulesOf, bandConflicts } from './spec.mjs';
import { beginSeatPlan, endSeatPlan, seatPlanOf, seatPlanRerolls } from '../../server/match/seatPlan.js';
import { pieceDir } from '../../server/match/board.js';
import { SPPolicy2 } from './infer2.mjs';

// Legacy prep with its procurement switched off — rewards too (the policy picks them, under the 让盟约 mask): a personal
// choice left over, arrange, equip, final arrange (+ rehearsal); prepEnd (legacy) applies the rehearsed plan, resolves
// temp, frees a hand slot and readies.
const noop = () => {};
// eslint-disable-next-line require-yield -- an empty step generator: the legacy step it replaces does nothing
const none = function* () { return null; };
const MAX_ECON = 40; // v4c: economy decisions per prep at most (MAX_PREP_ACTIONS of version 2)
export const LAYOUT_ONLY = createBotPrepSteps({ takeOffers: noop, sellJunk: noop, buyLoopSteps: none, levelUp: noop,
  advisorSpendSteps: none, spendDown: noop, maybeFreeze: noop });
// Legacy procurement only (the teacher): everything before the first arrange.
export const PROCURE_ONLY = createBotPrepSteps({ arrangeSteps: none, equipItems: noop });

// Rewards are mostly the team's (the owner's call of 2026-11-11: no learning to sacrifice a teammate): per decision,
// round × (team share of rounds reached + own share) − lp × (team share of LP lost + own share) / starting LP; a win +win.
// fa: the team's share of the round-14 Final Assault leader pool (1 − hp / maxHp of that SharedBossPool: 0 when untouched,
// 1 for a win), paid to every policy seat at the match end (ppo2: a signal between "reached the Final Assault and lost
// badly" and "nearly cleared it"). Every term can be overridden per environment (option `reward`).
export const REWARD = Object.freeze({ round: 0.1, lp: 1.0, win: 2.0, team: 0.75, fa: 0 });

/**
 * Bond potential (盟约激活 / 盟约叠层, the owner's call of 11-12): raw terms of a seat's own bond strip (privateView bonds,
 * which count the board as placed). T = Σ active tier / max tier; S = Σ over ACTIVE stackable bonds (noStack false and
 * per-stack parameters) of min(layers / next layer milestone, 1), the reference being STACK_REF when a bond has no
 * milestone left. Layers banked on an inactive bond do nothing in battle, so they do not count.
 */
export const STACK_REF = 40;
/** Bond log (opt-in, o.bondLog): a seat's active bonds [bondId, tier, layers, count], the main one (most units, then highest
 * tier), whether it is a bond the seat's band binds (null: the band binds none), the number of distinct active bonds. */
export function bondRow(gd, ps) {
  const list = (ps.privateView().bonds || []).filter((b) => b.active && !b.off);
  let main = null;
  for (const b of list) if (!main || (b.count || 0) > (main.count || 0) || ((b.count || 0) === (main.count || 0) && (b.tier || 0) > (main.tier || 0))) main = b;
  const bound = ps.bandId && gd.bandBondIds ? gd.bandBondIds(ps.bandId) : [];
  return { active: list.map((b) => [b.bondId, b.tier || 0, b.layers || 0, b.count || 0]), main: main ? main.bondId : null,
    mainIsBand: bound.length ? !!(main && bound.includes(main.bondId)) : null, distinct: list.length,
    // contention (coop4ai): operator base ids held (board, hand, temp) and the normal (not yet merged) copies among them
    ...(() => { const ch = ps.allChess(); return { ops: [...new Set(ch.map((p) => gd.baseIdOf(p.id)))], unmerged: ch.filter((p) => !gd.isGolden(p.id)).length, elites: ch.filter((p) => gd.isGolden(p.id)).length }; })() };
}
export function bondTerms(gd, bonds, uncapped = false) {
  let T = 0, S = 0;
  for (const b of bonds || []) {
    const max = Array.isArray(b.thresholds) && b.thresholds.length ? b.thresholds.length : 1;
    const tier = b.active ? Math.min(b.tier || 0, max) : 0;
    T += tier / max;
    if (!tier) continue;
    const rec = gd.bond(b.bondId);
    if (!rec || rec.noStack || !(rec.perStackParams || []).length) continue;
    const layers = Number(b.layers) || 0;
    if (uncapped) { S += layers / 100; continue; } // R2 (11-12): uncapped, Σ stacking-bond layers / 100
    const next = (rec.layerMilestones || []).map((x) => Number(x.layer)).filter((x) => x > layers).sort((x, y) => x - y)[0];
    const ref = next || ((rec.layerMilestones || []).length ? 0 : STACK_REF);
    S += ref ? Math.min(layers / ref, 1) : 1; // past the last milestone: full
  }
  return { T, S };
}

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function parseConfig(s) {
  const [mode, diff] = String(s).split(':');
  const solo = mode === 'solo';
  return { name: s, mode: solo ? 'solo' : 'coop', players: solo ? 1 : Math.max(1, Math.min(4, parseInt(mode.replace('coop', ''), 10) || 4)), // coop2l / coop4ai: the count before the suffix
    difficulty: String(diff || 'NORMAL').toUpperCase() };
}

export class SPEnv {
  /**
   * @param {object} o
   * @param {number} [o.envs] matches run side by side
   * @param {Array<string|{name:string,w:number}>} [o.configs] e.g. ['solo:NORMAL', {name:'coop4:HARD', w:2}]
   * @param {number} [o.seed] first match seed (each new match takes the next)
   * @param {number} [o.rehearsal] legacy layout rehearsal candidates (0 or 1 for training)
   * @param {number} [o.legacyMates] in co-op: chance that a seat other than seat 0 is a legacy bot
   * @param {'policy'|'teacher'|'legacy'} [o.mode] policy: decisions come from step(); teacher: legacy procurement with
   *   its calls recorded as env actions (no decisions returned); legacy: no RL seat at all (baseline)
   * @param {boolean} [o.check] collectViolations at every match end
   * @param {'random'|number} [o.humans] co-op seats other than seat 0 marked human (让盟约 training): 'random' = 0..n−1
   * @param {number} [o.humanRL] chance a marked human is driven by the policy instead of legacy (default 0: the policy
   *   drives AI seats only, as in a real room, where a human seat under AI 托管 is legacy's)
   * @param {object} [o.rules] { yieldMain, yieldMerge } (spec.mjs RULES, both on by default)
   * @param {object} [o.experimental] the room's experimental options (Match opts.experimental: disableSharedPool, poolPlan)
   * @param {boolean} [o.metrics] rule metrics for every AI seat and the humans' bond tiers (evaluation; off in training:
   *   they rebuild views on every AI purchase); with `scoutCheck` the direct 让合成 read is compared with the scout
   * @param {boolean} [o.enforce] teacher mode: a legacy call a rule forbids fails instead of running (legacy then does
   *   something else) — the rules' effect at legacy's strength, before any policy is trained
   */
  constructor(o = {}) {
    this.data = o.data || getData({ log: { warn() {}, error() {}, info() {} } });
    this.envs = Math.max(1, o.envs | 0 || 8);
    this.setConfigs(o.configs || ['solo:NORMAL']);
    this.seed = Number.isFinite(o.seed) ? o.seed : 1;
    this.rehearsal = o.rehearsal ?? 0;
    // A4: legacy layout rehearsal on the deciding fights only (Final Assault, Hidden Core), `rehearsal` elsewhere; null = off.
    // m.botRehearsal is read when a prep arranges (bot.js arrangeSteps / createRehearsalSteps); under the VirtualScheduler
    // no worker rehearsal reads it earlier (canUseWorkerRehearsal is false), so switching it at prepBegin is exact.
    this.bossRehearsal = Number.isInteger(o.bossRehearsal) && o.bossRehearsal >= 0 ? Math.min(o.bossRehearsal, 8) : null;
    // eval realism (11-12): the legacy (non-RL) seats rehearse this many layouts at normal rounds, as in production; RL seats keep o.rehearsal
    // deploy information set (11-13, the owner): the RL seats' obs, masks and legacy placement use only what a player sees
    this.visibleOnly = !!o.visibleOnly; // A1 humans' pairs via the scout view (board) only, A3 hidden boss id from its round on
    // strict preview (kept off: the owner rules enemy info, HP and the 机变 kill chance acceptable): B1 legacy fieldModel from the
    // preview (routes by gate, no spawn mods, no round HP scale), A2 bounty kill chance 0, A4 no spawn time
    this.strictPreview = !!o.strictPreview;
    // test only (infoguard.mjs): count reads of hidden match state while an RL seat's decision or placement runs
    this.infoGuard = o.infoGuard ? { routeIndex: 0, mods: 0, hpMul: 0, otherHand: 0, hiddenBoss: 0, scoutReads: 0, prepHandReads: 0, samples: [] } : null;
    this.mateRehearsal = Number.isInteger(o.mateRehearsal) && o.mateRehearsal >= 0 ? Math.min(o.mateRehearsal, 8) : null;
    this.legacyMates = o.legacyMates ?? 0;
    this.mode = o.mode || 'policy';
    this.check = !!o.check;
    this.humans = o.humans ?? 'random';
    this.humanRL = o.humanRL ?? 0;
    this.rules = rulesOf({ ...(o.yieldBonds === false ? { yieldMain: false } : {}), ...(o.rules || {}) });
    this.experimental = o.experimental || null;
    this.enforce = !!o.enforce;
    this.metrics = !!o.metrics;
    // potential-based shaping r = γ·Φ(s') − Φ(s), Φ = wTier·T + wStack·S (bondTerms), Φ(terminal) = 0; own seat only
    this.shaping = o.shaping && (o.shaping.wTier || o.shaping.wStack)
      ? { gamma: Number(o.shaping.gamma), wTier: Number(o.shaping.wTier) || 0, wStack: Number(o.shaping.wStack) || 0, uncapped: !!o.shaping.uncapped } : null;
    // reward version 2 (owner 11-12): team clear, team / own Final Assault damage, own leaks, 联防 saves, team-weighted LP
    // against a band-independent L_ref, elimination; optional direct layer reward (layer per stacking-bond layer gained)
    this.rewardV = Number(o.rewardV) || 1;
    this.R2 = Object.freeze({ win: 3, faTeam: 1.5, faOwn: 0.5, leak: 0.5, save: 0.25, lp: 1.0, team: 0.75, elim: 0.5, layer: 0, leakCap: false, ...(o.r2 || {}) });
    this.termWidth = this.rewardV === 2 ? 18 : 9;
    this.bondLog = !!o.bondLog; // bond log at the battles of rounds 6 / 9 / 12 and at the last battle (RL seats)
    if (this.shaping && !(this.shaping.gamma > 0 && this.shaping.gamma <= 1)) throw new Error('shaping.gamma must be the GAE gamma');
    this.phiSamples = o.phiSamples ? [] : null; // diagnostics: raw bond terms of every seat's final board
    // replay viewer: readable states of these seats' decisions (describe()), and the teacher's own per-prep log
    this.describeSeats = new Set(o.describeSeats || []);
    // diagnostics (ppo3 cost model): CPU ms per round x {prep, battle, teacherProcurement, encode}, decisions per round
    this.timing = o.timing ? {} : null;
    // S1 (ppo3): late-game starts. snapCollect { rounds, prob, configs (difficulty list), cap }: prep-end checkpoints
    // (server/match/snapshot.js, after legacy placement) of these matches go into a per-config pool, with the env's own
    // seat roles; snapStart { frac, configs }: that share of new matches of those difficulties restores a random pooled
    // checkpoint (restoreMatch replay: the checkpoint's round is fought again) instead of starting at round 1.
    // snapQueue: explicit checkpoints to start from, in order (tests).
    this.snapCollect = o.snapCollect ? { rounds: [9, 10, 11, 12, 13], prob: 0.25, configs: ['ABYSS'], cap: 64, ...o.snapCollect } : null;
    this.snapStart = o.snapStart ? { frac: 0.4, configs: ['ABYSS'], ...o.snapStart } : null;
    this.snaps = new Map(); // config name -> [{ doc, roles, round, seed }]
    this.snapQueue = o.snapQueue ? o.snapQueue.slice() : null;
    this.snapStats = { collected: 0, refused: 0, started: 0, restoreFailed: 0 };
    // S3 (ppo3): explicit seeds for the next matches (prioritized level replay; a seed fixes config and roles)
    this.seedQueue = [];
    // band choice (ppo3): bandPolicy { share: RL seats whose band the policy picks (the rest legacy's random pick),
    // planProb: chance a human proxy without a band-tied bond announces a random core 主盟约 }; null = legacy picks all
    this.obsVersion = o.obsVersion ?? 3; // 2: the ppo1/ppo2 layout and 60 actions; 3: S5 blocks + band choice; 4: full control
    // version 4 (the owner's call of 11-12): the policy takes every decision of its seat — economy, lineup, every tile and
    // facing, equipment, personal choices, temp / hand — and END is the seat's Ready; legacy never arranges a policy seat.
    // seatPlan (deploy semantics: one round trip per prep, server/match/seatPlan.js): the seat plans its whole prep from
    // the other seats as they stood at its prep start, its rerolls show its own refresh table, its effects draw its own
    // meta stream, and the draws a device cannot reproduce are counted stop points. A boss-round pair of two policy seats
    // plans L first; R sees L's planned board.
    this.full = this.obsVersion >= 4;
    this.seatPlan = this.full && o.seatPlan !== false;
    this.seatPlanParts = o.seatPlanParts || {}; // ablations: { table, offers, meta } (all on by default)
    // placement labels (version 4 teacher): a prep's samples are held until its end, then each MOVE sample gets the set of
    // owned rows legacy still moves in this prep (u32 `aux`, for a set-valued loss) and each tile sample the selected
    // piece's FINAL tile / facing (or the hand) when that is legal there — legacy's staging moves leave the targets
    this.placementLabels = !!o.placementLabels && this.full;
    this.labelStats = { moves: 0, tiles: 0, relabeled: 0, keptIllegal: 0 };
    // v4c teacher (version 4 teacher mode): the economy and the 机变 cards from a trained version 2 policy (ppo2, plain-JS
    // forward on version 2 observations, sampled at `temp`), legacy for everything else (lineup, tiles, equipment, the
    // rest of its prep); econPolicy { base, temp }
    this.econ = o.econPolicy && this.full ? { pol: SPPolicy2.load(o.econPolicy.base), temp: Number(o.econPolicy.temp ?? 1), rng: mulberry32(((o.envSeed ?? 12345) ^ 0xEC0) >>> 0), enc: null } : null;
    this.econStats = { decisions: 0, refused: 0, ends: 0, capped: 0, cards: 0 };
    this.planStats = { plans: 0, stops: { pool: 0, offer: 0, table: 0, foreign: 0 }, stopPreps: { pool: 0, offer: 0, table: 0, foreign: 0, any: 0 },
      rejected: 0, readyFallback: 0, bossDeferred: 0 };
    // uniform: the model-AI seats the policy does not pick for open with a uniformly random allowed band (robustness:
    // the policy sees every band), not legacy's weighted pick
    this.bandPolicy = o.bandPolicy && this.obsVersion >= 3 ? { share: 0.6, planProb: 0.5, uniform: true, ...o.bandPolicy } : null;
    this.bandStats = { decisions: 0, fallback: 0, humanConflict: 0, aiSameBond: 0, policySeats: 0, legacySeats: 0 };
    // Step 0 (band strength table): seat ai_0's legacy pick returns this band; teammates re-draw instead of taking it
    this.forceBand = o.forceBand || null;
    this.R = Object.freeze({ ...REWARD, ...(o.reward || {}) });
    // per-difficulty reward overrides on top of this.R (o.rewardBy { ABYSS: { round: 0, win: 1 } }); a seat keeps its match's weights
    this.RB = Object.fromEntries(Object.entries(o.rewardBy || {}).map(([d, r]) => [String(d).toUpperCase(), Object.freeze({ ...this.R, ...r })]));
    this.G0 = null; // the global block's offset (the own-LP guard), set with the encoder
    // own-LP obs field override (11-12 probe / --lp-obs): null = lp / lp0 (normal), 'raw' = raw LP (what the _seat bug fed
    // policy-band seats), a number k = (lp / lp0) · k; applied after the guard checked the normal value
    this.lpObs = o.lpObs === 'raw' ? 'raw' : Number.isFinite(Number(o.lpObs)) && o.lpObs !== null && o.lpObs !== undefined && o.lpObs !== '' ? Number(o.lpObs) : null;
    this.kindSeat = !!o.kindSeat; // cross-play evals: kinds[k] += 16 · seat index (ai_N), decoded by eval_cross.py
    this.tripLog = !!o.tripLog; // with metrics: every round trip's details in the match result (diagnostics)
    this.scoutCheck = !!o.scoutCheck;
    this.maxMatches = o.maxMatches ?? Infinity; // stop opening new matches after this many (smoke tests)
    this.until = o.until ?? Infinity; // performance.now() after which no new match opens (benchmarks)
    this.envSeed = (o.envSeed ?? 12345) >>> 0;
    this.rng = mulberry32(this.envSeed); // only the config pick of a match without a seed-derived stream
    this.recorder = o.recorder || null; // teacher mode: { sample(kind, traj, obs, mask, action, reward), term(t) }
    this.encoder = null;
    this.trajSeq = 0;
    this.started = 0;
    this.terms = [];
    this.results = [];
    this.stats = { decisions: 0, rejected: 0, forcedEnd: 0, matches: 0, errors: 0, failed: 0, violations: 0, callbacks: 0,
      teacher: { calls: {}, illegal: {}, unrepresented: {}, overCap: 0, preps: 0, blocked: {} }, scoutChecks: 0, scoutMismatch: 0 };
    this.errorSamples = [];
    this.slots = new Array(this.envs).fill(null);
    for (let i = 0; i < this.envs; i++) this.slots[i] = this._open(i);
    this.D = this.encoder.dim; this.A = this.encoder.A;
    this._buf(64);
  }

  setConfigs(list) {
    this.configs = list.map((c) => (typeof c === 'string' ? { ...parseConfig(c), w: 1 } : { ...parseConfig(c.name), w: c.w ?? 1 }));
    this.cw = this.configs.reduce((s, c) => s + c.w, 0);
  }

  spec() { return { ...this.encoder.spec(), reward: this.R, configs: this.configs.map((c) => c.name), termWidth: this.termWidth,
    ...(this.rewardV === 2 ? { rewardV: 2, r2: this.R2 } : {}) }; }

  _buf(n) {
    this.cap = n;
    this.obs = new Float32Array(n * this.D); this.mask = new Uint8Array(n * this.A);
    this.ids = new Int32Array(n); this.kinds = new Int32Array(n); this.rew = new Float32Array(n);
  }

  _pick(rng = this.rng) {
    let x = rng() * this.cw;
    for (let i = 0; i < this.configs.length; i++) { x -= this.configs[i].w; if (x < 0) return i; }
    return this.configs.length - 1;
  }

  _open(i) {
    if (this.started >= this.maxMatches || performance.now() > this.until) return null;
    let seed = this.seedQueue.length ? this.seedQueue.shift() : this.seed + this.started;
    this.started++;
    // config and seat roles come from the seed itself (not from the order matches open in), so a seed is the same
    // match under every mode — a policy evaluation and its legacy baseline see identical configs and roles
    const rng = mulberry32((this.envSeed ^ Math.imul(seed | 0, 0x9E3779B1)) >>> 0);
    rng(); rng();
    let ci = this._pick(rng), cfg = this.configs[ci];
    // S1: a late-game start (a pooled checkpoint of this config, or the next one of the test queue)
    let snap = null;
    if (this.snapQueue) {
      snap = this.snapQueue.shift() || null;
      if (snap) { ci = Math.max(0, this.configs.findIndex((c) => c.name === snap.config)); cfg = this.configs[ci]; }
    } else if (this.snapStart && this.snapStart.configs.includes(cfg.difficulty) && rng() < this.snapStart.frac) {
      const pool = this.snaps.get(cfg.name);
      if (pool && pool.length) snap = pool[Math.floor(rng() * pool.length)];
    }
    if (snap) seed = snap.seed;
    const sched = new VirtualScheduler();
    const seats = [];
    for (let s = 0; s < cfg.players; s++) seats.push({ seat: s, playerId: `ai_${s}`, name: `AI-${s + 1}`, isBot: true, connected: true });
    const slot = { i, ci, cfg, seed, sched, pending: [], seat: new Map(), rl: new Set(), humans: new Set(), role: new Map(), ended: false,
      summary: null, failed: null, cb: 0, m: null, seenRound: 0, yield: {}, humanTier: new Map(), trips: [],
      names: this.describeSeats.size ? new Names() : null, replay: [] };
    // seat roles; the env RNG is drawn the same way in every mode, so a seed gets the same roles under legacy / teacher / policy
    const mates = []; for (let s = 1; s < cfg.players; s++) mates.push(s);
    const legacyMate = new Map(mates.map((s) => [s, rng() < this.legacyMates]));
    for (let j = mates.length - 1; j > 0; j--) { const k = Math.floor(rng() * (j + 1)); [mates[j], mates[k]] = [mates[k], mates[j]]; }
    const nH = this.humans === 'random' ? Math.floor(rng() * cfg.players) : Math.min(Number(this.humans) || 0, mates.length);
    const allAI = String(cfg.name).split(':')[0].endsWith('ai');
    if (allAI) for (const k of legacyMate.keys()) legacyMate.set(k, false);
    // coop*l configs (11-12, the target): one RL seat (ai_0); every mate that is not a human proxy is a legacy rule-AI
    if (String(cfg.name).split(':')[0].endsWith('l')) for (const k of legacyMate.keys()) legacyMate.set(k, true);
    // coop*h<N> configs (11-12, product scenarios): exactly N human-proxy mates, every other seat an RL seat (no legacy mates)
    const hN = /h(\d)$/.exec(String(cfg.name).split(':')[0]);
    if (hN) for (const k of legacyMate.keys()) legacyMate.set(k, false);
    const humanSeats = new Set(allAI ? [] : hN ? mates.slice(0, Math.min(Number(hN[1]), mates.length)) : mates.slice(0, nH));
    const humanRL = new Map(mates.map((s) => [s, rng() < this.humanRL]));
    for (let s = 0; s < cfg.players; s++) {
      const pid = `ai_${s}`;
      const role = humanSeats.has(s) ? (humanRL.get(s) ? 'human-rl' : 'human-legacy') : s > 0 && legacyMate.get(s) ? 'legacy' : 'rl';
      if (humanSeats.has(s)) slot.humans.add(pid);
      slot.role.set(pid, this.mode === 'legacy' ? (role.startsWith('human') ? 'human-legacy' : 'legacy') : role);
      if (this.mode !== 'legacy' && (role === 'rl' || role === 'human-rl')) slot.rl.add(pid);
    }
    slot.deferredBand = []; slot.bandSeats = new Set(); slot.announced = new Set(); slot.deferredPrep = [];
    if (this.bandPolicy && !snap) for (const pid of slot.rl) {
      if (slot.humans.has(pid)) continue;
      if (rng() < this.bandPolicy.share) { slot.bandSeats.add(pid); this.bandStats.policySeats++; } else this.bandStats.legacySeats++;
    }
    if (snap) { // the checkpoint's own seat roles (human marks, legacy mates) and its round
      slot.humans = new Set(snap.roles.humans); slot.rl = new Set(this.mode === 'legacy' ? [] : snap.roles.rl);
      slot.role = new Map(snap.roles.role.map(([pid, r]) => [pid, this.mode === 'legacy' ? (r.startsWith('human') ? 'human-legacy' : 'legacy') : r]));
      slot.fromSnapshot = snap.round;
    }
    const env = this;
    const log = { info() {}, debug() {}, warn() {}, error: (...a) => { env.stats.errors++; if (env.errorSamples.length < 20) env.errorSamples.push(`[seed ${seed}] ${a.map(String).join(' ').slice(0, 400)}`); } };
    const strategy = defineBotStrategy({ id: 'rl-env',
      decide(m, ps, d) {
        // a human seat under AI 托管 (isBot false) stays legacy's, as every seat the env does not give the policy
        if (d.kind === 'band' && env.bandPolicy && ps.isBot && slot.bandSeats?.has(ps.playerId)) {
          // the model AI picks after every human(-proxy) seat (aiPicksLast semantics, the owner's call of 11-12)
          slot.deferredBand.push({ slot, ps, d, kind: 'band', actions: 0, freezes: 0, own: null, owned: null, cache: {} });
          return true;
        }
        if (!ps.isBot || !slot.rl.has(ps.playerId) || (d.kind !== 'prep' && d.kind !== 'card')) return false;
        if (env.mode === 'teacher') return env._teach(slot, ps, d);
        const dec = { slot, ps, d, kind: d.kind, actions: 0, freezes: 0, own: null, owned: null,
          startUids: new Set(ps.allChess().map((p) => p.uid)), soldBases: new Set(), cache: {},
          sel: -1, selUid: null, pubSnap: null, mateSnap: undefined, planned: false, opened: false };
        if (env.full && d.kind === 'prep' && env._bossWaitsForL(slot, ps)) { slot.deferredPrep.push(dec); env.planStats.bossDeferred++; return true; }
        slot.pending.push(dec);
        return true;
      },
      pickBand(m, ps) {
        if (env.bandPolicy?.uniform && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId) && !slot.bandSeats?.has(ps.playerId)) {
          const ok = m.gd.bandIds().filter((id) => m.gd.bandAllowed(id) && !m.bandTaken(id, ps.playerId));
          if (ok.length) return ok[Math.floor(slot.collectRng() * ok.length)];
        }
        const f = env.forceBand;
        if (f && ps.playerId === 'ai_0' && m.gd.bandAllowed(f) && !m.bandTaken(f, ps.playerId)) return f;
        let id = LEGACY_BOT.pickBand(m, ps);
        for (let k = 0; f && id === f && k < 8; k++) id = LEGACY_BOT.pickBand(m, ps);
        return id;
      },
      *prepBegin(m, ps, o) {
        // the rehearsal count is set for THIS seat before every resumption of its prep steps (per-seat rehearsal, mateRehearsal)
        return yield* env._perSeat(m, ps, slot, (slot.rl.has(ps.playerId) ? LAYOUT_ONLY : LEGACY_BOT).prepBegin(m, ps, o));
      },
      prepEnd(m, ps, job) {
        const g = LEGACY_BOT.prepEnd(m, ps, job); // the base hook as before, only wrapped
        return g && typeof g.next === 'function' && (env.bossRehearsal !== null || env.mateRehearsal !== null || env.infoGuard) ? env._perSeat(m, ps, slot, g) : g;
      },
    });
    const collect = this.snapCollect && this.snapCollect.configs.includes(cfg.difficulty) ? this.snapCollect : null;
    const onCheckpoint = collect ? (mm, reason) => {
      if (reason !== 'prep-end' || !collect.rounds.includes(mm.round) || slot.collectRng() >= collect.prob) return;
      const doc = snapshotMatch(mm);
      if (!doc) { this.snapStats.refused++; return; }
      const pool = this.snaps.get(cfg.name) || []; this.snaps.set(cfg.name, pool);
      const entry = { config: cfg.name, seed: doc.seed, round: mm.round, doc: JSON.parse(JSON.stringify(doc)),
        roles: { humans: [...slot.humans], rl: [...slot.rl], role: [...slot.role.entries()] } };
      if (pool.length < collect.cap) pool.push(entry); else pool[Math.floor(slot.collectRng() * pool.length)] = entry;
      this.snapStats.collected++;
    } : null;
    slot.collectRng = mulberry32((seed ^ 0x51A9) >>> 0);
    slot.m = new Match({ roomCode: 'RL', mode: snap ? snap.doc.mode : cfg.mode, difficulty: snap ? snap.doc.difficulty : cfg.difficulty,
      seats: snap ? snap.doc.seats : seats, seed, data: this.data, scheduler: sched, ...(onCheckpoint ? { onCheckpoint } : {}),
      ...(this.experimental && cfg.mode !== 'solo' ? { experimental: { revivalEnabled: false, disableSharedPool: false, ...this.experimental } } : {}),
      botRehearsal: this.rehearsal, botStrategy: strategy, log, send: () => true, broadcast: () => {},
      onEnd: (s) => { slot.summary = s; slot.ended = true; } });
    if (!this.encoder) this.encoder = new Encoder(slot.m.gd, { version: this.obsVersion });
    if (this.G0 === null) this.G0 = this.encoder.spec().blocks.global.offset;
    if (this.metrics) this._instrument(slot);
    if (this.rewardV === 2) this._wrapSettle(slot);
    if (this.strictPreview) slot.m._visibleOnly = slot.rl; // strict preview: read by legacy fieldModel (server/match/bot.js), absent in production
    if (this.infoGuard) this._guardMatch(slot);
    if (this.bondLog) { // diagnostics (bond log on): every seat's 联防 kills as a helper
      const m = slot.m, inner = m.settle.bind(m); slot.uniteKills = new Map();
      m.settle = (plan, ur) => {
        if (plan && ur && !ur.synthetic) for (const ps of plan.helpers || []) slot.uniteKills.set(ps.playerId, (slot.uniteKills.get(ps.playerId) || 0) + (Number(ur.perPlayer?.[ps.playerId]?.killed) || 0));
        return inner(plan, ur);
      };
    }
    if (this.timing) { // battle stepping time, rehearsal fields ('r:…') apart from real battles (as tools/botbench.mjs)
      const timing = this.timing, nb = slot.m.newBattle.bind(slot.m);
      slot.m.newBattle = (opts) => {
        const b = nb(opts);
        if (b && typeof b.step === 'function') {
          const reh = typeof opts?.fieldId === 'string' && opts.fieldId.startsWith('r:'), step = b.step.bind(b);
          b.step = (...args) => {
            const t0 = performance.now();
            try { return step(...args); } finally {
              const t = timing[slot.m.round || 0] ||= { prep: 0, battle: 0, other: 0, teach: 0, encode: 0, decisions: 0 };
              t[reh ? 'rehSim' : 'battleSim'] = (t[reh ? 'rehSim' : 'battleSim'] || 0) + performance.now() - t0;
            }
          };
        }
        return b;
      };
    }
    // version 4 plans (deploy semantics): every policy seat's plan and the view it plans from are fixed at the prep start
    // (the device plans all its seats at once), and a plan still open when the prep ends is closed first (its foreign effects)
    if (this.seatPlan) {
      const ep = slot.m.enterPrep, xp = slot.m.endPrep;
      slot.m.enterPrep = function (...a) { const r = ep.apply(this, a); env._prepStart(slot); return r; };
      slot.m.endPrep = function (...a) { env._prepClose(slot); return xp.apply(this, a); };
    }
    // the round-14 Final Assault's leader pool (m.bossPool is replaced by the Hidden Core's): kept for the fa reward
    const startFA = slot.m.startFinalAssault;
    slot.m.startFinalAssault = function (hidden) {
      const r = startFA.call(this, hidden);
      if (!hidden && !slot.faPool) slot.faPool = this.bossPool;
      return r;
    };
    if (snap) {
      let ok = false;
      try { ok = restoreMatch(slot.m, structuredClone(snap.doc), { createRngFromState, replay: true }); }
      catch (e) { if (this.errorSamples.length < 20) this.errorSamples.push(`[seed ${seed}] restore: ${String(e?.message || e).slice(0, 200)}`); }
      if (ok) { this.snapStats.started++; return slot; }
      this.snapStats.restoreFailed++;
      try { slot.m.dispose(); } catch { /* refused */ }
      return this._open(i); // a refused checkpoint: an ordinary match instead
    }
    slot.m.start();
    return slot;
  }

  /**
   * Rule metrics for every AI seat, whatever drives it (legacy or the policy): operators bought / taken, how many of them
   * a rule forbids (yieldMain, yieldMerge: same-pool living humans only — the policy's seats must count 0 for a rule that
   * is on), and round trips: within one prep, a sale and a purchase of the same operator (either order) — not a rule,
   * a watch for exploits (e.g. selling and buying to steer refreshes), compared with legacy.
   */
  _instrument(slot) {
    const m = slot.m, gd = m.gd, env = this;
    for (const ps of m.players.values()) {
      if (slot.humans.has(ps.playerId)) continue;
      const role = slot.role.get(ps.playerId);
      const y = slot.yield[role] ||= { buys: 0, sells: 0, yieldMain: 0, yieldMerge: 0, roundTrips: 0 };
      const prep = { key: null, buys: new Map(), sells: new Map() };
      const fresh = () => { // the seat's prep bookkeeping, reset at its first action of a new prep
        const key = `${m.round}:${m.phase}`;
        if (prep.key !== key) { prep.key = key; prep.buys = new Map(); prep.sells = new Map(); }
      };
      // one round trip per sale matched by a purchase of the same operator in the same prep (either order);
      // each side's list holds its events ({ funds delta, layers delta, id }) for the trip log
      const trip = (base, mine, other, ev) => {
        const o = other.get(base);
        if (o && o.length) {
          const first = o.shift(); y.roundTrips++;
          if (env.tripLog && slot.trips.length < 400) {
            const [b, s] = ev.side === 'buy' ? [ev, first] : [first, ev];
            slot.trips.push({ round: m.round, role, order: ev.side === 'buy' ? 'sell-buy' : 'buy-sell', id: b.id, golden: !!gd.isGolden?.(s.id),
              tier: gd.tierOf?.(b.id) || 0, paid: -b.funds, gain: s.funds, net: b.funds + s.funds, layersBuy: b.layers, layersSell: s.layers,
              fundsLeft: ps.funds, fundsBeforeBuy: b.before, handFreeAfterBuy: b.handFree, soldBy: s.by, band: ps.bandId,
              keepsLeftover: gd.leftoverKeptBands.includes(ps.bandId) });
          }
        } else { if (!mine.has(base)) mine.set(base, []); mine.get(base).push(ev); }
      };
      const layerSum = () => Object.values(ps.layers || {}).reduce((n, v) => n + (Number(v) || 0), 0);
      const proto = Object.getPrototypeOf(ps);
      const acquire = (name, idOf) => {
        ps[name] = function (arg) {
          fresh();
          const id = idOf(arg), chess = !!(id && gd.chess(id));
          let main = false, merge = false;
          if (chess && slot.humans.size) {
            const pub = m.publicView();
            main = chessInBonds(gd, id, humanBonds(gd, pub, ps.playerId, slot.humans));
            const pairs = humanPairs(m, pub, ps.playerId, slot.humans, { scout: env.visibleOnly });
            merge = pairs.has(gd.baseIdOf(id));
            if (env.scoutCheck) {
              const viaScout = humanPairs(m, pub, ps.playerId, slot.humans, { scout: true });
              env.stats.scoutChecks++;
              if ([...pairs].sort().join() !== [...viaScout].sort().join()) env.stats.scoutMismatch++;
            }
          }
          const f0 = ps.funds, l0 = layerSum();
          const r = proto[name].call(this, arg);
          if (!(r && r.error) && chess) {
            y.buys++; if (main) y.yieldMain++; if (merge) y.yieldMerge++;
            if (name === 'buy') trip(gd.baseIdOf(id), prep.buys, prep.sells, { side: 'buy', id, funds: ps.funds - f0, layers: layerSum() - l0,
              before: f0, handFree: ps.hand.filter((p) => !p).length });
          }
          return r;
        };
      };
      acquire('buy', (i) => ps.shop?.slots?.[i]?.id);
      acquire('pickReward', (i) => ps.offers?.[0]?.slots?.[i]?.id);
      ps.sell = function (uid) {
        fresh();
        const piece = ps.find(uid)?.piece;
        const f0 = ps.funds, l0 = layerSum();
        const r = proto.sell.call(this, uid);
        if (!(r && r.error) && piece) {
          y.sells++; trip(gd.baseIdOf(piece.id), prep.sells, prep.buys, { side: 'sell', id: piece.id, funds: ps.funds - f0, layers: layerSum() - l0,
            by: slot.acting === ps.playerId ? 'policy' : 'legacy' });
        }
        return r;
      };
    }
  }

  /** Band draft bookkeeping: human proxies announce a plan once picked; deferred model-AI band decisions become pending
   *  once every human proxy has picked (or the draft moved on). */
  _bandStep(slot) {
    const m = slot.m, d = m.draft, gd = m.gd;
    for (const pid of slot.humans) {
      const band = d.picks?.[pid];
      if (!band || slot.announced.has(pid)) continue;
      slot.announced.add(pid);
      const ps = m.players.get(pid);
      const tied = gd.bandBondIds(band).filter((b) => gd.bondPlannable?.(b));
      let main = tied[0] || null;
      if (!main && slot.collectRng() < this.bandPolicy.planProb) {
        const cores = gd.bondIds.filter((b) => gd.bond(b)?.isCore && gd.bondPlannable?.(b) && !gd.modeInactiveBonds?.has?.(b));
        main = cores.length ? cores[Math.floor(slot.collectRng() * cores.length)] : null;
      }
      if (main && ps) {
        try { m.planBond(ps, main, []); } catch { /* the plan stays display-only then */ }
        ps._envPlanBonds = [{ id: main, priority: 1 }]; // bot.js bondPlan (src-next copy): legacy's focus = the announced plan
      }
    }
    if (!slot.deferredBand.length) return;
    const proxiesDone = [...slot.humans].every((pid) => d.picks?.[pid] || !m.players.get(pid)?.alive);
    if (!proxiesDone && m.phase === 'BAND_DRAFT') return;
    // one model-AI band decision at a time: the next one is encoded after this one applied, so its AI-AI mask sees the pick
    // (releasing them together let two seats of one batch take the same band: an engine-rejected pick)
    while (slot.deferredBand.length) {
      const dec = slot.deferredBand.shift();
      if (dec.d.valid() && !dec.d.complete()) { slot.pending.push(dec); return; }
    }
  }

  /** The team's rounds reached and LP (every seat of the match), for the team share of the reward. */
  _team(m) {
    let rounds = 0, lp = 0, n = 0;
    for (const p of m.players.values()) {
      if (p.spectator) continue;
      n++; rounds += p.alive ? m.round : (p.eliminatedRound ?? m.round); lp += Math.max(0, p.lp || 0);
    }
    return { rounds, lp, n: n || 1 };
  }

  /** Reward of seat state `s` since its last decision (team-weighted), advancing its bookkeeping. */
  _reward(s, m, round, lp) {
    const v2 = this.rewardV === 2, R = s.R || this.R, t = this._team(m), w = v2 ? this.R2.team : R.team;
    const dRound = w * (t.rounds - s.teamRounds) / t.n + (1 - w) * (round - s.lastRound);
    const dLp = w * (t.lp - s.teamLp) / t.n + (1 - w) * (lp - s.lastLp);
    s.lastRound = round; s.lastLp = lp; s.teamRounds = t.rounds; s.teamLp = t.lp;
    const rRound = v2 ? 0 : R.round * dRound, rLp = v2 ? this.R2.lp * dLp / (m.gd.defaultStartLp || 28) : R.lp * dLp / s.lp0;
    s.cRound = (s.cRound || 0) + rRound; s.cLp = (s.cLp || 0) + rLp; // trajectory totals of the two terms (TensorBoard)
    if (!v2) return rRound + rLp;
    const bonus = s.bonus || 0; s.bonus = 0; s.R2.lp += rLp;
    return rLp + bonus;
  }

  /** Once per round: each marked human's main core bond and its tier (public strip). */
  _sampleHumans(slot) {
    const m = slot.m;
    slot.seenRound = m.round;
    if (!slot.humans.size) return;
    const pub = m.publicView();
    for (const p of pub.players || []) {
      if (!slot.humans.has(p.playerId) || !p.alive) continue;
      const main = mainCoreBondOf(m.gd, p), tier = main ? (p.bonds.find((b) => b.bondId === main)?.tier || 0) : 0;
      const h = slot.humanTier.get(p.playerId) || { max: 0 };
      h.max = Math.max(h.max, tier); h.last = tier;
      slot.humanTier.set(p.playerId, h);
    }
  }

  _rehearsalFor(m, ps, slot) {
    if (this.bossRehearsal !== null && (m.round === m.gd.bossRound || m.round === m.gd.hiddenRound)) return this.bossRehearsal;
    if (this.mateRehearsal !== null && !slot.rl.has(ps.playerId)) return this.mateRehearsal;
    return this.rehearsal;
  }

  *_perSeat(m, ps, slot, gen) {
    const set = () => { if (this.bossRehearsal !== null || this.mateRehearsal !== null) m.botRehearsal = this._rehearsalFor(m, ps, slot); };
    const rl = this.infoGuard && slot.rl.has(ps.playerId) ? ps.playerId : null, step = (v) => { if (this.infoGuard) { this._guardRefresh(slot); m._rlPath = rl; m._rlPhase = 'prep'; } try { return gen.next(v); } finally { if (this.infoGuard) m._rlPath = null; } };
    set(); let r = step();
    while (!r.done) { const v = yield r.value; set(); r = step(v); }
    return r.value;
  }

  /** infoguard: count hidden reads while m._rlPath (an RL seat's encode or placement step) is set; samples keep a stack. */
  _guardHit(kind, m) {
    const g = this.infoGuard;
    if (kind === 'otherHand') { // A1 covers the RL seat's encode; the scout view (prepFieldMeta) is what a teammate sees
      const st = String(new Error().stack);
      if (st.includes('prepFieldMeta')) { g.scoutReads++; return; }
      if (m && m._rlPhase === 'prep') { g.prepHandReads++; if (g.samples.length < 12) g.samples.push('prepHand ' + st.split('\n').slice(2, 12).map((x) => x.trim()).join(' < ')); return; }
    }
    g[kind]++;
    if (g.samples.length < 12) g.samples.push(kind + ' ' + String(new Error().stack).split('\n').slice(2, 12).map((x) => x.trim()).join(' < '));
  }

  _guardMatch(slot) {
    const m = slot.m, env = this, f = m.enemyHpMulAt;
    if (typeof f === 'function') m.enemyHpMulAt = function (...a) { if (m._rlPath) env._guardHit('hpMul'); return f.apply(this, a); };
  }

  _guardRefresh(slot) {
    const m = slot.m, env = this, T = Symbol.for('infoguard');
    const wrapSpawns = (wave) => {
      if (!wave || !Array.isArray(wave.spawns) || wave.spawns[T]) return;
      const arr = wave.spawns.map((s) => (s && typeof s === 'object' ? new Proxy(s, { get(t, k, rcv) { if (m._rlPath && (k === 'routeIndex' || k === 'mods')) env._guardHit(k); return Reflect.get(t, k, rcv); } }) : s));
      arr[T] = true; wave.spawns = arr;
    };
    wrapSpawns(m.wave);
    for (const g of Object.values(m.bossWaves || {})) wrapSpawns(g && g.wave);
    for (const ps of m.players.values()) for (const key of ['hand', 'temp']) {
      const a = ps[key]; if (!Array.isArray(a) || a[T]) continue;
      const owner = ps.playerId;
      const p = new Proxy(a, { get(t, k, rcv) { if (k === T) return true; if (m._rlPath && m._rlPath !== owner && typeof k === 'string' && /^\d+$/.test(k)) env._guardHit('otherHand', m); return Reflect.get(t, k, rcv); } });
      ps[key] = p;
    }
  }

  _guard(slot, msg) {
    this.stats.errors++; this.stats.guard = (this.stats.guard || 0) + 1;
    if (this.errorSamples.length < 20) this.errorSamples.push(`[seed ${slot.seed}] guard: ${msg}`);
  }

  _bondSample(slot) {
    const m = slot.m, ph = m.phase;
    slot.blPhase = ph; slot.blRound = m.round;
    if (!(ph === 'COMBAT' || ph === 'FINAL_ASSAULT' || ph === 'HIDDEN_CORE' || ph === 'UNITE')) return;
    slot.bl ||= new Map();
    const alive = new Set(m.alivePlayers());
    for (const pid of slot.rl) {
      const ps = m.players.get(pid);
      if (!ps || !alive.has(ps)) continue;
      const rec = slot.bl.get(pid) || {};
      const row = bondRow(m.gd, ps); rec.last = row; rec.lastRound = m.round; // end = the board of the seat's last battle
      // economy / LP trace at each battle start (behaviour diffs): cumulative counters, per-prep values by difference
      const st = ps.stats || {}, team = this._team(m);
      (rec.trace ||= []).push({ r: m.round, ph, lp: ps.lp, teamLp: team.lp, funds: ps.funds, level: ps.shop?.level ?? null, buys: st.buys, sells: st.sells,
        refreshes: st.refreshes, merges: st.merges, leaks: st.leaks, fundsGained: st.fundsGained, bossDamage: st.bossDamage, layers: Math.round(bondTerms(m.gd, ps.privateView().bonds, true).S * 100) });
      if ([6, 9, 12].includes(m.round) && !rec['r' + m.round]) rec['r' + m.round] = row;
      slot.bl.set(pid, rec);
    }
  }

  /** R2 direct layer reward: at each prep end, R2.layer per stacking-bond layer above the seat's high-water mark. */
  _layerHook(slot) {
    const m = slot.m, prev = slot.lhPhase; slot.lhPhase = m.phase;
    if (prev !== 'PREP' || m.phase === 'PREP') return;
    const alive = new Set(m.alivePlayers());
    for (const pid of slot.rl) {
      const s = slot.seat.get(pid), ps = m.players.get(pid);
      if (!s || !s.R2 || !ps || !alive.has(ps)) continue;
      const cur = Math.round(bondTerms(m.gd, ps.privateView().bonds, true).S * 100);
      if (s.layerHW === undefined) { s.layerHW = cur; s.ev.layerHW = cur; s.ev.layerHW0 = cur; continue; } // first prep end seen (a late start keeps its layers)
      if (cur > s.layerHW) { const v = this.R2.layer * (cur - s.layerHW); s.bonus += v; s.R2.layer += v; s.layerHW = cur; s.ev.layerHW = cur; }
    }
  }

  /** R2: own counted leaks of the normal combat and 联防 helper kills, charged when the round settles. */
  _wrapSettle(slot) {
    const m = slot.m, orig = m.settle.bind(m), R2 = this.R2;
    m.settle = (plan, uniteResult) => {
      if (!m.disposed && !m.ended && (m.phase === 'COMBAT' || m.phase === 'UNITE')) {
        const L = m.gd.defaultStartLp || 28;
        const ran = !!(plan && uniteResult && !uniteResult.synthetic);
        for (const ps of m.alivePlayers()) {
          const s = slot.rl.has(ps.playerId) ? slot.seat.get(ps.playerId) : null;
          if (!s || !s.R2) continue;
          const r = m.lastResults.get(ps.playerId);
          const raw = r ? (r.leaked || []).filter((l) => l && l.counted !== false).length : 0;
          const counted = R2.leakCap ? Math.min(raw, m.gd.lpCapPerRound || 10) : raw; // leakCap: the per-round LP cap applies
          if (counted) { const v = -R2.leak * counted / L; s.bonus += v; s.R2.leak += v; s.ev.leaks += counted; }
          const up = ran && plan.helpers.includes(ps) ? uniteResult.perPlayer?.[ps.playerId] : null;
          const k = up ? Math.max(0, Number(up.killed) || 0) : 0;
          if (k) { const v = R2.save * k / L; s.bonus += v; s.R2.save += v; s.ev.saves += k; }
        }
      }
      return orig(plan, uniteResult);
    };
  }

  _seat(slot, ps) {
    let s = slot.seat.get(ps.playerId);
    if (!s) {
      const t = this._team(slot.m);
      s = { ...(this.rewardV === 2 ? { R2: { win: 0, faTeam: 0, faOwn: 0, leak: 0, save: 0, lp: 0, elim: 0, layer: 0 }, bonus: 0, ev: { leaks: 0, saves: 0, layerHW: 0 } } : {}),
        R: this.RB[slot.cfg.difficulty] || this.R, traj: ++this.trajSeq, lp0: ps.bandId ? slot.m.gd.startLp(ps.bandId) : (ps.lp || 1), pre: !ps.bandId || !(ps.lp > 0), lastLp: Math.max(0, ps.lp || 0), lastRound: slot.m.round || 1, teamRounds: t.rounds, teamLp: t.lp, steps: 0 };
      slot.seat.set(ps.playerId, s);
    } else if (s.pre && ps.lp > 0 && ps.bandId) {
      // created at the band decision, before the band set the LP (lp 0): start the LP bookkeeping at its first decision after
      // it, so the obs LP field (lp / lp0) and the LP reward are the same as a seat whose band legacy picked (bug of 11-12).
      // lp0 is always the band's starting LP (gd.startLp), also for a late-game (snapshot) start (an audit, 11-12)
      const t = this._team(slot.m);
      s.pre = false; s.lp0 = ps.bandId ? slot.m.gd.startLp(ps.bandId) : ps.lp; s.lastLp = ps.lp; s.teamLp = t.lp; s.teamRounds = t.rounds; s.lastRound = slot.m.round || 1;
    }
    return s;
  }

  /** Run slot i until it waits on the policy or is finished and replaced. */
  _advance(i) {
    let slot = this.slots[i];
    for (let guard = 0; guard < 1e7 && slot; guard++) {
      if (!slot.pending.length && slot.deferredBand?.length && this.bandPolicy && slot.m.draft) this._bandStep(slot);
      if (!slot.pending.length && slot.deferredPrep.length) this._releasePrep(slot);
      if (slot.pending.length) return;
      if (slot.ended || slot.failed) { this._finish(slot); slot = this.slots[i] = this._open(i); continue; }
      let ok;
      if (this.timing) {
        const r = slot.m.round || 0, ph = slot.m.phase, t0 = performance.now();
        ok = slot.sched.runNext();
        const key = ph === 'PREP' ? 'prep' : ph === 'COMBAT' || ph === 'FINAL_ASSAULT' || ph === 'HIDDEN_CORE' || ph === 'UNITE' ? 'battle' : 'other';
        const t = this.timing[r] ||= { prep: 0, battle: 0, other: 0, teach: 0, encode: 0, decisions: 0 };
        t[key] += performance.now() - t0;
      } else ok = slot.sched.runNext();
      if (!ok) { slot.failed = 'stuck'; continue; }
      if (++slot.cb > 2e6) slot.failed = 'callbacks';
      if (this.metrics && slot.m.round !== slot.seenRound) this._sampleHumans(slot);
      if (this.bondLog && (slot.m.phase !== slot.blPhase || slot.m.round !== slot.blRound)) this._bondSample(slot);
      if (this.rewardV === 2 && this.R2.layer > 0 && slot.m.phase !== slot.lhPhase) this._layerHook(slot);
      if (this.bandPolicy && slot.m.draft) this._bandStep(slot);
    }
  }

  _finish(slot) {
    const m = slot.m;
    this.stats.matches++;
    this.stats.callbacks += slot.cb;
    if (slot.failed) { this.stats.failed++; if (this.errorSamples.length < 20) this.errorSamples.push(`[seed ${slot.seed}] ${slot.failed} in ${m.phase} R${m.round}`); }
    let violations = [];
    if (this.check) { try { violations = collectViolations(m); } catch (e) { violations = [`collectViolations threw ${e.message}`]; } }
    if (violations.length) { this.stats.violations += violations.length; if (this.errorSamples.length < 20) this.errorSamples.push(`[seed ${slot.seed}] violations ${violations.slice(0, 3).join(' | ')}`); }
    const rows = slot.summary?.players || [];
    if (this.phiSamples) for (const p of rows) {
      const ps = m.players.get(p.playerId);
      if (ps) this.phiSamples.push({ config: slot.cfg.name, role: slot.role.get(p.playerId), round: p.eliminatedRound ?? m.round,
        win: p.victory ? 1 : 0, ...bondTerms(m.gd, ps.privateView().bonds) });
    }
    const pool = slot.faPool;
    const faShare = pool && pool.maxHp > 0 ? Math.min(1, Math.max(0, 1 - Math.max(0, pool.hp) / pool.maxHp)) : 0;
    const cleared = rows.some((p) => p.victory) || !!(pool && pool.maxHp > 0 && pool.hp <= 0);
    const diag = this.bondLog ? { fa: pool ? { maxHp: pool.maxHp, hp: pool.hp, aliveAtFA: m.finalAliveCount ?? null, by: Object.fromEntries(pool.byPlayer || []) } : null,
      seatsAll: rows.map((p) => { const ps = m.players.get(p.playerId), st = ps?.stats || {};
        return { pid: p.playerId, role: slot.role.get(p.playerId), elim: p.eliminatedRound ?? null, win: p.victory ? 1 : 0, lp: p.lp, leaks: st.leaks, kills: st.kills, lpLost: st.lpLost, bossDamage: st.bossDamage, unite: slot.uniteKills?.get(p.playerId) || 0 }; }) } : null;
    const res = { config: slot.cfg.name, seed: slot.seed, failed: slot.failed, seats: [], yield: slot.yield, humans: slot.humans.size, faShare,
      ...(slot.fromSnapshot ? { fromSnapshot: slot.fromSnapshot } : {}),
      ...(diag || {}), ...(this.tripLog ? { trips: slot.trips } : {}), ...(slot.names ? { names: slot.names.list, replay: slot.replay } : {}) };
    for (const p of rows) {
      const round = p.eliminatedRound ?? m.round;
      const rl = slot.rl.has(p.playerId);
      res.seats.push({ playerId: p.playerId, rl, role: slot.role.get(p.playerId), band: m.players.get(p.playerId)?.bandId ?? null, survivalRound: round, roundsPassed: p.roundsPassed,
        lp: p.lp, win: p.victory ? 1 : 0, mainTier: slot.humanTier.get(p.playerId)?.max ?? null,
        ...(this.bondLog && rl ? (() => { const { last, lastRound, ...rs } = slot.bl?.get(p.playerId) || {}; return { bondLog: { ...rs, end: last || null, endRound: lastRound ?? null } }; })() : {}) });
      const s = slot.seat.get(p.playerId);
      if (!s) continue;
      // terminal shaping: γ·Φ(terminal = 0) − Φ(last decision), also for a seat eliminated earlier
      const shapeEnd = this.shaping && s.phi !== undefined ? -s.phi : 0;
      s.shape = (s.shape || 0) + shapeEnd;
      const R = s.R || this.R;
      let reward;
      if (this.rewardV === 2) {
        const R2 = this.R2, c = s.R2, base = this._reward(s, m, round, Math.max(0, p.lp || 0));
        const own = pool && pool.maxHp > 0 ? Math.max(0, (pool.byPlayer && pool.byPlayer.get ? pool.byPlayer.get(p.playerId) : 0) || 0) / pool.maxHp : 0;
        const win = cleared ? R2.win : 0, faT = R2.faTeam * faShare, faO = R2.faOwn * own, elim = p.eliminatedRound != null && !p.victory ? -R2.elim : 0;
        c.win += win; c.faTeam += faT; c.faOwn += faO; c.elim += elim;
        reward = base + win + faT + faO + elim + shapeEnd;
        if (rl) res.seats[res.seats.length - 1].r2 = { ...c, shape: (s.shape || 0), ev: { ...s.ev, ownFa: own, eliminated: p.eliminatedRound ?? null, cleared } };
      } else reward = this._reward(s, m, round, Math.max(0, p.lp || 0)) + (p.victory ? R.win : 0) + R.fa * faShare + shapeEnd;
      const t = { traj: s.traj, reward, win: p.victory ? 1 : 0, round, lp: p.lp || 0, config: slot.ci, steps: s.steps, faShare,
        rRound: s.cRound || 0, rLp: s.cLp || 0, shape: s.shape || 0, seedOffset: slot.seed - this.seed,
        ...(s.R2 ? { r2: [s.R2.win, s.R2.faTeam, s.R2.faOwn, s.R2.leak, s.R2.save, s.R2.lp, s.R2.elim, s.R2.layer, s.shape || 0] } : {}) };
      if (this.recorder) this.recorder.term(t); else this.terms.push(t);
    }
    this.results.push(res);
    if (this.results.length > 10000) this.results.shift();
    try { m.dispose(); } catch { /* finished */ }
  }

  /** Decisions waiting on the policy, encoded into this.obs / this.mask (n rows). Ended matches are replaced. */
  collect() {
    for (let i = 0; i < this.envs; i++) this._advance(i);
    const items = [];
    for (const slot of this.slots) if (slot) for (const dec of slot.pending) items.push(dec);
    if (items.length > this.cap) this._buf(Math.max(items.length, this.cap * 2));
    const D = this.D, A = this.A;
    for (let k = 0; k < items.length; k++) {
      const dec = items[k], { slot, ps } = dec, m = slot.m;
      const s = this._seat(slot, ps);
      this.rew[k] = this._reward(s, m, m.round, Math.max(0, ps.lp || 0));
      s.steps++;
      this.ids[k] = s.traj; this.kinds[k] = KIND[dec.kind] + (this.kindSeat ? 16 * (Number(String(dec.ps.playerId).split('_')[1]) || 0) : 0);
      const te0 = this.timing ? performance.now() : 0;
      const newUids = dec.startUids ? new Set(ps.allChess().map((p) => p.uid).filter((u) => !dec.startUids.has(u))) : null;
      const aiPicks = dec.kind === 'band' ? new Set([...m.players.keys()].filter((pid) => !slot.humans.has(pid) && m.draft?.picks?.[pid])) : null;
      if (this.full && dec.kind !== 'band' && dec.kind !== 'card' && !dec.opened) this._openPlan(dec);
      if (this.infoGuard) { this._guardRefresh(slot); m._rlPath = ps.playerId; m._rlPhase = 'enc'; }
      const r = this.encoder.encode(m, ps, { kind: dec.kind, actions: dec.actions, freezes: dec.freezes, lp0: s.lp0, humans: slot.humans, visibleOnly: this.visibleOnly, strictPreview: this.strictPreview,
        rules: this.rules, newUids, soldBases: dec.soldBases, cache: dec.cache || {}, aiPicks,
        ...(this.full ? { sel: dec.sel, pubSnap: dec.pubSnap, ...(dec.mateSnap !== undefined ? { mateSnap: dec.mateSnap } : {}),
          planned: dec.planned, rerolls: seatPlanRerolls(ps) } : {}) },
        this.obs.subarray(k * D, (k + 1) * D), this.mask.subarray(k * A, (k + 1) * A));
      dec.own = r.own; dec.owned = r.owned;
      if (this.infoGuard) {
        m._rlPath = null;
        const pub = r.pub || {}; if (this.visibleOnly && !(pub.hiddenRound && (pub.round || 0) >= pub.hiddenRound) && this.obs[k * D + this.encoder.lay.blocks.extra.offset + 2] !== 0) this.infoGuard.hiddenBoss++;
      }
      if (this.check) { // regression guard (11-12): a seat's LP bookkeeping starts at its real starting LP; the own-LP field stays in range
        if (!s.pre && !s.lp0Checked && ps.bandId) {
          s.lp0Checked = true;
          const want = m.gd.startLp(ps.bandId);
          if (s.lp0 !== want) this._guard(slot, `lp0 ${s.lp0} != starting LP ${want} of ${ps.bandId} (${ps.playerId}, ${dec.kind}, R${m.round})`);
        }
        const v = this.obs[k * D + this.G0 + 9];
        if (!(v >= 0 && v <= 1.05)) this._guard(slot, `own-LP obs ${v} outside [0, 1.05] (${ps.playerId}, ${dec.kind}, R${m.round}, lp ${ps.lp}, lp0 ${s.lp0})`);
      }
      if (this.lpObs !== null) { const i = k * D + this.G0 + 9; this.obs[i] = this.lpObs === 'raw' ? Math.max(0, r.own?.lp ?? ps.lp ?? 0) : this.obs[i] * this.lpObs; }
      if (dec.kind === 'band') { this.bandStats.decisions++; if (r.bandFallback) this.bandStats.fallback++; dec.aiPicks = aiPicks; }
      if (this.timing) { const t = this.timing[m.round || 0] ||= { prep: 0, battle: 0, other: 0, teach: 0, encode: 0, decisions: 0 };
        t.encode += performance.now() - te0; t.decisions++; }
      if (this.shaping) {
        const { T, S } = bondTerms(m.gd, r.own.bonds, this.shaping.uncapped);
        const phi = this.shaping.wTier * T + this.shaping.wStack * S;
        if (s.phi !== undefined) { const sh = this.shaping.gamma * phi - s.phi; this.rew[k] += sh; s.shape = (s.shape || 0) + sh; }
        s.phi = phi;
      }
    }
    this.items = items;
    this.stats.decisions += items.length;
    const terms = this.terms; this.terms = [];
    return { n: items.length, ids: this.ids, kinds: this.kinds, rew: this.rew, obs: this.obs, mask: this.mask, D, A, terms };
  }

  reset() { return this.collect(); }

  /** For the rows of the last collect(): a readable state of each describeSeats row (null for the others), with the
   *  labels of its legal actions. */
  describe() {
    return (this.items || []).map((dec, k) => {
      if (!this.describeSeats.has(dec.ps.playerId) || !dec.slot.names) return null;
      const s = summarize(dec.slot.m, dec.ps, dec.own, dec.slot.names, dec.kind);
      const mask = this.mask.subarray(k * this.A, (k + 1) * this.A), labels = {};
      for (let a = 0; a < this.A; a++) if (mask[a]) labels[a] = actionLabel(a, s, dec.slot.names);
      return { seed: dec.slot.seed, playerId: dec.ps.playerId, s, labels };
    });
  }

  /** Apply one action per row of the last collect(), then collect the next decisions. */
  step(actions) {
    const items = this.items || [];
    for (let k = 0; k < items.length; k++) this._apply(items[k], actions[k] | 0, this.mask.subarray(k * this.A, (k + 1) * this.A));
    this.items = null;
    return this.collect();
  }

  _drop(dec) { if (dec.planned) this._endPlan(dec); const p = dec.slot.pending, j = p.indexOf(dec); if (j >= 0) p.splice(j, 1); }

  // ---- version 4: full control ----
  /** A boss round's R seat waits while its partner L is a policy seat still planning (one device plans L, then R). */
  _bossWaitsForL(slot, ps) {
    const m = slot.m;
    if (m.round !== m.gd.bossRound && m.round !== m.gd.hiddenRound) return false;
    const g = typeof m.bossGroupOf === 'function' ? m.bossGroupOf(ps) : null;
    if (!g || g.side !== 'R' || !Array.isArray(g.players) || g.players.length < 2) return false;
    const pid = g.players.find((x) => x !== ps.playerId), mate = pid ? m.players.get(pid) : null;
    return !!mate && mate.alive && !mate.ready && slot.rl.has(pid) && !slot.humans.has(pid);
  }

  _releasePrep(slot) {
    for (let k = 0; k < slot.deferredPrep.length; k++) {
      const dec = slot.deferredPrep[k];
      if (!dec.d.valid() || dec.d.complete()) { slot.deferredPrep.splice(k--, 1); continue; }
      if (this._bossWaitsForL(slot, dec.ps)) continue;
      slot.deferredPrep.splice(k--, 1); slot.pending.push(dec);
    }
  }

  /** PREP begins (after onPrepStart): the view every policy seat plans from, and each one's plan (refresh table, streams). */
  _prepStart(slot) {
    const m = slot.m;
    slot.prepSnap = m.publicView(); slot.prepMate = new Map();
    for (const pid of slot.rl) {
      const ps = m.players.get(pid);
      if (!ps || !ps.alive || ps.ready || slot.humans.has(pid) || !ps.isBot) continue;
      slot.prepMate.set(pid, ps.privateView().bossMate ?? null);
      beginSeatPlan(m, ps, this.seatPlanParts); this.planStats.plans++;
    }
  }

  _prepClose(slot) {
    for (const pid of slot.rl) { const ps = slot.m.players.get(pid); if (ps && ps._seatPlan) this._closePlan(ps); }
  }

  /** The seat's first decision of the prep: its plan's view (a boss partner that planned first and is Ready: live). */
  _openPlan(dec) {
    const { slot, ps } = dec, m = slot.m;
    dec.opened = true;
    if (!this.seatPlan || !seatPlanOf(ps)) return;
    dec.pubSnap = slot.prepSnap || null; dec.planned = true;
    const g = typeof m.bossGroupOf === 'function' ? m.bossGroupOf(ps) : null;
    const pid = g && Array.isArray(g.players) ? g.players.find((x) => x !== ps.playerId) : null, mate = pid ? m.players.get(pid) : null;
    if (!(mate && mate.ready && slot.rl.has(pid))) dec.mateSnap = slot.prepMate?.get(ps.playerId) ?? null;
  }

  _endPlan(dec) { dec.planned = false; this._closePlan(dec.ps); }

  _closePlan(ps) {
    const st = endSeatPlan(ps);
    if (!st) return;
    let any = false;
    for (const k of Object.keys(this.planStats.stops)) {
      this.planStats.stops[k] += st[k] || 0;
      if (st[k] > 0) { this.planStats.stopPreps[k]++; any = true; }
    }
    if (any) this.planStats.stopPreps.any++;
  }

  /** Version 4 actions of a prep / tile / target decision; false = an economy action (the version 2/3 path). */
  _applyFull(dec, a) {
    const { slot, ps, d } = dec, m = slot.m;
    if (!d.valid()) { this._drop(dec); return true; }
    if (dec.kind === 'tile' || dec.kind === 'target') {
      const uid = dec.selUid; dec.kind = 'prep'; dec.sel = -1; dec.selUid = null; dec.actions++;
      let intent = null;
      if (a >= ACT.TILE && a < ACT.BENCH) {
        const t = Math.floor((a - ACT.TILE) / 4), [row, col] = tileRC(t);
        intent = { t: 'g.move', uid, to: { area: 'board', row, col }, dir: DIRS[(a - ACT.TILE) % 4] };
      } else if (a === ACT.BENCH) {
        const idx = ps.hand.findIndex((p) => p == null);
        intent = { t: 'g.move', uid, to: { area: 'hand', idx: idx >= 0 ? idx : 0 } };
      } else if (a >= ACT.TARGET && a < ACT.TARGET + OWNED) {
        const q = dec.owned?.[a - ACT.TARGET]; if (q) intent = { t: 'g.equip', itemUid: uid, targetUid: q.uid };
      }
      if (intent) { const r = m._handleSeatIntent(ps, intent); if (r?.error) { this.stats.rejected++; this.planStats.rejected++; } }
      if (!d.valid() || d.complete()) this._drop(dec);
      return true;
    }
    if (a === ACT.END) { // the seat's Ready, as a human's (what scheduleBotPrep's ready() does first: a left-over choice, temp)
      this._drop(dec);
      try { m.autoPickPersonalChoice(ps, 'random'); ps.resolveTemp(); } catch { /* refused: the engine's own Ready path */ }
      const r = m._handleSeatIntent(ps, { t: 'g.ready', ready: true });
      if (r?.error || !ps.ready) { this.planStats.readyFallback++; if (d.valid() && !d.complete()) d.fallback(); }
      return true;
    }
    if (a >= ACT.MOVE && a < ACT.TILE) {
      const i = a >= ACT.EQUIP ? a - ACT.EQUIP : a - ACT.MOVE, p = dec.owned?.[i];
      dec.actions++;
      if (!p) { this.stats.rejected++; return true; }
      dec.kind = a >= ACT.EQUIP ? 'target' : 'tile'; dec.sel = i; dec.selUid = p.uid;
      return true;
    }
    return false;
  }

  _apply(dec, a, mask) {
    const { slot, ps, d } = dec, m = slot.m;
    if (!(a >= 0 && a < this.A) || !mask[a]) { this.stats.rejected++; a = dec.kind === 'card' ? -1 : dec.kind === 'tile' || dec.kind === 'target' ? ACT.CANCEL : ACT.END; }
    if (dec.kind === 'band') {
      this._drop(dec);
      if (!d.valid()) return;
      const j = a - ACT.BAND, bandId = a >= ACT.BAND && j < BANDS ? this.encoder.vocab.bandList[j] : null;
      if (bandId) {
        const bc = bandConflicts(m.gd, m.publicView(), ps.playerId, slot.humans, dec.aiPicks);
        const tied = m.gd.bandBondIds(bandId);
        if (tied.some((b) => bc.humanBand.has(b) || bc.humanMain.has(b))) this.bandStats.humanConflict++;
        if (tied.some((b) => bc.aiPick.has(b))) this.bandStats.aiSameBond++;
        const r = m._handleSeatIntent(ps, { t: 'g.band', bandId });
        if (r?.error) this.stats.rejected++;
      }
      if (!d.complete()) d.fallback();
      return;
    }
    if (dec.kind === 'card') {
      this._drop(dec);
      if (!d.valid()) return;
      const c = a >= ACT.CARD ? m.sp?.cards?.[a - ACT.CARD] : null;
      if (c) { const r = m._handleSeatIntent(ps, { t: 'g.choice', idx: c.idx }); if (r?.error) this.stats.rejected++; }
      if (!d.complete()) d.fallback();
      return;
    }
    if (this.full && this._applyFull(dec, a)) return;
    if (!d.valid()) { this._drop(dec); return; }
    if (a === ACT.END) { this._drop(dec); if (!d.complete()) d.fallback(); return; }
    let intent = null;
    if (a === ACT.REFRESH) intent = { t: 'g.refresh' };
    else if (a === ACT.LEVEL) intent = { t: 'g.levelUp' };
    else if (a === ACT.FREEZE) { intent = { t: 'g.freeze' }; dec.freezes++; }
    else if (a < ACT.SELL) intent = { t: 'g.buy', slot: a - ACT.BUY };
    else if (a < ACT.REWARD) { const p = dec.owned?.[a - ACT.SELL]; if (p) { intent = { t: 'g.sell', uid: p.uid }; dec.soldBase = m.gd.baseIdOf(p.id); } }
    else if (a < ACT.PERSONAL) intent = { t: 'g.reward', idx: a - ACT.REWARD };
    else if (a < ACT.CARD) { const pc = dec.own?.personalChoice; if (pc) intent = { t: 'g.choice', idx: a - ACT.PERSONAL, choiceId: pc.id }; }
    dec.actions++;
    slot.acting = ps.playerId; // the trip log tells the policy's own sales from legacy's cleanup
    let r;
    try { r = intent ? m._handleSeatIntent(ps, intent) : { error: 'BAD_ACTION' }; } finally { slot.acting = null; }
    if (r?.error) this.stats.rejected++;
    else if (intent?.t === 'g.sell' && dec.soldBase) dec.soldBases.add(dec.soldBase);
    dec.soldBase = null;
    if (dec.actions >= this.encoder.maxActions) this.stats.forcedEnd++; // the next mask offers END only
    if (!d.valid() || d.complete()) this._drop(dec);
  }

  // ---- teacher: legacy procurement, every PlayerState call it makes recorded as the env action it equals ----
  _teach(slot, ps, d) {
    const m = slot.m, T = this.stats.teacher;
    const obs = new Float32Array(this.encoder.dim), mask = new Uint8Array(this.encoder.A);
    const s = this._seat(slot, ps);
    const startUids = new Set(ps.allChess().map((p) => p.uid)), soldBases = new Set();
    const ctx = { kind: d.kind, actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, soldBases, newUids: new Set(), cache: {},
      ...(this.full ? { sel: -1, pubSnap: slot.prepSnap || null, planned: false } : {}) };
    const rec = this.recorder;
    const log = slot.names && this.describeSeats.has(ps.playerId)
      ? (() => { const e = { r: m.round, k: d.kind === 'card' ? 1 : 0, before: null, acts: [] }; slot.replay.push(e); return e; })() : null;
    let lastOwn = null;
    // samples of this prep: straight to the recorder, or held for the placement labels (written at the prep's end)
    const held = this.placementLabels && rec ? [] : null;
    const put = (kind, o, mk, a, rw, info) => {
      if (!held) { rec.sample(kind, s.traj, o, mk, a, rw); return; }
      held.push({ kind, o: o === obs ? obs.slice() : o, mk: mk === mask ? mask.slice() : mk, a, rw, info });
    };
    const note = (a, label) => {
      if (log && a >= 0 && lastOwn) log.acts.push(actionLabel(a, summarize(m, ps, lastOwn, slot.names, d.kind), slot.names));
      T.calls[label] = (T.calls[label] || 0) + 1;
      if (a < 0 || !mask[a]) { T.illegal[label] = (T.illegal[label] || 0) + 1; return; }
      // behaviour-cloning sample: the observation and mask the call was made in, the env action it equals, and the
      // reward since this trajectory's previous sample (it belongs to the previous transition, as in step())
      if (rec) { s.steps++; put(KIND[ctx.kind] ?? KIND[d.kind], obs, mask, a, this._reward(s, m, m.round, Math.max(0, ps.lp || 0)), null); }
    };
    if (d.kind === 'card') {
      const sp = m.sp; if (!sp) return false;
      const avail = sp.cards.filter((c) => m.spCardAvailable(c, ps)).map((c) => c.idx);
      if (!avail.length) return false;
      let idx = LEGACY_BOT.pickCard(m, ps, sp.cards, avail);
      if (this.econ) { const a = this._econPick(m, ps, { ...ctx, kind: 'card' }); if (a >= ACT.CARD && a < ACT.CARD + CARDS && sp.cards[a - ACT.CARD]) { idx = sp.cards[a - ACT.CARD].idx; this.econStats.cards++; } }
      lastOwn = this.encoder.encode(m, ps, ctx, obs, mask).own;
      if (log) log.before = summarize(m, ps, lastOwn, slot.names, d.kind);
      const pos = sp.cards.findIndex((c) => c.idx === idx);
      const r = m._handleSeatIntent(ps, { t: 'g.choice', idx });
      if (r?.error) return false; // the engine falls back to the legacy pick
      note(pos >= 0 && pos < CARDS ? ACT.CARD + pos : -1, 'card');
      return true;
    }
    T.preps++;
    let owned = null, protect = new Set(), pairs = new Set();
    const before = () => {
      ctx.newUids = new Set(ps.allChess().map((p) => p.uid).filter((u) => !startUids.has(u)));
      const r = this.encoder.encode(m, ps, ctx, obs, mask); owned = r.owned; protect = r.protect; pairs = r.pairs;
      lastOwn = r.own;
      if (log && !log.before) log.before = summarize(m, ps, r.own, slot.names, d.kind);
    };
    // which rule a recorded legacy call breaks (it is outside the mask): the 2026-11-11 rule metrics of legacy itself
    const env = this;
    const why = (name, args) => {
      const gd = m.gd, out = [];
      const id = name === 'buy' ? ps.shop?.slots?.[args[0]]?.id : name === 'pickReward' ? ps.offers?.[0]?.slots?.[args[0]]?.id : null;
      if (id && gd.chess(id)) {
        if (this.rules.yieldMain && chessInBonds(gd, id, protect)) out.push('yieldMain');
        if (this.rules.yieldMerge && pairs.has(gd.baseIdOf(id))) out.push('yieldMerge');
      }
      return out.length ? out : ['other'];
    };
    // compose with the instance's own wrappers (the 让盟约 counter) and restore them afterwards
    const saved = new Map();
    const inner = (name) => { saved.set(name, Object.getOwnPropertyDescriptor(ps, name)); return ps[name]; };
    const wrap = (name, toAction) => {
      const base = inner(name);
      ps[name] = function (...args) {
        before();
        const a = toAction(...args);
        const reasons = a >= 0 && !mask[a] ? why(name, args) : null;
        const soldBase = name === 'sell' ? m.gd.baseIdOf(ps.find(args[0])?.piece?.id || '') : null;
        if (env.enforce && reasons && reasons[0] !== 'other') {
          for (const k of reasons) T.blocked[`${name}:${k}:refused`] = (T.blocked[`${name}:${k}:refused`] || 0) + 1;
          return { error: 'RULE' };
        }
        const r = base.apply(this, args);
        if (!(r && r.error)) {
          note(a, name); ctx.actions++;
          if (name === 'freeze') ctx.freezes++;
          if (soldBase) soldBases.add(soldBase);
          if (reasons) for (const k of reasons) T.blocked[`${name}:${k}`] = (T.blocked[`${name}:${k}`] || 0) + 1;
        }
        return r;
      };
    };
    const count = (name) => { const base = inner(name); ps[name] = function (...args) { const r = base.apply(this, args); if (!(r && r.error)) T.unrepresented[name] = (T.unrepresented[name] || 0) + 1; return r; }; };
    wrap('buy', (slotIdx) => (slotIdx >= 0 && slotIdx < SHOP ? ACT.BUY + slotIdx : -1));
    wrap('sell', (uid) => { const j = owned ? owned.findIndex((p) => p && p.uid === uid) : -1; return j >= 0 && j < OWNED ? ACT.SELL + j : -1; });
    wrap('refresh', () => ACT.REFRESH);
    wrap('freeze', () => ACT.FREEZE);
    wrap('levelUp', () => ACT.LEVEL);
    wrap('pickReward', (idx) => (idx >= 0 && idx < REWARDS ? ACT.REWARD + idx : -1));
    if (this.full) {
      // version 4: legacy's whole prep. A move / an equip is two env decisions — the piece (kind prep), then its tile or
      // target (kind tile / target, the piece selected) — recorded as such when both are legal; a no-op move is no action.
      const emit = (o, mk, a, kind, label, info = null) => {
        T.calls[label] = (T.calls[label] || 0) + 1;
        if (rec) { s.steps++; put(KIND[kind], o, mk, a, this._reward(s, m, m.round, Math.max(0, ps.lp || 0)), info); }
      };
      const two = (name, first) => {
        const base = inner(name);
        ps[name] = function (...args) {
          before();
          const f = first(...args);
          if (f.skip) return base.apply(this, args);
          let pair = null;
          if (f.i >= 0 && f.a1 >= 0 && mask[f.a1]) {
            const o1 = obs.slice(), m1 = mask.slice(), rows1 = owned ? owned.map((p) => (p ? p.uid : null)) : [];
            ctx.kind = f.kind2; ctx.sel = f.i; before();
            const a2 = f.a2();
            if (a2 >= 0 && mask[a2]) pair = { o1, m1, o2: obs.slice(), m2: mask.slice(), a2, owned: rows1 };
            ctx.kind = 'prep'; ctx.sel = -1;
          }
          const r = base.apply(this, args);
          if (!(r && r.error)) {
            if (pair) {
              const uid = args[0], rows = pair.owned;
              emit(pair.o1, pair.m1, f.a1, 'prep', name, name === 'move' ? { move: uid, rows } : null);
              emit(pair.o2, pair.m2, pair.a2, f.kind2, `${name}:to`, name === 'move' ? { tile: uid } : null);
              ctx.actions += 2;
            }
            else T.unrepresented[name] = (T.unrepresented[name] || 0) + 1;
          }
          return r;
        };
      };
      const idxOf = (uid) => (owned ? owned.findIndex((p) => p && p.uid === uid) : -1);
      two('move', (uid, to, dir) => {
        const loc = ps.find(uid), d0 = dir ?? to?.dir ?? 'RIGHT', i = idxOf(uid);
        if (loc && to?.area === 'board' && loc.area === 'board' && loc.key === `${to.row},${to.col}` && pieceDir(loc.piece) === d0) return { skip: true };
        return { i, a1: i >= 0 ? ACT.MOVE + i : -1, kind2: 'tile', a2: () => {
          if (to?.area === 'board') { const t = tileIndex(to.row, to.col), di = DIRS.indexOf(d0); return t >= 0 && di >= 0 ? ACT.TILE + t * 4 + di : -1; }
          return to?.area === 'hand' && loc && loc.area !== 'hand' ? ACT.BENCH : -1;
        } };
      });
      two('equip', (itemUid, targetUid, replaceUid = null) => {
        const i = idxOf(itemUid);
        return { i, a1: i >= 0 ? ACT.EQUIP + i : -1, kind2: 'target', a2: () => { const j = idxOf(targetUid); return replaceUid == null && j >= 0 ? ACT.TARGET + j : -1; } };
      });
      for (const name of ['destroy', 'unequip']) count(name);
    } else for (const name of ['move', 'destroy', 'equip', 'unequip']) count(name);
    const mPick = Object.getPrototypeOf(m).pickPersonalChoice;
    m.pickPersonalChoice = function (p, idx, id) {
      if (p !== ps) return mPick.call(this, p, idx, id);
      before();
      const r = mPick.call(this, p, idx, id);
      if (!(r && r.error)) { note(idx >= 0 && idx < PERSONAL ? ACT.PERSONAL + idx : -1, 'personal'); ctx.actions++; }
      return r;
    };
    const tt0 = this.timing ? performance.now() : 0;
    try {
      if (this.full) { // legacy's whole prep, its rehearsal (as the env sets it) run inline before prepEnd
        if (this.bossRehearsal !== null) m.botRehearsal = m.round === m.gd.bossRound || m.round === m.gd.hiddenRound ? this.bossRehearsal : this.rehearsal;
        if (this.econ) this._econPrep(m, ps, ctx, startUids, d);
        const job = runSteps((this.econ ? LAYOUT_ONLY : LEGACY_BOT).prepBegin(m, ps, {}));
        // an inline rehearsal job (virtual time: no worker) runs to its end now, so prepEnd applies the rehearsed plan
        if (job && !job.done && !job.remote && typeof job.run === 'function') for (let k = 0; k < 1e5 && !job.run(Infinity); k++); // loop-cap: job slices
        runSteps(LEGACY_BOT.prepEnd(m, ps, job && job.done ? job : null));
      } else runSteps(PROCURE_ONLY.prepBegin(m, ps, {}));
    }
    catch (e) { this.stats.errors++; if (this.errorSamples.length < 20) this.errorSamples.push(`teacher: ${e.stack?.slice(0, 300)}`); }
    finally {
      for (const [name, desc] of saved) { if (desc) Object.defineProperty(ps, name, desc); else delete ps[name]; }
      if (this.timing) { const t = this.timing[m.round || 0] ||= { prep: 0, battle: 0, other: 0, teach: 0, encode: 0, decisions: 0 };
        t.teach += performance.now() - tt0; t.decisions += ctx.actions + 1; }
      delete m.pickPersonalChoice;
    }
    if (ctx.actions > this.encoder.maxActions) T.overCap++;
    before(); note(ACT.END, 'end');
    if (held) this._flushLabels(ps, held, rec, s.traj);
    if (this.full) { if (!ps.ready && d.valid() && !d.complete()) { T.notReady = (T.notReady || 0) + 1; d.fallback(); } return true; }
    if (d.valid() && !d.complete()) d.fallback();
    return true;
  }

  /** v4c: the econ policy's action for seat ps (version 2 observation, sampled at temp; -1 = nothing legal). */
  _econPick(m, ps, ctx2) {
    const E = this.econ;
    if (!E.enc) { E.enc = new Encoder(m.gd, { version: 2 }); E.o = new Float32Array(E.enc.dim); E.mk = new Uint8Array(E.enc.A); }
    const r = E.enc.encode(m, ps, { ...ctx2, pubSnap: undefined, sel: undefined }, E.o, E.mk);
    const { logits } = E.pol.forward(E.o, E.mk, { value: false });
    let mx = -Infinity; for (let a = 0; a < E.enc.A; a++) if (E.mk[a] && logits[a] > mx) mx = logits[a];
    if (mx === -Infinity) return -1;
    let z = 0; const p = new Float64Array(E.enc.A);
    for (let a = 0; a < E.enc.A; a++) if (E.mk[a]) { p[a] = Math.exp((logits[a] - mx) / E.temp); z += p[a]; }
    let u = E.rng() * z;
    for (let a = 0; a < E.enc.A; a++) if (E.mk[a]) { u -= p[a]; if (u <= 0) { E.own = r.own; E.owned = r.owned; return a; } }
    E.own = r.own; E.owned = r.owned;
    for (let a = E.enc.A - 1; a >= 0; a--) if (E.mk[a]) return a;
    return -1;
  }

  /** v4c: the econ policy plays the seat's economy (its calls go through the teacher's wraps, which record them as
   *  version 4 samples) until it ends it; legacy's layout-only prep follows. */
  _econPrep(m, ps, ctx, startUids, d) {
    const S = this.econStats, E = this.econ;
    let bad = 0;
    for (let k = 0; k < MAX_ECON && d.valid() && !ps.ready; k++) {
      const newUids = new Set(ps.allChess().map((p) => p.uid).filter((u) => !startUids.has(u)));
      const a = this._econPick(m, ps, { kind: 'prep', actions: ctx.actions, freezes: ctx.freezes, lp0: ctx.lp0, humans: ctx.humans, rules: ctx.rules,
        newUids, soldBases: ctx.soldBases, cache: (ctx.econCache ||= {}) });
      if (a < 0 || a === ACT.END) { S.ends++; return; }
      S.decisions++;
      let intent = null;
      if (a === ACT.REFRESH) intent = { t: 'g.refresh' };
      else if (a === ACT.LEVEL) intent = { t: 'g.levelUp' };
      else if (a === ACT.FREEZE) intent = { t: 'g.freeze' };
      else if (a < ACT.SELL) intent = { t: 'g.buy', slot: a - ACT.BUY };
      else if (a < ACT.REWARD) { const p = E.owned?.[a - ACT.SELL]; if (p) intent = { t: 'g.sell', uid: p.uid }; }
      else if (a < ACT.PERSONAL) intent = { t: 'g.reward', idx: a - ACT.REWARD };
      else if (a < ACT.CARD) { const pc = E.own?.personalChoice; if (pc) intent = { t: 'g.choice', idx: a - ACT.PERSONAL, choiceId: pc.id }; }
      const r = intent ? m._handleSeatIntent(ps, intent) : { error: 'BAD_ACTION' };
      if (r?.error) { S.refused++; if (++bad >= 3) return; } // refused calls (a rule under the prep-start view): give up after 3
    }
    S.capped++;
  }

  /** Placement labels of one held prep (see placementLabels), then its samples to the recorder in order. */
  _flushLabels(ps, held, rec, traj) {
    const L = this.labelStats, later = new Set();
    for (let j = held.length - 1; j >= 0; j--) { // the uids still moved at or after each MOVE sample
      const h = held[j];
      if (h.info?.move != null) {
        later.add(h.info.move); L.moves++;
        let bits = 0;
        (h.info.rows || []).forEach((u, i) => { if (u != null && later.has(u) && i < 32) bits |= (1 << i); });
        h.aux = bits >>> 0;
      }
    }
    for (const h of held) {
      if (h.info?.tile == null) continue;
      L.tiles++;
      const loc = ps.find(h.info.tile);
      let b = -1;
      if (loc && loc.area === 'board') { const [r, c] = loc.key.split(',').map(Number), t = tileIndex(r, c), di = DIRS.indexOf(pieceDir(loc.piece)); if (t >= 0 && di >= 0) b = ACT.TILE + t * 4 + di; }
      else if (loc && loc.area === 'hand') b = ACT.BENCH;
      if (b >= 0 && b !== h.a) { if (h.mk[b]) { h.a = b; L.relabeled++; } else L.keptIllegal++; }
    }
    for (const h of held) rec.sample(h.kind, traj, h.o, h.mk, h.a, h.rw, h.aux || 0);
  }

  /** Survival summary of finished matches, per config and seat role (rl / legacy / human-legacy / human-rl). */
  summary() {
    const g = new Map();
    for (const r of this.results) for (const s of r.seats) {
      const key = `${r.config}:${s.role || (s.rl ? 'rl' : 'legacy')}`;
      const x = g.get(key) || { key, seats: 0, survival: 0, wins: 0, lp: 0, tier: 0, tiers: 0 };
      x.seats++; x.survival += s.survivalRound; x.wins += s.win; x.lp += s.lp;
      if (s.mainTier != null) { x.tier += s.mainTier; x.tiers++; }
      g.set(key, x);
    }
    return [...g.values()].map((x) => ({ key: x.key, seats: x.seats, survival: +(x.survival / x.seats).toFixed(2), winRate: +(x.wins / x.seats).toFixed(3),
      lp: +(x.lp / x.seats).toFixed(1), ...(x.tiers ? { humanMainTier: +(x.tier / x.tiers).toFixed(2) } : {}) }));
  }

  /** Rule metrics per driver of the AI seats (see _instrument), totals and per match. */
  yieldSummary() {
    const g = {};
    for (const r of this.results) for (const [role, y] of Object.entries(r.yield)) {
      const x = g[role] ||= { matches: 0, preps: 0, buys: 0, sells: 0, yieldMain: 0, yieldMerge: 0, roundTrips: 0 };
      x.matches++;
      // preps of this role's seats: one per round each reached
      x.preps += r.seats.filter((s) => s.role === role).reduce((n, s) => n + s.survivalRound, 0);
      for (const k of ['buys', 'sells', 'yieldMain', 'yieldMerge', 'roundTrips']) x[k] += y[k];
    }
    for (const x of Object.values(g)) {
      for (const k of ['yieldMain', 'yieldMerge']) x[`${k}PerMatch`] = +(x[k] / Math.max(1, x.matches)).toFixed(2);
      x.roundTripsPerPrep = +(x.roundTrips / Math.max(1, x.preps)).toFixed(3);
    }
    return g;
  }

  close() { for (const slot of this.slots) if (slot) try { slot.m.dispose(); } catch { /* */ } }
}

/** A uniformly random legal action per row. */
export function randomActions(batch, rng = Math.random, out = new Int32Array(batch.n)) {
  const { n, mask, A } = batch;
  for (let k = 0; k < n; k++) {
    let c = 0, pick = 0;
    for (let a = 0; a < A; a++) if (mask[k * A + a]) { c++; if (rng() * c < 1) pick = a; }
    out[k] = pick;
  }
  return out;
}
