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
import { xPlaceProblemForTraining as xPlaceProblemP, createBotPrepSteps, runSteps, layoutCandidatesForPicker, applyPlanForPicker, positionalCandidatesForTraining, layoutFeatures, rehearseCandidatesForTraining, candidatePoolForTraining, teacherSearchForTraining, oracleSigForTraining } from '../../server/match/bot.js';
import { pieceDir as pieceDirP } from '../../server/match/board.js';
import { appendFileSync as appendFileSyncP, writeFileSync as writeFileSyncP, existsSync as existsSyncP } from 'node:fs';
import { PickerHead, pickerRow } from './picker.mjs';
import { collectViolations } from '../../server/match/invariants.js';
import { snapshotMatch, restoreMatch } from '../../server/match/snapshot.js';
import { createRngFromState } from '../../server/sim/rng.js';
import { Encoder, ACT, KIND, SHOP, OWNED, REWARDS, PERSONAL, CARDS, BANDS, DIRS, tileRC, tileIndex, humanBonds, humanPairs, chessInBonds, mainCoreBondOf, rulesOf, bandConflicts } from './spec.mjs';
import { beginSeatPlan, endSeatPlan, seatPlanOf, seatPlanRerolls } from '../../server/match/seatPlan.js';
import { pieceDir } from '../../server/match/board.js';
import { withBounties } from '../../server/match/waves.js';
import { SPPolicy2 } from './infer2.mjs';
import { SPPolicy5 } from './infer5.mjs';
import { readFileSync as readFileSyncP } from 'node:fs';
import { recipePlan } from './recipes.mjs';
import { XPlacer } from './xplace.mjs';
import { xFeaturize as xFeaturizeP, xAssignOf as xAssignOfP, xCandFeatures as xCandFeaturesP } from './xfeat.mjs'; // (picker boss labels: the xplace Q on legacy K8)
import { createLabelHook } from './xsearch.mjs';
import { HumanProxies } from './proxies.mjs'; // (N8) co-op proxies teammate proxy pool (trackE-v2 cfabf0d3)
// (N5) the route bots' national-route bot (legacy buy loop + NAT_POLICY), the plan-known teacher of natPlan.teacher 'nat'
import { natSetup, NAT_PROCURE_ONLY, NATIONS as M_NATIONS, PAIRS as M_PAIRS } from './trackm/routebots.mjs';

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
// (N1) natPlan.slots: plan-block extras (distinct owned P / S members) and per-slot features (in P, in S, new / sole P member)
const PLAN_X = 2, PLAN_SW = 3;
// (N1) natPlan.head: one 'planNat' row per nation (PLAN_NF numeric features + a one-hot of the nation) and one extra action per
// nation after the encoder's actions; a plan decision (kind KIND_PLAN) picks the nation P (S: the rule in _ruleS)
const PLAN_NF = 8, KIND_PLAN = 5;
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
    // (A1, 11-13, arch) version 3 extension blocks (spec.mjs ext): obsExt { board, hist }; off (null) = version 3 as before.
    // hist keeps, per seat of a match, its own battles (leaks, LP lost, layers gained, kills) and the shop rolls it saw
    this.obsExt = o.obsExt && this.obsVersion === 3 && (o.obsExt.board || o.obsExt.hist) ? { board: !!o.obsExt.board, hist: !!o.obsExt.hist } : null;
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
    // band table (11-13): with bandPolicy on, ai_0's own band decision (after every human proxy, first of the model seats)
    // takes the forced band when the seat's mask allows it and no human's band bonds / 主盟约 meet its tied bonds (else the
    // policy's pick stands and the seed counts as not obtained); 'random' = a uniformly random allowed band (the baseline).
    // res.forced per match. Off (null) unless forceBand is set with bandPolicy.
    // team combo table (11-13): forceBand 'team:a,b,c' = the model seats' own band decisions take a, b, c in their pick order
    // (after the human proxies; each under its own mask and the human-bond rule, else the policy's pick stands: res.team per match)
    this.forceTeam = typeof this.forceBand === 'string' && this.forceBand.startsWith('team:') && this.bandPolicy ? this.forceBand.slice(5).split(',').filter(Boolean) : null;
    this.forceAt = this.forceBand && this.bandPolicy && !this.forceTeam ? 'policy' : null;
    // action histogram (11-13, diagnostics, off): per RL seat the policy's applied actions, buys / rewards by kind, 机变 picks
    // by family, its bounty units spawned / leaked at normal-round battles, the items legacy equips; res.act per match
    this.actLog = !!o.actLog;
    // bounty-aware legacy layout (11-13 experiment, off): an RL seat's legacy layout at a normal round models its own bounty
    // spawns too (bot.js fieldModel reads m.wave.spawns only; rehearsal adds the bounties) — env-side, bot.js unchanged
    this.bountyField = !!o.bountyField;
    this.bfStats = { swaps: 0, seatPreps: 0 };
    // boss-round layout picker (11-13, track 2, off): pickerHead { head: picker head JSON, trunk: frozen JS trunk export
    // base, K }; at the boss / hidden-core round an RL seat's legacy layout (after LAYOUT_ONLY's final arrange, before
    // prepEnd) is replaced by the head's pick among K legacy candidates (planLayout only: no battle may be built)
    // national-route plan (11-13, NAT arm; off): every RL seat (not a human proxy) gets a plan pair at its first decision after
    // the band — nation P (bondType SEASON; none tied to a human's band bonds / 主盟约, none another RL seat of the match holds)
    // and a class bond S carried by P's operators — shown in an obs block 'plan' appended after the encoder's layout. R2 terms
    // (rewardV 2): at match end, `bonus` x (0.5 + 0.5 faShare) if P (phase 'free': any nation) was at tier >= 2 at the Final
    // Assault start, + `sec` if S (free: any class stacking bond) was active there with >= secLayers layers; and the potential
    // phiCount x min(P on board, 6) / 6 + phiLayers x P layers / 100 (free: the best nation) inside the shaping term.
    this.natPlan = o.natPlan ? { phase: 'forced', bonus: 1.5, sec: 0.5, secLayers: 40, phiCount: 0.5, phiLayers: 0.3, ...o.natPlan } : null;
    // (N5) natPlan.teacher 'nat' (teacher mode): RL seats procure with the route bots' nat bot toward their plan nation (natSetup + NAT_PROCURE_ONLY)
    this.natTeach = !!(this.natPlan && this.natPlan.teacher === 'nat' && this.mode === 'teacher');
    // teammate proxy pool (11-13, co-op proxies; off): the human seats are driven by tools/rl/proxies.mjs kinds instead of
    // legacy (legacy kind: unchanged), with their own normal-round rehearsal and (log) a bond log in the match result
    this.humanProxy = o.humanProxy ? new HumanProxies(o.humanProxy) : null;
    this.picker = o.pickerHead ? { head: PickerHead.load(o.pickerHead.head), trunk: SPPolicy2.load(o.pickerHead.trunk), K: o.pickerHead.K || 8 } : null;
    // diagnostics (track 2): readyHook(m, ps, slot) runs right before a legacy-placed seat readies (its final layout,
    // after any rehearsal), e.g. a candidate pick; seatPlanBonds { ai_0: bondId }: legacy's focus bond for those seats
    this.readyHook = o.readyHook || null;
    this.equipHook = o.equipHook || null; // diagnostics: replaces equipItems for the policy (LAYOUT_ONLY) seats
    this.seatPlanBonds = o.seatPlanBonds || null;
    this.bountyAware = !!o.bountyAware; // diagnostics: legacy's field model of the policy seats adds their bounty spawns
    // normal-round layout picker (11-13, picker, off): normalPicker { head: <base>.json of a spatial head (export_p5.py, kind 'p5-picker'),
    // K, from, to, mode, margin }: at a normal round in [from, to] an RL seat's legacy layout (after LAYOUT_ONLY's final arrange, before
    // prepEnd) is replaced by the head's pick among K legacy candidates (planLayout only: no battle may be built). The head is the v5
    // tower over each candidate's version 3 + ext.board observation (Encoder.boardSlice); mode 'reg' (default): switch to the candidate
    // whose predicted −2 Δleaks + Δlayers / 5 + 0.05 Δkills vs the default exceeds `margin` (default 0.5); mode 'score': the score argmax.
    if (o.normalPicker && JSON.parse(readFileSyncP(o.normalPicker.head, 'utf8')).kind !== 'p5-picker') throw new Error('normalPicker: not a p5-picker head');
    this.npicker = o.normalPicker ? { spatial: SPPolicy5.load(o.normalPicker.head.replace(/\.json$/, '')), mode: o.normalPicker.mode || 'reg', margin: o.normalPicker.margin ?? 0.5,
      K: o.normalPicker.K || 8, from: o.normalPicker.from ?? 1, to: o.normalPicker.to ?? 13, enc: null, w: o.normalPicker.w || [2, 0.2, 0.05],
      pool: !!o.normalPicker.pool, edits: o.normalPicker.edits ?? 12, max: o.normalPicker.max ?? 32, tmpl: o.normalPicker.tmpl ?? 6 } : null;
    // normal-round labels (picker, off; training data only): pLabel { dir, K, tmpl, from, to, apply 'label'|'none', seed 'fixed'|'match' }: at every
    // normal round of an RL seat the K legacy candidates (+ tmpl stacking-aware) are rehearsed once each; one JSON line per decision in
    // <dir>/lab-<pid>.jsonl (version 3 + ext.board obs, per-candidate ext.board slices, assignments, scores); the D-rule best applied unless 'none'
    this.plabel = o.pLabel ? { K: 8, tmpl: 0, from: 1, to: 13, apply: 'label', seed: 'fixed', ...o.pLabel, enc: null } : null;
    // boss-round labels (picker, off; training only): pBossLabel { dir, K, seeds (1|2) }: at the boss / hidden-core round of an RL seat, BEFORE any boss
    // picker or the xplace ensemble acts, legacy's K candidates are each rehearsed with per-match boss seeds (mean of `seeds`); one line per decision in
    // <dir>/boss-<pid>.jsonl in posvalue pickrec's format (kind 'boss': v3 obs sparse, c of the boss picker's trunk, pickerRow rows, scores)
    this.pbl = o.pBossLabel ? { K: 8, seeds: 2, ...o.pBossLabel } : null;
    // (code-P9, picker, off) pBossLabel also takes qx { name: <X QNet model JSON> } (each net's Q on the same K, recorded as qx) and
    // rehearse: false (no rehearsal: scores null; records still written). bossSpatial { head: <base>.json (export_p5 of P/b1/bpick5.py,
    // kind 'p5-picker'), w (default 1), K (default 8), out 'score' (default, output 0) | 'share' (output 1), rounds 'fa' (default: the
    // Final Assault only) | 'boss' }: in the xplace QNet pick (xPlace) the spatial boss picker's z-scored output over legacy's K candidates (v3 +
    // ext.board obs with each candidate's board slice, as normalPicker) is added with weight w to z(Q) (+ ens x z(head) if xPlace ens > 0).
    if (o.bossSpatial && JSON.parse(readFileSyncP(o.bossSpatial.head, 'utf8')).kind !== 'p5-picker') throw new Error('bossSpatial: not a p5-picker head');
    this.bspatial = o.bossSpatial ? { spatial: SPPolicy5.load(o.bossSpatial.head.replace(/\.json$/, '')), w: o.bossSpatial.w ?? 1, K: o.bossSpatial.K || 8,
      out: o.bossSpatial.out === 'share' ? 1 : 0, rounds: o.bossSpatial.rounds || 'fa', enc: null } : null;
    // xplace (11-13, all off): placeHook(m, ps, slot) runs right after the prepBegin of an RL seat (not a human proxy) or a
    // recipe seat — the label generator (tools/rl/xlabel.mjs); recipeSeats { recipes, seats: { pid: k } } (L10 port): those
    // seats play legacy with recipe k's advisor plan and open with its first band; xPlace { model, rounds 'all' | 'normal' |
    // 'boss' }: the learned layout policy (tools/rl/xplace.mjs, no battle may be built) replaces an RL seat's legacy layout
    this.placeHook = o.placeHook || null;
    this.recipeSeats = o.recipeSeats && o.recipeSeats.recipes?.length ? o.recipeSeats : null;
    this.xplace = o.xPlace ? XPlacer.fromOptions(o.xPlace) : null;
    this.xplaceModel = o.xPlace && o.xPlace.model ? String(o.xPlace.model).split('/').slice(-2).join('/') : null; // (picker boss labels: tag)
    // xLabel { out, K, ls, seeds, ... } (tools/rl/xsearch.mjs): the search-label hook as placeHook, one <out>-<seed>-<pid> file pair per env
    if (o.xLabel) { this._xlab = createLabelHook(null, { ...o.xLabel, out: `${o.xLabel.out}-${o.seed}-${process.pid}` }); this.placeHook = this._xlab.hook; }
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
    this.D = this.encoder.dim + (this.natPlan ? this._planWidth() : 0); this.A = this.encoder.A + (this.natPlan?.head ? this._natBonds(this.encoder.gd).nations.length : 0);
    this._buf(64);
  }

  setConfigs(list) {
    this.configs = list.map((c) => (typeof c === 'string' ? { ...parseConfig(c), w: 1 } : { ...parseConfig(c.name), w: c.w ?? 1 }));
    this.cw = this.configs.reduce((s, c) => s + c.w, 0);
  }

  spec() { return { ...this._encSpec(), reward: this.R, configs: this.configs.map((c) => c.name), termWidth: this.termWidth,
    ...(this.rewardV === 2 ? { rewardV: 2, r2: this.R2 } : {}) }; }

  /** The encoder's spec, plus the appended 'plan' block when natPlan is on. */
  _encSpec() {
    const sp = this.encoder.spec();
    if (!this.natPlan) return sp;
    if (!this.natPlan.slots && !this.natPlan.head) return { ...sp, dim: this.D, blocks: { ...sp.blocks, plan: { offset: this.encoder.dim, count: 1, width: this._planWidth() } } };
    // (N1) plan slots: the plan block's own row, then one row per shop slot / reward slot / owned row (model.py appends each
    // to that entity's row input); plan head: one planNat row per nation (a pointer block: its logits are the extra actions)
    const w0 = 1 + 2 * this.encoder.vocab.bondIds.length + 6 + (this.natPlan.slots ? PLAN_X : 0); let q = this.encoder.dim + w0;
    const blocks = { ...sp.blocks, plan: { offset: this.encoder.dim, count: 1, width: w0 } };
    if (this.natPlan.slots) {
      blocks.planShop = { offset: q, count: SHOP, width: PLAN_SW }; q += PLAN_SW * SHOP;
      blocks.planRewards = { offset: q, count: REWARDS, width: PLAN_SW }; q += PLAN_SW * REWARDS;
      blocks.planOwned = { offset: q, count: OWNED, width: PLAN_SW }; q += PLAN_SW * OWNED;
    }
    if (this.natPlan.head) { const NN = this._natBonds(this.encoder.gd).nations.length; blocks.planNat = { offset: q, count: NN, width: PLAN_NF + NN }; }
    return { ...sp, dim: this.D, actions: this.A, blocks, ...(this.natPlan.head ? { planHead: { action0: this.encoder.A, nations: this._natBonds(this.encoder.gd).nations } } : {}) };
  }

  _planWidth() { const NN = this.natPlan?.head ? this._natBonds(this.encoder.gd).nations.length : 0;
    return 1 + 2 * this.encoder.vocab.bondIds.length + 6 + (this.natPlan?.slots ? PLAN_X + PLAN_SW * (SHOP + REWARDS + OWNED) : 0) + NN * (PLAN_NF + NN); }

  _natBonds(gd) {
    if (this._nat) return this._nat;
    const ids = this.encoder.vocab.bondIds, rec = (b) => gd.bond(b) || {};
    const nations = ids.filter((b) => rec(b).bondType === 'SEASON');
    const classes = new Set(ids.filter((b) => rec(b).bondType === 'REGULAR' && !rec(b).noStack && (rec(b).perStackParams || []).length));
    const secOf = new Map(nations.map((n) => {
      const mem = (rec(n).members || []).map((id) => gd.chess(id)).filter((c) => c && !c.isHidden);
      return [n, [...new Set(mem.flatMap((c) => (c.bonds || []).filter((b) => classes.has(b))))].sort()];
    }));
    this._nat = { nations, classes, secOf, index: new Map(ids.map((b, i) => [b, i])) };
    return this._nat;
  }

  /** (N1) natPlan.live: the bonds switched off in this match (drawn bans + static: their operators never enter the pool). */
  _offBonds(m) {
    const ex = this.natPlan?.exclude; // (N6) natPlan.exclude: nations never planned (e.g. ['egirShip']: devour order needs placement we lack)
    if (!this.natPlan?.live && !(ex && ex.length)) return null;
    return new Set([...(this.natPlan.live ? [...(m.disabledBonds || []), ...(m.staticInactiveBonds || [])] : []), ...(ex || [])]);
  }

  /** The seat's plan pair (P, S), drawn once per seat from a seat-and-seed stream. */
  _assignPlan(slot, ps, prev = null) {
    if (this.natPlan.assign === 'lead') return this._leadPlan(slot, ps, prev);
    if (this.natPlan.assign === 'auto' || this.natPlan.assign === 'mix') return this._autoPlan(slot, ps);
    const m = slot.m, N = this._natBonds(m.gd), bc = bandConflicts(m.gd, m.publicView(), ps.playerId, slot.humans, null);
    const taken = (slot.natTaken ||= new Set()), off = this._offBonds(m), nat = off ? N.nations.filter((n) => !off.has(n)) : N.nations;
    const hn = this.natPlan.coopReplan ? this._humanNat(slot, ps) : null; let ok = nat.filter((n) => !bc.humanBand.has(n) && !bc.humanMain.has(n) && !taken.has(n) && !(hn && hn.has(n)));
    if (!ok.length) ok = nat.filter((n) => !taken.has(n));
    if (!ok.length) ok = nat.length ? nat : N.nations;
    const rng = mulberry32((slot.seed ^ Math.imul((Number(String(ps.playerId).split('_')[1]) || 0) + 1, 0x9E3779B1) ^ 0x7A11) >>> 0);
    const P = ok[Math.floor(rng() * ok.length)], ss = (N.secOf.get(P) || []).filter((c) => !off || !off.has(c));
    let S = ss.length ? ss[Math.floor(rng() * ss.length)] : null;
    if (this.natPlan.pairS) S = this._pairS(m, P) ?? S;
    taken.add(P);
    return { P, S };
  }

  /** (N8) natPlan.coopReplan: the bonds living humans build or hold by band (their public main core bond, band bonds: the
   *  让盟约 protected set), never planned and a reason to replan. */
  _humanNat(slot, ps) { return slot.humans.size ? humanBonds(slot.m.gd, slot.m.publicView(), ps.playerId, slot.humans) : new Set(); }

  /** (N5) the route bots' community secondary for P (first live bond of its PAIRS list), or null. */
  _pairS(m, P) { const off = new Set([...(m.disabledBonds || []), ...(m.staticInactiveBonds || [])]); return (M_PAIRS[P] || []).find((b) => !off.has(b)) || null; }

  /** (N5) natPlan.assign 'auto': P = the first allowed (live with `live`) nation in the route bots' value order (recipes-M.json rank);
   *  'mix': a uniformly random allowed nation with probability mixP (default 0.5), else 'auto'. S = the route bots' pair (fallback: the
   *  first live class bond of P's operators). */
  _autoPlan(slot, ps) {
    const m = slot.m, N = this._natBonds(m.gd), bc = bandConflicts(m.gd, m.publicView(), ps.playerId, slot.humans, null);
    const taken = (slot.natTaken ||= new Set()), off = this._offBonds(m), nat = off ? N.nations.filter((n) => !off.has(n)) : N.nations;
    const hn = this.natPlan.coopReplan ? this._humanNat(slot, ps) : null; let ok = nat.filter((n) => !bc.humanBand.has(n) && !bc.humanMain.has(n) && !taken.has(n) && !(hn && hn.has(n)));
    if (!ok.length) ok = nat.filter((n) => !taken.has(n));
    if (!ok.length) ok = nat.length ? nat : N.nations;
    const rng = mulberry32((slot.seed ^ Math.imul((Number(String(ps.playerId).split('_')[1]) || 0) + 1, 0x9E3779B1) ^ 0xA070) >>> 0);
    const rand = this.natPlan.assign === 'mix' && rng() < (this.natPlan.mixP ?? 0.5);
    const P = rand ? ok[Math.floor(rng() * ok.length)] : (M_NATIONS.find((n) => ok.includes(n)) || ok[0]);
    const S = this._pairS(m, P) ?? ((N.secOf.get(P) || []).find((c) => !off || !off.has(c)) || null);
    taken.add(P);
    return { P, S };
  }

  /** (N5) natPlan.teacher 'nat': the seat's plan (natPlan.assign) handed to the route bots' natSetup once; s.plan = the bot's own (P, S). */
  _natTeachSetup(slot, ps, s, atBand = false) {
    if (ps._nat) return;
    if (atBand && this.natPlan.head && this.natPlan.planAtBand && this.natPlan.hindsight && this.recorder) { // (N7) the plan sample at the band stage
      const m = slot.m, obs = new Float32Array(this.encoder.dim), mask = new Uint8Array(this.encoder.A);
      const aiPicks = new Set([...m.players.keys()].filter((pid) => !slot.humans.has(pid) && m.draft?.picks?.[pid]));
      const r = this.encoder.encode(m, ps, { kind: 'band', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, newUids: new Set(),
        soldBases: new Set(), cache: {}, aiPicks }, obs, mask);
      (slot.natBuf ||= []).push({ pid: ps.playerId, kind: KIND_PLAN, traj: s.traj, o: obs.slice(), mk: null, a: -1, plan: true, first: true,
        rw: this._reward(s, m, m.round, Math.max(0, ps.lp || 0)), view: this._planView(r.own, r.owned), allowed: this._planAllowed(slot, ps, null), round: m.round || 0 });
      s.steps++; (s.hsPlans ||= new Set()).add(-1);
    }
    const plan = this._assignPlan(slot, ps);
    const N = natSetup(slot.m, ps, plan.P);
    s.plan = { P: N.P, S: N.S };
  }

  /** (N5) natPlan.band: the plan nation a seat's (legacy-picked) band should carry, fixing the plan now when needed. */
  _bandPlanP(slot, ps) {
    const s = this._seat(slot, ps);
    if (this.natTeach) this._natTeachSetup(slot, ps, s, true);
    else if (!s.plan && !this.natPlan.head && this.natPlan.assign !== 'lead') s.plan = this._assignPlan(slot, ps);
    return s.plan?.P || null;
  }

  /** (N1) natPlan.assign 'lead': P = the allowed nation with the most distinct owned members (board, hand, temp), the previous
   *  P on ties, then a seat-and-seed order; S = the class bond of P's operators with the most distinct owned members (the
   *  previous S on ties). Re-planned at every decision before round lockRound (default 4), then fixed. */
  _leadPlan(slot, ps, prev) {
    const m = slot.m, gd = m.gd, N = this._natBonds(gd), bc = bandConflicts(gd, m.publicView(), ps.playerId, slot.humans, null);
    const taken = (slot.natTaken ||= new Set()); if (prev) taken.delete(prev.P);
    const off = this._offBonds(m), nat = off ? N.nations.filter((n) => !off.has(n)) : N.nations;
    const hn = this.natPlan.coopReplan ? this._humanNat(slot, ps) : null; let ok = nat.filter((n) => !bc.humanBand.has(n) && !bc.humanMain.has(n) && !taken.has(n) && !(hn && hn.has(n)));
    if (!ok.length) ok = nat.filter((n) => !taken.has(n));
    if (!ok.length) ok = nat.length ? nat : N.nations;
    const bases = new Map(); for (const p of ps.allChess()) bases.set(gd.baseIdOf(p.id), p.id);
    const cnt = (b) => { let k = 0; for (const id of bases.values()) if ((gd.chess(id)?.bonds || []).includes(b)) k++; return k; };
    const rng = mulberry32((slot.seed ^ Math.imul((Number(String(ps.playerId).split('_')[1]) || 0) + 1, 0x9E3779B1) ^ 0x1EAD) >>> 0);
    const order = new Map(ok.map((n) => [n, rng()]));
    let P = null, pk = -Infinity;
    for (const n of ok) { const k = cnt(n) * 10 + (prev && prev.P === n ? 1 : 0) + order.get(n) * 0.5; if (k > pk) { pk = k; P = n; } }
    const ss = (N.secOf.get(P) || []).filter((c) => !off || !off.has(c));
    let S = null, sk = -Infinity;
    for (const c of ss) { const k = cnt(c) * 10 + (prev && prev.S === c ? 1 : 0) + rng() * 0.5; if (k > sk) { sk = k; S = c; } }
    if (this.natPlan.pairS) S = this._pairS(m, P) ?? S;
    taken.add(P);
    return { P, S };
  }

  /** (N1) The plan a seat played, read off its final board (hindsight label for plan-conditioned BC): P = the nation with
   *  the highest tier, then most units; S = the active class stacking bond with the most layers (null: none). */
  _hindsightPlan(gd, ps, fallback = null, known = null) {
    const N = this._natBonds(gd);
    let bonds = ps ? ps.privateView().bonds || [] : [];
    // an eliminated seat's board is gone at the end: then its last decision's bond view (board as placed)
    if (fallback && !N.nations.some((n) => SPEnv._bv(bonds, n).count > 0)) bonds = fallback;
    let P = N.nations[0], pv = null, pk = -1;
    for (const n of N.nations) { const v = SPEnv._bv(bonds, n), k = v.tier * 100 + v.count; if (k > pk) { pk = k; P = n; pv = v; } }
    let S = null, sv = null;
    for (const c of N.classes) { const v = SPEnv._bv(bonds, c); if (v.active && (!sv || v.layers > sv.layers)) { sv = v; S = c; } }
    if (known) { P = known.P; pv = SPEnv._bv(bonds, P); S = known.S || null; sv = S ? SPEnv._bv(bonds, S) : null; } // (N5) natPlan.known: the seat's own plan
    return { P, S, pTier: pv?.tier || 0, pCount: pv?.count || 0, pLayers: pv?.layers || 0, sLayers: sv?.layers || 0 };
  }

  /** (N1) What the plan-slot features need of a decision's state, as plain ids (a snapshot: hindsight samples are filled in
   *  at the match end). */
  _planView(own, owned) {
    const ch = (x) => (x && !x.sold && x.kind === 'chess' ? x.id : null);
    return { bonds: (own?.bonds || []).map((b) => ({ bondId: b.bondId, tier: b.tier, count: b.count, layers: b.layers, active: b.active })),
      shop: (own?.shop?.slots || []).slice(0, SHOP).map(ch), rewards: (own?.shop?.rewardOffer?.slots || []).slice(0, REWARDS).map(ch),
      owned: (owned || []).slice(0, OWNED).map((p) => (p && p.kind === 'chess' ? p.id : null)) };
  }

  /** (N1) Plan-slot features at out[q..]: distinct owned P / S members, then per shop slot / reward slot [in P, in S, a new
   *  distinct P member], per owned row [in P, in S, the only owned copy of a P member]. */
  _planSlots(view, plan, out, q) {
    const gd = this.encoder.gd, inB = (id, b) => !!(b && id && (gd.chess(id)?.bonds || []).includes(b));
    const own = new Map(); for (const id of view.owned) if (id) { const b = gd.baseIdOf(id); const e = own.get(b); if (e) e.n++; else own.set(b, { id, n: 1 }); }
    let dP = 0, dS = 0; for (const { id } of own.values()) { if (inB(id, plan.P)) dP++; if (inB(id, plan.S)) dS++; }
    out[q] = Math.min(dP, 9) / 6; out[q + 1] = Math.min(dS, 9) / 3; q += PLAN_X;
    const offer = (ids, n) => { for (let i = 0; i < n; i++, q += PLAN_SW) { const id = ids[i]; if (!id) continue;
      const p = inB(id, plan.P); out[q] = p ? 1 : 0; out[q + 1] = inB(id, plan.S) ? 1 : 0; out[q + 2] = p && !own.has(gd.baseIdOf(id)) ? 1 : 0; } };
    offer(view.shop, SHOP); offer(view.rewards, REWARDS);
    for (let i = 0; i < OWNED; i++, q += PLAN_SW) { const id = view.owned[i]; if (!id) continue;
      const p = inB(id, plan.P); out[q] = p ? 1 : 0; out[q + 1] = inB(id, plan.S) ? 1 : 0; out[q + 2] = p && own.get(gd.baseIdOf(id))?.n === 1 ? 1 : 0; }
  }

  /** (N1) natPlan.dense: one-time milestone rewards (no cycling): `unit` per new high of the target nation's units on board
   *  (cap 6), `t2` when it first reaches tier 2, `secOn` when the secondary (forced: S; free: the best active class bond)
   *  is first active, `secPay` per layer milestone of `secL` it first reaches. */
  _natDense(s, bonds) {
    const D = this.natPlan.dense, ms = (s.natMs ||= { cnt: 0, t2: false, sOn: false, sL: 0, sum: 0 });
    const t = SPEnv._bv(bonds, this._natTarget(bonds, s.plan)), c = Math.min(t.count, 6);
    let r = 0;
    if (c > ms.cnt) { r += (D.unit || 0) * (c - ms.cnt); ms.cnt = c; }
    if (t.tier >= 2 && !ms.t2) { r += D.t2 || 0; ms.t2 = true; }
    let sv = null;
    if (this.natPlan.phase === 'free') { for (const c2 of this._natBonds(this.encoder.gd).classes) { const v = SPEnv._bv(bonds, c2); if (v.active && (!sv || v.layers > sv.layers)) sv = v; } }
    else if (s.plan.S) sv = SPEnv._bv(bonds, s.plan.S);
    if (sv && sv.active) {
      if (!ms.sOn) { r += D.secOn || 0; ms.sOn = true; }
      for (const L of D.secL || []) if (L > ms.sL && sv.layers >= L) { r += D.secPay || 0; ms.sL = L; }
    }
    ms.sum += r;
    return r;
  }

  /** Bond view helpers: { tier, count, layers, active } of a bond in a privateView bonds list. */
  static _bv(bonds, id) { const b = (bonds || []).find((x) => x.bondId === id); return b ? { tier: b.active ? (b.tier || 0) : 0, count: b.count || 0, layers: b.layers || 0, active: !!b.active } : { tier: 0, count: 0, layers: 0, active: false }; }

  /** The nation the potential and the free-phase bonus follow: P (forced) or the seat's best nation (most units, then tier). */
  _natTarget(bonds, plan) {
    if (this.natPlan.phase !== 'free') return plan.P;
    const N = this._natBonds(this.slots.find(Boolean)?.m.gd || this.encoder.gd);
    let best = plan.P, bk = -1;
    for (const n of N.nations) { const v = SPEnv._bv(bonds, n), k = v.count * 10 + v.tier; if (k > bk) { bk = k; best = n; } }
    return best;
  }

  _natPhi(bonds, plan) {
    const t = SPEnv._bv(bonds, this._natTarget(bonds, plan));
    return this.natPlan.phiCount * Math.min(t.count, 6) / 6 + this.natPlan.phiLayers * t.layers / 100;
  }

  _planObs(bonds, plan, out, o, view = null) {
    const N = this._natBonds(this.encoder.gd), NB = this.encoder.vocab.bondIds.length;
    out[o] = 1; out[o + 1 + N.index.get(plan.P)] = 1; if (plan.S) out[o + 1 + NB + N.index.get(plan.S)] = 1;
    const p = SPEnv._bv(bonds, plan.P), sv = plan.S ? SPEnv._bv(bonds, plan.S) : { tier: 0, layers: 0, active: false }, q = o + 1 + 2 * NB;
    out[q] = Math.min(p.count, 9) / 6; out[q + 1] = p.tier / 2; out[q + 2] = Math.min(p.layers, 999) / 100; out[q + 3] = sv.active ? 1 : 0;
    out[q + 4] = Math.min(sv.layers, 999) / 100; out[q + 5] = this.natPlan.phase === 'free' ? 0 : 1;
    if (this.natPlan.slots && view) this._planSlots(view, plan, out, q + 6);
  }

  /** (N1) natPlan.head: allowed nations of a seat's plan decision (not tied to a human's band / 主盟约, not another RL seat's P). */
  _planAllowed(slot, ps, cur = null) {
    const m = slot.m, N = this._natBonds(m.gd), bc = bandConflicts(m.gd, m.publicView(), ps.playerId, slot.humans, null);
    const taken = new Set(slot.natTaken || []); if (cur) taken.delete(cur);
    const off = this._offBonds(m), nat = off ? N.nations.filter((n) => !off.has(n)) : N.nations;
    const hn = this.natPlan.coopReplan ? this._humanNat(slot, ps) : null; let ok = nat.filter((n) => !bc.humanBand.has(n) && !bc.humanMain.has(n) && !taken.has(n) && !(hn && hn.has(n)));
    if (!ok.length) ok = nat.filter((n) => !taken.has(n));
    return new Set(ok.length ? ok : nat.length ? nat : N.nations);
  }

  /** (N1) natPlan.head: S for a chosen P = P's class bond with the most distinct owned members (the previous S on ties, then a
   *  seat-and-seed order). */
  _ruleS(slot, ps, P, prev = null) {
    const gd = slot.m.gd, N = this._natBonds(gd), bases = new Map(); for (const p of ps.allChess()) bases.set(gd.baseIdOf(p.id), p.id);
    const cnt = (b) => { let k = 0; for (const id of bases.values()) if ((gd.chess(id)?.bonds || []).includes(b)) k++; return k; };
    const rng = mulberry32((slot.seed ^ Math.imul((Number(String(ps.playerId).split('_')[1]) || 0) + 1, 0x9E3779B1) ^ 0x5EC0 ^ (slot.m.round || 0)) >>> 0);
    let S = null, sk = -Infinity;
    const off = this._offBonds(slot.m);
    for (const c of N.secOf.get(P) || []) { if (off && off.has(c)) continue; const k = cnt(c) * 10 + (prev && prev.S === c ? 1 : 0) + rng() * 0.5; if (k > sk) { sk = k; S = c; } }
    if (this.natPlan.pairS) S = this._pairS(slot.m, P) ?? S;
    return S;
  }

  /** (N1) planNat rows at out[q..]: per nation [1, allowed, distinct owned / 6, on board / 6, tier / 2, layers / 100, is the
   *  plan's P, shop slots offering a new member / 3, one-hot of the nation]. */
  _planNatObs(view, plan, allowed, out, q) {
    const gd = this.encoder.gd, N = this._natBonds(gd), NN = N.nations.length, W = PLAN_NF + NN;
    const own = new Map(); for (const id of view.owned) if (id) own.set(gd.baseIdOf(id), id);
    N.nations.forEach((n, j) => {
      const r = q + j * W, inB = (id) => (gd.chess(id)?.bonds || []).includes(n), v = SPEnv._bv(view.bonds, n);
      let d = 0; for (const id of own.values()) if (inB(id)) d++;
      let nw = 0; for (const id of view.shop) if (id && inB(id) && !own.has(gd.baseIdOf(id))) nw++;
      out[r] = 1; out[r + 1] = allowed.has(n) ? 1 : 0; out[r + 2] = Math.min(d, 9) / 6; out[r + 3] = Math.min(v.count, 9) / 6; out[r + 4] = v.tier / 2;
      out[r + 5] = Math.min(v.layers, 999) / 100; out[r + 6] = plan && plan.P === n ? 1 : 0; out[r + 7] = Math.min(nw, 3) / 3; out[r + PLAN_NF + j] = 1;
    });
  }

  /** (N1) natPlan.head: does this seat take a plan decision now (no plan yet, or a replan round it has not replanned in)? */
  _needPlan(s, round) { return s.plan ? (this.natPlan.replan || []).includes(round) && s.planRound !== round : round >= (this.natPlan.planFrom ?? 1); }

  /** (N1) natPlan.head: the planNat block's offset. */
  _planNatAt() { const sp = this._encSpec(); return sp.blocks.planNat.offset; }

  /** At the Final Assault start: every RL seat's plan state (for the terminal bonus and the metrics). */
  _natAtFA(slot) {
    const m = slot.m, at = (slot.natFA ||= new Map()), N = this._natBonds(m.gd);
    for (const pid of slot.rl) {
      const ps = m.players.get(pid), s = slot.seat.get(pid);
      if (!ps || !ps.alive || !s || !s.plan || at.has(pid)) continue;
      const bonds = ps.privateView().bonds || [], P = SPEnv._bv(bonds, s.plan.P), S = s.plan.S ? SPEnv._bv(bonds, s.plan.S) : null;
      let bn = null, bv = null; for (const n of N.nations) { const v = SPEnv._bv(bonds, n); if (!bv || v.tier * 100 + v.count > bv.tier * 100 + bv.count) { bv = v; bn = n; } }
      let bs = null, bsv = null; for (const c of N.classes) { const v = SPEnv._bv(bonds, c); if (v.active && (!bsv || v.layers > bsv.layers)) { bsv = v; bs = c; } }
      at.set(pid, { P: s.plan.P, S: s.plan.S, pTier: P.tier, pCount: P.count, pLayers: P.layers, sActive: !!S?.active, sLayers: S?.layers || 0,
        bestNat: bn, bestNatTier: bv?.tier || 0, bestSec: bs, bestSecLayers: bsv?.layers || 0 });
    }
  }

  /** Terminal plan bonus of a seat (rewardV 2). */
  _natBonus(slot, pid, faShare) {
    const f = slot.natFA?.get(pid), N = this.natPlan; if (!f) return 0;
    const free = N.phase === 'free', natOk = free ? f.bestNatTier >= 2 : f.pTier >= 2;
    const secOk = free ? f.bestSecLayers >= N.secLayers : f.sActive && f.sLayers >= N.secLayers;
    return (natOk ? N.bonus * (0.5 + 0.5 * faShare) : 0) + (secOk ? N.sec : 0);
  }

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
    slot.recipeSeat = new Map();
    if (this.recipeSeats) for (const [pid, k] of Object.entries(this.recipeSeats.seats || {})) {
      if (!k || !slot.role.has(pid)) continue;
      slot.recipeSeat.set(pid, k); slot.rl.delete(pid); slot.humans.delete(pid); slot.role.set(pid, 'recipe');
    }
    slot.deferredBand = []; slot.bandSeats = new Set(); slot.announced = new Set(); slot.deferredPrep = [];
    if (this.humanProxy) slot.proxy = this.humanProxy.assign(slot, [...slot.humans], seed);
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
        const rk = env.recipeSeats && slot.recipeSeat.get(ps.playerId);
        if (rk) { const ok = (env.recipeSeats.recipes[rk - 1].bands || []).filter((id) => m.gd.bandAllowed(id) && !m.bandTaken(id, ps.playerId)); if (ok.length) return ok[0]; }
        const hp = slot.proxy?.get(ps.playerId);
        if (hp && hp.base !== 'legacy') return env.humanProxy.pickBand(m, ps, hp);
        if (env.bandPolicy?.uniform && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId) && !slot.bandSeats?.has(ps.playerId)) {
          const ok = m.gd.bandIds().filter((id) => m.gd.bandAllowed(id) && !m.bandTaken(id, ps.playerId));
          if (ok.length) return ok[Math.floor(slot.collectRng() * ok.length)];
        }
        if (env.natPlan?.band && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId)) { // (N5) the plan nation's band when offered (route bots)
          const P = env._bandPlanP(slot, ps);
          const ok = P ? m.gd.bandIds().filter((b) => m.gd.bandAllowed(b) && !m.bandTaken(b, ps.playerId) && m.gd.bandBondIds(b).includes(P)) : [];
          if (ok.length) return ok[0];
        }
        const f = env.forceBand;
        if (f && ps.playerId === 'ai_0' && m.gd.bandAllowed(f) && !m.bandTaken(f, ps.playerId)) return f;
        let id = LEGACY_BOT.pickBand(m, ps);
        for (let k = 0; f && !env.forceAt && id === f && k < 8; k++) id = LEGACY_BOT.pickBand(m, ps); // forceAt: the human proxies pick as in the baseline (a forced band they take: not obtained)
        return id;
      },
      *prepBegin(m, ps, o) {
        // the rehearsal count is set for THIS seat before every resumption of its prep steps (per-seat rehearsal, mateRehearsal)
        const hp = slot.proxy?.get(ps.playerId);
        if (hp && hp.base !== 'legacy' && !slot.rl.has(ps.playerId)) return yield* env._perSeat(m, ps, slot, env.humanProxy.prepBegin(m, ps, o, hp), true);
        const base = env.readyHook ? env._hookSteps(slot)[slot.rl.has(ps.playerId) ? 'layout' : 'legacy'] : (slot.rl.has(ps.playerId) ? LAYOUT_ONLY : LEGACY_BOT);
        return yield* env._perSeat(m, ps, slot, base.prepBegin(m, ps, o), true);
      },
      prepEnd(m, ps, job) {
        const hp = slot.proxy?.get(ps.playerId);
        const g = hp && hp.base !== 'legacy' && !slot.rl.has(ps.playerId) ? env.humanProxy.prepEnd(m, ps, job, hp) : (env.readyHook ? env._hookSteps(slot)[slot.rl.has(ps.playerId) ? 'layout' : 'legacy'] : LEGACY_BOT).prepEnd(m, ps, job); // the base hook as before, only wrapped
        return g && typeof g.next === 'function' && (env.bossRehearsal !== null || env.mateRehearsal !== null || env.infoGuard || env.bountyField || env.humanProxy || env.readyHook) ? env._perSeat(m, ps, slot, g) : g;
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
    if (this.humanProxy) this.humanProxy.setup(slot);
    if (!this.encoder) this.encoder = new Encoder(slot.m.gd, { version: this.obsVersion, ...(this.obsExt ? { ext: this.obsExt } : {}) });
    if (this.bountyAware) slot.m._bountyAware = new Set(slot.rl);
    if (slot.recipeSeat.size) { const rs = this.recipeSeats.recipes, mm = slot.m;
      mm.externalAI = { config: { mode: 'recipe' }, observe() {}, close() {}, planFor: (q) => { const k = slot.recipeSeat.get(q.playerId); return k ? recipePlan(mm, q, rs[k - 1]) : null; } }; }
    if (this.seatPlanBonds) for (const [pid, bond] of Object.entries(this.seatPlanBonds)) { const q = slot.m.players.get(pid); if (q) q._envPlanBonds = [{ id: bond, priority: 1 }]; }
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
    if (this.actLog) this._actInstrument(slot);
    if (this.timing) { // battle stepping time, rehearsal fields ('r:…') apart from real battles (as tools/botbench.mjs)
      const timing = this.timing, nb = slot.m.newBattle.bind(slot.m);
      slot.m.newBattle = (opts) => {
        const b = nb(opts);
        if (b && typeof b.step === 'function') {
          const reh = typeof opts?.fieldId === 'string' && (opts.fieldId.startsWith('r:') || opts.fieldId.startsWith('rb:')), step = b.step.bind(b);
          if (reh) { const t = timing[slot.m.round || 0] ||= { prep: 0, battle: 0, other: 0, teach: 0, encode: 0, decisions: 0 }; t.rehN = (t.rehN || 0) + 1; }
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
      if (!hidden && env.natPlan) env._natAtFA(slot);
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
      const hp = slot.proxy?.get(pid), own = hp && ps ? this.humanProxy.announceOf(m, ps, hp) : undefined;
      if (own !== undefined) main = own;
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
    if (this.forceAt) { // band table: ai_0 picks first of the model seats (no earlier AI pick constrains the forced band)
      const j = slot.deferredBand.findIndex((x) => x.ps.playerId === 'ai_0');
      if (j > 0) slot.deferredBand.unshift(...slot.deferredBand.splice(j, 1));
    }
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
    if (this.placeHook && slot.recipeSeat.has(ps.playerId)) return this.rehearsal; // xplace labels: a recipe seat keeps the RL seats' rehearsal
    if (this.bossRehearsal !== null && (m.round === m.gd.bossRound || m.round === m.gd.hiddenRound)) return this.bossRehearsal;
    const hp = slot.proxy?.get(ps.playerId);
    if (hp && hp.rehearsal !== null && !slot.rl.has(ps.playerId)) return hp.rehearsal;
    if (this.mateRehearsal !== null && !slot.rl.has(ps.playerId)) return this.mateRehearsal;
    return this.rehearsal;
  }

  *_perSeat(m, ps, slot, gen, begin = false) {
    const set = () => { if (this.bossRehearsal !== null || this.mateRehearsal !== null || this.humanProxy?.rehearsal != null) m.botRehearsal = this._rehearsalFor(m, ps, slot); };
    const bf = this.bountyField && slot.rl.has(ps.playerId) ? ps : null;
    const rl = this.infoGuard && slot.rl.has(ps.playerId) ? ps.playerId : null, step = (v) => { if (this.infoGuard) { this._guardRefresh(slot); m._rlPath = rl; m._rlPhase = 'prep'; } const sw = bf ? this._bfSwap(m, bf, slot) : null; try { return gen.next(v); } finally { if (sw) this._bfRestore(m, sw); if (this.infoGuard) m._rlPath = null; } };
    set(); let r = step();
    while (!r.done) { const v = yield r.value; set(); r = step(v); }
    if (begin && this.pbl && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId)) this._pBossLabel(m, ps, slot); // (picker) before the xplace / picker boss pickers
    if (begin && (this.placeHook || this.xplace) && !slot.humans.has(ps.playerId) && (slot.rl.has(ps.playerId) || slot.recipeSeat.has(ps.playerId))) this._xHook(m, ps, slot);
    if (begin && this.picker && !(this.xplace && this.xplace.ens > 0) && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId)) this._bossPick(m, ps, slot);
    if (begin && this.plabel && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId)) this._pLabel(m, ps, slot); // (picker) before normalPicker: DAgger order
    if (begin && this.npicker && slot.rl.has(ps.playerId) && !slot.humans.has(ps.playerId)) this._normalPick(m, ps, slot);
    if (begin && m.wave && (this.plabel?.pool || this.npicker?.pool) && slot.rl.has(ps.playerId)) { // picker pool: the seat's final layout of this round
      const pl = new Map(); for (const [k, p] of ps.board) if (p && p.kind === 'chess') pl.set(p.uid, [k, pieceDirP(p)]); (slot.prevPlace ||= new Map()).set(ps.playerId, pl); }
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

  // ---- band table (forceAt): ai_0's forced band at its own (policy) band decision ----
  _forcedBand(slot, ps, dec, mask, own) {
    const m = slot.m, f = this.forceBand, list = this.encoder.vocab.bandList;
    const bc = bandConflicts(m.gd, m.publicView(), ps.playerId, slot.humans, dec.aiPicks);
    const legal = (id) => { const j = list.indexOf(id); return j >= 0 && j < BANDS && !!mask[ACT.BAND + j] && !m.gd.bandBondIds(id).some((b) => bc.humanBand.has(b) || bc.humanMain.has(b)); };
    let band = own, how;
    if (f === 'random') {
      const ok = list.filter(legal), r = mulberry32((slot.seed ^ 0xBA4D) >>> 0);
      how = ok.length ? 'random' : 'none';
      if (ok.length) band = ok[Math.floor(r() * ok.length)];
    } else if (legal(f)) { band = f; how = 'forced'; }
    else how = !list.includes(f) ? 'unknown' : m.bandTaken?.(f, ps.playerId) ? 'taken' : !m.gd.bandAllowed(f) ? 'notAllowed' : 'masked';
    slot.forced = { band: f, how, pick: band, policy: own };
    return band;
  }

  /** The combo as a set: each model seat, in pick order, takes the first band of the combo not yet given out that its mask
   *  and the human-bond rule allow; none left = the policy's pick (the match then does not count as the combo). */
  _teamBand(slot, ps, dec, mask, own) {
    const left = (slot.teamLeft ||= this.forceTeam.slice());
    if (!left.length) return own;
    const saved = slot.forced, fb = this.forceBand; let band = own, rec = null;
    for (const f of left) {
      this.forceBand = f; const b = this._forcedBand(slot, ps, dec, mask, own);
      if (slot.forced.how === 'forced') { band = b; rec = slot.forced; left.splice(left.indexOf(f), 1); break; }
      rec ||= slot.forced;
    }
    this.forceBand = fb; (slot.team ||= []).push({ seat: ps.playerId, ...rec, ...(rec.how === 'forced' ? {} : { band: null }) }); slot.forced = saved;
    return band;
  }

  /** Track X: the label hook (placeHook) and the learned layout policy (xPlace, RL seats only; no battle may be built). */
  _xHook(m, ps, slot) {
    if (this.placeHook) {
      try { this.placeHook(m, ps, slot, this); } catch (e) { this.stats.errors++; if (this.errorSamples.length < 20) this.errorSamples.push(`placeHook: ${String(e && e.stack).slice(0, 300)}`); }
    }
    if (this.xplace && slot.rl.has(ps.playerId)) {
      const S = (slot.xp ||= { decisions: 0, changed: 0, moved: 0, pieces: 0, sims: 0, ms: 0, msMax: 0, errors: 0 }), t0 = performance.now();
      const nb = m.newBattle; let sims = 0;
      m.newBattle = function (...a) { sims++; return nb.apply(this, a); }; // deploy rule: no battle may be built at a placement
      try { this.xplace.apply(m, ps, slot, this, S); } catch (e) { S.errors++; this.stats.errors++; if (this.errorSamples.length < 20) this.errorSamples.push(`xPlace: ${String(e && e.stack).slice(0, 300)}`); }
      finally { if (m.newBattle !== nb) m.newBattle = nb; S.sims += sims; const dt = performance.now() - t0; S.ms += dt; S.msMax = Math.max(S.msMax, dt); }
    }
  }

  /** Boss-round labels (pBossLabel): see the constructor. */
  _pBossLabel(m, ps, slot) {
    const P = this.pbl;
    if (m.wave || (m.round !== m.gd.bossRound && m.round !== m.gd.hiddenRound) || !this.picker) return;
    if (P.hidden === false && m.round === m.gd.hiddenRound) return; // (code-P8) round 15 is played only in matches already won
    const S = (slot.pbl ||= { decisions: 0, labels: 0, ms: 0 }), t0 = performance.now();
    const C = layoutCandidatesForPicker(m, ps, P.K); if (!C || C.cands.length < 2) return;
    const R = P.rehearse === false ? [] : ['match', 'match2'].slice(0, P.seeds || 1).map((sd) => rehearseCandidatesForTraining(m, ps, C.chosen, C.cands, { seed: sd }) || []);
    const sc = C.cands.map((_, j) => { const xs = R.map((r) => r[j]).filter(Boolean); if (!xs.length) return null;
      const mu = (k) => xs.reduce((a, x) => a + (Number(x[k]) || 0), 0) / xs.length;
      return { leaks: mu('leaks'), boss: mu('boss'), kills: mu('kills'), lpc: mu('lpc'), cleared: xs.filter((x) => x.end === 'cleared').length / xs.length, per: xs.map((x) => [x.leaks, Math.round(x.boss), x.kills, x.lpc]) }; });
    S.decisions++;
    if ((!sc[0] && P.rehearse !== false) || !P.dir) return;
    const s = this._seat(slot, ps), ob = new Float32Array(this.encoder.dim), mk = new Uint8Array(this.encoder.A);
    this.encoder.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob, mk);
    this.picker.trunk.forward(ob, mk, { value: false }); const c = Array.from(this.picker.trunk.c, (x) => +x.toFixed(4));
    const oi = [], ov = []; for (let q = 0; q < ob.length; q++) if (ob[q] !== 0) { oi.push(q); ov.push(+ob[q].toFixed(5)); }
    const rows = C.cands.map((_, j) => Array.from(pickerRow(C.feats[j], C.variants[j], j === 0, C.feats[0]), (x) => +x.toFixed(4)));
    // the xplace QNet (env xPlace, e.g. xboss-v2 q5) on the same K8, as XPlacer._pick computes it before the ensemble (offline ensemble check)
    let q = null; const XP = this.xplace;
    if (XP && XP.q) try {
      const Pp = xPlaceProblemP(m, ps);
      if (Pp && Pp.fight && Pp.pieces.length >= XP.minPieces) {
        const h = (a) => Float32Array.from(Float16Array.from(a)), X0 = xFeaturizeP(m, ps, Pp, { lp0: s?.lp0 }), X = { ...X0, fp: h(X0.fp), ft: h(X0.ft), fe: h(X0.fe), fg: h(X0.fg) };
        const cfs = C.cands.map((pl) => (XP.net.cfg.cf ? h(xCandFeaturesP(m, ps, Pp.chosen, pl, C.cands[0])) : null));
        q = Array.from(XP.net.score(X, XP.net.encode(X), C.cands.map((pl) => xAssignOfP(X, pl)), cfs), (x) => +Number(x).toFixed(5));
      }
    } catch { q = null; }
    let qx = null;
    if (P.qx && XP) try { // (code-P9) extra X QNets' Q on the same K (offline comparison of boss scorers; the game still plays xPlace)
      P._qx ||= Object.entries(P.qx).map(([k, f]) => [k, XPlacer.fromOptions({ model: f }).net]);
      const Pp = xPlaceProblemP(m, ps);
      if (Pp && Pp.fight && Pp.pieces.length >= XP.minPieces) {
        const h = (a) => Float32Array.from(Float16Array.from(a)), X0 = xFeaturizeP(m, ps, Pp, { lp0: s?.lp0 }), X = { ...X0, fp: h(X0.fp), ft: h(X0.ft), fe: h(X0.fe), fg: h(X0.fg) };
        qx = {};
        for (const [k, net] of P._qx) {
          const cfs = C.cands.map((pl) => (net.cfg.cf ? h(xCandFeaturesP(m, ps, Pp.chosen, pl, C.cands[0])) : null));
          qx[k] = Array.from(net.score(X, net.encode(X), C.cands.map((pl) => xAssignOfP(X, pl)), cfs), (x) => +Number(x).toFixed(5));
        }
      }
    } catch { qx = null; }
    let ext = null;
    if (P.ext) { // (code-P8) p4-style spatial fields: the v3 + ext.board observation and each candidate's board slice (Encoder.boardSlice)
      if (!P.enc) { P.enc = new Encoder(m.gd, { version: 3, ext: { board: true } }); const sp = `${P.dir}/spec.json`; if (!existsSyncP(sp)) writeFileSyncP(sp, JSON.stringify(P.enc.spec())); }
      const E = P.enc, ob2 = new Float32Array(E.dim), mk2 = new Uint8Array(E.A);
      E.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob2, mk2);
      const sparse = (a) => { const i = [], v = []; for (let k = 0; k < a.length; k++) if (a[k] !== 0) { i.push(k); v.push(+a[k].toFixed(4)); } return { i, v }; };
      let boff = 0, blen = 0;
      const cands = C.cands.map((plan) => { const pl = new Map(), assign = [];
        for (const p of C.chosen) { const k = plan.get(p.uid); if (!k) continue; const d = plan.dirs?.get(p.uid) ?? 'RIGHT'; pl.set(p.uid, [k, d]); assign.push([p.uid, p.id, k, d]); }
        const bs = E.boardSlice(m, ps, pl); boff = bs.off; blen = bs.len; return { b: sparse(bs.v), a: assign }; });
      ext = { fmt: 'p4', D2: E.dim, obs2: sparse(ob2), board: { off: boff, len: blen }, cands, lp: ps.lp, teamLp: Number.isFinite(m.teamLp) ? m.teamLp : null,
        bonds: Object.entries(ps.bonds || {}).filter(([, b]) => b?.active).map(([id, b]) => [id, b.tier || 0, b.layers || 0]) };
    }
    appendFileSyncP(`${P.dir}/boss-${process.pid}.jsonl`, JSON.stringify({ kind: 'boss', trunk: 'SK-30M-js', seedMode: 'match', seeds: P.seeds || 1, cfg: slot.cfg.name, seed: slot.seed, ...(ext || {}),
      round: m.round, pid: ps.playerId, D: ob.length, obs: { i: oi, v: ov }, c, rows, scores: sc, ...(q ? { q, qModel: this.xplaceModel || null, ens: XP.ens } : {}), ...(qx ? { qx } : {}) }) + '\n');
    S.labels++; S.ms += performance.now() - t0;
  }

  /** Normal-round labels (pLabel): see the constructor. Training data only (rehearses every candidate). */
  _pLabel(m, ps, slot) {
    const P = this.plabel;
    if (!m.wave || m.round < P.from || m.round > P.to) return;
    const S = (slot.plab ||= { decisions: 0, labels: 0, changed: 0, ms: 0 }), t0 = performance.now();
    let C = P.pool ? candidatePoolForTraining(m, ps, { K: P.K, tmpl: P.tmpl, prev: slot.prevPlace?.get(ps.playerId), edits: P.edits ?? 12, max: P.max ?? 32 })
      : layoutCandidatesForPicker(m, ps, P.K); if (!C) return;
    if (!P.pool && P.tmpl > 0) { const ex = positionalCandidatesForTraining(m, ps, C, { N: P.tmpl });
      C = { chosen: C.chosen, cands: [...C.cands, ...ex], variants: [...C.variants, ...ex.map((q) => [q._tpl === 'egir' ? 101 : q._tpl === 'kjerag' ? 102 : 100])], feats: [...C.feats, ...ex.map((q) => layoutFeatures(m, ps, C.chosen, q, C.cands[0]))], nLegacy: C.cands.length }; }
    if (C.cands.length < 2) return;
    let sc = rehearseCandidatesForTraining(m, ps, C.chosen, C.cands, { seed: P.seed || 'fixed' }) || []; S.decisions++;
    if ((P.seeds || 1) >= 2) { // picker pool labels: the mean over two per-match seeds ('match', 'match2')
      const s2 = rehearseCandidatesForTraining(m, ps, C.chosen, C.cands, { seed: 'match2' }) || [];
      sc = sc.map((a, j) => { const b = s2[j]; if (!a || !b) return a || b || null; const lg = {}; for (const k of new Set([...Object.keys(a.lg || {}), ...Object.keys(b.lg || {})])) lg[k] = ((a.lg?.[k] || 0) + (b.lg?.[k] || 0)) / 2;
        return { leaks: (a.leaks + b.leaks) / 2, boss: (a.boss + b.boss) / 2, kills: (a.kills + b.kills) / 2, lg }; });
    }
    if (!sc[0]) return;
    const sum = (x) => Object.values(x?.lg || {}).reduce((a, b) => a + b, 0);
    let best = 0;
    for (let j = 1; j < sc.length; j++) { const x = sc[j], y = sc[best]; if (!x || x.leaks > sc[0].leaks) continue; if (sum(x) > sum(y) || (sum(x) === sum(y) && x.kills > y.kills)) best = j; }
    let srch = null; // picker teacher search (pLabel.search { methods, budget, every }): from the pool best, each method's top layouts join the labels
    if (P.search && sc[best] && ((Number(m.seed) + m.round * 31 + (ps.seat || 0) * 7) % (P.search.every || 1)) === 0) {
      srch = {}; const start = C.cands[best], st = sc[best], sigs = new Set(C.cands.map(oracleSigForTraining));
      if (!C.src) C.src = C.cands.map((_, j) => (j === 0 ? 'default' : 'legacy'));
      for (const meth of P.search.methods || ['hill']) {
        const r = teacherSearchForTraining(m, ps, C, start, st, sc[0].leaks, { method: meth, budget: P.search.budget ?? 32 });
        srch[meth] = { u0: +r.u0.toFixed(3), u: +r.u.toFixed(3), evals: r.evals };
        for (const t of r.top) { const sg = oracleSigForTraining(t.plan); if (sigs.has(sg)) continue; sigs.add(sg);
          C.cands.push(t.plan); C.src.push(`search:${meth}`); C.variants.push([]); C.feats.push(layoutFeatures(m, ps, C.chosen, t.plan, C.cands[0])); sc.push(t.score); }
      }
      for (let j = 1; j < sc.length; j++) { const x = sc[j], y = sc[best]; if (!x || x.leaks > sc[0].leaks) continue; if (sum(x) > sum(y) || (sum(x) === sum(y) && x.kills > y.kills)) best = j; }
    }
    if (P.dir) {
      if (!P.enc) { P.enc = new Encoder(m.gd, { version: 3, ext: { board: true } }); const sp = `${P.dir}/spec.json`; if (!existsSyncP(sp)) writeFileSyncP(sp, JSON.stringify(P.enc.spec())); }
      const E = P.enc, s = this._seat(slot, ps), ob = new Float32Array(E.dim), mk = new Uint8Array(E.A);
      E.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob, mk);
      const sparse = (a, base = 0) => { const i = [], v = []; for (let q = 0; q < a.length; q++) if (a[q] !== 0) { i.push(q + base); v.push(+a[q].toFixed(4)); } return { i, v }; };
      let boff = 0, blen = 0;
      const cands = C.cands.map((plan, j) => {
        const pl = new Map(), assign = [];
        for (const p of C.chosen) { const k = plan.get(p.uid); if (!k) continue; const d = plan.dirs?.get(p.uid) ?? 'RIGHT'; pl.set(p.uid, [k, d]); assign.push([p.uid, p.id, k, d]); }
        const bs = E.boardSlice(m, ps, pl); boff = bs.off; blen = bs.len;
        const f = C.feats[j] || {};
        return { b: sparse(bs.v), a: assign, var: C.variants[j], f: [f.value, f.cov, f.covMin, f.flyCov, f.onRoad, f.leader, f.diff, f.units].map((x) => +(+x || 0).toFixed(4)) };
      });
      const bonds = Object.entries(ps.bonds || {}).filter(([, b]) => b?.active).map(([id, b]) => [id, b.tier || 0, b.layers || 0]);
      appendFileSyncP(`${P.dir}/lab-${process.pid}.jsonl`, JSON.stringify({ kind: 'normal', fmt: 'p4', cfg: slot.cfg.name, seed: slot.seed, round: m.round, pid: ps.playerId,
        lp: ps.lp, seedMode: P.seed || 'fixed', seeds: P.seeds || 1, src: C.src || null, ...(srch ? { search: srch } : {}), D: E.dim, obs: sparse(ob), board: { off: boff, len: blen }, nLegacy: C.nLegacy || C.cands.length, cands, bonds, scores: sc, best }) + '\n');
      S.labels++;
    }
    if (best > 0 && P.apply === 'label') { applyPlanForPicker(m, ps, C.chosen, C.cands[best]); S.changed++; }
    S.ms += performance.now() - t0;
  }

  /** Normal-round picker (normalPicker, spatial head): see the constructor. No battle may be built (counted in S.sims). */
  _normalPick(m, ps, slot) {
    const P = this.npicker;
    if (!m.wave || m.round < P.from || m.round > P.to) return;
    const S = (slot.npick ||= { decisions: 0, nonDefault: 0, cands: 0, single: 0, sims: 0, ms: 0 }), t0 = performance.now();
    const nb = m.newBattle; let sims = 0;
    m.newBattle = function (...a) { sims++; return nb.apply(this, a); }; // deploy rule: no battle may be built at a pick
    try {
      const C = P.pool ? candidatePoolForTraining(m, ps, { K: P.K, tmpl: P.tmpl, prev: slot.prevPlace?.get(ps.playerId), edits: P.edits, max: P.max }) : layoutCandidatesForPicker(m, ps, P.K);
      if (!C || C.cands.length < 2) { S.single++; return; }
      const s = this._seat(slot, ps), E = (P.enc ||= new Encoder(m.gd, { version: 3, ext: { board: true } })), ob = new Float32Array(E.dim), mk = new Uint8Array(E.A);
      E.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob, mk);
      const out = C.cands.map((plan) => {
        const pl = new Map(); for (const p of C.chosen) { const k = plan.get(p.uid); if (k) pl.set(p.uid, [k, plan.dirs?.get(p.uid) ?? 'RIGHT']); }
        const bs = E.boardSlice(m, ps, pl), o2 = Float32Array.from(ob); o2.set(bs.v, bs.off); const r = P.spatial.forward(o2, null).logits; return [r[0], r[1], r[2], r[3]];
      });
      let idx = 0;
      if (P.mode === 'score') { for (let j = 1; j < out.length; j++) if (out[j][0] > out[idx][0]) idx = j; }
      else { let bu = P.margin; for (let j = 1; j < out.length; j++) { const u = -P.w[0] * (out[j][1] - out[0][1]) + P.w[1] * (out[j][2] - out[0][2]) + P.w[2] * (out[j][3] - out[0][3]); if (u > bu) { bu = u; idx = j; } } }
      if (idx > 0) { applyPlanForPicker(m, ps, C.chosen, C.cands[idx]); S.nonDefault++; }
      S.decisions++; S.cands += C.cands.length;
      const q = ((S.bySeat ||= {})[ps.playerId] ||= { decisions: 0, nonDefault: 0 }); q.decisions++; if (idx > 0) q.nonDefault++;
    } finally { if (m.newBattle !== nb) m.newBattle = nb; S.sims += sims; S.ms += performance.now() - t0; }
  }

  /** (code-P9) bossSpatial: the spatial boss picker's output per candidate of C (see the constructor); called from XPlacer._pick. */
  _bossSpatialScores(m, ps, slot, C) {
    const P = this.bspatial, s = this._seat(slot, ps), E = (P.enc ||= new Encoder(m.gd, { version: 3, ext: { board: true } })), ob = new Float32Array(E.dim), mk = new Uint8Array(E.A);
    E.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob, mk);
    return C.cands.map((plan) => {
      const pl = new Map(); for (const p of C.chosen) { const k = plan.get(p.uid); if (k) pl.set(p.uid, [k, plan.dirs?.get(p.uid) ?? 'RIGHT']); }
      const bs = E.boardSlice(m, ps, pl), o2 = Float32Array.from(ob); o2.set(bs.v, bs.off); return P.spatial.forward(o2, null).logits[P.out];
    });
  }

  /** Boss-round picker (pickerHead): the head's pick among legacy's candidate layouts of the fielded lineup. */
  _bossPick(m, ps, slot) {
    if (m.wave || (m.round !== m.gd.bossRound && m.round !== m.gd.hiddenRound)) return;
    const P = this.picker, S = (slot.pick ||= { decisions: 0, nonDefault: 0, cands: 0, single: 0, sims: 0, ms: 0 }), t0 = performance.now();
    const nb = m.newBattle; let sims = 0;
    m.newBattle = function (...a) { sims++; return nb.apply(this, a); }; // deploy rule: no battle may be built at a pick
    try {
      const C = layoutCandidatesForPicker(m, ps, P.K);
      if (!C || C.cands.length < 2) { S.single++; return; }
      const s = this._seat(slot, ps), ob = new Float32Array(this.encoder.dim), mk = new Uint8Array(this.encoder.A);
      this.encoder.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: s.lp0, humans: slot.humans, rules: this.rules, cache: {} }, ob, mk);
      P.trunk.forward(ob, mk, { value: false });
      const c = Float32Array.from(P.trunk.c), rows = C.cands.map((_, j) => pickerRow(C.feats[j], C.variants[j], j === 0, C.feats[0]));
      const lg = P.head.score(c, rows); let idx = 0;
      for (let j = 1; j < lg.length; j++) if (lg[j] > lg[idx]) idx = j;
      if (idx > 0) { applyPlanForPicker(m, ps, C.chosen, C.cands[idx]); S.nonDefault++; }
      S.decisions++; S.cands += C.cands.length;
    } finally { if (m.newBattle !== nb) m.newBattle = nb; S.sims += sims; S.ms += performance.now() - t0; }
  }

  // ---- bounty-aware legacy layout (bountyField): the seat's own bounty spawns in m.wave while its layout steps run ----
  _bfSwap(m, ps, slot) {
    const w = m.wave;
    if (!w || !Array.isArray(w.spawns) || !ps.bounties?.length) return null;
    const key = `${m.round}|${ps.bounties.map((b) => b.id).join(',')}`;
    const all = (slot.bfWave ||= new Map());
    let bw = all.get(ps.playerId);
    if (!bw || bw.key !== key || bw.src !== w) {
      bw = { key, src: w, wave: { ...w, spawns: withBounties(m.gd, m.round, w, ps.bounties, ps.playerId) }, path: null };
      all.set(ps.playerId, bw); this.bfStats.seatPreps++; slot.bfPreps = (slot.bfPreps || 0) + 1;
    }
    const sw = { wave: w, path: m._botPath, bw };
    m.wave = bw.wave; m._botPath = bw.path; this.bfStats.swaps++; slot.bfSwaps = (slot.bfSwaps || 0) + 1;
    return sw;
  }

  _bfRestore(m, sw) { sw.bw.path = m._botPath; m.wave = sw.wave; m._botPath = sw.path; }

  // ---- action histogram (actLog, diagnostics) ----
  _actRec(slot, pid) {
    const all = (slot.act ||= new Map());
    let x = all.get(pid);
    if (!x) {
      x = { refresh: 0, level: 0, freeze: 0, buy: { chess: 0, item: 0, other: 0 }, sell: { chess: 0, item: 0, other: 0 }, reward: { chess: 0, item: 0, other: 0 },
        personal: {}, cards: {}, byRound: {}, rounds: [],
        bounty: { cardBattles: 0, clean: 0, units: 0, leaked: 0, byPayout: {} },
        items: { gained: 0, bought: 0, equipCalls: 0, equipBoard: 0, equipBench: 0, consumed: 0, boughtEquipped: 0, boughtEquippedBoard: 0 },
        snap: { battles: 0, board: 0, bench: 0, loose: 0, looseEquip: 0 }, last: null, bought: new Map() };
      all.set(pid, x);
    }
    return x;
  }

  _itemsOf(gd, ps) {
    const out = { board: 0, bench: 0, loose: 0, looseEquip: 0, where: new Map() };
    for (const [, p] of ps.board || []) if (p && p.items) for (const it of p.items) { out.board++; out.where.set(it.uid, 'board'); }
    for (const arr of [ps.hand || [], ps.temp || []]) for (const p of arr) {
      if (!p) continue;
      if (p.kind === 'item') { out.loose++; out.where.set(p.uid, 'loose'); if (gd.item(p.id)?.itemType === 'EQUIP') out.looseEquip++; }
      else if (p.items) for (const it of p.items) { out.bench++; out.where.set(it.uid, 'bench'); }
    }
    return out;
  }

  _looseItems(ps) {
    const s = new Set();
    for (const arr of [ps.hand || [], ps.temp || []]) for (const p of arr) if (p && p.kind === 'item') s.add(p.uid);
    return s;
  }

  _actPre(slot, ps, a, dec) {
    const gd = slot.m.gd, kindOf = (id) => (!id ? 'other' : gd.chess(id) ? 'chess' : gd.item(id) ? 'item' : 'other');
    if (a === ACT.REFRESH) return { t: 'refresh' };
    if (a === ACT.LEVEL) return { t: 'level' };
    if (a === ACT.FREEZE) return { t: 'freeze' };
    if (a >= ACT.BUY && a < ACT.SELL) return { t: 'buy', kind: kindOf(ps.shop?.slots?.[a - ACT.BUY]?.id), loose: this._looseItems(ps) };
    if (a >= ACT.SELL && a < ACT.REWARD) { const p = dec.owned?.[a - ACT.SELL]; return { t: 'sell', kind: p ? kindOf(p.id) : 'other' }; }
    if (a >= ACT.REWARD && a < ACT.PERSONAL) return { t: 'reward', kind: kindOf(ps.offers?.[0]?.slots?.[a - ACT.REWARD]?.id), loose: this._looseItems(ps) };
    if (a >= ACT.PERSONAL && a < ACT.CARD) return { t: 'personal', kind: dec.own?.personalChoice?.cards?.[a - ACT.PERSONAL]?.kind || 'other' };
    return null;
  }

  _actPost(slot, ps, a, pre) {
    const m = slot.m, x = this._actRec(slot, ps.playerId);
    const R = (x.byRound[m.round] ||= { refresh: 0, level: 0, freeze: 0, buyChess: 0, buyItem: 0, sell: 0, reward: 0 });
    if (pre.t === 'refresh' || pre.t === 'level' || pre.t === 'freeze') { x[pre.t]++; R[pre.t]++; return; }
    if (pre.t === 'sell') { x.sell[pre.kind]++; R.sell++; return; }
    if (pre.t === 'personal') { x.personal[pre.kind] = (x.personal[pre.kind] || 0) + 1; return; }
    x[pre.t][pre.kind]++;
    if (pre.t === 'reward') R.reward++; else if (pre.kind === 'chess') R.buyChess++; else if (pre.kind === 'item') R.buyItem++;
    for (const u of this._looseItems(ps)) {
      if (pre.loose.has(u)) continue;
      x.items.gained++;
      if (pre.t === 'buy') { x.items.bought++; x.bought.set(u, { round: m.round, equipped: false, board: false, seenBoard: false, where: 'loose' }); }
    }
  }

  _actCard(slot, ps, c) {
    const x = this._actRec(slot, ps.playerId), k = c ? (c.kind || 'other') : 'fallback';
    x.cards[k] = (x.cards[k] || 0) + 1;
    if (c && c.kind === 'bounty') { const p = 'pick:' + (c.payout || 'kill'); x.cards[p] = (x.cards[p] || 0) + 1; }
  }

  _actInstrument(slot) {
    const m = slot.m, env = this;
    for (const ps of m.players.values()) {
      if (!slot.rl.has(ps.playerId)) continue;
      const base = ps.equip;
      if (typeof base !== 'function') continue;
      ps.equip = function (itemUid, targetUid, ...rest) {
        const il = this.find(itemUid), tl = this.find(targetUid), id = il?.piece?.id;
        const r = base.call(this, itemUid, targetUid, ...rest);
        if (!(r && r.error) && il) {
          const x = env._actRec(slot, ps.playerId), rec = m.gd.item(id);
          x.items.equipCalls++;
          if (typeof rec?.kind === 'string' && rec.kind.startsWith('consume_on_equip')) x.items.consumed++;
          if (tl?.area === 'board') x.items.equipBoard++; else x.items.equipBench++;
          const b = x.bought.get(itemUid);
          if (b && !b.equipped) { b.equipped = true; x.items.boughtEquipped++; }
          if (b && tl?.area === 'board' && !b.board) { b.board = true; x.items.boughtEquippedBoard++; }
        }
        return r;
      };
    }
    const inner = m.settle.bind(m);
    m.settle = (plan, ur) => {
      if (!m.disposed && !m.ended && (m.phase === 'COMBAT' || m.phase === 'UNITE')) env._actSettle(slot);
      return inner(plan, ur);
    };
  }

  /** At a normal round's settlement: each RL seat's bounty units of its own battle (spawned, leaked), and its counted leaks. */
  _actSettle(slot) {
    const m = slot.m, gd = m.gd;
    for (const ps of m.alivePlayers()) {
      if (!slot.rl.has(ps.playerId)) continue;
      const x = this._actRec(slot, ps.playerId), r = m.lastResults.get(ps.playerId), leaked = (r && r.leaked) || [];
      const counted = leaked.filter((l) => l && l.counted !== false).length;
      let bu = 0, bl = 0;
      for (const b of ps.bounties || []) {
        const c = b && b.card ? b.card : {};
        if (!gd.enemy(c.enemyKey)) continue;
        const units = Math.max(1, Math.min(20, Number.isInteger(c.count) ? c.count : 1));
        const lk = Math.min(units, leaked.filter((l) => l && l.mods && l.mods.bountyId === b.id).length);
        const P = (x.bounty.byPayout[c.payout || 'kill'] ||= { n: 0, clean: 0, units: 0, leaked: 0, perfect: 0 });
        x.bounty.cardBattles++; x.bounty.units += units; x.bounty.leaked += lk; if (!lk) x.bounty.clean++;
        P.n++; P.units += units; P.leaked += lk; if (!lk) P.clean++; if (counted === 0 && r && r.perfect !== false) P.perfect++;
        bu += units; bl += lk;
      }
      x.rounds.push([m.round, bu, bl, counted]);
    }
  }

  /** At each battle start: every RL seat's items (equipped on a board unit / on a bench unit / loose in hand or temp). */
  _actBattle(slot) {
    const m = slot.m, ph = m.phase;
    slot.alPhase = ph; slot.alRound = m.round;
    if (!(ph === 'COMBAT' || ph === 'FINAL_ASSAULT' || ph === 'HIDDEN_CORE')) return;
    const alive = new Set(m.alivePlayers());
    for (const pid of slot.rl) {
      const ps = m.players.get(pid);
      if (!ps || !alive.has(ps)) continue;
      const x = this._actRec(slot, pid), it = this._itemsOf(m.gd, ps);
      x.snap.battles++; x.snap.board += it.board; x.snap.bench += it.bench; x.snap.loose += it.loose; x.snap.looseEquip += it.looseEquip;
      x.last = { r: m.round, ph, board: it.board, bench: it.bench, loose: it.loose, looseEquip: it.looseEquip };
      for (const [u, b] of x.bought) { b.where = it.where.get(u) || 'gone'; if (b.where === 'board') b.seenBoard = true; }
    }
  }

  _actSummary(slot) {
    const m = slot.m, out = [];
    for (const pid of slot.rl) {
      const x = slot.act?.get(pid), ps = m.players.get(pid);
      if (!x) { out.push({ pid, role: slot.role.get(pid), none: true }); continue; }
      const { bought, ...rest } = x, bl = [...bought.values()];
      const st = ps?.stats || {};
      out.push({ pid, role: slot.role.get(pid), band: ps?.bandId ?? null, ...rest,
        boughtItems: { n: bl.length, equipped: bl.filter((b) => b.equipped).length, onBoardAtBattle: bl.filter((b) => b.seenBoard).length,
          neverEquippedLoose: bl.filter((b) => !b.equipped && b.where === 'loose').length, neverEquippedGone: bl.filter((b) => !b.equipped && b.where === 'gone').length,
          endWhere: bl.reduce((o, b) => { o[b.where] = (o[b.where] || 0) + 1; return o; }, {}) },
        stats: { buys: st.buys, sells: st.sells, refreshes: st.refreshes, merges: st.merges, leaks: st.leaks } });
    }
    return out;
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
    // legacy mode (no RL seat): the legacy AI seats (not the human proxies) are logged instead (11-13, stacking metrics of legacy fill)
    const logged = this.mode === 'legacy' ? [...slot.role.keys()].filter((x) => slot.role.get(x) === 'legacy' || (this.humanProxy?.log && slot.humans.has(x))) : this.humanProxy?.log ? [...slot.rl, ...[...slot.humans].filter((x) => !slot.rl.has(x))] : slot.rl;
    for (const pid of logged) {
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

  /** (A1 hist) At each phase change: leaving PREP notes every seat's LP / stacking layers; entering PREP closes the battle. */
  _histHook(slot) {
    const m = slot.m, prev = slot.hPhase; slot.hPhase = m.phase;
    slot.hist ||= new Map();
    const into = prev === 'PREP' && m.phase !== 'PREP', back = prev !== undefined && prev !== 'PREP' && m.phase === 'PREP';
    if (!into && !back) return;
    for (const ps of m.players.values()) {
      if (ps.spectator) continue;
      const h = this._histOf(slot, ps);
      const layers = Math.round(bondTerms(m.gd, ps.privateView().bonds, true).S * 100), lp = Math.max(0, ps.lp || 0);
      if (into) { h.pre = { round: m.round, layers, lp }; continue; }
      if (!h.pre) continue;
      const r = m.lastResults?.get(ps.playerId);
      const leaks = r ? (r.leaked || []).filter((l) => l && l.counted !== false).length : 0;
      h.battles.push({ round: h.pre.round, leaks, lpLost: Math.max(0, h.pre.lp - lp), dLayers: layers - h.pre.layers, layers, lp, kills: Math.max(0, Number(r?.killed) || 0) });
      if (h.battles.length > 20) h.battles.shift();
      h.pre = null;
    }
  }

  _histOf(slot, ps) {
    slot.hist ||= new Map();
    let h = slot.hist.get(ps.playerId);
    if (!h) { h = { battles: [], rolls: 0, seen: new Map(), shopKey: null, pre: null, prepRound: null }; slot.hist.set(ps.playerId, h); }
    return h;
  }

  /** (A1 hist) The seat's history at a decision: a shop it has not seen yet (round, rolls, slot ids) counts as one roll. */
  _histSeen(slot, ps) {
    const h = this._histOf(slot, ps), m = slot.m, slots = ps.shop?.slots || [];
    const key = `${m.round}|${ps.stats?.refreshes ?? 0}|${slots.map((x) => (x ? x.id : '')).join(',')}`;
    if (key !== h.shopKey) {
      h.shopKey = key; h.rolls++;
      if (h.prepRound !== m.round) { h.prepRound = m.round; for (const e of h.seen.values()) e.prep = 0; }
      for (const x of slots) {
        if (!x || x.kind !== 'chess') continue;
        const b = m.gd.baseIdOf(x.id), e = h.seen.get(b);
        if (e) { e.n++; e.prep = (e.prep || 0) + 1; } else h.seen.set(b, { n: 1, first: h.rolls, prep: 1 });
      }
    }
    return h;
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
      if (this.actLog && (slot.m.phase !== slot.alPhase || slot.m.round !== slot.alRound)) this._actBattle(slot);
      if (this.rewardV === 2 && this.R2.layer > 0 && slot.m.phase !== slot.lhPhase) this._layerHook(slot);
      if (this.obsExt?.hist && slot.m.phase !== slot.hPhase) this._histHook(slot);
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
      ...(diag || {}), ...(this.tripLog ? { trips: slot.trips } : {}),
      ...(this.forceAt ? { forced: { ...(slot.forced || { band: this.forceBand, how: 'none' }), got: m.players.get('ai_0')?.bandId ?? null } } : {}),
      ...(this.picker ? { picker: slot.pick || { decisions: 0 } } : {}), ...(this.npicker ? { npicker: slot.npick || { decisions: 0 } } : {}), ...(this.plabel ? { plab: slot.plab || { decisions: 0 } } : {}), ...(this.pbl ? { pbl: slot.pbl || { decisions: 0 } } : {}),
      ...(this.xplace ? { xplace: slot.xp || { decisions: 0 } } : {}),
      ...(this.forceTeam ? { team: (slot.team || []).map((t) => ({ ...t, got: m.players.get(t.seat)?.bandId ?? null })) } : {}),
      ...(this.actLog ? { act: this._actSummary(slot) } : {}), ...(this.bountyField ? { bf: { preps: slot.bfPreps || 0, swaps: slot.bfSwaps || 0 } } : {}), ...(slot.names ? { names: slot.names.list, replay: slot.replay } : {}) };
    // (N1) natPlan.hindsight (teacher recording): every held sample gets the plan its seat played, then goes to the recorder
    const hind = this.natPlan?.hindsight && slot.natBuf ? new Map() : null;
    if (hind) {
      const lastView = new Map(); for (const x of slot.natBuf) lastView.set(x.pid, x.view);
      for (const [pid, v] of lastView) hind.set(pid, this._hindsightPlan(m.gd, m.players.get(pid), v.bonds, this.natPlan.known ? slot.seat.get(pid)?.plan || null : null));
      const N = this.natPlan.head ? this._natBonds(m.gd) : null, at = this.natPlan.head ? (this._pnAt ??= this._planNatAt()) : 0;
      for (const x of slot.natBuf) {
        const o = new Float32Array(this.D); o.set(x.o); const h = hind.get(x.pid);
        // a first plan decision sees no plan yet; with natPlan.blind no plan decision does; before planFrom no sample has a plan
        const noPlan = (x.plan && (x.first || this.natPlan.blind)) || (this.natPlan.head && (x.round ?? 99) < (this.natPlan.planFrom ?? 1));
        if (!noPlan) this._planObs(x.view.bonds, h, o, this.encoder.dim, this.natPlan.slots ? x.view : null);
        let mk = x.mk, a = x.a;
        if (N) {
          this._planNatObs(x.view, noPlan ? null : h, x.allowed, o, at);
          if (x.plan) { // the plan sample's action: the hindsight nation (dropped when that nation was not allowed)
            if (!x.allowed.has(h.P)) continue;
            mk = new Uint8Array(this.A); N.nations.forEach((n, j) => { if (x.allowed.has(n)) mk[this.encoder.A + j] = 1; });
            a = this.encoder.A + N.nations.indexOf(h.P);
          }
        }
        this.recorder.sample(x.kind, x.traj, o, mk, a, x.rw);
      }
      slot.natBuf = null;
    }
    for (const p of rows) {
      const round = p.eliminatedRound ?? m.round;
      const rl = slot.rl.has(p.playerId);
      res.seats.push({ playerId: p.playerId, rl, role: slot.role.get(p.playerId), band: m.players.get(p.playerId)?.bandId ?? null, survivalRound: round, roundsPassed: p.roundsPassed,
        lp: p.lp, win: p.victory ? 1 : 0, mainTier: slot.humanTier.get(p.playerId)?.max ?? null,
        ...(slot.proxy?.has(p.playerId) ? { proxy: this.humanProxy.info(m, m.players.get(p.playerId), slot.proxy.get(p.playerId)) } : {}),
        ...(this.bondLog && (rl || (this.mode === 'legacy' && slot.role.get(p.playerId) === 'legacy') || (this.humanProxy?.log && slot.humans.has(p.playerId))) ? (() => { const { last, lastRound, ...rs } = slot.bl?.get(p.playerId) || {}; return { bondLog: { ...rs, end: last || null, endRound: lastRound ?? null } }; })() : {}) });
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
        const nb = this.natPlan && s.plan ? this._natBonus(slot, p.playerId, faShare) : 0;
        reward = base + win + faT + faO + elim + shapeEnd + nb;
        if (rl) res.seats[res.seats.length - 1].r2 = { ...c, shape: (s.shape || 0), ev: { ...s.ev, ownFa: own, eliminated: p.eliminatedRound ?? null, cleared } };
      } else reward = this._reward(s, m, round, Math.max(0, p.lp || 0)) + (p.victory ? R.win : 0) + R.fa * faShare + shapeEnd;
      if (rl && this.natPlan && s.plan) res.seats[res.seats.length - 1].nat = { ...s.plan, fa: slot.natFA?.get(p.playerId) || null, bonus: this._natBonus(slot, p.playerId, faShare), phase: this.natPlan.phase, ...(s.natMs ? { dense: +s.natMs.sum.toFixed(3) } : {}), ...(s.natReplans ? { replans: s.natReplans } : {}) };
      const t = { traj: s.traj, reward, win: p.victory ? 1 : 0, round, lp: p.lp || 0, config: slot.ci, steps: s.steps, faShare,
        rRound: s.cRound || 0, rLp: s.cLp || 0, shape: s.shape || 0, seedOffset: slot.seed - this.seed,
        ...(s.R2 ? { r2: [s.R2.win, s.R2.faTeam, s.R2.faOwn, s.R2.leak, s.R2.save, s.R2.lp, s.R2.elim, s.R2.layer, s.shape || 0] } : {}) };
      if (hind && hind.has(p.playerId)) t.natH = hind.get(p.playerId);
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
      if (this.natPlan && !this.natPlan.head && ps.bandId && !slot.humans.has(ps.playerId) && (!s.plan || (this.natPlan.assign === 'lead' && (m.round || 0) < (this.natPlan.lockRound ?? 4)))) s.plan = this._assignPlan(slot, ps, s.plan || null);
      // (N8) natPlan.coopReplan: a human now builds (or holds by band) our plan nation -> replan (head: a blind plan decision now)
      const hum = !!(this.natPlan?.coopReplan && dec.kind === 'prep' && s.plan && s.replanRound !== (m.round || 0) && !slot.humans.has(ps.playerId) && this._humanNat(slot, ps).has(s.plan.P));
      if (hum) s.replanRound = m.round || 0; // at most one co-op replan per round
      if (hum && !this.natPlan.head) { slot.natTaken?.delete(s.plan.P); s.plan = this._assignPlan(slot, ps, s.plan); s.natReplans = (s.natReplans || 0) + 1; }
      if (this.natPlan?.focus && s.plan && ps._envPlanBonds?.[0]?.id !== s.plan.P) ps._envPlanBonds = [{ id: s.plan.P, priority: 1 }]; // (N5) legacy's focus = P
      const pd = !!(this.natPlan?.head && dec.kind === 'prep' && ps.bandId && !slot.humans.has(ps.playerId) && this._needPlan(s, m.round || 0))
        || !!(this.natPlan?.head && this.natPlan.planAtBand && dec.kind === 'band' && !slot.humans.has(ps.playerId) && !s.plan) // (N7) plan before the band
        || (hum && !!this.natPlan.head); // (N8)
      this.rew[k] = this._reward(s, m, m.round, Math.max(0, ps.lp || 0));
      s.steps++;
      this.ids[k] = s.traj; this.kinds[k] = KIND[dec.kind] + (this.kindSeat ? 16 * (Number(String(dec.ps.playerId).split('_')[1]) || 0) : 0);
      const te0 = this.timing ? performance.now() : 0;
      const newUids = dec.startUids ? new Set(ps.allChess().map((p) => p.uid).filter((u) => !dec.startUids.has(u))) : null;
      const aiPicks = dec.kind === 'band' ? new Set([...m.players.keys()].filter((pid) => !slot.humans.has(pid) && m.draft?.picks?.[pid])) : null;
      if (this.full && dec.kind !== 'band' && dec.kind !== 'card' && !dec.opened) this._openPlan(dec);
      if (this.infoGuard) { this._guardRefresh(slot); m._rlPath = ps.playerId; m._rlPhase = 'enc'; }
      const r = this.encoder.encode(m, ps, { kind: dec.kind, actions: dec.actions, freezes: dec.freezes, lp0: s.lp0, humans: slot.humans, visibleOnly: this.visibleOnly, strictPreview: this.strictPreview,
        rules: this.rules, newUids, soldBases: dec.soldBases, cache: dec.cache || {}, aiPicks, ...(this.obsExt?.hist ? { hist: this._histSeen(slot, ps) } : {}),
        ...(this.full ? { sel: dec.sel, pubSnap: dec.pubSnap, ...(dec.mateSnap !== undefined ? { mateSnap: dec.mateSnap } : {}),
          planned: dec.planned, rerolls: seatPlanRerolls(ps) } : {}) },
        this.obs.subarray(k * D, (k + 1) * D), this.mask.subarray(k * A, (k + 1) * A));
      dec.own = r.own; dec.owned = r.owned;
      const pv = this.natPlan && (this.natPlan.slots || this.natPlan.head) ? this._planView(r.own, r.owned) : null;
      const blind = pd && (this.natPlan.blind || hum); // (N1) natPlan.blind: a plan decision does not see the current plan (N8: a co-op replan neither)
      if (this.natPlan && s.plan && !blind) this._planObs(r.own?.bonds, s.plan, this.obs, k * D + this.encoder.dim, this.natPlan.slots ? pv : null);
      if (this.natPlan?.head && !slot.humans.has(ps.playerId)) {
        const al = this._planAllowed(slot, ps, s.plan?.P || null);
        this._planNatObs(pv, blind ? null : s.plan || null, al, this.obs, k * D + (this._pnAt ??= this._planNatAt()));
        dec.planDec = pd;
        if (pd) { // a plan decision: only the allowed nations' actions
          const mk = this.mask.subarray(k * A, (k + 1) * A); mk.fill(0);
          this._natBonds(m.gd).nations.forEach((n, j) => { if (al.has(n)) mk[this.encoder.A + j] = 1; });
          this.kinds[k] = KIND_PLAN;
        }
      }
      if (this.natPlan?.dense && s.plan) this.rew[k] += this._natDense(s, r.own?.bonds);
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
        const phi = this.shaping.wTier * T + this.shaping.wStack * S + (this.natPlan && s.plan ? this._natPhi(r.own.bonds, s.plan) : 0);
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
    if (dec.planDec) { // (N1) natPlan.head: the plan decision sets the plan; the prep decision itself stays pending
      dec.planDec = false;
      const s = this._seat(slot, ps), N = this._natBonds(m.gd), j = a - this.encoder.A;
      let P = a >= this.encoder.A && a < this.A && mask[a] ? N.nations[j] : null;
      if (!P) { this.stats.rejected++; P = N.nations.find((n, i) => mask[this.encoder.A + i]) || N.nations[0]; }
      const taken = (slot.natTaken ||= new Set()); if (s.plan) taken.delete(s.plan.P); taken.add(P);
      if (s.plan && s.plan.P !== P && this.natPlan.coopReplan) s.natReplans = (s.natReplans || 0) + 1; // (N8)
      s.plan = { P, S: this._ruleS(slot, ps, P, s.plan && s.plan.P === P ? s.plan : null) }; s.planRound = m.round || 0;
      if (this.natPlan.focus) ps._envPlanBonds = [{ id: P, priority: 1 }]; // (N5)
      if (dec.kind === 'band') { // (N7) natPlan.planAtBand: the band follows the plan (P's band when offered, else legacy's pick)
        this._drop(dec);
        if (!d.valid()) return;
        const ok = m.gd.bandIds().filter((b) => m.gd.bandAllowed(b) && !m.bandTaken(b, ps.playerId) && m.gd.bandBondIds(b).includes(P));
        const bandId = ok.length ? ok[0] : LEGACY_BOT.pickBand(m, ps);
        if (bandId) { const r = m._handleSeatIntent(ps, { t: 'g.band', bandId }); if (r?.error) this.stats.rejected++; }
        if (!d.complete()) d.fallback();
      }
      return;
    }
    if (!(a >= 0 && a < this.A) || !mask[a]) { this.stats.rejected++; a = dec.kind === 'card' ? -1 : dec.kind === 'tile' || dec.kind === 'target' ? ACT.CANCEL : ACT.END; }
    if (dec.kind === 'band') {
      this._drop(dec);
      if (!d.valid()) return;
      const j = a - ACT.BAND; let bandId = a >= ACT.BAND && j < BANDS ? this.encoder.vocab.bandList[j] : null;
      if (this.forceAt && ps.playerId === 'ai_0') bandId = this._forcedBand(slot, ps, dec, mask, bandId);
      else if (this.forceTeam && !slot.humans.has(ps.playerId)) bandId = this._teamBand(slot, ps, dec, mask, bandId);
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
      if (c) { const r = m._handleSeatIntent(ps, { t: 'g.choice', idx: c.idx }); if (r?.error) this.stats.rejected++; else if (this.actLog) this._actCard(slot, ps, c); }
      else if (this.actLog) this._actCard(slot, ps, null);
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
    const pre = this.actLog && intent ? this._actPre(slot, ps, a, dec) : null;
    slot.acting = ps.playerId; // the trip log tells the policy's own sales from legacy's cleanup
    let r;
    try { r = intent ? m._handleSeatIntent(ps, intent) : { error: 'BAD_ACTION' }; } finally { slot.acting = null; }
    if (r?.error) this.stats.rejected++;
    else if (intent?.t === 'g.sell' && dec.soldBase) dec.soldBases.add(dec.soldBase);
    if (pre && !r?.error) this._actPost(slot, ps, a, pre);
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
      if (this.natPlan && rec) { // (N1) teacher mode with the plan block: hindsight (held to the match end) or the seat's own plan
        if (this.A > mk.length) { const m2 = new Uint8Array(this.A); m2.set(mk); mk = m2; }
        if (this.natPlan.hindsight) { (slot.natBuf ||= []).push({ pid: ps.playerId, kind, traj: s.traj, o: o.slice(), mk: mk === mask ? mk.slice() : mk, a, rw, view: ctx.planView,
          ...(this.natPlan.head ? { allowed: this._planAllowed(slot, ps, null), round: m.round || 0 } : {}) }); return; }
        if (!s.plan && ps.bandId) s.plan = this._assignPlan(slot, ps);
        const o2 = new Float32Array(this.D); o2.set(o); if (s.plan) this._planObs(ctx.planView.bonds, s.plan, o2, this.encoder.dim, ctx.planView);
        rec.sample(kind, s.traj, o2, mk, a, rw); return;
      }
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
      if (this.obsExt?.hist) ctx.hist = this._histSeen(slot, ps);
      lastOwn = this.encoder.encode(m, ps, ctx, obs, mask).own;
      if (this.natPlan) ctx.planView = this._planView(lastOwn, Encoder.owned(lastOwn));
      if (log) log.before = summarize(m, ps, lastOwn, slot.names, d.kind);
      const pos = sp.cards.findIndex((c) => c.idx === idx);
      const r = m._handleSeatIntent(ps, { t: 'g.choice', idx });
      if (r?.error) return false; // the engine falls back to the legacy pick
      note(pos >= 0 && pos < CARDS ? ACT.CARD + pos : -1, 'card');
      return true;
    }
    T.preps++;
    let owned = null, protect = new Set(), pairs = new Set();
    if (this.natTeach && ps.bandId && !slot.humans.has(ps.playerId)) this._natTeachSetup(slot, ps, s); // (N5) e.g. a band the env did not pick
    if (this.natPlan?.head && this.natPlan.hindsight && rec && ps.bandId && !slot.humans.has(ps.playerId) && ((s.hsPlans ||= new Set()).size === 0
      ? (m.round || 0) >= (this.natPlan.planFrom ?? 1) : (this.natPlan.replan || []).includes(m.round || 0) && !s.hsPlans.has(m.round || 0))) {
      // (N1) a synthetic plan-decision sample at the seat's first prep (and at the replan rounds): its action is the hindsight P
      s.hsPlans.add(m.round || 0);
      const r = this.encoder.encode(m, ps, ctx, obs, mask);
      (slot.natBuf ||= []).push({ pid: ps.playerId, kind: KIND_PLAN, traj: s.traj, o: obs.slice(), mk: null, a: -1, plan: true, first: s.hsPlans.size === 1,
        rw: this._reward(s, m, m.round, Math.max(0, ps.lp || 0)), view: this._planView(r.own, r.owned), allowed: this._planAllowed(slot, ps, null), round: m.round || 0 });
      s.steps++;
    }
    const before = () => {
      ctx.newUids = new Set(ps.allChess().map((p) => p.uid).filter((u) => !startUids.has(u)));
      if (this.obsExt?.hist) ctx.hist = this._histSeen(slot, ps);
      const r = this.encoder.encode(m, ps, ctx, obs, mask); owned = r.owned; protect = r.protect; pairs = r.pairs;
      lastOwn = r.own;
      if (this.natPlan) ctx.planView = this._planView(r.own, r.owned);
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
      } else runSteps((this.natTeach && ps._nat ? NAT_PROCURE_ONLY : PROCURE_ONLY).prepBegin(m, ps, {}));
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

  /** Diagnostics: per-match prep steps whose Ready first runs readyHook (legacy seats and LAYOUT_ONLY seats). */
  _hookSteps(slot) {
    const env = this;
    const ready = (ps) => { try { env.readyHook(slot.m, ps, slot); } catch (e) { env.stats.errors++; if (env.errorSamples.length < 20) env.errorSamples.push(`readyHook: ${e.stack?.slice(0, 200)}`); } try { ps.setReady(true); } catch { /* refused */ } };
    return slot.hookSteps ||= { legacy: createBotPrepSteps({ ready }), layout: createBotPrepSteps({ takeOffers: noop, sellJunk: noop, buyLoopSteps: none, levelUp: noop,
      advisorSpendSteps: none, spendDown: noop, maybeFreeze: noop, ready, ...(env.equipHook ? { equipItems: env.equipHook } : {}) }) };
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
