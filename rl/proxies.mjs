// tools/rl/proxies.mjs — co-op proxies (11-13): a pool of teammate proxies for the HUMAN seats of an eval (env option humanProxy;
// off by default, and with it off nothing here runs). Eval-only proxies; the model seats are never affected directly.
//   legacy          production legacy (the env's existing human proxy; only its rehearsal / bond log can change)
//   nat | nat:<P> | nat:rand   the route bots' national-route bot (routebots.mjs NAT_POLICY: every live member of P toward tier 2,
//                   secondary S, national band); nat = the route bots' rank order of live nations, nat:rand = a seeded random live nation
//   natb[:<P>]      the same with legacy's band pick
//   class | classb  the route bots' class bot (Ioleta's 助力 / 坚守 / 精准 elite plan forced on; band_ioleta | legacy's band)
//   recipe:<k|id>   the recipe AI (recipes.json: legacy + the recipe's advisor plan through m.externalAI.planFor, its first
//                   allowed band); recipe:rand = a seeded random recipe
//   noisy[:<sigma>] legacy with seeded noise N(0, sigma) on its buy / reward score per (seat, round, operator); default 25
//   mix:<kind>=<w>|<kind>=<w>...   a seeded per-seat draw from these kinds (population)
// o = { kinds: [kind per human seat, in seat order, cycled], rehearsal: normal-round rehearsal of the proxies (null = the
//   env's mateRehearsal), announce: 'own' (nat / recipe proxies announce their nation as 主盟约; others as the env) | 'env'
//   | 'none' (no proxy announces), log: bond log for the human seats too, recipes: path of a recipes JSON }
import { readFileSync } from 'node:fs';
import { LEGACY_BOT } from '../../server/match/botStrategyContract.js';
import * as IP from '../../server/match/botIoletaPolicy.js';
import { recipePlan } from './recipes.mjs';
import { createBotPrepSteps } from '../../server/match/bot.js';
import { NATIONS, natSetup, NAT_POLICY, CLASS_POLICY } from './trackm/routebots.mjs';

const NAT_PREP = createBotPrepSteps({}, NAT_POLICY), CLASS_PREP = createBotPrepSteps({}, CLASS_POLICY);

const hash32 = (s) => { let h = 0x811C9DC5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } return h >>> 0; };
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const gauss = (u1, u2) => Math.sqrt(-2 * Math.log(Math.max(u1, 1e-12))) * Math.cos(2 * Math.PI * u2);
const offBonds = (m) => new Set([...(m.disabledBonds || []), ...(m.staticInactiveBonds || []), ...m.gd.modeInactiveBonds]);

export class HumanProxies {
  constructor(o = {}) {
    this.kinds = (Array.isArray(o.kinds) ? o.kinds : [o.kinds || 'legacy']).map(String);
    this.rehearsal = Number.isInteger(o.rehearsal) && o.rehearsal >= 0 ? Math.min(o.rehearsal, 8) : null;
    this.announce = o.announce || 'own';
    this.log = o.log !== false;
    this.recipes = JSON.parse(readFileSync(o.recipes || new URL('./recipes.json', import.meta.url), 'utf8'));
    for (const k of this.kinds) this._parse(k); // validate up front
  }

  _parse(kind) {
    const [base, ...rest] = kind.split(':'), arg = rest.join(':') || null;
    if (base === 'mix') {
      const parts = arg.split('|').map((x) => { const j = x.lastIndexOf('='); return [x.slice(0, j), Number(x.slice(j + 1))]; });
      for (const [k, w] of parts) { if (!(w >= 0)) throw new Error(`humanProxy: bad weight in ${kind}`); this._parse(k); }
      return { base, arg, parts };
    }
    if (!['legacy', 'nat', 'natb', 'class', 'classb', 'recipe', 'noisy'].includes(base)) throw new Error(`humanProxy: unknown kind ${kind}`);
    if (base === 'recipe' && arg !== 'rand' && !this._recipe(arg)) throw new Error(`humanProxy: no recipe ${arg}`);
    return { base, arg };
  }

  _recipe(arg) {
    const k = Number(arg);
    return Number.isInteger(k) && k >= 1 && k <= this.recipes.length ? this.recipes[k - 1] : this.recipes.find((r) => r.id === arg) || null;
  }

  /** Proxy specs of a match's human seats (seeded by the match seed; the env RNG is not touched). */
  assign(slot, humanPids, seed) {
    const out = new Map(), rng = mulberry32(hash32(`proxy|${seed}`));
    humanPids.forEach((pid, j) => {
      let kind = this.kinds[j % this.kinds.length], p = this._parse(kind);
      if (p.base === 'mix') {
        const tot = p.parts.reduce((s, [, w]) => s + w, 0); let u = rng() * tot;
        kind = p.parts.find(([, w]) => (u -= w) < 0)?.[0] || p.parts[p.parts.length - 1][0]; p = this._parse(kind);
      }
      const spec = { pid, kind, base: p.base, arg: p.arg, rehearsal: this.rehearsal, u: rng() };
      if (p.base === 'recipe') spec.recipe = p.arg === 'rand' ? this.recipes[Math.floor(spec.u * this.recipes.length)] : this._recipe(p.arg);
      if (p.base === 'noisy') spec.sigma = p.arg !== null ? Number(p.arg) : 25;
      out.set(pid, spec);
    });
    return out;
  }

  /** After the Match exists: advisor plans of the recipe seats, buy noise of the noisy seats. */
  setup(slot) {
    const m = slot.m, specs = slot.proxy;
    if (!specs || !specs.size) return;
    const rec = [...specs.values()].filter((s) => s.base === 'recipe');
    if (rec.length) {
      const by = new Map(rec.map((s) => [s.pid, s.recipe]));
      m.externalAI = { config: { mode: 'recipe' }, observe() {}, close() {}, planFor: (q) => { const rc = by.get(q.playerId); return rc ? recipePlan(m, q, rc) : null; } };
    }
    for (const s of specs.values()) if (s.base === 'noisy') {
      const ps = m.players.get(s.pid); if (!ps) continue;
      const key = `${slot.seed}|${s.pid}`, sigma = s.sigma;
      ps._buyNoise = (id) => { const h = hash32(`${key}|${m.round}|${m.gd.baseIdOf(id)}`), r = mulberry32(h); return sigma * gauss(r(), r()); };
    }
  }

  _natSetup(m, ps, spec) {
    if (ps._nat) return ps._nat;
    let forced = spec.arg && spec.arg !== 'rand' ? spec.arg : null;
    if (spec.arg === 'rand') { const o = offBonds(m), live = NATIONS.filter((b) => !o.has(b)); forced = live.length ? live[Math.floor(spec.u * live.length)] : null; }
    return natSetup(m, ps, forced);
  }

  pickBand(m, ps, spec) {
    const b = spec.base;
    if (b === 'classb' || b === 'natb') { if (b === 'natb') this._natSetup(m, ps, spec); return LEGACY_BOT.pickBand(m, ps); }
    if (b === 'class') { const id = 'band_ioleta'; if (m.gd.bandIds().includes(id) && !m.bandTaken(id, ps.playerId)) return id; }
    else if (b === 'nat') {
      const N = this._natSetup(m, ps, spec);
      const ok = m.gd.bandIds().filter((x) => m.gd.bandAllowed(x) && !m.bandTaken(x, ps.playerId) && m.gd.bandBondIds(x).includes(N.P));
      if (ok.length) return ok[0];
    } else if (b === 'recipe') {
      const ok = (spec.recipe.bands || []).filter((x) => m.gd.bandAllowed(x) && !m.bandTaken(x, ps.playerId));
      if (ok.length) return ok[0];
    }
    return LEGACY_BOT.pickBand(m, ps);
  }

  /** The proxy's prep generators (legacy, recipe and noisy seats: legacy's; their plan / noise lives on the match / seat). */
  *prepBegin(m, ps, opts, spec) {
    const b = spec.base;
    if (b === 'class' || b === 'classb') { ps._ioletaForce = true; if (!ps.alive || ps.ready) return null; IP.refreshIoletaPlan(m, ps); return yield* CLASS_PREP.prepBegin(m, ps, opts); }
    if (b === 'nat' || b === 'natb') { this._natSetup(m, ps, spec); return yield* NAT_PREP.prepBegin(m, ps, opts); }
    return yield* LEGACY_BOT.prepBegin(m, ps, opts);
  }

  prepEnd(m, ps, job, spec) {
    const b = spec.base;
    if (b === 'class' || b === 'classb') return CLASS_PREP.prepEnd(m, ps, job);
    if (b === 'nat' || b === 'natb') return NAT_PREP.prepEnd(m, ps, job);
    return LEGACY_BOT.prepEnd(m, ps, job);
  }

  /** The 主盟约 a proxy announces: undefined = the env's default rule, null = none, else a bond id. */
  announceOf(m, ps, spec) {
    if (this.announce === 'none') return null;
    if (this.announce !== 'own') return undefined;
    if ((spec.base === 'nat' || spec.base === 'natb') && ps._nat?.P) return ps._nat.P;
    if (spec.base === 'recipe') return spec.recipe.bond;
    // the class bots: the band's tied bond if any, never the env's random core announcement (it would become legacy's focus)
    if (spec.base === 'class' || spec.base === 'classb') return m.gd.bandBondIds(ps.bandId).filter((b) => m.gd.bondPlannable?.(b))[0] || null;
    return undefined;
  }

  /** Per-seat record for the match result. */
  info(m, ps, spec) {
    const o = offBonds(m);
    return { kind: spec.kind, base: spec.base, ...(ps?._nat ? { P: ps._nat.P, S: ps._nat.S, forcedLive: ps._nat.forcedLive } : {}),
      ...(spec.recipe ? { recipe: spec.recipe.id, P: spec.recipe.bond, live: !o.has(spec.recipe.bond) } : {}),
      ...(spec.sigma !== undefined ? { sigma: spec.sigma } : {}), plan: ps?.bondPlan ?? null };
  }
}
