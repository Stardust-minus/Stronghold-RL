// tools/rl/release/play.mjs (11-14): play matches in-process with a release directory (export_release.py) under the headline
// protocol: the JS policy (infer2.mjs for the MLP, infer5.mjs for v5, greedy over the legal actions) plus the two deploy pickers
// (c9 on normal rounds, xboss-v3 on boss rounds). No search, no rehearsal, no live seed: every pick counts the battles it built
// (sims), which must stay 0. Prints team win per config and the per-decision cost; writes an eval.py-style JSON (stackrep.py
// reads it) and, with --dump, every policy decision (obs, mask, JS logits) for the torch parity check.
//   node tools/rl/release/play.mjs <release dir> [--configs solo:ABYSS,coop4h1:ABYSS] [--matches 8] [--seed 300000000]
//        [--envs 4] [--no-pickers] [--out <eval json>] [--dump <base>]
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { SPEnv } from '../env.mjs';
import { SPPolicy2 } from '../infer2.mjs';
import { SPPolicy5 } from '../infer5.mjs';
import { XPlacer } from '../xplace.mjs';

const argv = process.argv.slice(2), dir = resolve(argv[0] || '.');
const opt = (k, d) => { const i = argv.indexOf(k); return i > 0 ? argv[i + 1] : d; };
const configs = opt('--configs', 'solo:ABYSS,coop4h1:ABYSS').split(','), matches = Number(opt('--matches', 8));
const seed0 = Number(opt('--seed', 300_000_000)), nenv = Number(opt('--envs', 4)), pickers = !argv.includes('--no-pickers');
const outPath = opt('--out', null), dumpBase = opt('--dump', null);

const man = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8')), E = man.env;
// the code this release was exported and checked with (manifest.code: md5 per file of the game checkout); a mismatch is a warning
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
for (const [f, h] of Object.entries(man.code || {})) {
  let got = null; try { got = createHash('md5').update(readFileSync(resolve(root, f))).digest('hex'); } catch { /* missing */ }
  if (got !== h) console.warn(`warning: ${f} differs from the release's code (${got ? got.slice(0, 8) : 'missing'} vs ${h.slice(0, 8)})`);
}
const pm = JSON.parse(readFileSync(`${dir}/policy/policy.json`, 'utf8')), mb = () => process.memoryUsage().rss / 2 ** 20;
const rss0 = mb(), net = (pm.format === 'sp-policy5/1' ? SPPolicy5 : SPPolicy2).load(`${dir}/policy/policy`), rss1 = mb();
const xdir = `${dir}/${dirname(E.xPlace)}`, xp = JSON.parse(readFileSync(`${dir}/${E.xPlace}`, 'utf8'));
xp.model = resolve(xdir, xp.model); if (xp.pool?.gen) xp.pool.gen = resolve(xdir, xp.pool.gen);
const npick = { ...E.normalPicker, head: resolve(dir, E.normalPicker.head) };
// resident memory of the loaded models (information): the policy, then the two pickers loaded once more on their own
let keep = [SPPolicy5.load(npick.head.replace(/\.json$/, ''))]; const rss2 = mb(); keep.push(XPlacer.fromOptions(xp)); const rss3 = mb(); keep = null;

// eval.py's seed rule: a config's seeds depend on its name (CANON index, else a hash), so these matches pair with eval.py runs
const CANON = ['solo:NORMAL', 'solo:HARD', 'solo:ABYSS', 'coop4:NORMAL', 'coop4:HARD', 'coop4:ABYSS'];
const cidx = (c) => (CANON.includes(c) ? CANON.indexOf(c) : 100 + parseInt(createHash('sha256').update(c).digest('hex').slice(0, 6), 16) % 900);

const dump = dumpBase ? { obs: [], mask: [], logits: [], cfg: [] } : null;
const report = { release: man.name, memoryMB: { policy: +(rss1 - rss0).toFixed(1), c9: +(rss2 - rss1).toFixed(1), xboss: +(rss3 - rss2).toFixed(1) }, configs: {} };
console.log(`resident memory of the loaded models (MB): policy ${report.memoryMB.policy}, c9 ${report.memoryMB.c9}, xboss-v3 ${report.memoryMB.xboss}`);
for (const cfg of configs) {
  const coop = cfg.startsWith('coop');
  const env = new SPEnv({
    envs: Math.min(nenv, matches), configs: [cfg], seed: seed0 + cidx(cfg) * 1_000_000, envSeed: seed0, maxMatches: matches,
    humans: cfg.split(':')[0].endsWith('ai') ? 0 : 'random', legacyMates: 0, rehearsal: E.rehearsal, metrics: true, check: true,
    obsVersion: E.obsVersion, ...(E.obsExt ? { obsExt: E.obsExt } : {}), bondLog: true, natPlan: E.natPlan,
    ...(E.bandPolicy > 0 ? { bandPolicy: { share: E.bandPolicy, uniform: false } } : {}), bossRehearsal: E.bossRehearsal,
    ...(coop ? { mateRehearsal: E.mateRehearsal, humanProxy: { kinds: ['legacy'], log: true } } : {}),
    ...(pickers ? { normalPicker: npick, xPlace: xp } : {}), mode: 'policy',
  });
  let b = env.reset(), dec = 0, ms = 0, msMax = 0;
  if (b.n && b.D !== net.D) throw new Error(`observation size ${b.D} != policy D ${net.D} (env settings or dataId differ)`);
  const t0 = performance.now();
  while (b.n > 0) {
    const act = new Int32Array(b.n);
    for (let k = 0; k < b.n; k++) {
      const o = b.obs.subarray(k * b.D, (k + 1) * b.D), m = b.mask.subarray(k * b.A, (k + 1) * b.A);
      const t = performance.now(), { logits } = net.forward(o, m, { value: false }), dt = performance.now() - t;
      ms += dt; msMax = Math.max(msMax, dt); dec++;
      let best = -1, bv = -Infinity;
      for (let a = 0; a < b.A; a++) if (m[a] && logits[a] > bv) { bv = logits[a]; best = a; }
      act[k] = best;
      if (dump) { dump.obs.push(Float32Array.from(o)); dump.mask.push(Uint8Array.from(m)); dump.logits.push(Float32Array.from(logits.subarray ? logits.subarray(0, b.A) : logits)); dump.cfg.push(cfg); }
    }
    b = env.step(act);
  }
  const wall = (performance.now() - t0) / 1000, res = env.results;
  const sum = (k, f) => res.reduce((s, r) => s + (r[k] ? f(r[k]) : 0), 0);
  const np = { decisions: sum('npicker', (x) => x.decisions), ms: sum('npicker', (x) => x.ms), cands: sum('npicker', (x) => x.cands), sims: sum('npicker', (x) => x.sims) };
  const xb = { decisions: sum('xplace', (x) => x.decisions), ms: sum('xplace', (x) => x.ms), sims: sum('xplace', (x) => x.sims), msMax: Math.max(0, ...res.map((r) => r.xplace?.msMax || 0)) };
  const win = res.filter((r) => r.seats.some((s) => s.win)).length / Math.max(1, res.length);
  const per = (x) => (x.decisions ? +(x.ms / x.decisions).toFixed(1) : null);
  report.configs[cfg] = {
    matches: res.length, teamWin: win, errors: env.stats?.errors ?? null, wallSeconds: +wall.toFixed(1),
    policy: { decisions: dec, msPerDecision: +(ms / Math.max(1, dec)).toFixed(3), msMax: +msMax.toFixed(2) },
    ...(pickers ? { normalPicker: { ...np, msPerDecision: per(np), candsPerDecision: np.decisions ? +(np.cands / np.decisions).toFixed(1) : null },
      bossPlacement: { ...xb, msPerDecision: per(xb) } } : {}),
    perMatch: res.map((r) => ({ seed: r.seed, win: +r.seats.some((s) => s.win),
      bondLog: r.seats.filter((s) => s.rl && s.bondLog).map((s) => ({ ...s.bondLog, band: s.band, win: s.win })) })),
  };
  const c = report.configs[cfg];
  console.log(`${cfg}: ${c.matches} matches, team win ${win.toFixed(3)}, policy ${c.policy.msPerDecision} ms/decision (${dec})` +
    (pickers ? `, c9 ${c.normalPicker.msPerDecision} ms (${np.decisions}, sims ${np.sims}), xboss-v3 ${c.bossPlacement.msPerDecision} ms (${xb.decisions}, sims ${xb.sims})` : '') + `, ${c.wallSeconds} s`);
  env.close();
}
report.memoryMB.processPeak = +(process.resourceUsage().maxRSS / 1024).toFixed(1); // the whole process: engine, env and models
if (outPath) writeFileSync(outPath, JSON.stringify(report));
if (dump) {
  const n = dump.obs.length, D = net.D, A = dump.mask[0]?.length || 0, cat = (L, T, w) => { const o = new T(n * w); L.forEach((x, i) => o.set(x, i * w)); return o; };
  writeFileSync(`${dumpBase}.obs.f32`, Buffer.from(cat(dump.obs, Float32Array, D).buffer));
  writeFileSync(`${dumpBase}.mask.u8`, Buffer.from(cat(dump.mask, Uint8Array, A).buffer));
  writeFileSync(`${dumpBase}.logits.f32`, Buffer.from(cat(dump.logits, Float32Array, A).buffer));
  writeFileSync(`${dumpBase}.json`, JSON.stringify({ n, D, A, cfg: dump.cfg }));
}
