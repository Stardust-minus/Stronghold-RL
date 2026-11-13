// Version 4 (full control) smoke test: a random policy and the full-prep teacher on obs v4 — errors, engine-rejected
// actions, invariant violations, Ready fallbacks, the teacher's unrepresented calls, and the one-round-trip plan's stop
// rates per prep (pool / offer / table / foreign).
//   node tools/rl/smoke4.mjs [--matches 8] [--configs coop4:HARD,solo:NORMAL] [--seat-plan 1|0] [--only policy|teacher]
import { SPEnv, randomActions } from './env.mjs';
import { KIND } from './spec.mjs';
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const matches = Number(arg('matches', 8)), configs = arg('configs', 'coop4:HARD,solo:NORMAL').split(','), only = arg('only', '');
const seatPlan = arg('seat-plan', '1') !== '0';
for (const mode of ['policy', 'teacher']) {
  if (only && only !== mode) continue;
  const t0 = performance.now();
  const env = new SPEnv({ envs: 4, configs, seed: 4100, envSeed: 41, maxMatches: matches, humans: 'random', legacyMates: 0.3, mode,
    obsVersion: 4, seatPlan, check: true, metrics: true, ...(mode === 'teacher' ? { enforce: true, recorder: { sample() {}, term() {} } } : {}) });
  let b = env.reset(); const kinds = {}; let rows = 0, steps = 0;
  while (b.n) {
    for (let k = 0; k < b.n; k++) kinds[b.kinds[k]] = (kinds[b.kinds[k]] || 0) + 1;
    rows += b.n; steps++;
    b = env.step(randomActions(b));
  }
  const ps = env.planStats, per = (x) => (ps.plans ? +(x / ps.plans).toFixed(3) : null);
  console.log(JSON.stringify({ mode, seatPlan, matches: env.stats.matches, sec: +((performance.now() - t0) / 1000).toFixed(1), dim: env.D, actions: env.A,
    decisionsByKind: Object.fromEntries(Object.entries(kinds).map(([k, v]) => [Object.keys(KIND).find((n) => KIND[n] === +k), v])),
    errors: env.stats.errors, rejected: env.stats.rejected, failed: env.stats.failed, violations: env.stats.violations, forcedEnd: env.stats.forcedEnd,
    plans: ps.plans, stopRatePerPrep: { any: per(ps.stopPreps.any), pool: per(ps.stopPreps.pool), offer: per(ps.stopPreps.offer), table: per(ps.stopPreps.table), foreign: per(ps.stopPreps.foreign) },
    stops: ps.stops, planRejected: ps.rejected, readyFallback: ps.readyFallback, bossDeferred: ps.bossDeferred,
    teacher: mode === 'teacher' ? { preps: env.stats.teacher.preps, calls: env.stats.teacher.calls, illegal: env.stats.teacher.illegal, unrepresented: env.stats.teacher.unrepresented, notReady: env.stats.teacher.notReady || 0 } : undefined,
    summary: env.summary().slice(0, 6), samples: env.errorSamples.slice(0, 4) }));
  env.close();
}
