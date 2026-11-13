// Teacher sanity check for version 4: the full-prep teacher (one-round-trip plan on / off) vs production legacy on the
// same seeds. node tools/rl/tcheck.mjs <mode teacher|legacy> <config> <seatPlan 1|0> [matches 96] [rehearsal 0] [bossRehearsal]
import { SPEnv } from './env.mjs';
const [mode, config, sp, nArg, rArg, brArg] = process.argv.slice(2);
const names = ['solo:NORMAL', 'solo:HARD', 'solo:ABYSS', 'coop4:NORMAL', 'coop4:HARD', 'coop4:ABYSS'];
const t0 = performance.now();
const env = new SPEnv({ envs: 16, configs: [config], seed: 7_000_000 + Math.max(0, names.indexOf(config)) * 1_000_000, envSeed: 7, maxMatches: Number(nArg || 96),
  humans: 'random', legacyMates: 0, mode, obsVersion: 4, seatPlan: sp !== '0', check: true, metrics: true, rehearsal: Number(rArg || 0),
  ...(brArg !== undefined ? { bossRehearsal: Number(brArg) } : {}), ...(mode === 'teacher' ? { enforce: true, recorder: { sample() {}, term() {} } } : {}) });
let b = env.reset(); while (b.n) b = env.step(new Int32Array(b.n));
const rows = env.results; let win = 0, fa = 0;
for (const r of rows) { if (r.seats.some((s) => s.win)) win++; fa += r.faShare || 0; }
const ps = env.planStats, per = (x) => (ps.plans ? +(x / ps.plans).toFixed(3) : null);
console.log(JSON.stringify({ mode, config, seatPlan: sp !== '0', n: rows.length, teamWin: +(win / rows.length).toFixed(3), fa: +(fa / rows.length).toFixed(3),
  sec: Math.round((performance.now() - t0) / 1000), errors: env.stats.errors, failed: env.stats.failed, violations: env.stats.violations,
  plans: ps.plans, stopRate: { any: per(ps.stopPreps.any), pool: per(ps.stopPreps.pool), offer: per(ps.stopPreps.offer), table: per(ps.stopPreps.table), foreign: per(ps.stopPreps.foreign) },
  teacher: mode === 'teacher' ? { preps: env.stats.teacher.preps, illegal: env.stats.teacher.illegal, unrepresented: env.stats.teacher.unrepresented, notReady: env.stats.teacher.notReady || 0 } : undefined,
  summary: env.summary(), samples: env.errorSamples.slice(0, 2) }));
