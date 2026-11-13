// LP regression check (11-12): band policy 1 in every config, random legal actions, --check on; the env's lp0 / own-LP
// guard must stay silent. Phase 2: late-game (snapshot) starts in ABYSS (collect every eligible checkpoint, restore half). node lpguard.mjs [env module] [matches per config]
const mod = process.argv[2] || './env.mjs', N = Number(process.argv[3] || 4);
const { SPEnv } = await import(mod);
const cfgs = ['solo:NORMAL', 'solo:HARD', 'solo:ABYSS', 'coop2:HARD', 'coop2:ABYSS', 'coop3:HARD', 'coop4:NORMAL', 'coop4:HARD', 'coop4:ABYSS', 'coop4ai:HARD', 'coop4ai:ABYSS'];
let rng = 12345; const rnd = () => ((rng = (rng * 1103515245 + 12345) >>> 0) / 4294967296);
const out = {};
for (const [ci, cfg] of cfgs.entries()) {
  const env = new SPEnv({ envs: 2, configs: [cfg], seed: 9_900_000 + ci * 1000, envSeed: 99, maxMatches: N, humans: 'random', legacyMates: 0.3, obsVersion: 3,
    rehearsal: 0, bossRehearsal: 3, mode: 'policy', check: true, bandPolicy: { share: 1, uniform: false } });
  const A = env.A; let b = env.reset(), rows = 0, outOfRange = 0; const G0 = env.G0;
  while (b.n) {
    const act = new Int32Array(b.n);
    for (let k = 0; k < b.n; k++) {
      const v = b.obs[k * env.D + G0 + 9]; rows++; if (!(v >= 0 && v <= 1.05)) outOfRange++;
      const legal = []; for (let j = 0; j < A; j++) if (b.mask[k * A + j]) legal.push(j);
      act[k] = legal.length ? legal[Math.floor(rnd() * legal.length)] : 0;
    }
    b = env.step(act);
  }
  out[cfg] = { matches: env.stats.matches, rows, outOfRange, guard: env.stats.guard || 0, errors: env.stats.errors, sample: env.errorSamples.filter((x) => x.includes('guard')).slice(0, 2) };
}
for (const [ci, cfg] of ['solo:ABYSS', 'coop4:ABYSS', 'coop4ai:ABYSS'].entries()) {
  const env = new SPEnv({ envs: 2, configs: [cfg], seed: 9_950_000 + ci * 1000, envSeed: 77, maxMatches: 3 * N + 4, humans: 'random', legacyMates: 0.3, obsVersion: 3,
    rehearsal: 0, bossRehearsal: 3, mode: 'policy', check: true, bandPolicy: { share: 1, uniform: false },
    snapCollect: { prob: 1, rounds: [4, 5, 6, 7, 8, 9, 10] }, snapStart: { frac: 0.5 } });
  const A = env.A; let b = env.reset(), rows = 0, outOfRange = 0, below1 = 0; const G0 = env.G0;
  while (b.n) {
    const act = new Int32Array(b.n);
    for (let k = 0; k < b.n; k++) {
      const v = b.obs[k * env.D + G0 + 9]; rows++; if (!(v >= 0 && v <= 1.05)) outOfRange++; if (v < 0.999) below1++;
      const legal = []; for (let j = 0; j < A; j++) if (b.mask[k * A + j]) legal.push(j);
      act[k] = legal.length ? legal[Math.floor(rnd() * legal.length)] : 0;
    }
    b = env.step(act);
  }
  const snapRes = env.results.filter((r) => r.fromSnapshot);
  out['snap ' + cfg] = { matches: env.stats.matches, snapStarts: env.snapStats.started, collected: env.snapStats.collected, rows, outOfRange, lpFieldBelow1: below1,
    guard: env.stats.guard || 0, errors: env.stats.errors, snapRounds: snapRes.map((r) => r.fromSnapshot), sample: env.errorSamples.filter((x) => x.includes('guard')).slice(0, 2) };
}
const bad = Object.values(out).reduce((n, x) => n + x.guard + x.outOfRange, 0) + (Object.entries(out).some(([k, x]) => k.startsWith('snap') && !x.snapStarts) ? 1 : 0);
console.log(JSON.stringify({ module: mod, ok: bad === 0, out }));
process.exit(bad ? 1 : 0);
