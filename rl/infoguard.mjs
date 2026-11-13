// infoguard (11-13): mode on = visibleOnly: the RL seat's encode never reads another seat's hand / temp except through the
// scout view, and the hidden boss id stays 0 before its round; strict = + strictPreview: legacy placement for the RL seat never
// reads spawn routeIndex / mods or enemyHpMulAt; off = the positive control (hits expected). node infoguard.mjs [on|strict|off] [n]
const { SPEnv } = await import('./env.mjs');
const mode = process.argv[2] || 'on', vis = mode !== 'off', strict = mode === 'strict', N = Number(process.argv[3] || 3);
let rng = 4242; const rnd = () => ((rng = (rng * 1103515245 + 12345) >>> 0) / 4294967296);
const out = {}; let bad = 0;
for (const [ci, cfg] of ['solo:ABYSS', 'coop4l:ABYSS', 'coop4:ABYSS', 'coop4ai:ABYSS', 'solo:HARD'].entries()) {
  const env = new SPEnv({ envs: 2, configs: [cfg], seed: 9_970_000 + ci * 1000, envSeed: 97, maxMatches: N, humans: 'random', legacyMates: 0.3, obsVersion: 3,
    rehearsal: 0, bossRehearsal: 0, mode: 'policy', check: true, bandPolicy: { share: 1, uniform: false }, visibleOnly: vis, strictPreview: strict, infoGuard: true });
  const A = env.A; let b = env.reset(), rows = 0;
  while (b.n) { const act = new Int32Array(b.n); for (let k = 0; k < b.n; k++) { const legal = []; for (let j = 0; j < A; j++) if (b.mask[k * A + j]) legal.push(j); act[k] = legal.length ? legal[Math.floor(rnd() * legal.length)] : 0; } rows += b.n; b = env.step(act); }
  const g = env.infoGuard; const hits = g.otherHand + g.hiddenBoss + (strict ? g.routeIndex + g.mods + g.hpMul : 0); bad += hits;
  out[cfg] = { matches: env.stats.matches, rows, otherHand: g.otherHand, hiddenBoss: g.hiddenBoss, routeIndex: g.routeIndex, mods: g.mods, hpMul: g.hpMul, scoutReads: g.scoutReads, prepHandReads: g.prepHandReads, errors: env.stats.errors, samples: g.samples.slice(0, 4) };
}
console.log(JSON.stringify({ mode, ok: vis ? bad === 0 : bad > 0, out }));
process.exit(vis ? (bad === 0 ? 0 : 1) : 0);
