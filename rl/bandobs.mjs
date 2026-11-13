// Band-decision observations for the offline band-head warm start: obs v3, the policy picks every model seat's band
// (bandPolicy share 1), random play otherwise; writes <out>.f32 (obs), <out>.u8 (mask), <out>.json (configs per row, bandList).
//   node tools/rl/bandobs.mjs <out prefix> [matches per config 120]
import { SPEnv, randomActions } from './env.mjs';
import { KIND } from './spec.mjs';
import { writeFileSync } from 'node:fs';
const [out, nArg] = process.argv.slice(2);
const configs = ['solo:HARD', 'solo:ABYSS', 'coop4:HARD', 'coop4:ABYSS', 'solo:NORMAL', 'coop4:NORMAL', 'coop2:ABYSS', 'coop2:HARD'];
const obs = [], masks = [], cfgs = [];
let bandList = null, D = 0, A = 0;
for (const cfg of configs) {
  const env = new SPEnv({ envs: 8, configs: [cfg], seed: 5_500_000 + configs.indexOf(cfg) * 10_000, envSeed: 55, maxMatches: Number(nArg || 120), humans: 'random',
    legacyMates: 0.3, obsVersion: 3, bandPolicy: { share: 1, planProb: 0.5, uniform: false } });
  bandList = env.encoder.vocab.bandList; D = env.D; A = env.A;
  let b = env.reset();
  while (b.n) {
    for (let k = 0; k < b.n; k++) if (b.kinds[k] === KIND.band) { obs.push(b.obs.slice(k * D, (k + 1) * D)); masks.push(b.mask.slice(k * A, (k + 1) * A)); cfgs.push(cfg); }
    b = env.step(randomActions(b));
  }
  env.close();
}
const O = new Float32Array(obs.length * D), M = new Uint8Array(obs.length * A);
obs.forEach((o, i) => O.set(o, i * D)); masks.forEach((m, i) => M.set(m, i * A));
writeFileSync(out + '.f32', Buffer.from(O.buffer)); writeFileSync(out + '.u8', Buffer.from(M.buffer));
writeFileSync(out + '.json', JSON.stringify({ n: obs.length, D, A, configs: cfgs, bandList, bandAction0: 60 }));
console.log(JSON.stringify({ n: obs.length, D, A, perConfig: Object.fromEntries(configs.map((c) => [c, cfgs.filter((x) => x === c).length])) }));
