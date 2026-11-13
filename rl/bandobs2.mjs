// Band-decision observations v2 (11-13): obs v3 at every model seat's own band decision (bandPolicy share 1, after the
// human proxies); a match is closed at its first prep (random actions until then); legacyMates 0 as in training (mp1). Per row: config, the human-proxy count, seat.
// Writes <out>.f32 (obs), <out>.u8 (mask), <out>.json (configs, humans, seat per row, bandList).
//   node tools/rl/bandobs2.mjs <out prefix> <matches per config> <config,config,...>
import { SPEnv, randomActions } from './env.mjs';
import { KIND } from './spec.mjs';
import { writeFileSync } from 'node:fs';
const [out, nArg, cfgArg] = process.argv.slice(2);
const configs = cfgArg.split(',');
const obs = [], masks = [], cfgs = [], hum = [], seat = [];
let bandList = null, D = 0, A = 0;
for (const cfg of configs) {
  const env = new SPEnv({ envs: 8, configs: [cfg], seed: 5_600_000 + configs.indexOf(cfg) * 10_000, envSeed: 56, maxMatches: Number(nArg || 150), humans: 'random',
    legacyMates: 0, obsVersion: 3, bandPolicy: { share: 1, planProb: 0.5, uniform: false } });
  bandList = env.encoder.vocab.bandList; D = env.D; A = env.A;
  let b = env.reset();
  while (b.n) {
    for (let k = 0; k < b.n; k++) if (b.kinds[k] === KIND.band) {
      obs.push(b.obs.slice(k * D, (k + 1) * D)); masks.push(b.mask.slice(k * A, (k + 1) * A)); cfgs.push(cfg);
      const dec = env.items[k]; hum.push(dec.slot.humans.size); seat.push(dec.ps.playerId);
    }
    b = env.step(randomActions(b));
    // only the band decisions matter: a match whose first prep began is closed (counted failed: 'bandobs')
    for (const s of env.slots) if (s && !s.failed && !s.ended && s.m && s.m.phase === 'PREP' && !s.deferredBand?.length) { s.pending.length = 0; s.failed = 'bandobs'; }
  }
  env.close();
}
const O = new Float32Array(obs.length * D), M = new Uint8Array(obs.length * A);
obs.forEach((o, i) => O.set(o, i * D)); masks.forEach((m, i) => M.set(m, i * A));
writeFileSync(out + '.f32', Buffer.from(O.buffer)); writeFileSync(out + '.u8', Buffer.from(M.buffer));
writeFileSync(out + '.json', JSON.stringify({ n: obs.length, D, A, configs: cfgs, humans: hum, seats: seat, bandList, bandAction0: 60 }));
console.log(JSON.stringify({ n: obs.length, D, A, perConfig: Object.fromEntries(configs.map((c) => [c, cfgs.filter((x) => x === c).length])) }));
