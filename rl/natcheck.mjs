import { SPEnv, randomActions } from './env.mjs';
const env = new SPEnv({ envs: 4, configs: ['solo:ABYSS', 'coop4h1:ABYSS'], seed: 8_100_000, maxMatches: 16, obsVersion: 3, natPlan: { phase: 'forced' }, rewardV: 2,
  r2: { leak: 0.15, save: 0.075, leakCap: true, layer: 0 }, shaping: { gamma: 0.99, wTier: 0.06, wStack: 0, uncapped: true }, bandPolicy: { share: 1 }, legacyMates: 0 });
const E = env.encoder.dim, D = env.D; let n = 0, planOn = 0, phiN = [], cnt = [], steps = 0;
let b = env.reset();
while (b.n && steps < 4000) {
  for (let k = 0; k < b.n; k++) {
    const dec = env.items[k]; if (dec.kind !== 'prep') continue; n++;
    if (b.obs[k * D + E] === 1) planOn++;
    const s = dec.slot.seat.get(dec.ps.playerId); if (s?.plan && dec.own) { phiN.push(env._natPhi(dec.own.bonds, s.plan)); cnt.push(SPEnv._bv(dec.own.bonds, s.plan.P).count); }
  }
  b = env.step(randomActions(b)); steps++;
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
console.log(JSON.stringify({ prepDecisions: n, planBlockOn: +(planOn / Math.max(1, n)).toFixed(3), phiNatMean: +mean(phiN).toFixed(4), phiNatMax: Math.max(0, ...phiN), shareWithPUnits: +(cnt.filter((c) => c > 0).length / Math.max(1, cnt.length)).toFixed(3), D, E }));
process.exit(0);
