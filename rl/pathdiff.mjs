// band-path obs diff (11-12): the same solo:ABYSS seed, (A) the policy picks sarkazb (bandPolicy share 1), (B) the band forced
// to sarkazb; both then play END at every decision. Prints which observation fields differ at the first decisions.
import { SPEnv } from './env.mjs';
import { ACT } from './spec.mjs';
const seed = Number(process.argv[2] || 7_000_000 + 2 * 1_000_000), base = { envs: 1, configs: ['solo:ABYSS'], seed, envSeed: 7_000_000, maxMatches: 1, humans: 'random',
  legacyMates: 0, obsVersion: 3, rehearsal: 0, bossRehearsal: 3, mode: 'policy', metrics: true, check: true };
const A = new SPEnv({ ...base, bandPolicy: { share: 1, uniform: false } }), B = new SPEnv({ ...base, forceBand: 'band_sarkazb' });
const spec = A.spec(), blocks = spec.blocks, D = spec.D ?? A.encoder.dim;
const where = (i) => { for (const [k, b] of Object.entries(blocks)) { const w = (b.count || 1) * (b.width || b.dim || 0); if (i >= b.offset && i < b.offset + w) return `${k}[${Math.floor((i - b.offset) / (b.width || w))}].${(i - b.offset) % (b.width || w)}`; } return `?${i}`; };
let a = A.reset(), b = B.reset(); const bi = A.encoder.vocab.bandList.indexOf('band_sarkazb');
const diffs = new Map(); let steps = 0;
for (let t = 0; t < 400 && a.n && b.n; t++) {
  const actA = new Int32Array(a.n), actB = new Int32Array(b.n);
  for (let k = 0; k < a.n; k++) actA[k] = a.kinds[k] === 2 ? ACT.BAND + bi : ACT.END;
  for (let k = 0; k < b.n; k++) actB[k] = ACT.END;
  if (a.kinds[0] !== 2 && a.n === b.n) {
    steps++;
    for (let i = 0; i < D; i++) if (Math.abs(a.obs[i] - b.obs[i]) > 1e-6) { const w = where(i); const e = diffs.get(w) || { n: 0, ex: [] }; e.n++; if (e.ex.length < 3) e.ex.push([+a.obs[i].toFixed(3), +b.obs[i].toFixed(3), t]); diffs.set(w, e); }
    a = A.step(actA); b = B.step(actB);
  } else a = A.step(actA);
}
console.log(JSON.stringify({ seed, comparedSteps: steps, bandsA: A.results[0]?.seats?.map((s) => s.band), bandsB: B.results[0]?.seats?.map((s) => s.band),
  fieldsDiffering: diffs.size, top: [...diffs.entries()].sort((x, y) => y[1].n - x[1].n).slice(0, 25) }));
