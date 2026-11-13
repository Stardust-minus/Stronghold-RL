// infer3 checks on real prep sequences: incremental == full (bit-exact), batch == single (bit-exact), infer3 vs infer2,
// timing per decision (full / incremental / batch of 3 seats) and the single-legal share. node parity3.mjs <policy base> [n]
import { performance } from 'node:perf_hooks';
import { SPEnv, randomActions } from './env.mjs';
import { SPPolicy2 } from './infer2.mjs';
import { SPPolicy3 } from './infer3.mjs';
const base = process.argv[2], N = Number(process.argv[3] || 2000);
const p2 = SPPolicy2.load(base), p3 = SPPolicy3.load(base);
const env = new SPEnv({ envs: 6, configs: ['coop4:ABYSS', 'solo:HARD', 'coop4:HARD', 'solo:ABYSS'], seed: 3900000000, envSeed: 6, humans: 'random', obsVersion: 2 });
const seq = new Map(); let b = env.reset(), n = 0; // trajectory -> its decisions in order
while (n < N && b.n) {
  for (let k = 0; k < b.n && n < N; k++, n++) { const t = b.ids[k]; if (!seq.has(t)) seq.set(t, []); seq.get(t).push({ obs: b.obs.slice(k * b.D, (k + 1) * b.D), mask: b.mask.slice(k * b.A, (k + 1) * b.A).subarray(0, p3.A) }); }
  b = env.step(randomActions(b));
}
env.close();
const all = [...seq.values()].flat();
let exactInc = 0, exactBatch = 0, worst2 = 0, single = 0;
for (const list of seq.values()) {
  const cache = p3.newCache();
  for (const d of list) {
    const full = p3.forward(d.obs, d.mask), inc = p3.forward(d.obs, d.mask, { cache });
    let same = true; for (let a = 0; a < p3.A; a++) if (full.logits[a] !== inc.logits[a]) same = false; exactInc += same;
    const r2 = p2.forward(d.obs, d.mask, { value: false });
    for (let a = 0; a < p3.A; a++) if (d.mask[a]) worst2 = Math.max(worst2, Math.abs(r2.logits[a] - full.logits[a]));
    let legal = 0; for (let a = 0; a < p3.A; a++) legal += d.mask[a]; single += legal === 1;
  }
}
for (let i = 0; i + 2 < all.length; i += 3) {
  const bat = p3.forwardBatch([0, 1, 2].map((j) => ({ obs: all[i + j].obs, mask: all[i + j].mask, cache: null })));
  let same = true; for (let j = 0; j < 3; j++) { const s = p3.forward(all[i + j].obs, all[i + j].mask); for (let a = 0; a < p3.A; a++) if (s.logits[a] !== bat[j].logits[a]) same = false; }
  exactBatch += same;
}
const time = (f) => { const t0 = performance.now(); const k = f(); return (performance.now() - t0) / k; };
for (let w = 0; w < 2; w++) for (const d of all.slice(0, 200)) p3.forward(d.obs, d.mask);
const msFull = time(() => { for (const d of all) p3.forward(d.obs, d.mask); return all.length; });
p3.stats = { rows: 0, rowsReused: 0 };
const msInc = time(() => { for (const list of seq.values()) { const c = p3.newCache(); for (const d of list) p3.forward(d.obs, d.mask, { cache: c }); } return all.length; });
const reuse = p3.stats.rowsReused / p3.stats.rows;
const msBatch = time(() => { let k = 0; for (let i = 0; i + 2 < all.length; i += 3, k += 3) p3.forwardBatch([0, 1, 2].map((j) => ({ obs: all[i + j].obs, mask: all[i + j].mask }))); return k; });
const msInfer2 = time(() => { for (const d of all) p2.forward(d.obs, d.mask, { value: false }); return all.length; });
console.log(JSON.stringify({ decisions: all.length, trajectories: seq.size, incrementalBitExact: `${exactInc}/${all.length}`, batchBitExact: `${exactBatch}/${Math.floor(all.length / 3)}`,
  maxAbsDiffVsInfer2: worst2, singleLegalShare: +(single / all.length).toFixed(4), rowReuse: +reuse.toFixed(3),
  msPerDecision: { infer2: +msInfer2.toFixed(3), infer3Full: +msFull.toFixed(3), infer3Incremental: +msInc.toFixed(3), infer3Batch3: +msBatch.toFixed(3) } }));
