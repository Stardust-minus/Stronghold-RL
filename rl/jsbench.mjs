// jsbench.mjs <policy base> <dump prefix> [reps]: plain-JS forward (infer2) per decision: ms / decision, max |d logit| vs torch
import { SPPolicy2 } from './infer2.mjs';
import { readFileSync } from 'node:fs';
const [base, dump, repsArg] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(dump + '.json', 'utf8')), { n, D, A } = meta;
const rd = (f, T) => { const b = readFileSync(f); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const O = rd(dump + '.obs.f32', Float32Array), M = rd(dump + '.mask.u8', Uint8Array), R = rd(dump + `.${meta.ref}.f32`, Float32Array);
const net = SPPolicy2.load(base);
let maxd = 0;
for (let i = 0; i < n; i++) {
  const { logits } = net.forward(O.subarray(i * D, (i + 1) * D), M.subarray(i * A, (i + 1) * A), { value: false });
  for (let a = 0; a < A; a++) if (M[i * A + a]) maxd = Math.max(maxd, Math.abs(logits[a] - R[i * A + a]));
}
const reps = Number(repsArg || 3); let best = Infinity;
for (let r = 0; r < reps; r++) {
  const t0 = performance.now();
  for (let i = 0; i < n; i++) net.forward(O.subarray(i * D, (i + 1) * D), M.subarray(i * A, (i + 1) * A), { value: false });
  best = Math.min(best, performance.now() - t0);
}
console.log(JSON.stringify({ base, rows: n, msPerDecision: +(best / n).toFixed(4), maxDLogitVsTorch: maxd }));
