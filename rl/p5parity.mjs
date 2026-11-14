import { readFileSync } from 'node:fs';
const [tree, base, dump] = process.argv.slice(2);
const { SPPolicy5 } = await import(`${tree}/tools/rl/infer5.mjs`);
const net = SPPolicy5.load(base), J = JSON.parse(readFileSync(dump, 'utf8'));
let mx = 0, arg = 0;
J.obs.forEach((sp, n) => { const o = new Float32Array(J.D); for (const [i, v] of sp) o[i] = v; const r = net.forward(o, null).logits;
  for (let k = 0; k < 4; k++) mx = Math.max(mx, Math.abs(r[k] - J.out[n][k])); });
console.log(JSON.stringify({ rows: J.obs.length, maxAbsDiff: mx }));
