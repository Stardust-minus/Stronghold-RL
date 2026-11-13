// tools/rl/picker_reencode.mjs — re-encode stored picker labels for another trunk: every record's sparse observation
// (posvalue pickrec --rec, `obs: { i, v }`, `D`) through the trunk's forward → its context `c`; everything else unchanged.
//   node tools/rl/picker_reencode.mjs <trunk JS export base> <in.jsonl> <out.jsonl>   (records without `obs` are skipped)
import { readFileSync, writeFileSync } from 'node:fs';
import { SPPolicy2 } from './infer2.mjs';
const [base, inp, out] = process.argv.slice(2);
const net = SPPolicy2.load(base), D = net.D, A = net.A, ob = new Float32Array(D), mk = new Uint8Array(A).fill(1);
let n = 0, skipped = 0; const lines = [];
for (const line of readFileSync(inp, 'utf8').split('\n')) {
  if (!line) continue;
  const d = JSON.parse(line);
  if (!d.obs || d.D !== D) { skipped++; continue; }
  ob.fill(0); d.obs.i.forEach((q, k) => { ob[q] = d.obs.v[k]; });
  net.forward(ob, mk, { value: false });
  d.c = Array.from(net.c, (x) => +x.toFixed(4)); lines.push(JSON.stringify(d)); n++;
}
writeFileSync(out, lines.join('\n') + (lines.length ? '\n' : ''));
console.log(JSON.stringify({ in: inp, out, reencoded: n, skipped }));
