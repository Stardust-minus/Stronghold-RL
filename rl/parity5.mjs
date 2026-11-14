// (A1) infer5.mjs vs torch: node parity5.mjs <policy5 base> <dump base> — max |Δlogit| over legal actions, argmax agreement, ms / decision
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { SPPolicy5 } from './infer5.mjs';
const [base, dump] = process.argv.slice(2);
const net = SPPolicy5.load(base), M = JSON.parse(readFileSync(dump + '.json', 'utf8')), buf = readFileSync(dump + '.bin');
const { n, D, A } = M;
const f = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + 4 * n * D));
const mk = new Uint8Array(buf.buffer.slice(buf.byteOffset + 4 * n * D, buf.byteOffset + 4 * n * D + n * A));
const ref = new Float32Array(buf.buffer.slice(buf.byteOffset + 4 * n * D + n * A, buf.byteOffset + 4 * n * D + n * A + 4 * n * A));
let maxd = 0, agree = 0, toks = 0; const t0 = performance.now();
for (let i = 0; i < n; i++) {
  const o = f.subarray(i * D, (i + 1) * D), m = mk.subarray(i * A, (i + 1) * A);
  const { logits, tokens } = net.forward(o, m); toks += tokens;
  let bj = -1, bt = -1, vj = -Infinity, vt = -Infinity;
  for (let a = 0; a < A; a++) { if (!m[a]) continue; const d = Math.abs(logits[a] - ref[i * A + a]); if (d > maxd) maxd = d;
    if (logits[a] > vj) { vj = logits[a]; bj = a; } if (ref[i * A + a] > vt) { vt = ref[i * A + a]; bt = a; } }
  if (bj === bt) agree++;
}
const ms = (performance.now() - t0) / n;
console.log(JSON.stringify({ n, maxAbsDiff: maxd, argmaxAgree: agree / n, msPerDecision: +ms.toFixed(2), meanTokens: +(toks / n).toFixed(1) }));
