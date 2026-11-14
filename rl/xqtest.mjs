// xplace: JS QNet scores of stored candidates (parity with xqtest.py). node xqtest.mjs model.json file.jsonl N
import { readFileSync } from 'node:fs';
import { QNetJS } from './xplace.mjs';
const [mp, f, N] = process.argv.slice(2);
const net = QNetJS.load(mp), raw = readFileSync(f.replace(/\.jsonl$/, '.bin'));
for (const line of readFileSync(f, 'utf8').split('\n').filter(Boolean).slice(0, Number(N || 20))) {
  const r = JSON.parse(line), { FP, FT, FE, FG } = r.dims, n = r.n; let o = r.bin[0];
  const take = (cnt, T, sz) => { const b = raw.subarray(o, o + cnt * sz); o += cnt * sz; const ab = new ArrayBuffer(cnt * sz); new Uint8Array(ab).set(b); return new T(ab); };
  const f16 = (cnt) => Float32Array.from(take(cnt, Float16Array, 2));
  const X = { n, fp: f16(n * FP), ft: f16(28 * FT), fe: f16(n * 112 * FE), fg: f16(FG), ids: take(n, Int32Array, 4), legal: take(n * 112, Uint8Array, 1) };
  const cs = r.cands.slice(0, 8), cfs = cs.map((c) => Float32Array.from(Float16Array.from(c.f || new Array(26).fill(0))));
  const t0 = performance.now(), enc = net.encode(X), q = net.score(X, enc, cs.map((c) => c.a), cfs);
  console.log(JSON.stringify({ q: q.map((x) => +x.toFixed(4)), ms: +(performance.now() - t0).toFixed(2) }));
}
