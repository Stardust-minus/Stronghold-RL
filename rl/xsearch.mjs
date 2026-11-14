// tools/rl/xsearch.mjs — xplace (11-13): the search-label hook (training only; sims are the teacher), shared by the JS-policy
// driver xlabel.mjs and the env option xLabel (eval.py --xlabel: any trunk eval.py runs, e.g. the plan-conditioned policies).
// createLabelHook(env, o) -> { hook(m, ps, slot), S, close() }; o: { out, K, KL, ls, seeds, top, simplify, wleak, nkey, apply,
// rounds, maxMs, storeObs, candFeats, xmodel, xmargin, play, cfg, net (optional JS trunk for c) }. Files: <out>.jsonl / .bin.
import { appendFileSync, writeFileSync, openSync, writeSync, closeSync, existsSync, statSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { xPlaceProblemForTraining, xRehearseForTraining, xPerturbForTraining, layoutCandidatesForPicker, applyPlanForPicker, LayoutPlan } from '../../server/match/bot.js';
import { xFeaturize, xAssignOf, xCandFeatures } from './xfeat.mjs';
import { XNetJS } from './xplace.mjs';
import { createTeachHook } from './xteach.mjs'; // X7: o.teach -> the teacher-search bake-off hook (off by default)

export function createLabelHook(env, o = {}) {
  if (o.teach) return createTeachHook(env, o);
  const OUT = o.out; if (!OUT) throw new Error('xsearch: out required');
  const K = Number(o.K ?? 24), KL = Number(o.KL ?? 12), LS = Number(o.ls ?? 40), SEEDS = Number(o.seeds ?? 3), TOP = Number(o.top ?? 3);
  const SIMPLIFY = o.simplify !== false && o.simplify !== 0 && o.simplify !== '0', WLEAK = Number(o.wleak ?? 8), APPLY = o.apply || 'best', ROUNDS = o.rounds || 'all';
  const MAXMS = Number(o.maxMs ?? 60000), NKEY = o.nkey || 'd', STORE_OBS = !!o.storeObs, CANDF = o.candFeats !== false && o.candFeats !== 0;
  const XSAMPLES = Number(o.xsamples || 0), XTEMP = Number(o.xtemp || 1), SALT = !!o.seedSalt; // seedSalt: per-match rehearsal seeds
  const XM = o.xmodel ? XNetJS.load(o.xmodel) : null, XMARGIN = Number(o.xmargin || 0), PLAY = o.play || 'best', cfg = o.cfg || null, net = o.net || null;
  if (existsSync(OUT + '.jsonl') && statSync(OUT + '.jsonl').size > 0 && !o.overwrite) throw new Error(`${OUT}.jsonl exists (overwrite)`);
  writeFileSync(OUT + '.jsonl', ''); const binFd = openSync(OUT + '.bin', 'w'); let binOff = 0;
  const S = { decisions: 0, normal: 0, boss: 0, changed: 0, rehearsals: 0, ms: 0, msMax: 0, lsImproved: 0, valFlip: 0, simplified: 0, model: 0, modelCur: 0, modelPick: 0, modelBetter: 0, modelWorse: 0, modelIsCur: 0, played: 0, skipped: 0, dKey: 0, dLayers: 0, dLeaks: 0, dBoss: 0 };
  function rngOf(str) { // mulberry32 seeded by an FNV hash of str
    let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    let a = h || 1;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const sigOf = (plan) => [...plan.entries()].sort((a, b) => a[0] - b[0]).map(([u, k]) => `${u}@${k}:${plan.dirs?.get(u) ?? 'RIGHT'}`).join(' ');
  const mean = (xs) => { const ok = xs.filter(Boolean); if (!ok.length) return null; const o = { leaks: 0, boss: 0, kills: 0, layers: 0 }; for (const x of ok) for (const k in o) o[k] += x[k] / ok.length; return o; };
  // normal-round key: --nkey d (default; the picker's adopted rule "D"): no more leaks than the current (legacy) layout on the same
  // seeds, then the most layers, then kills; --nkey soft: layers + 0.01 kills - wleak x leaks (label set 1). s0 = the
  // current layout's score on the same seed(s).
  const keyOf = (boss, s, s0) => (!s ? -Infinity : boss ? -s.leaks * 1e7 + s.boss + 1e-3 * s.kills
    : NKEY === 'd' ? s.layers + 0.01 * s.kills - 1000 * Math.max(0, s.leaks - (s0 ? s0.leaks : s.leaks) - 1e-9)
    : -WLEAK * s.leaks + s.layers + 0.01 * s.kills);
  const pack = (s) => (s ? [s.leaks, +s.boss.toFixed(1), s.kills, s.layers] : null);
  function writeBin(arrs) {
    const off = binOff, parts = [];
    for (const a of arrs) { const f = a instanceof Float32Array ? Float16Array.from(a) : a; const b = Buffer.from(f.buffer, f.byteOffset, f.byteLength); parts.push(b); }
    const buf = Buffer.concat(parts); writeSync(binFd, buf); binOff += buf.length; return [off, buf.length];
  }

  const hook = (m, ps, slot, envIn = null) => {
    const E = env || envIn;
    if (!slot.rl.has(ps.playerId) && !slot.recipeSeat?.has(ps.playerId)) return;
    if (slot.humans.has(ps.playerId)) return;
    const boss = !m.wave;
    if (ROUNDS !== 'all' && (ROUNDS === 'boss') !== boss) return;
    const t0 = performance.now();
    const P = xPlaceProblemForTraining(m, ps); if (!P || !P.fight) { S.skipped++; return; }
    const seat = E._seat(slot, ps), X = xFeaturize(m, ps, P, { lp0: seat?.lp0 });
    // --store-obs 1: the seat's sparse prep observation and the policy trunk's c (re-encodable for another trunk)
    let obsRec = null;
    if (STORE_OBS) {
      const ob = new Float32Array(E.encoder.dim), mk = new Uint8Array(E.encoder.A);
      E.encoder.encode(m, ps, { kind: 'prep', actions: 0, freezes: 0, lp0: seat?.lp0, humans: slot.humans, rules: E.rules, cache: {} }, ob, mk);
      const oi = [], ov = []; for (let q = 0; q < ob.length; q++) if (ob[q] !== 0) { oi.push(q); ov.push(+ob[q].toFixed(5)); }
      obsRec = { D: ob.length, i: oi, v: ov };
      if (net) { net.forward(ob, mk, { value: false }); obsRec.c = Array.from(net.c, (x) => +x.toFixed(4)); }
    }
    const C = layoutCandidatesForPicker(m, ps, KL); if (!C) { S.skipped++; return; }
    const chosen = C.chosen, rng = rngOf(`x:${slot.seed}:${m.round}:${ps.playerId}`);
    const cands = C.cands.map((plan, j) => ({ plan, src: j === 0 ? 'cur' : 'leg', s: [] })), seen = new Set(cands.map((c) => sigOf(c.plan)));
    let mi = -1;
    if (XM && XSAMPLES > 0) { // expert iteration: the generator's sampled layouts join the candidates (src 'gen')
      const Xg = { ...X, fp: Float32Array.from(Float16Array.from(X.fp)), ft: Float32Array.from(Float16Array.from(X.ft)), fe: Float32Array.from(Float16Array.from(X.fe)), fg: Float32Array.from(Float16Array.from(X.fg)) };
      for (let g = 0; g < XSAMPLES; g++) {
        const asg = XM.sample(Xg, XTEMP, rng), q = new LayoutPlan();
        X.uids.forEach((u, i) => { const [t, d] = asg[i]; if (t < 0) return; q.set(u, `${9 + Math.floor(t / 7)},${3 + (t % 7)}`); q.dirs.set(u, ['UP', 'RIGHT', 'DOWN', 'LEFT'][d]); });
        const sg = sigOf(q); if (!seen.has(sg)) { seen.add(sg); cands.push({ plan: q, src: 'gen', s: [] }); }
      }
    }
    if (XM) {
      const Xr = { ...X, fp: Float32Array.from(Float16Array.from(X.fp)), ft: Float32Array.from(Float16Array.from(X.ft)), fe: Float32Array.from(Float16Array.from(X.fe)), fg: Float32Array.from(Float16Array.from(X.fg)) };
      const asg = XM.decode(Xr, XMARGIN), q = new LayoutPlan();
      X.uids.forEach((u, i) => { const [t, d] = asg[i]; if (t < 0) return; q.set(u, `${9 + Math.floor(t / 7)},${3 + (t % 7)}`); q.dirs.set(u, ['UP', 'RIGHT', 'DOWN', 'LEFT'][d]); });
      const sg = sigOf(q), at = cands.findIndex((c) => sigOf(c.plan) === sg);
      if (at >= 0) mi = at; else { seen.add(sg); cands.push({ plan: q, src: 'model', s: [] }); mi = cands.length - 1; }
    }
    for (let g = 0; cands.length < K && g < K * 30; g++) {
      const q = xPerturbForTraining(m, ps, chosen, cands[Math.floor(rng() * cands.length)].plan, rng), s = sigOf(q);
      if (!seen.has(s)) { seen.add(s); cands.push({ plan: q, src: 'pert', s: [] }); }
    }
    let nreh = 0;
    const run = (list, seedIx) => { const r = xRehearseForTraining(m, ps, chosen, list.map((c) => c.plan), [seedIx], SALT ? `${slot.seed}` : '') || []; nreh += list.length; list.forEach((c, j) => { c.s[seedIx] = r[j] ? r[j][0] : null; }); };
    run(cands, 0);
    if (!cands[0].s[0]) { S.skipped++; return; }
    const c00 = cands[0].s[0];
    let best = 0; cands.forEach((c, j) => { if (keyOf(boss, c.s[0], c00) > keyOf(boss, cands[best].s[0], c00)) best = j; });
    // hill-climb: perturbations of the incumbent (seed 0), first improvement
    for (let it = 0, g = 0; it < LS && g < LS * 20 && performance.now() - t0 < MAXMS; g++) {
      const q = xPerturbForTraining(m, ps, chosen, cands[best].plan, rng), s = sigOf(q);
      if (seen.has(s)) continue; seen.add(s); it++;
      const c = { plan: q, src: 'ls', s: [] }; cands.push(c); run([c], 0);
      if (keyOf(boss, c.s[0], c00) > keyOf(boss, cands[best].s[0], c00)) { best = cands.length - 1; S.lsImproved++; }
    }
    // validation on more fixed seeds: the current layout, the seed-0 best and the next TOP-1
    const rank = cands.map((c, j) => j).filter((j) => cands[j].s[0]).sort((a, b) => keyOf(boss, cands[b].s[0], c00) - keyOf(boss, cands[a].s[0], c00));
    const val = [...new Set([0, ...(mi >= 0 ? [mi] : []), ...rank.slice(0, TOP)])];
    for (let s = 1; s < SEEDS; s++) run(val.map((j) => cands[j]), s);
    const mc0 = () => mean(cands[0].s.slice(0, SEEDS));
    let pick = 0; const mk = (j) => keyOf(boss, mean(cands[j].s.slice(0, SEEDS)), mc0());
    for (const j of val) if (mk(j) > mk(pick)) pick = j;
    if (pick !== rank[0]) S.valFlip++;
    const pick0 = pick;
    // simplification (--simplify 1): revert each piece the pick moved back to its current tile / facing (swapping the
    // occupant into its place when legal) while the mean key over the SEEDS seeds does not drop — minimal-edit labels
    if (SIMPLIFY && pick !== 0) {
      const base = cands[0].plan, legalOf = new Map(P.pieces.map((q) => [q.uid, new Set(q.legal)]));
      for (const u of [...base.keys()]) {
        const cp = cands[pick].plan, ku = cp.get(u), k0 = base.get(u), d0 = base.dirs.get(u) ?? 'RIGHT';
        if (ku === k0 && (cp.dirs.get(u) ?? 'RIGHT') === d0) continue;
        const q = new LayoutPlan([...cp.entries()]); for (const [x, d] of cp.dirs) q.dirs.set(x, d);
        const occ = [...q.entries()].find(([x, k]) => x !== u && k === k0);
        if (occ) { if (!legalOf.get(occ[0])?.has(ku)) continue; q.set(occ[0], ku); q.dirs.set(occ[0], cp.dirs.get(u) ?? 'RIGHT'); }
        q.set(u, k0); q.dirs.set(u, d0);
        const sg = sigOf(q); if (seen.has(sg)) continue; seen.add(sg);
        const c = { plan: q, src: 'simp', s: [] }; cands.push(c);
        for (let s = 0; s < SEEDS; s++) run([c], s);
        if (keyOf(boss, mean(c.s), mc0()) >= keyOf(boss, mean(cands[pick].s), mc0()) - 1e-9) { pick = cands.length - 1; S.simplified++; }
      }
    }
    const m0 = mean(cands[0].s), mp = mean(cands[pick].s);
    S.decisions++; S[boss ? 'boss' : 'normal']++; S.rehearsals += nreh;
    if (pick !== 0) { S.changed++; S.dLayers += mp.layers - m0.layers; S.dLeaks += mp.leaks - m0.leaks; S.dBoss += mp.boss - m0.boss; }
    // record
    const [off, len] = writeBin([X.fp, X.ft, X.fe, X.fg, X.ids, X.legal, X.cur]);
    const rec = { salt: SALT ? 1 : 0, nkey: NKEY, seed: slot.seed, cfg: cfg || slot.cfg?.name, round: m.round, pid: ps.playerId, recipe: slot.recipeSeat?.get(ps.playerId) || 0, boss, n: X.n, v: X.version, vh: X.vocabHash,
      dims: X.dims, bin: [off, len], pick, best0: rank[0], val,
      pick0: pick0, cands: cands.map((c) => ({ a: xAssignOf(X, c.plan), src: c.src, s: c.s.map(pack), ...(CANDF ? { f: Array.from(xCandFeatures(m, ps, chosen, c.plan, cands[0].plan), (x) => +x.toFixed(4)) } : {}) })),
      ...(obsRec ? { obs: obsRec } : {}),
      bt: Object.fromEntries(Object.entries(ps.bonds || {}).filter(([, v]) => v && v.active).map(([b, v]) => [b, [v.tier || 0, v.layers || 0, m.gd.bond(b)?.isCore ? 1 : 0]])),
      mi, lg0: cands[0].s[0]?.lg, lgp: cands[pick].s[0]?.lg, ms: Math.round(performance.now() - t0), nreh };
    appendFileSync(OUT + '.jsonl', JSON.stringify(rec) + '\n');
    let play = APPLY === 'best' ? pick : 0;
    if (mi >= 0) {
      const km = keyOf(boss, mean(cands[mi].s), m0), k0 = keyOf(boss, m0, m0), kp = keyOf(boss, mp, m0);
      S.model++; if (mi === 0) S.modelIsCur++; if (km > k0 + 1e-9) S.modelBetter++; else if (km < k0 - 1e-9) S.modelWorse++;
      if (!boss) { S.modelCur += km - k0; S.modelPick += kp - km; }
      if (PLAY === 'model' || (PLAY.startsWith('mix:') && rng() >= Number(PLAY.slice(4)))) play = mi;
    }
    if (play !== 0) { applyPlanForPicker(m, ps, chosen, cands[play].plan); S.played++; }
    const dt = performance.now() - t0; S.ms += dt; S.msMax = Math.max(S.msMax, dt);
  };

  return { hook, S, close() { closeSync(binFd); } };
}
