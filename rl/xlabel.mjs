// tools/rl/xlabel.mjs — xplace (11-13): search labels for the learned layout policy (training only; sims are the teacher).
// Each RL seat (and recipe seat) prep with a fight: candidates = the current (legacy) layout, legacy's generator variants
// (layoutCandidatesForPicker K), random legal perturbations (move / swap / turn) up to --K; each rehearsed once on fixed
// seed 0 (the label seed, never the live one); then --ls hill-climb proposals (perturbations of the incumbent best);
// then the --top best (plus the current layout) re-rehearsed on --seeds - 1 more fixed seeds; the pick = best mean key.
// Key: normal rounds layers + 0.01 kills - --wleak x leaks; boss rounds leaks asc, then leader damage, then kills.
// --apply best: the pick is played (teacher states); none: legacy keeps playing (labels only).
// Output: <out>.jsonl (one record per decision: candidates, their assignments and scores) + <out>.bin (float16 features).
//   node tools/rl/xlabel.mjs --policy <js export base> --config solo:ABYSS --seed0 9100000 --matches 8 --out /path/x
import { SPEnv } from './env.mjs';
import { SPPolicy2 } from './infer2.mjs';
import { readFileSync, appendFileSync, writeFileSync, openSync, writeSync, closeSync, existsSync, statSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createLabelHook } from './xsearch.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const T0 = performance.now();
const cfg = arg('config', 'solo:ABYSS'), OUT = arg('out');
const net = arg('policy') ? SPPolicy2.load(arg('policy')) : null;
const K = Number(arg('K', 16)), KL = Number(arg('KL', 12)), LS = Number(arg('ls', 16)), SEEDS = Number(arg('seeds', 3)), TOP = Number(arg('top', 3));
const SIMPLIFY = arg('simplify', '1') !== '0', WLEAK = Number(arg('wleak', 8)), APPLY = arg('apply', 'best'), ROUNDS = arg('rounds', 'all'), MAXMS = Number(arg('max-ms', 60000));
const RECIPES = JSON.parse(readFileSync(arg('recipes', new URL('./recipes.json', import.meta.url)), 'utf8'));
const recipe = Number(arg('recipe', 0));
// DAgger (--xmodel <xplace model JSON>): the policy's own layout joins the candidates (src 'model', always validated);
// --play best (the pick) | model (the policy's layout: its own state distribution) | mix:<p> (the pick with probability p)
const STORE_OBS = arg('store-obs', '0') === '1', CANDF = arg('cand-feats', '1') === '1';
const PLAY = arg('play', 'best');
const coopAi = cfg.split(':')[0].endsWith('ai');
if (!OUT) throw new Error('--out required');

const LAB = createLabelHook(null, { out: OUT, K, KL, ls: LS, seeds: SEEDS, top: TOP, simplify: SIMPLIFY, wleak: WLEAK, nkey: arg('nkey', 'd'), apply: APPLY,
  rounds: ROUNDS, maxMs: MAXMS, storeObs: STORE_OBS, candFeats: CANDF, xmodel: arg('xmodel'), xmargin: Number(arg('xmargin', 0)), play: PLAY, cfg, net,
  overwrite: arg('overwrite') === '1' });
const S = LAB.S, hook = (m, ps, slot) => LAB.hook(m, ps, slot, env);
const env = new SPEnv({ envs: Number(arg('envs', 2)), configs: [cfg], seed: Number(arg('seed0', 9_100_000)), envSeed: Number(arg('seed0', 9_100_000)),
  maxMatches: Number(arg('matches', 4)), humans: coopAi ? 0 : 'random', rehearsal: 0, bossRehearsal: 0, mateRehearsal: Number(arg('mate-rehearsal', 3)),
  bandPolicy: net ? { share: 1, uniform: false } : undefined, mode: net ? 'policy' : 'policy', check: true, metrics: true, bondLog: true, placeHook: hook,
  ...(recipe ? { recipeSeats: { recipes: RECIPES, seats: { ai_0: recipe } } } : {}) });
let b = env.reset();
while (b.n) {
  const a = new Int32Array(b.n);
  if (net) for (let k = 0; k < b.n; k++) {
    const { logits } = net.forward(b.obs.subarray(k * b.D, (k + 1) * b.D), b.mask.subarray(k * b.A, (k + 1) * b.A), { value: false });
    let bi = 0, bv = -Infinity; for (let j = 0; j < b.A; j++) if (b.mask[k * b.A + j] && logits[j] > bv) { bv = logits[j]; bi = j; }
    a[k] = bi;
  }
  b = env.step(a);
}
env.close(); LAB.close();
const seeds = env.results.map((r) => ({ seed: r.seed, win: r.seats.some((x) => x.win) ? 1 : 0, fa: +(r.faShare || 0).toFixed(4) }));
const out = { config: cfg, recipe, apply: APPLY, K, KL, LS, SEEDS, TOP, WLEAK, NKEY, seconds: +((performance.now() - T0) / 1000).toFixed(1), matches: env.results.length,
  win: seeds.reduce((a, x) => a + x.win, 0) / Math.max(1, seeds.length), stats: S, seeds, errors: env.stats.errors, samples: env.errorSamples.slice(0, 3) };
writeFileSync(OUT + '.summary.json', JSON.stringify(out));
console.log(JSON.stringify({ ...out, seeds: undefined }));
