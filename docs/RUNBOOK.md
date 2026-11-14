# Runbook

All commands run inside the game checkout that `setup.sh` prepares, unless they start with `scripts/` or `./`. Paths
such as `runs/`, `work/` and `data/` are your own working directories. Keep them outside the checkout, or let the
repository's `.gitignore` patterns cover them.

## 1. Requirements

- **Node ≥ 24** (tested: 24.9 for training, 25.9 for the release checks). The environment needs no npm packages; `npm
  ci` (`./setup.sh --npm`) is only for the game's own tests and for `chrome_bench.mjs` (puppeteer-core plus a local
  Chrome).
- **Python 3.12** with `requirements.txt` (PyTorch, numpy, tensorboard). The learner uses one GPU (`--device cuda`).
  Evaluation (`--device cpu` works), export, parity and surgery run on CPU.
- **CPU and memory.** The environments are CPU-bound Node processes, one core each. A final-recipe run uses 50 of them
  plus the learner and needs about 35 GB of RAM. §8 has the measured numbers per stage. Run heavy jobs through your
  machine's job queue, never as an unbounded load on a shared host.
- **Inference at deploy** needs no GPU: the policy and both pickers are plain JS (§8.5).

## 2. Set up

```sh
./setup.sh [game dir] [--reset] [--npm]
```

1. Clones `Stardust-minus/Stronghold-Protocol`, or reuses the clone at `[game dir]`.
2. Checks out `9d6a1b4e` detached.
3. Replaces `tools/rl` with `rl/`.
4. Applies `patches/*.patch` (`git apply --check` first).
5. `--reset` discards local changes in an existing checkout first.
6. `GAME_COMMIT=<sha>` tries another game commit. The patches may then need a refresh, and the game data, and so the
   dataId, may change.

## 3. Checks (seconds to minutes each)

```sh
node tools/rl/vcheck.mjs                         # dataIds: v2 act2autochess-b0c30136ef88, v3 act2autochess-75d38e72a6cf
node tools/rl/smoke.mjs --matches 4 --configs coop4:HARD,solo:NORMAL --humans 1   # random policy; teacher == legacy
node tools/rl/lpguard.mjs ./env.mjs 4            # the lp0 / own-LP regression check (must report no guard hits)
node tools/rl/snaptest.mjs --matches 4           # snapshot starts finish exactly like the originals
node tools/rl/bandtest.mjs --matches 16          # band masks, conflicts, fallbacks
node tools/rl/r2check-cap.mjs solo:ABYSS,coop4ai:ABYSS 6   # every R2 term against the engine's own counters
node tools/rl/pathdiff.mjs                       # policy-picked vs forced band: observations must not differ
node tools/rl/infoguard.mjs on                   # the visible-only information set: no hidden reads (exit 1 on a hit)
node tools/rl/infoguard.mjs off                  # positive control: hits expected ("ok": true means some were found)
python tools/rl/final/smoke.py <work>/runs/<run>/<stage>/train.jsonl   # first updates of a stage: env errors 0, KL sane
```

Run `lpguard`, `snaptest`, `infoguard on` and `vcheck` after any change to `env.mjs` or `spec.mjs`. A changed dataId
means old checkpoints no longer match the observation layout. After any change to the model, the export or the JS
forwards, run the release parity check (§6).

## 4. The pipeline from scratch

### 4.1 The final pipeline

[rl/final/README.md](../rl/final/README.md) has the work-dir layout, the exact arguments, the seeds and the md5s.
In short:

```sh
./setup.sh work/code-A6          # the training tree (pinned game commit + rl/ + patches/)
./setup.sh work/code-A7          # the eval tree (the same code)
cp -R rl/final work/FINAL
# each tree needs its manifest: md5sum of every file under data packs plugins schemas server shared tools + package.json
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh data    # record the BC data (112 single-core workers)
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh bc      # MLP and v5 BC
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh rl      # NM -> NMA -> co-op stage, both architectures
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh status
```

```text
data (national-route teacher) → BC → NM (solo:HARD, 5 M) → NMA (auto1, 60 M) → [extension] → [co-op stage, 10 M]
  → milestone evals (settings a / b) → pick → held-out confirmation → release export (§6)
```

- The launchers check the tree manifests (`MAN_MD5` / `MAN_A6` / `MAN_A7` override them for a rebuilt tree) and refuse a
  mismatch.
- Every stage is restart-safe: a restart resumes from the newest complete checkpoint with a fresh Adam and the resume
  warm-up. Each stage dir holds `meta.json` (full args, tree manifest md5, init and KL-ref md5), `train.jsonl`,
  `eval-*.json`, `model-*.pt` (every ~1 M) and `DONE`.
- Give each run and each resume its own seed offset (the launchers do), so no two runs replay the same matches.
- BC is not bit-deterministic for the MLP (an equivalent BC, not the same bytes); the v5 BC runs under torch's
  deterministic algorithms (`v5/detrun.py`) and reproduces exactly.

### 4.2 The earlier hybrid line (Part B of RESULTS)

The lineage of the 11-12 / 11-13 models (details and flags: [EXPERIMENTS.md](EXPERIMENTS.md) Phases 1–3,
`docs/curves/hyperparams.csv`):

```text
record (teacher, obs v2) → bc1 → ppo1 → ppo2 @ 15 M ──surgery──→ obs v3
   → ppo3h (band choice) → 6.0 M + offline-fitted band head
   → m3 (R2 + layer, mix v2) @ 10 M → m3s (mix v3) / S (mix v3s) → SK (KL 0.003) / SC (v3sc) / SO1 (solo1)
```

```sh
# 1. BC data from the rule-enforcing legacy teacher (version 2 observations; bc1 used 4.8 M records)
node tools/rl/record.mjs --out data --tag s1 --matches 400 --seed 100000 --envs 8 --obs 2
python tools/rl/py/train_bc.py --data data/<dataId> --out runs/bc

# 2. ppo1: staged curriculum NORMAL → HARD → ABYSS, KL to BC decaying over 5 M
python tools/rl/py/train_ppo.py --init runs/bc/bc.pt --out runs/ppo1 --procs 24 --envs 16 --max-steps 3e7 \
  --bc-kl 0.1 --bc-kl-steps 5e6 --baseline runs/base/teacher.json

# 3. ppo2: fixed mix v1, Final Assault reward, constant KL 0.01 to ppo1, bond shaping, boss rehearsal 3
python tools/rl/py/train_ppo.py --init runs/ppo1/model-final.pt --out runs/ppo2 --curriculum fixed --mix v1 \
  --reward '{"round":0.05,"fa":1.5}' --bc-kl 0.01 --bc-kl-steps 0 --rehearsal 0 --boss-rehearsal 3 \
  --shape-tier 0.06 --shape-stack 0.08 --adv-norm global --vclip 0 --vf-warmup 5 --procs 60 --envs 16 --max-steps 5e7

# 4. grow to observation v3 (new blocks zero-initialised; outputs checked identical)
python tools/rl/py/surgery.py --ckpt runs/ppo2/model-<15M>.pt --out runs/v3/ppo2-v3.pt

# 5. band head: strength table on paired seeds, band-decision observations, offline fit, transplant
python tools/rl/py/band_table.py --ckpt runs/ppo2/model-<15M>.pt --out runs/band0 --matches 48
node tools/rl/bandobs.mjs runs/v3/bandobs 150
python tools/rl/py/bandfit.py --ckpt runs/v3/ppo2-v3.pt --obs runs/v3/bandobs --table runs/band0/band_table.json --out runs/v3/ppo2-v3-ref.pt
python tools/rl/py/bandxplant.py --dst <checkpoint> --head runs/v3/ppo2-v3-ref.pt --out <checkpoint>-bh.pt \
  --obs runs/v3/bandobs --table runs/band0/band_table.json

# 6. the hybrid line (restart-safe; prints its usage without arguments)
MIX=v2  scripts/train.sh runs/m3  <init>              runs/v3/ppo2-v3-ref.pt <init steps>
MIX=v3s scripts/train.sh runs/S   runs/m3/model-<10M>.pt runs/v3/ppo2-v3-ref.pt <10M steps>
MIX=v3s scripts/train.sh runs/SK  runs/S/model-<20M>.pt  runs/v3/ppo2-v3-ref.pt <20M steps> --bc-kl 0.003
```

Our m3 started from ppo3h @ 6.0 M, which was trained under the `_seat` bug ([FINDINGS.md](FINDINGS.md) §1), with its
band head replaced by the fitted one. From scratch, start m3 from `ppo2-v3-ref.pt` and expect a different curve.

### 4.3 Monitoring

`tensorboard --logdir runs`. Each run writes `<run>/tb`, `<run>/train.jsonl` (one line per update: steps, sps, rolling
win / fa per difficulty, losses, KL to the anchor, entropy, environment errors / rejections / violations, the auto1
mix) and `<run>/eval-<steps>.json` (the in-run evaluations). Alert on:

- a Traceback / OOM / Killed line in the logs, even if the process exited 0;
- environment errors > 0;
- rule violations above `--viol-rate`;
- a stalled step counter;
- a launcher restart loop (repeated `rc` lines in `launcher.log`).

## 5. Evaluate

The headline protocol of the final models (480 matches in 28 chunks = 476 played; `rl/final/evq2.sh` runs exactly this):

```sh
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
SH=(--greedy --bond-log --rehearsal 0 --boss-rehearsal 0 --band-policy 0 --nat-plan "$NM" --matches 480 --chunks 28 --seed 7000000)
CO=(--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}')
PK=(--npicker-head <rel>/pickers/c9/c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool
    --xplace "$(python tools/rl/release/xplace_arg.py <rel>)")          # <rel> = a release dir (§6)
python tools/rl/py/eval.py --ckpt <model.pt> --configs solo:ABYSS    "${SH[@]}"                    --out a-solo.json   # setting a
python tools/rl/py/eval.py --ckpt <model.pt> --configs coop4h1:ABYSS "${SH[@]}" "${CO[@]}"         --out a-coop.json
python tools/rl/py/eval.py --ckpt <model.pt> --configs solo:ABYSS    "${SH[@]}" "${PK[@]}"         --out b-solo.json   # setting b (deploy)
python tools/rl/py/eval.py --ckpt <model.pt> --configs coop4h1:ABYSS "${SH[@]}" "${CO[@]}" "${PK[@]}" --out b-coop.json
python tools/rl/py/stackrep.py b-solo.json b-coop.json            # stacking metrics
python tools/rl/py/coopstats.py b-coop.json                       # collision / replans
python tools/rl/py/compare.py <dir> <stem A>:<stem B>             # per-seed paired comparison
```

- The held-out set is the same with `--seed 200000000` (`rl/final/heldout.sh`).
- Compare checkpoints only on the same `--seed / --matches / --chunks`, the same setting and the same band policy.
- With the pickers, do not pass `--picker-head` / `--picker-trunk` (the older boss picker would re-pick over xboss-v3's
  board). Use `eval.py`, not train_ppo's in-run evals, for setting b: train_ppo has no `--npicker-pool`.
- `rl/final/records_final.py` turns a set of eval outputs into `records/final/` ([its README](../records/final/README.md)
  defines every column).
- The earlier hybrid line's `scripts/eval.sh solo480 | deploy288` (b3 / b0, `--visible-only`, band chosen by the
  policy) stays for Part B of RESULTS.

## 6. Release export and JS parity

```sh
python tools/rl/release/export_release.py --ckpt <model.pt> --c9 <dir with c9-js.json/.bin> \
  --xboss <dir with dw.json, g3b.json, xplace.json> --out <rel> --name <name> --repo-commit <sha> \
  --matches 8 --cpu <one core>
node tools/rl/release/play.mjs <rel> --configs solo:ABYSS,coop4h1:ABYSS --matches 8      # play with the release in JS
```

`export_release.py`:

1. exports the policy for its JS forward (`py/export.py` → `infer2.mjs` for the MLP, `py/export5.py` → `infer5.mjs` for
   v5; the critic is not exported) and copies the two pickers;
2. writes `manifest.json` (env settings, files with md5 and bytes, repo and game commits, source md5s, the md5s of the
   JS code files) and `MD5SUMS`;
3. plays `--matches` matches per config in JS (`play.mjs`: the JS policy, greedy, with c9 and xboss-v3, headline
   settings) pinned to one core, records every policy decision, and re-scores the decisions in torch;
4. writes `parity.json` and fails unless max |JS − torch| logit over legal actions ≤ 1e-3, greedy agreement ≥ 0.999, and
   no picker built a battle.

The release asset layout and how to load it: [RELEASE.md](../RELEASE.md). Older parity tools stay for diagnostics:
`py/parity.py` (infer.mjs vs torch on data shards), `parity2.mjs` / `parity3.mjs` (JS variants agree; `infer3.mjs` does
not support plan-slot exports), `parity5.mjs` + `py/parity5_dump.py` (v5), `py/p5parity.py` + `p5parity.mjs` (picker
heads), `chrome_bench.mjs` (ms per decision under Chrome CPU throttling).

## 7. Updating the server patches

Four patches, all against `9d6a1b4e`:

- `0001-seat-plan-hooks.patch`: `server/match/seatPlan.js` and gated hooks in `choices.js`, `effectsMeta.js`,
  `player/acquire.js` and `player/economy.js`;
- `0002-visible-only-hooks.patch`: the strict-preview path of `bot.js` `fieldModel`, gated on `m._visibleOnly`;
- `0003-placement-picker-oracle-hooks.patch`: training-only `bot.js` exports for the placement pickers and oracles;
- `0004-label-xplace-proxy-hooks.patch`: the candidate pool, label seeds, placement exports, proxy buy noise / trade /
  spendDown and the Ioleta force flag.

All hooks are gated on fields the server never sets. To refresh them against a newer game commit:

1. Check out that commit.
2. Apply with `git apply --3way`.
3. Resolve any conflicts.
4. Regenerate each patch from its own files (`git diff -- <files> > patches/<name>.patch`; `git add -N` new files first).
5. Re-run `smoke4.mjs`, `tcheck.mjs teacher coop4:HARD 1 24`, `infoguard.mjs strict`, a seeded flags-off identity run
   against the old tree, and the game's own tests (`npm test`). The hooks must stay inert for legacy and human seats.

## 8. Hardware

Measured on the final runs unless marked *estimated*. Hosts: an 8× H200 host with 224 hardware threads (Xeon Platinum
8581C; seed 1, extensions, evals) and an 8× RTX 4090 host with 128 threads (Xeon Gold 6548Y+; seed 2, one 24 GB GPU and
96 threads used). Each run had its own CPU set; several runs shared a host.

### 8.1 Minimum and what we used

| | minimum (one run at a time) | what we used |
|---|---|---|
| GPU | one 24 GB GPU (an RTX 4090 trained the MLP and the v5 seed-2 runs side by side, 9.5 GB together) | one H200 per seed-1 run, one RTX 4090 for both seed-2 runs |
| CPU threads | 32 (the seed-2 MLP run: 32 threads, 3.3 k steps/s) | 50–64 per run, plus 30–60 per eval lane |
| RAM | about 40 GB per run (*estimated*: the measured 33–37 GB plus headroom) | 33–37 GB per run |
| wall time, MLP | NM + NMA to 60 M about 5.1 h, co-op stage 1.5 h | the same, four runs in parallel |
| wall time, v5 | about 6.1 h to 60 M | the same |

### 8.2 Training, per stage (50 environment processes × 64 matches each)

| stage | arch | CPU threads | env steps/s | wall time | notes |
|---|---|---|---|---|---|
| data recording | (teacher) | 112 single-core workers | | 10.5 min | 1.61 M decisions, 7.7 GB on disk |
| BC, 20 epochs | MLP | 1 GPU | | 110 s | H200 |
| BC, 20 epochs | v5 | 1 GPU | | 13 min | H200, deterministic algorithms |
| NM, 0 → 5 M | MLP | 56 (H200 host) / 32 (4090 host) | 2.7 k / 3.3 k | 0.51 h / 0.43 h | update 2.6–2.8 s |
| V5NM, 0 → 5 M | v5 | 56 / 64 | 2.3 k / 2.9 k | 0.60 h / 0.47 h | update 12–19 s |
| NMA, 5 → 60 M | MLP | 56 / 32 | 2.8 k / 3.3 k | 5.4 h / 4.7 h | |
| V5NMA, 5 → 60 M | v5 | 56 / 64 | 2.3 k / 2.7 k | 6.7 h / 5.6 h | the H200 run was learner-bound (20 s updates) |
| extension, per 10 M | MLP | 50 | 2.6 k | 1.06 h | |
| extension, per 10 M | v5 | 50 | 2.2 k | 1.3 h | |
| co-op stage, 10 M | MLP | 32 (4090 host) / 70 (CB arm) | 1.9 k / 2.3 k | 1.46 h / 1.23 h | co-op matches cost more per step |
| co-op stage, 10 M | v5 | 64 / 70 | 2.0 k / 2.1–2.2 k | about 1.4 h / 1.3 h | |

- **GPU memory** (peak of the learner process, including in-run evals): MLP 4.3 GB without `--micro`, 2.9 GB with
  `--micro 2048`, up to 5.4 GB seen during an extension; v5 4.5–4.6 GB with `--micro 1024`; each in-run eval adds about
  1.2 GB. The runs capped memory with `--mem-frac` / `SPRL_CUDA_MEM_FRAC`; a cap below the peak ends in an OOM restart
  loop.
- **Host RAM** (resident set of a live run): the learner 2.1–3.9 GB plus 50 Node workers of 0.6–0.7 GB each, 33–37 GB in
  total.
- Steps/s depends on hyperthreads: 32 physical cores (the 4090 host) beat 28 cores × 2 threads (the H200 host).

### 8.3 Evaluation

One checkpoint at 480 seeds (476 played), 28 Node processes plus the policy (one GPU share, or CPU):

| setting | solo:ABYSS | coop4h1:ABYSS |
|---|---|---|
| a (no pickers) | about 70 s | about 220 s |
| b (c9 + xboss-v3) | about 340 s | about 15 min |

Medians over 10 checkpoints on 32–58-thread lanes. A full milestone (a and b, solo and co-op) takes about 27 min. On a
16-thread CPU-only box, setting a took 61–71 s solo and 4–5 min co-op.

### 8.4 Picker training

| step | cost |
|---|---|
| normal-round labels (c9: 41 k pool decisions, every candidate rehearsed on 2 seeds) | 28 single-thread jobs, about 1.4 h (33 k decisions / h) |
| normal-round labels (Sall: about 300 k K8 decisions) | about 3 k decisions / min across the label jobs |
| c9 fit | 6 epochs, under 10 min on one H200 (the first epoch 58 s), 722 k parameters |
| boss-round search labels | one thread per job, about 10–20 s per boss decision (hill-climb), 9–14 k decisions / h over 24–30 jobs including normal rounds |
| teacher bake-off | 61 s per decision for 3 teachers × 2 budgets |
| Dw fit | 12 epochs, about 9 min on one GPU (30 k records) |

### 8.5 Inference at deploy

CPU only; no GPU needed. Measured by the release dry run (`play.mjs`) on one pinned core of a Xeon Gold 5420+, Node
25.9, 8 + 8 matches (MLP) and 4 + 4 (v5):

| model | multiply-accumulates | ms per decision | decisions per solo match | files (bytes) | resident memory |
|---|---|---|---|---|---|
| policy, MLP (`infer2.mjs`) | 1.71 M | 1.7–1.9 | about 105 | policy.bin 6.2 M + policy.json 8 k | about 20 MB |
| policy, v5 (`infer5.mjs`) | 25.0 M | 23–24 | about 105 | policy.bin 4.7 M | about 9 MB |
| normal-round picker c9 | about 21 candidates × a v5 tower | 750–800 | 13 | c9-js.bin 2.9 M + .json 12 k | 9–14 MB |
| boss-round placement xboss-v3 | 25–54 candidates × the QNet | 150–240 (max 385) | 1.75 | dw.json 2.4 M + g3b.json 1.5 M | 12–14 MB |

- Per solo match that is about 10.5 s of model time on one core with the MLP (the c9 picker dominates), about 13 s with
  v5. The whole Node process (engine, environment and models) peaked at 460–540 MB.
- Neither picker is optimised: both re-encode every candidate from scratch. Deploy cost was not a constraint for the
  final choice (it can move to WebGPU or be optimised later).
- Earlier measurements on loaded hosts: MLP 4.2–5.4 ms, v5 84 ms, c9 about 1.8 s, xboss-v3 0.35–1.0 s per decision.
