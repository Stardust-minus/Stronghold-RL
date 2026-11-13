# Runbook

All commands run inside the game checkout that `setup.sh` prepares, unless they start with `scripts/` or `./`. Paths
such as `runs/` and `data/` are your own working directories. Keep them outside the checkout, or let the
repository's `.gitignore` patterns cover them.

## 1. Requirements

- **Node ≥ 24** (tested: 24.9). The environment needs no npm packages; `npm ci` (`./setup.sh --npm`) is only for the
  game's own tests and for `chrome_bench.mjs` (puppeteer-core plus a local Chrome).
- **Python 3.12** with `requirements.txt` (PyTorch, numpy, tensorboard). The learner uses one GPU (`--device cuda`).
  Evaluation, export and surgery run on CPU.
- **CPU.** One environment process runs 64 matches and takes about one core. The current recipe uses 25 processes
  and collects about 900 decisions/s. Each process peaks at about 330–420 MiB. Run heavy jobs through your machine's
  job queue, never as an unbounded load on a shared host.

## 2. Set up

```sh
./setup.sh [game dir] [--reset] [--npm]
```

1. Clones `Stardust-minus/Stronghold-Protocol`, or reuses the clone at `[game dir]`.
2. Checks out `9d6a1b4e` detached.
3. Replaces `tools/rl` with `rl/`.
4. Applies `patches/*.patch` (`git apply --check` first).
5. `--reset` discards local changes in an existing checkout first.
6. `GAME_COMMIT=<sha>` tries another game commit. The patch may then need a refresh, and the game data, and so the
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
node tools/rl/infoguard.mjs on                   # deploy information set: no hidden reads by an RL seat (exit 1 on a hit)
node tools/rl/infoguard.mjs strict               # + strict preview (A2 / A4 / B1)
node tools/rl/infoguard.mjs off                  # positive control: hits expected ("ok": true means some were found)
```

Run `lpguard`, `snaptest`, `infoguard on` and `vcheck` after any change to `env.mjs` or `spec.mjs`. A changed dataId means old
checkpoints no longer match the observation layout.

## 4. The pipeline from scratch

No checkpoints are published, so a reproduction starts from behaviour cloning. The lineage of the current models
(details and exact flags: [EXPERIMENTS.md](EXPERIMENTS.md), `docs/curves/hyperparams.csv`):

```text
record (teacher, obs v2) → bc1 → ppo1 → ppo2 @ 15 M ──surgery──→ obs v3
   → ppo3h (band choice) → 6.0 M + offline-fitted band head
   → m3 (R2 + layer, mix v2) @ 10 M → m3s (mix v3) / S (mix v3s) → SK (KL 0.003) / SC (v3sc) / SO1 (solo1)
```

The KL anchor of every arm from m3 on is ppo2 @ 15 M, grown to obs v3, with the fitted band head.

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

# 6. the hybrid line with the current recipe (restart-safe; prints its usage without arguments)
MIX=v2  scripts/train.sh runs/m3  <init>              runs/v3/ppo2-v3-ref.pt <init steps>
MIX=v3s scripts/train.sh runs/S   runs/m3/model-<10M>.pt runs/v3/ppo2-v3-ref.pt <10M steps>
MIX=v3s scripts/train.sh runs/SK  runs/S/model-<20M>.pt  runs/v3/ppo2-v3-ref.pt <20M steps> --bc-kl 0.003
```

**Deploy settings.** The arms of [RESULTS.md](RESULTS.md) trained at b3 without the visible-only switch, and
`scripts/train.sh` reproduces them by default. A run for deployment trains as it will play:
`BOSS_REHEARSAL=0 VISIBLE_ONLY=1 scripts/train.sh …` (`--boss-rehearsal 0 --visible-only`). `--strict-preview` exists
for experiments, but it is not part of the deploy rule ([DESIGN.md](DESIGN.md) §3).

Our m3 started from ppo3h @ 6.0 M, which was trained under the `_seat` bug ([FINDINGS.md](FINDINGS.md) §1), with its
band head replaced by the fitted one. From scratch, start m3 from `ppo2-v3-ref.pt`, i.e. ppo2 grown to v3 with the
fitted head, and expect a different curve. `scripts/train.sh` sets `--seed-offset`. Give each arm and each resume
its own offset, so that no two runs replay the same matches.

**Monitoring.** `tensorboard --logdir runs`. Each run writes `<run>/tb`, `<run>/train.jsonl` (one line per update:
steps, sps, rolling win / fa per difficulty, losses, KL to the anchor, entropy, environment errors / rejections /
violations) and `<run>/eval-<steps>[-b0].json` (the in-run evaluations). Alert on:

- a Traceback / OOM / Killed line in the logs, even if the process exited 0;
- environment errors > 0;
- rule violations above `--viol-rate`;
- a stalled step counter.

## 5. Evaluate

```sh
scripts/eval.sh solo480   runs/SK/model-<steps>.pt runs/eval     # → runs/eval/SK-model-<steps>-solo480-b3-vis.json / -b0-vis.json
scripts/eval.sh deploy288 runs/SK/model-<steps>.pt runs/eval     # solo / coop4l / coop2l, 288 seeds, mates r3, b3
python tools/rl/py/eval.py --teacher --configs solo:ABYSS --matches 480 --rehearsal 0 --boss-rehearsal 3 --out runs/base/teacher.json
python tools/rl/py/eval.py --legacy  --configs solo:ABYSS --matches 480 --rehearsal 3 --boss-rehearsal 3 --out runs/base/live.json
python tools/rl/py/compare.py runs/eval S-model-<steps>-solo480-b0-vis:SK-model-<steps>-solo480-b0-vis   # file stems; per-seed paired (fa: bootstrap CI; win: McNemar)
python tools/rl/py/eval_report.py --www www/eval --name SK-30M --title "SK @ 30 M" --run SK=runs/eval/SK-model-<steps>-solo480-b0-vis.json
```

`scripts/eval.sh` adds `--visible-only` unless `VISIBLE_ONLY=0`. The deploy number is the b0 file. Compare checkpoints
only on the same seeds, the same rehearsal settings and the same information set. Report the 480 / 288-seed numbers,
not the 96-seed in-run ones.

## 6. Export and JS parity

```sh
python tools/rl/py/export.py --ckpt runs/SK/model-<steps>.pt --out runs/export/SK      # SK.json (manifest) + SK.bin (f32)
python tools/rl/py/parity.py --ckpt runs/SK/model-<steps>.pt --data data/<dataId> --n 64   # infer.mjs vs PyTorch (needs record.mjs shards of the same dataId)
node tools/rl/parity2.mjs runs/export/SK 3000      # infer2 == infer on real observations
node tools/rl/parity3.mjs runs/export/SK 2400      # infer3: incremental == full, batch == single (bit-exact)
node tools/rl/chrome_bench.mjs runs/export/SK 1500 <chrome>   # ms per decision at 1× / 4× / 6× CPU throttling
```

Before a model can ship:

- the JS forward must agree with PyTorch (logit differences around 1e-6, identical greedy actions);
- it must stay within the 10 ms per decision budget at 4×;
- `infer3` must stay bit-identical to a full recompute.

## 7. Updating the server patch

Two patches, both against `9d6a1b4e` and independent of each other:

- `patches/0001-seat-plan-hooks.patch` adds `server/match/seatPlan.js` and gated hooks in `choices.js`,
  `effectsMeta.js`, `player/acquire.js` and `player/economy.js`;
- `patches/0002-visible-only-hooks.patch` adds the strict-preview path of `bot.js` `fieldModel`, gated on
  `m._visibleOnly`.

To refresh them against a newer game commit:

1. Check out that commit.
2. Apply with `git apply --3way`.
3. Resolve any conflicts.
4. Regenerate each patch from its own files, e.g. `git add -N server/match/seatPlan.js && git diff -- server/match/seatPlan.js
   server/match/choices.js server/match/effectsMeta.js server/match/player > patches/0001-seat-plan-hooks.patch` and
   `git diff -- server/match/bot.js > patches/0002-visible-only-hooks.patch`.
5. Re-run `smoke4.mjs`, `tcheck.mjs teacher coop4:HARD 1 24`, `infoguard.mjs strict` and the game's own tests
   (`npm test`). The hooks must stay inert for legacy and human seats.
