# Final training pipeline (2026-11-13 / 14)

The scripts in this directory trained and evaluated the final models. They are the scripts as run, with host paths
turned into variables. Copy this directory to `$B/FINAL` and run it from there.

## Work dir layout

Every script needs `B` set to the work dir. They expect:

| path | what |
|---|---|
| `$B/code-A6` | training tree: a game checkout with this repository's `rl/` as `tools/rl` and `patches/` applied, plus `MANIFEST-A6.md5` |
| `$B/code-A7` | eval tree for setting b (the same code; it adds the deploy pickers' eval flags), plus `MANIFEST-A7.md5` |
| `$B/prefix/venv/bin/python`, `$B/prefix/node/bin/node`, `$B/site` | Python 3.12 + PyTorch, Node 24, extra Python packages |
| `$B/N/data-plan5/act2autochess-75d38e72a6cf` | the BC data (or `ROOT/data-plan5/...` after `launch.sh data`) |
| `$B/inbox/P-spatial/npool-c9/c9-js.{json,bin}` | the normal-round picker head c9 |
| `$B/X/release/xboss-v3/xplace.json` | the boss-round placement model xboss-v3 |
| `$B/N/ev11/NMA-30000000*` | the NMA 30M reference checkpoint and its 480-seed evals (setting a) |

Build a tree with `./setup.sh <dir>` (the pinned game commit `9d6a1b4e`). A manifest is `md5sum` over every file under
`data packs plugins schemas server shared tools` and `package.json`, sorted by path. The launchers check the
manifest's md5. Set `MAN_MD5`, `MAN_A6` and `MAN_A7` to override the frozen values for a rebuilt tree.

The frozen trees were code-A6 (manifest md5 `2ed55409`) and code-A7 (`cfe0dda0`). They were built on game commit
`d40e3061`, the parent of `8594b517` (shadow verification: result hashes, battle CPU by field kind). The pinned commit
`9d6a1b4e` is that commit plus the `tools/rl` commit. A rebuilt tree therefore differs from the frozen manifests in
two ways. The server and shared files touched by `8594b517` differ. In `tools/rl` the differences are comment edits
and extra tools that nothing imports. `records/provenance.csv` maps each file.

Seeded random-action matches are byte-identical between the rebuilt tree and the frozen code-A7. That covers the
observations, masks, rewards and match results of 18 setup × config cases. Flags-off settings, the NM natPlan and the
teammate proxies (legacy and the CB pool) are among them.

## Stages

1. **Data**: `ROOT=<dir> bash launch.sh data` re-records data-plan5 on the earlier tree code-N5 (manifest `ec17d896`).
   That is 112 single-core workers at seeds 90000000 + i×10000, with the national-route bot as teacher. Expect data
   manifest `0d275a93` (md5 of `md5sum *.bin *.meta.json *.terms.jsonl`): 112 shards, 1,613,015 samples. Recording is
   deterministic per seed.
2. **BC**: `ROOT=<dir> bash launch.sh bc`.
   - MLP BC "PB5": `train_bc.py --epochs 20 --seed 1`, best val, CE .354. md5 `eaef4a20`. Not bit-deterministic, so a
     re-run gives an equivalent BC, not these bytes.
   - v5 BC: `v5/detrun.py train_bc5.py`, 20 epochs, torch deterministic algorithms. md5 `81270028`. Two replicas
     gave identical tensors.
   - The v5 member table is `v5/lookahead.json` (`2912c4aa`).
3. **NM** (stage 1): BC init, solo:HARD, `--mix hard1`, natPlan NM, to 5M.
4. **NMA** (stage 2): NM's final checkpoint, `--mix auto1` (the ABYSS curriculum), to 60M.
   `ROOT=<dir> bash launch.sh rl` runs stages 3 and 4 for both architectures.
5. **Extension** (optional, 60M → 90M): `ext/launch.sh`, for runs that were still rising at 60M (rule below).
6. **Co-op stage** (60M → 70M, arm CB): `coop/launch.sh` (`ext/coop-launch.sh` for an extended checkpoint,
   `coop/launch-xa.sh` with extra args `XARGS`).
7. **Evals and pick**: `evq2.sh` (milestone lanes), `gate.sh` (reference reproduction), `finsum.py` (milestone
   table), `finalpick.py` (ranking), `heldout.sh` (held-out confirmation).

`launch.sh rl|coop|stop|status` drives the runs. Every stage is restart-safe: a restart resumes from the newest
checkpoint with a fresh Adam. Each stage dir holds `meta.json` (full args, tree, manifest md5, init and KL-ref md5),
`train.jsonl`, `eval-*.json`, `model-*.pt` (every ~1M) and `DONE`.

## Recipes (exact train_ppo arguments)

`<S1>` / `<S2>` are the seed offsets and `<BC>` is the run's BC.

Shared base:

```
--kl-ref <BC> --seed-offset <S1> --mix hard1 --curriculum fixed --rehearsal 0 --boss-rehearsal 0 --reward-v 2
--r2 {"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002} --shape-uncapped --shape-tier 0.06 --shape-stack 0.3 --band-policy 0
--bc-kl 0.01 --bc-kl-steps 0 --ent 0.003 --lr 1e-4 --gamma 0.997 --lam 0.95 --clip 0.2 --target-kl 0.02 --vf 0.5 --vf-warmup 5
--lr-warmup 20 --vclip 0 --adv-norm global --overlap --legacy-mates 0 --envs 64 --rollout 65536 --minibatch 8192 --epochs 3
--hours 24 --ckpt-every 1e6 --eval-every 1e6 --eval-at 0 --eval-matches 96 --eval-chunks 4 --eval-configs solo:HARD --log-diag --check
--procs <P>
```

natPlan NM = `{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}`.

**MLP** (`mlp/launch.sh`, model.py, 1.55M parameters):

- NM: base + `--lr 3e-4 --nat-plan <NM> --max-steps 5000000`, init PB5.
- NMA: base with `--seed-offset <S2>`, then `--mix auto1 --eval-configs solo:HARD,solo:ABYSS --eval-every 2e6 --nat-plan <NM>
  --lr 3e-4 --max-steps 60000000`, init NM's `model-final.pt` with `--start-steps 5000000 --eval-at ''`.
- Host flags: `--learner-cpus --env-cpus --node --device cuda --mem-frac 0.06`, `OMP_NUM_THREADS=2 SPRL_CUDA_MEM_FRAC=0.06`.

**v5** (`v5/launch.sh`, model5.SPNet5, 2.72M parameters: entity transformer d128 × 2 layers, 4 heads, unit–bond
member bias, separate critic d192 × 1):

- Base with `--lr 1e-4` as is, plus `--amp --micro 1024 --fwd-chunk 2048 --clip-sep --nat-plan <NM>`.
- V5NM: + `--max-steps 5000000`. V5NMA: + `--seed-offset <S2> --mix auto1 --eval-configs solo:HARD,solo:ABYSS --eval-every 2e6
  --max-steps 60000000`, init V5NM's final checkpoint.
- Host flags: `--mem-frac 0.05`, `SPRL_CUDA_MEM_FRAC=0.01`.

**Seeds** (stage 1 / stage 2; co-op stage):

| run | NM / NMA | co-op stage |
|---|---|---|
| FIN-MLP-s1 | 61000000 / 62000000 | 71000000 |
| FIN-V5-s1 | 65000000 / 66000000 | 72000000 |
| FIN-MLP-s2 | 63000000 / 64000000 | 73000000 |
| FIN-V5-s2 | 67000000 / 68000000 | 74000000 |

The seed-1 and seed-2 runs use the same tree, BCs and argument lists. Only `--procs`, the CPU lists and the GPU
memory caps differ per host.

These recipes differ from the 30M NMA run they come from only in the seed offsets, stage 2's `--max-steps`
(30M → 60M), `--hours` (12 → 24, a wall-clock cap) and the tree. `argcmp.py` compares two argument lists with
train_ppo's own parser.

## Extension (60M → 90M)

Decided per run once its 50M and 60M evals are in:

- S1 = paired delta (60M − 50M) of mean(setting b solo, setting b co-op), 476 seeds per cell, SE ½·√(SE_solo² + SE_coop²).
- S2 = rolling ABYSS training win (`rolling.ABYSS.win`, n ≥ 100) meaned over the 55–60M updates, minus the 45–50M mean.
- Extend if S1 ≥ +.02 or S2 ≥ +.02. At most 2 runs are extended: the best qualifying run by 60M setting-b mean, and
  one MLP and one v5 may each take a slot if both qualify.
- Stop early when a 10M segment gains < +.01 on both S1 and S2. Segment versions (11-14): S1 = the paired Δ of the b
  mean against the previous milestone; S2 = the rolling ABYSS win over the segment's last 5M minus the previous 5M,
  measured inside the extension. S1 decides the first segment.

| run | S1 | S2 | 60M b mean | extended |
|---|---|---|---|---|
| FIN-MLP-s1 | −.014 ± .019 | +.007 | .663 | no |
| FIN-MLP-s2 | −.005 ± .018 | +.027 | .672 | yes: FIN-MLPX-s2 (seed offset 64,500,000) |
| FIN-V5-s1 | +.012 ± .019 | +.027 | .696 | qualified; extended later as FIN-V5X-s1 (66,500,000) when a slot freed up |
| FIN-V5-s2 | +.045 ± .019 | +.041 | .708 | yes: FIN-V5X-s2 (68,500,000) |

<!-- pick:BEGIN extension -->
Outcomes (milestone rows in `records/final/milestones.csv`, gates in `coopgate.csv`):

- FIN-MLPX-s2: stopped at 80M (S1 −.041 at 80M). Its best milestone is 70M.
- FIN-V5X-s2: ran to 90M (S1 +.022 at 80M, +.028 at 90M).
- FIN-V5X-s1: ran to 90M (S1 +.052 / +.012 / −.002 at 70 / 80 / 90M; S2 +.022 at 80M and 90M).
- XC, the co-op stage on FIN-MLPX-s2 70M (`ext/coop-launch.sh`): failed its gate, setting b solo −.052 ± .024.
<!-- pick:END extension -->

`ext/launch.sh` builds the argument list as:

```
--init <newest checkpoint> --start-steps <its steps> <the run's stage-2 meta.json "args", in order; --kl-ref -> a local md5-checked copy>
--eval-at '' --seed-offset <stage-2 offset + 500000> --auto-smin 0.85 --max-steps 90000000
+ host flags: --out <run>/EXT --learner-cpus --env-cpus --node --device cuda --mem-frac (MLP .06 / v5 .05)
```

What carries over and what restarts (also in each `EXT/meta.json` under `continuation`):

- Checkpoints hold weights only, so Adam restarts. The recipe's warm-up applies: 5 value-only updates, then a 20-update
  lr ramp.
- `--auto-smin 0.85` pins the auto1 ABYSS share at its 60M value. A plain resume would ramp it from 0 again.
- The +500,000 seed offset continues on unused match seeds. A resume with the same offset would replay stage 2's seeds.
- The rolling window (3000) and the minibatch RNG restart. There is no init eval.

## Co-op stage (arm CB, 60M → 70M)

`coop/coopstage.py <stage-2 meta.json> <checkpoint> <start steps> <seed offset> [--check <CB meta.json>]` prints the
argument list. It takes the run's own stage-2 args in order and appends CB's overrides. The last occurrence wins.

- `--init X --kl-ref X`: the KL anchor is the 60M start.
- `--bc-kl 0.01 --bc-kl-steps 0`, `--start-steps 60M --max-steps 70M`, `--mix mp2`, `--band-policy 0`.
- natPlan NM without coopReplan.
- `--human-proxy {"kinds":["mix:nat:lateranoShip=0.25|nat:kjeragShip=0.25|nat:rand=0.2|legacy=0.15|class=0.1|noisy:50=0.05"],"announce":"own"}`.
- In-run evals solo:ABYSS + coop4h1:ABYSS 96 every 2M (`--eval-at 1e12`).
- `--micro 2048` if the base has no `--micro` (memory only; v5 keeps 1024).

The launcher waits for `SRC/DONE` and copies `SRC/model-final.pt` in as init and anchor. It writes `meta.json` with
the source meta and the coopstage.py md5, then trains in `<run>/COOP`. The frozen coopstage.py had md5 `fc57b219`.
The copy here has comment-only edits (md5 `cb426b2b`, the launchers' `CS_MD5` default) and prints the same arguments.

Gate: each 70M is evaluated against its own 60M (solo:ABYSS 480 + coop4h1 480, settings a and b). Adopt the stage if
co-op rises and solo drops by no more than .02.

Validation before the final runs reached 60M (arms CM2 / CV5 / CV5b: the stage for 10M from their 30M checkpoints, setting b): the MLP gained
+.118 ± .026 on co-op (solo −.040 ± .027); v5 lost co-op at lr 1e-4 (−.027 ± .026) and at 3e-4 (−.048 ± .027,
`coop/launch-xa.sh` with `XARGS='--lr 3e-4'`). So the stage is for the MLP; a v5 run keeps its 60M checkpoint unless its
own 70M gate passes. FIN-MLP-s2's stage passed (b coop +.101 ± .027, b solo −.019 ± .026).
`ext/coop-launch.sh` runs the stage on an extended checkpoint.

## Evals

- Milestones: the first checkpoint at or past 30M, 40M, 50M and 60M. Also 70M for the co-op stage and 70/80/90M for
  extensions.
- Configs: solo:ABYSS and coop4h1:ABYSS, `--matches 480 --chunks 28` (476 played), `--seed 7000000`.
- Headline protocol: greedy, `--rehearsal 0 --boss-rehearsal 0 --bond-log`, the policy's own flags
  `--band-policy 0 --nat-plan <NM>`. Co-op adds `--mate-rehearsal 3 --human-proxy {"kinds":["legacy"],"log":true}`.
- Setting **a**: no pickers (code-A6).
- Setting **b**: the deploy pickers (code-A7):
  - normal rounds: c9 over the candidate pool, `--npicker-head <c9-js.json> --npicker-mode reg --npicker-margin 0.5
    --npicker-pool` (c9-js.json `35c6b24d`, bin `230067ab`);
  - boss rounds: xboss-v3, `--xplace "$(cat xplace.json)"` (`aa5fcc3d`);
  - no `--picker-head` / `--picker-trunk`.
- References, paired on the same seeds: a = NMA 30M without pickers, b = NMA 30M with c9 + xboss-v3. `gate.sh`
  checks per match that both trees reproduce them.
- Every eval reports the stacking metrics next to win: main bond type and tier, natMain and natMainT2 shares, main
  and secondary layers, Σ layers, elites.

## Final pick

- Candidates: the four runs at 60M and 70M (and extension milestones), plus two seed-0 candidates: NMA 30M (the MLP
  recipe to 30M, `73cff28c`) and CB10 (NMA 30M + the CB stage, `476ae0a3`).
- Rule: rank by setting b. solo:ABYSS and coop4h1 must both be ≥ .50. Their mean breaks ties. Stacking metrics are
  reported next to every win.
- Held-out confirmation (winner's curse): the pick uses the milestone seeds, and NMA was selected on them too. So
  the leading candidates are re-run on a fresh set with `--seed 200000000`, setting b, solo:ABYSS 480 + coop4h1 480.
  - Match seed = `--seed` + config index × 1M + chunk × 17 + i. coop4h1:ABYSS has config index 580.
  - The held-out seeds are solo:ABYSS 202,000,000–202,000,475 and coop4h1:ABYSS 780,000,000–780,000,475.
  - No earlier eval base or record seed reaches them. Training env seeds start at 1.01e9.
  - The reported number is the held-out one. The final choice between the held-out candidates was itself made on
    these seeds, so the released model is confirmed on a third set, `--seed 300000000` (below).
- `rl/final/records_final.py` regenerates `records/final/` from the eval outputs; `records/final/README.md` defines the
  columns.
- The chosen checkpoint and its pickers are exported with `rl/release/export_release.py` ([RELEASE.md](../../RELEASE.md)).

<!-- pick:BEGIN pick -->
What ran:

- 11 candidates reached the held-out set: the leaders on the milestone seeds as the runs finished, and CB10.
- From 06:55 the rule was: the current pick stays unless a candidate beats it on held-out, paired (paired mean Δ > 0,
  both ≥ .50). `hopair.py` prints that table with the held-out stacking and the pooled view
  (`records/final/heldout-paired.md`).
- By that rule: FIN-V5X-s1-90M (`1e19b8aa`), held-out .788 / .769, Δ mean vs FIN-MLP-s2-70M +.018 ± .017.
- **Released: FIN-MLP-s2-70M** (`514db10d`), held-out .784 / .737 (the owner's choice: the MLP for its inference cost,
  policy 1.8 vs 23 ms per decision in JS). Lineage metas: `records/meta/FIN-MLP-s2-NM-meta.json`,
  `FIN-MLP-s2-NMA-meta.json`, `FIN-MLP-s2-COOP-meta.json`.
- Confirmation on a fresh seed set, `--seed 300000000`, same protocol: FIN-MLP-s2-70M 0.769 / 0.731, FIN-V5X-s1-90M
  0.790 / 0.744; paired Δ mean +0.017 ± 0.018, the same gap as held-out. Real-player-weighted coop4h1 0.702 ± 0.010 vs
  0.713 ± 0.010. With a 谢拉格 human the MLP leads (0.652 vs 0.619, likely from its co-op stage), with a 拉特兰 human they
  tie (0.645) `confirm.py` prints the tables (`records/final/confirm.md`).
- Export: `rl/release/export_release.py` ([RELEASE.md](../../RELEASE.md)).
<!-- pick:END pick -->
