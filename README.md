# Stronghold-RL

Reinforcement learning for the AI seats of [Stronghold Protocol](https://github.com/Stardust-minus/Stronghold-Protocol),
an unofficial, non-commercial fan remake of the Arknights auto-chess mode *Stronghold Protocol: Alliance* (卫戍协议：盟约).
This repository holds the environment, the PPO trainer, the evaluation tools, the plain-JS inference and the deploy
pickers, plus the documentation of how the policies were tuned, failures included, and the pipeline of the final
models. The trained models are published as release assets ([RELEASE.md](RELEASE.md)).

The code moved here from the game repository's `tools/rl/`. That directory only ever had one commit there
([`9d6a1b4e`](https://github.com/Stardust-minus/Stronghold-Protocol/commit/9d6a1b4e75daf3f77fd00336a5d305d015efec12),
"tools/rl: the reinforcement-learning environment and trainer for the AI seats"), so no history was imported: `rl/`
here is that directory as it stands now, and the game repository no longer carries it.

## The AI at deploy

Every model seat runs one policy network plus two placement pickers. All three are pure models. Deploy runs no
search, no simulation, no layout rehearsal and no live seed. The seat uses only what a player sees in the game:
monster info and HP, the 机变 kill chance, teammates' cards and flags, and the boss. Everything runs in plain JS on one
CPU core; no GPU is needed.

- **Policy**: an MLP (`rl/py/model.py`, 1.55M parameters, 1.51M at deploy) or v5 (`rl/py/model5.py`, an entity
  transformer, 2.72M parameters, 1.18M at deploy: its separate critic is training-only), greedy. It reads the game
  state plus a national plan (a main nation and a secondary class bond, chosen by rule). JS inference is
  `rl/infer2.mjs` (MLP, about 2 ms per decision) / `rl/infer5.mjs` (v5, about 24 ms), checked against torch.
- **Normal-round picker c9**: a spatial head (`rl/py/picker_train5j.py`, exported by `rl/py/export_p5.py`). It scores
  a candidate pool of legacy layouts, stacking templates and one-step edits, generated without simulation. It
  switches from legacy's default layout only when the predicted gain exceeds a margin.
- **Boss-round placement xboss-v3**: a learned scorer over a pool of deploy-legal boss layouts (`rl/xplace.mjs`),
  trained on search labels scored by the win probability (`rl/py/xtrain.py`, `rl/xlabel.mjs`).
- Together the pickers add +0.13 to +0.20 win over legacy placement on the final models.

The training steers the policy toward the core loop: a national main bond plus a stacked secondary class bond. Every
eval reports the stacking metrics next to win ([records/final/README.md](records/final/README.md) defines them).
Design: [docs/DESIGN.md](docs/DESIGN.md).

## Final recipe

1. **BC init**: behaviour cloning on recorded matches with a national-route rule bot as teacher.
2. **NM**: PPO from the BC, solo:HARD, with the national plan block (`--nat-plan`), to 5M steps.
3. **NMA**: the ABYSS curriculum (`--mix auto1`) to 60M steps.
4. **Extension** (optional): 60M → 90M, 10M at a time, for runs still improving (rule in
   [rl/final/README.md](rl/final/README.md)).
5. **Co-op stage** (arm CB, used on the MLP; it failed on v5): 10M steps of co-op training with a pool of teammate
   proxies (`rl/proxies.mjs`), KL-anchored to its start, kept only if co-op rises and solo drops by ≤ 0.02.

Two architectures × two seeds were trained. Candidates are ranked on paired 480-seed evals of solo:ABYSS and
coop4h1:ABYSS with both pickers, the leaders are compared on a held-out seed set, and the two finalists are confirmed
on a fresh one. The exact arguments, seeds, md5s, the pick rule and the seed sets are in
[rl/final/README.md](rl/final/README.md).

## Reproducing from scratch

```sh
./setup.sh work/code-A6         # the training tree (pinned game commit + rl/ + patches/)
./setup.sh work/code-A7         # the eval tree (the same code)
cp -R rl/final work/FINAL
# each tree needs its manifest: md5sum of every file under data packs plugins schemas server shared tools + package.json
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh data    # record the BC data
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh bc      # MLP and v5 BC
B=$PWD/work ROOT=$PWD/work/FINAL bash work/FINAL/launch.sh rl      # NM -> NMA -> co-op stage, both architectures
```

- The data step runs on the earlier recording tree code-N5 (manifest `ec17d896`). This repository does not rebuild
  that tree byte for byte. Its check is the data manifest `0d275a93`.
- The launchers check tree manifests. A rebuilt tree has its own manifest md5 (`MAN_MD5` / `MAN_A6` / `MAN_A7`).
- The rebuilt tree plays byte-identical seeded matches to the frozen trees the final runs used. The frozen trees sit
  on the game commit before the pinned one. [rl/final/README.md](rl/final/README.md) has the work-dir layout and the
  details.
- Every new option in `rl/` is off by default. With all of them off, the environment's outputs equal the previous
  commit's.
- No checkpoints are in git. The final model and its pickers are a release asset ([RELEASE.md](RELEASE.md));
  `records/weights-md5.csv` lists the md5 of every model and picker file the final evals used.

## Final results

<!-- pick:BEGIN headline -->
The released model is **FIN-MLP-s2-70M**: the MLP (1.55 M parameters), seed 2, NM 5 M → NMA 60 M → co-op stage 70 M.
It was chosen for its inference cost. The v5 run FIN-V5X-s1-90M scored a little higher on the fresh seeds (0.790 /
0.744; paired Δ mean +0.017 ± 0.018, about one standard error; held-out +0.018 ± 0.017), but its policy costs 23 ms
per decision in JS against 1.8 ms for the MLP.

| | solo:ABYSS | coop4h1:ABYSS |
|---|---|---|
| **FIN-MLP-s2-70M**, deploy setting (both pickers), fresh seeds (`--seed 300000000`, 476 each) | **0.769 ± 0.019** | **0.731 ± 0.020** |
| FIN-MLP-s2-70M, same setting, held-out seeds (`--seed 200000000`) | 0.784 ± 0.019 | 0.737 ± 0.020 |
| FIN-MLP-s2-70M, same setting, milestone seeds (`--seed 7000000`) | 0.723 ± 0.021 | 0.702 ± 0.021 |
| coop4h1 with real-player-like humans (proxy cells weighted by 126 real matches), fresh seeds | | 0.702 ± 0.010 |
| natMain / natMainT2 (national main bond; at tier ≥ 2), fresh | 0.99 / 0.97 | 0.92 / 0.82 |
| main / secondary / Σ layers, elites, fresh | 185 / 87 / 325, 5.6 | 185 / 75 / 302, 5.7 |
| without the pickers (milestone seeds) | 0.595 | 0.508 |
| not shipped: FIN-V5X-s1-90M (v5), deploy setting, fresh seeds | 0.790 ± 0.019 | 0.744 ± 0.020 |
| reference: the legacy rule AI in the model seats (11-13 protocol, b0) | 0.010 | 0.071 |
| reference: NAT 88M + boss picker (elite-merge route, not the target play) | 0.596 | 0.581 |

The target was ≥ 0.50 on both. ± is one standard error; stacking is averaged over the model seats.

**Three seed sets, 476 seeds per config each.** The 19 candidates were ranked on the milestone seeds, and the leading
11 were compared on the held-out seeds. Nothing was chosen on the fresh seeds, so they give the headline. A number
from a set that was used to choose is biased upward for whatever was chosen on it (selection bias): FIN-MLP-s2-70M
read 0.784 / 0.737 on the held-out seeds, where the two finalists were chosen, and 0.769 / 0.731 on the fresh ones. On
the milestone seeds, where it ranked 10th of 19 and so was not favoured by the selection, it read 0.723 / 0.702; the
fresh number sits between the two. Every candidate, milestone and gate: [docs/RESULTS.md](docs/RESULTS.md) Part A and
`records/final/`.
<!-- pick:END headline -->

## Hardware

| | to train one run | what we used | to play (deploy) |
|---|---|---|---|
| GPU | one 24 GB GPU | one H200 per seed-1 run; one RTX 4090 shared by both seed-2 runs | none |
| CPU | 32 threads | 50–64 threads per run, 30–60 per eval lane | one core per seat |
| RAM | about 40 GB | 33–37 GB per run | about 0.5 GB per Node process (engine included) |
| time | MLP: 5.1 h to 60M + 1.5 h co-op stage; v5: 6.1 h to 60M | four runs in parallel | about 10–13 s of model time per solo match |

Measured on the final runs; [docs/RUNBOOK.md](docs/RUNBOOK.md) §8 has every stage, the evals and the picker training.

## Records

| Path | Content |
|---|---|
| `records/final/` | milestone evals of the final runs (settings a / b, stacking metrics), the candidate ranking, the co-op stage gate, held-out results; [its README](records/final/README.md) defines every metric. `rl/final/records_final.py` regenerates it. |
| `records/meta/` | `meta.json` of every final stage and co-op arm (full args, tree manifest md5, init and KL-ref md5), paths reduced to `<work>/...` |
| `records/train/` | train curves as CSV (`final/`, `coop/` and earlier arms) |
| `records/results/` | result tables of the experiments since the previous release: NAT plan, v5, pickers, boss placement, co-op robustness, scratch runs. Real-player data appears only as aggregate counts. |
| `records/coop/` | co-op stage arms |
| `records/provenance.csv` | which frozen tree file each repo file comes from, and how it was ported |
| `records/index.csv` | one line per record file |

## What is in here

| Path | Content |
|---|---|
| `rl/` | Overlaid onto the game checkout as `tools/rl/`. Node: environment, observation / action spec, stdin/stdout server, BC recorder, JS inference, placement pickers and parity checks. Python: PPO / BC trainers, evaluation, export, picker and placement trainers, diagnostics. File map: [rl/README.md](rl/README.md). `rl/final/` is the final pipeline, `rl/release/` the release export and the JS player. |
| `patches/` | Server changes the environment needs, against Stronghold-Protocol `9d6a1b4e`: the seat plan of a model-AI seat, the strict-preview field model, and training-only hooks (placement candidates, label seeds, teammate-proxy options). All are inert for legacy and human seats. |
| `setup.sh` | Clones the game repository, pins `9d6a1b4e`, replaces `tools/rl` with `rl/` and applies `patches/`. |
| `scripts/` | Generic launchers from the earlier recipe: `train.sh` and `eval.sh`. |
| `docs/` | [DESIGN](docs/DESIGN.md) (the system), [FINDINGS](docs/FINDINGS.md) (what we learned), [RESULTS](docs/RESULTS.md) (the numbers), [EXPERIMENTS](docs/EXPERIMENTS.md) (every run in order, failures included), [RUNBOOK](docs/RUNBOOK.md) (commands and hardware); `docs/curves/` the in-run eval CSVs. |
| `RELEASE.md` | the release asset layout and how to load and run a released model with its pickers |

## Quick start

```sh
./setup.sh                      # → ./Stronghold-Protocol at 9d6a1b4e with tools/rl and the patches
cd Stronghold-Protocol
node tools/rl/vcheck.mjs        # observation layouts and their dataIds (Node 24)
node tools/rl/smoke.mjs --matches 1 --configs solo:NORMAL --only random
node tools/rl/release/play.mjs <release dir> --matches 4   # a released model with its pickers, in JS (RELEASE.md)
```

Training needs Python 3.12 with PyTorch, numpy and tensorboard (`pip install -r requirements.txt`). Most of the cost
is the environments, which run on CPU. One GPU is enough for the learner.

## Licence

Code: GPL-3.0-or-later ([LICENSE](LICENSE)), as for the game repository. The released model files are under the same
licence ([RELEASE.md](RELEASE.md)). Arknights names, art, audio, text and data are © their owners and are not covered by this
licence. This project is not affiliated with or endorsed by Hypergryph or Yostar.
