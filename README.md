# Stronghold-RL

Reinforcement learning for the AI seats of [Stronghold Protocol](https://github.com/Stardust-minus/Stronghold-Protocol),
an unofficial, non-commercial fan remake of the Arknights auto-chess mode *Stronghold Protocol: Alliance* (卫戍协议：盟约).
This repository holds the environment, the PPO trainer, the evaluation tools and the plain-JS inference, plus the
documentation of how the policies were tuned, failures included, and the pipeline of the final models.

The code moved here from the game repository's `tools/rl/`. That directory only ever had one commit there
([`9d6a1b4e`](https://github.com/Stardust-minus/Stronghold-Protocol/commit/9d6a1b4e75daf3f77fd00336a5d305d015efec12),
"tools/rl: the reinforcement-learning environment and trainer for the AI seats"), so no history was imported: `rl/`
here is that directory as it stands now, and the game repository no longer carries it.

## The AI at deploy

Every model seat runs one policy network plus two placement pickers. All three are pure models. Deploy runs no
search, no simulation, no layout rehearsal and no live seed. The seat sees only what a player sees in the game:
monster info and HP, the 机变 kill chance, teammates' cards and flags, and the boss.

- **Policy**: an MLP (`rl/py/model.py`, 1.55M parameters, 1.51M at deploy) or v5 (`rl/py/model5.py`, an entity
  transformer, 2.72M parameters, 1.18M at deploy: its separate critic is training-only), greedy. JS inference is `rl/infer2.mjs` (MLP) / `rl/infer5.mjs` (v5), checked against torch.
- **Normal-round picker c9**: a spatial head (`rl/py/picker_train5j.py`, exported by `rl/py/export_p5.py`). It scores
  a candidate pool of legacy layouts, stacking templates and one-step edits, generated without simulation. It
  switches from legacy's default layout only when the predicted gain exceeds a margin.
- **Boss-round placement xboss-v3**: a learned scorer over the deploy-legal boss layouts (`rl/xplace.mjs`), trained on
  search labels (`rl/py/xtrain.py`, `rl/xlabel.mjs`).

The training steers the policy toward the core loop: a national main bond plus a stacked secondary class bond. Every
eval reports the stacking metrics next to win.

## Final recipe

1. **BC init**: behaviour cloning on recorded matches with a national-route rule bot as teacher.
2. **NM**: PPO from the BC, solo:HARD, with the national plan block (`--nat-plan`), to 5M steps.
3. **NMA**: the ABYSS curriculum (`--mix auto1`) to 60M steps.
4. **Extension** (optional): 60M → 90M for runs still improving at 60M.
5. **Co-op stage** (arm CB, used on the MLP): 10M steps of co-op training with a pool of teammate proxies
   (`rl/proxies.mjs`), KL-anchored to its start.

Two architectures × two seeds were trained. The pick is made on paired 480-seed evals of solo:ABYSS and
coop4h1:ABYSS with both pickers, then confirmed on a held-out seed set. The exact arguments, seeds, md5s, the pick rule
and the held-out protocol are in [rl/final/README.md](rl/final/README.md).

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
- No weights or checkpoints are published. `records/weights-md5.csv` lists the md5 of every model and picker file the
  final evals used.

## Final results

*Placeholder: filled in by the follow-up commit after the final pick.* The tables will be in `records/final/`.

## Records

| Path | Content |
|---|---|
| `records/final/` | milestone evals of the final runs (settings a / b, stacking metrics), the candidate ranking, the co-op stage gate, held-out results. `rl/final/records_final.py` regenerates it. |
| `records/meta/` | `meta.json` of every final stage and co-op arm (full args, tree manifest md5, init and KL-ref md5), paths reduced to `<work>/...` |
| `records/train/` | train curves as CSV (`final/`, `coop/` and earlier arms) |
| `records/results/` | result tables of the experiments since the previous release: NAT plan, v5, pickers, boss placement, co-op robustness, scratch runs. Real-player data appears only as aggregate counts. |
| `records/coop/` | co-op stage arms |
| `records/provenance.csv` | which frozen tree file each repo file comes from, and how it was ported |
| `records/index.csv` | one line per record file |

## What is in here

| Path | Content |
|---|---|
| `rl/` | Overlaid onto the game checkout as `tools/rl/`. Node: environment, observation / action spec, stdin/stdout server, BC recorder, JS inference, placement pickers and parity checks. Python: PPO / BC trainers, evaluation, export, picker and placement trainers, diagnostics. File map: [rl/README.md](rl/README.md). `rl/final/` is the final pipeline. |
| `patches/` | Server changes the environment needs, against Stronghold-Protocol `9d6a1b4e`: the seat plan of a model-AI seat, the strict-preview field model, and training-only hooks (placement candidates, label seeds, teammate-proxy options). All are inert for legacy and human seats. |
| `setup.sh` | Clones the game repository, pins `9d6a1b4e`, replaces `tools/rl` with `rl/` and applies `patches/`. |
| `scripts/` | Generic launchers from the earlier recipe: `train.sh` and `eval.sh`. |
| `docs/` | [DESIGN](docs/DESIGN.md), [FINDINGS](docs/FINDINGS.md), [RESULTS](docs/RESULTS.md), [EXPERIMENTS](docs/EXPERIMENTS.md), [RUNBOOK](docs/RUNBOOK.md) up to 11-13. |

## Quick start

```sh
./setup.sh                      # → ./Stronghold-Protocol at 9d6a1b4e with tools/rl and the patches
cd Stronghold-Protocol
node tools/rl/vcheck.mjs        # observation layouts and their dataIds (Node 24)
node tools/rl/smoke.mjs --matches 1 --configs solo:NORMAL --only random
```

Training needs Python 3.12 with PyTorch, numpy and tensorboard (`pip install -r requirements.txt`). Most of the cost
is the environments, which run on CPU. One GPU is enough for the learner.

## Licence

GPL-3.0-or-later ([LICENSE](LICENSE)), as for the game repository. Arknights names, art, audio, text and data are ©
their owners and are not covered by this licence. This project is not affiliated with or endorsed by Hypergryph or
Yostar.
