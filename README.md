# Stronghold-RL

Reinforcement learning for the AI seats of [Stronghold Protocol](https://github.com/Stardust-minus/Stronghold-Protocol),
an unofficial, non-commercial fan remake of the Arknights auto-chess mode *Stronghold Protocol: Alliance* (卫戍协议：盟约).
This repository holds the environment, the PPO trainer, the evaluation tools and the plain-JS inference, plus the
documentation of how the current policies were tuned, failures included.

The code moved here from the game repository's `tools/rl/`. That directory only ever had one commit there
([`9d6a1b4e`](https://github.com/Stardust-minus/Stronghold-Protocol/commit/9d6a1b4e75daf3f77fd00336a5d305d015efec12),
"tools/rl: the reinforcement-learning environment and trainer for the AI seats"), so no history was imported: `rl/`
here is that directory as it stands now, and the game repository no longer carries it.

## Status (2026-11-13)

Target: **pure-model inference** wins more than half of the games at the highest difficulty (ABYSS, 终极模拟), solo and
as the one model seat next to rule-AI / human teammates. Pure means a greedy policy, no search, and legacy placement
without layout rehearsal (`r0`). The boss-round rehearsal of legacy placement is reported both on (`b3`) and off
(`b0`); `b0` is the deploy setting because the deployed AI must not use any simulation or information a player
cannot see.

| solo:ABYSS, team win, 480 paired seeds | b3 | b0 |
|---|---|---|
| production legacy AI (96 seeds) | 0.010 | 0.010 |
| ppo2 @ 15 M, the previous public model (96 seeds) | 0.042 | 0.031 |
| m3s @ 40 M | 0.375 ± 0.022 | 0.310 ± 0.021 |
| S @ 30 M | 0.346 ± 0.022 | 0.292 ± 0.021 |
| **SK @ 30 M** (best so far) | **0.394 ± 0.022** | **0.319 ± 0.021** |

- With the deploy information set (`--visible-only`: no hidden boss, no other players' hands), SK @ 30 M wins
  **0.329 ± 0.021** at b0. The switch moves the b0 numbers by −0.015 ± 0.015 (S) and +0.010 ± 0.016 (SK), both within
  noise.
- One model seat with legacy rule-AI mates is much harder: about 0.12–0.13 in 4-player and 2-player co-op
  (288 seeds, S @ 20 M). Most of that gap is structural; see [docs/FINDINGS.md](docs/FINDINGS.md) §6.
- Four model seats together (coop4ai:ABYSS) reach 0.73 (96 seeds, SK @ 30 M).
- Training continues. No weights or checkpoints are published.

Full tables: [docs/RESULTS.md](docs/RESULTS.md). The chronological log of every run: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## What is in here

| Path | Content |
|---|---|
| `rl/` | Overlaid onto the game checkout as `tools/rl/`. Node: environment, observation / action spec, stdin/stdout server, BC recorder, JS inference and parity checks. Python: PPO / BC trainers, evaluation, export, band-head fitting, diagnostics. File map: [rl/README.md](rl/README.md). |
| `patches/` | Server changes the environment needs, against Stronghold-Protocol `9d6a1b4e`: the one-round-trip seat plan of a model-AI seat (`server/match/seatPlan.js`) with its gated hooks, and the strict-preview path of legacy's field model. Both are inert for legacy and human seats. |
| `setup.sh` | Clones the game repository, pins `9d6a1b4e`, replaces `tools/rl` with `rl/` and applies `patches/`. |
| `scripts/` | Generic launchers: `train.sh` (the current recipe, restart-safe) and `eval.sh` (the headline evaluations). |
| `docs/` | [DESIGN](docs/DESIGN.md), [FINDINGS](docs/FINDINGS.md), [RESULTS](docs/RESULTS.md), [EXPERIMENTS](docs/EXPERIMENTS.md), [RUNBOOK](docs/RUNBOOK.md); `docs/curves/` holds small CSVs: eval history per arm, the 480-seed results, paired differences and each arm's hyperparameters. |

## Quick start

```sh
./setup.sh                      # → ./Stronghold-Protocol at 9d6a1b4e with tools/rl and the patches
cd Stronghold-Protocol
node tools/rl/vcheck.mjs        # observation layouts and their dataIds (Node 24)
node tools/rl/smoke.mjs --matches 1 --configs solo:NORMAL --only random
```

Training needs Python 3.12 with PyTorch, numpy and tensorboard (`pip install -r requirements.txt`). Most of the cost
is the environments, which run on CPU. One GPU is enough for the learner. [docs/RUNBOOK.md](docs/RUNBOOK.md) covers
the whole pipeline from behaviour cloning to the current recipe, with the parity and regression checks.

## Licence

GPL-3.0-or-later ([LICENSE](LICENSE)), as for the game repository. Arknights names, art, audio, text and data are ©
their owners and are not covered by this licence. This project is not affiliated with or endorsed by Hypergryph or
Yostar.
