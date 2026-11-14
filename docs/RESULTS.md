# Results

Team win rate (any seat of the team clears ABYSS) on paired seeds, greedy policy. ± is one standard error: √(p(1−p)/n)
for a rate, sd / √n of the per-seed differences for a paired Δ.

- **Part A** (11-14) is the final pipeline: headline protocol, 476 paired seeds per cell, setting a (no pickers) and
  setting b (the deploy pickers c9 + xboss-v3), stacking metrics next to every win. Protocol and metric definitions:
  [records/final/README.md](../records/final/README.md). Source tables: `records/final/`.
- **Part B** (11-12 to 11-13 morning) is the earlier hybrid line under its own protocol (band chosen by the policy, legacy
  placement, b3 / b0, no plan). Its numbers do not compare with Part A.

## Final result

<!-- pick:BEGIN results -->
The released model is **FIN-MLP-s2-70M**: MLP (`rl/py/model.py`), 1.55 M parameters, 70 M steps; MLP BC (PB5) → NM to
5 M → NMA (auto1) to 60 M → co-op stage (arm CB, seed offset 73,000,000) to 70 M. It was chosen for its inference
cost. The v5 run FIN-V5X-s1-90M scored a little higher on the fresh seeds (0.790 / 0.744; paired Δ mean +0.017 ±
0.018, about one standard error; held-out +0.018 ± 0.017), but its policy costs 23 ms per decision in JS against 1.8
ms for the MLP. The normal-round picker c9 dominates a match's model time with either policy (about 772 ms per
normal-round decision).

| | solo:ABYSS | coop4h1:ABYSS |
|---|---|---|
| FIN-MLP-s2-70M, setting b, fresh seeds (`--seed 300000000`, 476 each) | 0.769 ± 0.019 | 0.731 ± 0.020 |
| FIN-MLP-s2-70M, setting b, held-out (`--seed 200000000`; the last choice was made here) | 0.784 ± 0.019 | 0.737 ± 0.020 |
| same, milestone seeds (`--seed 7000000`; 10th of 19 on the b mean) | 0.723 ± 0.021 | 0.702 ± 0.021 |
| same, milestone + held-out pooled (952 seeds each) | 0.753 | 0.720 |
| real-player-weighted coop4h1 (human-proxy cells, fresh seeds; A1c) | | 0.702 ± 0.010 |
| setting a (no pickers), milestone seeds | 0.595 | 0.508 |
| picker lift b − a (milestone seeds, paired) | +0.128 ± 0.022 | +0.193 ± 0.025 |
| natMain / natMainT2 (fresh) | 0.99 / 0.97 | 0.92 / 0.82 |
| main tier (fresh) | 1.98 | 1.89 |
| main / secondary / Σ layers (fresh) | 185 / 87 / 325 | 185 / 75 / 302 |
| elites (fresh) | 5.6 | 5.7 |
| not shipped: FIN-V5X-s1-90M (v5, 90 M), setting b, fresh | 0.790 ± 0.019 | 0.744 ± 0.020 |
| FIN-V5X-s1-90M − FIN-MLP-s2-70M, fresh, paired | +0.021 ± 0.024 | +0.013 ± 0.027 |
| FIN-V5X-s1-90M, held-out | 0.788 ± 0.019 | 0.769 ± 0.019 |
| FIN-V5X-s1-90M − FIN-MLP-s2-70M, held-out, paired | +0.004 ± 0.024 | +0.032 ± 0.023 |
| same, milestone + held-out pooled, paired | +0.033 ± 0.018 | +0.029 ± 0.017 |
| reference: NMA 30 M (seed 0), setting b, milestone seeds | 0.679 | 0.540 |
| reference: NAT 88 M + boss picker (elite route, not the target play) | 0.596 | 0.581 |
| legacy fill (every model seat legacy; 11-13 protocol, b0, 480 seeds) | 0.010 | 0.071 |

- **Why the three seed sets differ.** Each has 476 seeds per config, so a rate has an SE of about 0.02 on its own, and
  the sets differ in difficulty; only paired numbers on one set compare candidates. On top of that, picking the best
  of many on a set biases the pick's number on that set upward. The milestone set ranked 19 candidates (its leader
  FIN-V5X-s1-80M, 0.805 / 0.712, fell to 0.756 / 0.685 held-out). The held-out set compared 11, and the two finalists
  are its top two: FIN-MLP-s2-70M 0.784 / 0.737 there and 0.769 / 0.731 fresh, FIN-V5X-s1-90M 0.788 / 0.769 there and
  0.790 / 0.744 fresh. The fresh set chose nothing: its numbers are the headline. The paired gap to FIN-V5X-s1-90M is
  the same on held-out (+0.018 ± 0.017) and fresh (+0.017 ± 0.018).
- FIN-MLP-s2-70M plays the national route: main bonds 拉特兰 0.63, 维多利亚 0.26 (held-out solo), secondaries 灵巧 0.46, 坚守
  0.22, 精准 0.20. No drift toward the elite route. FIN-V5X-s1-90M (fresh): natMain 0.99 / 0.89, Σ layers 336 / 295,
  elites 5.5 / 4.6.
- Deploy cost, measured on the release check matches (8 + 8 per model, `parity.json` of each export; the pickers build
  no battles, sims = 0):

| one pinned core, JS, release check matches | FIN-MLP-s2-70M (MLP) | FIN-V5X-s1-90M (v5) |
|---|---|---|
| policy, ms per decision (all check-match decisions) | 1.8 | 23 |
| c9 (normal rounds), ms per decision | 772 | 714 |
| xboss-v3 (boss rounds), ms per decision | 191 | 191 |
| model time per solo:ABYSS match, s (policy + c9 + xboss-v3) | 10.6 | 11.9 |
| of which the policy, s | 0.2 | 2.3 |
| model time per coop4h1:ABYSS match, s (three model seats) | 31.5 | 33.3 |

The v5 export passed the same parity gate (`docs/RUNBOOK.md` §6) but is not released.
<!-- pick:END results -->

<!-- pick:BEGIN heldout -->
## A1. Held-out confirmation (`--seed 200000000`, setting b, 476 seeds per cell)

The leading candidates on the milestone seeds were re-run on a fresh seed set as the runs finished: 11 in all, CB10
included. Paired Δ vs FIN-MLP-s2-70M, the pick at the time the round ran. The table with the held-out stacking and the
pooled view: `records/final/heldout-paired.md` (`rl/final/hopair.py`).

| candidate | solo:ABYSS | coop4h1:ABYSS | mean | Δ solo | Δ coop | Δ mean |
|---|---|---|---|---|---|---|
| FIN-V5X-s1-90M | 0.788 | 0.769 | 0.778 | +0.004 ± 0.024 | +0.032 ± 0.023 | +0.018 ± 0.017 |
| **FIN-MLP-s2-70M** | 0.784 | 0.737 | 0.761 |  |  |  |
| FIN-V5X-s2-80M | 0.773 | 0.712 | 0.743 | −0.011 ± 0.024 | −0.025 ± 0.025 | −0.018 ± 0.017 |
| FIN-V5X-s2-90M | 0.765 | 0.714 | 0.739 | −0.019 ± 0.024 | −0.023 ± 0.025 | −0.021 ± 0.017 |
| FIN-V5X-s1-70M | 0.769 | 0.708 | 0.738 | −0.015 ± 0.025 | −0.029 ± 0.026 | −0.022 ± 0.018 |
| FIN-MLPX-s2-70M | 0.765 | 0.704 | 0.734 | −0.019 ± 0.026 | −0.034 ± 0.026 | −0.026 ± 0.018 |
| FIN-MLP-s1-70M | 0.744 | 0.714 | 0.729 | −0.040 ± 0.024 | −0.023 ± 0.025 | −0.032 ± 0.017 |
| FIN-V5-s1-70M | 0.725 | 0.718 | 0.722 | −0.059 ± 0.025 | −0.019 ± 0.023 | −0.039 ± 0.017 |
| FIN-V5X-s1-80M | 0.756 | 0.685 | 0.721 | −0.027 ± 0.026 | −0.053 ± 0.026 | −0.040 ± 0.018 |
| FIN-V5-s2-60M | 0.723 | 0.639 | 0.681 | −0.061 ± 0.026 | −0.099 ± 0.025 | −0.080 ± 0.018 |
| CB10 | 0.704 | 0.616 | 0.660 | −0.080 ± 0.025 | −0.122 ± 0.025 | −0.101 ± 0.018 |

- The held-out set removes the milestone set's selection bias ([FINDINGS.md](FINDINGS.md) §13); choosing the finalists
  on it adds its own, which the fresh set in A1b removes. FIN-V5X-s1-80M ranked 1st on the milestone seeds (0.805 /
  0.712) and read 0.756 / 0.685 here. FIN-MLP-s2-70M ranked 10th (0.723 / 0.702) and read 0.784 / 0.737.
- FIN-V5X-s1-90M is the only candidate ahead of FIN-MLP-s2-70M on both seed sets: pooled (952 seeds per cell) +0.033 ±
  0.018 solo, +0.029 ± 0.017 co-op, +0.031 ± 0.012 on the mean. Its held-out lead is about one standard error, and it
  is the maximum of 11 held-out runs, so the pooled number is the conservative one.

### A1b. Fresh-seed confirmation (`--seed 300000000`, setting b, 476 seeds per cell)

The release's JS parity check matches also ran on seeds of this set, for both exports: solo:ABYSS
302,000,000–302,000,007 and coop4h1:ABYSS 880,000,000–880,000,007. They check JS against torch and chose nothing.

Both finalists, setting b, `rl/final/confirm.py` (`records/final/confirm.md`). Paired Δ = FIN-V5X-s1-90M −
FIN-MLP-s2-70M on the same seeds.

| candidate | solo:ABYSS | coop4h1:ABYSS | mean | solo / coop w/o the check-match seeds |
|---|---|---|---|---|
| FIN-MLP-s2-70M | 0.769 | 0.731 | 0.750 | 0.765 / 0.729 |
| FIN-V5X-s1-90M | 0.790 | 0.744 | 0.767 | 0.791 / 0.746 |

Paired V5X - MLP: solo +0.021 ± 0.024 (n 476; w/o check-match seeds +0.026 ± 0.024), coop +0.013 ± 0.027 (n 476; w/o
+0.017 ± 0.028), mean +0.017 ± 0.018

Stacking (fresh set; solo / coop rows):

| candidate | cfg | natMain / T2 | main tier | main / sec / Σ layers | elites | main bond |
|---|---|---|---|---|---|---|
| FIN-MLP-s2-70M | solo | 0.99 / 0.97 | 1.98 | 185 / 87 / 325 | 5.6 | 拉特兰 0.64, 维多利亚 0.24 |
| FIN-MLP-s2-70M | coop | 0.92 / 0.82 | 1.89 | 185 / 75 / 302 | 5.7 | 维多利亚 0.20, 拉特兰 0.20 |
| FIN-V5X-s1-90M | solo | 0.99 / 0.97 | 1.98 | 191 / 89 / 336 | 5.5 | 拉特兰 0.65, 维多利亚 0.24 |
| FIN-V5X-s1-90M | coop | 0.89 / 0.75 | 1.80 | 184 / 68 / 295 | 4.6 | 拉特兰 0.20, 维多利亚 0.20 |

### A1c. Robustness to the human teammate (coop4h1:ABYSS, the same fresh seeds)

The human seat is a scripted proxy that builds one nation (`nat:<P>Ship`; scored on the seeds where that nation is
live) or a class route (`class`). Hole = the cell's win minus the model's own win with the legacy human on the same
seeds.

| cell | live seeds | MLP-s2-70M | V5X-s1-90M | Δ V5X - MLP (paired) | MLP hole vs own legacy | V5X hole vs own legacy | MLP natMain / T2, Σ, elites | V5X natMain / T2, Σ, elites |
|---|---|---|---|---|---|---|---|---|
| 叙拉古 | 269 | 0.736 | 0.773 | +0.037 ± 0.030 | +0.022 ± 0.034 | +0.063 ± 0.033 | 0.98 / 0.89, 300, 5.7 | 0.95 / 0.84, 319, 5.0 |
| 炎 | 306 | 0.771 | 0.758 | −0.013 ± 0.030 | +0.075 ± 0.031 | +0.046 ± 0.030 | 0.95 / 0.85, 322, 5.9 | 0.92 / 0.81, 313, 4.9 |
| 谢拉格 | 302 | 0.652 | 0.619 | −0.033 ± 0.034 | −0.132 ± 0.032 | −0.175 ± 0.032 | 0.92 / 0.77, 278, 5.5 | 0.88 / 0.69, 253, 4.0 |
| 阿戈尔 | 301 | 0.704 | 0.681 | −0.023 ± 0.032 | −0.007 ± 0.032 | −0.010 ± 0.033 | 0.96 / 0.86, 285, 5.7 | 0.93 / 0.78, 285, 4.6 |
| 卡西米尔 | 298 | 0.661 | 0.681 | +0.020 ± 0.032 | −0.084 ± 0.032 | −0.077 ± 0.031 | 0.91 / 0.79, 290, 5.6 | 0.90 / 0.74, 277, 4.4 |
| 维多利亚 | 301 | 0.694 | 0.688 | −0.007 ± 0.033 | −0.023 ± 0.032 | −0.066 ± 0.033 | 0.91 / 0.77, 282, 5.5 | 0.93 / 0.70, 293, 4.6 |
| 拉特兰 | 304 | 0.645 | 0.645 | +0.000 ± 0.031 | −0.128 ± 0.032 | −0.141 ± 0.033 | 0.85 / 0.71, 290, 5.5 | 0.88 / 0.66, 284, 4.3 |
| 萨尔贡 | 299 | 0.722 | 0.766 | +0.043 ± 0.030 | +0.017 ± 0.029 | +0.027 ± 0.030 | 0.96 / 0.87, 300, 5.8 | 0.94 / 0.81, 309, 5.1 |
| class | all 476 | 0.674 | 0.704 | +0.029 ± 0.026 | −0.057 ± 0.025 | −0.040 ± 0.026 | 0.91 / 0.79, 293, 5.5 | 0.90 / 0.76, 301, 4.6 |

Real-player-weighted coop4h1: the cells weighted by the nation mix of 126 real matches (aggregates only; 78 %
national, 22 % class).

| candidate | real-player-weighted coop4h1 | legacy-proxy coop (fresh) |
|---|---|---|
| FIN-MLP-s2-70M | 0.702 ± 0.010 | 0.731 |
| FIN-V5X-s1-90M | 0.713 ± 0.010 | 0.744 |

Paired weighted Δ V5X - MLP = +0.012 ± 0.012
<!-- pick:END heldout -->

## A2. Milestones of the final runs (476 paired seeds, `--seed 7000000`)

Setting b unless noted. Δ = paired vs NMA 30 M in the same setting.

| checkpoint | b solo | Δ | b coop | Δ | a solo | a coop | natMain / T2 (b solo) | natMain / T2 (b coop) | Σ layers b solo / coop | elites b solo / coop |
|---|---|---|---|---|---|---|---|---|---|---|
| NMA 30 M (reference) | 0.679 | | 0.540 | | 0.504 | 0.382 | 0.99 / 0.96 | 0.90 / 0.73 | 304 / 243 | 6.0 / 5.1 |
| FIN-MLP-s1 60 M | 0.708 | +0.029 ± 0.027 | 0.618 | +0.078 ± 0.027 | 0.544 | 0.489 | 0.99 / 0.96 | 0.90 / 0.75 | 314 / 264 | 5.5 / 5.0 |
| FIN-MLP-s2 60 M | 0.742 | | 0.601 | | 0.569 | 0.424 | 0.99 / 0.96 | 0.88 / 0.66 | 328 / 253 | 5.7 / 4.9 |
| FIN-MLP-s2 70 M (co-op stage) | 0.723 | | 0.702 | | 0.595 | 0.508 | | 0.91 / 0.81 | 296 (coop) | 5.6 (coop) |
| FIN-V5-s1 60 M | 0.704 | | 0.687 | | 0.561 | 0.511 | 0.99 / 0.97 | 0.87 / 0.71 | 334 / 295 | 6.0 / 4.8 |
| FIN-V5-s2 60 M | 0.754 | | 0.662 | | 0.592 | 0.511 | 0.98 / 0.96 | 0.88 / 0.75 | 331 / 281 | 5.3 / 4.6 |

Every milestone (30–90 M), the Δ columns and the top main / secondary bonds: `records/final/milestones.csv` and
`summary.md`. The main nations are 拉特兰 (about 0.63 of solo seats) and 维多利亚 (0.28); co-op seats spread over
谢拉格 / 拉特兰 / others because the human's nation is yielded. The commonest secondaries are 灵巧, 精准 and 坚守.

## A3. Co-op stage gate (the stage output vs its start, paired)

| run | b solo | b coop | a solo | a coop | adopted |
|---|---|---|---|---|---|
| FIN-MLP-s2 | −0.019 ± 0.026 | +0.101 ± 0.027 | +0.025 ± 0.028 | +0.084 ± 0.026 | yes |
| CB10 (NMA 30 M → 40 M) | +0.008 ± 0.026 | +0.080 ± 0.028 | +0.010 ± 0.028 | +0.073 ± 0.025 | yes |
| CM2 (FIN-MLP-s1 30 M → 40 M) | −0.040 ± 0.027 | +0.118 ± 0.026 | +0.025 ± 0.027 | +0.075 ± 0.025 | (validation) |
| CV5 (FIN-V5-s1 30 M → 40 M, lr 1e-4) | +0.046 ± 0.027 | −0.027 ± 0.026 | +0.046 | +0.006 ± 0.027 | no |
| CV5b (same, lr 3e-4) | −0.017 ± 0.029 | −0.048 ± 0.027 | −0.029 | −0.023 | no |

The other runs' stage results: `records/final/coopgate.csv`. Gate: co-op Δ > 0 and solo Δ ≥ −0.02.

## A4. Picker lift (setting b − a, same checkpoint, paired)

+0.13 to +0.20 at every milestone of every final run, solo and co-op. Σ layers rise by 30–51 with the pickers; natMain,
natMainT2 and the main nations do not change. Picker development, against the previous picker on the same boards:
c9 +0.025 ± 0.009 over the K8 picker Sall (2,856 pairs), xboss-v3 +0.029 ± 0.007 over xboss-v2 ([EXPERIMENTS.md](EXPERIMENTS.md)
Phase 6).

## A5. Co-op diagnostics (`py/coopstats.py`, setting b, coop4h1:ABYSS)

| checkpoint | collision | win, collided matches | win, other matches | replans |
|---|---|---|---|---|
| FIN-MLP-s2 60 M | 0.079 | 0.540 (113 matches) | 0.620 | 0 |
| FIN-MLP-s2 70 M | 0.088 | 0.632 (125) | 0.726 | 0 |

Robustness to the teammate type (setting a, NMA 30 M and the CB / CA arms): `records/results/coop-*.csv`.

# Part B: the hybrid line (11-12 to 11-13 morning)

Team win rate on paired seeds, greedy policy that also picks its own band, legacy placement for the RL seat at r0. `b3`
/ `b0` = legacy's boss-round rehearsal on / off; b0 was the deploy setting at the time, with `--visible-only` since
11-13 (§1b). `fa` is the team's share of the Final Assault pool. Steps are environment decisions of the RL seats.
Source CSVs: `docs/curves/`.

## 1. Headline: solo:ABYSS, 480 paired seeds (2026-11-13)

| Checkpoint | b3 win | b0 win | b3 fa | b0 fa |
|---|---|---|---|---|
| m3s @ 20.0 M | 0.290 ± 0.021 | 0.246 ± 0.020 | 0.772 | 0.741 |
| m3s @ 30.0 M | 0.338 ± 0.022 | 0.287 ± 0.021 | 0.792 | 0.764 |
| m3s @ 40.0 M | 0.375 ± 0.022 | 0.310 ± 0.021 | 0.815 | 0.780 |
| S @ 20.0 M | 0.308 ± 0.021 | 0.263 ± 0.020 | 0.780 | 0.746 |
| S @ 25.0 M | 0.317 ± 0.021 | 0.252 ± 0.020 | 0.793 | 0.756 |
| S @ 30.0 M | 0.346 ± 0.022 | 0.292 ± 0.021 | 0.798 | 0.762 |
| SC @ 25.0 M | 0.319 ± 0.021 | 0.265 ± 0.020 | 0.775 | 0.744 |
| SK @ 25.0 M | 0.304 ± 0.021 | 0.256 ± 0.020 | 0.767 | 0.735 |
| **SK @ 30.0 M** | **0.394 ± 0.022** | **0.319 ± 0.021** | 0.823 | 0.784 |

Paired differences on the same 480 seeds (`paired.csv`):

| A − B | b3 Δwin | b0 Δwin | b3 Δfa |
|---|---|---|---|
| S @ 20 M − m3s @ 20 M (mix v3s vs v3) | +0.019 ± 0.021 | +0.017 ± 0.020 | +0.008 ± 0.010 |
| S @ 30 M − m3s @ 30 M | +0.008 ± 0.022 | +0.004 ± 0.020 | +0.007 ± 0.010 |
| SC @ 25 M − S @ 25 M (mix v3sc vs v3s) | +0.002 ± 0.022 | +0.013 ± 0.020 | −0.018 ± 0.010 |
| SK @ 25 M − S @ 25 M (KL 0.003 vs 0.01) | −0.013 ± 0.022 | +0.004 ± 0.020 | −0.025 ± 0.009 |
| SK @ 30 M − S @ 30 M | **+0.048 ± 0.024** | +0.027 ± 0.024 | +0.024 ± 0.009 |
| SK @ 30 M − m3s @ 40 M | +0.019 ± 0.023 | +0.008 ± 0.024 | +0.008 ± 0.010 |
| m3s @ 40 M − m3s @ 30 M | +0.038 ± 0.023 | +0.023 ± 0.023 | +0.024 ± 0.011 |
| S @ 30 M − S @ 20 M | +0.038 ± 0.021 | +0.029 ± 0.022 | +0.018 ± 0.010 |

Steps are matched to within 0.1 % except where noted (SK @ 30 M vs m3s @ 40 M). Still running at the time of writing:
m3s, S, SK, SC, SO1 (mix solo1, from S @ 20 M).

## 1b. The deploy information set (`--visible-only`, 11-13)

§1–§3 were measured before the visible-only switch existed ([DESIGN.md](DESIGN.md) §3). The same checkpoints with
it on, solo:ABYSS b0, 480 paired seeds (`curves/visible_only.csv`):

| Checkpoint | b0 plain | b0 visible-only | Δ (paired) | seeds that change |
|---|---|---|---|---|
| S @ 30 M | 0.292 ± 0.021 | 0.277 ± 0.020 | −0.015 ± 0.015 | 51 |
| SK @ 30 M | 0.319 ± 0.021 | **0.329 ± 0.021** | +0.010 ± 0.016 | 61 |

Both differences are within noise. From here on, `scripts/eval.sh` evaluates with `--visible-only` by default, and
`VISIBLE_ONLY=0` reproduces the plain tables.

## 2. One model seat with rule-AI mates, deploy-realistic (288 seeds, 11-13)

RL seat r0, legacy mates r3 (production), b3. The solo row is the same protocol on 288 seeds.

| config | S @ 20 M | m3s @ 20 M | Δ (paired) |
|---|---|---|---|
| solo:ABYSS | 0.309 ± 0.027 | 0.285 ± 0.027 | +0.024 ± 0.028 |
| coop4l:ABYSS | 0.132 ± 0.020 | 0.122 ± 0.019 | +0.010 ± 0.023 |
| coop2l:ABYSS | 0.118 ± 0.019 | 0.097 ± 0.017 | +0.021 ± 0.023 |
| solo:ABYSS at b0 | 0.267 ± 0.026 | 0.226 ± 0.025 | +0.042 ± 0.028 |

The gap to the 50 % target is about 11–19 points solo (b3 / b0) and about 37 points in one-RL co-op. Why co-op is
hard: [FINDINGS.md](FINDINGS.md) §6.

## 3. In-run evaluations, 96 seeds (SE ≈ ±0.045 at these rates)

All configs at r0. coop4l / coop2l mates at r0 here (unlike §2). Per-arm CSVs have fa and the b0 runs.

| arm @ steps | solo:ABYSS b3 | solo:ABYSS b0 | coop4l:ABYSS b3 | coop2l:ABYSS b3 | coop4ai:ABYSS b3 | coop4:ABYSS b3 | solo:HARD b3 | coop4l:HARD b3 |
|---|---|---|---|---|---|---|---|---|
| m3 @ 10.0 M | 0.219 | – | – | – | 0.427 | 0.135 | 0.854 | – |
| m3 @ 20.0 M | 0.281 | – | – | – | 0.615 | 0.177 | 0.885 | – |
| m3 @ 30.0 M | 0.365 | – | – | – | 0.583 | 0.229 | 0.896 | – |
| m3 @ 40.0 M | 0.281 | – | – | – | 0.615 | 0.219 | 0.875 | – |
| m3s @ 20.0 M | 0.312 | 0.271 | 0.104 | 0.094 | 0.562 | 0.198 | 0.875 | 0.802 |
| m3s @ 30.0 M | 0.292 | 0.260 | 0.135 | 0.062 | 0.552 | 0.219 | 0.802 | 0.854 |
| m3s @ 40.0 M | 0.396 | 0.354 | 0.219 | 0.156 | 0.688 | 0.302 | 0.865 | 0.865 |
| S @ 20.0 M | 0.354 | 0.281 | 0.146 | 0.062 | 0.531 | 0.167 | 0.917 | 0.802 |
| S @ 30.0 M | 0.406 | 0.354 | 0.146 | 0.125 | 0.635 | 0.229 | 0.906 | 0.875 |
| SK @ 30.0 M | 0.469 | 0.396 | 0.115 | 0.240 | 0.729 | 0.240 | 0.906 | 0.729 |
| KL @ 20.0 M | 0.344 | 0.302 | 0.104 | 0.125 | 0.583 | 0.260 | 0.875 | 0.760 |
| P @ 20.0 M | 0.302 | 0.250 | 0.073 | 0.062 | 0.521 | 0.198 | 0.865 | 0.917 |
| P0 @ 20.0 M | 0.354 | 0.240 | 0.115 | 0.094 | 0.490 | 0.188 | 0.885 | 0.833 |
| ALL @ 20.0 M | 0.333 | 0.281 | 0.094 | 0.042 | 0.562 | 0.240 | 0.885 | 0.781 |

The 96-seed numbers run high for the leading arm. For example, SK @ 30 M reads 0.469 here but 0.394 on 480 seeds.
Rank arms with §1 instead.

## 4. Target configs, baselines (96 seeds, 11-12)

| config | production legacy | teacher | ppo2 @ 15 M | ppo3h @ 6.0 M (fixed env) | m3 @ 10 M |
|---|---|---|---|---|---|
| solo:ABYSS b3 | 0.010 | 0.010 | 0.042 | 0.094 | 0.219 |
| solo:ABYSS b0 | 0.010 | 0.010 | 0.031 | 0.062 | 0.219 |
| coop4l:ABYSS b3 | 0.042 | 0.010 | 0.010 | 0.062 | 0.135 |
| coop4l:ABYSS b0 | 0.042 | 0.010 | 0.010 | 0.052 | 0.094 |
| coop2l:ABYSS b3 | 0.000 | 0.000 | 0.010 | 0.021 | 0.073 |
| coop2l:ABYSS b0 | 0.010 | 0.000 | 0.010 | 0.021 | 0.052 |
| coop4ai:ABYSS b3 | 0.000 | 0.000 | 0.073 | 0.135 | 0.427 |

Production legacy here is evaluated at r0 like the RL seat. With its production rehearsal (r3) it wins 1.0 % of
solo:ABYSS (384 seeds). The teacher is legacy with the AI-seat rules enforced.

## 5. Earlier line, for reference (11-12)

ppo2 against production legacy, 96 seeds, r0b3:

| config | production legacy | teacher | ppo2 @ 15 M |
|---|---|---|---|
| solo:HARD | 0.344 | 0.344 | 0.719 |
| solo:ABYSS | 0.010 | 0.010 | 0.042 |
| coop4:HARD | 0.521 | 0.406 | 0.865 |
| coop4:ABYSS | 0.062 | 0.031 | 0.021 |

The final ppo2 (13 / 14 / 15 M averaged) against production legacy at its production rehearsal r3:

| config | seeds | legacy win | ppo2 win | legacy fa | ppo2 fa |
|---|---|---|---|---|---|
| solo:ABYSS | 384 | 1.0 % | 4.4 % | 0.218 | 0.514 |
| coop4:ABYSS | 384 | 3.1 % | 4.9 % | 0.200 | 0.368 |
| solo:HARD | 192 | 0.391 | 0.769 | | |
| coop4:HARD | 192 | 0.646 | 0.899 | | |

## 6. Placement and rehearsal (m3 @ 10 M)

| setting | config | seeds | win |
|---|---|---|---|
| r0b3 (eval default) | solo:ABYSS | 288 | 0.181 |
| production rehearsal r3 | solo:ABYSS | 288 | 0.208 (+0.028) |
| peek hill-climb K16 from R1 (live seed: teacher only) | solo:ABYSS | 288 | 0.306 (+0.125 ± 0.023) |
| r0b3 | coop4ai:ABYSS | 96 | 0.427 |
| production rehearsal r3b3 | coop4ai:ABYSS | 96 | 0.500 (+0.073 ± 0.043; fa 0.835) |
| v6 per-piece placement BC (vs m3) | solo:ABYSS | 288 | −0.080 (p 0.0001) |

## CSV files (`docs/curves/`)

Part B's tables come from these files; Part A's from `records/final/` and `records/results/`.

| File | Columns |
|---|---|
| `final.csv` | the columns of `<arm>.csv`: the 96-seed in-run evaluations (solo:HARD, solo:ABYSS; coop4h1:ABYSS in the co-op stage) of the final runs, their extensions and co-op stages and the co-op arms CA / CB / CM2 / CV5 / CV5b, from `records/train/*/*-runs.json` |
| `<arm>.csv` | arm, steps, rehearsal, boss_rehearsal, config, n_seeds, win, win_se, fa: the in-run evaluations (boss_rehearsal blank = not recorded by that run's evaluator; fa 0 for ppo1 = not measured then) |
| `solo480.csv` | the same columns, solo:ABYSS on 480 seeds |
| `deploy288.csv` | the same + mates (`mates r3`: legacy mates at production rehearsal) |
| `visible_only.csv` | checkpoint, steps, config, boss_rehearsal, n_seeds, plain vs visible-only win, paired dwin ± SE, seeds that change, fa |
| `paired.csv` | eval, A, B, boss_rehearsal, config, n_seeds, dwin, dwin_se, dfa, dfa_se: per-seed paired differences A − B |
| `baselines.csv` | policy, rehearsal, boss_rehearsal, config, n_seeds, win, win_se, fa: legacy, teacher, BC, ppo2, the 6.0 M init (bugged and fixed env), m3 @ 10 M |
| `hyperparams.csv` | one row per launch (resumes included): arm, launch, parent, start_steps, last_steps, kl_ref and every recipe flag |
