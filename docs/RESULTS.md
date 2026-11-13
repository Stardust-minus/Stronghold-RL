# Results

Team win rate (any seat of the team clears ABYSS / HARD) on paired seeds, greedy policy that also picks its own band,
legacy placement for the RL seat at r0. `b3` / `b0` = legacy's boss-round rehearsal on / off; **b0 is the deploy
setting** (see [DESIGN.md](DESIGN.md) §9); since 11-13 the deploy number is b0 with `--visible-only` (§1b). ± is one
standard error: √(p(1−p)/n) for a rate, sd / √n of the per-seed differences for a paired Δ. `fa` is the team's share
of the Final Assault pool. Steps are environment decisions of the RL seats. Source CSVs: `docs/curves/`.

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

| File | Columns |
|---|---|
| `<arm>.csv` | arm, steps, rehearsal, boss_rehearsal, config, n_seeds, win, win_se, fa: the in-run evaluations (boss_rehearsal blank = not recorded by that run's evaluator; fa 0 for ppo1 = not measured then) |
| `solo480.csv` | the same columns, solo:ABYSS on 480 seeds |
| `deploy288.csv` | the same + mates (`mates r3`: legacy mates at production rehearsal) |
| `visible_only.csv` | checkpoint, steps, config, boss_rehearsal, n_seeds, plain vs visible-only win, paired dwin ± SE, seeds that change, fa |
| `paired.csv` | eval, A, B, boss_rehearsal, config, n_seeds, dwin, dwin_se, dfa, dfa_se: per-seed paired differences A − B |
| `baselines.csv` | policy, rehearsal, boss_rehearsal, config, n_seeds, win, win_se, fa: legacy, teacher, BC, ppo2, the 6.0 M init (bugged and fixed env), m3 @ 10 M |
| `hyperparams.csv` | one row per launch (resumes included): arm, launch, parent, start_steps, last_steps, kl_ref and every recipe flag |
