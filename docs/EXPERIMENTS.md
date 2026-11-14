# Experiment log

This is every training run and side study in order, failures included. Phases 1–3 (11-11 to 11-13 morning) give times
in UTC+8; from Phase 4 on (11-13 to 11-14) times are UTC. The final pipeline's exact arguments are in
[rl/final/README.md](../rl/final/README.md), its results in `records/final/`, and the result tables of Phases 4–8 in
`records/results/`.

For each run the entry gives:

- **parent**: the checkpoint the run starts from, and its step;
- **change**: the single factor changed against the run it is compared with, as exact flags;
- **why**: the reason it was tried;
- **result**: team win, paired seeds, ± one SE;
- **decision**: what was done next.

Where nothing else is said, evaluation in Phases 1–3 is greedy, r0b3, 96 seeds; from Phase 4 on it is greedy, b0,
480 paired seeds. Steps count RL-seat decisions. The full flags of
every launch are in [curves/hyperparams.csv](curves/hyperparams.csv), the in-run evaluations in `curves/<arm>.csv`,
and the protocol in [DESIGN.md](DESIGN.md) §9.

Abbreviations: `fa` = the team's Final Assault pool share; `b3` / `b0` = legacy boss-round rehearsal 3 / 0; teacher =
legacy with the AI-seat rules enforced; production legacy = the shipped rule AI.

## Phase 1: economy-only policy, observation v2 (11-11 → 11-12 morning)

### bc1: behaviour cloning (11-11 23:14)
- **Parent:** none.
- **Data:** 4.8 M decisions of the rule-enforcing teacher (`record.mjs`, obs v2: 920 inputs, 60 actions).
- **Change:** SPNet, 1.32 M parameters, `train_bc.py` (4 epochs, lr 1e-3, value weight 0.5).
- **Result:** best validation CE 0.538 after 95 s (accuracy 0.80: buy 0.68, sell 0.58). Plays worse than legacy.
- **Decision:** use it only as the PPO start. The low accuracy later turned out to be partly label ambiguity: identical
  shop cards encode identically (`twin.py`).

### ppo1 (11-11 23:18 → 11-12 04:13)
- **Parent:** bc1.
- **Change:** PPO with a staged curriculum NORMAL → HARD → ABYSS, promotion at 95 % of the teacher's win.
  - `--bc-kl 0.1 --bc-kl-steps 5e6` (decaying anchor); reward version 1 (round 0.1, LP 1.0, win 2.0, team 0.75);
  - 24 processes × 16 matches, γ 0.997, λ 0.95, clip 0.2, ent 0.003, lr 1e-4, 3e7 steps.
- **Why:** first RL baseline.
- **Result:** 3.0e7 steps in 4.9 h (about 1780 steps/s). Final, 192 seeds, against the teacher:
  - solo HARD 0.33 → 0.58 and coop4 HARD 0.42 → 0.76, both significant;
  - solo NORMAL 0.76 → 0.86;
  - ABYSS about 0–3 %, no gain.
- **Decision:** ppo2 with a dense ABYSS signal.

### ppo2 (11-12 08:51 → 10:30, stopped at 15.7 M)
- **Parent:** ppo1 final.
- **Change:**
  - `--curriculum fixed` (mix v1: ABYSS .60 / HARD .30 / NORMAL .10);
  - `--reward '{"round":0.05,"fa":1.5}'` (team Final Assault share);
  - `--bc-kl 0.01 --bc-kl-steps 0` (constant anchor to ppo1);
  - `--shape-tier 0.06 --shape-stack 0.08` (potential-based bond shaping);
  - `--boss-rehearsal 3 --adv-norm global --vclip 0`, 60 processes.
- **Why:** ABYSS was flat; give it a dense signal and keep the HARD gains.
- **Result:** r0b3, 96 seeds, against production legacy:

| config | production legacy | ppo2 @ 15 M |
|---|---|---|
| solo HARD | 0.344 | 0.719 ± 0.046 |
| solo ABYSS | 0.010 | 0.042 ± 0.020 |
| coop4 HARD | 0.521 | 0.865 ± 0.035 |
| coop4 ABYSS | 0.062 | 0.021 ± 0.015 |

  Averaged over 13 / 14 / 15 M, against production legacy at its production rehearsal r3: solo ABYSS (384 seeds) win
  1.0 % → 4.4 %, fa 0.218 → 0.514.
- **Decision:** the owner stopped it for a retrain with every optimisation integrated. ppo2 @ 15 M stays the anchor of
  everything after it, grown to obs v3.

### Band strength table, Step 0 (11-12 midday)
- **Parent:** ppo2 @ 15 M.
- **Change:** seat 0 forced to each of the 40 bands; 48 paired seeds × {solo, coop4} × {HARD, ABYSS}; `band_table.py`.
- **Why:** before the policy is given band choice, measure what choice is worth.
- **Result:**
  - ABYSS: solo best ioleta, fa 0.637 vs the random pick's 0.502 (+0.135, 95 % CI +0.055 … +0.216); co-op best
    amedic, 0.400 vs 0.272.
  - HARD: within noise.
- **Decision:** the policy picks the band. The table is the band head's fit target from 16:56 on.

### Architecture / capacity probe (11-12 11:00 → 15:00, BC only)
- **Parent:** none (BC on the bc1 data).
- **Change:** variants trained in lock-step on the same batches (`bc_probe2.py`, `model2.py`):
  - width ×0.5 – ×8;
  - a residual LayerNorm trunk;
  - entity self-attention;
  - bond look-ahead features;
  - 25 / 50 / 100 % of the data.
- **Why:** was capacity or missing information the limit?
- **Result:**
  - The first round's "attention leads by far" was wrong: the large variants had diverged at lr 1e-3 (CE in the
    thousands).
  - Among the stable runs, aL was best (val CE 0.319 / acc 0.883), then b 0.324, w15 0.337, c 0.343, a 0.376.
  - More data always helped (c4 at 25 % overfits).
- **Decision:** add z-loss, per-model lr, gradient clipping and warm-up. Bigger models are teacher-only until the JS
  budget allows them.

## Phase 2: full control, observation v4 (11-12 11:30 → 16:00), abandoned

Version 4 gave the policy every decision of the seat, including each piece's tile and facing: 3397 inputs,
295 actions, about 28 decisions per prep instead of 7.

### v4 BC (11-12 11:36 → 14:20)
- **Parent:** none.
- **Data:** full-prep teacher, 9.41 M records, 14 400 matches.
- **Change:** BC on xL / dLxs / bx / c4x / … for 12 epochs.
- **Result:** xL epoch 3 had solo HARD 0.292 (= teacher4 0.292) and coop4 HARD 0.365 (teacher4 0.448, about 1.7 SE);
  `curves/baselines.csv`.
- **Variants:**
  - **v4b labels.** A set loss over the pieces legacy will still move, plus final-tile targets. Label accuracy rose
    (move 0.77 → 0.87), play fell: xL solo HARD 0.292 → 0.104 (p 5e-4), dLxs coop4 HARD 0.365 → 0.208 (p 0.003).
    Dropped.
  - **v4c.** ppo2's economy relabelled into v4, economy-only fine-tune. Solo HARD fell to 0.042 (xL) / 0.083 (dLxs):
    the shared trunk moved the placement heads (move accuracy 0.27 → 0.08). Dropped.

### ab-xL / ab-dLxs: first v4 PPO A/B (11-12 13:21 → 14:15)
- **Parent:** BC v4 xL / dLxs at epoch 3.
- **Change:** `--adv-norm group --ent 0.003 --bc-kl 0.01`, 16 × 16, 4e6 steps planned. The first attempts died of GPU
  out-of-memory next to the BC job (`*.oom*` dirs).
- **Why:** pick the v4 architecture.
- **Result:** 48 seeds; both arms below their BC start of 0.292 / 0.365. kl_bc reached 0.87 by 2.5 M while entropy
  rose.

| arm @ steps | solo HARD | coop4 HARD |
|---|---|---|
| xL @ 2.0 M | 0.000 | 0.229 ± 0.061 |
| xL @ 2.56 M | 0.000 | 0.229 ± 0.061 |
| dLxs @ 2.0 M | 0.062 ± 0.035 | 0.188 ± 0.056 |

- **Decision:** the diagnosis was that entropy 0.003 swamped the weak advantage over 112 tile actions, that the start
  was high-entropy, that KL 0.01 was too weak, and that group normalisation amplified noise. Try recipes R1–R3.

### v4-R1 / R2 / R3: recipe arms (11-12 14:25 → 14:40, 1.2 M each)
- **Parent:** BC v4 xL epoch 4 (R1, R2: logits ×2 via `sharpen.py`).
- **Change:**
  - R1: `--ent 0.0005 --bc-kl 0.03`;
  - R2: R1 with `--ent 0`;
  - R3: `--ent 0.0005 --bc-kl 0.05 --bc-kl-min 0.01 --bc-kl-steps 5e6 --adv-norm global`.
- **Result** at 1.2 M, 48 seeds (BC epoch 4: 0.229 / 0.312):

| arm | solo HARD | coop4 HARD |
|---|---|---|
| R1 | 0.250 | 0.292 |
| R2 | 0.271 | 0.292 |
| R3 | 0.229 | 0.229 |

  R1 holds the BC level but does not improve; R2 oscillates; R3 declines. The rising entropy sat in the placement
  and 机变 decisions; the economy stayed flat.
- **Decision:** main run on R1.

### ppo3: v4 main run (11-12 15:04 → 15:59, stopped at 15.2 M)
- **Parent:** BC v4 xL epoch 4, logits ×2.
- **Change:**
  - the R1 recipe with `--rollout 131072 --epochs 2 --lr 5e-5`;
  - band choice: `--band-policy 0.7 --band-warmstart band_table.json`;
  - 60 × 64, about 5.8 k steps/s.
- **Result:** flat for 15 M steps: sampled HARD win about 0.12, fa about 0.40, ABYSS fa about 0.035; kl_bc 0.50.
  The 10 M evaluation hung. Its band seats also carried the `_seat` bug (Phase 3).
- **Decision:** stop full control; switch to the hybrid design.

### v4-EI-A / EI-B: expert iteration v0 (11-12 15:19 → 15:33)
- **Parent:** ppo3 @ 5 M (frozen copy).
- **Change:** B adds `--expert-coef 0.3 --expert-rounds boss`: at boss rounds, the policy's layout and 5 legacy
  variants are simulated, and the best one is the label. A is the plain continuation. Both use the R1 recipe at 12 × 32.
- **Result** at 2 M, 48 seeds:

| arm | solo HARD | coop4 HARD |
|---|---|---|
| A | 0.208 | 0.292 |
| B | 0.271 | 0.354 |

  The difference is not significant, ABYSS is unchanged, and A itself was degrading.
- **Decision:** EI is not worth it on this line. The code lives outside this snapshot.

### v4-DR-A / DR-B: per-round discount (11-12 15:38 → 15:53)
- **Parent:** ppo3 @ 10 M.
- **Change:** B: `--discount-per-round 1 --gamma-round 0.979 --lam-round 0.70`, i.e. γ = λ = 1 inside a prep. With
  28 decisions per prep, λ^28 = 0.24 had shortened the effective horizon about 4×.
- **Result** at 2 M, 48 seeds: solo HARD 0.125 / 0.125, coop4 HARD 0.188 / 0.167. Both below BC.
- **Decision:** no help. Phase 2 closed.

## Phase 3: hybrid model, observation v3 (11-12 16:00 → 11-13 09:30)

The policy does band, economy and 机变; legacy places ([DESIGN.md](DESIGN.md) §1).

### ppo3h (11-12 16:10 → 17:39)
- **Parent:** ppo2 @ 15 M grown to obs v3 (`surgery.py`; new inputs zero-initialised, outputs identical).
- **Change:** `--band-policy 0.7 --band-warmstart band_table.json`, KL 0.01 to the init, 60 processes (3.0–3.3 k
  steps/s).
- **Result 1:** the band head stayed uniform (band CE ≈ ln 38): too few band samples per update.
- **Fix:** at 16:56, from the 6.0 M checkpoint, a band head fitted offline (`bandobs.mjs` + `bandfit.py`) was swapped in.
  The KL anchor became ppo2-v3 with the same fitted head, and Adam was reset.
- **Result 2** at 10 M:

| config | win | fa |
|---|---|---|
| solo HARD | 0.708 ± 0.046 | |
| solo ABYSS | 0.000 | 0.226 |
| coop4 HARD | 0.865 ± 0.035 | |
| coop4 ABYSS | 0.031 ± 0.018 | 0.343 |

  PPO had collapsed the band head onto sarkazb (96 / 96 solo picks). ppo2 had solo ABYSS 4.2 % / fa 0.517.
- **Decision:** freeze the band head and restart from the 6.0 M transplant point. The 6.0 M checkpoint with the fitted
  head ("6.0 M init") is the parent of everything below.

### fb / fr (11-12 17:39 → 19:45, about 15 M)
- **Parent:** 6.0 M init.
- **Change:**
  - fb: `--freeze-band`;
  - fr: fb + `--reward-by '{"ABYSS":{"round":0,"win":1.0}}'`. This was misconfigured: a win of 1.0 is lower than the
    default 2.0.
  - 30 processes each.
- **Result** at 10 M:

| arm | coop4ai ABYSS | solo ABYSS | coop4 ABYSS |
|---|---|---|---|
| fb | 0.354 ± 0.049 | 0.031 ± 0.018 | 0.083 ± 0.028 |
| fr | 0.375 ± 0.049 | 0.062 ± 0.025 | 0.135 ± 0.035 |

  The rolling ABYSS fa sat at 0.33–0.40 from 7.6 M on. **These co-op numbers were later found inflated by the
  `_seat` bug.**
- **Decision:** the owner stopped both at 19:45. A reward redesign followed (R2, [FINDINGS.md](FINDINGS.md) §2).

### m2a / m3a → m2b / m3b (11-12 19:56 → 20:08, discarded)
- **Parent:** 6.0 M init.
- **Change:** `--reward-v 2 --shape-uncapped --shape-stack 0.3 --mix v2 --vf-warmup 20` (m3a adds `--r2 '{"layer":0.002}'`),
  first with the R2 default leak 0.5 / save 0.25 uncapped.
- **Result:** the leak term cost −1.0 to −1.7 per seat-match. It was restarted at 20:00 (m2b / m3b) with
  `"leak":0.15,"save":0.075,"leakCap":true`, then again at 20:08 when the `_seat` bug was found.
- **Decision:** the archives are kept; there are no evaluations.

### Bug: `_seat` lp0 (found 11-12 20:08)
Policy-band seats had lp0 = 1, which gave a raw-LP observation, a 20–45× LP reward and a free +start-LP reward after
the band pick ([FINDINGS.md](FINDINGS.md) §1). It was fixed in the env, and `lpguard.mjs` was added. **Invalidated:**
every `--band-policy` evaluation and training run before 20:08 (ppo3, ppo3h, fb, fr, the 6.0 M init numbers):

| 6.0 M init, coop4ai:ABYSS | win |
|---|---|
| bugged env | 0.354 ± 0.049 |
| fixed env, 96 seeds | 0.135 ± 0.035 |
| fixed env, 192 seeds | 0.130 ± 0.024 |

The LP-scale probe (fixed env, 6.0 M init, coop4ai:ABYSS, 96 seeds) reproduced the bugged result with raw LP:

| k | 1 | 3 | 10 | 25 | raw |
|---|---|---|---|---|---|
| win | 0.135 | 0.083 | 0.146 | 0.344 | 0.354 (fa 0.625) |

The decision was to keep normalised LP and let R2 teach the invest-and-stack regime. A second, same-class bug was
found in the audit that followed: a snapshot start took lp0 from the LP at restore. It was fixed in "env-next" and
deployed at the 21:42 resume.

### m2 / m3: R2 on the fixed env (11-12 20:08 →)
- **Parent:** 6.0 M init; KL anchor ppo2-v3 + fitted head.
- **Change (m2):**
  - `--freeze-band --reward-v 2 --r2 '{"leak":0.15,"save":0.075,"leakCap":true}' --shape-uncapped --shape-stack 0.3`;
  - `--shape-tier 0.06 --mix v2 --band-policy 0.7 --bc-kl 0.01 --snap-start 0.3 --adv-norm global`, 25 × 64.
- **Change (m3):** m2 + `"layer":0.002`.
- **Why:** R2 (team win 3.0, Final Assault team 1.5 / own 0.5, leaks, saves, LP / 28, elimination −0.5, round 0), plus
  rewarded stacking.
- **Result** at 10 M:

| config | m2 | m3 |
|---|---|---|
| coop4ai ABYSS | 0.333 ± 0.048 | 0.427 ± 0.050 (paired Δ +0.094, 95 % CI +0.01 … +0.18) |
| solo ABYSS | 0.156 ± 0.037 | 0.219 ± 0.042 (ppo2: 0.042) |
| coop4 ABYSS | 0.104 ± 0.031 | 0.135 ± 0.035 |

  Compared with the 6.0 M init (k = 1), m3 gains +0.29 on coop4ai:ABYSS (95 % CI +0.19 … +0.40). Both arms learned
  invest-and-stack on normalised LP: shop level from R4, fewer refreshes, fewer leaks, team LP 52 vs 35 at the Final
  Assault.
- **Decision:** keep the layer term.
  - m2 and m3 were resumed at 21:42 from 10 M on env-next.
  - m2 was stopped at 11.5 M (22:06) to make room for m3s.
  - m3 (mix v2) continued as the reference until the owner stopped it at 42.7 M (11-13 07:45).
  - m3's later in-run evaluations: solo ABYSS 0.281 / 0.365 / 0.281 at 20 / 30 / 40 M; coop4ai 0.615 / 0.583 / 0.615.
- **Note:** production rehearsal at m3 @ 10 M (r3b3, 96 seeds): coop4ai 0.500 (+0.073 ± 0.043 over r0b3), solo +0.000.
  Rehearsal helps co-op, not solo.

### Target change (11-12 22:0x)
The owner set the target: pure-model inference wins more than 50 % solo, or as the one model seat next to rule-AI /
human teammates. Headline configs became solo:ABYSS, coop4l:ABYSS and coop2l:ABYSS, with coop4ai secondary, and mix v3
was written for them.

Baselines on the new configs (96 seeds, `curves/baselines.csv`):

| config | m3 @ 10 M (b3 / b0) | teacher, live and ppo2 |
|---|---|---|
| solo | 0.219 / 0.219 | ≤ 0.042 |
| coop4l | 0.135 / 0.094 | ≤ 0.042 |
| coop2l | 0.073 / 0.052 | ≤ 0.042 |

### m3s (11-12 22:06 →, stopped 11-13)
- **Parent:** m3 @ 10.02 M.
- **Change:** `--mix v3` (ABYSS .65: solo .25, coop4l .12, coop4 .08, coop2l .07, coop2 .05, coop4ai .08) and
  `--seed-offset 11000000`. A second in-run evaluation, `--eval2-configs solo:ABYSS,coop4l:ABYSS,coop2l:ABYSS
  --eval2-boss-rehearsal 0`, was added.
- **Why:** train on the target configs. m3 (mix v2) stays as the control.
- **Result:**

| eval | solo ABYSS b3 | solo ABYSS b0 | coop4l b3 | coop2l b3 | coop4ai b3 |
|---|---|---|---|---|---|
| in-run @ 20 M | 0.312 | 0.271 | 0.104 | 0.094 | 0.562 |
| in-run @ 40 M | 0.396 | 0.354 | 0.219 | 0.156 | 0.688 |
| solo480 @ 20 / 30 / 40 M | 0.290 / 0.338 / **0.375** ± 0.022 | 0.246 / 0.287 / **0.310** ± 0.021 | | | |

  m3s @ 20 M against m3 @ 10 M on solo: Δwin +0.094 (95 % CI −0.02 … +0.21), Δfa +0.091 (significant).
- **Decision:** the control arm for S and its descendants.

### P / P0 / KL / ALL / R3: single-factor arms on the m3s line (11-12 22:15 → 11-13 04:08)
- **Parent:** m3 @ 10.02 M, mix v3, the m3s flags otherwise; 25 processes each, on two other large CPU hosts with one
  GPU each.
- **Change:**
  - **P:** `--priv-critic detached --priv-hidden 256 --vf-priv 0.5`, a critic with privileged inputs and its gradient
    detached from the policy trunk;
  - **P0:** P with `--critic-priv-zero` (privileged inputs zeroed: the control for the extra critic);
  - **KL:** `--bc-kl 0.003`;
  - **ALL:** P + KL 0.003;
  - **R3:** `--rehearsal 3` in training (legacy rehearses three layouts at normal rounds).
- **Why:** test each recipe factor against m3s. Arms were sample-matched.
- **Result** at 20 M, paired against m3s @ 20 M on the same 96 seeds (`curves/paired.csv`):

| arm | solo ABYSS Δwin | solo ABYSS Δfa | coop4ai ABYSS Δwin | coop4l ABYSS Δwin |
|---|---|---|---|---|
| P | −0.010 ± 0.054 | **−0.053 ± 0.024** | −0.042 ± 0.051 | −0.031 ± 0.038 |
| P0 | +0.042 ± 0.053 | −0.034 ± 0.025 | −0.073 ± 0.043 | +0.010 ± 0.031 |
| KL | +0.031 ± 0.054 | 0.000 ± 0.022 | +0.021 ± 0.051 | +0.000 ± 0.033 |
| ALL | +0.021 ± 0.044 | −0.019 ± 0.022 | +0.000 ± 0.055 | −0.010 ± 0.038 |

  R3 ran at about 300 steps/s instead of 650.
- **Decision:**
  - R3 was stopped at 10.2 M: the owner prefers a pure-model deploy, and r3 adds +0.000 solo.
  - P, P0 and ALL were stopped at about 04:07 (20–22 M): the privileged critic leans negative on ABYSS.
  - KL continues (28.5 M by 11-13 09:00, no further evaluation yet).
  - The privileged-critic code is not part of this snapshot.

### S: solo-heavy, sample-matched mix (11-12 22:29 →, stopped 11-13)
- **Parent:** m3 @ 10.02 M.
- **Change:** `--mix v3s` (ABYSS .75: solo .45, coop4l .15, coop2l .10, coop4ai / coop4 .02, coop2 .01). This is
  about 55 % solo and 28 % one-RL co-op of the ABYSS samples.
- **Why:** pure-model deploy. Mix v3 under-weights the target configs once samples per match are counted.
- **Result:**
  - In-run @ 20 M against m3s @ 20 M (96 seeds): solo 0.354 vs 0.312, coop4l 0.146 vs 0.104 (mates r0). Promising,
    but see the 288-seed correction below.
  - 288 seeds @ 20 M, mates r3: solo 0.309 ± 0.027 (m3s 0.285), coop4l 0.132 ± 0.020 (0.122), coop2l 0.118 ± 0.019
    (0.097), solo b0 0.267 ± 0.026 (0.226). Paired Δ +0.024 / +0.010 / +0.021 / +0.042, every one < 1.5 SE.
  - solo480 b3: 0.308 / 0.317 / 0.346 at 20 / 25 / 30 M. S − m3s at matched steps: +0.019 ± 0.021 (20 M),
    +0.008 ± 0.022 (30 M).
- **Decision:** S is the default recipe. The mix effect is small and positive.

### Correction: 96 seeds against 288 / 480 seeds (11-13 ~05:00)
The 96-seed S @ 20 M figures (solo 0.354, coop4l 0.219 with mates r3) were lucky draws: on 288 seeds they are 0.309 and
0.132. At 30 M the in-run S − m3s read +0.115 ± 0.055 on solo b3, against +0.008 ± 0.022 on 480 seeds. From here on,
decisions use 288 / 480 paired seeds, and 96-seed evaluations are trend only.

### SC / SK (11-13 04:14 →, stopped 11-13)
- **Parent:** S @ 20.03 M.
- **Change:**
  - **SC:** `--mix v3sc` (ABYSS .78: solo .35, coop4l .25, coop2l .15). Why: heavier one-RL co-op, the binding gap.
  - **SK:** `--bc-kl 0.003`. Why: the KL arm was neutral at 20 M, so retest the looser anchor on the S line.
- **Result** (solo480):

| Δ (paired, 480 seeds) | b3 | b0 |
|---|---|---|
| SC @ 25 M − S @ 25 M | +0.002 ± 0.022 | +0.013 ± 0.020 |
| SK @ 25 M − S @ 25 M | −0.013 ± 0.022 | +0.004 ± 0.020 |
| SK @ 30 M − S @ 30 M | **+0.048 ± 0.024** | +0.027 ± 0.024 |

  - SK @ 30 M is the best checkpoint so far: 0.394 ± 0.022 b3, 0.319 ± 0.021 b0.
  - Its in-run (96 seeds) coop4ai is 0.729 ± 0.045.
- **Decision:** open. The rule was:
  - pick by the best paired solo b3 Δ at matched steps;
  - default to S's recipe;
  - add KL 0.003 only if it is significant.

  SK's +2 SE at 30 M after −0.6 SE at 25 M needs the next matched checkpoint.

### Restart incident (11-13 06:01)
The host's unattended upgrades restarted the m3 and m3s services (needrestart). The launcher restarted both from their
original init (10 M), discarding the 38.0 M / 34.0 M progress for 18 minutes until they were resumed by hand at
06:19 / 06:20 from the newest checkpoints. The launcher now resumes from the newest complete checkpoint on any start
(`scripts/train.sh`, [FINDINGS.md](FINDINGS.md) §8).

### Focus on solo; SO1 (11-13 07:49 →, stopped 11-13)
- **Owner, 07:4x:** focus on solo only, keep training, no from-scratch run. Two large solo arms (≈ 100 processes each)
  were planned and then put on hold until the controls settle the recipe.
- **SO1:**
  - **Parent:** S @ 20.03 M.
  - **Change:** `--mix solo1` (ABYSS .80, of which solo .70; HARD .15; NORMAL .05).
  - **Why:** solo is the target; the small co-op share only guards against forgetting.
  - **Result:** none yet (23.7 M at 09:00).
  - **Decision rule:** prefer solo1 if SO1 ≥ S − 0.5 SE at matched steps.
- m3 was stopped at 42.7 M (07:45). m3s keeps running as the control.

### Zero-extra-information deploy rule and the visible-only switch (11-13 ~09:00 → 09:30)
- **Owner, ~09:00:** the deployed AI may use nothing a player cannot see, and no rehearsal or simulation of any kind.
  The headline moves to solo:ABYSS **b0**, and the final recipe will train at b0.
- **Audit:** five reads beyond a player's view were found (A1–A4, B1; [FINDINGS.md](FINDINGS.md) §7).
- **Owner's ruling:** enemy information, HP and the 机变 kill chance are acceptable; the hidden boss and other players'
  hands are not.
- **Change:**
  - `--visible-only` (A1: human pairs only through the scout view; A3: hidden boss 0 before its round) is always on for
    deploy;
  - `--strict-preview` (A2, A4, B1; `patches/0002`) is optional and off;
  - `infoguard.mjs` checks both;
  - no layout or dataId change.
- **Result:** solo:ABYSS b0, 480 paired seeds, visible-only − plain:

| checkpoint | plain | visible-only | Δ |
|---|---|---|---|
| S @ 30 M | 0.292 | 0.277 | −0.015 ± 0.015 |
| SK @ 30 M | 0.319 | 0.329 | +0.010 ± 0.016 |

- **Decision:** deploy numbers are b0 + visible-only from here on (`scripts/eval.sh` default). Training for
  deployment uses `--boss-rehearsal 0 --visible-only`.

## Side studies of Phases 1–3 (not training runs)

| Date | Study | Result | Decision |
|---|---|---|---|
| 11-12 15:0x | Placement ceiling: ppo2 economy, oracle layouts simulated on the real battle seeds | solo:ABYSS 4.2 % → 8–12 % with placement, 26–30 % with the best band too (O3 30.2 %, CI 22–40 %); coop4 2.1 % → 4–10 % → 10–15 % | Placement is a second lever; band and economy dominate |
| 11-12 afternoon | Economy rollout search (ppo2, solo ABYSS, 170 decisions, about 10 k rollouts) | No candidate class beats greedy; ≤ +0.004 fa per decision (95 %) | No-go |
| 11-12 evening | Opening plans (bond plan biasing purchases) | about +0.03 on fresh seeds (1 of about 60 combinations passed, by chance) | No-go |
| 11-12 evening | Recipe bots (5 community stacking strategies) | none beat legacy; layer p90 120–190 | No-go, not even as teachers |
| 11-12 evening | Positional-trait placement heuristics | slightly worse on model lineups | Dropped |
| 11-12 22:3x | Peek hill-climb K16 from R1, m3 @ 10 M, solo ABYSS, 288 seeds | r0b3 0.181 → K16 0.306 (+0.125 ± 0.023); production r3 0.208 | Teacher only (it reads the live seed); gate for v6 passed |
| 11-12 22:3x | Same, coop4l ABYSS, 288 seeds | r0b3 0.090, r3 all seats 0.132; K16 for the RL seat only +0.01–0.024 over r3 | Most of r3's gain is the mates' own rehearsal |
| 11-13 ~01:xx | Final-Assault-prep joint search (R13–R14), with an honest check that swaps shop RNG and battle seeds | naive +0.188 per R14 game; honest economy-only +0.000 ± 0.017 (R13), −0.006 (R14); placement part +0.075 | No-go for the economy |
| 11-13 ~03:xx | v6 per-piece placement BC from the K16 layouts (about 3 M samples) | economy Δ −0.011; placement −0.080 (288 seeds, p 0.0001); 36 % tile + facing match | Track paused; retry only as a whole-layout pick or DAgger |
| 11-13 early | Product scenarios, m3s @ 20 M | coop4h1 (1 proxy + 3 RL) 0.323; coop2h1 (1 proxy + 1 RL) 0.083 | Proxies die early, so these understate play with humans |
| 11-13 early | 1-RL co-op diagnostic | Final Assault pool × alive players; mates net −0.45 each; co-op ABYSS HP 2.1–2.5× | Structural; levers are game-side ([FINDINGS.md](FINDINGS.md) §6) |

## Phase 4: the elite route and placement (11-13, about 01:00 → 10:00 UTC)

From here on, times are UTC. Evals are b0 (`--boss-rehearsal 0`), greedy, 480 paired seeds unless noted, and they now
report the stacking metrics ([records/final/README.md](../records/final/README.md)). Earlier phases did not log them.
SK @ 30 M (the best checkpoint of Phase 3) became the parent of the next line. The Phase 3 arms (m3s, S, SK, SC, SO1, KL)
were stopped without further matched evals, so the open KL 0.003 and solo1 questions were not settled; the final recipe
keeps KL 0.01.

### LIN and LIN-C
- **LIN.** Parent SK @ 30 M. Change: `--mix solo1` for 30–36 M, then `--mix mp1` (humans + model seats), 90 processes,
  `--boss-rehearsal 0`. Why: train on the product configs at deploy settings.
  - Result: solo:ABYSS 0.333 at 45 M, 0.315 at 62 M (b0, no picker).
- **LIN-C.** Parent LIN @ 54 M. Change: a separate critic trunk (`model2.py`, `py/critic.py`; 2.52 M parameters, the
  critic training-only). Result: +0.025 ± 0.010 pooled over LIN, mostly coop4h1. Adopted. LIN-C final (78 M) with the
  boss picker: solo 0.477, coop4h1 0.510.
- Baseline SK @ 30 M against legacy fill (every model seat replaced by legacy):

| config | SK @ 30 M | legacy fill |
|---|---|---|
| solo:ABYSS | 0.319 | 0.010 |
| coop4h1 | 0.358 | 0.071 |
| coop4h2 | 0.206 | 0.033 |
| coop4h3 | 0.085 | 0.033 |
| coop2h1 | 0.056 | 0.017 |

### Boss-round picker in the loop
- **Change:** at the boss rounds, a learned head scores legacy's K8 layouts (`planLayout`, no battle built) and plays
  the best. Labels: fixed-seed rehearsal of each candidate; joint head, leak weight 2 (`--picker-head` /
  `--picker-trunk`, `picker.mjs`, `py/picker_train.py`).
- **Result:** on SK @ 30 M, solo +0.123 (0.442), coop4h1 +0.109 (0.474). A stale (not refitted) head on LIN 58 M keeps
  85–90 % of the gain. K10 / K12 add ≤ 1 point, so K8 stays. 35–45 ms per boss prep in JS.
- **Decision:** the boss picker joins every eval and, in NAT, the training loop.

### Width and structure A/B
- **Change:** three arms from the same checkpoint, +15 M steps each: W2 = 2× width (5.8 M parameters, `py/widen.py`), AT
  = entity attention (`infer2at.mjs`), W1 = control.
- **Result** at +15 M (55.03 M, 480 + 480 paired seeds, b0, no picker): W1 0.325 / 0.429, W2 0.333 / 0.458, AT 0.323 /
  0.421 (solo / coop4h1). W2 − W1 +0.019 ± 0.015, the gap growing (+0.002 / +0.004 / +0.019 at +5 / +10 / +15 M).
  AT − W1 −0.005 ± 0.015.
- **Decision:** AT closed. The W2 extension was not run (handover); width was not carried into the final recipe.

### Stacking diagnosis
The owner set a hard requirement here (later relaxed to a trend, see Phase 5): the AI must play a national main bond at
tier ≥ 2 plus a stacked secondary class bond, not the elite-merge route.

| seat | national main at end | national tier ≥ 2 | Σ layers | elites | win |
|---|---|---|---|---|---|
| model (SK / LIN line) | 4 % | 0 % | ~380 | ~6 | 0.32 |
| legacy fill | 90 % | 78 % (reached at R9–R12) | 186 | 2.4 | 0.01 |

The models' main bonds were 坚守 / 精准. Causes found: national tier 2 needs 6 distinct units of one nation and pays
only late; the class route is 1.7–2× cheaper per layer; the R2 stacking shaping is route-agnostic; exploration at
entropy 0.003 never finds the national route; legacy placement is blind to in-battle layer engines.

### NAT: the national push on the elite line
- **Parent:** LIN-C final (78 M). **Change:** `--nat-plan '{"phase":"forced"}'` for 78 → 81 M, then `"free"` to 88 M;
  `--mix mp2`, the boss picker in the loop, `--bc-kl-from 78M --bc-kl-steps 5e6` (KL decaying to 0), late-lr flags.
- **Result** (b0, boss picker, 480 seeds):

| step | solo | coop4h1 | natMainT2 |
|---|---|---|---|
| LIN-C final | 0.477 | 0.510 | 0 |
| 81.0 M | 0.473 | 0.527 | 0 |
| 84.1 M | 0.500 | 0.560 | 0 |
| **88.0 M** | **0.596** | **0.581** | **0** |

  National main 6–9 %, national tier 2 at 0 %, Σ layers 320–390, about 6 elites.
- **Decision:** the best win so far, on the elite route. Not acceptable under the requirement; kept as the win reference.
  The win likely came from mp2, the KL decay and the picker in the loop, not from the national push (no ablation).

### Placement ceilings (teacher-only, live seed)
- Oracle placement on LIN-C 74 M, 96 seeds, solo: none 0.292, every round 0.625, boss rounds only 0.490, normal rounds
  only 0.385 (+47 layers). coop4h1 0.490 → 0.677. Placement is a much bigger lever than the 11-12 study suggested.
- Normal-round ablation (LIN-C 74 M, normal rounds only, 96 seeds): K8 legacy candidates, layer scoring, fixed seed
  ("D") +0.073 ± 0.031 solo / +0.062 ± 0.049 coop4h1; wider candidates, outcome scoring or the live seed add only
  0.01–0.03. **Decision:** build a learned normal-round picker from D (Phase 6).

### Other
- **Old scratch run SCR** (fresh BC → PPO): killed at 20 M (HARD 0.625, ABYSS 0.000). Causes: weak BC start, a KL decay
  coinciding with a mix switch, almost no ABYSS win signal, entropy collapse.
- Closed: an equipment-pick ceiling (+0.015 solo), bounty-round placement (≈ 0), a per-seat band-table refit (−0.125 on
  coop4h1: seats pile onto the same bands), proxy handicaps (none that matters).

## Phase 5: the national route (11-13 10:00 → 21:15 UTC)

### Diagnosis: the BC start loses tier 2
- solo:HARD, 192 seeds: legacy keeps national tier 2 (natMainT2 0.85, win 0.333); the BC start "bc-fresh" drops it to
  about 0 (natMainT2 0.00–0.03, win 0.14), under greedy and sampled play and either band policy.
- First reading: legacy follows a hidden focus bond the observation does not show. A **plan block** in the observation
  (hindsight plan labels, data-plan1) lifted the BC's agreement on "buy a new member of the eventual nation" from 0.21
  to 0.88.
- **Correction:** bc-fresh was under-trained (about 4–6 epochs). A plain MLP BC trained to convergence (40 epochs, CE
  0.532) reaches natMainT2 0.81; a v5 BC 0.84. Plan features still add about +0.18 agreement on commitment buys and a
  controllable handle. From then on every BC trains 20–40 epochs to the best validation CE. A reproduction found
  bc-fresh itself to be a lucky draw (three 4-epoch reruns: CE 0.79–0.81, HARD 0.146 each vs 0.177).

### Route economics (no RL)
- Shop Monte Carlo (2000 matches per cell, solo ABYSS, full commitment): P(tier 2 by R12) 0.94–0.97 for 叙拉古 / 阿戈尔 /
  萨尔贡 / 谢拉格 / 卡西米尔, 0.82 维多利亚, 0.72 拉特兰, 0.49 炎. Half commitment is far too slow (≤ 0.15).
- Controlled Final Assault boards (48 paired seeds): tier 2 is where the value is (about +0.2 pool share, against +0.03
  per unit before it). Best pairs: 卡西米尔 + 突袭 0.49, 谢拉格 + 奥术 0.42, 拉特兰 + 精准 0.42.
- Full-match route bots (480 seeds, r0): the national bot wins HARD 0.608 (legacy 0.388, class bot 0.223) and reaches tier
  2 by R14 in 95 % of matches. Ranked by 0.5 HARD win + 0.3 ABYSS FA share + 0.2 controlled tier-2 win, the nations
  order 拉特兰 1.0, 维多利亚 0.93, 谢拉格 0.83, 卡西米尔 0.82, 炎 0.66, 萨尔贡 0.52, 叙拉古 0.40, 阿戈尔 0.21. This is the
  `assign: "auto"` order and `trackm/recipes-M.json`. The national bot became the BC teacher.

### Plan-block arms (solo:HARD, 5 M from a plan BC, 192 seeds)
- **NR / NL** (random / lead plan, KL to the plan BC): NL 0.484 (natMainT2 0.73), NR 0.333. Scratch-recipe baselines
  without a plan: 0.68–0.75 but natMainT2 0.02. Tier 2 passes, win fails. Two bugs found and fixed on the way: plans
  included disabled nations (→ `live`), and a plan head leaked the hindsight plan (→ `planFrom`, `blind`).
- **NH** (a learned plan head): 0.641, natMainT2 0.99. NL10 (10 M) 0.635.
- **NM** (BC "PB5" on the national-teacher data + `{slots, live, assign auto, band, focus, pairS}` + the plan reward
  terms, lr 3e-4): **0.797** (+0.052 ± 0.036 vs the best scratch baseline), natMain 1.00, natMainT2 1.00, layers
  121 / 68 / 223, elites 4.3. A no-plan control (MLP40 BC) reached 0.766 with natMainT2 drifting 0.81 → 0.50.
- **NMA** = NM 5 M → 30 M with `--mix auto1` (the ABYSS curriculum, §8 of DESIGN). solo:ABYSS 0.296 at 12 M, 0.382 at
  20 M (natMainT2 0.92); coop4h1 0.338 at 20 M.
- **NH2 / NH2s** (a plan head at the band decision; NH2s adds a secondary push: secondary bonus 1.0 and dense pays):
  0.818 / 0.781 at 5 M. NH2s has more secondary layers (87 vs 57), NH2s − NH2 −0.036 ± 0.036. **NH2sA** = NH2s → 30 M
  on auto1; **NH2sC** = NH2s → mp2 with teammate proxies and `coopReplan`.

### NMA against NH2sA at 30 M (480 seeds per cell)
none = no pickers; ens = boss scorer + boss picker ensemble (xboss-v1); ensP = ens + the spatial normal picker S80.

| arm | config | none | ens | ensP | natMainT2 | main / sec / Σ (ensP) |
|---|---|---|---|---|---|---|
| NMA | solo:ABYSS | 0.504 | 0.613 | **0.653** | 0.94–0.95 | 158 / 72 / 276 |
| NMA | coop4h1 | 0.382 | 0.454 | **0.492** | 0.68–0.70 | 136 / 56 / 221 |
| NH2sA | solo:ABYSS | 0.553 | 0.576 | 0.590 | 0.97–0.98 | 176 / 123 / 405 |
| NH2sA | coop4h1 | 0.347 | 0.399 | 0.460 | 0.67–0.68 | 134 / 71 / 247 |

- NH2sA − NMA, solo / coop: none +0.048 ± 0.030 / −0.036 ± 0.026; ensP −0.063 ± 0.029 / −0.032 ± 0.028.
- An earlier read at 20 M ("NH2sA dominates NMA on solo", +0.071 ± 0.031 without pickers) did not hold with the
  pickers.
- **Decision:** the final recipe is NMA: it wins more in the deployed stack (with pickers), and it is simpler (a rule
  plan, no plan head, band policy 0). The pickers were fitted on NMA-type boards too, which favours it in this
  comparison. The owner approved "the NMA recipe to 60 M, MLP and v5, two seeds" at about 20:00.
- The owner relaxed the requirement at about 17:05: the national main + stacked secondary style only needs to be broadly
  followed, and win rate decides between candidates. Stacking metrics stay next to every win.

### Reproducibility (from-scratch pipeline)
- Deterministic BC (`train_bc.py --seed --deterministic --data-cpu`): bit-identical weights across reruns. v5 BC
  through `rl/final/v5/detrun.py` (torch deterministic algorithms): identical tensors over three runs.
- NM reproduced from scratch on a second host: data re-recorded byte-identical (224 / 224 files), BC 40 epochs (CE 0.329,
  BC win 0.682), RL 0.786 / 0.781 on two seeds against NM 0.797 (both within ±0.03). NH2s reproduced likewise (−0.008
  ± 0.029 over two seeds).
- One-factor ablations on NM (480 seeds pooled, none adopted): plan-buy prior +0.000 ± 0.014, secondary pay ×2 +0.006,
  economy masks +0.005 (192 seeds), per-round γ −0.033 ± 0.019.
- PPO learning rate: 3e-4 learns faster than 1e-4 on the MLP (rolling HARD win at 1.6 M 0.38 vs 0.25); kept 3e-4.
- Negative: BC on national-tier-2 trajectories only (natMainT2 0.01); forcing the plan in RL (forced p = 1 collapses
  unforced play; nothing remains once p → 0).

### Architecture: v5 and memory
- **v5 BC** on the same data: val CE 0.425 (8 epochs) vs the MLP's 0.532 at 40 epochs; init solo:HARD win 0.365 vs
  0.250 (96 seeds).
- **Wave 1** (5 M, scratch recipe, 2 seeds each, 480 seeds): MLP from bc-fresh 0.758, MLP from a converged BC 0.735, v5
  0.733. No win difference; v5 keeps national tier 2 on about 70 % of seats (MLP-b 32 %).
- **Recurrent arms** (LSTM / GRU 128, zero-init residual branch, `--seq-len 32 --burn-in 16`): 0.621 / 0.627, −0.11 vs
  v5. The branch stayed unused (zeroing it moves the policy by KL 0.0005); the loss is the cost of the sequence
  training (fewer, slower updates). No evidence that memory helps; not pursued.
- **V5P** (plan token, NH recipe, lr 1e-4): +0.031 ± 0.025 vs NH (one seed each). **V5Pbig** (d192 × 3): −0.079 ±
  0.023, learns slower.
- **v5 learning rate** (NH2s recipe, 480 seeds): at lr 3e-4 v5 is −0.036 ± 0.018 vs NH2s (KL to BC 0.46–0.60 at 2 M vs
  0.19: it drifts fast); at **lr 1e-4** it is **+0.035 ± 0.017** (three seeds: 0.823 / 0.821 / 0.823). The final v5
  trains at 1e-4.
- **History features** (`obsExt.hist`: the seat's last battle outcomes and shop history): V5H +0.006 ± 0.020 without
  the plan; V5PH (plan + hist) −0.022 ± 0.013 vs V5P6L. Not additive with the plan; dropped.
- **v5 on the NMA recipe** (5 M, 480 seeds): V5NM 0.795 (two seeds) vs NM 0.808, −0.014 ± 0.019. No edge at 5 M on this
  recipe. Both architectures went into the final runs.

### Throughput
Eight engine patches (memo, deopt and allocation fixes) plus V8 tiering flags: +8.2 % environment steps per second,
byte-identical outputs on solo and long co-op cases. Because the outputs are identical, no result depends on them.

## Phase 6: the deploy pickers (11-13 10:05 → 21:15 UTC)

Both pickers replace legacy's layout choice at decision time with a learned score over rule-generated candidates. No
battle is built at decision time. Design: [DESIGN.md](DESIGN.md) §9.

### Normal rounds (rounds 1–13)
- **Label rule.** The ablation's rule D (no extra leaks, max Σ layers gained, ties to most kills) on K8 legacy
  candidates: +0.073 ± 0.017 solo / +0.075 ± 0.020 coop4h1 on LIN-C 74 M (480 seeds, legacy boss rounds). An earlier
  96-seed read of a layer-pick rule (−0.062) was a bad draw (+0.017 ± 0.017 on 480).
- **Summary-feature heads** (N1 / N2 / v3 / a set encoder V4b, 89 k labels): pooled +0.007 / +0.012 / −0.001. They kept
  4–14 % of the label's layer gain; val R² ≤ 0. Closed.
- **Spatial picker** (a v5 tower over each candidate's board): 40 k labels +0.008 ± 0.009 pooled; S80 (80 k) +0.032 ±
  0.009; **Sall** (about 300 k labels, with DAgger states) +0.054 ± 0.009, oracle +0.101 ± 0.010. Sall became the default.
- **Bug: label seeds.** Rehearsal seeds had no match term, so the first 196 k labels shared about 52 random streams.
  Mild impact (a fixed-seed pick keeps 92–95 % of its gain under the live seed); fixed with per-match seeds.
- **Candidate pool** (templates, last board, one-step edits; about 21.5 per decision): the best candidate is non-legacy
  24–35 % of the time; the pool's best adds +1.4 to +3.5 layers over K8's best. Sall, trained on K8 only, does not gain
  from the pool (pool − K8 −0.005 ± 0.011).
- **c8** (refit on NH2sA labels): +0.005 ± 0.009 over Sall. Not adopted.
- **c9** (trained on 41 k pool labels): offline utility 1.32 vs Sall-on-pool 1.17 and Sall-on-K8 0.90. In game (both
  arms on xboss-v3):

| board | config | Sall + v3 | c9 + v3 | Δ (paired) | n |
|---|---|---|---|---|---|
| NMA 30 M | solo:ABYSS | 0.674 | 0.680 | +0.005 ± 0.014 | 952 (2 seed sets) |
| NMA 30 M | coop4h1 | 0.537 | 0.571 | **+0.035 ± 0.017** | 952 |
| NH2sA 30 M | solo:ABYSS | 0.645 | 0.681 | +0.036 ± 0.020 | 476 |
| NH2sA 30 M | coop4h1 | 0.483 | 0.515 | +0.032 ± 0.024 | 476 |

  All 2,856 pairs +0.025 ± 0.009; main layers +9 to +28, Σ layers +11 to +37, natMain and elites unchanged. Released as
  the default normal-round picker at 21:15. A DAgger refit (c10) was started and not finished.

### Boss rounds (Final Assault, hidden core)
- **Search labels.** Hill-climb on fixed rehearsal seeds: leader damage +35–49 %, boss leaks 0.87 → 0.01 per decision.
- **QNet scorers over legacy's K12 / K8.** q2 boss-only +0.083 ± 0.038 vs legacy (LIN-C, 96 seeds), below the boss
  picker; q3 / q4b about equal to the picker. **xboss-v1** = q4b + picker ensemble: +0.012 ± 0.006 over the picker. Q
  hill-climb and random perturbations as extra candidates hurt an unadapted scorer.
- **Bug: label seeds again.** Every solo boss label had been scored on the same 3 rehearsal seeds; fixed with salted
  per-match seeds (`xRehearseForTraining`). The honest re-check of the generator matched the old one.
- **Generator g3b** (imitation + one AWR step): +0.040 ± 0.017 over legacy in game, the first direct generator to beat
  legacy; later AWR steps diverged.
- **Lineup ceiling.** Placement oracle on legacy's lineup 0.533 vs 0.360; a lineup + placement oracle +0.060 ± 0.024
  more. Left unexploited.
- **xboss-v2** (scorer q5) +0.007 ± 0.004 over v1. **q8d** (recipe D: from scratch, more labels): offline +0.6 per
  decision, in game +0.006 ± 0.004. Offline gains on K8 bought little.
- **Teacher bake-off** (hill-climb / CEM / PUCT at 64 and 256 simulations, re-scored on fresh seeds): equal within
  ±0.25; hill-climb 256 beats the best K8 candidate by +3.4–3.5 utility. Teacher = hill-climb 256.
- **Win utility.** Only the round-14 pick moves the win rate. P(win | rehearsed FA share) is steep around share 0.85;
  the earlier utilities over-weighted leaks (fitted: −0.1 per leak vs +11.7 per unit of share). Re-scored with it, the
  in-game order of earlier scorers was reproduced.
- **xboss-v3** = Dw (recipe D on the win utility, including pool labels) over the deploy pool (476 paired seeds, Sall
  normal picker in every arm):

| board | config | v2 (q5 ens) | q8d | Dw on K8 | **Dw on pool** | Dw pool − v2 |
|---|---|---|---|---|---|---|
| NMA 30 M | solo:ABYSS | 0.658 | 0.672 | 0.662 | **0.685** | +0.027 ± 0.013 |
| NMA 30 M | coop4h1 | 0.521 | 0.534 | 0.523 | **0.534** | +0.013 ± 0.010 |
| NH2sA 30 M | solo:ABYSS | 0.588 | 0.595 | 0.607 | **0.645** | +0.057 ± 0.018 |
| NH2sA 30 M | coop4h1 | 0.464 | 0.464 | 0.471 | **0.483** | +0.019 ± 0.012 |

  Pooled +0.029 ± 0.007 over v2; all of it from the pool (Dw pool − Dw K8 +0.021 ± 0.006). Stacking identical across
  arms. Released at 19:35.

### The pickers on the final models
Setting b (c9 + xboss-v3) − setting a (no pickers), same checkpoint, 476 paired seeds: +0.13 to +0.20 on every final
checkpoint, solo and co-op (`records/final/milestones.csv`, column `lift_b_minus_a`). Stacking rises with the pickers
(Σ layers +30 to +51), natMain is unchanged.

## Phase 7: co-op robustness and the co-op stage (11-13 11:40 → 11-14 04:15 UTC)

### Teammate proxies and the robustness suite
- A pool of scripted teammates (`proxies.mjs`) and a suite of cells (legacy r3 / r0, national, class, recipe, noisy,
  each nation forced live). Measured as the "hole": Δ vs the model's own legacy-r3 result on the same seeds.
- **Elite-route models** (NAT88, LIN-C): a 谢拉格 human costs −0.212 ± 0.031 / −0.271 ± 0.032. The yield rule hands the
  human's nation to the human, and the elite builds held 2.3–2.5 谢拉格 operators per seat.
- **National models** (NH2sA, NMA): robust to the teammate type (all Δ within ±0.04 for NH2sA). Their weakness is the
  co-op level itself (0.35–0.38 without pickers).
- **NMA nation cells** (480 seeds, no pickers): legacy-r3 0.383; 谢拉格 live 0.299 (hole −0.121 ± 0.030), 拉特兰 live
  0.371 (−0.089 ± 0.034); collision 0.097 / 0.168 / 0.247. A human taking a carry nation is NMA's hole.
- **Replanning on collision** (`coopReplan`), as a trained arm (NH2sC) and as an eval-time rule on NMA (NMArp):
  collisions drop about 10×, win drops in every cell (NMArp legacy-r3 −0.044 ± 0.018, 拉特兰 −0.055 ± 0.025).
  Rejected both times.
- Real-player-weighted co-op (proxy cells weighted by the nation mix of 126 real matches, aggregates only): NMA 0.367 ±
  0.010, NH2sA 0.354 ± 0.010. Lift over legacy fill (96 seeds, NMA): coop4h1 +0.296, coop4h2 +0.186, coop2h1 +0.131.

### Co-op fine-tune arms (10 M from NMA 30 M, KL to the start)
- **C1** (from NH2sA 30 M, replan on, mixed proxy pool, lr 3e-4): coop +0.069 ± 0.026 at 10 M, solo +0.013.
- **CA** = C1's recipe on NMA (replan on, `nat:rand` heavy pool). **CB** = no replan; half the proxies ask for a carry
  nation (`nat:lateranoShip`, `nat:kjeragShip`). Both `--mix mp2 --band-policy 0`, KL anchor = the start, `--micro 2048`
  after two out-of-memory starts.

| arm @ 10 M | solo | coop legacy-r3 | 谢拉格 live | 拉特兰 live | coop natMain / T2 / Σ / elites | collision / replans |
|---|---|---|---|---|---|---|
| CA | +0.013 ± 0.028 | +0.067 ± 0.025 | +0.028 | +0.021 | 0.95 / 0.82 / 238 / 5.7 | 0.005 / 0.74 |
| CB | +0.010 ± 0.028 | +0.073 ± 0.025 | +0.065 | +0.034 | 0.89 / 0.75 / 247 / 5.5 | 0.091 / 0 |

- At 5 M neither passed; CA's 拉特兰 cell was −0.076 there.
- **Decision:** CB. It is ≥ CA in every co-op cell and needs no plan-rule change. Real-player-weighted coop4h1 0.367 →
  0.427. With the pickers (setting b, 476 seeds), CB10 vs NMA 30 M: solo 0.687 (+0.008 ± 0.026), coop 0.620 (+0.080 ±
  0.028).
- **Validation on the final runs' 30 M checkpoints** (setting b, vs the 30 M start):

| run | b solo | b coop | a solo | a coop |
|---|---|---|---|---|
| CM2 = FIN-MLP-s1 30 M + CB | −0.040 ± 0.027 | **+0.118 ± 0.026** | +0.025 | +0.075 |
| CV5 = FIN-V5-s1 30 M + CB, lr 1e-4 | +0.046 ± 0.027 | −0.027 ± 0.026 | +0.046 | +0.006 |
| CV5b = CV5 with lr 3e-4 | −0.017 ± 0.029 | −0.048 ± 0.027 | −0.029 | −0.023 |

- **Decision:** the co-op stage applies to the MLP. On v5 it failed at both learning rates. CM2's solo dip with pickers
  (−0.040) was flagged as a risk for the 70 M gate.

## Phase 8: the final runs (11-13 21:35 UTC →)

Recipe: the NMA recipe to 60 M, MLP and v5, seeds 1 and 2 (`rl/final/`). Seed 1 ran on an 8× H200 host (one GPU and 56
threads per run), seed 2 on one RTX 4090 (both architectures sharing it, 32 + 64 threads). Gates before launch: the training tree reproduces the NMA 30 M
reference evals per match, and the eval tree reproduces the c9 + xboss-v3 reference (476 / 476 matches each).

### Milestones (setting b unless noted; 476 paired seeds; Δ vs NMA 30 M on the same setting)

| checkpoint | b solo | b coop | a solo | a coop | lift solo / coop | natMain / T2 solo | natMain / T2 coop |
|---|---|---|---|---|---|---|---|
| NMA 30 M (reference) | 0.679 | 0.540 | 0.504 | 0.382 | +0.175 / +0.158 | 0.99 / 0.96 | 0.90 / 0.73 |
| FIN-MLP-s1 30 M | 0.622 | 0.485 | 0.429 | 0.355 | +0.193 / +0.130 | 0.97 / 0.94 | 0.89 / 0.69 |
| FIN-MLP-s2 30 M | 0.632 | 0.483 | 0.431 | 0.309 | +0.202 / +0.174 | 0.98 / 0.94 | 0.85 / 0.60 |
| FIN-V5-s1 30 M | 0.605 | 0.580 | 0.452 | 0.382 | +0.153 / +0.197 | 0.99 / 0.96 | 0.83 / 0.70 |
| FIN-V5-s2 30 M | 0.586 | 0.561 | 0.416 | 0.387 | +0.170 / +0.174 | 0.96 / 0.93 | 0.84 / 0.69 |
| FIN-MLP-s1 60 M | 0.708 | 0.618 | 0.544 | 0.489 | +0.164 / +0.128 | 0.99 / 0.96 | 0.90 / 0.75 |
| FIN-MLP-s2 60 M | 0.742 | 0.601 | 0.569 | 0.424 | +0.172 / +0.176 | 0.99 / 0.96 | 0.88 / 0.66 |
| FIN-V5-s1 60 M | 0.704 | 0.687 | 0.561 | 0.511 | +0.143 / +0.176 | 0.99 / 0.97 | 0.87 / 0.71 |
| FIN-V5-s2 60 M | 0.754 | 0.662 | 0.592 | 0.511 | +0.162 / +0.151 | 0.98 / 0.96 | 0.88 / 0.75 |

- At 30 M every run trailed NMA 30 M on solo by 0.05–0.09, while still rising; NMA 30 M looks like an above-average seed.
- No drift toward the elite route at any milestone: natMain 0.96–0.99 solo, Σ layers above NMA's, elites 4.6–6.2.
- The full table, with every 40 / 50 M milestone and the stacking columns: `records/final/milestones.csv`.

### Extension (60 M → 90 M)
The owner approved continuing runs that were still rising ("you decide"). Rule ([rl/final/README.md](../rl/final/README.md)):
S1 = paired Δ (60 M − 50 M) of the setting-b mean; S2 = rolling ABYSS training win, 55–60 M minus 45–50 M; extend if
either ≥ +0.02; at most two runs (one MLP and one v5 may each take a slot).

| run | S1 | S2 | 60 M b mean | extended |
|---|---|---|---|---|
| FIN-MLP-s1 | −0.014 ± 0.019 | +0.007 | 0.663 | no |
| FIN-MLP-s2 | −0.005 ± 0.018 | +0.027 | 0.672 | yes (FIN-MLPX-s2) |
| FIN-V5-s1 | +0.012 ± 0.019 | +0.027 | 0.696 | first no (V5-s2 had the higher mean), then yes after a rule change (FIN-V5X-s1) |
| FIN-V5-s2 | +0.045 ± 0.019 | +0.041 | 0.708 | yes (FIN-V5X-s2) |

- Continuation: weights only (fresh Adam with the resume warm-up), `--auto-smin 0.85` pins the curriculum at its 60 M
  state, the seed offset moves by +500,000 so no match seed is replayed. Continuity checked on the first updates
  (KL, clip fraction, explained variance, sps against the 55–60 M updates).
- Rule change (11-14 04:56): the segment's S2 is measured within the extension (its last 5 M against the previous 5 M),
  S1 decides the first segment, and a run stops at the next milestone if its segment gains < +0.01 on both. MLPX-s2's
  within-extension S2 was −0.002.

### Co-op stage on the 60 M checkpoints
- **FIN-MLP-s2**: 70 M vs its own 60 M, setting b solo −0.019 ± 0.026, coop **+0.101 ± 0.027**; setting a +0.025 /
  +0.084. Adopted by the gate (the solo dip is at the edge, the same pattern as CM2). Stacking at 70 M (b coop): natMain
  0.91, natMainT2 0.81, Σ 296, elites 5.6 (60 M: 0.88 / 0.66 / 253 / 4.9).
- FIN-MLP-s1, FIN-V5-s1 (lr 1e-4) and FIN-V5-s2: see `records/final/coopgate.csv`.

<!-- pick:BEGIN heldout -->
### Held-out confirmation (`--seed 200000000`, setting b, 476 + 476 seeds)

The leading candidates on the milestone seeds went to the held-out set as the runs finished, 11 in all (CB10
included). From 06:55 the rule was: the current pick stays unless a candidate beats it there, paired (paired mean Δ >
0, both ≥ 0.50). Final table: [RESULTS.md](RESULTS.md) A1 and `records/final/heldout-paired.md`.

- Selection bias again: FIN-V5X-s1-80M, 1st on the milestone seeds (0.805 / 0.712), read 0.756 / 0.685; FIN-V5-s2-60M
  0.754 / 0.662 → 0.723 / 0.639. FIN-MLP-s2-70M went the other way: 10th (0.723 / 0.702) → 0.784 / 0.737.
<!-- pick:END heldout -->

### Final pick

<!-- pick:BEGIN pick -->
- **Extensions:** FIN-MLPX-s2 stopped at 80 M (S1 −0.041; best milestone 70 M). FIN-V5X-s2 and FIN-V5X-s1 ran to 90 M
  (V5X-s1: S1 +0.052 / +0.012 / −0.002 at 70 / 80 / 90 M). XC, the co-op stage on FIN-MLPX-s2 70 M, failed its gate
  (setting b solo −0.052 ± 0.024).
- **Co-op stage:** adopted on both MLP seeds. On v5 it failed (V5-s2) or barely moved co-op (V5-s1, setting b co-op
  +0.011), so the extended v5 runs did not take it (`records/final/coopgate.csv`).
- **By the win rule (09:00):** FIN-V5X-s1-90M. It beat FIN-MLP-s2-70M on held-out, paired (Δ mean +0.018 ± 0.017; solo
  +0.004 ± 0.024, co-op +0.032 ± 0.023), led on the milestone seeds too, and pooled +0.031 ± 0.012.
- **Released (the owner's choice): FIN-MLP-s2-70M**, held-out 0.784 / 0.737. It was chosen for its inference cost. The
  v5 run FIN-V5X-s1-90M scored a little higher on the fresh seeds (0.790 / 0.744; paired Δ mean +0.017 ± 0.018, about
  one standard error; held-out +0.018 ± 0.017), but its policy costs 23 ms per decision in JS against 1.8 ms for the
  MLP.
- Stacking of FIN-MLP-s2-70M, held-out (solo / co-op): natMain 0.99 / 0.91, natMainT2 0.97 / 0.81, Σ layers 331 / 295,
  elites 5.7 / 5.6; main bonds 拉特兰 0.63, 维多利亚 0.26, secondaries 灵巧 0.46, 坚守 0.22, 精准 0.20.
- **Confirmation on a fresh seed set** (`--seed 300000000`, same protocol): FIN-MLP-s2-70M 0.769 / 0.731,
  FIN-V5X-s1-90M 0.790 / 0.744; paired Δ mean +0.017 ± 0.018, the same gap as held-out. Real-player-weighted coop4h1
  0.702 ± 0.010 vs 0.713 ± 0.010. With a 谢拉格 human the MLP leads (0.652 vs 0.619, likely from its co-op stage), with a
  拉特兰 human they tie (0.645) ([RESULTS.md](RESULTS.md) A1b, A1c).
<!-- pick:END pick -->
