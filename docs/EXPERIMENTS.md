# Experiment log

This is every training run and side study in order, failures included. Times are UTC+8 in November 2026 (11-11 to
11-13).

For each run the entry gives:

- **parent**: the checkpoint the run starts from, and its step;
- **change**: the single factor changed against the run it is compared with, as exact flags;
- **why**: the reason it was tried;
- **result**: team win, paired seeds, ± one SE;
- **decision**: what was done next.

Where nothing else is said, evaluation is greedy, r0b3, 96 seeds. Steps count RL-seat decisions. The full flags of
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

## Phase 3: hybrid model, observation v3 (11-12 16:00 → now)

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

### m3s (11-12 22:06 →, running)
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

### S: solo-heavy, sample-matched mix (11-12 22:29 →, running)
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

### SC / SK (11-13 04:14 →, running)
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

### Focus on solo; SO1 (11-13 07:49 →, running)
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

## Side studies (not training runs)

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
