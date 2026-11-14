# Design

This document describes the system of the final models (11-14): the national-plan policy (MLP or v5), the two deploy
pickers and the co-op stage. Earlier designs (ppo1 / ppo2, the version 4 full-control attempt, the hybrid line m3 → S / SK,
the elite-route arms LIN / LIN-C / NAT) appear only where they explain a choice. For the history, see
[EXPERIMENTS.md](EXPERIMENTS.md). The exact arguments of every final stage are in [rl/final/README.md](../rl/final/README.md).

## 1. What the model controls

The model drives AI seats ("model-AI seats") in ordinary matches. The legacy rule AI (`server/match/bot.js`) keeps the
rest of the seat's turn.

| Decision | Who | Notes |
|---|---|---|
| Opening band (分队 / strategy) | rule, from the national plan | `natPlan.band`: when the plan fixes a nation, the seat opens with that nation's band. The policy's band head is not used (`--band-policy 0`). Masks forbid a band whose bond clashes with a human's band or announced main bond. |
| National plan (P, S) | rule | §3.1. Fixed at the seat's first decision after the band. |
| Prep economy: buy, sell, refresh, level up, freeze | policy | one flat action space, masked per decision |
| Merge rewards, personal choices, the 机变 draft card | policy | |
| Normal-round layout (rounds 1–13) | picker **c9** | §9.1; legacy's layout when the picker does not switch |
| Boss-round layout (Final Assault, hidden core) | picker **xboss-v3** | §9.2 |
| Lineup, equipment, temp / hand resolution, Ready | legacy | `LAYOUT_ONLY` steps of `createBotPrepSteps` (legacy with its procurement switched off) |

The AI-seat rules of the game's owner hold as masks: 让主盟约 (`yieldMain`: never buy, or take as a reward, an operator of a
living same-pool human's main core bond or opening-strategy bonds) and 让合成 (`yieldMerge`: never take the third copy
of an operator a human holds two of). Training marks 0..n−1 teammates as humans at random (`humans: 'random'`), so the
masks are always exercised.

Deploy is pure model. The policy and both pickers run no search, no simulation, no layout rehearsal and no live seed.
Every picker decision counts the battles it built (`sims`); the count is 0 in every reported eval. Search and simulation
are used only at training time, as teachers and label generators.

The version 4 design let the policy take every decision of the seat, placement included (`obsVersion: 4`, 3397
inputs, 295 actions). It is still in `spec.mjs` / `env.mjs`, but PPO did not learn on it ([FINDINGS.md](FINDINGS.md) §9).
Placement came back later as separate learned pickers trained on search labels (§9).

## 2. Environment

- `SPEnv` (`env.mjs`) runs `envs` matches per Node process in virtual time (`Match` + `VirtualScheduler`). A match with
  a decision pending runs no scheduler callback, so its clock stands still until the policy acts. `server.mjs` serves
  it to Python over stdin/stdout (`py/spenv.py`); `AsyncPool` steps each process as soon as its batch is in.
- **A seed fixes the match**: its config, seat roles, human marks, teammate-proxy kinds and every game RNG stream are
  derived from the seed, not from the order matches open in. The same seed is the same match under any policy, so evals
  pair per seed. Eval match seed = `--seed` + config index × 1,000,000 + chunk × (matches / chunks) + i
  (`py/eval.py`). The config index is the position in a fixed list, or a hash of the name for newer configs:
  solo:ABYSS is 2, coop4h1:ABYSS is 580. Training seeds start at 1.01 × 10⁹.
- **Configs** (`<mode>:<difficulty>`, difficulty NORMAL / HARD / ABYSS; ABYSS is 终极模拟):

| Mode | Seats |
|---|---|
| `solo` | one model seat |
| `coopN` | seat 0 model; each other seat is a legacy mate with probability `legacyMates` (0 in the final recipe), else a model seat; 0..N−1 mates marked human |
| `coopNl` | seat 0 is the only model seat; every mate is legacy or a human proxy |
| `coopNai` | all N seats model, no humans |
| `coopNh<k>` | exactly k human seats (proxies), every other seat a model seat. **coop4h1** (one human + three model seats) is the co-op headline. |

- **Human seats in training and evals** are played by teammate proxies (`proxies.mjs`, `--human-proxy`, §10.1). Without
  the option a human seat is legacy, optionally announcing a main-bond plan (`ps._envPlanBonds`).
- **Late-game starts** (`--snap-start`): restores of snapshotted ABYSS preps (R9–R13). Not used by the final recipe.
- **Seat plan** (`patches/0001`, `server/match/seatPlan.js`): the one-round-trip deploy protocol of version 4 (per-seat
  refresh / offer tables drawn at prep start). Gated on `ps._seatPlan`; version 3 trains without it.
- **Training-only hooks** (`patches/0003`, `0004`): placement candidates, label seeds, placement exports, proxy buy
  noise / trade / spendDown, the Ioleta force flag. They are gated on fields the server never sets, so legacy and human
  seats and golden replays are unchanged.

### 2.1 Information rule

The owner's rule for the deployed AI: everything a player sees in the game is allowed, and nothing else.

- **Allowed:** monster information and HP, the 机变 kill chance, human teammates' cards and flags, the boss (it is not
  hidden in the game).
- **Not allowed at deploy:** search, simulation, layout rehearsal, the live battle seed.

An earlier reading (11-13 morning) also excluded the hidden boss before its round and humans' hands. The
`--visible-only` switch implements that reading (A1: human pairs only through the scout view; A3: hidden boss 0 before
its round). `--strict-preview` (`patches/0002`) further removes the kill chance and the spawn details. Neither switch
changes the layout or the dataId. The final pipeline sets neither, following the later rule. `infoguard.mjs` tests both.

## 3. Observation (version 3 + national plan)

2542 floats for the final models: the version 3 layout (2358 floats) plus the national-plan blocks (184 floats).
`dataId` `act2autochess-75d38e72a6cf` hashes the layout of the version 3 encoder together with the game-data version. A
model reads only its own dataId (`vcheck.mjs`); the size check in `play.mjs` catches a missing plan block.

| Block | Rows × width | Content |
|---|---|---|
| `global` | 1 × 38 | round, decision kind, difficulty, co-op flag, own LP / LP₀ and LP / 50, funds, shop level, upgrade / refresh price, free refreshes, freeze, deploy cap / count, actions this prep, boss / hidden / special round flags, living mates and humans, band, stage, rule switches |
| `shop` | 10 × 13 | shop slots: kind, piece / item code, elite, tier, price, sold, frozen, copies held, rule flags (protected bond, human pair, sold this prep) |
| `owned` | 27 × 13 | hand 10 + temp 5 + board 12 |
| `bonds` | 23 × 5 | own bond strip (count, tier, layers, …) |
| `enemies` | 8 × 8 | next round's enemy preview |
| `rewards` | 6 × 8 | merge reward offer |
| `personal` | 5 × 9 | personal choice cards |
| `cards` | 8 × 12 | 机变 draft cards (bounty kill chance for bounties) |
| `mates` | 3 × 11 | teammates' public rows |
| `extra` | 1 × 8 | boss and hidden-boss ids, rounds to the boss round, human / AI mate counts, band-decision and concurrent-draft flags |
| `mateBonds` | 3 × 50 | teammates' bond bars and announced plans |
| `bands` | 40 × 32 | band choices (band decision only) |
| `plan` | 1 × 55 | national plan: flag, P one-hot (23 bonds), S one-hot, P count / 6, P tier / 2, P layers / 100, S active, S layers / 100, forced flag, distinct owned P and S members |
| `planShop` / `planRewards` / `planOwned` | 10 × 3 / 6 × 3 / 27 × 3 | per row: in P, in S, a new distinct P member (owned: the only owned copy of a P member). Appended to the `shop` / `rewards` / `owned` rows. |

Version 2 (ppo1 / ppo2) is the first nine blocks: 920 floats, 60 actions, `act2autochess-b0c30136ef88`. The pickers read
version 3 plus `ext.board` (each owned row's tile, facing, place class and block; 28 tile rows) and a per-candidate
board slice (`Encoder.boardSlice`).

### 3.1 The national plan (`--nat-plan`)

The final recipe's plan options are `{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}`
("natPlan NM").

- **P (main nation)**: `assign: "auto"` takes the first allowed nation in a fixed value order from the route study
  (`trackm/recipes-M.json`): 拉特兰 > 维多利亚 > 谢拉格 > 卡西米尔 > 炎 > 萨尔贡 > 叙拉古 > 阿戈尔.
  - Allowed: live in this match (`live`: its operators can enter the pool), not tied to a living human's band or
    announced 主盟约, not held by another model seat of the match.
- **S (secondary class bond)**: `pairS` takes the community pair for P from the same table, else P's class bond with
  the most owned members.
- `band`: the seat opens with P's band. `focus`: legacy's internal focus bond is set to P.
- Solo plan shares come out at about 拉特兰 0.63 and 维多利亚 0.28.
- The plan does not change after it is fixed. A co-op replan rule (`coopReplan`) exists and was rejected
  ([FINDINGS.md](FINDINGS.md) §14).
- Not used by the final recipe: a learned plan head (`head`, the `planNat` block and 8 more actions; NH / NH2 arms),
  `lead` assignment, the dense milestone pays (`dense`) and the forced → free phases of the NAT arm.

The teacher of the BC data (`natPlan.teacher: "nat"`) is the national-route bot of `trackm/routebots.mjs`: legacy's buy
loop with a policy object that procures all live members of P toward tier 2 (`NAT_PROCURE_ONLY`).

## 4. Actions

100 discrete actions, masked per decision. Decision kinds are `prep`, `card` (机变 draft) and `band`.

| Index | Action |
|---|---|
| 0–3 | END (finish the prep's economy; the pickers and legacy then place and ready), REFRESH, LEVEL, FREEZE |
| 4–13 | BUY shop slot i |
| 14–40 | SELL owned piece j |
| 41–46 | take merge reward k |
| 47–51 | personal choice p |
| 52–59 | 机变 card c |
| 60–99 | band b (not used by the final models) |

At most 40 economy actions per prep (`MAX_PREP_ACTIONS`); a seat at the cap is forced to END.

## 5. Networks

Both architectures read the same observation and produce the same 100 logits. Acting is greedy (argmax over the legal
actions).

### 5.1 MLP (`py/model.py`, `SPNet`)

1,547,115 parameters with the plan blocks (the value head, about 33 k of them, is not needed at deploy).

- every integer-coded field is embedded (piece 32, enemy 16, band 16, stage 8, card 16, kind 8, bond 16, boss 16);
- each entity block goes through its own 2-layer MLP encoder (global 128, shop 128, owned 128, enemies 64, rewards 96,
  personal 64, cards 64, mates 64, extra 32, mateBonds 64, bands 32, plan 32); the plan-slot rows are appended to their
  `shop` / `rewards` / `owned` rows before the encoder;
- masked mean + max pooling per block, and the bond strip through an MLP to 128;
- a trunk of 512 → 256 (ReLU);
- heads: 4 scalar actions from the trunk; each slot action from that row's encoding concatenated with the trunk output
  (2-layer pointer head, 128); a 2-layer value head.

JS: `infer2.mjs` (exact rewrite of the reference `infer.mjs`: folded embeddings, absent rows skipped). 1.71 M
multiply-accumulates per decision.

### 5.2 v5 (`py/model5.py`, `SPNet5`)

2,719,295 parameters: a policy tower of 1,177,870 and a separate critic of 1,541,425 that is never exported.

- every entity row becomes a token (shop, owned, bonds, enemies, rewards, personal, cards, mates, plan), plus one global
  token (global + extra); companion rows merge into their token (owned ← planOwned, shop ← planShop, rewards ←
  planRewards, mates ← mateBonds); about 76 tokens per decision;
- each token: a 2-layer row encoder over its numeric and embedded fields, plus a type embedding; a unit token adds a
  projection of its catalogue bond membership (`v5/lookahead.json`);
- **member bias**: each head's attention logit between a unit token and the token of a bond it belongs to gets a learned
  bias; the plan token links to the bond tokens of P and S the same way;
- 2 pre-LN transformer layers, d 128, 4 heads, FFN ×2, QK-LayerNorm, GELU-tanh; a final LayerNorm; the context is the
  global token plus the masked mean / max of the present tokens;
- heads as in the MLP (scalars from the context, pointers from slot tokens + context);
- critic: its own token encoder and a d192 × 1 transformer over the observation (training only).

JS: `infer5.mjs`. 25.0 M multiply-accumulates per decision. v5 trains at lr 1e-4 with `--amp --micro 1024
--fwd-chunk 2048 --clip-sep` ([FINDINGS.md](FINDINGS.md) §12 for the lr).

`infer2.mjs` uses only + − × ÷ and comparisons, so its outputs are bit-identical in every JS engine. `infer5.mjs` and
the pickers also need `Math.exp` / `Math.tanh` / `Math.sqrt` (softmax, GELU-tanh, LayerNorm), so their outputs agree
across engines only to float rounding. Every forward is checked against torch on recorded decisions before a release
([RUNBOOK.md](RUNBOOK.md) §6). Deploy cost was not a constraint for the final choice; it is reported for information
([RUNBOOK.md](RUNBOOK.md) §8).

## 6. Training (PPO)

`py/train_ppo.py`. The final recipe (exact flags: [rl/final/README.md](../rl/final/README.md)):

| Setting | Value |
|---|---|
| init | BC on 1.61 M decisions of the national-route teacher (§3.1), 20 epochs: MLP "PB5" (val CE 0.354), v5 (val CE 0.226) |
| collection | 50 Node processes × 64 matches, asynchronous; one trajectory per policy seat |
| advantage | GAE per trajectory, per decision: γ 0.997, λ 0.95; global normalisation |
| PPO | clip 0.2, 3 epochs, rollout 65 536 samples, minibatch 8192, entropy 0.003, value coef 0.5, no value clipping, target KL 0.02 |
| learning rate | MLP 3e-4, v5 1e-4 |
| KL anchor | 0.01 × KL(BC ‖ policy), constant (`--bc-kl 0.01 --bc-kl-steps 0`) |
| resume | value-only for 5 updates (fresh Adam), then lr ramps over 20 updates |
| placement in training | legacy, `--rehearsal 0 --boss-rehearsal 0` (the pickers are eval / deploy only) |
| band | `--band-policy 0`: the plan's band (§3.1) |
| other | update / collection overlap (`--overlap`), legacy mates 0, `--check` (rule violations at every match end) |

## 7. Reward

Reward version 2 (R2) plus the plan terms. Per model seat (`env.mjs`, `--reward-v 2 --r2
'{"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002}' --shape-uncapped --shape-tier 0.06 --shape-stack 0.3`).
L_ref = 28, the default starting LP.

| Term | Weight | When | Definition |
|---|---|---|---|
| win | +3.0 | match end | the team cleared (any seat won, or the Final Assault pool reached 0); paid to every seat, eliminated ones included |
| faTeam | +1.5 × share | match end | team share of the round-14 Final Assault pool: 1 − hp / maxHp |
| faOwn | +0.5 × share | match end | this seat's damage / pool maxHp |
| LP | 1.0 × ΔLP / L_ref | every decision | ΔLP = 0.75 × Δ(team LP) / n_seats + 0.25 × Δ(own LP) |
| leak | −0.15 × leaks / L_ref | each round's settlement | own counted leaks, at most 10 per round |
| save | +0.075 × kills / L_ref | each 联防 round | kills this seat made as a 联防 helper |
| elimination | −0.5 | match end | the seat was eliminated and did not win |
| layer | +0.002 per layer | each prep end | Σ layers of active stacking bonds above the seat's high-water mark |
| **plan bonus** | +1.5 × (0.5 + 0.5 × team FA share) | match end | P was at tier ≥ 2 at the Final Assault start |
| **plan secondary** | +0.5 | match end | S was active with ≥ 40 layers at the Final Assault start |
| shaping | γΦ(s′) − Φ(s), Φ(terminal) = 0 | every decision | Φ = 0.06 · T + 0.3 · S + **0.5 · min(P on board, 6) / 6 + 0.3 · P layers / 100**; T = Σ active tier / max tier; S = Σ layers of active stacking bonds / 100. Own seat only, γ = the GAE γ. |

The plan terms are the defaults of `natPlan` (`bonus`, `sec`, `secLayers`, `phiCount`, `phiLayers`). Why they were
needed: [FINDINGS.md](FINDINGS.md) §10. Version 1 (ppo1 / ppo2) and the redesign to R2: [FINDINGS.md](FINDINGS.md) §2.

## 8. Mixes and the curriculum

`--curriculum fixed --mix <name>`. Weights are per match.

| Mix | Used by | Content |
|---|---|---|
| `hard1` | stage 1 (NM) | solo:HARD only |
| `auto1` | stage 2 (NMA), extension | solo only. ABYSS switches on once the rolling HARD win reaches 0.60 (`--auto-hard-gate`). Its share then moves by at most 0.02 per update (`--auto-step`) toward the value that keeps the expected win of the mix at 0.35 (`--auto-target`), within [0.10, 0.85] (`--auto-smin` / `--auto-smax`). NORMAL keeps ≥ 0.05, rising to 0.30 while HARD wins less than 0.30. The rest is HARD. Difficulties mix inside each update. |
| `mp2` | co-op stage | ABYSS: solo .44, coop4 .16, coop2 .08, coop3 .03, coop4l .04, coop2l .03; HARD .16; NORMAL .06. With `--human-proxy` the human-marked seats are proxies, so about 45 % of samples are solo, 50 % humans + model seats, 5 % one-model co-op. |
| v1 / v2 / v3 / v3s / v3sc / solo1 / mp1 | earlier arms | `py/train_ppo.py` has the exact weights |

In every final run the auto1 ABYSS share reached 0.85 long before 60 M, with the rolling ABYSS win above 0.35.

## 9. Deploy pickers

Both pickers score a pool of candidate layouts generated by rules at decision time, with no battle built, and play the
best one. Both were trained on labels from training-time search with simulations on salted per-match seeds, never the
live seed. Both run in plain JS inside the environment (`env.mjs`; `--npicker-*`, `--xplace` in `eval.py`).

### 9.1 Normal rounds: c9 over the candidate pool

- **Pool** (`bot.js candidatePoolForTraining`, `--npicker-pool`; K 8, edits 12, max 32): legacy's K8 layouts; stacking
  templates (阿戈尔 fodder in front of members, 谢拉格 collector range, a positional hill-climb on features, 卡西米尔
  redeploy, 突袭 idle start, 莫斯提马 neighbours); last round's board re-fitted; 12 one-step edits (swap, turn, move) of
  legacy's default. Distinct candidates only, about 21 per decision.
- **Labels** (`env pLabel`, training only): each candidate rehearsed on 2 salted per-match seeds; label rule "D" = no
  extra leaks, then most Σ layers gained, ties to most kills. 41 k pool decisions (l8, plus 6 k of l7) on NMA and
  NH2sA boards, played by oracle and DAgger states.
- **Model**: a v5 tower (d128, the version 3 + `ext.board` observation with the candidate's board slice) and an MLP head
  with 4 outputs (score, Δleaks, Δlayers, Δkills). Listwise cross-entropy on the label utility plus auxiliary
  regression (`py/picker_train5j.py`), warm-started from the K8 head "Sall" (about 300 k K8 labels). 722 k parameters.
- **Rule** (`--npicker-mode reg --npicker-margin 0.5`): switch from legacy's default to the best candidate only if its
  predicted −2 Δleaks + 0.2 Δlayers + 0.05 Δkills exceeds 0.5. Rounds 1–13.
- Export `py/export_p5.py` → `c9-js.{json,bin}` (`p5-picker`), parity 3.8e-5 (`p5parity`). Gate `py/poolgate.py`.

### 9.2 Boss rounds: xboss-v3

- **Pool** (`xpool.mjs`): the current layout, legacy's K8, the generator g3b's greedy decode and 8 samples, stacking
  templates, last round's board, and 6 one-step edits of the scorer's own pick. 25–54 candidates per decision.
- **Labels** (`eval.py --xlabel`, `xsearch.mjs`, `xteach.mjs`): a hill-climb teacher with 256 simulations per decision,
  each leaf the mean of 2 salted rehearsal seeds. Records keep every candidate's per-seed leaks / boss damage / kills /
  layers and are re-scored at train time.
- **Win utility** (`py/winutil.py`): only the round-14 pick moves the win rate (round 15 is reached only in matches
  already won). u = P(win | FA share), a logistic fit on in-game picks and outcomes: solo sigmoid(−13.05 + 15.56 ×
  share), coop4h1 sigmoid(−13.77 + 17.32 × team share). Solo share = the candidate's rehearsed boss damage / FA pool max
  HP; co-op adds the other seats' real FA damage (assumed 0.635 at search time).
- **Scorer Dw** (`py/xtrain.py --wutil --winfit`): a QNet over `xfeat.mjs` features (per piece, tile and pair, the
  legacy layout as an input, per-candidate no-sim features), auxiliary heads, listwise ranking; pick = argmax. From
  scratch on all boss labels plus the pool labels ×3, boss weight 4, 12 epochs. Gates: `py/xkfold.py` (6 folds by match
  seed) and a pool gate.
- **Generator g3b**: an autoregressive (tile, facing) pointer over the fielded pieces, imitation then a small AWR step on
  search labels (lr 1e-4, 4 epochs, τ 5).
- Release files: `dw.json` (scorer), `g3b.json` (generator), `xplace.json` (the `--xplace` options:
  `{"rounds":"boss","margin":0,"pool":{"K":8,"genN":8,"genT":1,"tpl":1,"prev":1,"edits":6,"max":48}}`). The QNet JS
  port matches torch exactly. The older boss picker (`--picker-head` / `--picker-trunk`) must be off with xboss-v3.

## 10. Co-op stage

The national-plan policy trains solo only (hard1, auto1). The co-op stage then adapts it to playing next to humans.

### 10.1 Teammate proxies (`proxies.mjs`, `--human-proxy`)

Each human seat is played by a scripted proxy, its kind drawn per match from a seeded RNG keyed on the match seed (the
env RNG is untouched).

| Kind | Play |
|---|---|
| `legacy` | the production legacy bot (the eval proxy) |
| `nat`, `nat:<P>`, `nat:rand`, `nat:<P>Ship` | the national-route bot: all live members of P toward tier 2, S from the community pairs, national band, focus P; `<P>` forces a nation when live, `rand` draws one |
| `class` | the elite plan (助力 / 坚守 / 精准) with Ioleta's band |
| `recipe:<k>` | a community recipe AI (legacy plus the recipe's advisor plan) |
| `noisy:<σ>` | legacy with seeded N(0, σ) noise on its buy scores |
| `mix:<kind>=<w>\|…` | a weighted draw over the kinds |

`announce: "own"` makes nat and recipe proxies announce their nation as 主盟约, as a player would. The nat and class bots
use legacy's buy loop with `trade` (a layer trade) and `spendDown` (spend expiring funds on refreshes for a pair); both are
gated flags in `bot.js`.

### 10.2 The stage (arm CB)

`rl/final/coop/coopstage.py` builds the argument list from the run's own stage-2 arguments plus these overrides:

- init = KL anchor = the 60 M checkpoint, `--bc-kl 0.01 --bc-kl-steps 0`; 60 M → 70 M;
- `--mix mp2`, `--band-policy 0`, natPlan NM without `coopReplan`;
- `--human-proxy {"kinds":["mix:nat:lateranoShip=0.25|nat:kjeragShip=0.25|nat:rand=0.2|legacy=0.15|class=0.1|noisy:50=0.05"],"announce":"own"}`:
  half the draws ask for a carry nation (拉特兰, 谢拉格: the nations the plan wants most), and about 40 % of human seats
  hold one (the nation is not always live);
- `--micro 2048` (gradient accumulation, memory only).

Gate: the 70 M checkpoint against its own 60 M start on the same seeds, settings a and b (§11). Adopt if co-op rises
and solo drops by no more than 0.02. The stage applies to the MLP. On v5 it failed at both learning rates
([FINDINGS.md](FINDINGS.md) §14).

## 11. Evaluation protocol

- **Headline**: `eval.py --greedy --bond-log --rehearsal 0 --boss-rehearsal 0 --band-policy 0 --nat-plan <NM>`, configs
  solo:ABYSS and coop4h1:ABYSS, `--matches 480 --chunks 28` (476 played), `--seed 7000000`. Co-op adds
  `--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}'`: the human seat is legacy at its production
  rehearsal. Eval protocol details and the metrics: [records/final/README.md](../records/final/README.md).
- **Setting a**: no pickers (legacy places). **Setting b**: c9 + xboss-v3, the deploy setting. Picker lift = b − a on the
  same seeds.
- **Paired seeds.** Every comparison is a mean of per-seed differences on identical seeds, SE = sd / √n. The pairing
  only holds for the same `--seed`, `--matches` and `--chunks`.
- **Stacking metrics** next to every win (`--bond-log`, `py/stackrep.py`): main bond type and tier, natMain, natMainT2,
  main / secondary / Σ layers, elites.
- **Milestones**: 30 / 40 / 50 / 60 M per run, 70 M for the co-op stage, 70 / 80 / 90 M for extensions. In-run evals (96
  seeds, solo:HARD and solo:ABYSS every 2 M) are trend only.
- **Pick**: rank candidates by setting b; solo:ABYSS and coop4h1:ABYSS must both be ≥ 0.50; the mean breaks ties.
- **Held-out confirmation**: the pick and the NMA recipe were both selected on the milestone seeds, so the top
  candidates are re-run on a fresh seed set, `--seed 200000000` (solo:ABYSS match seeds 202,000,000–202,000,475,
  coop4h1:ABYSS 780,000,000–780,000,475). No earlier eval base or record seed reaches them. **The held-out number is the
  reported one.**
- **Statistics.** Win SE = √(p(1−p)/n), about ±0.022 at 476 seeds. Seed-to-seed training noise was about 0.03–0.10
  depending on stage, so single-seed arms are read with care.

## 12. Final pipeline

```text
record (national-route teacher, natPlan data, 1.61 M decisions, 112 workers)
  → BC (MLP PB5 / v5, 20 epochs)
  → NM:  PPO solo:HARD (hard1), natPlan NM, 5 M          (MLP lr 3e-4, v5 lr 1e-4)
  → NMA: auto1 curriculum to 60 M
  → [extension 60 M → 90 M if still rising, rule in rl/final/README.md]
  → [co-op stage 10 M, CB, MLP only; gated vs its start]
  → milestone evals (a, b) → pick (setting b) → held-out confirmation
```

Two architectures × two seeds were trained (FIN-MLP-s1/s2, FIN-V5-s1/s2). `rl/final/` holds the launchers, the
argument checks and the pick tools. [RUNBOOK.md](RUNBOOK.md) §4 has the commands, §8 the hardware.

## Appendix A: the hybrid line (11-12 / 11-13 morning)

The recipe and protocol of the m3 → S / SK line, kept for the results of [RESULTS.md](RESULTS.md) Part B. The policy
picked its own band (a band head fitted offline and frozen), legacy placed every round, and evals used b3 / b0.

### A.1 Training (PPO), the hybrid line

`py/train_ppo.py`, current recipe (`scripts/train.sh`):

| Setting | Value |
|---|---|
| collection | 25 Node processes × 64 matches, asynchronous; one trajectory per policy seat |
| advantage | GAE per trajectory, per decision: γ 0.997, λ 0.95; global normalisation |
| PPO | clip 0.2, 3 epochs, rollout 32 768 samples, minibatch 4096, Adam lr 1e-4, entropy 0.003, value coef 0.5, no value clipping, target KL 0.02 |
| KL anchor | 0.01 × KL(anchor ‖ policy), constant. The anchor is ppo2 @ 15 M grown to observation v3 with the fitted band head. SK and KL use 0.003. |
| resume | the value head trains alone for 20 updates (fresh Adam moments), then lr ramps up over 20 updates |
| band head | `--freeze-band`: no gradient on the band encoder or band pointer head, and band rows are left out of the policy-gradient and entropy terms. 70 % of RL seats let the policy pick (`--band-policy 0.7`); the rest open with a uniformly random allowed band. |
| legacy placement in training | `--rehearsal 0 --boss-rehearsal 3` |
| other | update/collection overlap (`--overlap`), late-game starts 0.3, legacy mates 0.3, `collectViolations` at every match end (`--check`) |

**Band head.** `band_table.py` measures every band on paired seeds (ppo2 @ 15 M with seat 0 forced to the band, 48
seeds × 4 configs). `bandobs.mjs` records band decisions, and `bandfit.py` fits only the band encoder and head to
softmax(Δfa / 0.03) on ABYSS and softmax(Δwin / 0.05) on HARD. `bandxplant.py` copies the fitted head into a
checkpoint. Non-band outputs stay unchanged.

### A.2 Training mixes of the hybrid line

`--curriculum fixed --mix <name>`. Weights are per match; one coop4ai match yields about four seats of samples.

| Mix | ABYSS | HARD | NORMAL | ABYSS breakdown |
|---|---|---|---|---|
| v1 (ppo2) | .60 | .30 | .10 | solo .15, coop2 .15, coop4 .30 |
| v2 (m2, m3) | .60 | .30 | .10 | solo .12, coop2 .10, coop4 .18, coop4ai .20 |
| v3 (m3s, P, P0, KL, ALL, R3) | .65 | .25 | .10 | solo .25, coop4l .12, coop4 .08, coop2l .07, coop2 .05, coop4ai .08 |
| v3s (S, SK) | .75 | .20 | .05 | solo .45, coop4l .15, coop2l .10, coop4ai .02, coop4 .02, coop2 .01 (≈ 55 % solo, 28 % one-RL co-op by samples) |
| v3sc (SC) | .78 | .17 | .05 | solo .35, coop4l .25, coop2l .15, coop4ai / coop4 / coop2 .01 each |
| solo1 (SO1) | .80 | .15 | .05 | solo .70, coop4l .05, coop2l .03, coop4ai .02 |

The HARD and NORMAL parts follow the same pattern; `py/train_ppo.py` has the exact weights.

### A.3 Evaluation protocol of the hybrid line

- **Paired seeds.** Runs are compared on identical seeds (§2), per seed: `py/compare.py`, or the mean of per-seed
  differences with SE = sd / √n (`docs/curves/paired.csv`). The 96 in-run seeds are the first 96 of the 480.
- **Greedy** policy, the policy picks its own band (`--band-policy 1`), CPU.
- **Deploy setting:** the RL seat's legacy placement never rehearses at normal rounds (`--rehearsal 0`). The boss
  rounds are reported at `--boss-rehearsal 3` (b3, legacy's three-candidate rehearsal) and 0 (b0). Since 11-13 the
  deployed AI may use no simulation of any kind and only the information set of §2.1 (the 11-13 morning reading), so
  **b0 with `--visible-only` was the deploy number** of that line (`scripts/eval.sh` default). b3 stays for continuity. The tables measured before the switch
  existed are plain b0; the switch moves them by ±0.015 ([RESULTS.md](RESULTS.md) §1b).
- **Production legacy** (the baseline AI the game ships) rehearses three layouts with the live battle seed at normal
  rounds (r3). Our r0 numbers therefore give the RL seat less placement help than production legacy gets.
- **Headline:** solo:ABYSS, 480 seeds, b3 and b0 (`scripts/eval.sh solo480`). One-RL co-op: coop4l / coop2l ABYSS,
  288 seeds, RL seat r0, legacy mates r3 as in production (`--mate-rehearsal 3`), b3 (`scripts/eval.sh deploy288`).
- **In-run:** every 1e7 steps, 96 seeds × 7 configs at b3, plus solo / coop4l / coop2l at b0.
- **Statistics:** win SE = √(p(1−p)/n). With 96 seeds the SE is about ±0.05, which is too wide for the differences we
  care about, and several early 96-seed reads did not survive 288 / 480 seeds ([FINDINGS.md](FINDINGS.md) §9).
