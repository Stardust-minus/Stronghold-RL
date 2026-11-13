# Design

This document describes the current line, the hybrid model ("ppo3h": m3 → m3s / S → SK / SC / SO1). Earlier designs
(ppo1 / ppo2, the version 4 full-control attempt) appear only where they explain a choice. For the history, see
[EXPERIMENTS.md](EXPERIMENTS.md).

## 1. What the policy controls

The model drives AI seats ("model-AI seats") in ordinary matches. The legacy rule AI (`server/match/bot.js`) keeps the
rest of the seat's turn.

| Decision | Who | Notes |
|---|---|---|
| Opening band (分队 / strategy) | policy, **head frozen** | Fitted offline to a paired-seed strength table (§6); PPO never updates it. The policy picks after every human seat (aiPicksLast semantics). Masks forbid a band whose bond clashes with a human's band or announced main bond. |
| Prep economy: buy, sell, refresh, level up, freeze | policy | One flat action space, masked per decision |
| Merge rewards, personal choices, the 机变 draft card | policy | |
| Lineup, tiles and facings, equipment, temp / hand resolution, Ready | legacy | `LAYOUT_ONLY` steps of `createBotPrepSteps` (legacy with its procurement switched off) |
| Layout rehearsal (simulating candidate layouts before committing) | legacy, configurable | `rehearsal` at normal rounds (`r`), `bossRehearsal` at the Final Assault and Hidden Core rounds (`b`) |

The AI-seat rules of the game's owner hold as masks: 让主盟约 (`yieldMain`: never buy, or take as a reward, an operator of a
living same-pool human's main core bond or opening-strategy bonds) and 让合成 (`yieldMerge`: never take the third copy
of an operator a human holds two of). Training marks 0..n−1 teammates as humans at random (`humans: 'random'`) so the
masks are always exercised.

The version 4 design let the policy take every decision of the seat, placement included (`obsVersion: 4`, 3397
inputs, 295 actions, 28 tiles × 4 facings). It is still in `spec.mjs` / `env.mjs`, but PPO did not learn on it:
[FINDINGS.md](FINDINGS.md) §9.

## 2. Environment

- `SPEnv` (`env.mjs`) runs `envs` matches per Node process in virtual time (`Match` + `VirtualScheduler`). A match with
  a decision pending runs no scheduler callback, so its clock stands still until the policy acts. `server.mjs` serves
  it to Python over stdin/stdout (`py/spenv.py`); `AsyncPool` steps each process as soon as its batch is in.
- **A seed fixes the match**: its config, seat roles, human marks and every game RNG stream are derived from the seed,
  not from the order matches open in. The same seed is therefore the same match under the policy, the teacher and
  legacy. Evaluation seeds are `7,000,000 + 1,000,000 × c + i` (`py/eval.py`). Here c is the config's index in a fixed
  list, or a hash of the name for newer configs, so solo:ABYSS uses 9,000,000 + i. Training seeds start at 10⁹.
- **Configs** (`<mode>:<difficulty>`, difficulty NORMAL / HARD / ABYSS; ABYSS is 终极模拟):

| Mode | Seats |
|---|---|
| `solo` | one RL seat |
| `coopN` | seat 0 RL; each other seat is a legacy mate with probability `legacyMates` (0.3), else RL; 0..N−1 mates marked human |
| `coopNl` | the target co-op setting: seat 0 is the **only** RL seat; every mate is legacy (rule AI) or a human proxy |
| `coopNai` | all N seats RL, no humans |
| `coopNh<k>` | exactly k human proxies, every other seat RL (product scenarios) |

- **Human proxies** are legacy seats marked human. They may announce a main-bond plan (`ps._envPlanBonds`), which
  legacy then pursues. This is the one-line hook in `bot.js` that the game repository keeps.
- **Late-game starts** (S1): 25 % of eligible ABYSS prep ends at R9–R13 are snapshotted (`server/match/snapshot.js`);
  30 % of new ABYSS training matches restore one and replay from that round.
- **Seat plan** (`patches/`, `server/match/seatPlan.js`): the deploy protocol is one round trip per prep, with
  inference on the player's device. A model-AI seat's refreshes and reward offers then come from a per-seat refresh /
  offer table drawn at prep start, and its own effects draw a per-seat meta stream. Draws a device cannot reproduce are
  counted as stop points. Writes from other players wait until the plan ends. The hooks are gated on
  `ps._seatPlan`, which only model-AI seats ever set, so legacy and human seats and golden replays are unchanged.
  `env.mjs` turns the plan on for version 4 only. The hybrid line trains without it.
- **Strict preview** (`patches/0002-visible-only-hooks.patch`): `bot.js` `fieldModel` gets a preview-only path for
  the seats in `m._visibleOnly`. Only `env.mjs` sets it, under `strictPreview`. Unset, the production path computes
  exactly as before.

## 3. Observation (version 3)

2358 floats, `dataId` `act2autochess-75d38e72a6cf`. The id hashes the layout together with the game-data version, and
a model reads only its own dataId (`vcheck.mjs`).

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

Version 2 (ppo1 / ppo2) is the first nine blocks: 920 floats, 60 actions, `act2autochess-b0c30136ef88`.

### Information set

The owner's rule for the deployed AI (11-13) is that a model-AI seat may use what a player sees and nothing else.

- **Allowed:** enemy information from the preview, HP, and the 机变 bounty kill chance (the same estimate the
  external-AI digest hands out).
- **Not allowed:** the hidden boss before its round, other players' hands and temp slots, and any rehearsal or
  simulation of layouts, the boss rounds included.

Two encoder switches implement it. Neither changes the layout or the dataId, so existing checkpoints run under them.

| Switch | Items | Deploy |
|---|---|---|
| `visibleOnly` (`--visible-only`) | **A1** the 让合成 human-pair flag reads a human's units only through the prep-scout view (the board), never the hand / temp; **A3** the hidden-boss id stays 0 until its round | **always on** |
| `strictPreview` (`--strict-preview`) | **A2** bounty kill chance 0; **A4** no enemy spawn time; **B1** legacy placement for the RL seat models the field from the wave preview only (routes by gate and flight, no per-spawn route index, no spawn modifiers, no round HP scale; `patches/0002`, `server/match/bot.js` `fieldModel`, gated on `m._visibleOnly`) | optional, **off**: the owner rules enemy information, HP and the kill chance acceptable |

Boss-round rehearsal is off at deploy (`--boss-rehearsal 0`, "b0"). `infoguard.mjs` checks the switches: with
`visibleOnly` on, an RL seat's encoding must never read another seat's hand or temp outside the scout view, and the
hidden-boss field must stay 0 before its round; `strict` adds the field-model reads; `off` is the positive control.
Training may still use privileged signals, e.g. search as a teacher.

## 4. Actions

100 discrete actions, masked per decision. Decision kinds are `prep`, `card` (机变 draft) and `band`.

| Index | Action |
|---|---|
| 0–3 | END (finish the prep's economy; legacy then places and readies), REFRESH, LEVEL, FREEZE |
| 4–13 | BUY shop slot i |
| 14–40 | SELL owned piece j |
| 41–46 | take merge reward k |
| 47–51 | personal choice p |
| 52–59 | 机变 card c |
| 60–99 | band b |

At most 40 economy actions per prep (`MAX_PREP_ACTIONS`); a seat at the cap is forced to END.

## 5. Network

`SPNet` (`py/model.py`), about 1.51 M parameters at version 3:

- every integer-coded field is embedded (piece 32, enemy 16, band 16, stage 8, card 16, kind 8, bond 16, boss 16);
- each entity block goes through its own 2-layer MLP encoder (global 128, shop 128, owned 128, enemies 64, rewards 96,
  personal 64, cards 64, mates 64, extra 32, mateBonds 64, bands 32);
- masked mean + max pooling per block, and the bond strip through an MLP to 128;
- a trunk of 512 → 256 (ReLU);
- heads: 4 scalar actions from the trunk; each slot action from that row's encoding concatenated with the trunk output
  (2-layer pointer head, 128); a 2-layer value head.

Only Embedding, Linear, ReLU, masked mean / max and concatenation are used, so `infer.mjs`, `infer2.mjs` and
`infer3.mjs` mirror the network in plain JS. Deployment constraints for that forward pass:

- it runs on the player's device: ≤ 10 ms per decision on a mid-range phone (Chrome CPU throttling 4×,
  `chrome_bench.mjs`);
- it must be bit-identical in every JS engine, because the server re-runs it to check a client by hash. So the
  operation order is fixed, only + − × ÷, comparisons and `Math.sqrt` are allowed (no `exp` / `log` / `tanh` / `pow`), and
  sampling is the argmax of the logits plus server-supplied noise;
- it reads only the seat's own view and its own dataId.

## 6. Training (PPO)

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

## 7. Reward version 2 (R2)

Terms, per RL seat (`env.mjs`, `--reward-v 2 --r2 '{"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002}'`).
L_ref = 28, the default starting LP, is the same constant for every band.

| Term | Weight | When | Definition |
|---|---|---|---|
| win | +3.0 | match end | the team cleared (any seat won, or the Final Assault pool reached 0); paid to every seat, eliminated ones included |
| faTeam | +1.5 × share | match end | team share of the round-14 Final Assault pool: 1 − hp / maxHp |
| faOwn | +0.5 × share | match end | this seat's damage / pool maxHp |
| LP | 1.0 × ΔLP / L_ref | every decision | ΔLP = 0.75 × Δ(team LP) / n_seats + 0.25 × Δ(own LP) |
| leak | −0.15 × leaks / L_ref | each round's settlement | own counted leaks, at most 10 per round (the game's per-round LP cap) |
| save | +0.075 × kills / L_ref | each 联防 round | kills this seat made as a 联防 helper (half the leak weight, so it cannot be farmed) |
| elimination | −0.5 | match end | the seat was eliminated and did not win |
| layer | +0.002 per layer | each prep end | Σ layers of active stacking bonds above the seat's high-water mark |
| round | 0 | | (version 1: 0.1, ppo2: 0.05) |
| shaping | γΦ(s′) − Φ(s), Φ(terminal) = 0 | every decision | Φ = 0.06 · T + 0.3 · S; T = Σ active tier / max tier; S = Σ layers of active stacking bonds / 100 (uncapped, `--shape-uncapped`). Own seat only, γ = the GAE γ. |

`--reward '{"round":0.05,"fa":1.5}'` still appears in the launch lines. It sets version-1 weights that R2 does not
read.

Version 1 (ppo1 / ppo2): per decision, `round × Δrounds − lp × ΔLP / LP₀` (team-weighted 0.75 / 0.25), +2.0 for a win,
+`fa` × the team FA share. [FINDINGS.md](FINDINGS.md) §2 explains why it was replaced.

## 8. Training mixes

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

## 9. Evaluation protocol

- **Paired seeds.** Runs are compared on identical seeds (§2), per seed: `py/compare.py`, or the mean of per-seed
  differences with SE = sd / √n (`docs/curves/paired.csv`). The 96 in-run seeds are the first 96 of the 480.
- **Greedy** policy, the policy picks its own band (`--band-policy 1`), CPU.
- **Deploy setting:** the RL seat's legacy placement never rehearses at normal rounds (`--rehearsal 0`). The boss
  rounds are reported at `--boss-rehearsal 3` (b3, legacy's three-candidate rehearsal) and 0 (b0). Since 11-13 the
  deployed AI may use no simulation of any kind and only the information set of §3, so **b0 with `--visible-only` is
  the deploy number** (`scripts/eval.sh` default). b3 stays for continuity. The tables measured before the switch
  existed are plain b0; the switch moves them by ±0.015 ([RESULTS.md](RESULTS.md) §1b).
- **Production legacy** (the baseline AI the game ships) rehearses three layouts with the live battle seed at normal
  rounds (r3). Our r0 numbers therefore give the RL seat less placement help than production legacy gets.
- **Headline:** solo:ABYSS, 480 seeds, b3 and b0 (`scripts/eval.sh solo480`). One-RL co-op: coop4l / coop2l ABYSS,
  288 seeds, RL seat r0, legacy mates r3 as in production (`--mate-rehearsal 3`), b3 (`scripts/eval.sh deploy288`).
- **In-run:** every 1e7 steps, 96 seeds × 7 configs at b3, plus solo / coop4l / coop2l at b0.
- **Statistics:** win SE = √(p(1−p)/n). With 96 seeds the SE is about ±0.05, which is too wide for the differences we
  care about, and several early 96-seed reads did not survive 288 / 480 seeds ([FINDINGS.md](FINDINGS.md) §9).
