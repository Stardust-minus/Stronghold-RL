# records/final: metrics of the final evaluations

The files here summarise the milestone evals, the co-op stage gate, the candidate ranking and the held-out
confirmation of the final runs. `rl/final/records_final.py` regenerates them from the eval outputs. No per-match rows
are published: the eval JSONs hold them.

| File | Content |
|---|---|
| `milestones.csv` | every milestone eval of the final runs: win, paired Δ vs the reference, picker lift, Δ vs the run's own 60 M, stacking |
| `candidates.csv` | each final candidate in settings a and b, stacking of setting b |
| `coopgate.csv` | the co-op stage (and extension) gate: the stage output vs its start, settings a and b |
| `ranking.csv` | the pick rule: both setting-b wins ≥ 0.50, ranked by their mean; paired Δ vs rank 1 |
| `heldout.csv` | the held-out confirmation (`--seed 200000000`), setting b |
| `heldout-paired.md` | the same held-out runs with full stacking, paired Δ vs FIN-MLP-s2-70M and the pooled view (milestone + held-out seeds); `rl/final/hopair.py` |
| `repro-gate.csv` | the eval trees reproduce the reference evals match by match |
| `summary.md` | the milestone table as markdown |

## How an eval is run

`py/eval.py` with the headline protocol ([docs/DESIGN.md](../../docs/DESIGN.md) §11):

```sh
python tools/rl/py/eval.py --ckpt <model.pt> --configs solo:ABYSS --matches 480 --chunks 28 --seed 7000000 \
  --greedy --bond-log --rehearsal 0 --boss-rehearsal 0 --band-policy 0 --nat-plan '<natPlan NM>' [pickers] --out solo.json
# co-op: --configs coop4h1:ABYSS --mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}'
# pickers (setting b): --npicker-head c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool --xplace '<xboss-v3>'
```

- **Configs.** `solo:ABYSS`: one model seat. `coop4h1:ABYSS`: one human seat played by the legacy bot at its production
  rehearsal (r3), three model seats. A match is won if the team clears ABYSS (any seat wins).
- **Seeds.** 480 matches in 28 chunks of 17, so 476 are played. Match seed = `--seed` + config index × 1,000,000 +
  chunk × 17 + i. With `--seed 7000000`: solo:ABYSS 9,000,000 + …, coop4h1:ABYSS 587,000,000 + …. The held-out set
  uses `--seed 200000000`. Another `--matches` / `--chunks` gives another seed set, which does not pair.
- **Settings.** **a** = no pickers: legacy places every round. **b** = the deploy pickers: c9 over the candidate pool on
  rounds 1–13, xboss-v3 on the boss rounds. Both settings run no search, rehearsal or live seed for the model seats
  (`sims` = 0 in every b file).
- **Trees.** Setting a ran on the training tree, setting b on the eval tree (the same code plus the pickers' eval
  options). `repro-gate.csv`: both reproduce the reference files match by match.

## The metrics

| Column | Meaning | Computed by |
|---|---|---|
| `win` | team win rate over the played matches | `eval.py` (`teamWin`) |
| `n` | matches played (476) | |
| `d_vs_nma30`, `se` | paired Δ win vs the reference NMA 30 M (seed 0 of the MLP recipe) in the **same setting**: mean of per-seed differences on the common seeds, SE = sd / √n | `records_final.py delta()` |
| `lift_b_minus_a`, `se_lift` | **picker lift**: setting b − setting a of the same checkpoint, paired per seed | `delta()` |
| `d_vs_own60`, `se_own60` | 70 M (co-op stage or extension) − the same run's 60 M, paired, same setting | `delta()` |
| `natMain` | share of model seats whose main bond is national | `py/stackrep.py` |
| `natMainT2` | share of model seats whose main bond is national **and** at tier ≥ 2 (6 distinct units of the nation) | `stackrep.py` |
| `mainTier` | mean tier of the main bond | `stackrep.py` |
| `mainLayers` / `secLayers` / `sumLayers` | layers of the main bond; of the secondary (the best other stacking bond, by tier then layers); Σ over every active stacking bond | `stackrep.py` |
| `elites` | golden (elite) units on the board | `stackrep.py` |
| `topMain` / `topSec` | the most common main bonds / secondaries and their shares | `stackrep.py` |
| `b_mean` | (setting-b solo win + setting-b co-op win) / 2: the pick score | `records_final.py` |
| `both_ge_0.50` | both setting-b wins ≥ 0.50 (the target) | `records_final.py` |
| `adopt_b` / `adopt_a` | gate: co-op Δ > 0 and solo Δ ≥ −0.02 vs the start | `records_final.py` |
| `result`, `differing` (repro gate) | IDENTICAL when every per-match record equals the reference | `rl/final/gate.sh` |

**Stacking metrics** come from `--bond-log`. Each model seat logs its bonds at its **last battle** (the Final Assault
for a seat that reaches it, else the round it was eliminated). The main bond is the active bond with the most units,
then the highest tier; noStack bonds such as 绝技 are not main bonds. National = `bondType SEASON` in `data/bonds.json`.
The averages run over model seats: one per solo match, three per coop4h1 match. `stackrep.py` also prints the shares at
R9 / R12 and the win of seats with a tier-2 national main against the rest.

**Co-op diagnostics** (`py/coopstats.py`, in the co-op result tables of `records/results/`):

- **collision**: the share of model seats whose plan nation P is the human seat's nation. That is the proxy's own plan
  P when it has one, else the human's main bond at its last battle when that bond is national.
- **replans per match**: plan changes of the model seats under `natPlan.coopReplan` (0 without it; the final models do
  not replan).
- `winCollided` / `winOther`: team win of matches with and without a collision.

**Held-out** (`heldout.csv`): the same setting-b protocol on `--seed 200000000`: solo:ABYSS match seeds
202,000,000–202,000,475, coop4h1:ABYSS 780,000,000–780,000,475. Training seeds start at 1.01 × 10⁹, and no other eval
base used by the project reaches these seeds.

## How to read them

- **Standard errors.** A win rate over 476 seeds has SE √(p(1−p)/476): ±0.023 at p = 0.5, ±0.019 at p = 0.75. A paired
  Δ has a smaller SE when the two arms win the same seeds: ±0.025–0.028 for checkpoint comparisons, ±0.007–0.025 for
  picker comparisons. Treat |Δ| < 2 SE as not shown.
- **Compare within a setting.** a and b differ by the pickers (+0.13 to +0.20 on these models). A setting-a number and a
  setting-b number of two different checkpoints do not compare.
- **Paired only on the same seeds.** Δ columns use the seeds both files share. Numbers from different seed sets (the
  milestone set, the held-out set, the 96-seed in-run evals) can be compared as rates, not paired.
- **Selection bias (winner's curse).** The pick takes the best of many candidates on the milestone seeds. Its milestone
  number is biased upward. NMA 30 M, the reference, was itself picked on the same seeds. The held-out set measures the
  pick without that bias, and **the held-out number is the one to report**. Example: FIN-V5-s2-60M read 0.754 / 0.662 on
  the milestone seeds and 0.723 / 0.639 held-out. Comparing 11 candidates on the held-out set used it up in turn, so
  the released model's headline comes from a third set (`--seed 300000000`).
- **In-run evals** (`docs/curves/final.csv`, 96 seeds, SE ≈ ±0.05) show trends only.
- **Stacking is a trend, not a gate.** The pick is by win. The stacking columns are reported next to every win to check
  that the model plays the national main + stacked secondary route, not the elite-merge route (many elites, class main
  bonds 坚守 / 精准, natMain near 0).
