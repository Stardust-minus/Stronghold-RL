# Findings

The key findings and fixes so far. Dates are 2026-11 (UTC+8). Unless stated otherwise, numbers are team win rates on
paired seeds, greedy policy, RL seat r0. Run names are explained in [EXPERIMENTS.md](EXPERIMENTS.md), and the raw
numbers are in `docs/curves/`.

## 1. The `_seat` lp0 bug (found and fixed 11-12)

**Bug.** `SPEnv._seat` creates a seat's reward bookkeeping at the seat's first decision. When the policy picks the
band, that first decision is the band pick, and at that point the band has not set the seat's LP yet (`ps.lp = 0`). So
`lp0 = 1` for the rest of the match, with three effects:

- the own-LP observation field (`lp / lp0`) showed the **raw** LP, up to 45, instead of a fraction;
- under reward version 1 the LP term (`ΔLP / lp0`) was **20–45× too large**;
- the first decision after the pick collected **+ΔLP equal to the band's starting LP** for free: 45 for sarkazb, 25 for
  most bands.

**Effect.**

- About 70 % of training seats were affected from ppo3 on (every seat whose band the policy picked).
- So was every evaluation with `--band-policy 1`: ppo3h, fb, fr and the 6.0 M init.
- ppo2, the teacher, production legacy and the forced-band table were not affected.
- The free +45 is the main reason PPO collapsed the band head onto sarkazb (ppo3h @ 10 M: 96 / 96 solo picks, solo
  ABYSS 0 %).

**Fix.** The seat's LP bookkeeping is re-initialised at its first decision after the band is set, and `lp0` is always
the band's starting LP. With the fix, the policy-picked and forced-band paths give identical observations
(`pathdiff.mjs`).

A second bug of the same class turned up in the audit that followed. A late-game (snapshot) start took `lp0` from
the LP at restore, so in 30 % of training matches the observation LP read 1.0 all match. It was fixed the same way
(`lp0 = gd.startLp(band)`). `lpguard.mjs` is the regression check: band policy 1 in every config, random legal actions,
snapshot starts. The env's guard must stay silent.

**Numbers it invalidated.**

- coop4ai:ABYSS fell from 0.354 to 0.135 for the 6.0 M init under the fixed env (96 seeds; 192 seeds: 0.130).
- The fb / fr co-op numbers (0.35–0.38) are inflated the same way.

A probe on the fixed env showed why the bugged numbers were higher. It scaled the LP field by k, on 96 seeds of
coop4ai:ABYSS:

| k | 1 | 3 | 10 | 25 | raw |
|---|---|---|---|---|---|
| win | 0.135 | 0.083 | 0.146 | 0.344 | 0.354 |

Raw LP reproduces the bugged result. The model had learned two regimes:

- **Normalised-LP regime: defensive.** Many refreshes, low shop level.
- **Raw-LP regime: invest-and-stack.** Shop level 1.6–2.2 at R3–R5 instead of 1.0, fewer early refreshes, more leaks
  mid-game, Σ stacking layers 296 vs 200 at R14, team LP 40 vs 35 at the Final Assault.

The decision was to keep normalised LP and fix the incentives instead (§2).

## 2. Reward redesign (R2) and the invest-and-stack regime

**Problems with version 1.**

1. The LP term divided by the seat's own starting LP, so a 45-LP band felt about 40 % less pain per LP lost.
2. The win bonus went only to seats still alive at the end.
3. On ABYSS, the survival and LP terms dominated the return variance; the win and Final Assault signals were weak.
4. The Final Assault pool scales with the number of living players, which rewards a weak seat for dying early (§6).

**R2** fixes these (full table in [DESIGN.md](DESIGN.md) §7):

- a team win of 3.0, paid to every seat;
- Final Assault damage, team 1.5 and own 0.5;
- leaks and 联防 saves;
- LP against a constant L_ref = 28;
- elimination −0.5;
- no round reward.

The first leak weight (0.5 per leak, uncapped) cost each seat −1.0 to −1.7 per match, which pushes towards quick,
low-ceiling play. It was cut to 0.15 per leak with the per-round cap of 10, and saves to 0.075.

**Stacking.** The owner wanted bond stacking rewarded without dictating play. So the potential-based term stays
(Φ = 0.06 · tiers + 0.3 · Σ layers / 100, uncapped, which cannot change the optimal policy), and m3 adds a small direct
reward of 0.002 per new layer. At 10 M, m3 against m2 (96 seeds, r0b3):

| config | m2 (no layer term) | m3 (+ layer 0.002) |
|---|---|---|
| coop4ai:ABYSS | 0.333 | 0.427 (Δ +0.094, 95 % CI +0.01 … +0.18) |
| solo:ABYSS | 0.156 | 0.219 |
| coop4:ABYSS | 0.104 | 0.135 |

Under R2 and normalised LP, both arms learned invest-and-stack:

- shop level-ups from R4, fewer refreshes, fewer leaks;
- team LP 52 vs 35 at the Final Assault start;
- Σ layers per seat about 290 at R14 at 10 M, rising to 363 (solo) by 20 M.

**Bonds.** The models play elite operators (the 绝技 bond counts every elite) plus 坚守 / 精准. The teacher plays
nation bonds with about two elites. Σ layers correlate with the Final Assault share at 0.37–0.82, depending on the
config.

## 3. Band head: fit offline, then freeze

- Trained by PPO, the band head drifted within about 1 M steps despite a KL anchor of 0.01. The band strength table
  showed that the drift made ABYSS worse for most bands. Part of the drift was the `_seat` bonus (§1), but the head
  also learned too little from too few band samples: warm-started in-run, its cross-entropy stayed at ln 38, i.e. uniform.
- Now the head is fitted offline to a paired-seed table and frozen (`--freeze-band`; [DESIGN.md](DESIGN.md) §6).
  70 % of the RL seats use it; the other 30 % open with a uniformly random band, so the policy keeps seeing every band.
- Table, ppo2 @ 15 M, 48 paired seeds per band and config:
  - band choice matters on ABYSS. Solo: ioleta fa 0.637 vs the random pick's 0.502 (+0.135, CI +0.055 … +0.216);
    worst quintus, 0.259. Co-op: amedic 0.400 vs 0.272.
  - On HARD the differences are within noise.
  - Only ioleta, amedic and amiya are significant on ABYSS. Forcing ioleta gives solo:ABYSS 16.7 % against 6.3 % for
    a random band (ppo2 economy).
- The table is re-measured and the head re-fitted periodically; PPO never updates it.

## 4. Placement studies

Legacy placement is the main lever the policy does not control.

- **Legacy's blind spots.**
  - 43 of the 266 piece entries carry positional traits (in front, same row, the four neighbours, …). Legacy's
    candidate layouts are purely geometric and never read those keys.
  - It does not recognise IN_BATTLE engines.
  - It never redeploys on its own.
  - Production legacy compensates partly by rehearsing three layouts with the live battle seed (r3).
- **Ceilings with a peek** (search that simulates layouts with the real battle seed).
  - With the ppo2 economy and an oracle placement, solo:ABYSS rises from 4.2 % to 8–12 %. Adding the best band takes
    it to 26–30 %.
  - With the 6.0 M lineups, hill-climbing K16 from R1 gave coop4ai +0.20 fa and +9 points of win, at about 0.95 s per
    seat-prep.
  - On m3 @ 10 M (solo:ABYSS, 288 paired seeds): r0b3 0.181, production r3 0.208 (+0.028), peek K16 0.306
    (**+0.125 ± 0.023** over r0b3).
  - The peek reads the live seed, so it is a teacher, not a deployable policy. The learnable share is probably about
    +4–6 points.
- **Per-piece BC from searched layouts failed** (v6).
  - Data: about 3 M samples (40 recording jobs) of the searched layouts, replayed piece by piece in a fixed order.
  - Result: the economy matched m3 (Δ −0.011), but learned placement lost **−0.080** in solo:ABYSS (288 seeds,
    p = 0.0001). Only 36 % of held-out pieces matched the teacher's tile and facing.
  - Why: the peek gain belongs to the whole layout and to the seed. Imitating it piece by piece cannot carry it.
  - The same lesson appeared earlier in version 4 BC. Labels with better accuracy (a set loss over the pieces still to
    move, final-tile targets) played worse: solo HARD 0.292 → 0.104 (p 5e-4).
  - Next attempt, if any: pick a whole layout among legacy's coherent candidates (v5-style), or DAgger-style
    relabelling. Not more per-piece BC.
- **Trait heuristics** (re-placing pieces by their positional traits) were slightly worse on the models' lineups, and
  were dropped.

## 5. Search for the economy: no-go

- **Rollout search over economy decisions** (ppo2 greedy, solo:ABYSS, 170 decisions, about 10 k rollouts). It tried
  the top-5 actions, 8 whole-prep macros and switching to one of the seat's own three largest bonds. Nothing beat greedy; the 95 % bound is
  ≤ +0.004 fa per decision.
- **Final-Assault-prep joint search** (R13–R14). Naive search gained +0.188 per game that reached R14 in solo. An
  honest check then swapped both the shop RNG and the battle seeds to unseen futures:
  - economy-only Δ at R13: +0.000 ± 0.017; at R14: −0.006. The whole naive gain was luck borrowed from the real future.
  - Only the placement part (hill-climb at R14) survived unseen battle seeds, at +0.075. That belongs to §4.
  - No label set was made.
- **Opening plans** (a fixed bond plan biasing purchases from R1): about +0.03 on fresh seeds, below the bar.
- **Recipe bots** (five community-style stacking strategies: 谢拉格, 卡西米尔, 叙拉古, 阿戈尔, 萨尔贡): none beat legacy. Layer
  p90 was only 120–190, so they are not usable as teachers either.

## 6. One model seat in co-op: a structural ceiling

1-RL co-op (`coop4l` / `coop2l`: the model plus legacy rule-AI or human-proxy mates) sits at about 0.10–0.13. Solo is
at 0.30–0.39 and four model seats at 0.56–0.73. A diagnostic split the gap:

- **The pool.** The Final Assault pool is bloodPoint × the players alive at its start (`bossPoolHp` in
  `finalAssault.js`), and bots count.
- **The mates.** A rule-AI mate deals about 0.55 of its share, so every surviving mate is a net −0.45. Final Assaults
  with 4 alive win 5 %; with 2–3 alive, 27 %.
- **The enemies.** Co-op ABYSS enemy HP is 2.1–2.5× solo from R7 on.
- **The RL seat's own part.** In co-op it dies before the Final Assault 34 % of the time, leaks 42 times (solo: 7) and
  has 286 layers at R14 (solo: 363).
- **Mate rehearsal.** With the mates at production rehearsal r3, coop4l is 0.146 against 0.104 all-r0 (m3s @ 20 M,
  96 seeds), so +4 points come from the mates.

The in-RL lever is a larger share of one-RL co-op in the mix. S (v3s) against m3s (v3), 288 seeds:

| config | Δ win | ± SE |
|---|---|---|
| solo | +0.024 | 0.028 |
| coop4l | +0.010 | 0.023 |
| coop2l | +0.021 | 0.023 |

All within noise. SC (v3sc, heavier co-op) matched S at 25 M on solo (+0.002 ± 0.022).

Product scenarios at m3s @ 20 M: one human proxy + 3 model seats wins 0.323, one proxy + 1 model seat 0.083. The
proxies are weak bots that die before the Final Assault 27–52 % of the time, so these numbers understate play next to
real humans. 联防 assignment is automatic (`planUnite`: leaks go to the players who had a perfect round), not a policy
decision.

The remaining levers are game-side and await the owner's decision:

- use the model as the teammate AI (four model seats already reach ≥ 0.56);
- a stronger rule AI;
- a Final Assault pool balance for bots.

## 7. Information audit: the deploy information set

Since 11-13 the deployed AI must run on **zero extra information**: nothing a player cannot see, and no rehearsal or
simulation of any kind, the boss rounds included. Training may still use privileged signals, e.g. search as a
teacher.

- **Headline.** It is now solo:ABYSS at b0 with `--visible-only`. b3 (legacy's boss-round rehearsal) is reported for
  continuity only.
- **Audit.** It checked every input on the inference path and found five reads beyond what a player sees:
  - A1: the 让合成 human-pair flag read the humans' hands, not only the board the prep scout shows;
  - A2: the bounty kill chance (engine wave internals);
  - A3: the hidden boss id before its round;
  - A4: the enemy spawn time (not in the UI preview);
  - B1: legacy placement modelled the field from the real spawns: route index, spawn modifiers, round HP scale.
- **The owner's rule.** Enemy information, HP and the 机变 kill chance are acceptable. The hidden boss and other
  players' hands are not.
- **The switches.**
  - `visibleOnly` fixes A1 and A3. It is always on for deploy.
  - `strictPreview` covers A2, A4 and B1 (B1 through `patches/0002`). It stays optional and off.
  - Neither changes the observation layout or the dataId ([DESIGN.md](DESIGN.md) §3).
  - `infoguard.mjs` is the check: it counts hidden reads during an RL seat's encode and placement, and the positive
    control (`off`) must find some.
- **Impact.** Existing checkpoints evaluated with `visibleOnly` lose nothing measurable. solo:ABYSS b0, 480 paired
  seeds:

  | checkpoint | plain | visible-only | Δ paired | seeds that change |
  |---|---|---|---|---|
  | S @ 30 M | 0.292 | 0.277 | −0.015 ± 0.015 | 51 |
  | SK @ 30 M | 0.319 | 0.329 | +0.010 ± 0.016 | 61 |

  The final recipe will train with `--visible-only` at b0.
- **Placement.** The model must later learn boss-round placement itself, choosing among legacy's coherent candidate
  layouts. The plan is to distil it from b3 picks, then fine-tune with PPO.

## 8. Restart safety

On 11-13 at 06:01 the host's unattended upgrades restarted both training services (needrestart). The launcher started
again from its original init (the 10 M checkpoint), not from the 38.0 M / 34.0 M checkpoints the runs had reached. This
was caught and resumed by hand 18 minutes later.

The lesson is in `scripts/train.sh`: on any start it resumes from the newest **complete** `model-<steps>.pt` in the run
directory. A checkpoint smaller than 95 % of its predecessor counts as cut short and is skipped. The match-seed offset
must also move on a resume (`--seed-offset`), so a resumed run does not replay its first matches.

Smaller operational lessons:

- an evaluation that ran out of GPU memory still exited 0, so monitors also grep the logs for Traceback / OOM / Killed;
- one-round-trip plans make a few rule violations inherent, so the violation alarm is a rate (`--viol-rate`), not 0.

## 9. Other lessons

- **96 seeds mislead.**
  - S @ 20 M read 0.354 (solo) and 0.219 (coop4l, mates r3) on 96 seeds, but 0.309 and 0.132 on 288.
  - The in-run 96-seed S @ 30 M against m3s @ 30 M read +0.115 ± 0.055 on solo:ABYSS b3; on 480 seeds it is
    +0.008 ± 0.022.
  - Decisions now use 288 / 480 paired seeds.
- **Full control (version 4) did not learn.**
  - Setup: the policy also placed every piece (28 tiles × 4 facings), with about 28 decisions per prep instead of 7.
  - PPO never improved on its BC start in any recipe tried: entropy / KL / logit sharpening (R1–R3; R1 held the BC
    level, the others fell below it), expert iteration at the boss rounds, a per-round semi-MDP discount.
  - The 15 M main run stayed flat.
  - The diagnosis was credit-assignment noise on the tile actions, so the hybrid design (§1 of DESIGN) replaced it.
- **The privileged critic did not help.** A detached critic with privileged inputs (P), or the same with those inputs
  zeroed (P0), leaned negative on ABYSS at 20 M: P solo fa −0.053 ± 0.024 (96 seeds). P, P0 and P + KL were stopped.
  That code is not part of this snapshot.
- **KL 0.003 instead of 0.01.** At 20 M (KL arm, from m3) it was neutral. SK (from S) on solo:ABYSS, 480 seeds, b3:
  −0.013 ± 0.022 against S at 25 M, then **+0.048 ± 0.024** at 30 M (b0: +0.027 ± 0.024). It is borderline, and one
  more matched checkpoint will decide it.
- **Training with rehearsal** (R3: r3 inside training) ran at about 300 steps/s instead of 650. Rehearsal also adds
  nothing to solo at evaluation (m3 @ 10 M: +0.000). R3 was stopped at 10.2 M.
