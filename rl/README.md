# rl/ (tools/rl in a game checkout): reinforcement learning for the AI seats

`setup.sh` copies this directory into a Stronghold-Protocol checkout as `tools/rl/`. Every path below is relative to
that checkout, because the scripts import the game's own `server/` modules.

The environment drives ordinary `Match`es through the bot-strategy `decide` hook. Every action goes through the same
validation as a human intent. The policy sees only that seat's view: the public view, the private view, the prep
scout and the static catalogue. Nothing here changes the game for players. The repository's `patches/` add the gated
seat-plan hooks, the strict-preview path of `fieldModel` (gated on `m._visibleOnly`) and the training-only exports the
pickers, the label generators and the teammate proxies call (gated on fields the server never sets).

Comments that cite `plan §…` refer to the maintainers' design notes. [docs/DESIGN.md](../docs/DESIGN.md) is the public
summary, and the docstring at the top of each script lists its switches.

## Environment and inference (Node 24)

| File | |
|---|---|
| `spec.mjs` | observation / action layout (version 2 = ppo1 / ppo2; 3 = the hybrid line and the final models: + boss, teammates' bond bars and plans, band choice, `ext.board`; 4 = full control), masks, AI-seat rules (让主盟约 `yieldMain`, 让合成 `yieldMerge`, card-pool settings), band conflicts, `Encoder.boardSlice` for the pickers |
| `env.mjs` | `SPEnv`: N matches per process in virtual time, `reset()` / `step(actions)`. Rewards: version 1, version 2 (R2) with potential-based bond shaping, the national-plan terms. Modes `policy` / `teacher` / `legacy`; configs `solo`, `coopN`, `coopNl`, `coopNai`, `coopNh<k>`. Options: `natPlan` (the plan blocks, the rule plan, the plan reward, the national-route teacher), `humanProxy` (teammate proxies), `normalPicker` (c9 over the candidate pool), `xPlace` (xboss-v3), `pickerHead` (the older boss picker), the label hooks `pLabel` / `pBossLabel` / `xLabel`, rehearsal per round kind, late-game snapshot starts, the bond log, `visibleOnly` / `strictPreview`. Every option is off by default. |
| `server.mjs` | the environment behind a binary stdin/stdout protocol (frame format in its header) |
| `infer2.mjs` | the MLP's JS forward: folded embeddings, absent rows and masked actions skipped, split pointer heads, plan-slot rows; only + − × ÷ |
| `infer5.mjs` | v5's JS forward (`SPPolicy5`), also the encoder of the c9 picker head |
| `infer.mjs` | the reference JS forward of the MLP (used by `py/parity.py`) |
| `infer3.mjs` | `infer2` plus per-seat incremental caching and multi-seat batching, bit-identical to a full recompute (no plan-slot exports) |
| `infer2at.mjs` | `infer2` plus the residual entity attention of the AT arm |
| `proxies.mjs` | teammate proxies for human seats: legacy, the national-route bot, the elite plan, recipe AIs, noisy legacy, weighted mixes |
| `trackm/` | the route bots (`routebots.mjs`: national / class procurement on legacy's buy loop; the national-plan teacher), full-match runner `routematch.mjs`, the nation order and community pairs `recipes-M.json` |
| `xplace.mjs` | xboss-v3: the boss-round QNet scorer (`QNetJS`) and the g3b generator over the candidate pool, in plain JS |
| `xpool.mjs` | the deploy-legal boss-round candidate pool (no battle built) |
| `xfeat.mjs` | placement features shared by the scorer, the generator and the label tools |
| `xsearch.mjs` / `xteach.mjs` / `xlabel.mjs` | training-only search labels for the boss rounds (hill-climb / CEM / PUCT teachers on salted rehearsal seeds) |
| `xlineup.mjs` | the lineup ceiling oracle (diagnostics) |
| `picker.mjs` | the older boss-round picker head on the policy trunk (`pickerHead`) |
| `posvalue.mjs`, `picker_reencode.mjs`, `recipes.mjs`, `recipes.json` | placement diagnostics, picker label re-encoding, community recipe bots |
| `record.mjs` | behaviour-cloning shards from the teacher (rule-enforcing legacy, or the national-route bot with `--nat-plan`) |
| `describe.mjs`, `viewer.html` | readable decision states (Chinese catalogue names) and a static per-round replay viewer |
| `lookahead_table.mjs` | static tables for the bond look-ahead features and v5's unit–bond membership |
| `bandobs.mjs`, `bandobs2.mjs` | band-decision observations for the offline band-head fits |
| `release/` | `export_release.py` (release export + JS-vs-torch parity), `play.mjs` (play a release in JS with both pickers), `xplace_arg.py` ([RELEASE.md](../RELEASE.md)) |
| `final/` | the final pipeline ([final/README.md](final/README.md)) |

Checks and diagnostics: `smoke.mjs` (random policy, and teacher == legacy), `smoke4.mjs` / `tcheck.mjs` (version 4 and
the seat plan), `vcheck.mjs` (dataIds), `bandtest.mjs` (band masks / conflicts), `snaptest.mjs` (snapshot starts
replay exactly), `lpguard.mjs` (the lp0 / own-LP regression check), `infoguard.mjs` (no hidden reads under
`visibleOnly` / `strictPreview`), `pathdiff.mjs` (policy-picked vs forced band), `r2check.mjs` / `r2check-cap.mjs` (R2
terms against the engine's own counters), `natcheck.mjs` (natPlan smoke), `parity2.mjs` / `parity3.mjs` /
`parity5.mjs` / `p5parity.mjs` / `xqtest.mjs` (JS forwards against each other or against torch dumps), `bench.mjs` /
`timing.mjs` / `maccount.mjs` / `jsbench.mjs` / `chrome_bench.mjs` (throughput, CPU per round, MACs, ms per decision),
`eval-rules.mjs` (the AI-seat rules one by one).

## Python (PyTorch)

| File | |
|---|---|
| `py/spenv.py` | client: `NodeEnv`, `AsyncPool` |
| `py/model.py` | the MLP policy / value network (`SPNet`), plan blocks and plan-slot rows included |
| `py/model5.py` | v5 (`SPNet5`): entity transformer with unit–bond membership bias and a separate training-only critic |
| `py/model2.py` | architecture-probe variants (width, residual LayerNorm trunk, entity attention, separate critic trunk, version 4 heads) |
| `py/data.py`, `py/train_bc.py`, `py/train_bc5.py` | shard reader, behaviour cloning (MLP; v5 with optional deterministic algorithms) |
| `py/train_ppo.py` | PPO: asynchronous collection, GAE, fixed mixes and the auto1 / hard1 curricula (`--mix`, `--auto-*`), KL anchor, value-only and lr warm-up, advantage normalisation, update / collection overlap, the recurrent path (`--seq-len`, `--burn-in`), `--amp` / `--micro` / `--fwd-chunk` / `--clip-sep`, `--nat-plan`, `--human-proxy`, the older boss picker in the loop, TensorBoard |
| `py/eval.py` | same-seed evaluation of a checkpoint / the teacher / production legacy, with the pickers (`--npicker-*`, `--xplace`), teammate proxies, the bond log and the label hooks |
| `py/stackrep.py`, `py/coopstats.py` | stacking metrics; co-op collision / replans ([records/final/README.md](../records/final/README.md)) |
| `py/compare.py`, `py/stats.py`, `py/eval_cross.py`, `py/eval_report.py`, `py/replay.py`, `py/eval_dump.py` | paired comparison, statistics, cross-play, HTML reports, decision replays, eval with an observation dump |
| `py/export.py`, `py/export5.py`, `py/parity.py`, `py/parity5_dump.py` | JS exports of the MLP / v5 policy and their parity checks |
| `py/picker_train5j.py`, `py/export_p5.py`, `py/p5parity.py`, `py/poolgate.py` | the normal-round picker (c9): training on pool labels, export, parity, the offline gate |
| `py/picker_train.py` | the older boss-round picker head |
| `py/winutil.py` | the win utility: P(win) as a function of the Final Assault share |
| `py/xtrain.py`, `py/xkfold.py`, `py/xbake.py`, `py/xnoise.py`, `py/xbossgap.py`, `py/xqtest.py` | the boss-round scorer and generator: training, k-fold gate, teacher bake-off, seed-noise and headroom studies, parity |
| `py/init5.py`, `py/addrnn.py`, `py/rnntest.py` | v5 initialisation, adding a zero-initialised recurrent branch, its sequence / step check |
| `py/surgery.py`, `py/widen.py`, `py/critic.py`, `py/natsurg.py`, `py/swaphead.py`, `py/obsmerge.py`, `py/obsrec.py`, `py/dumpref.py` | function-preserving checkpoint surgery: observation growth, width, a separate critic, plan blocks, head swaps; observation tools |
| `py/band_table.py`, `py/band_table2.py`, `py/bandfit.py`, `py/bandfit2.py`, `py/bandxplant.py`, `py/combo.py`, `py/actrep.py` | band strength tables, offline band-head fits, transplants, team-combo screen, action reports |
| `py/sharpen.py`, `py/kl_drift.py`, `py/twin.py`, `py/bc_probe.py`, `py/bc_probe2.py`, `py/bc_probe_variants.py`, `py/prep_stats.py`, `py/bench_random.py`, `py/macs.py`, `py/memprobe.py`, `py/tb_from_jsonl.py`, `py/tb.py` | diagnostics: logit temperature, drift by decision kind, label ambiguity, architecture probes, decisions per prep, throughput, MACs, memory, TensorBoard |

Record layout of `record.mjs`: f16 obs[D] · u8 mask bits[⌈A/8⌉] · u8 action · u8 kind · u16 0 · u32 trajectory · f32
reward since the trajectory's previous record. Trajectory ends go to `<tag>.terms.jsonl` and the spec to
`<tag>.meta.json`.
