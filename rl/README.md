# rl/ (tools/rl in a game checkout): reinforcement learning for the AI seats

`setup.sh` copies this directory into a Stronghold-Protocol checkout as `tools/rl/`. Every path below is relative to
that checkout, because the scripts import the game's own `server/` modules.

The environment drives ordinary `Match`es through the bot-strategy `decide` hook. Every action goes through the same
validation as a human intent. The policy sees only that seat's view: the public view, the private view, the prep
scout and the static catalogue. Nothing here changes the game for players. The repository's `patches/` add the gated
seat-plan hooks and the strict-preview path of `fieldModel` (gated on `m._visibleOnly`). `server/match/bot.js` also
reads `ps._envPlanBonds`, which only `env.mjs` sets, for the human proxies of training.

Comments that cite `plan §…` refer to the maintainers' design notes. [docs/DESIGN.md](../docs/DESIGN.md) is the public
summary, and the docstring at the top of each script lists its switches.

## Environment (Node 24)

| File | |
|---|---|
| `spec.mjs` | observation / action layout (version 2 = ppo1 / ppo2; 3 = the hybrid line: + boss, teammates' bond bars and plans, band choice; 4 = full control), masks, AI-seat rules (让主盟约 `yieldMain`, 让合成 `yieldMerge`, card-pool settings), band conflicts |
| `env.mjs` | `SPEnv`: N matches per process in virtual time, `reset()` / `step(actions)`. Rewards: version 1, and version 2 (R2) with potential-based bond shaping. Modes `policy` / `teacher` / `legacy`; configs `solo`, `coopN`, `coopNl`, `coopNai`, `coopNh<k>`. Also: rehearsal per round kind (`rehearsal`, `bossRehearsal`, `mateRehearsal`), late-game snapshot starts, the seed queue, band choice with human proxies, the one-round-trip seat plan (version 4), the LP guard, the bond log / regime trace, per-round CPU timing. The deploy information set is `visibleOnly` (always on for deploy); `strictPreview` is optional and off; the `infoGuard` counters exist for tests only. |
| `server.mjs` | the environment behind a binary stdin/stdout protocol (frame format in its header) |
| `describe.mjs` | readable decision states (Chinese catalogue names) for the replay viewer |
| `record.mjs` | behaviour-cloning shards from the rule-enforcing legacy teacher |
| `infer.mjs` | plain-JS forward pass of an exported policy (reference) |
| `infer2.mjs` | exact rewrite of `infer.mjs`: folded embeddings, absent rows and masked actions skipped, split pointer heads; same outputs at about a quarter of the MACs |
| `infer3.mjs` | deployment forward: `infer2` plus per-seat incremental caching and multi-seat batching, bit-identical to a full recompute |
| `lookahead_table.mjs` | static tables for the bond look-ahead probe features |
| `bandobs.mjs` | band-decision observations for the offline band-head fit (`py/bandfit.py`) |
| `viewer.html` | static per-round replay viewer (policy and legacy side by side) |

Checks and diagnostics: `smoke.mjs` (random policy, and teacher == legacy), `smoke4.mjs` / `tcheck.mjs` (version 4 and
the seat plan), `vcheck.mjs` (dataIds), `bandtest.mjs` (band masks / conflicts), `snaptest.mjs` (snapshot starts
replay exactly), `lpguard.mjs` (the lp0 / own-LP regression check), `infoguard.mjs` (no hidden reads under `visibleOnly` / `strictPreview`), `pathdiff.mjs` (policy-picked vs forced band:
observation diff, the `_seat` bug's probe), `r2check-cap.mjs` (R2 terms against the engine's own counters), `parity2.mjs` /
`parity3.mjs` (JS forward variants agree), `bench.mjs` / `timing.mjs` / `maccount.mjs` / `chrome_bench.mjs` (throughput,
CPU per round, MACs, ms per decision under Chrome CPU throttling), `eval-rules.mjs` (the AI-seat rules one by one).

## Python (PyTorch)

| File | |
|---|---|
| `py/spenv.py` | client: `NodeEnv`, `AsyncPool` |
| `py/model.py`, `py/model2.py` | policy / value network (`SPNet`); `model2.SPNet2` = architecture-probe variants (width, residual LayerNorm trunk, entity self-attention, version 4 heads) |
| `py/data.py`, `py/train_bc.py` | shard reader, behaviour cloning |
| `py/train_ppo.py` | PPO: asynchronous collection, GAE, curricula / fixed mixes (`--mix`), KL anchor, value-only and lr warm-up, advantage normalisation, update/collection overlap, PLR, SIL, late-game starts, band choice and `--freeze-band`, reward version 2, TensorBoard (`tb.py`) |
| `py/eval.py`, `py/compare.py`, `py/stats.py` | same-seed evaluation of a checkpoint / the teacher / production legacy; paired comparison (fa: mean, IQM, bootstrap CI; win: Wilson, McNemar) |
| `py/eval_cross.py` | cross-play evaluation: chosen seats play one checkpoint, the others another (forked from an older `eval.py`) |
| `py/eval_report.py`, `py/replay.py` | self-contained HTML evaluation pages; decision replays for `viewer.html` |
| `py/export.py`, `py/parity.py` | export for the JS forward and its parity check against PyTorch |
| `py/surgery.py` | function-preserving growth of a checkpoint into a newer observation layout (zero-initialised new inputs) |
| `py/band_table.py`, `py/bandfit.py`, `py/bandxplant.py` | band strength table from paired seeds; offline fit of the band head to it; transplant of the fitted head into another checkpoint |
| `py/sharpen.py` | scale the final logit layers (temperature) of a checkpoint |
| `py/kl_drift.py`, `py/twin.py`, `py/bc_probe.py`, `py/bc_probe2.py`, `py/bc_probe_variants.py`, `py/prep_stats.py`, `py/bench_random.py`, `py/tb_from_jsonl.py` | diagnostics: drift by decision kind, label ambiguity, architecture probes, decisions per prep, throughput, TensorBoard from logs |

Record layout of `record.mjs`: f16 obs[D] · u8 mask bits[⌈A/8⌉] · u8 action · u8 kind · u16 0 · u32 trajectory · f32
reward since the trajectory's previous record. Trajectory ends go to `<tag>.terms.jsonl` and the spec to
`<tag>.meta.json`.
