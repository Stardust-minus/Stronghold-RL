# Model releases

The trained models are published as GitHub release assets, not in git. One release holds one policy (MLP or v5) and the
two deploy pickers it was evaluated with. `rl/release/export_release.py` builds the release directory from the
checkpoints and checks it ([docs/RUNBOOK.md](docs/RUNBOOK.md) §6).

<!-- pick:BEGIN release -->
**First release: `FIN-MLP-s2-70M`**: MLP, 1.55 M parameters, 70,000,000 steps (NM 5 M → NMA 60 M → co-op stage 70 M).
Asset `stronghold-rl-FIN-MLP-s2-70M.tar.gz` (17.1 MB, md5 `4d535b73d8fba3c932bcd1720d11efeb`). Repository commit
`f3e0730` (`repoCommit` in `manifest.json`); the files it runs on (`manifest.json` → `code`) are unchanged in later
commits. Numbers and the choice of model: [docs/RESULTS.md](docs/RESULTS.md) "Final result".
<!-- pick:END release -->

## Asset layout

```text
<name>/
  manifest.json                        what this is: architecture, steps, dataId, env settings, eval flags, files (md5, bytes),
                                       repository and game commits, source md5s, weight md5s, parity and cost summary
  MD5SUMS                              md5sum -c MD5SUMS
  parity.json                          JS vs torch on the decisions of the check matches, picker counters, ms per decision
  policy/policy.json + policy.bin      the policy for the JS forward: infer2.mjs (MLP, format sp-policy/1) or infer5.mjs
                                       (v5, sp-policy5/1); float32 little-endian tensors; v5's separate critic is not
                                       included (the MLP's small value head is, 33k parameters, unused at play)
  policy/model.pt                      the PyTorch checkpoint (eval.py, fine-tuning); plain tensors and strings, so it loads
                                       with torch.load(..., weights_only=True); training-host paths in its args are
                                       replaced by <work>/...
  pickers/c9/c9-js.json + c9-js.bin    normal-round picker c9 (rounds 1–13), an infer5.mjs 'p5-picker' head
  pickers/xboss-v3/dw.json             boss-round scorer Dw (QNet weights, base64 float32)
  pickers/xboss-v3/g3b.json            boss-round pool generator g3b
  pickers/xboss-v3/xplace.json         the xboss-v3 options; file names relative to this directory
```

The picker files are the ones the final evals used. Only metadata strings changed (training-host paths); `manifest.json`
lists the source md5s and the weight md5s, which are equal before and after.

## Requirements

- A game checkout prepared by `./setup.sh` from this repository at the commit in `manifest.json` (`repoCommit`). The
  policy reads only observations of its own `dataId` (`act2autochess-75d38e72a6cf`): the game data must match the
  pinned game commit.
- Node ≥ 24 to play. Python 3.12 + PyTorch only for `eval.py` and the parity check.
- No GPU. One CPU core plays a match ([docs/RUNBOOK.md](docs/RUNBOOK.md) §8.5).

## Verify

```sh
cd <name> && md5sum -c MD5SUMS && cd -
node tools/rl/release/play.mjs <name> --matches 2        # warns if your tools/rl or bot.js differ from the release's code
```

To repeat the parity check on your machine, re-export from `policy/model.pt` with `export_release.py` (it needs the
two picker directories of the release as `--c9` / `--xboss`).

## Run

**In JS, with both pickers** (no Python; the headline protocol, greedy, in-process):

```sh
node tools/rl/release/play.mjs <name> --configs solo:ABYSS,coop4h1:ABYSS --matches 32 --out play.json
python tools/rl/py/stackrep.py play.json                  # stacking metrics of the played matches
```

`play.mjs` builds `SPEnv` with the settings in `manifest.json` → `env`: version 3 observation with the national plan
(`natPlan`), band from the plan (band policy 0), no rehearsal (`rehearsal 0`, `bossRehearsal 0`), the normal-round
picker (`normalPicker`: c9 over the candidate pool, rule `reg`, margin 0.5) and the boss-round placement (`xPlace`).
In co-op the human seat is the legacy bot at its production rehearsal (`mateRehearsal 3`). Every picker decision counts
the battles it built; the count must stay 0.

**In PyTorch, as evaluated** (`eval.py`; the commands of [docs/RUNBOOK.md](docs/RUNBOOK.md) §5):

```sh
python tools/rl/py/eval.py --ckpt <name>/policy/model.pt --configs solo:ABYSS --matches 480 --chunks 28 --seed 200000000 \
  --greedy --bond-log --rehearsal 0 --boss-rehearsal 0 --band-policy 0 --nat-plan '<manifest.env.natPlan>' \
  --npicker-head <name>/pickers/c9/c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool \
  --xplace "$(python tools/rl/release/xplace_arg.py <name>)" --out heldout-solo.json
```

With `--seed 200000000` this repeats the held-out solo:ABYSS evaluation (co-op: `--configs coop4h1:ABYSS
--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}'`). Do not add `--picker-head` / `--picker-trunk`.

**The policy alone, in your own JS code:**

```js
import { SPPolicy2 } from './tools/rl/infer2.mjs';        // v5: import { SPPolicy5 } from './tools/rl/infer5.mjs'
const net = SPPolicy2.load('<name>/policy/policy');       // reads policy.json + policy.bin
const { logits } = net.forward(obs, mask, { value: false });
// obs: Float32Array(net.D), the seat's observation from env.mjs's encoder with manifest.env's settings;
// mask: Uint8Array(net.A), the legal actions. Greedy: the legal action with the largest logit.
```

The pickers are not standalone modules. They run inside `env.mjs` (`_normalPick`, `_xHook`) and build their candidate
pools with the `bot.js` functions of `patches/0003` / `0004`. A game-server integration needs the same hooks.

## What the model may use

Everything a player sees in the game: monster information and HP, the 机变 kill chance, teammates' cards and flags, the
boss. No search, no simulation, no layout rehearsal and no live battle seed at decision time
([docs/DESIGN.md](docs/DESIGN.md) §2.1).

## Licence and attribution

- Code in this repository: GPL-3.0-or-later ([LICENSE](LICENSE)).
- Model files (the release assets): GPL-3.0-or-later, the same licence as the code.
- The model files contain weights and layout metadata (catalogue ids, block layouts), no game art, text, audio or data
  files. They were trained on simulated matches of Stronghold Protocol, an unofficial, non-commercial fan remake of the
  Arknights auto-chess mode. Arknights names and data are © their owners. This project is not affiliated with or
  endorsed by Hypergryph or Yostar.

## Release notes (template)

```text
Stronghold-RL <name>

Policy: <arch> (<params> parameters, <steps> steps), trained by the final pipeline (rl/final/): national-route BC →
NM (5 M) → NMA (auto1 to 60 M)<extension><co-op stage>. Pickers: c9 (normal rounds), xboss-v3 (boss rounds).

Held-out (setting b, 476 seeds each): solo:ABYSS <x> ± <se>, coop4h1:ABYSS <y> ± <se>.
Stacking (held-out): natMain <a> / <b>, natMainT2 <c> / <d>, Σ layers <e> / <f>, elites <g> / <h>.
Picker lift (b − a, milestone seeds): solo <+l1>, co-op <+l2>.

Assets: stronghold-rl-<name>.tar.gz (<size>), md5 <md5>. manifest.json inside lists every file.
Parity: JS vs torch max |Δlogit| <p>, greedy agreement <q> on <n> decisions.
Cost on one CPU core: policy <ms> ms, c9 <ms> ms, xboss-v3 <ms> ms per decision. No GPU needed.
Code: repository commit <sha>. Licence of the model files: <licence>.
```
