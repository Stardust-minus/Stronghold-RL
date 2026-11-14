"""PPO from the behaviour-cloned policy (plan §8.9, B2).
python train_ppo.py --init runs/bc/bc.pt --out runs/ppo-1 [--procs 24 --envs 16 --max-steps 3e7 --hours 24]
Asynchronous collection (AsyncPool: each environment process is stepped as soon as its batch is in), one trajectory per
policy seat, GAE per trajectory (a segment that is still running bootstraps from the value of its next decision),
clipped PPO with masked logits, an optional KL pull towards the BC policy that decays, a curriculum NORMAL → HARD →
ABYSS with legacy teammates mixed in and humans marked at random, checkpoints and same-seed evaluations (eval.py) on
the way. Logs: <out>/train.jsonl (one line per update), <out>/eval-<steps>.json."""
import argparse, json, math, os, re, subprocess, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn.functional as F
import contextlib
from spenv import NodeEnv, AsyncPool
from model import load, save
from tb import Board, TYPES
GROUPS = [f'{d}_{m}' for d in ('FUNNY', 'NORMAL', 'HARD', 'ABYSS') for m in ('solo', 'coop')]  # index = difficulty * 2 + co-op
torch.set_num_threads(4)  # the learner runs on the GPU; a wide CPU pool would only compete with the environments

def W(**kw):
    return [{'name': k.replace('_', ':'), 'w': v} for k, v in kw.items()]

STAGES = [
    {'name': 'normal', 'key': 'NORMAL', 'configs': W(solo_NORMAL=1, coop2_NORMAL=1, coop4_NORMAL=2, solo_HARD=0.25, coop4_HARD=0.5)},
    {'name': 'hard', 'key': 'HARD', 'configs': W(solo_NORMAL=0.2, coop4_NORMAL=0.3, solo_HARD=1, coop2_HARD=1, coop3_HARD=1, coop4_HARD=2, coop4_ABYSS=0.3)},
    {'name': 'abyss', 'key': 'ABYSS', 'configs': W(solo_NORMAL=0.1, coop4_NORMAL=0.1, solo_HARD=0.3, coop4_HARD=0.5, solo_ABYSS=1, coop2_ABYSS=1, coop4_ABYSS=2)},
]

# ppo2 (approved 11-12): one fixed mix by match, ABYSS 60 % / HARD 30 % / NORMAL 10 %
FIXED = [{'name': 'mix', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.15, coop2_ABYSS=0.15, coop4_ABYSS=0.30,
                                                      solo_HARD=0.075, coop2_HARD=0.075, coop3_HARD=0.05, coop4_HARD=0.10,
                                                      solo_NORMAL=0.03, coop2_NORMAL=0.03, coop4_NORMAL=0.04)}]

# v2 mix (11-12): 4-RL-seat co-op (the target, coop4ai = no humans, no legacy mates) ≈ 36 % of the ABYSS games
FIXED2 = [{'name': 'mix2', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.12, coop2_ABYSS=0.10, coop4_ABYSS=0.18, coop4ai_ABYSS=0.20,
                                                        solo_HARD=0.06, coop2_HARD=0.06, coop3_HARD=0.04, coop4_HARD=0.08, coop4ai_HARD=0.06,
                                                        solo_NORMAL=0.03, coop2_NORMAL=0.03, coop4_NORMAL=0.04)}]

# v3 mix (11-12, the target: one RL seat solo or with rule-AI / human-proxy mates): coop*l = ai_0 the only RL seat
FIXED3 = [{'name': 'mix3', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.25, coop4l_ABYSS=0.12, coop4_ABYSS=0.08, coop2l_ABYSS=0.07, coop2_ABYSS=0.05, coop4ai_ABYSS=0.08,
                                                        solo_HARD=0.08, coop4l_HARD=0.05, coop4_HARD=0.04, coop2l_HARD=0.03, coop2_HARD=0.02, coop3_HARD=0.01, coop4ai_HARD=0.02,
                                                        solo_NORMAL=0.03, coop2_NORMAL=0.03, coop4_NORMAL=0.04)}]
# v3s (11-12, pure-model deploy, arm S): solo-heavy, weights matched to training samples (a coop4ai match gives 4 RL
# seats of decisions, coop4 ~2, coop2 ~1.35, solo / coop*l 1): ABYSS .75, HARD .20, NORMAL .05 — ~55 % solo, ~28 % 1-RL co-op of ABYSS samples
FIXED3S = [{'name': 'mix3s', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.45, coop4l_ABYSS=0.15, coop2l_ABYSS=0.10, coop4ai_ABYSS=0.02, coop4_ABYSS=0.02, coop2_ABYSS=0.01, solo_HARD=0.10, coop4l_HARD=0.05, coop2l_HARD=0.03, coop4ai_HARD=0.02, solo_NORMAL=0.03, coop4l_NORMAL=0.02)}]
# v3sc (11-13, arm SC): v3s heavier on 1-RL co-op: ABYSS .78, HARD .17, NORMAL .05
FIXED3SC = [{'name': 'mix3sc', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.35, coop4l_ABYSS=0.25, coop2l_ABYSS=0.15, coop4ai_ABYSS=0.01, coop4_ABYSS=0.01, coop2_ABYSS=0.01, solo_HARD=0.08, coop4l_HARD=0.05, coop2l_HARD=0.03, coop4ai_HARD=0.01, solo_NORMAL=0.03, coop4l_NORMAL=0.02)}]
# solo1 (11-13, the owner: solo:ABYSS only; the small co-op share only against forgetting): ABYSS .80, HARD .15, NORMAL .05
FIXED_SOLO1 = [{'name': 'mixsolo1', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.70, coop4l_ABYSS=0.05, coop2l_ABYSS=0.03, coop4ai_ABYSS=0.02, solo_HARD=0.12, coop4l_HARD=0.03, solo_NORMAL=0.05)}]
# solo1w (11-13, from-scratch warm phase, first ~10M): solo1 shifted to easier difficulties — ABYSS .50, HARD .35, NORMAL .15
FIXED_SOLO1W = [{'name': 'mixsolo1w', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.44, coop4l_ABYSS=0.03, coop2l_ABYSS=0.02, coop4ai_ABYSS=0.01, solo_HARD=0.30, coop4l_HARD=0.05, solo_NORMAL=0.15)}]
# mp1 (11-13 rev 2, multiplayer-first, every empty seat filled by the model; --legacy-mates 0): game weights from SAMPLE targets
# solo 30 (ABYSS 23 / HARD 5 / NORMAL 2), humans+model 65 (ABYSS 49 / HARD 12 / NORMAL 4; coop4 2/3, coop2 1/4, coop3 1/12),
# rule-AI mates 5 (coop4l ABYSS 3, coop2l ABYSS 2), divided by measured samples per game (coop4 2.06x, coop2 1.46x, coop3 1.44x solo)
FIXED_MP1 = [{'name': 'mixmp1', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.3233, coop4_ABYSS=0.2233, coop2_ABYSS=0.1183, coop3_ABYSS=0.0399, coop4l_ABYSS=0.0431, coop2l_ABYSS=0.0292, solo_HARD=0.0703, coop4_HARD=0.0547, coop2_HARD=0.029, coop3_HARD=0.0098, solo_NORMAL=0.0281, coop4_NORMAL=0.0182, coop2_NORMAL=0.0097, coop3_NORMAL=0.0033)}]
# mp1w: the same room-type split (30 / 65 / 5) at ABYSS .50 / HARD .35 / NORMAL .15 (from-scratch warm phase)
FIXED_MP1W = [{'name': 'mixmp1w', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.2108, coop4_ABYSS=0.1481, coop2_ABYSS=0.0785, coop3_ABYSS=0.0264, coop4l_ABYSS=0.0215, coop2l_ABYSS=0.0146, solo_HARD=0.1476, coop4_HARD=0.1037, coop2_HARD=0.0549, coop3_HARD=0.0185, coop4l_HARD=0.0151, coop2l_HARD=0.0102, solo_NORMAL=0.0633, coop4_NORMAL=0.0444, coop2_NORMAL=0.0235, coop3_NORMAL=0.0079, coop4l_NORMAL=0.0065, coop2l_NORMAL=0.0044)}]
FIXED_AUTO1 = [{'name': 'mixauto1', 'key': 'ABYSS', 'configs': W(solo_NORMAL=0.15, solo_HARD=0.85, solo_ABYSS=0.0)}]  # (scr0) weights set per update by auto_mix()
FIXED_HARD1 = [{'name': 'mixhard1', 'key': 'HARD', 'configs': W(solo_HARD=1.0)}]  # (scr0) proxy task: solo HARD only
FIXED_MP2 = [{'name': 'mixmp2', 'key': 'ABYSS', 'configs': W(solo_ABYSS=0.4444, coop4_ABYSS=0.157, coop2_ABYSS=0.0832, coop3_ABYSS=0.0281, coop4l_ABYSS=0.0384, coop2l_ABYSS=0.026, solo_HARD=0.0966, coop4_HARD=0.0385, coop2_HARD=0.0204, coop3_HARD=0.0069, solo_NORMAL=0.0386, coop4_NORMAL=0.0128, coop2_NORMAL=0.0068, coop3_NORMAL=0.0023)}]  # (11-13) final phase: solo ~45% of samples, humans + model ~50%, coop*l ~5% (mp1's per-game sample factors)

def cpu_list(s):
    out = set()
    for part in s.split(','):
        lo, _, hi = part.partition('-'); out.update(range(int(lo), int(hi or lo) + 1))
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--procs', type=int, default=24); ap.add_argument('--envs', type=int, default=16)
ap.add_argument('--rollout', type=int, default=32768); ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--minibatch', type=int, default=4096); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--vf', type=float, default=0.5)
ap.add_argument('--ent', type=float, default=0.003); ap.add_argument('--gamma', type=float, default=0.997)
ap.add_argument('--lam', type=float, default=0.95); ap.add_argument('--target-kl', type=float, default=0.02)
ap.add_argument('--bc-kl', type=float, default=0.1); ap.add_argument('--bc-kl-steps', type=float, default=5e6)
ap.add_argument('--max-steps', type=float, default=3e7); ap.add_argument('--hours', type=float, default=24)
ap.add_argument('--ckpt-every', type=float, default=1e6); ap.add_argument('--eval-every', type=float, default=3e6)
ap.add_argument('--baseline', help='eval.py --teacher JSON: promotion = 95%% of its team win at the stage difficulty')
ap.add_argument('--stage-min', type=float, default=2e6); ap.add_argument('--stage-max', type=float, default=1e7)
ap.add_argument('--legacy-mates', type=float, default=0.3); ap.add_argument('--node', default='node')
ap.add_argument('--eval-matches', type=int, default=24); ap.add_argument('--eval-chunks', type=int, default=2)
ap.add_argument('--eval-configs', default='solo:NORMAL,solo:HARD,solo:ABYSS,coop4:NORMAL,coop4:HARD,coop4:ABYSS')
ap.add_argument('--curriculum', choices=['staged', 'fixed'], default='staged')
ap.add_argument('--rehearsal', type=int, default=0, help='legacy layout rehearsal candidates in the training matches')
ap.add_argument('--reward', default='{}', help='JSON overrides of env.mjs REWARD, e.g. {"round":0.05,"fa":1.5}')
ap.add_argument('--vf-warmup', type=int, default=0, help='first updates train the value head only (a new reward)')
ap.add_argument('--learner-cpus', default='', help='CPU list for the learner after the environments are spawned')
ap.add_argument('--env-cpus', default='', help='CPU list the environment processes are spawned with (default: the whole unit)')
ap.add_argument('--check', action='store_true', help='collectViolations at every match end in the environments')
ap.add_argument('--eval-baselines', default='', help='name=path,... eval.py outputs on the in-run eval seeds (paired differences in TensorBoard)')
ap.add_argument('--hist-every', type=int, default=10, help='TensorBoard histograms every n updates')
ap.add_argument('--overlap', action='store_true', help='T1: update on a learner copy in a thread while the actor keeps collecting')
ap.add_argument('--device', default='cuda', help='cuda, or cpu for smoke tests while the GPU is taken')
ap.add_argument('--kl-ref', default='', help='checkpoint of the KL anchor policy (default: --init)')
ap.add_argument('--strict-preview', action='store_true', help='strict preview for the RL seats (env strictPreview), off by default')
ap.add_argument('--xplace', default=None, help='xplace: env xPlace JSON (learned layout scorer, no sims), training envs and in-run evals')
ap.add_argument('--picker-head', default=None, help='boss-round layout picker in the training envs and in-run evals (env pickerHead; with --picker-trunk, a frozen JS trunk export; boss rehearsal 0)')
ap.add_argument('--picker-trunk', default=None); ap.add_argument('--picker-k', type=int, default=8)
ap.add_argument('--nat-plan', default=None, help='(11-13) NAT arm: env natPlan JSON (plan pair obs block, national / secondary terms, potential); in-run evals too')
ap.add_argument('--visible-only', action='store_true', help='deploy information set for the RL seats (env visibleOnly), in training and in-run evals')
ap.add_argument('--lp-obs', default=None, help="own-LP obs field for every seat: 'raw' (raw LP) or a scale k on lp / lp0 (env lpObs)")
ap.add_argument('--reward-v', type=int, default=1, help='2: reward version 2 (team clear, FA team/own, leaks, 联防 saves, LP vs L_ref, elimination)')
ap.add_argument('--r2', default='{}', help='reward version 2 weight overrides, e.g. {"layer": 0.002} (direct layer reward)')
ap.add_argument('--shape-uncapped', action='store_true', help='bond shaping stack term uncapped: Σ stacking-bond layers / 100 (potential-based)')
ap.add_argument('--eval2-configs', default='', help='a second in-run eval (eval-<steps>-b<N>.json, TB eval_b<N>/…) of these configs with --eval2-boss-rehearsal')
ap.add_argument('--eval2-boss-rehearsal', type=int, default=0)
ap.add_argument('--npicker-head', default=None, help='normal-round layout picker (picker spatial head) in the training envs and in-run evals; rehearsal 0')
ap.add_argument('--npicker-k', type=int, default=8); ap.add_argument('--npicker-from', type=int, default=1); ap.add_argument('--npicker-to', type=int, default=13)
ap.add_argument('--npicker-mode', default=None); ap.add_argument('--npicker-margin', type=float, default=None)
ap.add_argument('--human-proxy', default=None, help="co-op proxies: env humanProxy JSON for the human-proxy seats (tools/rl/proxies.mjs), e.g. a teammate population '{\"kinds\":[\"mix:legacy=0.4|nat:rand=0.4|class=0.1|noisy:50=0.1\"],\"log\":false}'; off = legacy proxies")
ap.add_argument('--mix', default='v1', choices=['v1', 'v2', 'v3', 'v3s', 'v3sc', 'solo1', 'solo1w', 'mp1', 'mp1w', 'mp2', 'hard1', 'auto1'], help='v2 (11-12): fixed mix with coop4ai (4 RL seats) ABYSS .20 / HARD .06')
ap.add_argument('--reward-by', default='{}', help='per-difficulty reward overrides, e.g. {"ABYSS": {"round": 0, "win": 1.0}} (env rewardBy)')
ap.add_argument('--freeze-band', action='store_true', help='band choice fixed: no gradient on enc.bands / ptr.bands (no optimizer entries), band-decision rows out of the policy-gradient and entropy terms (their KL anchor stays)')
ap.add_argument('--lr-warmup', type=int, default=0, help='resume: ramp lr linearly over this many updates after the value warm-up (fresh Adam moments)')
ap.add_argument('--start-steps', type=float, default=0, help='resume: the step counter starts here (checkpoint / eval cadence continues)')
ap.add_argument('--seed-offset', type=int, default=0, help='resume: added to the training match seeds (envSeed += offset // 1000), so a resumed run does not replay its first matches')
ap.add_argument('--bc-kl-from', type=float, default=None, help='(11-13) --bc-kl-steps counts from this absolute step (the phase start; restart-safe: pass the same value on every resume), not from step 0')
ap.add_argument('--lr-final', type=float, default=None, help='(11-13) anneal lr linearly to this value from --lr-anneal-start (absolute steps) to --max-steps')
ap.add_argument('--lr-anneal-start', type=float, default=None)
ap.add_argument('--bc-kl-min', type=float, default=0.0, help='with --bc-kl-steps > 0: the KL anchor decays linearly from --bc-kl to this floor')
ap.add_argument('--viol-rate', type=float, default=0.6, help='in-run eval: rule violations per match above this alert (one-round-trip plans make a few inherent; teacher4 0.28)')
ap.add_argument('--micro', type=int, default=0, help='gradient accumulation chunk inside a minibatch (memory); 0 = the whole minibatch')
ap.add_argument('--mem-frac', type=float, default=0.0, help='cap this process\'s share of the GPU memory (a shared GPU); 0 = no cap')
# ppo3 candidates (off by default)
ap.add_argument('--ent-final', type=float, default=None, help='(scr0) anneal the entropy coefficient linearly from --ent to this over start..max steps')
ap.add_argument('--auto-hard-gate', type=float, default=0.60, help='(scr0, auto1) HARD rolling win that switches ABYSS on')
ap.add_argument('--auto-target', type=float, default=0.35, help='(scr0, auto1) expected win rate of the mix (band 0.15-0.60)')
ap.add_argument('--auto-smin', type=float, default=0.10); ap.add_argument('--auto-smax', type=float, default=0.85)
ap.add_argument('--auto-step', type=float, default=0.02, help='(scr0, auto1) max change of the ABYSS share per update')
ap.add_argument('--auto-nfloor', type=float, default=0.05, help='(scr0, auto1) NORMAL share floor')
ap.add_argument('--log-diag', action='store_true', help='(scr0) update events carry entropy by kind / chosen type, advantage and return scale, EV by group')
ap.add_argument('--eval-at', default='', help='(scr0) extra in-run evals at these absolute steps (comma list; 0 or the start step = the init, before training)')
ap.add_argument('--plr', type=float, default=0.0, help='S3: share of new matches that replay a buffered level (seed)')
ap.add_argument('--plr-buffer', type=int, default=256); ap.add_argument('--plr-beta', type=float, default=0.1)
ap.add_argument('--plr-staleness', type=float, default=0.3)
ap.add_argument('--sil', type=float, default=0.0, help='S2: self-imitation loss weight (0 = off)')
ap.add_argument('--sil-buffer', type=int, default=100000); ap.add_argument('--sil-batches', type=int, default=4)
ap.add_argument('--sil-vf', type=float, default=0.01)
ap.add_argument('--snap-start', type=float, default=0.0, help='S1: share of ABYSS matches started from late-game checkpoints')
ap.add_argument('--snap-collect', type=float, default=0.25, help='S1: chance an eligible ABYSS prep-end (R9-13) is pooled')
ap.add_argument('--band-policy', type=float, default=0.0, help='share of RL seats whose opening band the policy picks (0 = legacy)')
ap.add_argument('--band-warmstart', default='', help='Step 2: band_table.json; the band head first fits softmax(fa / tau) per config')
ap.add_argument('--band-tau', type=float, default=0.05); ap.add_argument('--band-warm-steps', type=int, default=300)
ap.add_argument('--www', default='', help='eval report pages: a directory (eval_report.py), one page per in-run evaluation')
ap.add_argument('--vclip', type=float, default=0.2, help='PPO value clipping (0 = plain MSE)')
ap.add_argument('--adv-norm', choices=['global', 'group'], default='global', help='advantage normalisation (group = difficulty x solo/co-op)')
ap.add_argument('--boss-rehearsal', type=int, default=None, help='legacy rehearsal on the Final Assault / Hidden Core rounds only')
ap.add_argument('--shape-tier', type=float, default=0.0, help='bond shaping: weight of Σ active tier / max tier (potential-based)')
ap.add_argument('--shape-stack', type=float, default=0.0, help='bond shaping: weight of Σ min(layers / next milestone, 1), active bonds')
ap.add_argument('--amp', action='store_true', help='(A1) bf16 autocast for every forward (actor, old log-probs, minibatches, KL anchor) and the rollout observations kept fp16 on the GPU')
ap.add_argument('--fwd-chunk', type=int, default=8192, help='(A1) rows per no-grad forward of the old log-probs pass (memory)')
ap.add_argument('--seq-len', type=int, default=32, help='(A3) recurrent policy: loss steps per training sequence (truncated BPTT)')
ap.add_argument('--burn-in', type=int, default=16, help='(A3) recurrent policy: steps before each sequence that rebuild its state from the stored one (no gradient)')
ap.add_argument('--clip-sep', action='store_true', help='(A1) a separate critic: clip the policy and the value gradients each to 0.5 (else one global norm)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
AMP = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if a.amp else contextlib.nullcontext  # (A1) --amp
if a.curriculum == 'fixed':
    STAGES = FIXED_AUTO1 if a.mix == 'auto1' else FIXED_HARD1 if a.mix == 'hard1' else FIXED if a.mix == 'v1' else FIXED2 if a.mix == 'v2' else FIXED3S if a.mix == 'v3s' else FIXED3SC if a.mix == 'v3sc' else FIXED_SOLO1 if a.mix == 'solo1' else FIXED_SOLO1W if a.mix == 'solo1w' else FIXED_MP1 if a.mix == 'mp1' else FIXED_MP1W if a.mix == 'mp1w' else FIXED_MP2 if a.mix == 'mp2' else FIXED3
reward_over = json.loads(a.reward)
reward_by = json.loads(a.reward_by)
dev = torch.device(a.device)
logf = open(os.path.join(a.out, 'train.jsonl'), 'a')
def emit(**kw):
    kw['t'] = round(time.time(), 1); line = json.dumps(kw); print(line, flush=True); logf.write(line + '\n'); logf.flush()

if a.mem_frac > 0 and str(dev).startswith('cuda'):
    torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
net, ck = load(a.init, dev)
bc_net = None
REC = getattr(net, 'rnn', None) is not None  # (A3) a recurrent policy (model5 cfg rnn): stored states, sequence minibatches
assert not (REC and (a.sil > 0 or a.band_warmstart)), 'recurrent policy: SIL / band warm start not supported'
if a.bc_kl > 0:
    bc_net, _ = load(a.kl_ref or a.init, dev); bc_net.eval()
    if hasattr(bc_net, 'with_value'): bc_net.with_value = False  # (A1) SPNet5: the KL anchor never needs its critic
    assert getattr(bc_net, 'rnn', None) is None, '(A3) the KL anchor must be a feed-forward policy'
    for p in bc_net.parameters():
        p.requires_grad_(False)
if a.freeze_band:
    for p_ in list(net.enc['bands'].parameters()) + list(net.ptr['bands'].parameters()):
        p_.requires_grad_(False)
opt = torch.optim.Adam([p_ for p_ in net.parameters() if p_.requires_grad], lr=a.lr, eps=1e-5)
import copy, threading
actor = copy.deepcopy(net) if a.overlap else net  # T1: the network that acts; net is the learner
actor.eval()
value_params = list(net.value.parameters()) + (list(net.vtrunk.parameters()) if hasattr(net, 'vtrunk') else [])  # LIN-C: the critic trunk warms up with its head
opt_v = torch.optim.Adam(value_params, lr=a.lr, eps=1e-5)  # A1: the value-only warm-up
pi_params = [p_ for p_ in net.parameters() if p_.requires_grad and all(p_ is not q for q in value_params)]  # (A1) --clip-sep
spec = net.spec
threshold = {}
if a.baseline:
    base = json.load(open(a.baseline))['configs']
    for key in ('NORMAL', 'HARD', 'ABYSS'):
        ws = [c['teamWin'] for name, c in base.items() if name.endswith(':' + key)]
        if ws:
            threshold[key] = 0.95 * sum(ws) / len(ws)
stage, stage_since = 0, 0
cfg_names = [c['name'] for c in STAGES[0]['configs']]
full_cpus = os.sched_getaffinity(0)  # the unit's whole CPU set: in-run evaluations get it back
if a.env_cpus:
    os.sched_setaffinity(0, cpu_list(a.env_cpus) & full_cpus)  # the environment processes inherit this at spawn
# P10: training seeds from 1e9, far from every evaluation range (7e6 + config index x 1e6)
envs = [NodeEnv({'envs': a.envs, 'configs': STAGES[0]['configs'], 'seed': 1_000_000_000 + 10_000_000 * (i + 1) + a.seed_offset, 'envSeed': 1000 + i + a.seed_offset // 1000,
                 'humans': 'random', 'legacyMates': a.legacy_mates, 'rehearsal': a.rehearsal, 'reward': reward_over, **({'rewardBy': reward_by} if reward_by else {}),
                 'obsVersion': 4 if 'tiles' in spec['blocks'] else 3 if 'bands' in spec['blocks'] else 2,
                 **({'obsExt': spec['ext']} if spec.get('ext') else {}),  # (A1) the checkpoint's version 3 extension blocks
                 'check': a.check, 'shaping': {'gamma': a.gamma, 'wTier': a.shape_tier, 'wStack': a.shape_stack, 'uncapped': a.shape_uncapped},
                 **({'rewardV': 2, 'r2': json.loads(a.r2)} if a.reward_v == 2 else {}),
                 **({'lpObs': a.lp_obs if a.lp_obs == 'raw' else float(a.lp_obs)} if a.lp_obs else {}),
                 **({'visibleOnly': True} if a.visible_only else {}), **({'strictPreview': True} if a.strict_preview else {}),
                 **({'pickerHead': {'head': a.picker_head, 'trunk': a.picker_trunk, 'K': a.picker_k}} if a.picker_head else {}),
                 **({'normalPicker': {'head': a.npicker_head, 'K': a.npicker_k, 'from': a.npicker_from, 'to': a.npicker_to, **({'mode': a.npicker_mode} if a.npicker_mode else {}), **({'margin': a.npicker_margin} if a.npicker_margin is not None else {})}} if a.npicker_head else {}),
                 **({'xPlace': json.loads(a.xplace)} if a.xplace else {}),
                 **({'natPlan': json.loads(a.nat_plan)} if a.nat_plan else {}),
                 **({'humanProxy': json.loads(a.human_proxy)} if getattr(a, 'human_proxy', None) else {}),
                 **({'snapCollect': {'prob': a.snap_collect}, 'snapStart': {'frac': a.snap_start}} if a.snap_start > 0 else {}),
                 **({'bandPolicy': {'share': a.band_policy}} if a.band_policy > 0 else {}),
                 **({'bossRehearsal': a.boss_rehearsal} if a.boss_rehearsal is not None else {})}, node=a.node)
        for i in range(a.procs)]
env_cpu_count = len(os.sched_getaffinity(0))
if a.learner_cpus:
    os.sched_setaffinity(0, cpu_list(a.learner_cpus) & full_cpus)
env_counts = [(0, 0, 0)] * len(envs)  # errors + failed, rejected, violations (batch header) per environment process
Rw = envs[0].spec['reward']
board = Board(os.path.join(a.out, 'tb'), envs[0].spec['act'],
              {k: v for k, v in (x.split('=', 1) for x in a.eval_baselines.split(',') if x)})
# S3 PLR: per process, seed -> {score, last}; trajectory scores (mean |advantage|) gathered over updates
proc_seed_base = [1_000_000_000 + 10_000_000 * (i + 1) + a.seed_offset for i in range(a.procs)]
levels = [dict() for _ in range(a.procs)]; traj_score = defaultdict(lambda: [0.0, 0]); last_matches = [0] * a.procs
plr_stats = {'replayed': 0, 'fresh': 0}
plr_rng = np.random.default_rng(11)

def plr_pick(i):
    L = levels[i]
    if not L:
        return None
    seeds = list(L.keys()); sc = np.array([L[x]['score'] for x in seeds]); st = np.array([update_no - L[x]['last'] for x in seeds], dtype=np.float64)
    rank = np.empty(len(sc)); rank[np.argsort(-sc)] = np.arange(1, len(sc) + 1)
    ps = (1.0 / rank) ** (1.0 / a.plr_beta); ps /= ps.sum()
    pc = st / st.sum() if st.sum() > 0 else np.full(len(st), 1.0 / len(st))
    p = (1 - a.plr_staleness) * ps + a.plr_staleness * pc
    s_ = seeds[int(plr_rng.choice(len(seeds), p=p / p.sum()))]; L[s_]['last'] = update_no
    return s_

def plr_add(i, seed, score):
    L = levels[i]
    e = L.get(seed)
    L[seed] = {'score': score if e is None else 0.5 * (e['score'] + score), 'last': e['last'] if e else update_no}
    if len(L) > a.plr_buffer:
        del L[min(L, key=lambda x: L[x]['score'])]

# S2 SIL: transitions whose return beat the value estimate, prioritised by (R - V)+
sil = {'obs': None, 'mask': None, 'act': None, 'ret': None, 'pri': None, 'n': 0, 'pos': 0}

def sil_add(obs_np, mask_np, act_np, ret_np, v_np):
    pos_ = ret_np - v_np
    sel = pos_ > 0
    if not sel.any():
        return
    o, m_, ac, r_, pr = obs_np[sel], mask_np[sel], act_np[sel], ret_np[sel], pos_[sel]
    if sil['obs'] is None:
        cap = a.sil_buffer
        sil.update(obs=np.zeros((cap,) + o.shape[1:], np.float16), mask=np.zeros((cap,) + m_.shape[1:], bool), act=np.zeros(cap, np.int64),
                   ret=np.zeros(cap, np.float32), pri=np.zeros(cap, np.float32))
    cap = len(sil['act'])
    for j in range(len(ac)):
        k = sil['pos']; sil['obs'][k] = o[j]; sil['mask'][k] = m_[j]; sil['act'][k] = ac[j]; sil['ret'][k] = r_[j]; sil['pri'][k] = pr[j]
        sil['pos'] = (k + 1) % cap; sil['n'] = min(cap, sil['n'] + 1)

recent_terms = []  # episodes ended since the last update: (config, win, fa, round term, lp term, bond shaping)
BOND = envs[0].spec['blocks']['bonds']  # tier field 2 = tier / 3
G0 = envs[0].spec['blocks']['global']['offset']  # field 0 = round / 15
assert envs[0].spec['dataId'] == spec['dataId'], ('observation layout changed since the checkpoint', envs[0].spec['dataId'], spec['dataId'])
assert envs[0].spec['dim'] == spec['dim'], ('observation width differs from the checkpoint (A1: version 3 extension blocks)', envs[0].spec['dim'], spec['dim'])
env_stage = [0] * len(envs)
env_mix = [0] * len(envs)  # (scr0) auto1 mix version applied per process
pool = AsyncPool(envs); pool.reset()
hstate = {}; h_zero = np.zeros(net.state_size if REC else 0, np.float32)  # (A3) (process, trajectory) -> state after its last decision
pending, chains = {}, defaultdict(list)
ready = 0
episodes = deque(maxlen=3000)
steps = int(a.start_steps); t0 = time.time(); update_no = 0
next_ckpt = (math.floor(steps / a.ckpt_every) + 1) * a.ckpt_every; next_eval = (math.floor(steps / a.eval_every) + 1) * a.eval_every
gen = torch.Generator(device=dev); gen.manual_seed(7)
evals = []
eval_at = sorted(float(x) for x in a.eval_at.split(',') if x.strip())  # (scr0)

def finalize(key, r, nv, done):
    global ready
    t = pending.pop(key); t['r'] = r; t['nv'] = nv; t['done'] = done; chains[key].append(t); ready += 1

def rolling(key):
    xs = [e for e in episodes if e[0].endswith(':' + key)]
    if not xs:
        return None
    return {'n': len(xs), 'win': float(np.mean([e[1] for e in xs])), 'survival': float(np.mean([e[2] for e in xs])), 'lp': float(np.mean([e[3] for e in xs])),
            'fa': float(np.mean([e[4] for e in xs]))}

auto_state = {'on': False, 's': 0.0, 'ver': 0}
def auto_mix():  # (scr0) --mix auto1: in-update difficulty mixing gated on rolling win rates
    rN, rH, rA = rolling('NORMAL'), rolling('HARD'), rolling('ABYSS')
    pN = rN['win'] if rN and rN['n'] >= 100 else 0.9; pH = rH['win'] if rH and rH['n'] >= 300 else 0.0
    pA = rA['win'] if rA and rA['n'] >= 100 else 0.0
    if not auto_state['on'] and pH >= a.auto_hard_gate:
        auto_state['on'] = True; emit(event='mix-gate', steps=steps, reason='HARD rolling win %.3f >= %.2f' % (pH, a.auto_hard_gate))
    n = max(a.auto_nfloor, 0.30 if pH < 0.30 else a.auto_nfloor)
    if auto_state['on']:
        pR = (n * pN + (1 - n) * pH)  # win rate of the non-ABYSS part
        want = a.auto_smax if pA >= a.auto_target else (pR - a.auto_target) / max(1e-6, pR - pA)
        want = min(a.auto_smax, max(a.auto_smin, want))
        s_ = auto_state['s'] if auto_state['s'] > 0 else a.auto_smin
        auto_state['s'] = s_ + max(-a.auto_step, min(a.auto_step, want - s_))
    sA = auto_state['s']; wN = (1 - sA) * n; wH = (1 - sA) - wN
    cfg = W(solo_NORMAL=round(wN, 4), solo_HARD=round(wH, 4), solo_ABYSS=round(sA, 4))
    STAGES[0]['configs'] = cfg
    auto_state['ver'] += 1  # applied per process at its next batch (the protocol is request / response)
    emit(event='mix', steps=steps, w={'NORMAL': round(wN, 4), 'HARD': round(wH, 4), 'ABYSS': round(sA, 4)}, p={'NORMAL': round(pN, 3), 'HARD': round(pH, 3), 'ABYSS': round(pA, 3)},
         n={k: (r or {}).get('n', 0) for k, r in (('NORMAL', rN), ('HARD', rH), ('ABYSS', rA))}, expected=round(wN * pN + wH * pH + sA * pA, 3))

def checkpoint(tag):
    path = os.path.join(a.out, f'model-{tag}.pt')
    save(path, net, {'steps': steps, 'stage': STAGES[stage]['name'], 'args': vars(a)})
    save(os.path.join(a.out, 'model-last.pt'), net, {'steps': steps, 'stage': STAGES[stage]['name'], 'args': vars(a)})
    return path

def launch_eval(path, block=False, second=False):
    if not second and a.eval2_configs:
        launch_eval(path, block=False, second=True)
    out = os.path.join(a.out, f'eval-{steps}-b{a.eval2_boss_rehearsal}.json' if second else f'eval-{steps}.json')
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'eval.py'), '--ckpt', path, '--out', out, '--greedy',
           '--matches', str(a.eval_matches), '--chunks', str(a.eval_chunks), '--envs', '6', '--node', a.node,
           '--bond-log', *(['--lp-obs', str(a.lp_obs)] if a.lp_obs else []), *(['--visible-only'] if a.visible_only else []), *(['--strict-preview'] if a.strict_preview else []), *(['--picker-head', a.picker_head, '--picker-trunk', a.picker_trunk, '--picker-k', str(a.picker_k)] if a.picker_head else []), *(['--npicker-head', a.npicker_head, '--npicker-k', str(a.npicker_k), '--npicker-from', str(a.npicker_from), '--npicker-to', str(a.npicker_to), *(['--npicker-mode', a.npicker_mode] if a.npicker_mode else []), *(['--npicker-margin', str(a.npicker_margin)] if a.npicker_margin is not None else [])] if a.npicker_head else []), *(['--xplace', a.xplace] if a.xplace else []), *(['--nat-plan', a.nat_plan] if a.nat_plan else []), '--configs', a.eval2_configs if second else a.eval_configs, '--rehearsal', str(a.rehearsal), '--device', a.device] + (['--boss-rehearsal', str(a.eval2_boss_rehearsal if second else a.boss_rehearsal)] if second or a.boss_rehearsal is not None else []) \
          + (['--band-policy', '1'] if a.band_policy > 0 else [])  # the policy's own band picks are evaluated too
    p = subprocess.Popen(cmd, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT,
                         env={**os.environ, 'OMP_NUM_THREADS': '4'}, preexec_fn=lambda: os.sched_setaffinity(0, full_cpus))
    evals.append((p, out))
    if block:
        p.wait()

def eval_steps_of(path):
    try:
        return int(os.path.basename(path).split('-')[1].split('.')[0])
    except Exception:
        return 0

def eval_problems():
    """Finished in-run evaluations: (path, rule violations, errors) of each one with a problem; finished ones are dropped."""
    bad = []
    for p, out in list(evals):
        if p.poll() is None:
            continue
        evals.remove((p, out))
        try:
            r = json.load(open(out))['configs']
        except Exception as e:
            bad.append((out, 'unreadable', str(e)[:200])); continue
        viol = sum(int(c['ai'].get('rl', {}).get('violations', 0)) for c in r.values())
        errs = sum(int(c['errors']) for c in r.values())
        n_matches = sum(int(c.get('matches', 0)) for c in r.values()) or 1
        summary = {k: {'win': round(c['teamWin'], 3), 'survival': round(c['roles'].get('rl', {}).get('survival', 0), 2),
                       'fa': round(c.get('faShare', 0), 3), 'rt': round(c['ai'].get('rl', {}).get('roundTripsPerPrep', 0), 3)} for k, c in r.items()}
        emit(event='eval', path=out, violations=viol, errors=errs, result=summary)
        board.eval(out, tag=('_' + out.rsplit('-', 1)[1][:-5]) if re.search(r'-b\d+\.json$', out) else '')
        if a.www:  # the HTML report of this evaluation (eval_report.py, numpy only; failures never stop training)
            try:
                import eval_report
                base = {k: json.load(open(v)) for k, v in (x.split('=', 1) for x in a.eval_baselines.split(',') if x) if os.path.exists(v)}
                label = {'teacher': '规则版 legacy', 'live': '无规则 legacy'}
                name = f'{os.path.basename(a.out)}-{eval_steps_of(out)}'
                eval_report.add(a.www, name, f'{os.path.basename(a.out)} 训练中评估 · {eval_steps_of(out) / 1e6:.1f}M 步',
                                [(os.path.basename(a.out), json.load(open(out)))], [(label.get(k, k), v) for k, v in base.items()],
                                note=f'贪心策略，rehearsal {a.rehearsal}' + (f'，首领回合 {a.boss_rehearsal}' if a.boss_rehearsal is not None else '') +
                                     f'，每个配置 {a.eval_matches} 个种子')
            except Exception as e:
                emit(event='www-error', path=out, error=str(e)[:300])
        if errs or viol / n_matches > a.viol_rate:  # rule violations: a rate (the one-round-trip semantics make a few inherent)
            bad.append((out, viol, errs))
    return bad

def update(rollout):
    """One PPO update on `rollout` (trajectory key -> finalized transitions), handed over by the main loop."""
    global update_no
    keys = list(rollout.keys())
    obs, mask, act, logp_old, v_old, adv, ret = [], [], [], [], [], [], []
    kinds, preps = [], []  # preps: (buys, sells, refreshes) of every prep that ended (END) in this rollout
    spans_, hs = [], []  # (A3) per chain: (first flat row, length); per row: the stored state before the decision
    for key in keys:
        ch = rollout.pop(key)
        cnt = [0, 0, 0]
        for t in ch:
            if t['kind'] != 0:
                continue
            x = t['a']
            if x == board.act['END']:
                preps.append(tuple(cnt)); cnt = [0, 0, 0]
            elif x == board.act['REFRESH']:
                cnt[2] += 1
            elif board.act['BUY'] <= x < board.act['SELL']:
                cnt[0] += 1
            elif board.act['SELL'] <= x < board.act['REWARD']:
                cnt[1] += 1
        a_next = 0.0
        advs = np.empty(len(ch), dtype=np.float32)
        for k in range(len(ch) - 1, -1, -1):
            t = ch[k]
            nd = 0.0 if t['done'] else 1.0
            delta = t['r'] + a.gamma * nd * t['nv'] - t['v']
            a_next = delta + a.gamma * a.lam * nd * a_next
            advs[k] = a_next
        if a.plr > 0:
            ts = traj_score[key]; ts[0] += float(np.abs(advs).sum()); ts[1] += len(advs)
        if REC:
            spans_.append((len(obs), len(ch))); hs.extend(t['h'] for t in ch)
        for t, ad in zip(ch, advs):
            obs.append(t['obs']); mask.append(t['mask']); act.append(t['a']); logp_old.append(t['logp']); v_old.append(t['v'])
            adv.append(ad); ret.append(ad + t['v']); kinds.append(t['kind'])
    extra = {'act': np.array(act), 'kind': np.array(kinds), 'v': np.array(v_old, dtype=np.float32), 'adv': np.array(adv, dtype=np.float32),
             'preps': np.array(preps, dtype=np.int32).reshape(-1, 3)}
    obs_np = np.stack(obs)
    O = torch.from_numpy(obs_np).to(dev) if a.amp else torch.from_numpy(obs_np).to(dev).float(); M = torch.from_numpy(np.stack(mask)).to(dev)
    # own bond tiers (Σ active tiers) at prep decisions, by round: does the shaping change the builds
    pk = np.array(kinds) == 0
    tiers = (obs_np[pk][:, BOND['offset'] + 2:BOND['offset'] + BOND['count'] * BOND['width']:BOND['width']].astype(np.float32) * 3).sum(1)
    rounds = np.rint(obs_np[pk][:, G0].astype(np.float32) * 15).astype(int)
    Ac = torch.tensor(act, device=dev, dtype=torch.long); V0 = torch.tensor(v_old, device=dev); R = torch.tensor(ret, device=dev)
    AD_raw = torch.tensor(adv, device=dev)
    N = O.shape[0]
    # groups: difficulty one-hot (global fields 4..7) x solo / co-op (global field 8)
    gnp = obs_np[:, G0 + 4:G0 + 8].astype(np.float32).argmax(1) * 2 + (obs_np[:, G0 + 8] > 0.5)
    G = torch.from_numpy(gnp.astype(np.int64)).to(dev)
    diag = {}
    # P9: the old log-probabilities from the stored (fp16) observations, so the first epoch starts at ratio 1; the same
    # pass gives the rollout policy's entropy per decision kind and per chosen action type (B1)
    if REC:
        # (A3) training sequences: each chain (a trajectory's decisions in this rollout, in order) cut into runs of seq_len loss
        # steps; each run is preceded by up to burn_in earlier steps of the same chain (left-padded) and starts from the state
        # the actor stored before its first step (burn-in start); padding steps carry the state unchanged
        Lq, Kb = a.seq_len, a.burn_in; Tq = Kb + Lq; rows_l, h0_l = [], []
        for f0, n_ in spans_:
            for s0 in range(0, n_, Lq):
                nb = min(Kb, s0); nl = min(Lq, n_ - s0)
                rows_l.append([-1] * (Kb - nb) + list(range(f0 + s0 - nb, f0 + s0 + nl)) + [-1] * (Lq - nl)); h0_l.append(hs[f0 + s0 - nb])
        SEQ = torch.tensor(rows_l, device=dev, dtype=torch.long); SVAL = SEQ >= 0; SEQc = SEQ.clamp_min(0); NS = SEQ.shape[0]
        H0 = torch.from_numpy(np.stack(h0_l).astype(np.float32)).to(dev)
        assert int(SVAL[:, Kb:].sum()) == N, ('every row is a loss step exactly once', int(SVAL[:, Kb:].sum()), N)
        def seq_fwd(sq):
            r = SEQc[sq]; vv = SVAL[sq]
            with AMP():
                lg_, v_ = net.forward_seq(O[r].float(), M[r], H0[sq], vv, Kb)
            lv = vv[:, Kb:]
            return lg_.float()[lv], v_.float()[lv], SEQ[sq][:, Kb:][lv]
        diag['diag/seqs'] = NS; diag['diag/seq_fill'] = N / max(1, NS * Lq)
    with torch.no_grad():
        lps, ents = [], []
        FC = a.fwd_chunk
        wv_ = getattr(net, 'with_value', None)
        if wv_ is not None: net.with_value = False  # (A1) SPNet5: this pass needs the policy only
        if REC:
            LP = torch.zeros(N, device=dev); ENT = torch.zeros(N, device=dev)
            for i in range(0, NS, max(1, FC // Tq)):
                lg, _, rid = seq_fwd(torch.arange(i, min(NS, i + max(1, FC // Tq)), device=dev))
                la = F.log_softmax(lg, 1)
                LP[rid] = la.gather(1, Ac[rid, None]).squeeze(1); ENT[rid] = -(la.exp() * la).sum(1)
        else:
            for i in range(0, N, FC):
                with AMP():
                    lg, _ = net(O[i:i + FC].float(), M[i:i + FC])
                la = F.log_softmax(lg.float(), 1)
                lps.append(la.gather(1, Ac[i:i + FC, None]).squeeze(1)); ents.append(-(la.exp() * la).sum(1))
            LP = torch.cat(lps); ENT = torch.cat(ents)
        if wv_ is not None: net.with_value = wv_
        LP_rec = torch.tensor(logp_old, device=dev)
        diag['diag/logp_fp16_gap'] = float((LP - LP_rec).abs().mean())
        if a.overlap:  # T1: samples were drawn by an actor up to one update older than this net: keep the recorded log-probs
            LP = LP_rec
    ent_np = ENT.cpu().numpy(); kinds_np = np.array(kinds)
    NBW = torch.from_numpy((kinds_np != 2).astype(np.float32)).to(dev) if a.freeze_band else None
    types = np.searchsorted(np.array(board.bounds), np.array(act), side='right')
    for kname, kv in (('prep', 0), ('card', 1), ('band', 2), ('tile', 3), ('target', 4), ('plan', 5)):  # plan: N1 natPlan.head
        sel = kinds_np == kv
        if sel.any():
            diag[f'diag/entropy_kind_{kname}'] = float(ent_np[sel].mean()); diag[f'diag/share_kind_{kname}'] = float(sel.mean())
    act_np = np.array(act); ends = int(((kinds_np == 0) & (act_np == 0)).sum())
    if ends and spec['actions'] >= 295:  # version 4: moves (MOVE picks) and equips per finished prep in this rollout
        diag['diag/moves_per_prep'] = float(((act_np >= 100) & (act_np < 127)).sum() / ends)
        diag['diag/equips_per_prep'] = float(((act_np >= 127) & (act_np < 154)).sum() / ends)
        diag['diag/econ_per_prep'] = float(((kinds_np == 0) & (act_np > 0) & (act_np < 100)).sum() / ends)
    for ti, tname in enumerate(TYPES):
        sel = types == ti
        if sel.any():
            diag[f'diag/entropy_chosen_{tname}'] = float(ent_np[sel].mean())
    r_np, v_np, adv_np = np.array(ret, np.float64), np.array(v_old, np.float64), np.array(adv, np.float64)
    def explained(sel):
        var = r_np[sel].var()
        return float(1 - (r_np[sel] - v_np[sel]).var() / var) if var > 1e-12 else float('nan')
    diag['diag/ev_all'] = explained(np.ones(N, bool)); diag['diag/adv_std_all'] = float(adv_np.std())
    if a.log_diag:  # (scr0) return scale and raw advantage magnitude
        diag['diag/ret_mean'] = float(r_np.mean()); diag['diag/ret_std'] = float(r_np.std()); diag['diag/ret_p95'] = float(np.percentile(r_np, 95))
        diag['diag/adv_absmean_all'] = float(np.abs(adv_np).mean()); diag['diag/adv_mean_all'] = float(adv_np.mean())
    for gi, gname in enumerate(GROUPS):
        sel = gnp == gi
        if sel.sum() >= 32:
            diag[f'diag/ev_{gname}'] = explained(sel); diag[f'diag/adv_std_{gname}'] = float(adv_np[sel].std())
            diag[f'diag/samples_{gname}'] = int(sel.sum())
    # advantage normalisation: one batch-wide scale, or per group (A3 switch; a group under 32 samples uses the batch's)
    gm, gs = AD_raw.mean(), AD_raw.std() + 1e-8
    if a.adv_norm == 'group':
        AD = (AD_raw - gm) / gs
        for gi in range(len(GROUPS)):
            sel = G == gi
            if int(sel.sum()) >= 32:
                x = AD_raw[sel]; AD[sel] = (x - x.mean()) / (x.std() + 1e-8)
    else:
        AD = (AD_raw - gm) / gs
    # bc_kl_steps <= 0: a persistent anchor (ppo2: 0.01 towards the init policy, ppo1-final)
    bc_coef = 0.0 if bc_net is None else a.bc_kl if a.bc_kl_steps <= 0 else a.bc_kl_min + (a.bc_kl - a.bc_kl_min) * max(0.0, 1.0 - ((steps - a.bc_kl_from) if a.bc_kl_from is not None else steps) / a.bc_kl_steps)
    warm = update_no < a.vf_warmup
    ent_now = a.ent if a.ent_final is None else a.ent + (a.ent_final - a.ent) * min(1.0, max(0.0, steps - a.start_steps) / max(1.0, a.max_steps - a.start_steps))  # (scr0) linear entropy anneal over the run
    if a.ent_final is not None:
        diag['diag/ent_coef'] = ent_now
    if a.lr_warmup > 0:
        lr_now = a.lr * min(1.0, max(1, update_no - a.vf_warmup + 1) / a.lr_warmup)
        if a.lr_final is not None and a.lr_anneal_start is not None and steps > a.lr_anneal_start:  # (11-13) final-phase anneal
            lr_now = min(lr_now, a.lr + (a.lr_final - a.lr) * min(1.0, (steps - a.lr_anneal_start) / max(1.0, a.max_steps - a.lr_anneal_start)))
        for g_ in opt.param_groups: g_['lr'] = lr_now
        diag['diag/lr'] = lr_now
    TB = None
    if band_fa is not None and update_no < a.band_warm_steps:
        selb = kinds_np == 2
        if selb.any():
            TB = torch.zeros((N, a_dim := np.stack(mask).shape[1]), device=dev)
            TB[torch.from_numpy(np.flatnonzero(selb)).to(dev)] = torch.from_numpy(band_targets(obs_np, np.stack(mask), selb)).to(dev)
            diag['diag/band_warm_rows'] = int(selb.sum())
    info = defaultdict(list); stop = False
    for ep in range(a.epochs):
        if REC and not warm:  # (A3) sequence minibatches; as many per epoch as row minibatches would be
            permS = torch.randperm(NS, device=dev); nmb = max(1, math.ceil(N / a.minibatch)); mbS = math.ceil(NS / nmb)
            muS = max(1, (a.micro if a.micro else 10 ** 9) // Tq)
            for i in range(0, NS, mbS):
                sqs = permS[i:i + mbS]; B_ = int(SVAL[sqs][:, Kb:].sum())
                opt.zero_grad(set_to_none=True)
                acc = defaultdict(float)
                for j in range(0, len(sqs), muS):
                    logits, v, idx = seq_fwd(sqs[j:j + muS])
                    f = len(idx) / B_
                    vl = 0.5 * ((v - R[idx]) ** 2).mean() if a.vclip <= 0 else 0.5 * torch.max((v - R[idx]) ** 2, (V0[idx] + (v - V0[idx]).clamp(-a.vclip, a.vclip) - R[idx]) ** 2).mean()
                    logp_all = F.log_softmax(logits, 1)
                    logp = logp_all.gather(1, Ac[idx, None]).squeeze(1)
                    logratio = logp - LP[idx]; ratio = torch.exp(logratio)
                    pgr = torch.min(ratio * AD[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * AD[idx])
                    p = logp_all.exp()
                    entr = -(p * logp_all).sum(1)
                    if NBW is not None:
                        pgr = pgr * NBW[idx]; entr = entr * NBW[idx]
                    pg = -pgr.mean(); ent = entr.mean()
                    loss = pg + a.vf * vl - ent_now * ent
                    if bc_coef > 0:
                        with torch.no_grad():
                            with AMP():
                                lb, _ = bc_net(O[idx][:, :bc_net.D].float() if bc_net.D < O.shape[1] else O[idx].float(), M[idx])
                            lpb = F.log_softmax(lb.float(), 1)
                        kl_bc = (lpb.exp() * (lpb - logp_all)).sum(1).mean()
                        loss = loss + bc_coef * kl_bc; acc['kl_bc'] += kl_bc.item() * f
                    (loss * f).backward()
                    with torch.no_grad():
                        acc['kl'] += (-logratio).mean().item() * f; acc['kl3'] += ((ratio - 1) - logratio).mean().item() * f
                        acc['clipfrac'] += ((ratio - 1).abs() > a.clip).float().mean().item() * f
                    acc['pg'] += pg.item() * f; acc['vloss'] += vl.item() * f; acc['ent'] += ent.item() * f
                if a.clip_sep:
                    acc['gn_pi'] = float(torch.nn.utils.clip_grad_norm_(pi_params, 0.5)); acc['gn_v'] = float(torch.nn.utils.clip_grad_norm_(value_params, 0.5))
                else:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                for k_, v_ in acc.items():
                    info[k_].append(v_)
                if acc['kl3'] > 1.5 * a.target_kl:
                    stop = True; break
            if stop:
                break
            continue
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, a.minibatch):
            idx_mb = perm[i:i + a.minibatch]; B_ = len(idx_mb)
            # --micro: gradient accumulation over the minibatch (every term is a mean, so the chunks weighted by their
            # share give the same loss, gradient and statistics as one pass; only the activation memory shrinks)
            mu = a.micro if a.micro and a.micro < B_ else B_
            chunks = [idx_mb[j:j + mu] for j in range(0, B_, mu)]
            if warm:
                # A1: only the value head learns; the shared trunk and so the policy stay as initialised
                net.zero_grad(set_to_none=True); vls = 0.0
                for idx in chunks:
                    f = len(idx) / B_
                    with AMP():
                        v = net(O[idx].float(), M[idx])[1]  # (A3) a recurrent net also returns its state; the critic is feed-forward
                    v = v.float()
                    if a.vclip > 0:
                        vc = V0[idx] + (v - V0[idx]).clamp(-a.vclip, a.vclip)
                        vl = 0.5 * torch.max((v - R[idx]) ** 2, (vc - R[idx]) ** 2).mean()
                    else:
                        vl = 0.5 * ((v - R[idx]) ** 2).mean()  # A2: no value clipping (unnormalised returns)
                    (a.vf * vl * f).backward(); vls += vl.item() * f
                torch.nn.utils.clip_grad_norm_(value_params, 0.5); opt_v.step()
                info['vloss'].append(vls)
                continue
            opt.zero_grad(set_to_none=True)
            acc = defaultdict(float)
            for idx in chunks:
                f = len(idx) / B_
                with AMP():
                    logits, v = net(O[idx].float(), M[idx])
                logits, v = logits.float(), v.float()
                if a.vclip > 0:
                    vc = V0[idx] + (v - V0[idx]).clamp(-a.vclip, a.vclip)
                    vl = 0.5 * torch.max((v - R[idx]) ** 2, (vc - R[idx]) ** 2).mean()
                else:
                    vl = 0.5 * ((v - R[idx]) ** 2).mean()
                logp_all = F.log_softmax(logits, 1)
                logp = logp_all.gather(1, Ac[idx, None]).squeeze(1)
                logratio = logp - LP[idx]; ratio = torch.exp(logratio)
                pgr = torch.min(ratio * AD[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * AD[idx])
                p = logp_all.exp()
                entr = -(p * logp_all).sum(1)
                if NBW is not None:  # --freeze-band: band rows give no policy-gradient / entropy signal
                    pgr = pgr * NBW[idx]; entr = entr * NBW[idx]
                pg = -pgr.mean(); ent = entr.mean()
                loss = pg + a.vf * vl - ent_now * ent
                if TB is not None:  # Step 2 warm start: the band head towards the Step 0 table on band rows
                    tb = TB[idx]; has = tb.sum(1) > 0
                    if has.any():
                        lb_ = -(tb[has] * logp_all[has].clamp_min(-1e4)).sum(1).mean(); loss = loss + lb_; acc['band_warm_ce'] += lb_.item() * f
                if bc_coef > 0:
                    with torch.no_grad():
                        with AMP():
                            lb, _ = bc_net(O[idx][:, :bc_net.D].float() if bc_net.D < O.shape[1] else O[idx].float(), M[idx])  # a ref without the appended plan block reads the obs it knows
                        lpb = F.log_softmax(lb.float(), 1)
                    kl_bc = (lpb.exp() * (lpb - logp_all)).sum(1).mean()
                    loss = loss + bc_coef * kl_bc; acc['kl_bc'] += kl_bc.item() * f
                (loss * f).backward()
                with torch.no_grad():
                    acc['kl'] += (-logratio).mean().item() * f; acc['kl3'] += ((ratio - 1) - logratio).mean().item() * f  # P6: k3, non-negative
                    acc['clipfrac'] += ((ratio - 1).abs() > a.clip).float().mean().item() * f
                acc['pg'] += pg.item() * f; acc['vloss'] += vl.item() * f; acc['ent'] += ent.item() * f
            if a.clip_sep:  # (A1) value and policy parameters clipped apart (a separate critic must not eat the policy's norm budget)
                acc['gn_pi'] = float(torch.nn.utils.clip_grad_norm_(pi_params, 0.5)); acc['gn_v'] = float(torch.nn.utils.clip_grad_norm_(value_params, 0.5))
            else:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            for k_, v_ in acc.items():
                info[k_].append(v_)
            kl3 = acc['kl3']
            if kl3 > 1.5 * a.target_kl:
                stop = True; break
        if stop:
            break
    extra['diag'] = diag
    if a.sil > 0 and not warm:
        sil_add(obs_np, np.stack(mask), np.array(act), np.array(ret, np.float32), np.array(v_old, np.float32))
        if sil['n'] >= a.minibatch:
            pri = sil['pri'][:sil['n']] + 1e-6; p = pri / pri.sum(); sl = []
            for _ in range(a.sil_batches):
                idx = np.random.choice(sil['n'], size=a.minibatch, p=p)
                so = torch.from_numpy(sil['obs'][idx]).to(dev).float(); sm = torch.from_numpy(sil['mask'][idx]).to(dev)
                sa = torch.from_numpy(sil['act'][idx]).to(dev); sr = torch.from_numpy(sil['ret'][idx]).to(dev)
                lg, sv = net(so, sm); lpa = F.log_softmax(lg, 1).gather(1, sa[:, None]).squeeze(1)
                clipped = (sr - sv).clamp_min(0)
                loss = a.sil * (-(lpa * clipped.detach()).mean() + a.sil_vf * 0.5 * (clipped ** 2).mean())
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                sl.append(loss.item())
            diag['diag/sil_loss'] = float(np.mean(sl)); diag['diag/sil_buffer'] = int(sil['n'])
    update_no += 1
    extra['tiers'] = tiers; extra['rounds'] = rounds
    return N, {k: round(float(np.mean(v)), 5) for k, v in info.items()}, ep + 1, bc_coef, extra

# Step 2: band-head warm start from the Step 0 table (band_table.json): target = softmax(fa(config, band) / tau) over legal bands
band_fa = None
if a.band_warmstart:
    _t = json.load(open(a.band_warmstart))
    _bl = sorted(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/bands.json'))).keys())[:40]
    band_fa = {cfg: np.array([{r['band']: r['fa'] for r in v['bands']}.get(b, np.nan) for b in _bl], dtype=np.float64) for cfg, v in _t.items()}

def band_targets(obs_np, mask_np, sel):
    """(rows, A) targets for band-decision rows `sel`; the config from the observation (difficulty one-hot, co-op flag)."""
    A0 = envs[0].spec['act']['BAND']; out = np.zeros((int(sel.sum()), mask_np.shape[1]), np.float32)
    for r, k in enumerate(np.flatnonzero(sel)):
        diff = ('FUNNY', 'NORMAL', 'HARD', 'ABYSS')[int(obs_np[k, G0 + 4:G0 + 8].astype(np.float32).argmax())]
        key = ('coop4' if obs_np[k, G0 + 8] > 0.5 else 'solo') + ':' + diff
        fa = band_fa.get(key)
        legal = mask_np[k, A0:A0 + 40]
        if fa is None or not legal.any():
            continue
        v = np.where(legal & np.isfinite(fa), fa, np.nan); v = np.where(np.isfinite(v), v, np.nanmin(v) if np.isfinite(v).any() else 0)
        z = np.exp((v - v[legal].max()) / a.band_tau) * legal; out[r, A0:A0 + 40] = z / z.sum()
    return out

emit(event='start', init=a.init, params=sum(p.numel() for p in net.parameters()), thresholds=threshold, args=vars(a),
     learner_cpus=len(os.sched_getaffinity(0)), env_cpus=env_cpu_count, unit_cpus=len(full_cpus), stages=[s['name'] for s in STAGES])
last_log = time.time(); sps_steps = 0; sps_t = time.time()
t_sel = t_fwd = 0.0; prev_update = time.time()
upd_thread = None; upd_box = {}; lag_steps = 0  # main-loop time in select waits / forward passes since the last update
if eval_at and eval_at[0] <= steps:  # (scr0) the init's own in-run eval, before any update
    launch_eval(checkpoint(f'{int(steps)}')); eval_at = [x for x in eval_at if x > steps]
while steps < a.max_steps and time.time() - t0 < a.hours * 3600:
    ts = time.time(); batch_list = list(pool.poll(timeout=60)); t_sel += time.time() - ts
    for i, b in batch_list:
        for tid, tr, info in zip(b.term_ids, b.term_rew, b.term_info):
            key = (i, int(tid))
            if key in pending:
                finalize(key, float(tr), 0.0, True)
            if REC:
                hstate.pop(key, None)  # (A3) the trajectory ended: its state goes
            ci = int(info[3]); names = [c['name'] for c in STAGES[env_stage[i]]['configs']]
            if a.plr > 0:
                ts = traj_score.pop(key, None)
                if ts and ts[1]:
                    plr_add(i, proc_seed_base[i] + int(round(float(info[8]))), ts[0] / ts[1])
            name = names[ci] if ci < len(names) else '?'
            episodes.append((name, float(info[0]), float(info[1]), float(info[2]), float(info[4])))
            recent_terms.append((name, float(info[0]), float(info[4]), float(info[5]), float(info[6]), float(info[7]),
                                 [float(x) for x in info[9:18]] if len(info) >= 18 else None))
        env_counts[i] = (b.errors, b.rejected, b.violations)
        if a.plr > 0:  # matches opened since this process's last batch: a share of them replays a buffered level
            opened = b.matches - last_matches[i]; last_matches[i] = b.matches; queue = []
            for _ in range(max(0, opened)):
                if plr_rng.random() < a.plr:
                    s_ = plr_pick(i)
                    if s_ is not None:
                        queue.append(s_); plr_stats['replayed'] += 1; continue
                plr_stats['fresh'] += 1
            if queue:
                envs[i].set_configs(seeds=queue)
        if env_stage[i] != stage:
            envs[i].set_configs(STAGES[stage]['configs']); env_stage[i] = stage
        if a.mix == 'auto1' and env_mix[i] != auto_state['ver']:  # (scr0) the newest in-update mix
            envs[i].set_configs(STAGES[0]['configs']); env_mix[i] = auto_state['ver']
        if b.n == 0:
            pool.step(i, np.zeros(0, dtype=np.int32)); continue
        tf = time.time()
        with torch.no_grad():
            if REC:  # (A3) each trajectory's state before this decision (zeros at its first), stored with the transition
                H_in = np.stack([hstate.get((i, int(b.ids[k])), h_zero) for k in range(b.n)])
                with AMP():
                    logits, v, h_new = actor(torch.from_numpy(np.array(b.obs)).to(dev), torch.from_numpy(np.array(b.mask)).to(dev), torch.from_numpy(H_in).to(dev))
                h_new = h_new.float().cpu().numpy()
            else:
                with AMP():
                    logits, v = actor(torch.from_numpy(np.array(b.obs)).to(dev), torch.from_numpy(np.array(b.mask)).to(dev))
            logits, v = logits.float(), v.float()
            probs = torch.softmax(logits, 1)
            acts = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            lp = torch.log(probs.gather(1, acts[:, None]).squeeze(1).clamp_min(1e-12))
        acts_np = acts.int().cpu().numpy(); lp_np = lp.cpu().numpy(); v_np = v.cpu().numpy(); t_fwd += time.time() - tf
        for k in range(b.n):
            key = (i, int(b.ids[k]))
            if key in pending:
                finalize(key, float(b.rew[k]), float(v_np[k]), False)
            pending[key] = {'obs': b.obs[k].astype(np.float16), 'mask': b.mask[k].copy(), 'a': int(acts_np[k]), 'logp': float(lp_np[k]), 'v': float(v_np[k]),
                            'kind': int(b.kinds[k])}
            if REC:
                pending[key]['h'] = H_in[k]; hstate[key] = h_new[k]
        pool.step(i, acts_np)
        steps += b.n; sps_steps += b.n
    if a.overlap and upd_thread is not None:
        if upd_thread.is_alive() and ready < 2 * a.rollout:
            continue  # keep collecting while the learner works (at most one rollout ahead)
        upd_thread.join()
        with torch.no_grad():
            actor.load_state_dict(net.state_dict())  # the actor catches up with the learner
        n, info, epochs_done, bc_coef, extra = upd_box['result']; upd_s = upd_box['seconds']; upd_thread = None
        lag_steps = steps - upd_box['steps_at_start']
    elif ready >= a.rollout and a.overlap:
        upd_box = {'steps_at_start': steps}
        rollout_, chains, ready = chains, defaultdict(list), 0  # the learner gets its own rollout; collection goes on
        def _run():
            t_ = time.time(); upd_box['result'] = update(rollout_); upd_box['seconds'] = time.time() - t_
        upd_thread = threading.Thread(target=_run, daemon=True); upd_thread.start()
        continue
    elif ready >= a.rollout:
        rollout_, chains, ready = chains, defaultdict(list), 0
        tu = time.time(); n, info, epochs_done, bc_coef, extra = update(rollout_); upd_s = time.time() - tu; lag_steps = 0
    else:
        continue
    if True:
        interval = max(1e-6, time.time() - prev_update); prev_update = time.time()
        loop = {'diag/stall_ratio': (0.0 if a.overlap else upd_s) / interval, 'diag/frac_select_wait': t_sel / interval,
                'diag/frac_forward': t_fwd / interval, 'diag/policy_lag_steps': lag_steps}
        if a.plr > 0:
            tot_ = plr_stats['replayed'] + plr_stats['fresh']
            loop.update({'diag/plr_replay_share': plr_stats['replayed'] / max(1, tot_), 'diag/plr_levels': sum(len(L) for L in levels),
                         'diag/plr_score_mean': float(np.mean([e['score'] for L in levels for e in L.values()] or [0]))})
        t_sel = t_fwd = 0.0
        st = STAGES[stage]; roll = rolling(st['key'])
        diagj = {'diag': {k[5:]: round(v, 4) for k, v in extra['diag'].items() if isinstance(v, float) and k.startswith(('diag/entropy', 'diag/adv_', 'diag/ev_', 'diag/ret_', 'diag/share_kind', 'diag/lr', 'diag/ent_coef') + (('diag/logp', 'diag/seq') if REC else ()))}} if a.log_diag else {}
        emit(event='update', update=update_no, steps=steps, sps=round(sps_steps / max(1e-6, time.time() - sps_t)), samples=n, epochs=epochs_done,
             update_s=round(upd_s, 2), bc_coef=round(bc_coef, 4), stage=st['name'], stage_steps=steps - stage_since,
             rolling={k: rolling(k) for k in ('NORMAL', 'HARD', 'ABYSS')}, hours=round((time.time() - t0) / 3600, 3),
             env_errors=sum(c[0] for c in env_counts), env_rejected=sum(c[1] for c in env_counts), env_violations=sum(c[2] for c in env_counts),
             warmup=update_no <= a.vf_warmup, ev=round(extra['diag'].get('diag/ev_all', float('nan')), 4),
             stall=round(loop['diag/stall_ratio'], 4), **info, **diagj)
        board.update(steps, {'sps': sps_steps / max(1e-6, time.time() - sps_t), 'update_s': upd_s, 'bc_coef': bc_coef, 'samples': n,
                             'epochs': epochs_done, 'stage': stage, 'warmup': float(update_no <= a.vf_warmup),
                             'env_errors': sum(c[0] for c in env_counts), 'env_rejected': sum(c[1] for c in env_counts),
                             'env_violations': sum(c[2] for c in env_counts), **info, **extra['diag'], **loop},
                     {k: rolling(k) for k in ('NORMAL', 'HARD', 'ABYSS')}, recent_terms, Rw,
                     extra if update_no % a.hist_every == 0 else None, bonds=(extra['tiers'], extra['rounds']), R_by=reward_by)
        recent_terms.clear()
        if a.mix == 'auto1':
            auto_mix()
        bad = eval_problems()
        if sum(c[0] for c in env_counts) or sum(c[2] for c in env_counts) or bad:
            path = checkpoint(f'{int(steps)}-alert')
            emit(event='alert', steps=steps, env_errors=sum(c[0] for c in env_counts), env_violations=sum(c[2] for c in env_counts),
                 evals=bad, checkpoint=path)
            break
        sps_steps = 0; sps_t = time.time()
        # curriculum
        if stage < len(STAGES) - 1:
            in_stage = steps - stage_since
            thr = threshold.get(st['key'], 0.8 if st['key'] == 'NORMAL' else 0.4)
            if (in_stage >= a.stage_min and roll and roll['n'] >= 500 and roll['win'] >= thr) or in_stage >= a.stage_max:
                stage += 1; stage_since = steps; episodes.clear()
                emit(event='stage', stage=STAGES[stage]['name'], steps=steps, reason='threshold' if in_stage < a.stage_max else 'max-steps', rolling=roll, threshold=thr)
        if steps >= next_ckpt:
            path = checkpoint(f'{int(steps)}'); next_ckpt += a.ckpt_every
            if steps >= next_eval or (eval_at and steps >= eval_at[0]):  # (scr0) --eval-at adds evals, never removes one
                launch_eval(path)
                if steps >= next_eval:
                    next_eval += a.eval_every
                eval_at = [x for x in eval_at if x > steps]
if upd_thread is not None:  # T1: the last overlapped update finishes before the final checkpoint
    upd_thread.join()
path = checkpoint('final')
emit(event='end', steps=steps, hours=round((time.time() - t0) / 3600, 3), stage=STAGES[stage]['name'])
launch_eval(path, block=True)
for p, _ in list(evals):
    p.wait()
eval_problems()
board.close()
pool.close()
