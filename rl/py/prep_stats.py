"""Decisions per prep per AI seat, refreshes per prep, decisions per shop state, single-legal-action share (ppo3 sizing):
python prep_stats.py --ckpt ck.pt [--matches 24]. Greedy policy (deployment), CPU."""
import argparse, json
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(2)
from spenv import NodeEnv
from model import load
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--matches', type=int, default=24)
ap.add_argument('--configs', default='solo:HARD,solo:ABYSS,coop4:HARD,coop4:ABYSS,coop4:NORMAL')
a = ap.parse_args()
net, _ = load(a.ckpt); net.eval()
v = 4 if 'tiles' in net.spec['blocks'] else 3 if 'bands' in net.spec['blocks'] else 2
env = NodeEnv({'envs': 6, 'configs': a.configs.split(','), 'seed': 3800000000, 'envSeed': 5, 'maxMatches': a.matches, 'humans': 'random',
               'legacyMates': 0, 'rehearsal': 0, 'bossRehearsal': 3, 'obsVersion': v})
act = env.spec['act']; G0 = env.spec['blocks']['global']['offset']
prep = defaultdict(lambda: {'dec': 0, 'ref': 0, 'single': 0})  # (traj, round) -> counts
single = total = 0
env.reset_async(); b = env.wait()
while b.n:
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(np.array(b.obs)), torch.from_numpy(np.array(b.mask)))
    acts = lg.argmax(1).int().numpy(); rounds = np.rint(np.array(b.obs)[:, G0] * 15).astype(int)
    for k in range(b.n):
        legal = int(b.mask[k].sum()); total += 1; single += legal == 1
        if b.kinds[k] != 0:
            continue
        x = prep[(int(b.ids[k]), int(rounds[k]))]; x['dec'] += 1; x['single'] += legal == 1
        if acts[k] == act['REFRESH']:
            x['ref'] += 1
    env.step_async(acts); b = env.wait()
env.close()
d = np.array([x['dec'] for x in prep.values()]); r = np.array([x['ref'] for x in prep.values()])
q = lambda x, p: float(np.percentile(x, p))
print(json.dumps({'preps': len(d), 'decisionsPerPrep': {'mean': float(d.mean()), 'p50': q(d, 50), 'p90': q(d, 90), 'max': int(d.max())},
                  'refreshesPerPrep': {'mean': float(r.mean()), 'p50': q(r, 50), 'p90': q(r, 90), 'max': int(r.max())},
                  'shopStatesPerPrep': float((r + 1).mean()), 'decisionsPerShopState': float(d.sum() / (r + 1).sum()),
                  'singleLegalShare': single / total, 'decisionsTotal': total}, indent=1))
