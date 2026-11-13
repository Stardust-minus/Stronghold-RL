"""Record decision observations (obs, mask, kind) of a checkpoint's own greedy play, for architecture equivalence checks.
python obsrec.py --ckpt ck.pt --out obs.npz [--configs solo:ABYSS,coop4h1:ABYSS,coop4:ABYSS,coop4h3:ABYSS] [--matches 8] [--keep 6000]"""
import argparse, os, numpy as np, torch
import eval as E
from model import load

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--configs', default='solo:ABYSS,coop4h1:ABYSS,coop4:ABYSS,coop4h3:ABYSS'); ap.add_argument('--matches', type=int, default=8)
ap.add_argument('--keep', type=int, default=6000); ap.add_argument('--node', default='node'); ap.add_argument('--device', default='cpu')
a = ap.parse_args()
torch.set_num_threads(4)
net, _ = load(a.ckpt, a.device); net.eval()
O, M, K = [], [], []
@torch.no_grad()
def policy(b):
    obs = np.array(b.obs); mask = np.array(b.mask)
    O.append(obs.copy()); M.append(mask.copy()); K.append(np.array(b.kinds).copy() if hasattr(b, 'kinds') else np.zeros(len(obs), np.int32))
    lg, _ = net(torch.from_numpy(obs).to(a.device), torch.from_numpy(mask).to(a.device))
    return lg.argmax(1).int().cpu().numpy()
ns = argparse.Namespace(configs=a.configs, matches=a.matches, chunks=min(4, a.matches), envs=6, seed=7_300_000, legacy_mates=0.0, rehearsal=0, trips=False,
                        obs_version=3, force_band=None, band_policy=1.0, boss_rehearsal=0, mate_rehearsal=3, legacy=False, node=a.node)
r = E.run(ns, policy)
O = np.concatenate(O); M = np.concatenate(M).astype(bool); K = np.concatenate(K)
rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(O), min(a.keep, len(O)), replace=False))
np.savez_compressed(a.out, obs=O[idx], mask=M[idx], kinds=K[idx])
print({'decisions': int(len(O)), 'kept': int(len(idx)), 'kinds': {int(k): int((K[idx] % 16 == k).sum()) for k in np.unique(K[idx] % 16)},
       'win': {c: round(v['teamWin'], 3) for c, v in r['configs'].items()}})
