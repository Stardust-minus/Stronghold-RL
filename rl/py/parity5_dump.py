"""(A1) Reference for infer5.mjs parity: N records of a shard -> <out>.bin (f32 obs | u8 mask | f32 torch CPU logits) + <out>.json.
python parity5_dump.py --ckpt bc.pt --shard <dir>/<tag> --n 500 --out p5"""
import argparse, json, numpy as np, torch
from data import record_dtype, unpack_mask
from model import load
torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--shard', required=True)
ap.add_argument('--n', type=int, default=500); ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
meta = json.load(open(a.shard + '.meta.json')); rec = np.memmap(a.shard + '.bin', dtype=record_dtype(meta), mode='r')
idx = np.sort(np.random.default_rng(a.seed).choice(len(rec), a.n, replace=False))
obs = np.asarray(rec['obs'][idx]).astype(np.float32); A = meta['A']; mask = unpack_mask(np.asarray(rec['mask'][idx]), A)
net, _ = load(a.ckpt, 'cpu'); net.eval(); net.with_value = False
with torch.no_grad():
    lg, _ = net(torch.from_numpy(obs), torch.from_numpy(mask))
open(a.out + '.bin', 'wb').write(obs.tobytes() + mask.astype(np.uint8).tobytes() + lg.numpy().astype('<f4').tobytes())
json.dump({'n': a.n, 'D': obs.shape[1], 'A': A, 'kinds': np.asarray(rec['kind'][idx]).tolist()}, open(a.out + '.json', 'w'))
print('dumped', a.n, obs.shape)
