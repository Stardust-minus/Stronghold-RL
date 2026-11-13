"""dumpref.py <obs.npz> <out prefix> <ckpt> <name>...: raw obs / mask and each checkpoint's torch logits for jsbench.mjs"""
import sys, json, numpy as np, torch
from model import load
torch.set_num_threads(4)
d = np.load(sys.argv[1]); out = sys.argv[2]; O, M = d['obs'], d['mask']
O.astype('<f4').tofile(out + '.obs.f32'); M.astype(np.uint8).tofile(out + '.mask.u8')
for ck, name in zip(sys.argv[3::2], sys.argv[4::2]):
    net, _ = load(ck); net.eval()
    with torch.no_grad(): lg, _ = net(torch.from_numpy(O), torch.from_numpy(M))
    lg.numpy().astype('<f4').tofile(out + f'.{name}.f32')
json.dump({'n': int(len(O)), 'D': int(O.shape[1]), 'A': int(M.shape[1]), 'ref': 'W1'}, open(out + '.json', 'w'))
