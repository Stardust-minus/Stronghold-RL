"""Concatenate bandobs2 outputs: python obsmerge.py <out prefix> <in prefix> [<in prefix> ...]"""
import json, sys, numpy as np
out, ins = sys.argv[1], sys.argv[2:]
metas = [json.load(open(p + '.json')) for p in ins]
D, A = metas[0]['D'], metas[0]['A']
assert all(m['D'] == D and m['A'] == A and m['bandList'] == metas[0]['bandList'] for m in metas)
O = np.concatenate([np.fromfile(p + '.f32', np.float32).reshape(m['n'], D) for p, m in zip(ins, metas)])
M = np.concatenate([np.fromfile(p + '.u8', np.uint8).reshape(m['n'], A) for p, m in zip(ins, metas)])
O.tofile(out + '.f32'); M.tofile(out + '.u8')
meta = {'n': int(O.shape[0]), 'D': D, 'A': A, 'bandList': metas[0]['bandList'], 'bandAction0': metas[0]['bandAction0'],
        **{k: sum((m[k] for m in metas), []) for k in ('configs', 'humans', 'seats')}}
json.dump(meta, open(out + '.json', 'w')); print(meta['n'], 'rows')
