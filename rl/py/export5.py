"""(A1, 11-13) Export the policy of a model5.SPNet5 checkpoint for the plain-JS forward (tools/rl/infer5.mjs): <out>.json
(manifest: layout, cfg, membership table) + <out>.bin (f32 LE tensors, policy only: the critic is never exported).
python export5.py --ckpt run/model.pt --out run/policy5"""
import argparse, json
import torch
from model import load, EMB, SCALARS
from model5 import PIECE_FIELD
torch.set_num_threads(4)

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
net, ck = load(a.ckpt)
assert net.cfg.get('arch') == 'v5', 'not an SPNet5 checkpoint'
tensors, chunks, off = {}, [], 0
for name, t in net.state_dict().items():
    if name.startswith('value.'):
        continue
    arr = t.detach().cpu().float().numpy().astype('<f4')
    tensors[name] = {'offset': off, 'shape': list(arr.shape)}
    chunks.append(arr.tobytes()); off += arr.size
tw = net.tower
lay = {n: {'offset': o, 'count': c, 'width': w, 'ids': [[f, v] for f, v in ids], 'numeric': getattr(tw, f'num_{n}').tolist()}
       for n, (o, c, w, ids) in tw.lay.items()}
band = None
if 'bands' in net.pointer:
    o, c, w, ids = net.band_lay
    band = {'offset': o, 'count': c, 'width': w, 'ids': [[f, v] for f, v in ids], 'numeric': net.num_bands.tolist()}
cfg = {k: v for k, v in net.cfg.items() if k not in ('member', 'critic')}
member = {str(k): [int(x) for x in v] for k, v in (net.cfg.get('member') or {}).items()}
manifest = {'format': 'sp-policy5/1', 'dataId': net.spec['dataId'], 'D': net.D, 'A': net.A, 'emb': EMB, 'scalars': SCALARS, 'cfg': cfg,
            'tok': tw.tok, 'merge': tw.merge, 'gmerge': tw.gmerge, 'pieceField': {k: v for k, v in PIECE_FIELD.items() if k in tw.tok},
            'NB': tw.NB, 'pieceVocab': int(tw.member.shape[0]), 'member': member, 'layout': lay, 'band': band, 'pointer': net.pointer,
            'memberBias': tw.beta is not None, 'tensors': tensors, 'steps': ck.get('steps')}
open(a.out + '.bin', 'wb').write(b''.join(chunks))
json.dump(manifest, open(a.out + '.json', 'w'))
print('exported', len(tensors), 'tensors', off, 'floats', round(off * 4 / 1e6, 2), 'MB')
