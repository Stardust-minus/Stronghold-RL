"""Export a picker_train5 checkpoint (the spatial normal-round picker) in infer5.mjs's sp-policy5/1 format: the tower as is,
the head's three Linear layers as trunk.0 / trunk.2 (ReLU, ReLU) and scalar (4 outputs: score, leaks, layers, kills), no pointers.
  python export_p5.py --ckpt head5.pt --member $B/A/lookahead.json --out <base>   ->  <base>.json + <base>.bin  (SPPolicy5.load(base))"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EMB, SCALARS
from model5 import Tower, PIECE_FIELD
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--member', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location='cpu'); args = ck['args']; spec = json.load(open(ck['spec'])); member = json.load(open(a.member))['pieceBonds']
tw = Tower(spec, member, args['d'], args['layers'], args['heads'], 2, True, True)
st = ck['state']; tw.load_state_dict({k[len('tower.'):]: v for k, v in st.items() if k.startswith('tower.')})
ren = {'head.0': 'trunk.0', 'head.2': 'trunk.2', 'head.4': 'scalar'}
tensors, chunks, off = {}, [], 0
for name, t in st.items():
    p = name.rsplit('.', 1)[0]; nn_ = ren.get(p, p) + '.' + name.rsplit('.', 1)[1] if p in ren else name
    arr = t.detach().float().numpy().astype('<f4'); tensors[nn_] = {'offset': off, 'shape': list(arr.shape)}; chunks.append(arr.tobytes()); off += arr.size
lay = {n: {'offset': o, 'count': c, 'width': w, 'ids': [[f, v] for f, v in ids], 'numeric': getattr(tw, f'num_{n}').tolist()} for n, (o, c, w, ids) in tw.lay.items()}
cfg = {'d': args['d'], 'layers': args['layers'], 'heads': args['heads'], 'ffn': 2, 'qkn': True}
mem = {str(k): [int(x) for x in v] for k, v in member.items()}
manifest = {'format': 'sp-policy5/1', 'kind': 'p5-picker', 'dataId': spec['dataId'], 'D': spec['dim'], 'A': 4, 'emb': EMB, 'scalars': SCALARS, 'cfg': cfg,
            'tok': tw.tok, 'merge': tw.merge, 'gmerge': tw.gmerge, 'pieceField': {k: v for k, v in PIECE_FIELD.items() if k in tw.tok},
            'NB': tw.NB, 'pieceVocab': int(tw.member.shape[0]), 'member': mem, 'layout': lay, 'band': None, 'pointer': [],
            'memberBias': tw.beta is not None, 'tensors': tensors, 'trainArgs': args, 'val': ck.get('val')}
open(a.out + '.bin', 'wb').write(b''.join(chunks)); json.dump(manifest, open(a.out + '.json', 'w'))
print('exported', len(tensors), 'tensors', off, 'floats')
