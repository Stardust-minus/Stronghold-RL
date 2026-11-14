"""(A3) checks: python rnntest.py <A1 py dir> <v5 ckpt> <shard base> <tmpdir>"""
import json, sys, importlib, os, subprocess, numpy as np, torch
a1, ck, shard, tmp = sys.argv[1:5]
from data import record_dtype, unpack_mask
import model as M3, model5 as M5_3
torch.set_num_threads(4); torch.manual_seed(0)
meta = json.load(open(shard + '.meta.json')); rec = np.memmap(shard + '.bin', dtype=record_dtype(meta), mode='r'); A = meta['A']
idx = np.sort(np.random.default_rng(0).choice(len(rec), 300, replace=False))
O = torch.from_numpy(np.asarray(rec['obs'][idx]).astype(np.float32)); MK = torch.from_numpy(unpack_mask(np.asarray(rec['mask'][idx]), A))
net3, _ = M3.load(ck); net3.eval()
# 1. flags-off: code-A1 model5 vs this tree's model5 on the same checkpoint (a subprocess with the A1 path)
np.save(tmp + '/o.npy', O.numpy()); np.save(tmp + '/m.npy', MK.numpy())
code = f"import sys; sys.path.insert(0, '{a1}'); import numpy as np, torch; torch.set_num_threads(4); from model import load; n,_=load('{ck}'); n.eval();\nwith torch.no_grad(): l,v=n(torch.from_numpy(np.load('{tmp}/o.npy')), torch.from_numpy(np.load('{tmp}/m.npy')))\nnp.save('{tmp}/l1.npy', l.numpy()); np.save('{tmp}/v1.npy', v.numpy())"
subprocess.run([sys.executable, '-c', code], check=True, cwd=tmp)
with torch.no_grad(): l3, v3 = net3(O, MK)
print('1 flags-off identical logits/value:', bool(np.array_equal(np.load(tmp + '/l1.npy'), l3.numpy())), bool(np.array_equal(np.load(tmp + '/v1.npy'), v3.numpy())))
# 2. recurrent init is function-preserving
for cell in ('lstm', 'gru'):
    subprocess.run([sys.executable, 'addrnn.py', '--init', ck, '--cell', cell, '--hidden', '128', '--out', f'{tmp}/{cell}.pt'], check=True, capture_output=True)
    nr, _ = M3.load(f'{tmp}/{cell}.pt'); nr.eval()
    with torch.no_grad(): lr, vr, hn = nr(O, MK, None)
    print(f'2 {cell} init identical logits/value:', bool(torch.equal(lr, l3)), bool(torch.equal(vr, v3)), 'state', tuple(hn.shape), 'nonzero state', bool(hn.abs().sum() > 0))
    # 3. sequence forward == step forward with the state carried (non-zero rnn_out so the memory matters)
    with torch.no_grad():
        nr.rnn_out.weight.normal_(0, 0.5); nr.rnn_out.bias.normal_(0, 0.1)
    traj = np.asarray(rec['traj']); t0 = traj[idx[0]]; rows = np.flatnonzero(traj == t0)[:60]
    To = torch.from_numpy(np.asarray(rec['obs'][rows]).astype(np.float32)); Tm = torch.from_numpy(unpack_mask(np.asarray(rec['mask'][rows]), A))
    st = torch.zeros(1, nr.state_size); outs, states = [], []
    with torch.no_grad():
        for k in range(len(rows)):
            states.append(st); lg, _, st = nr(To[k:k + 1], Tm[k:k + 1], st); outs.append(lg)
    LS = torch.cat(outs); S = torch.cat(states)
    worst = 0.0
    for (s0, Kb, Lq) in [(0, 16, 32), (16, 16, 32), (40, 16, 32), (5, 16, 32), (28, 8, 32)]:
        nb = min(Kb, s0); nl = min(Lq, len(rows) - s0)
        r = [-1] * (Kb - nb) + list(range(s0 - nb, s0 + nl)) + [-1] * (Lq - nl)
        R = torch.tensor(r); V = R >= 0
        ob = To[R.clamp_min(0)].unsqueeze(0); mb = Tm[R.clamp_min(0)].unsqueeze(0)
        with torch.no_grad(): lgs, _ = nr.forward_seq(ob, mb, S[s0 - nb:s0 - nb + 1], V.unsqueeze(0), Kb)
        lv = V[Kb:]
        ref = LS[R[Kb:][lv]]; got = lgs[0][lv]
        d = float(((got - ref).abs() * Tm[R[Kb:][lv]]).max()); worst = max(worst, d)
    print(f'3 {cell} forward_seq vs stepping (burn-in, left/right padding): max |dlogit| {worst:.2e}')
