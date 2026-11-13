"""Deployed multiply-adds per decision (infer2 rules) for W1 / W2, and AT's extra (entity attention over present rows)."""
import sys, numpy as np, torch
from model import load
from model2 import SPNet2, deployed_macs
torch.set_num_threads(4)
d = np.load(sys.argv[1]); I = sys.argv[2]
O = torch.from_numpy(d['obs'][:1500]); M = torch.from_numpy(d['mask'][:1500])
w1, _ = load(I + '/W1.pt'); s1 = SPNet2(w1.spec, width=1.0); s1.load_state_dict(w1.state_dict())
w2, _ = load(I + '/W2.pt'); at, _ = load(I + '/AT.pt')
m1, m2 = deployed_macs(s1, O, M), deployed_macs(w2, O, M)
# AT extra per decision: in-proj of present rows, bond-row tokens, global token, L layers over T present tokens, out-proj
da = at.cfg['attn_dim']; L = at.cfg['entattn']; lay = at.ea_layers[0]
lin = lambda m_: m_.in_features * m_.out_features
per_layer_tok = lin(lay.qkv) + lin(lay.o) + lin(lay.f1) + lin(lay.f2)
ex, Ts = [], []
off, nb, wb = at.ea_bonds_rows
for k in range(len(O)):
    o = O[k]; T = 1; t = lin(at.ea_g)
    for n in at.ea_ents:
        off_, cnt, w, _ = at.layout[n]; rows = int((o[off_:off_ + cnt * w].reshape(cnt, w)[:, 0] > 0.5).sum())
        T += rows; t += rows * (lin(at.ea_in[n]) + lin(at.ea_out[n]))
    br = int((o[off:off + nb * wb].reshape(nb, wb)[:, 0] > 0).sum()); T += br; t += br * lin(at.ea_bond)
    t += L * (T * per_layer_tok + 2 * T * T * da)
    ex.append(t); Ts.append(T)
print({'W1_macs': round(m1), 'W2_macs': round(m2), 'W2/W1': round(m2 / m1, 2), 'AT_extra_macs': round(float(np.mean(ex))), 'AT/W1': round((m1 + np.mean(ex)) / m1, 2),
       'AT_tokens_mean': round(float(np.mean(Ts)), 1), 'AT_tokens_p95': float(np.percentile(Ts, 95)), 'ea_ents': at.ea_ents,
       'bond_rows': nb, 'rows_per_block': {n: at.layout[n][1] for n in at.ea_ents}})
