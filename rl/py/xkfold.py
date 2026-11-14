"""xplace (X7, 11-13): offline k-fold gate for boss layout scorers (QNet).

Every held-out decision is scored by the model(s) of its fold: fold = hash(match seed) % val-mod (xtrain --val-fold).
The pick is the argmax Q over the deploy candidates (cur + the first 7 legacy = K8, as the boss picker sees them; also K12).
Utility = seed-0 boss utility vs the current layout (xtrain.util, clipped to +-uclip), so it is unbiased for held-out decisions.
  python xkfold.py --data 'a/*.jsonl,b/*.jsonl' --model q5=models/q5.pt --model r1=models/r1-f{f}.pt --ens e=r1+r2 --out rep.json
A model path with {f} is per fold; without it the same model scores every fold (a model trained without this data).
"""
import argparse, json, sys, os, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtrain as X


def load(p, dev):
    ck = torch.load(p, map_location=dev); c = ck['cfg']
    n = X.QNet(c['FP'], c['FT'], c['FE'], c['FG'], c['V'], d=c['d'], de=c['de'], attn=c.get('attn', 0), heads=c.get('heads', 4), qattn=c.get('qattn', 1), cf=c.get('cf', 0)).to(dev)
    n.load_state_dict(ck['state']); n.eval(); return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--model', action='append', default=[]); ap.add_argument('--ens', action='append', default=[])
    ap.add_argument('--val-mod', type=int, default=6); ap.add_argument('--folds', default=''); ap.add_argument('--uclip', type=float, default=30.0)
    ap.add_argument('--out', default=''); ap.add_argument('--device', default='cuda'); ap.add_argument('--per-dec', default='')
    ap.add_argument('--wutil', default=''); ap.add_argument('--winfit', default='')  # X10: win utility (round 14, FA known), in P(win) points
    a = ap.parse_args()
    X.UCLIP = a.uclip
    if a.wutil: X.WFA = json.load(open(a.wutil)); X.WFIT = json.load(open(a.winfit))
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    recs = []
    for p in a.data.split(','): recs += X.load_records(p, 'boss')
    recs = [r for r in recs if r.get('salt')]
    if a.wutil: recs = [r for r in recs if X.wctx(r) is not None]
    fold = np.array([(r['seed'] * 2654435761) % (2 ** 32) % a.val_mod for r in recs])
    folds = [int(x) for x in a.folds.split(',')] if a.folds else list(range(a.val_mod))
    models = [m.split('=', 1) for m in a.model]; ens = [(e.split('=', 1)[0], e.split('=', 1)[1].split('+')) for e in a.ens]
    names = [n for n, _ in models] + [n for n, _ in ens]
    out = {k: {'K8': [], 'K12': []} for k in names}; meta = []
    for f in folds:
        ii = np.nonzero(fold == f)[0]
        if not len(ii): continue
        nets = {n: load(p.replace('{f}', str(f)), dev) for n, p in models}
        R = [recs[i] for i in ii]; D = X.Data(R, R[0]['dims'], seed0=True)
        for q, r in enumerate(R):
            cs = r['cands']; s0 = cs[0]['s'][0] if cs[0]['s'] else None
            if not s0: continue
            legs = [j for j, c in enumerate(cs) if c['src'] == 'leg' and c['s'] and c['s'][0]]
            k12 = [0] + legs[:11]
            if len(k12) < 2: continue
            X.WCTX = X.wctx(r); u = np.array([X.util(True, cs[j]['s'][0], s0) for j in k12]); n = min(r['n'], X.NMAX)
            tg = np.full((len(k12), X.NMAX, 2), -1, np.int64)
            for t, j in enumerate(k12): tg[t, :n] = np.array(cs[j]['a'][:n])
            cfa = np.zeros((len(k12), X.XCF), np.float32)
            for t, j in enumerate(k12):
                if cs[j].get('f') and len(cs[j]['f']) == X.XCF: cfa[t] = cs[j]['f']
            cft = torch.from_numpy(cfa).to(dev); b, cur, boss, w = D.batch(np.array([q]), dev); M = len(k12)
            rep = lambda x: x.expand(M, *x.shape[1:]).contiguous()
            Qs = {}
            with torch.no_grad():
                for nm, net in nets.items():
                    Qs[nm] = net.score(rep(b['fp']), rep(b['ids']), rep(b['ft']), rep(b['fe']), rep(b['fg']), rep(b['legal']), rep(b['pm']), torch.from_numpy(tg).to(dev), cft, cft[:1].expand(M, X.XCF)).cpu().numpy()
            for en, mem in ens:  # z-scored sum over the K8 (resp. K12) candidates
                Qs[en] = None
            m8 = min(8, M)
            for nm in names:
                for key, mm in (('K8', m8), ('K12', M)):
                    if nm in dict(ens):
                        zs = [(Qs[x][:mm] - Qs[x][:mm].mean()) / (Qs[x][:mm].std() + 1e-6) for x in dict(ens)[nm]]; Q = np.sum(zs, 0)
                    else: Q = Qs[nm][:mm]
                    out[nm][key].append(float(u[int(np.argmax(Q))]))
            meta.append(dict(f=int(f), seed=int(r['seed']), cfg=r['cfg'], round=r['round'], o8=float(u[:m8].max()), o12=float(u.max()), m=M))
    cfgs = np.array([m['cfg'] for m in meta]); res = {'n': len(meta), 'folds': folds, 'models': dict(models), 'ens': {k: v for k, v in ens}}
    se = lambda v: float(np.std(v) / max(1, len(v)) ** 0.5)
    base = names[0]
    for split in ['all'] + sorted(set(cfgs.tolist())):
        sel = np.ones(len(meta), bool) if split == 'all' else cfgs == split
        o8 = np.array([m['o8'] for m in meta])[sel]; o12 = np.array([m['o12'] for m in meta])[sel]
        line = f'{split}: n {int(sel.sum())} oracle K8 {o8.mean():+.2f} K12 {o12.mean():+.2f} |'
        rs = {'n': int(sel.sum()), 'oracleK8': float(o8.mean()), 'oracleK12': float(o12.mean())}
        for nm in names:
            v8 = np.array(out[nm]['K8'])[sel]; v12 = np.array(out[nm]['K12'])[sel]; b8 = np.array(out[base]['K8'])[sel]
            rs[nm] = {'K8': float(v8.mean()), 'K8se': se(v8), 'K12': float(v12.mean()), 'dK8': float((v8 - b8).mean()), 'dK8se': se(v8 - b8)}
            line += f' {nm} {v8.mean():+.2f}±{se(v8):.2f}' + (f' (d {(v8 - b8).mean():+.2f}±{se(v8 - b8):.2f})' if nm != base else '') + f' K12 {v12.mean():+.2f};'
        res[split] = rs; print(line, flush=True)
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)
    if a.per_dec: json.dump({'meta': meta, 'K8': {k: v['K8'] for k, v in out.items()}}, open(a.per_dec, 'w'))


if __name__ == '__main__':
    main()
