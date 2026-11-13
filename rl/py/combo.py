"""Team band combos (11-13): every model seat of a room switches together. For each 3-band set of --bands the model seats'
own band decisions (after the human proxies, in pick order) take the set's bands (env forceBand 'team:...'; masks and the
human-bond rule apply; a seed where the set cannot be completed is dropped), against the checkpoint's own head on the same
paired seeds. Judged on win. Yields to a shared eval lane (no new job while a lane eval.py runs).
  python combo.py --ckpt ck.pt --out dir --config coop4h1:ABYSS --bands a,b,... [--size 3] [--matches 96] [--only 'a+b+c;...'] [--par 40]"""
import argparse, itertools, json, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--config', default='coop4h1:ABYSS')
ap.add_argument('--bands', default=''); ap.add_argument('--size', type=int, default=3); ap.add_argument('--matches', type=int, default=96)
ap.add_argument('--only', default='', help="explicit sets 'a+b+c;d+e+f' instead of all --size subsets of --bands")
ap.add_argument('--par', type=int, default=40); ap.add_argument('--node', default='node'); ap.add_argument('--lane-q', default='')
ap.add_argument('--report-only', action='store_true'); ap.add_argument('--seed', type=int, default=7_000_000, help='eval seed base (a disjoint confirmation: e.g. 7500000)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
sets = [tuple(s.split('+')) for s in a.only.split(';') if s] if a.only else list(itertools.combinations(a.bands.split(','), a.size))
ch = next((c for c in (8, 12, 6, 4, 2) if a.matches % c == 0 and a.matches // c >= 4), 1)
name = lambda s: 'team-' + '+'.join(x.replace('band_', '') for x in s)
jobs = [('base', [])] + [(name(s), ['--force-band', 'team:' + ','.join(s)]) for s in sets]
cmd0 = [sys.executable, os.path.join(here, 'eval.py'), '--ckpt', a.ckpt, '--greedy', '--device', 'cuda', '--node', a.node, '--configs', a.config,
        '--matches', str(a.matches), '--chunks', str(ch), '--envs', '6', '--rehearsal', '0', '--boss-rehearsal', '0', '--mate-rehearsal', '3', '--band-policy', '1', '--seed', str(a.seed)]
lane_busy = lambda: bool(a.lane_q) and subprocess.run(['pgrep', '-u', os.environ.get('USER') or str(os.getuid()), '-f', f'eval.py --ckpt {a.lane_q}'], capture_output=True).returncode == 0
if not a.report_only:
    todo = [(n, x) for n, x in jobs if not os.path.exists(os.path.join(a.out, n + '.json'))]
    print(f'{time.strftime("%H:%M:%S")} {len(todo)} of {len(jobs)} jobs, {a.matches} seeds, {ch} chunks', flush=True)
    run, t0 = [], time.time()
    while todo or run:
        while todo and sum(c for c, _ in run) + ch <= a.par and not lane_busy():
            n, x = todo.pop(0); out = os.path.join(a.out, n + '.json')
            run.append((ch, subprocess.Popen(cmd0 + x + ['--out', out], stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT)))
        time.sleep(3)
        run = [(c, p) for c, p in run if p.poll() is None]
    print(f'{time.strftime("%H:%M:%S")} done in {time.time() - t0:.0f} s', flush=True)
def pm(n):
    p = os.path.join(a.out, n + '.json')
    return {m['seed']: m for m in json.load(open(p))['configs'][a.config]['perMatch']} if os.path.exists(p) else {}
B = pm('base'); rows = []
for s in sets:
    R = pm(name(s))
    got = [k for k, m in R.items() if k in B and len(m.get('team', [])) >= len(s) and all(t.get('how') == 'forced' and t.get('got') == t.get('band') for t in m['team'])]
    why = {}
    for k, m in R.items():
        if k in got: continue
        hs = [t.get('how') for t in m.get('team', []) if t.get('how') != 'forced'] or ['incomplete']
        for h in set(hs): why[h] = why.get(h, 0) + 1
    if len(got) < 8: rows.append({'set': s, 'n': len(got), 'dropped': len(R) - len(got), 'why': why}); continue
    d = np.array([R[k]['win'] - B[k]['win'] for k in got], float)
    rows.append({'set': s, 'n': len(got), 'win': float(np.mean([R[k]['win'] for k in got])), 'base': float(np.mean([B[k]['win'] for k in got])),
                 'dwin': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))), 'dfa': float(np.mean([R[k]['faShare'] - B[k]['faShare'] for k in got])),
                 'dropped': len(R) - len(got), 'why': why, 'baseWinDropped': float(np.mean([B[k]['win'] for k in R if k not in got and k in B])) if len(R) > len(got) else None})
rows.sort(key=lambda r: -r.get('dwin', -9))
bw = np.mean([m['win'] for m in B.values()]) if B else float('nan')
print(f'{a.config}: base (own head) win {bw:.3f} on {len(B)} seeds; {len(sets)} sets')
for r in rows:
    if 'dwin' in r: print(f"  {'+'.join(x.replace('band_', '') for x in r['set']):32s} n {r['n']:3d} win {r['win']:.3f} vs {r['base']:.3f}  dwin {r['dwin']:+.3f} ± {r['se']:.3f} ({r['dwin'] / max(r['se'], 1e-9):+.1f} SE)  dfa {r['dfa']:+.3f}"
                          f"  | dropped {r['dropped']} {r['why']} base win on dropped {r['baseWinDropped'] if r['baseWinDropped'] is None else round(r['baseWinDropped'], 3)}")
    else: print(f"  {'+'.join(x.replace('band_', '') for x in r['set']):32s} n {r['n']:3d} (too few seeds complete the set)")
json.dump({'config': a.config, 'baseWin': bw, 'rows': rows}, open(os.path.join(a.out, 'combo.json'), 'w'), indent=1)
