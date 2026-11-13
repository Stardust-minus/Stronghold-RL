"""Band table v2 (11-13, deploy band semantics): seat ai_0 forced to each band at its own policy band decision (after the
human proxies, first of the model seats; env forceAt), and a uniformly random allowed band as the baseline, all on one
paired seed set per config. A band whose tied bonds meet a human seat's band bonds or announced 主盟约 is masked: that seed
counts as not obtained (the policy's own pick plays it and it is dropped). The other model seats pick with the checkpoint's
head (band policy 1); greedy; rehearsal 0, boss rehearsal 0, human proxies at --mate-rehearsal (3, as the headline evals).
  python band_table2.py --ckpt ck.pt --out dir [--groups 'solo:ABYSS*96;coop4h1:ABYSS,coop4h2:ABYSS,coop4h3:ABYSS*16;coop2h1:ABYSS*48']
    [--par 40] [--bands all|a,b] [--node node]
Restart-safe: a finished (band, group) json is kept. Output dir/band_table2.json: per table key (coop4h1..3 pooled as
coop4h:ABYSS) the baseline and per band n obtained, paired d fa / d win vs the baseline (mean, SE), fa, win, how often masked."""
import argparse, json, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--groups', default='solo:ABYSS*96;coop4h1:ABYSS,coop4h2:ABYSS,coop4h3:ABYSS*16;coop2h1:ABYSS*48')
ap.add_argument('--par', type=int, default=40, help='node processes (= chunks) running at once')
ap.add_argument('--bands', default='all'); ap.add_argument('--node', default='node'); ap.add_argument('--device', default='cuda')
ap.add_argument('--mate-rehearsal', type=int, default=3); ap.add_argument('--min-n', type=int, default=0, help='0: 1/4 of the group seeds')
ap.add_argument('--report-only', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
bands = sorted(json.load(open(os.path.join(here, '../../../data/bands.json'))).keys()) if a.bands == 'all' else a.bands.split(',')
groups = []
for g in a.groups.split(';'):
    cf, n = g.split('*'); n = int(n); ch = next((c for c in (8, 6, 4, 3, 2) if n % c == 0 and n // c >= 4), 1)
    groups.append({'configs': cf.split(','), 'matches': n, 'chunks': ch, 'name': cf.split(',')[0].split(':')[0] + (f'+{len(cf.split(","))-1}' if ',' in cf else '')})
def key_of(cfg):
    mode, diff = cfg.split(':')
    return (('coop4h' if mode.startswith('coop4h') else mode) + ':' + diff)
jobs = []
for b in ['random'] + bands:
    for g in groups:
        out = os.path.join(a.out, g['name'], f'{b}.json'); os.makedirs(os.path.dirname(out), exist_ok=True)
        jobs.append((g['chunks'] * len(g['configs']), out, [sys.executable, os.path.join(here, 'eval.py'), '--ckpt', a.ckpt, '--greedy', '--device', a.device, '--node', a.node,
            '--configs', ','.join(g['configs']), '--matches', str(g['matches']), '--chunks', str(g['chunks']), '--envs', '6', '--rehearsal', '0',
            '--boss-rehearsal', '0', '--mate-rehearsal', str(a.mate_rehearsal), '--band-policy', '1', '--force-band', b, '--out', out]))
if not a.report_only:
    todo = [j for j in jobs if not os.path.exists(j[1])]
    print(f'{time.strftime("%H:%M:%S")} {len(todo)} of {len(jobs)} jobs to run, par {a.par}', flush=True)
    run, t0, done = [], time.time(), 0
    while todo or run:
        while todo and (not run or sum(c for c, _, _ in run) + todo[0][0] <= a.par):
            c, out, cmd = todo.pop(0)
            run.append((c, out, subprocess.Popen(cmd, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT)))
        time.sleep(2)
        for r in run[:]:
            if r[2].poll() is not None:
                run.remove(r); done += 1
                if r[2].returncode: print(f'  rc {r[2].returncode}: {r[1]}', flush=True)
                if done % 10 == 0: print(f'{time.strftime("%H:%M:%S")} {done} done, {len(todo)} left, {time.time() - t0:.0f} s', flush=True)
    print(f'{time.strftime("%H:%M:%S")} all jobs done in {time.time() - t0:.0f} s', flush=True)

def per_match(path):
    if not os.path.exists(path): return {}
    r = json.load(open(path))['configs']; out = {}
    for cfg, c in r.items():
        for m in c['perMatch']: out[(cfg, m['seed'])] = m
    return out
table = {}
for g in groups:
    base = per_match(os.path.join(a.out, g['name'], 'random.json'))
    keys = sorted({key_of(c) for c in g['configs']})
    for key in keys:
        B = {k: m for k, m in base.items() if key_of(k[0]) == key and (m.get('forced') or {}).get('how') == 'random'}
        nseeds = len([k for k in base if key_of(k[0]) == key])
        T = table.setdefault(key, {'seeds': nseeds, 'random': {'n': len(B), 'fa': float(np.mean([m['faShare'] for m in B.values()])) if B else None,
                                                        'win': float(np.mean([m['win'] for m in B.values()])) if B else None}, 'bands': []})
        minn = a.min_n or max(4, nseeds // 4)
        for b in bands:
            R = per_match(os.path.join(a.out, g['name'], f'{b}.json'))
            R = {k: m for k, m in R.items() if key_of(k[0]) == key}
            how = {}
            for m in R.values(): h = (m.get('forced') or {}).get('how', 'none'); how[h] = how.get(h, 0) + 1
            got = [k for k, m in R.items() if (m.get('forced') or {}).get('how') == 'forced' and m.get('band0') == b and k in B]
            row = {'band': b, 'n': len(got), 'how': how}
            if len(got) >= 2:
                d = np.array([R[k]['faShare'] - B[k]['faShare'] for k in got]); w = np.array([R[k]['win'] - B[k]['win'] for k in got])
                row.update({'dfa': float(d.mean()), 'dfaSE': float(d.std(ddof=1) / np.sqrt(len(d))), 'dwin': float(w.mean()), 'dwinSE': float(w.std(ddof=1) / np.sqrt(len(w))),
                            'fa': float(np.mean([R[k]['faShare'] for k in got])), 'win': float(np.mean([R[k]['win'] for k in got])), 'baseFa': float(np.mean([B[k]['faShare'] for k in got]))})
            row['used'] = len(got) >= minn
            T['bands'].append(row)
        T['minN'] = minn
for key, T in table.items():
    rows = sorted([r for r in T['bands'] if r['used']], key=lambda r: -r['dfa'])
    print(f"{key}: seeds {T['seeds']} random fa {T['random']['fa']} win {T['random']['win']} | bands used {len(rows)} of {len(T['bands'])} (n >= {T['minN']})")
    for r in rows[:6]: print(f"   + {r['band']:18s} n {r['n']:3d} dfa {r['dfa']:+.3f} ± {r['dfaSE']:.3f} dwin {r['dwin']:+.3f} ± {r['dwinSE']:.3f} win {r['win']:.3f}")
    for r in rows[-3:]: print(f"   - {r['band']:18s} n {r['n']:3d} dfa {r['dfa']:+.3f} ± {r['dfaSE']:.3f} dwin {r['dwin']:+.3f} win {r['win']:.3f}")
json.dump(table, open(os.path.join(a.out, 'band_table2.json'), 'w'), indent=1)
print('saved', os.path.join(a.out, 'band_table2.json'))
