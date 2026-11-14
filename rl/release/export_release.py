"""(11-14) Release export: one policy checkpoint plus the two deploy pickers -> the directory the JS inference loads, a manifest
(files with md5 and bytes, repo / game commits, source md5s, env settings) and a JS-vs-torch parity check on decisions that the
release itself records in play (play.mjs: JS policy + c9 + xboss-v3, headline protocol).

python tools/rl/release/export_release.py --ckpt <model.pt> --c9 <dir: c9-js.json, c9-js.bin> --xboss <dir: dw.json, g3b.json,
    xplace.json> --out <release dir> [--name <model name>] [--repo-commit <sha>] [--matches 8] [--configs solo:ABYSS,coop4h1:ABYSS]
    [--seed 300000000] [--node node] [--cpu <core for the timing run>] [--no-pt] [--skip-parity]

<out>/
  policy/policy.json + policy.bin      the policy (MLP: infer2.mjs, format sp-policy/1; v5: infer5.mjs, sp-policy5/1; no critic)
  policy/model.pt                      the torch checkpoint (eval.py, fine-tuning); left out with --no-pt
  pickers/c9/c9-js.json + c9-js.bin    normal-round picker c9 (rounds 1-13; an infer5.mjs 'p5-picker' head)
  pickers/xboss-v3/dw.json, g3b.json   boss-round scorer Dw and pool generator g3b (xplace.mjs)
  pickers/xboss-v3/xplace.json         the --xplace options, file names relative to this directory (xplace_arg.py resolves them)
  manifest.json                        env settings, files, commits, sources, parity and cost summary
  MD5SUMS                              md5sum -c MD5SUMS (inside the release dir)
  parity.json                          per-config parity, picker counters and JS cost (play.mjs + torch)
Parity gate: max |JS - torch| logit over legal actions <= --tol (default 1e-3) and greedy action agreement >= 0.999."""
import argparse, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.join(HERE, '..', 'py'); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, PY)
import torch
from model import load
torch.set_num_threads(4)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--c9', required=True); ap.add_argument('--xboss', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--name', default=None); ap.add_argument('--repo-commit', default=os.environ.get('SPRL_REPO_COMMIT'))
ap.add_argument('--matches', type=int, default=8); ap.add_argument('--configs', default='solo:ABYSS,coop4h1:ABYSS')
ap.add_argument('--seed', type=int, default=300_000_000); ap.add_argument('--node', default='node'); ap.add_argument('--cpu', default=None)
ap.add_argument('--no-pt', action='store_true'); ap.add_argument('--skip-parity', action='store_true'); ap.add_argument('--tol', type=float, default=1e-3)
a = ap.parse_args()

md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
# Shipped metadata must not carry the training hosts' paths: absolute paths in string fields (training args) become
# <work>/<last two components>. Weights are never touched (base64 blobs and tensors are skipped); weightsMd5 proves it.
ABS = re.compile(r'(?<![\w<.])/(?:[^\s,;:"\'/]+/)*[^\s,;:"\'/]*')
BLOB = re.compile(r'^[A-Za-z0-9+/=]{1000,}$')
LEAK = re.compile(r'/(?:data|home|root|opt|hdd\w*|mnt|srv|tmp|scratch|workspace|Users)/')
def scrub(o):
    if isinstance(o, dict): return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, list): return [scrub(v) for v in o]
    if isinstance(o, tuple): return tuple(scrub(v) for v in o)
    if isinstance(o, str) and not BLOB.match(o):
        return ABS.sub(lambda m: '<work>/' + '/'.join([p for p in m.group(0).split('/') if p][-2:]) if m.group(0).count('/') > 1 else m.group(0), o)
    return o
def leaks(o, path=''):
    if isinstance(o, dict): return [x for k, v in o.items() for x in leaks(v, f'{path}.{k}')]
    if isinstance(o, (list, tuple)): return [x for i, v in enumerate(o) for x in leaks(v, f'{path}[{i}]')]
    return [(path, o[:120])] if isinstance(o, str) and not BLOB.match(o) and LEAK.search(o) else []
def ship_json(src, dst, weights=('data', 'tensors')):
    d = json.load(open(src)); w0 = hashlib.md5(json.dumps([d.get(k) for k in weights]).encode()).hexdigest()
    d = scrub(d); w1 = hashlib.md5(json.dumps([d.get(k) for k in weights]).encode()).hexdigest()
    assert w0 == w1, f'{src}: scrubbing changed the weights'
    bad = leaks(d); assert not bad, f'{src}: host paths left: {bad[:3]}'
    json.dump(d, open(dst, 'w'), separators=(',', ':')); return w0
def ship_pt(src, dst):
    c = torch.load(src, map_location='cpu', weights_only=False)
    tw = lambda c: hashlib.md5(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in c['state'].values())).hexdigest()
    w0 = tw(c); c = {k: (v if k == 'state' else scrub(v)) for k, v in c.items()}
    bad = leaks({k: v for k, v in c.items() if k != 'state'}); assert not bad, f'{src}: host paths left: {bad[:3]}'
    torch.save(c, dst); assert tw(torch.load(dst, map_location='cpu', weights_only=False)) == w0; return w0
def git(*args, cwd=ROOT):
    try: return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return None

out = os.path.abspath(a.out)
if os.path.exists(out) and os.listdir(out): sys.exit(f'{out} exists and is not empty')
for d in ('policy', 'pickers/c9', 'pickers/xboss-v3'): os.makedirs(os.path.join(out, d), exist_ok=True)

# 1. policy: the JS export of the architecture (the critic is training-only and never exported)
net, ck = load(a.ckpt, 'cpu'); net.eval()
arch = 'v5' if (getattr(net, 'cfg', None) or {}).get('arch') == 'v5' else 'mlp'
exporter = os.path.join(PY, 'export5.py' if arch == 'v5' else 'export.py')
subprocess.run([sys.executable, exporter, '--ckpt', a.ckpt, '--out', os.path.join(out, 'policy', 'policy')], check=True)
weights = {'policy/policy.json': ship_json(os.path.join(out, 'policy', 'policy.json'), os.path.join(out, 'policy', 'policy.json'), ('tensors',))}
if not a.no_pt: weights['policy/model.pt'] = ship_pt(a.ckpt, os.path.join(out, 'policy', 'model.pt'))
targs = ck.get('args') or {}
nat_plan = json.loads(targs['nat_plan']) if targs.get('nat_plan') else None

# 2. pickers
shutil.copyfile(os.path.join(a.c9, 'c9-js.bin'), os.path.join(out, 'pickers', 'c9', 'c9-js.bin'))
weights['pickers/c9/c9-js.json'] = ship_json(os.path.join(a.c9, 'c9-js.json'), os.path.join(out, 'pickers', 'c9', 'c9-js.json'), ('tensors',))
assert json.load(open(os.path.join(out, 'pickers', 'c9', 'c9-js.json'))).get('kind') == 'p5-picker', 'c9-js.json is not a p5-picker head'
src_xp = json.load(open(os.path.join(a.xboss, 'xplace.json')))
xp = json.loads(json.dumps(src_xp)); xp['model'] = os.path.basename(src_xp['model'])
if (xp.get('pool') or {}).get('gen'): xp['pool']['gen'] = os.path.basename(src_xp['pool']['gen'])
for f in sorted({xp['model'], (xp.get('pool') or {}).get('gen')} - {None}):
    weights[f'pickers/xboss-v3/{f}'] = ship_json(os.path.join(a.xboss, f), os.path.join(out, 'pickers', 'xboss-v3', f), ('data', 'tensors', 'cfg'))
json.dump(xp, open(os.path.join(out, 'pickers', 'xboss-v3', 'xplace.json'), 'w'))

# 3. manifest (parity added below)
env = {'obsVersion': 3, 'obsExt': net.spec.get('ext'), 'natPlan': nat_plan, 'bandPolicy': float(targs.get('band_policy') or 0),
       'rehearsal': 0, 'bossRehearsal': 0, 'mateRehearsal': 3,
       'normalPicker': {'head': 'pickers/c9/c9-js.json', 'mode': 'reg', 'margin': 0.5, 'pool': True},
       'xPlace': 'pickers/xboss-v3/xplace.json'}
code = ['tools/rl/env.mjs', 'tools/rl/spec.mjs', 'tools/rl/infer2.mjs', 'tools/rl/infer5.mjs', 'tools/rl/xplace.mjs', 'tools/rl/xpool.mjs',
        'tools/rl/xfeat.mjs', 'tools/rl/picker.mjs', 'tools/rl/release/play.mjs', 'server/match/bot.js']
man = {'format': 'sp-release/1', 'name': a.name or os.path.splitext(os.path.basename(a.ckpt))[0], 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'arch': arch, 'params': int(sum(p.numel() for p in net.parameters())), 'steps': ck.get('steps'), 'dataId': net.spec['dataId'], 'D': net.D, 'A': net.A,
       'repoCommit': a.repo_commit, 'gameCommit': git('rev-parse', 'HEAD'), 'env': env,
       'evalFlags': ('--greedy --bond-log --rehearsal 0 --boss-rehearsal 0 --band-policy %g%s --npicker-head pickers/c9/c9-js.json --npicker-mode reg '
                     '--npicker-margin 0.5 --npicker-pool --xplace "$(python tools/rl/release/xplace_arg.py <release dir>)"; co-op adds --mate-rehearsal 3 '
                     '--human-proxy \'{"kinds":["legacy"],"log":true}\'') % (env['bandPolicy'], " --nat-plan '%s'" % json.dumps(nat_plan, separators=(',', ':')) if nat_plan else ''),
       'sources': {'ckpt': {'file': os.path.basename(a.ckpt), 'md5': md5(a.ckpt)},
                   'c9': {f: md5(os.path.join(a.c9, f)) for f in ('c9-js.json', 'c9-js.bin')},
                   'xboss-v3': {**{f: md5(os.path.join(a.xboss, f)) for f in {xp['model'], (xp.get('pool') or {}).get('gen')} - {None}}, 'xplace.json': md5(os.path.join(a.xboss, 'xplace.json'))}},
       'code': {p: md5(os.path.join(ROOT, p)) for p in code if os.path.exists(os.path.join(ROOT, p))},
       'weightsMd5': weights, 'note': 'metadata strings with training-host paths are rewritten to <work>/...; weightsMd5 is the md5 of '
                                      'the weight fields (JSON) or of the state tensors (model.pt), equal before and after'}
man_path = os.path.join(out, 'manifest.json')
def write_manifest():
    files = sorted(p for p in glob.glob(os.path.join(out, '**', '*'), recursive=True) if os.path.isfile(p) and os.path.basename(p) not in ('manifest.json', 'MD5SUMS'))
    man['files'] = {os.path.relpath(p, out): {'md5': md5(p), 'bytes': os.path.getsize(p)} for p in files}
    json.dump(man, open(man_path, 'w'), indent=1, ensure_ascii=False)
    with open(os.path.join(out, 'MD5SUMS'), 'w') as f:  # md5sum -c MD5SUMS inside the release dir
        for p in files + [man_path]: f.write(f'{md5(p)}  {os.path.relpath(p, out)}\n')
write_manifest()

# 4. parity + cost: play.mjs records every policy decision (JS logits) of real matches with both pickers; torch re-scores them
if not a.skip_parity:
    work = tempfile.mkdtemp(prefix='sprl-parity-'); base = os.path.join(work, 'dec')
    cmd = [a.node, os.path.join(HERE, 'play.mjs'), out, '--configs', a.configs, '--matches', str(a.matches), '--seed', str(a.seed),
           '--envs', '1', '--out', os.path.join(work, 'play.json'), '--dump', base]
    if a.cpu is not None: cmd = ['taskset', '-c', str(a.cpu)] + cmd
    t0 = time.time(); r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr[-2000:])
    if r.returncode: sys.exit(f'play.mjs failed ({r.returncode})')
    M = json.load(open(base + '.json')); n, D, A = M['n'], M['D'], M['A']
    obs = np.fromfile(base + '.obs.f32', np.float32).reshape(n, D); mask = np.fromfile(base + '.mask.u8', np.uint8).reshape(n, A)
    lj = np.fromfile(base + '.logits.f32', np.float32).reshape(n, A)
    if hasattr(net, 'with_value'): net.with_value = False
    with torch.no_grad():
        lt = np.concatenate([net(torch.from_numpy(obs[i:i + 512]), torch.from_numpy(mask[i:i + 512].astype(bool)))[0].float().numpy() for i in range(0, n, 512)])
    legal = mask.astype(bool); diff = np.where(legal, np.abs(lj - lt), 0)
    aj = np.where(legal, lj, -np.inf).argmax(1); at = np.where(legal, lt, -np.inf).argmax(1)
    cfg = np.array(M['cfg']); play = json.load(open(os.path.join(work, 'play.json')))
    P = {'decisions': n, 'maxAbsLogitDiffLegal': float(diff.max()), 'argmaxAgree': float((aj == at).mean()), 'tol': a.tol,
         'perConfig': {c: {'decisions': int((cfg == c).sum()), 'maxAbsLogitDiffLegal': float(diff[cfg == c].max()), 'argmaxAgree': float((aj == at)[cfg == c].mean())} for c in sorted(set(M['cfg']))},
         'play': {c: {k: v for k, v in x.items() if k != 'perMatch'} for c, x in play['configs'].items()}, 'memoryMB': play.get('memoryMB'),
         'cpu': (open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(' :\t') if os.path.exists('/proc/cpuinfo') else platform.processor()),
         'pinnedCore': a.cpu, 'node': subprocess.run([a.node, '--version'], capture_output=True, text=True).stdout.strip(), 'seed': a.seed, 'wallSeconds': round(time.time() - t0, 1)}
    P['pass'] = bool(P['maxAbsLogitDiffLegal'] <= a.tol and P['argmaxAgree'] >= 0.999 and all((x.get('normalPicker') or {}).get('sims', 0) == 0 and (x.get('bossPlacement') or {}).get('sims', 0) == 0 for x in P['play'].values()))
    json.dump(P, open(os.path.join(out, 'parity.json'), 'w'), indent=1)
    man['parity'] = {k: P[k] for k in ('decisions', 'maxAbsLogitDiffLegal', 'argmaxAgree', 'pass', 'cpu', 'pinnedCore', 'node', 'memoryMB')}
    man['cost'] = {c: {'policyMs': x['policy']['msPerDecision'], 'c9Ms': (x.get('normalPicker') or {}).get('msPerDecision'),
                       'xbossMs': (x.get('bossPlacement') or {}).get('msPerDecision')} for c, x in P['play'].items()}
    shutil.rmtree(work, ignore_errors=True)
    write_manifest()
    print(json.dumps(man['parity']))
    if not P['pass']: sys.exit('PARITY FAILED')
print('release', out, len(man['files']), 'files')
