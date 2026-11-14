"""Same-seed evaluation (plan §8.9, B2): a checkpoint's policy, or the rule-enforcing legacy teacher as the baseline,
on a fixed set of matches. A seed fixes its config and seat roles (env.mjs), so both runs face identical matches.
python eval.py --ckpt run/model.pt --out run/eval.json   |   python eval.py --teacher --out baseline.json
  |   python eval.py --legacy --rehearsal 3 --out live.json   (production legacy: no RL seat, rules off)
  [--configs solo:NORMAL,...] [--matches 24] [--chunks 4] [--envs 6] [--seed 7000000] [--greedy]"""
import argparse, hashlib, json, os, time
from collections import defaultdict
import numpy as np
from spenv import NodeEnv, AsyncPool

DEFAULT_CONFIGS = 'solo:NORMAL,solo:HARD,solo:ABYSS,coop4:NORMAL,coop4:HARD,coop4:ABYSS'
CANON = DEFAULT_CONFIGS.split(',')


def run(a, policy=None):
    configs = a.configs.split(',')
    per = max(1, a.matches // a.chunks)
    envs, owner = [], []
    for cfg in configs:
        # a config's seeds depend on its name, never on its position in --configs: any subset run pairs with a full run
        ci = CANON.index(cfg) if cfg in CANON else 100 + int(hashlib.sha256(cfg.encode()).hexdigest()[:6], 16) % 900
        for k in range(a.chunks):
            opts = {'envs': min(a.envs, per), 'configs': [cfg], 'seed': a.seed + ci * 1_000_000 + k * per, 'envSeed': a.seed,
                    'maxMatches': per, 'humans': 0 if cfg.split(':')[0].endswith('ai') else 'random', 'legacyMates': a.legacy_mates,  # coop4ai: every seat an AI seat
                    'rehearsal': a.rehearsal,
                    'metrics': True, 'check': True, 'tripLog': a.trips, 'obsVersion': a.obs_version,
                    **({'forceBand': a.force_band} if a.force_band else {}),
                    **({'bandPolicy': {'share': a.band_policy, 'uniform': False}} if a.band_policy > 0 and policy is not None else {}),
                    **({'bossRehearsal': a.boss_rehearsal} if a.boss_rehearsal is not None else {})}
            opts.update({'mode': 'legacy'} if a.legacy else {'mode': 'teacher', 'enforce': True} if policy is None else {'mode': 'policy'})
            envs.append(NodeEnv(opts, node=a.node)); owner.append(cfg)
    stats = [None] * len(envs)
    pool = AsyncPool(envs)
    pool.reset()
    left, decisions = len(envs), 0
    while left:
        for i, b in pool.poll():
            if b.n == 0:
                envs[i]._send(8); t, payload = envs[i]._recv(); stats[i] = json.loads(payload); left -= 1
                pool.sel.unregister(envs[i].p.stdout)
                continue
            decisions += b.n
            pool.step(i, policy(b))
    for e in envs:
        e.close()
    return aggregate(configs, owner, stats, decisions, 'legacy' if a.legacy else 'rl')


def aggregate(configs, owner, stats, decisions, ai_role='rl'):
    # the evaluated AI seats: the policy's / the teacher's ('rl'), or production legacy's ('legacy'); reported as 'rl'
    out = {'decisions': decisions, 'configs': {}}
    for cfg in configs:
        res = [r for s, o in zip(stats, owner) if o == cfg for r in s['results']]
        y = defaultdict(lambda: defaultdict(float))
        for s, o in zip(stats, owner):
            if o != cfg:
                continue
            for role, v in s['yield'].items():
                for k in ('matches', 'preps', 'buys', 'sells', 'yieldMain', 'yieldMerge', 'roundTrips'):
                    y[role][k] += v.get(k, 0)
        roles = defaultdict(lambda: {'seats': 0, 'survival': 0.0, 'lp': 0.0, 'wins': 0, 'tier': 0.0, 'tiers': 0})
        for r in res:
            for st in r['seats']:
                x = roles[st['role']]
                x['seats'] += 1; x['survival'] += st['survivalRound']; x['lp'] += st['lp']; x['wins'] += st['win']
                if st.get('mainTier') is not None:
                    x['tier'] += st['mainTier']; x['tiers'] += 1
        n = len(res)
        c = {'matches': n, 'teamWin': sum(1 for r in res if any(st['win'] for st in r['seats'])) / max(1, n),
             'roles': {k: {'seats': v['seats'], 'survival': v['survival'] / max(1, v['seats']), 'lp': v['lp'] / max(1, v['seats']),
                           'win': v['wins'] / max(1, v['seats']), **({'humanMainTier': v['tier'] / v['tiers']} if v['tiers'] else {})}
                       for k, v in roles.items()},
             'ai': {k: {'buysPerMatch': v['buys'] / max(1, v['matches']), 'yieldMainPerMatch': v['yieldMain'] / max(1, v['matches']),
                        'yieldMergePerMatch': v['yieldMerge'] / max(1, v['matches']), 'roundTripsPerPrep': v['roundTrips'] / max(1, v['preps']),
                        'violations': v['yieldMain'] + v['yieldMerge']} for k, v in y.items()},
             'errors': sum(s['stats']['errors'] + s['stats']['failed'] + s['stats']['violations'] for s, o in zip(stats, owner) if o == cfg),
             'rejected': sum(s['stats']['rejected'] for s, o in zip(stats, owner) if o == cfg)}
        # per match, for paired comparisons across runs on the same seeds
        c['perMatch'] = [{'seed': r['seed'], 'win': int(any(st['win'] for st in r['seats'])),
                          'rlSurvival': float(np.mean([st['survivalRound'] for st in r['seats'] if st['role'] == ai_role] or [0])),
                          'rlLp': float(np.mean([st['lp'] for st in r['seats'] if st['role'] == ai_role] or [0])),
                          'faShare': r.get('faShare', 0),
                          'bands': [st.get('band') for st in r['seats'] if st.get('rl')],
                          'band0': next((st.get('band') for st in r['seats'] if st['playerId'] == 'ai_0'), None),
                          **({'trips': r['trips']} if 'trips' in r else {})} for r in res]
        if ai_role != 'rl':
            if ai_role in c['roles']:
                c['roles']['rl'] = c['roles'].pop(ai_role)
            if ai_role in c['ai']:
                c['ai']['rl'] = c['ai'].pop(ai_role)
        c['faShare'] = float(np.mean([r.get('faShare', 0) for r in res])) if res else 0.0
        # per band of the evaluated seats (band-conditioned play): matches, team win, fa
        bb = {}
        for pm in c['perMatch']:
            for b in set(x for x in pm.get('bands') or [] if x):
                e = bb.setdefault(b, [0, 0, 0.0]); e[0] += 1; e[1] += pm['win']; e[2] += pm['faShare']
        c['byBand'] = {b: {'n': n, 'win': w / n, 'fa': f / n} for b, (n, w, f) in sorted(bb.items(), key=lambda x: -x[1][0])}
        out['configs'][cfg] = c
    return out


def table(r, label):
    lines = [f'== {label}: {r["decisions"]} decisions']
    for cfg, c in r['configs'].items():
        rl = c['roles'].get('rl', {})
        ai = c['ai'].get('rl', {})
        h = c['roles'].get('human-legacy', {})
        lines.append(f'{cfg:13s} n={c["matches"]:3d} teamWin {c["teamWin"]:.3f} | AI survival {rl.get("survival", 0):.2f} lp {rl.get("lp", 0):.1f}'
                     f' | human survival {h.get("survival", float("nan")):.2f} tier {h.get("humanMainTier", float("nan")):.2f}'
                     f' | AI buys/match {ai.get("buysPerMatch", 0):.1f} violations {int(ai.get("violations", 0))} roundTrips/prep {ai.get("roundTripsPerPrep", 0):.3f}'
                     f' | errors {c["errors"]} rejected {c["rejected"]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt'); ap.add_argument('--teacher', action='store_true'); ap.add_argument('--out')
    ap.add_argument('--legacy', action='store_true', help='production legacy (rules off); pair with --rehearsal 3 for the live setting')
    ap.add_argument('--configs', default=DEFAULT_CONFIGS); ap.add_argument('--matches', type=int, default=24)
    ap.add_argument('--chunks', type=int, default=4); ap.add_argument('--envs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7_000_000); ap.add_argument('--legacy-mates', type=float, default=0.0)
    ap.add_argument('--rehearsal', type=int, default=0); ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--boss-rehearsal', type=int, default=None, help='rehearsal on the Final Assault / Hidden Core rounds only')
    ap.add_argument('--node', default='node'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--trips', action='store_true', help='log every round trip (diagnostics)')
    ap.add_argument('--band-policy', type=float, default=0.0, help='the policy picks the bands of this share of its seats (1 = all, as deployed); 0 = legacy picks')
    ap.add_argument('--force-band', default=None, help='Step 0: seat ai_0 opens with this band (band strength table)')
    a = ap.parse_args()
    DUMP_PATH = os.environ.get('DUMP'); DUMP = {'obs': [], 'mask': []} if DUMP_PATH else None
    a.obs_version = int(os.environ.get('EVAL_OBS', '3'))  # teacher / legacy baselines: the env layout (4 = full control, one-round-trip plans)
    t0 = time.time()
    policy = None
    if not a.teacher and not a.legacy:
        import torch
        torch.set_num_threads(4)  # small batches: a wide intra-op pool only starves the environment processes
        from model import load
        dev = torch.device(a.device)
        net, _ = load(a.ckpt, dev); net.eval()
        a.obs_version = 4 if 'tiles' in net.spec['blocks'] else 3 if 'bands' in net.spec['blocks'] else 2
        gen = torch.Generator(device=dev); gen.manual_seed(1)
        @torch.no_grad()
        def policy(b):
            if DUMP is not None:
                k = np.flatnonzero(np.asarray(b.kinds) == 2)
                if len(k): DUMP['obs'].append(np.array(b.obs)[k].astype(np.float32)); DUMP['mask'].append(np.array(b.mask)[k].astype(np.uint8))
            logits, _ = net(torch.from_numpy(np.array(b.obs)).to(dev), torch.from_numpy(np.array(b.mask)).to(dev))
            if a.greedy:
                return logits.argmax(1).int().cpu().numpy()
            return torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1).int().cpu().numpy()
    r = run(a, policy)
    if DUMP is not None and DUMP['obs']:
        O = np.concatenate(DUMP['obs']); M = np.concatenate(DUMP['mask']); G0 = net.spec['blocks']['global']['offset']
        cfgs = [('coop4' if o[G0 + 8] > 0.5 else 'solo') + ':' + ('FUNNY', 'NORMAL', 'HARD', 'ABYSS')[int(o[G0 + 4:G0 + 8].argmax())] for o in O]
        O.tofile(DUMP_PATH + '.f32'); M.tofile(DUMP_PATH + '.u8')
        bl = sorted(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/bands.json'))).keys())[:40]
        json.dump({'n': len(O), 'D': O.shape[1], 'A': M.shape[1], 'configs': cfgs, 'bandList': bl, 'bandAction0': 60, 'from': 'eval_dump ' + str(a.ckpt)}, open(DUMP_PATH + '.json', 'w'))
        print('dumped', len(O), 'band states', flush=True)
    r['seconds'] = round(time.time() - t0, 1)
    r['label'] = ('legacy-live' if a.legacy else 'teacher-enforce' if a.teacher else f'{a.ckpt} {"greedy" if a.greedy else "sampled"}') + f' rehearsal {a.rehearsal}'
    r['cpus'] = len(os.sched_getaffinity(0)); r['rehearsal'] = a.rehearsal; r['bossRehearsal'] = a.boss_rehearsal
    if a.boss_rehearsal is not None:
        r['label'] += f' boss {a.boss_rehearsal}'
    print(table(r, r['label']), flush=True)
    if a.out:
        json.dump(r, open(a.out, 'w'), indent=1)
