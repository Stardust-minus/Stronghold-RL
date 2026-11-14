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
                    'metrics': True, 'check': True, 'tripLog': a.trips, **({'lpObs': a.lp_obs if a.lp_obs == 'raw' else float(a.lp_obs)} if getattr(a, 'lp_obs', None) else {}), 'obsVersion': a.obs_version, **({'obsExt': a.obs_ext} if getattr(a, 'obs_ext', None) else {}), **({'bondLog': True} if getattr(a, 'bond_log', False) else {}),
                    **({'forceBand': a.force_band} if a.force_band else {}),
                    **({'actLog': True} if getattr(a, 'act_log', False) else {}),
                    **({'pickerHead': {'head': a.picker_head, 'trunk': a.picker_trunk, 'K': a.picker_k}} if getattr(a, 'picker_head', None) else {}),
                    **({'normalPicker': {'head': a.npicker_head, 'K': a.npicker_k, 'from': a.npicker_from, 'to': a.npicker_to, **({'mode': a.npicker_mode} if a.npicker_mode else {}), **({'margin': a.npicker_margin} if a.npicker_margin is not None else {}), **({'w': [float(x) for x in a.npicker_w.split(',')]} if getattr(a, 'npicker_w', None) else {}), **({'pool': True, 'edits': a.npicker_edits, 'max': a.npicker_max} if getattr(a, 'npicker_pool', False) else {})}} if getattr(a, 'npicker_head', None) else {}),
                    **({'pLabel': json.loads(a.plabel)} if getattr(a, 'plabel', None) else {}),
                    **({'pBossLabel': json.loads(a.pbosslabel)} if getattr(a, 'pbosslabel', None) else {}),
                    **({'bossSpatial': json.loads(a.boss_spatial)} if getattr(a, 'boss_spatial', None) else {}),
                    **({'xPlace': json.loads(a.xplace)} if getattr(a, 'xplace', None) else {}),
                    **({'xLabel': json.loads(a.xlabel)} if getattr(a, 'xlabel', None) else {}),
                    **({'natPlan': json.loads(a.nat_plan)} if getattr(a, 'nat_plan', None) else {}), **({'bountyField': True} if getattr(a, 'bounty_field', False) else {}),
                    **({'bandPolicy': {'share': a.band_policy, 'uniform': False, **({'planProb': a.plan_prob} if getattr(a, 'plan_prob', None) is not None else {})}} if a.band_policy > 0 and policy is not None else {}),
                    **({'bossRehearsal': a.boss_rehearsal} if a.boss_rehearsal is not None else {}),
                    **({'mateRehearsal': a.mate_rehearsal} if a.mate_rehearsal is not None else {}),
                    **({'humanProxy': json.loads(a.human_proxy)} if getattr(a, 'human_proxy', None) else {}),
                    **({'visibleOnly': True} if getattr(a, 'visible_only', False) else {}), **({'strictPreview': True} if getattr(a, 'strict_preview', False) else {})}
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
            pool.step(i, policy(b, i) if getattr(policy, 'rec', False) else policy(b))
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
                          **({'fa': r['fa'], 'seatsAll': r['seatsAll']} if 'seatsAll' in r else {}),
                          **({'bondLog': [dict(st['bondLog'], band=st.get('band'), win=st['win']) for st in r['seats'] if (st.get('rl') or st.get('role') == 'legacy') and 'bondLog' in st]}
                             if any('bondLog' in st for st in r['seats']) else {}),
                          **({'trips': r['trips']} if 'trips' in r else {}),
                          **({'forced': r['forced']} if 'forced' in r else {}), **({'team': r['team']} if 'team' in r else {}), **({'picker': r['picker']} if 'picker' in r else {}), **({'plab': r['plab']} if 'plab' in r else {}), **({'npicker': r['npicker']} if 'npicker' in r else {}), **({'xplace': r['xplace']} if 'xplace' in r else {}), **({'nat': [st['nat'] for st in r['seats'] if 'nat' in st]} if any('nat' in st for st in r['seats']) else {}), **({'act': r['act']} if 'act' in r else {}), **({'bf': r['bf']} if 'bf' in r else {}),
                          **({'human': [{'pid': st['playerId'], 'band': st.get('band'), 'win': st['win'], 'surv': st['survivalRound'], 'mainTier': st.get('mainTier'), 'proxy': st['proxy'], **({'bondLog': st['bondLog']} if 'bondLog' in st else {})}
                                         for st in r['seats'] if 'proxy' in st]} if any('proxy' in st for st in r['seats']) else {})} for r in res]
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
        if any(pm.get('bondLog') for pm in c['perMatch']):
            c['bondSummary'] = bond_summary(c['perMatch'])
            try:  # (11-13) stacking metrics: national main bond and tier, main / secondary / Σ layers, elites (stackrep.py)
                from stackrep import summarize as _stack
                c['stack'] = _stack(c['perMatch'], c['teamWin'])
            except Exception as e:
                c['stack'] = {'error': str(e)[:200]}
        if any(h.get('bondLog') for pm in c['perMatch'] for h in pm.get('human') or []):  # co-op proxies: the human proxies' own stacking
            try:
                from stackrep import summarize as _stack
                c['humanStack'] = _stack([{'win': pm['win'], 'bondLog': [h['bondLog'] for h in pm.get('human') or [] if h.get('bondLog')]} for pm in c['perMatch']], c['teamWin'])
            except Exception as e:
                c['humanStack'] = {'error': str(e)[:200]}
        out['configs'][cfg] = c
    return out


STACKING = None


def bond_summary(per):
    """Stacking at the seats' last battle: Σ layers of the layer-stacking bonds (perStackParams, not noStack), elite units
    (the 绝技 suntShip count), the main bond excluding 绝技 (most units, then tier, then layers), and fa by layer tercile."""
    global STACKING
    if STACKING is None:
        B = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/bonds.json')))
        STACKING = {k for k, v in B.items() if v.get('perStackParams') and not v.get('noStack')}
    seats = [(s.get('end') or {}, m['faShare'], m['win']) for m in per for s in m.get('bondLog', [])]
    if not seats: return None
    lay = [sum(l for b, t, l, n in e.get('active', []) if b in STACKING) for e, _, _ in seats]
    eli = [next((n for b, t, l, n in e.get('active', []) if b == 'suntShip'), 0) for e, _, _ in seats]
    def mx(e):
        best = max(((n, t, l, b) for b, t, l, n in e.get('active', []) if b != 'suntShip'), default=None)
        return best[3] if best else None
    main = defaultdict(int)
    for e, _, _ in seats: main[mx(e)] += 1
    o = np.argsort(lay); k = len(o) // 3 or 1
    fa = [x[1] for x in seats]
    # contention between the AI seats of one match (all 4 run the same policy in coop4ai): seats sharing the main bond
    # (excl. 绝技), operator base ids held by 2+ seats at R6 / R9 / R12, normal (unmerged) copies held at the last battle
    cont = []
    for m in per:
        bl = m.get('bondLog') or []
        if len(bl) < 2: continue
        mains = [mx(s.get('end') or {}) for s in bl]; mc = max((mains.count(x) for x in set(mains) if x), default=0)
        row = {'sameMain': mc, 'win': m['win'], 'fa': m['faShare'], 'unmerged': float(np.mean([(s.get('end') or {}).get('unmerged', 0) for s in bl]))}
        for r in ('r6', 'r9', 'r12'):
            sets = [set((s.get(r) or {}).get('ops', [])) for s in bl if s.get(r)]
            if len(sets) >= 2:
                cnt = defaultdict(int)
                for st in sets:
                    for x in st: cnt[x] += 1
                row['shared_' + r] = sum(1 for v in cnt.values() if v >= 2)
        cont.append(row)
    def corr(k, y):
        xs = [(c[k], c[y]) for c in cont if k in c]
        if len(xs) < 8: return None
        a = np.array(xs, dtype=np.float64)
        return None if a[:, 0].std() == 0 or a[:, 1].std() == 0 else float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    contention = {'matches': len(cont), **{f'{k}Mean': float(np.mean([c[k] for c in cont if k in c])) for k in ('sameMain', 'shared_r6', 'shared_r9', 'shared_r12', 'unmerged') if any(k in c for c in cont)},
                  'corr': {f'{k}~{y}': corr(k, y) for k in ('sameMain', 'shared_r9', 'shared_r12', 'unmerged') for y in ('win', 'fa')},
                  'faBySameMain': {str(v): float(np.mean([c['fa'] for c in cont if c['sameMain'] == v])) for v in sorted({c['sameMain'] for c in cont})},
                  'winBySameMain': {str(v): float(np.mean([c['win'] for c in cont if c['sameMain'] == v])) for v in sorted({c['sameMain'] for c in cont})}} if cont else None
    # regime trace (11-12, invest-and-stack vs defensive): shop level R3-R6, rerolls / leaks per seat-round, Σ stacking layers
    # at R9 / R12 / R14 (battle start), team LP at the Final Assault start
    lv = defaultdict(list); rr = []; lk = []; lyr = defaultdict(list); fat = []
    for m in per:
        seats_ = m.get('bondLog') or []
        fa0 = [x for s in seats_ for x in (s.get('trace') or []) if x.get('ph') == 'FINAL_ASSAULT']
        if fa0: fat.append(fa0[0]['teamLp'])
        for s in seats_:
            prev = None
            for x in (s.get('trace') or []):
                if x.get('level') is not None and 3 <= x['r'] <= 6: lv[x['r']].append(x['level'])
                if x['r'] in (9, 12, 14) and x.get('layers') is not None: lyr[x['r']].append(x['layers'])
                if prev is not None and x['r'] > prev['r']:
                    if x.get('refreshes') is not None and prev.get('refreshes') is not None: rr.append(x['refreshes'] - prev['refreshes'])
                    if x.get('leaks') is not None and prev.get('leaks') is not None: lk.append(x['leaks'] - prev['leaks'])
                prev = x
    trace = {'levelR3_6': {str(r): float(np.mean(v)) for r, v in sorted(lv.items())}, 'rerollsPerPrep': float(np.mean(rr)) if rr else None,
             'leaksPerRound': float(np.mean(lk)) if lk else None, 'layersAt': {str(r): float(np.mean(v)) for r, v in sorted(lyr.items())},
             'teamLpAtFA': float(np.mean(fat)) if fat else None, 'faStarts': len(fat)}
    return {'seats': len(seats), 'layersMean': float(np.mean(lay)), 'eliteMean': float(np.mean(eli)), 'contention': contention, 'trace': trace,
            'mainExSunt': {str(b): n / len(seats) for b, n in sorted(main.items(), key=lambda x: -x[1])[:6]},
            'faByLayerTercile': [float(np.mean([fa[i] for i in o[:k]])), float(np.mean([fa[i] for i in o[k:len(o) - k]] or [np.nan])), float(np.mean([fa[i] for i in o[len(o) - k:]]))]}


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
    ap.add_argument('--strict-preview', action='store_true', help='strict preview (off by default): legacy placement from the wave preview only, no bounty kill chance, no spawn time')
    ap.add_argument('--visible-only', action='store_true', help='deploy information set: the RL seats see only what a player sees (env visibleOnly)')
    ap.add_argument('--mate-rehearsal', type=int, default=None, help='legacy (non-RL) seats rehearse this many layouts at normal rounds (production); RL seats keep --rehearsal')
    ap.add_argument('--lp-obs', default=None, help="own-LP obs field override: 'raw' (raw LP) or a scale k on lp / lp0")
    ap.add_argument('--bond-log', action='store_true', help='per RL seat: active bonds at the battles of R6 / R9 / R12 and the last one (perMatch bondLog, bondSummary)')
    ap.add_argument('--force-band', default=None, help="Step 0: seat ai_0 opens with this band (band strength table); with --band-policy at ai_0's own band decision, 'random' = a random allowed band")
    ap.add_argument('--plan-prob', type=float, default=None, help='with --band-policy: chance a human proxy whose band has no tied bond announces a random core 主盟约 (env default 0.5)')
    ap.add_argument('--picker-head', default=None, help="boss-round layout picker: the head JSON (with --picker-trunk, the frozen JS trunk export base); boss rehearsal 0")
    ap.add_argument('--human-proxy', default=None, help='co-op proxies: env humanProxy JSON (tools/rl/proxies.mjs: kinds per human seat, rehearsal, announce, log); off = legacy human proxies')
    ap.add_argument('--npicker-head', default=None, help='normal-round layout picker (picker spatial head, <base>.json from export_p5.py); rehearsal 0')
    ap.add_argument('--npicker-k', type=int, default=8); ap.add_argument('--npicker-from', type=int, default=1); ap.add_argument('--npicker-to', type=int, default=13)
    ap.add_argument('--npicker-mode', default=None, help="'reg' (default) or 'score'"); ap.add_argument('--npicker-margin', type=float, default=None)
    ap.add_argument('--npicker-w', default=None, help='reg weights wLeak,wLayer,wKill (default 2,0.2,0.05)')
    ap.add_argument('--npicker-pool', action='store_true', help='picker: the picker scores the candidate pool (bot.js candidatePoolForTraining)'); ap.add_argument('--npicker-edits', type=int, default=12); ap.add_argument('--npicker-max', type=int, default=32)
    ap.add_argument('--pbosslabel', default=None, help='picker boss-round labels (env pBossLabel JSON: dir, K, seeds)')
    ap.add_argument('--boss-spatial', default=None, help='picker (code-P9): env bossSpatial JSON {head, w, K, out, rounds}: spatial boss picker in the xPlace QNet ensemble')
    ap.add_argument('--plabel', default=None, help='picker training labels (env pLabel JSON: dir, K, tmpl, from, to, apply, seed); never for headline evals')
    ap.add_argument('--picker-trunk', default=None); ap.add_argument('--picker-k', type=int, default=8)
    ap.add_argument('--xlabel', default=None, help='xplace: env xLabel JSON (search labels at the RL seats, tools/rl/xsearch.mjs; training only, builds rehearsals)')
    ap.add_argument('--xplace', default=None, help='xplace: env xPlace JSON {"model": <xplace model JSON>, "rounds": "all|normal|boss"} (learned layout policy, no sims)')
    ap.add_argument('--nat-plan', default=None, help='NAT arm: env natPlan JSON (plan obs block + terms); a NAT checkpoint needs it (its obs has the plan block)')
    ap.add_argument('--act-log', action='store_true', help='per RL seat: applied actions, buys by kind, 机变 picks, bounty outcomes, items equipped (perMatch act)')
    ap.add_argument('--bounty-field', action='store_true', help="experiment: an RL seat's legacy layout models its own bounty spawns at normal rounds (env bountyField)")
    a = ap.parse_args()
    a.obs_version = int(os.environ.get('EVAL_OBS', '3'))  # teacher / legacy baselines: the env layout (4 = full control, one-round-trip plans)
    t0 = time.time()
    policy = None
    if not a.teacher and not a.legacy:
        import torch
        torch.set_num_threads(4)  # small batches: a wide intra-op pool only starves the environment processes
        from model import load
        dev = torch.device(a.device)
        net, _ = load(a.ckpt, dev); net.eval()
        if hasattr(net, 'with_value'): net.with_value = False  # (A1) SPNet5: acting never needs the critic
        a.obs_version = 4 if 'tiles' in net.spec['blocks'] else 3 if 'bands' in net.spec['blocks'] else 2
        a.obs_ext = net.spec.get('ext')  # (A1) version 3 extension blocks of the checkpoint (None: off)
        gen = torch.Generator(device=dev); gen.manual_seed(1)
        rec = getattr(net, 'rnn', None) is not None  # (A3) a recurrent policy: per (env process, trajectory) state
        hst = {}; hz = np.zeros(getattr(net, 'state_size', 0), np.float32)
        @torch.no_grad()
        def policy(b, i=None):
            if rec:
                for tid in b.term_ids: hst.pop((i, int(tid)), None)
                H_in = np.stack([hst.get((i, int(b.ids[k])), hz) for k in range(b.n)])
                logits, _, h_new = net(torch.from_numpy(np.array(b.obs)).to(dev), torch.from_numpy(np.array(b.mask)).to(dev), torch.from_numpy(H_in).to(dev))
                h_new = h_new.float().cpu().numpy()
                for k in range(b.n): hst[(i, int(b.ids[k]))] = h_new[k]
            else:
                logits, _ = net(torch.from_numpy(np.array(b.obs)).to(dev), torch.from_numpy(np.array(b.mask)).to(dev))
            if a.greedy:
                return logits.argmax(1).int().cpu().numpy()
            return torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1).int().cpu().numpy()
        policy.rec = rec
    r = run(a, policy)
    r['seconds'] = round(time.time() - t0, 1)
    r['label'] = ('legacy-live' if a.legacy else 'teacher-enforce' if a.teacher else f'{a.ckpt} {"greedy" if a.greedy else "sampled"}') + f' rehearsal {a.rehearsal}'
    r['cpus'] = len(os.sched_getaffinity(0)); r['rehearsal'] = a.rehearsal; r['bossRehearsal'] = a.boss_rehearsal
    if a.boss_rehearsal is not None:
        r['label'] += f' boss {a.boss_rehearsal}'
    print(table(r, r['label']), flush=True)
    if a.out:
        json.dump(r, open(a.out, 'w'), indent=1)
