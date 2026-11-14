import json, sys
for p in sys.argv[1:]:
    L = [json.loads(l) for l in open(p)]
    ups = [u for u in L if u.get('event') == 'update']; ev = [u for u in L if u.get('event') == 'eval']
    print('==', p.split('/runs/')[1], 'updates', len(ups), 'evals', len(ev))
    if ups: print('   keys:', ' '.join(k for k in ups[-1] if k not in ('event',))[:400])
    for u in ups[-4:]:
        d = u.get('diag', {}) if isinstance(u.get('diag'), dict) else {}
        g = lambda k: u.get(k, d.get(k))
        f = lambda v: round(v, 4) if isinstance(v, float) else v
        print('  ', {k: f(g(k)) for k in ['update', 'steps', 'sps', 'update_s', 'kl', 'kl3', 'kl_bc', 'clipfrac', 'ev', 'entropy', 'env_errors', 'env_violations', 'lr'] if g(k) is not None})
    for e in ev[-2:]: print('   eval', json.dumps({k: v for k, v in e.items() if k != 'event'})[:300])
